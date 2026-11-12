"""LLM2Vec on the shared testbed. Writes runs/<name>/<dataset>.json and the document embeddings for the demo.

The 6 variants of the ablation on Qwen2.5-0.5B (full is our official row):
  python method.py --name uni --mode uni
  python method.py --name bi --mode bi
  python method.py --name bi-mntp --mode bi --adapters outputs/q05/mntp
  python method.py --name uni-simcse --mode uni --adapters outputs/q05/uni-simcse
  python method.py --name bi-simcse --mode bi --adapters outputs/q05/bi-simcse
  python method.py --name full --mode bi --adapters outputs/q05/mntp outputs/q05/mntp-simcse

Then, from the testbed folder:
  python testbed.py score <path>/runs/full --method "LLM2Vec (Qwen2.5-0.5B)" --pair 02-llm2vec --against bge --out <path>/results.json
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

from encoder import encode, load_model, load_tokenizer

HERE = Path(__file__).resolve().parent
# in the class repo we are in pairs/02-llm2vec, on our side the testbed is next to our folder
for folder in [HERE.parents[1], HERE.parent / "ir2-testbed"]:
    if (folder / "testbed.py").exists():
        sys.path.insert(0, str(folder))
import testbed

# Table 10 of the paper, for the queries only. Encoders do not get them.
INSTRUCTIONS = {
    "scifact": "Given a scientific claim, retrieve documents that support or refute the claim:",
    "nfcorpus": "Given a question, retrieve relevant documents that best answer the question:",
    "arguana": "Given a claim, find documents that refute the claim:",
}
MAX_LENGTH = 512  # like the authors. Documents have 200 words, almost none is longer


def search(doc_emb, query_emb, doc_ids, query_ids):
    # same as testbed.dense_search: K + 1 because on ArguAna the query's own document is removed
    scores = query_emb @ doc_emb.T
    top = np.argsort(-scores, axis=1)[:, :testbed.K + 1]
    return {q: {doc_ids[j]: float(scores[i, j]) for j in top[i]} for i, q in enumerate(query_ids)}


def embed_dataset(model, tok, name, mode, how, corpus, queries):
    instruction = None if mode == "encoder" else INSTRUCTIONS[name]
    docs = encode(model, tok, testbed.doc_texts(corpus), mode, how=how, max_length=MAX_LENGTH)
    qs = encode(model, tok, list(queries.values()), mode, instruction=instruction, how=how, max_length=MAX_LENGTH)
    return docs, qs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--mode", required=True, choices=["uni", "bi", "encoder"])
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--adapters", nargs="*", default=[])
    p.add_argument("--pooling", default="mean", choices=["mean", "weighted_mean", "last"])
    p.add_argument("--datasets", nargs="+", default=testbed.DATASETS, choices=testbed.DATASETS)
    args = p.parse_args()

    out = HERE / "runs" / args.name
    out.mkdir(parents=True, exist_ok=True)
    tok = load_tokenizer(args.model)
    model = load_model(args.model, args.adapters)
    for name in args.datasets:
        t0 = time.time()
        corpus, queries, _ = testbed.load(name)  # the qrels are not used here
        docs, qs = embed_dataset(model, tok, name, args.mode, args.pooling, corpus, queries)
        np.save(out / f"{name}_docs.npy", docs)
        testbed.save_run(search(docs, qs, list(corpus), list(queries)), out / f"{name}.json")
        print(f"{name}: {len(corpus)} docs, {len(queries)} queries, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
