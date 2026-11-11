"""BM25, bge-small, Qwen3-Embedding 0.6B and 8B (with their Matryoshka truncations) and GTE-ModernColBERT on LIMIT.
Recall@2/10/20/100 as in the paper (Tables 4 and 5). One file per system: outputs/limit/<split>/<system>.json.

  python pairs/04-limit/scripts/limit_eval.py --split limit-small                 all systems but qwen3-8b (it needs ~16 GB of GPU memory)
  python pairs/04-limit/scripts/limit_eval.py --split limit --systems qwen3-8b     a subset"""
import argparse
import time

import numpy as np

import limitlib
from limitlib import testbed

SYSTEMS = ["bm25", "bge", "qwen3", "qwen3-8b", "colbert"]
QWEN_SYSTEMS = {"qwen3": (limitlib.QWEN, "Qwen3-Embedding-0.6B", 16), "qwen3-8b": (limitlib.QWEN8, "Qwen3-Embedding-8B", 8)}
LABELS = {"bm25": "BM25", "bge": "bge-small-en-v1.5 (384d)", "colbert": "GTE-ModernColBERT (multi-vector, MaxSim)"}


def evaluate(split, system):
    corpus, queries, qrels = limitlib.load_limit(split)
    qids, dids = list(queries), list(corpus)
    labels = dict(LABELS)
    t0 = time.time()
    if system == "bm25":
        runs = {"bm25": testbed.bm25_search(split, {d: {"title": "", "text": t} for d, t in zip(dids, limitlib.texts_of(split, corpus))}, queries)}
    elif system == "bge":
        D, Q = limitlib.dense_embeddings(limitlib.BGE, split, corpus, queries)
        runs = {"bge": limitlib.to_run(limitlib.cosine_scores(Q, D), qids, dids)}
    elif system in QWEN_SYSTEMS:
        model, label, batch = QWEN_SYSTEMS[system]
        D, Q = limitlib.dense_embeddings(model, split, corpus, queries, batch_size=batch)
        runs = {}
        for dim in limitlib.QWEN_DIMS[model]:
            runs[f"{system}@{dim}"] = limitlib.to_run(limitlib.cosine_scores(Q, D, dim), qids, dids)
            labels[f"{system}@{dim}"] = f"{label} ({dim}d)"
    else:
        D, Q = limitlib.colbert_embeddings(split, corpus, queries)
        runs = {"colbert": limitlib.to_run(limitlib.maxsim_scores(Q, D), qids, dids)}
    seconds = round(time.time() - t0, 1)
    result = {"split": split, "n_docs": len(corpus), "n_queries": len(queries), "seconds": seconds, "systems": {}}
    for name, run in runs.items():
        r = limitlib.recall_at(run, qrels)
        result["systems"][name] = {"label": labels[name], **{f"recall@{k}": round(float(v.mean()) * 100, 2) for k, v in r.items()},
                                   "per_query_recall@2": r[2].round(2).tolist(), "per_query_recall@10": r[10].round(2).tolist()}
        limitlib.save_json(run, limitlib.CACHE / split / f"run.{name}.json")
        print(f"{split:11} {labels[name]:45} R@2 {r[2].mean() * 100:5.1f}  R@10 {r[10].mean() * 100:5.1f}  "
              f"R@20 {r[20].mean() * 100:5.1f}  R@100 {r[100].mean() * 100:5.1f}", flush=True)
    result["query_ids"] = list(qrels)
    limitlib.save_json(result, limitlib.OUT / "limit" / split / f"{system}.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", choices=limitlib.LIMIT_SPLITS, default="limit-small")
    parser.add_argument("--systems", nargs="+", choices=SYSTEMS, default=[s for s in SYSTEMS if s != "qwen3-8b"])
    args = parser.parse_args()
    np.random.seed(limitlib.SEED)
    for system in args.systems:
        evaluate(args.split, system)
