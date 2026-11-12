"""Per-query scores and compute time for the notebooks.

The run files and the training logs are not in the repo (too big), so we compute
everything once from runs/ and outputs/ and keep the small results in results/:
  results/per_query.json  nDCG@10, Recall@100 and rank of the first relevant doc, per query and per system
  results/compute.json    hours of training and evaluation, without the time the Mac was sleeping
  results/encoder_geometry.json  mean cosines of queries and documents for the encoders
Run: python analysis.py
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
for folder in [HERE.parents[1], HERE.parent / "ir2-testbed"]:
    if (folder / "testbed.py").exists():
        sys.path.insert(0, str(folder))
import testbed

RESULTS = HERE / "results"
OURS = ["uni", "bi", "bi-mntp", "uni-simcse", "bi-simcse", "full", "q15-full", "sllama",
        "modernbert", "modernbert-simcse", "modernbert-simcse-d01", "ettin", "ettin-simcse", "ettin-simcse-d01"]
BASELINES = ["bm25", "minilm", "bge", "rrf", "ce"]
NO_RANK = 101  # the relevant doc is not in the top 100

# seconds for each dataset, copied from outputs/q05/eval.log, outputs/run2.log and outputs/run3.log
EVAL_SECONDS = {
    "full": [700, 522, 830], "uni": [685, 516], "bi": [691, 534], "bi-mntp": [726, 534],
    "bi-simcse": [711, 530], "uni-simcse": [723, 571], "modernbert": [435, 327], "ettin": [426, 300],
    "modernbert-simcse": [394, 300, 509], "ettin-simcse": [395, 298, 512], "q15-full": [1985, 1555, 2398],
    "sllama": [2371, 1767, 2754], "ettin-simcse-d01": [410, 310, 524], "modernbert-simcse-d01": [412, 308, 525],
}


def first_relevant_rank(run, qrels, qids, name):
    ranks = []
    for q in qids:
        ranked = sorted(run[q], key=run[q].get, reverse=True)
        if name == "arguana":  # same rule as testbed.score_run
            ranked = [d for d in ranked if d != q]
        hits = [i + 1 for i, d in enumerate(ranked[:testbed.K]) if qrels[q].get(d, 0) > 0]
        ranks.append(hits[0] if hits else NO_RANK)
    return ranks


def build_per_query():
    base = testbed.stored()
    table = {}
    for name in testbed.DATASETS:
        _, queries, qrels = testbed.load(name)
        qids = list(queries)
        entry = {"qids": qids}
        for system in BASELINES:
            entry[system] = {"ndcg10": base[name][system]["ndcg10"], "recall100": base[name][system]["recall100"]}
            path = testbed.BASE_RUNS / name / f"{system}.json"
            if path.exists():  # only bm25, minilm and bge have run files
                entry[system]["rank"] = first_relevant_rank(json.load(open(path)), qrels, qids, name)
        for system in OURS:
            path = HERE / "runs" / system / f"{name}.json"
            if not path.exists():
                continue
            run = json.load(open(path))
            _, nd, rc = testbed.score_run(run, name)
            entry[system] = {"ndcg10": nd.round(6).tolist(), "recall100": rc.round(6).tolist(),
                             "rank": first_relevant_rank(run, qrels, qids, name),
                             # hubs: how many different docs we find in all the top 10 lists together
                             "distinct_top10": len({d for q in qids for d in sorted(run[q], key=run[q].get, reverse=True)[:10]})}
        table[name] = entry
    return table


def training_hours(log):
    # When the Mac was sleeping, one step can take more than 1000 s.
    # We replace these steps by the median step, so we only count real compute.
    seconds = np.array([json.loads(line)["seconds"] for line in open(log)])
    median = np.median(seconds)
    slept = seconds > 5 * median
    clean = np.where(slept, median, seconds)
    return round(seconds.sum() / 3600, 2), round(clean.sum() / 3600, 2), int(slept.sum())


def build_compute():
    rows = []
    for log in sorted((HERE / "outputs").glob("*/*/log.jsonl")):
        args = json.load(open(log.parent / "args.json"))
        raw, clean, slept = training_hours(log)
        rows.append({"what": "train", "run": str(log.parent.relative_to(HERE / "outputs")), "model": args["model"],
                     "step": args["step"], "steps": sum(1 for _ in open(log)), "hours_raw": raw, "hours": clean,
                     "sleep_steps": slept})
    for run, seconds in EVAL_SECONDS.items():
        rows.append({"what": "eval", "run": run, "datasets": len(seconds), "hours": round(sum(seconds) / 3600, 2)})
    return rows


def stem_tool():
    # same preprocessing as the BM25 baseline of the testbed
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer
    from sklearn.feature_extraction.text import CountVectorizer
    nltk.download("stopwords", quiet=True)
    sw, porter, analyze = set(stopwords.words("english")), PorterStemmer(), CountVectorizer().build_analyzer()
    return lambda t: {porter.stem(x) for x in analyze(t) if x not in sw}


def overlap(name):
    """For each query: the part of its word stems that we find in its best labelled document."""
    stems = stem_tool()
    corpus, queries, qrels = testbed.load(name)
    texts = dict(zip(corpus, testbed.doc_texts(corpus)))
    out = []
    for q in queries:
        qs = stems(queries[q])
        rel = [d for d, g in qrels[q].items() if g > 0 and d in texts]
        out.append(max(len(qs & stems(texts[d])) / max(len(qs), 1) for d in rel) if rel else 0.0)
    return np.array(out)


def build_geometry():
    # Mean cosine between documents, between queries and between a query and a document,
    # on the first 150 SciFact queries. Encoders get no instruction (like in method.py).
    from encoder import encode, load_model, load_tokenizer
    from method import INSTRUCTIONS
    corpus, queries, qrels = testbed.load("scifact")
    doc_ids = list(corpus)
    qids = list(queries)[:150]
    rows = []
    for run, model_name, adapters in [
        ("full", "Qwen/Qwen2.5-0.5B", ["outputs/q05/mntp", "outputs/q05/mntp-simcse"]),
        ("ettin", "jhu-clsp/ettin-encoder-150m", []),
        ("ettin-simcse", "jhu-clsp/ettin-encoder-150m", ["outputs/ettin/simcse"]),
        ("ettin-simcse-d01", "jhu-clsp/ettin-encoder-150m", ["outputs/ettin/simcse-d01"]),
        ("modernbert", "answerdotai/ModernBERT-base", []),
        ("modernbert-simcse", "answerdotai/ModernBERT-base", ["outputs/modernbert/simcse"]),
        ("modernbert-simcse-d01", "answerdotai/ModernBERT-base", ["outputs/modernbert/simcse-d01"]),
    ]:
        model = load_model(model_name, [str(HERE / a) for a in adapters])
        D = np.load(HERE / "runs" / run / "scifact_docs.npy")
        mode, instruction = ("bi", INSTRUCTIONS["scifact"]) if run == "full" else ("encoder", None)
        Q = encode(model, load_tokenizer(model_name), [queries[q] for q in qids], mode, instruction=instruction, max_length=512)
        i, j = np.random.default_rng(0).integers(0, len(D), (2, 2000))
        S = Q @ D.T
        rel = [S[k, doc_ids.index(d)] for k, q in enumerate(qids) for d in qrels[q]]
        rows.append({"run": run, "doc-doc": float((D[i] * D[j]).sum(1).mean()),
                     "query-query": float((Q @ Q.T)[np.triu_indices(len(Q), 1)].mean()),
                     "query-doc": float(S.mean()), "query-labelled doc": float(np.mean(rel))})
        del model
    return rows


def cached(path, build, rebuild=False):
    if rebuild or not path.exists():
        json.dump(build(), open(path, "w"))
    return json.load(open(path))


def per_query(rebuild=False):
    return cached(RESULTS / "per_query.json", build_per_query, rebuild)


def compute(rebuild=False):
    return cached(RESULTS / "compute.json", build_compute, rebuild)


def geometry(rebuild=False):
    return cached(RESULTS / "encoder_geometry.json", build_geometry, rebuild)


if __name__ == "__main__":
    per_query(rebuild=True)
    geometry(rebuild=True)  # needs the models, about 3 minutes
    rows = compute(rebuild=True)
    print("training", round(sum(r["hours"] for r in rows if r["what"] == "train"), 1), "h,",
          "evaluation", round(sum(r["hours"] for r in rows if r["what"] == "eval"), 1), "h")
