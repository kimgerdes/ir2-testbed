import json
import subprocess
import sys
from pathlib import Path
import numpy as np
from sentence_transformers import CrossEncoder, SentenceTransformer

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import testbed
import hyde

OUT_DIR = HERE / "results_variants"
OUT_DIR.mkdir(exist_ok=True)
TMP_RUNS = HERE / "runs" / "variants"


def run_dense_hyde(model_name, folder, prefix="", out_run_dir=None):
    model = SentenceTransformer(model_name, device=testbed.device())
    runs = {}
    for name in testbed.DATASETS:
        corpus, queries, _ = testbed.load(name)
        ids, qids = list(corpus), list(queries)
        D = model.encode(testbed.doc_texts(corpus), batch_size=64, normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True)
        Q = model.encode([prefix + queries[q] for q in qids], batch_size=64, normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True)
        gens = hyde.load_generations(name, folder=folder)
        E = hyde.embed_generations(model, qids, gens, normalize=True)
        V = hyde.hyde_vectors(E, Q, n=8)
        V = V / np.linalg.norm(V, axis=1, keepdims=True)
        run = hyde.search(V, D, qids, ids)
        runs[name] = run
        if out_run_dir:
            testbed.save_run(run, out_run_dir / f"{name}.json")
    return runs


def run_dense_plain(model_name, prefix=""):
    model = SentenceTransformer(model_name, device=testbed.device())
    runs = {}
    for name in testbed.DATASETS:
        corpus, queries, _ = testbed.load(name)
        runs[name] = testbed.dense_search(model, corpus, queries, query_prefix=prefix)
    return runs


def run_bm25_and_ce_hyde(folder="generations"):
    ce_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device=testbed.device())
    for name in testbed.DATASETS:
        corpus, queries, _ = testbed.load(name)
        gens = hyde.load_generations(name, folder=folder)
        expanded_queries = {q: queries[q] + " " + " ".join(gens[q][:8]) for q in queries}
        bm25_hyde = testbed.bm25_search(name, corpus, expanded_queries)
        testbed.save_run(bm25_hyde, TMP_RUNS / "bm25_hyde" / f"{name}.json")
        ce_hyde = testbed.rerank(ce_model, corpus, queries, bm25_hyde)
        testbed.save_run(ce_hyde, TMP_RUNS / "ce_bm25_hyde" / f"{name}.json")
        print(f"finished bm25+hyde & ce(bm25+hyde) on {name}")


def score_contriever():
    plain = run_dense_plain("facebook/contriever")
    with_hyde = run_dense_hyde("facebook/contriever", "generations")
    rows = {}
    for name in testbed.DATASETS:
        qids, nd_p, rc_p = testbed.score_run(plain[name], name)
        _, nd_h, rc_h = testbed.score_run(with_hyde[name], name)
        rows[name] = {
            "n_queries": len(qids),
            "ndcg10": round(float(nd_h.mean()), 4),
            "recall100": round(float(rc_h.mean()), 4),
            "baseline": {"name": "contriever", "ndcg10": round(float(nd_p.mean()), 4), "recall100": round(float(rc_p.mean()), 4)},
            "ndcg10_vs_baseline": testbed.compare(nd_h, nd_p),
            "recall100_vs_baseline": testbed.compare(rc_h, rc_p),
        }
    json.dump({"pair": "15-hyde", "method": "HyDE + Contriever (Qwen2.5-3B, N=8)", "datasets": rows}, open(OUT_DIR / "contriever_hyde.json", "w"), indent=1)
    print("wrote contriever_hyde.json")


if __name__ == "__main__":
    print("1/4 Running BGE-small + generations_plain...")
    run_dense_hyde("BAAI/bge-small-en-v1.5", "generations_plain", prefix=testbed.BGE_QUERY, out_run_dir=TMP_RUNS / "bge_plain")
    print("2/4 Running MiniLM + generations...")
    run_dense_hyde("all-MiniLM-L6-v2", "generations", prefix="", out_run_dir=TMP_RUNS / "minilm_hyde")
    print("3/4 Running BM25+HyDE and Cross-Encoder(BM25+HyDE)...")
    run_bm25_and_ce_hyde("generations")
    print("4/4 Running Contriever vs Contriever+HyDE...")
    score_contriever()
    for folder, method, against in [
        ("bge_plain", "HyDE + BGE-small (plain, N=8)", "bge"),
        ("minilm_hyde", "HyDE + MiniLM (N=8)", "minilm"),
        ("bm25_hyde", "BM25 + HyDE (N=8)", "bm25"),
        ("ce_bm25_hyde", "Cross-Encoder on BM25+HyDE (N=8)", "ce"),
    ]:
        subprocess.run([sys.executable, str(ROOT / "testbed.py"), "score", str(TMP_RUNS / folder), "--method", method,
                        "--pair", "15-hyde", "--against", against, "--out", str(OUT_DIR / f"{folder}.json")], check=True)
