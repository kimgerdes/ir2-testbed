"""NanoBEIR retrieval evaluation (blog §Evaluator Selection), from scratch.

NanoBEIR = 13 small BEIR subsets (50 queries, a few thousand documents each)
from zeta-alpha-ai. The blog reports the mean NDCG@10 over the 13 datasets.
Here any model is evaluated through one interface, ``score_fn(queries, corpus)
-> (n_queries, n_docs) score matrix``, so dense models, our static models and
BM25 all use exactly the same metric code.

    DCG@10  = sum_{r=1}^{10} rel_r / log2(r + 1)     (binary relevance, as BEIR / ST)
    NDCG@10 = DCG@10 / IDCG@10
"""
from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache

import numpy as np
from datasets import load_dataset

NANOBEIR_DATASETS = [
    "ClimateFEVER", "DBPedia", "FEVER", "FiQA2018", "HotpotQA", "MSMARCO", "NFCorpus",
    "NQ", "QuoraRetrieval", "SCIDOCS", "ArguAna", "SciFact", "Touche2020",
]

ScoreFn = Callable[[list[str], list[str]], np.ndarray]


@lru_cache(maxsize=None)
def load_nanobeir(name: str) -> tuple[list[str], list[str], list[str], list[str], dict[str, set[str]]]:
    """Returns (query_ids, queries, doc_ids, docs, qrels: query_id -> set(relevant doc ids))."""
    repo = f"zeta-alpha-ai/Nano{name}"
    corpus = load_dataset(repo, "corpus", split="train")
    queries = load_dataset(repo, "queries", split="train")
    qrels_ds = load_dataset(repo, "qrels", split="train")
    qrels: dict[str, set[str]] = {}
    for qid, did in zip(qrels_ds["query-id"], qrels_ds["corpus-id"]):
        qrels.setdefault(str(qid), set()).add(str(did))
    # Text field only, empty texts dropped: same as sentence-transformers'
    # NanoBEIREvaluator (nano_beir.py L455-456), so numbers are comparable to the blog.
    docs = [(str(i), t) for i, t in zip(corpus["_id"], corpus["text"]) if len(t) > 0]
    qs = [(str(i), t) for i, t in zip(queries["_id"], queries["text"]) if len(t) > 0]
    return [i for i, _ in qs], [t for _, t in qs], [i for i, _ in docs], [t for _, t in docs], qrels


def ndcg_at_k(scores: np.ndarray, query_ids: list[str], doc_ids: list[str], qrels: dict[str, set[str]], k: int = 10) -> float:
    """Mean NDCG@k over the queries that have at least one relevant document."""
    doc_ids_arr = np.asarray(doc_ids)
    topk = np.argsort(-scores, axis=1, kind="stable")[:, :k]
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    values = []
    for i, qid in enumerate(query_ids):
        rel = qrels.get(qid)
        if not rel:
            continue
        gains = np.isin(doc_ids_arr[topk[i]], list(rel)).astype(float)
        idcg = discounts[: min(len(rel), k)].sum()
        values.append((gains * discounts).sum() / idcg)
    return float(np.mean(values))


def dense_score_fn(encode_queries: Callable[[list[str]], np.ndarray], encode_docs: Callable[[list[str]], np.ndarray] | None = None) -> ScoreFn:
    """Cosine similarity between query and document embeddings."""
    encode_docs = encode_docs or encode_queries

    def score(queries: list[str], docs: list[str]) -> np.ndarray:
        q = np.asarray(encode_queries(queries), dtype=np.float32)
        d = np.asarray(encode_docs(docs), dtype=np.float32)
        q /= np.linalg.norm(q, axis=1, keepdims=True) + 1e-12
        d /= np.linalg.norm(d, axis=1, keepdims=True) + 1e-12
        return q @ d.T

    return score


def bm25_score_fn() -> ScoreFn:
    """BM25 with bm25s + English PyStemmer, the blog's BM25 setup (§NanoBEIR, BM25 details)."""
    import bm25s
    import Stemmer

    stemmer = Stemmer.Stemmer("english")

    def score(queries: list[str], docs: list[str]) -> np.ndarray:
        retriever = bm25s.BM25()
        retriever.index(bm25s.tokenize(docs, stopwords="en", stemmer=stemmer, show_progress=False), show_progress=False)
        q_tokens = bm25s.tokenize(queries, stopwords="en", stemmer=stemmer, return_ids=False, show_progress=False)
        return np.stack([retriever.get_scores(q) for q in q_tokens])

    return score


def evaluate(score_fn: ScoreFn, datasets: list[str] | None = None, k: int = 10, verbose: bool = False) -> dict[str, float]:
    """NDCG@k per NanoBEIR dataset, plus their mean under key 'mean'."""
    results = {}
    for name in datasets or NANOBEIR_DATASETS:
        qids, queries, dids, docs, qrels = load_nanobeir(name)
        results[name] = ndcg_at_k(score_fn(queries, docs), qids, dids, qrels, k=k)
        if verbose:
            print(f"  Nano{name:<15} NDCG@{k} = {results[name]:.4f}", flush=True)
    results["mean"] = float(np.mean([results[n] for n in (datasets or NANOBEIR_DATASETS)]))
    return results
