"""BM25 first stage with bm25s (Lucene scoring, k1=0.9, b=0.4 by default) and TREC run-file IO."""
from collections import defaultdict

from .common import BM25_K1, BM25_B


def bm25_search(doc_ids, doc_texts, queries, k=100, k1=BM25_K1, b=BM25_B, drop_self=False):
    """queries: {qid: text}. Returns run {qid: [(docid, score), ...]} sorted by score desc.

    drop_self removes a hit whose docid equals the qid (ArguAna: the query is itself a corpus document)."""
    import bm25s
    import Stemmer

    stemmer = Stemmer.Stemmer("english")
    corpus_tokens = bm25s.tokenize(doc_texts, stopwords="en", stemmer=stemmer, show_progress=False)
    retriever = bm25s.BM25(method="lucene", k1=k1, b=b)
    retriever.index(corpus_tokens, show_progress=False)

    qids = list(queries)
    q_tokens = bm25s.tokenize([queries[q] for q in qids], stopwords="en", stemmer=stemmer,
                              return_ids=False, show_progress=False)
    kk = min(k + (1 if drop_self else 0), len(doc_ids))
    results, scores = retriever.retrieve(q_tokens, k=kk, show_progress=False)
    run = {}
    for i, q in enumerate(qids):
        hits = [(doc_ids[int(di)], float(s)) for di, s in zip(results[i], scores[i]) if s > 0]
        if drop_self:
            hits = [h for h in hits if h[0] != q]
        run[q] = hits[:k]
    return run


def write_trec_run(run, path, tag="run"):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf8") as f:
        for qid, hits in run.items():
            for rank, (docid, score) in enumerate(hits, 1):
                f.write(f"{qid} Q0 {docid} {rank} {score:.6f} {tag}\n")


def read_trec_run(path):
    run = defaultdict(list)
    with open(path, encoding="utf8") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 6:
                continue
            qid, _, docid, rank, score, _ = parts[:6]
            run[qid].append((int(rank), docid, float(score)))
    return {q: [(d, s) for _, d, s in sorted(v)] for q, v in run.items()}


def ranks_to_run(order):
    """{qid: [docid, ...]} in final order -> {qid: [(docid, score)]} with descending integer scores (no ties)."""
    return {q: [(d, float(len(docs) - i)) for i, d in enumerate(docs)] for q, docs in order.items()}
