"""nDCG@k (trec_eval definition), paired t-test and bootstrap confidence intervals."""
import math

import numpy as np


def _ndcg_query(ranked_docids, rels, k):
    gains = [rels.get(d, 0) for d in ranked_docids[:k]]
    dcg = sum(g / math.log2(i + 2) for i, g in enumerate(gains))
    ideal = sorted(rels.values(), reverse=True)[:k]
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def ndcg_per_query(run, qrels, k=10):
    """run: {qid: [(docid, score), ...]}. Only queries present in qrels count; a query missing from the run scores 0."""
    out = {}
    for q, rels in qrels.items():
        hits = run.get(q, [])
        ranked = [d for d, _ in sorted(hits, key=lambda x: -x[1])]
        out[q] = _ndcg_query(ranked, rels, k)
    return out


def ndcg_per_query_pytrec(run, qrels, k=10):
    import pytrec_eval
    ev = pytrec_eval.RelevanceEvaluator({q: {d: int(r) for d, r in rels.items()} for q, rels in qrels.items()},
                                        {f"ndcg_cut.{k}"})
    res = ev.evaluate({q: {d: float(s) for d, s in hits} for q, hits in run.items()})
    return {q: res.get(q, {}).get(f"ndcg_cut_{k}", 0.0) for q in qrels}


def evaluate_run(run, qrels, ks=(1, 5, 10), use_pytrec=True):
    """Returns {'nDCG@k': mean, ..., 'per_query': {str(k): {qid: value}}, 'n_queries': N}."""
    out, per = {}, {}
    for k in ks:
        pq = None
        if use_pytrec:
            try:
                pq = ndcg_per_query_pytrec(run, qrels, k)
            except Exception:
                pq = None
        if pq is None:
            pq = ndcg_per_query(run, qrels, k)
        per[k] = pq
        out[f"nDCG@{k}"] = float(np.mean(list(pq.values()))) if pq else 0.0
    out["per_query"] = {str(k): v for k, v in per.items()}
    out["n_queries"] = len(qrels)
    return out


def paired_ttest(a, b):
    """a, b: {qid: value}. Returns (mean_diff a-b, t, p) over common queries."""
    from scipy import stats
    qs = sorted(set(a) & set(b))
    x = np.array([a[q] for q in qs]); y = np.array([b[q] for q in qs])
    if len(qs) < 2:
        return (float(np.mean(x - y)) if qs else 0.0), float("nan"), float("nan")
    t, p = stats.ttest_rel(x, y)
    return float(np.mean(x - y)), float(t), float(p)


def bootstrap_ci(a, b, n_boot=10000, seed=0, alpha=0.05):
    """Percentile bootstrap CI of the mean paired difference a-b."""
    qs = sorted(set(a) & set(b))
    d = np.array([a[q] - b[q] for q in qs])
    if len(d) == 0:
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    means = d[idx].mean(axis=1)
    return float(d.mean()), float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def compare(a, b, label_a, label_b, n_boot=10000):
    diff, t, p = paired_ttest(a, b)
    _, lo, hi = bootstrap_ci(a, b, n_boot=n_boot)
    return {"a": label_a, "b": label_b, "mean_diff": diff, "t": t, "p": p, "ci95": [lo, hi], "n": len(set(a) & set(b))}
