"""Bridge to the course testbed (github.com/kimgerdes/ir2-testbed, cloned next to this project).

- candidates come from the testbed's own BM25 run (k1=1.2, b=0.75, Porter, no stopwords; top-101 per query)
  with the testbed's document text (title + abstract, first 200 words);
- run files are written in the testbed format {qid: {docid: score}} with at least 100 documents per query;
- scoring calls `testbed.py score ... --against ce`, so p-values and CIs are the shared script's.
"""
import importlib.util
import json
import os
import pathlib
import subprocess
import sys

from .common import ROOT, PROC, jload

TESTBED = pathlib.Path(os.environ.get("RANKGPT_TESTBED", ROOT.parent / "ir2-testbed"))


def load_testbed_module():
    spec = importlib.util.spec_from_file_location("testbed", TESTBED / "testbed.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def testbed_candidates(name, drop_self=True):
    """Rows {qid, query, hits:[{docid, score, rank, text}]} from the testbed BM25 run (all ~100 candidates, in BM25 order).
    On ArguAna the query's own document is removed first (the testbed removes it at scoring time only)."""
    tb = load_testbed_module()
    corpus, queries, qrels = tb.load(name)
    texts = dict(zip(corpus, tb.doc_texts(corpus)))
    run = tb.baseline_run(name, "bm25")
    rows = []
    for q, text in queries.items():
        ranked = sorted(run[q], key=run[q].get, reverse=True)
        if drop_self and name == "arguana":
            ranked = [d for d in ranked if d != q]
        ranked = ranked[: tb.K]
        rows.append({"qid": q, "query": " ".join(text.split()),
                     "hits": [{"docid": d, "score": float(run[q][d]), "rank": r, "text": texts[d]} for r, d in enumerate(ranked, 1)]})
    return rows, qrels


def to_testbed_run(order, candidates):
    """order: {qid: [docid, ...]} (the re-ranked part, possibly followed by more docs); candidates: {qid: [docid, ...]} in
    first-stage order. Returns {qid: {docid: score}} with every candidate: ranked docs first, then the untouched tail."""
    run = {}
    for q, cands in candidates.items():
        head = [d for d in order.get(q, []) if d in set(cands)]
        seen = set(head)
        tail = [d for d in cands if d not in seen]
        docs = head + tail
        run[q] = {d: float(len(docs) - i) for i, d in enumerate(docs)}
    return run


def candidate_order(name):
    """{qid: [docid, ...]} from data/processed/<name>/candidates.jsonl."""
    out = {}
    with open(PROC / name / "candidates.jsonl", encoding="utf8") as f:
        for line in f:
            r = json.loads(line)
            out[r["qid"]] = [h["docid"] for h in r["hits"]]
    return out


def score_with_testbed(run_dir, method, pair="13-rankgpt", against="ce", out=None):
    """Runs `testbed.py score` (the shared script) and returns the parsed results.json."""
    run_dir = pathlib.Path(run_dir).resolve()
    out = pathlib.Path(out or run_dir.parent / f"{run_dir.name}_results.json").resolve()
    cmd = [sys.executable, "testbed.py", "score", str(run_dir), "--method", method, "--pair", pair, "--out", str(out)]
    if against:
        cmd += ["--against", against]
    proc = subprocess.run(cmd, cwd=TESTBED, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"testbed.py score failed:\n{proc.stdout}\n{proc.stderr}")
    return jload(out), proc.stdout
