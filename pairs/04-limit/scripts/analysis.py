"""Per-query error analysis of a testbed run against bge-small: where the method helps, where it hurts, and two
simple covariates (query length, number of relevant documents) to test hypotheses. Writes outputs/error_analysis_<system>.md.

  python pairs/04-limit/scripts/analysis.py            GTE-ModernColBERT (runs/)
  python pairs/04-limit/scripts/analysis.py qwen3      Qwen3-Embedding-0.6B (runs/qwen3/)"""
import json
import sys

import numpy as np

import limitlib
from limitlib import testbed

TOP = 5


def first_relevant_rank(run, rel, query_id):
    """Rank as the scorer sees it: on ArguAna the query's own document is removed first."""
    ranked = [d for d in sorted(run, key=run.get, reverse=True) if d != query_id]
    return next((i + 1 for i, d in enumerate(ranked) if rel.get(d, 0) > 0), None)


def correlation(x, y):
    return "n/a (constant)" if x.std() == 0 or y.std() == 0 else f"{np.corrcoef(x, y)[0, 1]:+.2f}"


def analyse(name, system):
    corpus, queries, qrels = testbed.load(name)
    folder = limitlib.HERE / "runs" / ("" if system == "colbert" else system)
    ours = json.load(open(folder / f"{name}.json"))
    base = testbed.baseline_run(name, "bge")
    qids, nd_ours, _ = testbed.score_run(ours, name)
    _, nd_base, _ = testbed.score_run(base, name)
    delta = nd_ours - nd_base
    lengths = np.array([len(queries[q].split()) for q in qids])
    n_rel = np.array([sum(g > 0 for g in qrels[q].values()) for q in qids])
    lines = [f"## {name} ({len(qids)} queries)", "",
             f"nDCG@10 {nd_ours.mean():.3f} vs bge {nd_base.mean():.3f}: better on {(delta > 0).sum()}, "
             f"worse on {(delta < 0).sum()}, equal on {(delta == 0).sum()} queries.", ""]
    terciles = np.quantile(lengths, [1 / 3, 2 / 3])
    groups = np.digitize(lengths, terciles, right=True)
    rows = [f"| {label} | {(groups == g).sum()} | {delta[groups == g].mean():+.3f} |"
            for g, label in enumerate([f"≤ {terciles[0]:.0f} words", f"≤ {terciles[1]:.0f} words",
                                       f"> {terciles[1]:.0f} words"]) if (groups == g).any()]
    lines += ["| Query length | Queries | Mean Δ nDCG@10 |", "|---|:-:|:-:|", *rows, "",
              f"Correlation of Δ with query length: {correlation(lengths, delta)}; "
              f"with the number of relevant documents: {correlation(n_rel, delta)}.", ""]
    order = np.argsort(delta)
    for title, picks in [("Largest gains", order[::-1][:TOP]), ("Largest losses", order[:TOP])]:
        lines += [f"**{title}**", "", "| Δ | Query | First relevant: bge → ours | Relevant document |", "|:-:|---|:-:|---|"]
        for i in picks:
            q = qids[i]
            rel = qrels[q]
            doc = next(d for d, g in rel.items() if g > 0)
            text = " ".join((corpus[doc]["title"] + " " + corpus[doc]["text"]).split()[:20]).replace("|", "/")
            query = " ".join(queries[q].split()[:30]).replace("|", "/")
            own = q if name == "arguana" else None
            lines.append(f"| {delta[i]:+.2f} | {query} | {first_relevant_rank(base[q], rel, own)} → "
                         f"{first_relevant_rank(ours[q], rel, own)} | {text} … |")
        lines.append("")
    return lines


if __name__ == "__main__":
    system = sys.argv[1] if len(sys.argv) > 1 else "colbert"
    label = {"colbert": "LIMIT (GTE-ModernColBERT, MaxSim)", "qwen3": "LIMIT (Qwen3-Embedding-0.6B)"}[system]
    out = [f"# Error analysis: {label} vs bge-small-en-v1.5", "",
           "Per-query nDCG@10 from testbed.score_run; ranks are those of the first relevant document (None: not in the top 101).", ""]
    for name in testbed.DATASETS:
        out += analyse(name, system)
    path = limitlib.OUT / f"error_analysis_{system}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(out), encoding="utf-8")
    print(f"wrote {path}")
