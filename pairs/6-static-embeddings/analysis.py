"""Reads results.json, results_variants/*.json, latency.json and the local runs; writes results_variants/summary.json,
summary.md, errors.md and figures/. Usage: python pairs/6-static-embeddings/analysis.py"""
import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))
import testbed
from variants import VARIANTS

RV, FIG = HERE / "results_variants", HERE / "figures"
FIG.mkdir(exist_ok=True)
DS = testbed.DATASETS
BASE = testbed.stored()
OFFSET = {"mpnet": (5, -14), "e5": (5, 5), "minilm": (5, 6)}            # year panel
OFFSET_LAT = {"mpnet": (5, -14), "e5": (2, -17), "minilm": (5, 6)}   # latency panel
COLORS = {"sparse lexical": "#795548", "static (word vectors)": "#9e9e9e", "transformer": "#43a047", "LLM-based": "#1b5e20",
          "static (distilled)": "#64b5f6", "static (distilled + Tokenlearn)": "#1e88e5", "static (contrastive)": "#e53935",
          "static (this pair)": "#fb8c00", "hybrid": "#6a1b9a"}

# ---------------------------------------------------------------- all rows and Holm correction
rows = {  # testbed baselines (no comparison of their own)
    "bm25": {"label": "BM25", "year": 1994, "family": "sparse lexical"},
    "minilm": {"label": "all-MiniLM-L6-v2 (spring)", "year": 2021, "family": "transformer"},
    "bge": {"label": "bge-small-en-v1.5", "year": 2023, "family": "transformer"},
}
for k in rows:
    rows[k]["datasets"] = {d: {"ndcg10": float(np.mean(BASE[d][k]["ndcg10"])), "recall100": float(np.mean(BASE[d][k]["recall100"]))} for d in DS}
official = json.load(open(HERE / "results.json"))
rows["official"] = {"label": "static-retrieval-mrl-en-v1 (official row)", "year": 2025, "family": "static (contrastive)", "datasets": official["datasets"]}
for name, (label, year, family, _) in VARIANTS.items():
    p = RV / f"{name}.json"
    if p.exists():
        rows[name] = {"label": label, "year": year, "family": family, "datasets": json.load(open(p))["datasets"]}

# Direct pairwise comparisons that the story needs (also in the Holm family below).
PAIRS = [("official", "bm25"), ("potion_ret32m", "official"), ("ours_trained", "official"),
         ("rrf_bm25_static", "bm25"), ("static_d384", "official"), ("qwen3", "bge")]


def per_query_ndcg(name, d):
    if name in BASE[d]:
        return np.array(BASE[d][name]["ndcg10"])
    path = HERE / "runs" / f"{d}.json" if name == "official" else HERE / "runs_variants" / name / f"{d}.json"
    return testbed.score_run(json.load(open(path)), d)[1]


pairwise = {f"{a}_vs_{b}": {d: testbed.compare(per_query_ndcg(a, d), per_query_ndcg(b, d)) for d in DS}
            for a, b in PAIRS if a in rows and b in rows}

tests = [(n, d) for n, r in rows.items() for d in DS if "ndcg10_vs_baseline" in r["datasets"][d]]
pvals = [rows[n]["datasets"][d]["ndcg10_vs_baseline"]["p"] for n, d in tests]
pair_tests = [(k, d) for k in pairwise for d in DS]
pvals = np.array(pvals + [pairwise[k][d]["p"] for k, d in pair_tests])


def holm_adjust(p):
    """Holm step-down: p_(i) * (m - i), made monotone."""
    p = np.asarray(p, dtype=float)
    out, running = np.empty_like(p), 0.0
    for rank, i in enumerate(np.argsort(p)):
        running = max(running, min(1.0, p[i] * (len(p) - rank)))
        out[i] = running
    return out


# Two families: the pre-specified primary test (the official row: 3 datasets) and everything exploratory.
is_primary = np.array([n == "official" for n, _ in tests] + [False] * len(pair_tests))
holm = np.empty_like(pvals)
holm[is_primary] = holm_adjust(pvals[is_primary])
holm[~is_primary] = holm_adjust(pvals[~is_primary])
for (n, d), ph in zip(tests, holm[: len(tests)]):
    rows[n]["datasets"][d]["ndcg10_vs_baseline"]["p_holm"] = float(ph)
for (k, d), ph in zip(pair_tests, holm[len(tests):]):
    pairwise[k][d]["p_holm"] = float(ph)
N_PRIMARY, N_EXPLORATORY = int(is_primary.sum()), int((~is_primary).sum())

latency = json.load(open(RV / "latency.json")) if (RV / "latency.json").exists() else None
summary = {"n_tests_primary": N_PRIMARY, "n_tests_exploratory": N_EXPLORATORY, "rows": rows, "pairwise": pairwise, "latency": latency}
for k, v in pairwise.items():
    print(k, "  ".join(f"{d}: {v[d]['delta']:+.3f} {v[d]['ci95']} p={v[d]['p']:.2g} holm={v[d]['p_holm']:.2g}" for d in DS))

# ---------------------------------------------------------------- per-query error analysis of the official row
NEG = re.compile(r"\b(not|no|never|without|n't|cannot|neither|nor|lack|fail\w*|absent|unrelated|independent)\b", re.I)
NUM = re.compile(r"\d")


def per_query(run_path, name):
    return testbed.score_run(json.load(open(run_path)), name)


def first_rel_rank(run, q, qrels):
    ranked = sorted(run[q], key=run[q].get, reverse=True)
    for i, d in enumerate(ranked, 1):
        if qrels[q].get(d, 0) > 0:
            return i
    return None


errors = {}
md = ["# Error analysis: static-retrieval-mrl-en-v1 vs MiniLM and BM25 (per-query nDCG@10)", ""]
for d in DS:
    corpus, queries, qrels = testbed.load(d)
    qids, nd_static, _ = per_query(HERE / "runs" / f"{d}.json", d)
    nd_minilm, nd_bm25 = np.array(BASE[d]["minilm"]["ndcg10"]), np.array(BASE[d]["bm25"]["ndcg10"])
    delta_m, delta_b = nd_static - nd_minilm, nd_static - nd_bm25
    qtext = [queries[q] for q in qids]
    length = np.array([len(t.split()) for t in qtext])
    neg = np.array([bool(NEG.search(t)) for t in qtext])
    num = np.array([bool(NUM.search(t)) for t in qtext])
    rho, p_rho = stats.spearmanr(length, delta_m)
    # Are the static model's successes and failures those of BM25 (a learned bag of words) or of MiniLM?
    corr = {"static~bm25": float(stats.pearsonr(nd_static, nd_bm25)[0]), "static~minilm": float(stats.pearsonr(nd_static, nd_minilm)[0]),
            "minilm~bm25": float(stats.pearsonr(nd_minilm, nd_bm25)[0])}
    mw = lambda mask: float(stats.mannwhitneyu(delta_m[mask], delta_m[~mask]).pvalue) if 0 < mask.sum() < len(mask) else None
    e = {"wins_vs_minilm": int((delta_m > 0).sum()), "losses_vs_minilm": int((delta_m < 0).sum()), "ties": int((delta_m == 0).sum()),
         "spearman_len_vs_delta": float(rho), "spearman_p": float(p_rho),
         "mean_delta_negation": float(delta_m[neg].mean()) if neg.any() else None, "n_negation": int(neg.sum()),
         "mean_delta_no_negation": float(delta_m[~neg].mean()),
         "mean_delta_with_digits": float(delta_m[num].mean()) if num.any() else None, "n_digits": int(num.sum()),
         "mean_delta_without_digits": float(delta_m[~num].mean()),
         "mannwhitney_p_negation": mw(neg), "mannwhitney_p_digits": mw(num), "per_query_pearson": corr}
    errors[d] = e
    md += [f"## {d}", "", f"static beats MiniLM on {e['wins_vs_minilm']} queries, loses on {e['losses_vs_minilm']}, ties on {e['ties']}. "
           f"Spearman(query length, Δ) = {rho:.2f} (p = {p_rho:.2g}). Mean Δ with a negation word: "
           f"{e['mean_delta_negation'] if e['mean_delta_negation'] is not None else float('nan'):+.3f} (n = {e['n_negation']}) vs {e['mean_delta_no_negation']:+.3f}; "
           f"with digits: {e['mean_delta_with_digits'] if e['mean_delta_with_digits'] is not None else float('nan'):+.3f} (n = {e['n_digits']}) vs {e['mean_delta_without_digits']:+.3f} "
           f"(Mann-Whitney p: negation {e['mannwhitney_p_negation']}, digits {e['mannwhitney_p_digits']}). "
           f"Per-query Pearson r: static~BM25 {corr['static~bm25']:.2f}, static~MiniLM {corr['static~minilm']:.2f}, MiniLM~BM25 {corr['minilm~bm25']:.2f}.", ""]
    if d == "arguana":
        continue  # ArguAna queries are whole arguments: listing them is not readable
    runs = {"static": json.load(open(HERE / "runs" / f"{d}.json")),
            "minilm": testbed.baseline_run(d, "minilm"), "bm25": testbed.baseline_run(d, "bm25")}
    for title, idx in [("Biggest wins vs MiniLM", np.argsort(-delta_m)[:8]), ("Biggest losses vs MiniLM", np.argsort(delta_m)[:8])]:
        md += [f"### {title}", "", "| Δ vs MiniLM | Δ vs BM25 | first relevant rank (static / MiniLM / BM25) | query |", "|---|---|---|---|"]
        for i in idx:
            q = qids[i]
            ranks = " / ".join(str(first_rel_rank(runs[s], q, qrels) or ">100") for s in ["static", "minilm", "bm25"])
            md.append(f"| {delta_m[i]:+.2f} | {delta_b[i]:+.2f} | {ranks} | {qtext[i]} |")
        md.append("")
summary["errors"] = errors
json.dump(summary, open(RV / "summary.json", "w"), indent=1)
(RV / "errors.md").write_text("\n".join(md) + "\n")

# ---------------------------------------------------------------- summary table (markdown)
def cell(r, d):
    x = r["datasets"][d]
    s = f"{x['ndcg10']:.3f}"
    c = x.get("ndcg10_vs_baseline")
    if c:
        star = "**" if c.get("p_holm", 1) < 0.05 and (c["ci95"][0] > 0 or c["ci95"][1] < 0) else ""
        s += f" ({star}{c['delta']:+.3f}{star}, p<sub>Holm</sub>={c['p_holm']:.2g})"
    return s


lines = [f"Δ vs `minilm` (RRF row: vs `rrf`); bold = Holm-corrected p < 0.05 and CI excluding 0. Holm families: the official row ({N_PRIMARY} tests, pre-specified) and the exploratory rest ({N_EXPLORATORY} tests).", "",
         "| System | Year | Family | SciFact | NFCorpus | ArguAna | mean | CPU ms/query |", "|---|---|---|---|---|---|---|---|"]
lat = (latency or {}).get("systems", {})
for n, r in sorted(rows.items(), key=lambda x: -np.mean([x[1]["datasets"][d]["ndcg10"] for d in DS])):
    ms = lat.get(n, {}).get("query_ms_median")
    lines.append(f"| {r['label']} | {r['year']} | {r['family']} | " + " | ".join(cell(r, d) for d in DS)
                 + f" | {np.mean([r['datasets'][d]['ndcg10'] for d in DS]):.3f} | {f'{ms:.1f}' if ms else '–'} |")
(RV / "summary.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))

# ---------------------------------------------------------------- figures
ERA = ["bm25", "glove", "minilm", "mpnet", "contriever", "e5", "bge", "m2v_base", "potion8m", "potion_ret32m", "official", "qwen3", "ours_m2v_bge", "ours_trained"]
era = [n for n in ERA if n in rows]
mean = {n: np.mean([rows[n]["datasets"][d]["ndcg10"] for d in DS]) for n in era}
plt.rcParams.update({"font.size": 12}); fig, axes = plt.subplots(1, 2 if lat else 1, figsize=(14 if lat else 7, 6.2), squeeze=False)
ax = axes[0, 0]
ax.axhline(mean["bm25"], color=COLORS["sparse lexical"], ls="--", lw=1.2, label="BM25 (1994): reference line")
for n in era:
    if n == "bm25":
        continue
    r = rows[n]
    ax.scatter(r["year"], mean[n], s=90, color=COLORS[r["family"]], edgecolor="k", zorder=3, label=r["family"])
    ax.annotate(r["label"].replace(" (official row)", "").replace("Model2Vec ", ""), (r["year"], mean[n]), textcoords="offset points", xytext=OFFSET.get(n, (5, 4)), fontsize=9.5)
ax.set_xlabel("release year"); ax.set_ylabel("nDCG@10, mean of SciFact · NFCorpus · ArguAna"); ax.set_title("One model per era on the testbed")
ax.grid(alpha=.3)
h, l = ax.get_legend_handles_labels(); u = dict(zip(l, h)); ax.legend(u.values(), u.keys(), fontsize=9, loc="upper left", framealpha=0.9)
ax.set_xlim(2013, 2027)
if lat:
    ax = axes[0, 1]
    for n in era:
        key = "official" if n == "official" else n
        if key in lat:
            r = rows[n]
            ax.scatter(lat[key]["query_ms_median"], mean[n], s=90, color=COLORS[r["family"]], edgecolor="k", zorder=3)
            ax.annotate(r["label"].replace(" (official row)", "").replace("Model2Vec ", ""), (lat[key]["query_ms_median"], mean[n]), textcoords="offset points", xytext=OFFSET_LAT.get(n, (5, 4)), fontsize=9.5)
    ax.set_xscale("log"); ax.set_xlabel(f"CPU latency per query, ms (batch 1, encode + search 5k docs; {latency['torch_threads']} threads)")
    ax.set_ylabel("nDCG@10 (mean of 3 datasets)"); ax.set_title("Quality vs CPU latency"); ax.grid(alpha=.3, which="both")
fig.tight_layout(); fig.savefig(FIG / "era_testbed.png", dpi=170)

dims = [d for d in (1024, 512, 384, 256, 128, 64)]
names = {1024: "official", **{d: f"static_d{d}" for d in dims[1:]}}
if all(names[d] in rows for d in dims):
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for d, c in zip(DS, ["#1e88e5", "#43a047", "#e53935"]):
        ax.plot(dims, [rows[names[k]]["datasets"][d]["ndcg10"] for k in dims], marker="o", color=c, label=f"static · {d}")
        ax.axhline(rows["minilm"]["datasets"][d]["ndcg10"], color=c, ls="--", lw=1)
    ax.axvline(384, color="grey", ls=":", lw=1); ax.text(390, ax.get_ylim()[0] + 0.01, "384 = MiniLM index size", fontsize=7, color="grey")
    ax.set_xscale("log", base=2); ax.set_xticks(dims, [str(d) for d in dims]); ax.set_xlabel("dimensions kept (Matryoshka prefix)")
    ax.set_ylabel("nDCG@10"); ax.set_title("Truncating static-retrieval-mrl-en-v1 (dashed: MiniLM)"); ax.legend(fontsize=7); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(FIG / "matryoshka_testbed.png", dpi=170)
print(json.dumps(errors, indent=1))
