"""Why does GTE-ModernColBERT reach R@2 99.8 on LIMIT-small with us and 83.5 in the paper (Table 4)?
The same checkpoint is scored five ways; only the query-side settings and the scoring pipeline change:

  default         exact MaxSim, today's configuration: query length 48, no query expansion (our limit_eval row)
  qlen32          exact MaxSim, query length 32, no query expansion
  paper-era       exact MaxSim, the configuration when the paper was written: query length 32, and PyLate pads
                  every query with [MASK] tokens to 32 (expansion; switched off by the model's authors on 2025-09-09)
  mteb            the paper's evaluation pipeline, MTEB 2.27.7 (PLAID index, not exact), today's configuration
  mteb-paper-era  MTEB 2.27.7 with the paper-era configuration

  python pairs/04-limit/scripts/colbert_diagnosis.py --split limit-small [--variants default paper-era]
One file per variant: outputs/limit/<split>/colbert_diagnosis/<variant>.json."""
import argparse
import time

import limitlib

EXACT = ["default", "qlen32", "paper-era"]
MTEB = {"mteb": "default", "mteb-paper-era": "paper-era"}
MTEB_TASKS = {"limit-small": "LIMITSmallRetrieval", "limit": "LIMITRetrieval"}
KS = (2, 10, 20, 100)


def exact(split, variant):
    corpus, queries, qrels = limitlib.load_limit(split)
    D, Q = limitlib.colbert_embeddings(split, corpus, queries, variant=variant)
    run = limitlib.to_run(limitlib.maxsim_scores(Q, D), list(queries), list(corpus))
    return {k: round(float(v.mean()) * 100, 2) for k, v in limitlib.recall_at(run, qrels, KS).items()}, Q[0].shape[0]


def through_mteb(split, variant):
    import mteb
    settings = dict(limitlib.COLBERT_VARIANTS[variant], document_length=limitlib.COLBERT_DOC_LENGTH)
    model = mteb.get_model(limitlib.COLBERT, device=limitlib.device(),
                           index_dir=str(limitlib.CACHE / split / "mteb-index"), **settings)
    result = mteb.evaluate(model, tasks=mteb.get_tasks(tasks=[MTEB_TASKS[split]]), cache=None)
    scores = result.task_results[0].scores["test"][0]
    return {k: round(scores[f"recall_at_{k}"] * 100, 2) for k in KS}, None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", choices=limitlib.LIMIT_SPLITS, default="limit-small")
    parser.add_argument("--variants", nargs="+", choices=EXACT + list(MTEB), default=EXACT + list(MTEB))
    args = parser.parse_args()
    for variant in args.variants:
        t0 = time.time()
        recall, query_tokens = exact(args.split, variant) if variant in EXACT else through_mteb(args.split, MTEB[variant])
        settings = limitlib.COLBERT_VARIANTS[MTEB.get(variant, variant)]
        limitlib.save_json({"split": args.split, "variant": variant, "settings": settings or "model defaults",
                            "pipeline": "MTEB 2.27.7" if variant in MTEB else "exact MaxSim (limitlib)",
                            "query_tokens": query_tokens, "seconds": round(time.time() - t0, 1),
                            **{f"recall@{k}": v for k, v in recall.items()}},
                           limitlib.OUT / "limit" / args.split / "colbert_diagnosis" / f"{variant}.json")
        print(f"{args.split:11} {variant:15} " + "  ".join(f"R@{k} {v:5.1f}" for k, v in recall.items()), flush=True)
