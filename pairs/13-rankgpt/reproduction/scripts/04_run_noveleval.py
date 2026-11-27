"""Step 4: NovelEval (Table 3) with a listwise LLM: 21 questions x 20 passages, one window per question.

Input orders: 'shuffled' (paper protocol; one run per seed), 'bm25' (BM25 order from step 1), 'released' (Google order).

Usage:
  python scripts/04_run_noveleval.py --model castorini/rank_zephyr_7b_v1_full --quant 4bit --style zephyr \
      --tag rankzephyr --seeds 0 1 2 --orders shuffled bm25
"""
import argparse
import random
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rankrepro.common import CACHE, RUNS, RESULTS, ensure_dirs, jdump, write_jsonl
from rankrepro.data import load_noveleval
from rankrepro.bm25 import read_trec_run, write_trec_run
from rankrepro.evaluate import evaluate_run
from rankrepro.listwise import WindowCache, sliding_window_rerank, summarize_logs

PAPER = {"bm25": [33.33, 45.96, 55.77], "monoT5-3B": [83.33, 78.38, 84.62], "gpt-3.5-turbo": [76.19, 74.15, 75.71],
         "gpt-4": [85.71, 87.49, 90.45], "RankZephyr (its paper, Table 8)": [92.86, 86.15, 89.34]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="castorini/rank_zephyr_7b_v1_full")
    ap.add_argument("--quant", default="4bit", choices=["4bit", "8bit", "none"])
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--style", default="zephyr", choices=["zephyr", "rankgpt", "plain"])
    ap.add_argument("--passage_words", type=int, default=None)
    ap.add_argument("--context_limit", type=int, default=4096, help="passages are shortened so that each prompt fits (RankZephyr: 4096); 0 = no limit")
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    ap.add_argument("--orders", nargs="*", default=["shuffled", "bm25"])
    ap.add_argument("--tag", default="rankzephyr")
    ap.add_argument("--device_map", default="cuda")
    args = ap.parse_args()
    ensure_dirs()

    from rankrepro.llm_backend import HFChatLLM
    queries, qrels, cands = load_noveleval()
    bm25 = read_trec_run(RUNS / "noveleval" / "bm25.trec") if "bm25" in args.orders else {}
    cache = WindowCache(CACHE / "listwise" / f"{args.tag}_noveleval.jsonl")
    llm = HFChatLLM(args.model, quant=args.quant, dtype=args.dtype,
                    device_map=({"": 0} if args.device_map == "cuda" else "auto"))
    print("tokenizer ready; model weights load on the first uncached window", flush=True)

    report = {"model": args.model, "quant": args.quant, "style": args.style, "paper": PAPER, "orders": {}}
    for order in args.orders:
        seeds = args.seeds if order == "shuffled" else [None]
        per_seed = []
        for seed in seeds:
            run, logs, t0 = {}, [], time.time()
            for q in queries:
                hits = list(cands[q])
                if order == "shuffled":
                    random.Random(seed * 1000 + int(q)).shuffle(hits)
                elif order == "bm25":
                    pos = {d: i for i, (d, _) in enumerate(bm25[q])}
                    hits.sort(key=lambda h: pos.get(h["docid"], 10 ** 6))
                new_hits, lg = sliding_window_rerank(llm, queries[q], hits, window=20, step=10, style=args.style,
                                                     passage_words=args.passage_words, cache=cache, model_tag=f"{args.model}|{args.quant}|{args.style}", context_limit=args.context_limit or None)
                run[q] = [(h["docid"], float(len(new_hits) - r)) for r, h in enumerate(new_hits)]
                for x in lg:
                    x.update(qid=q, seed=seed, order=order)
                logs += lg
            ev = evaluate_run(run, qrels)
            name = f"{args.tag}_{order}" + (f"_seed{seed}" if seed is not None else "")
            write_trec_run(run, RUNS / "noveleval" / f"{name}.trec", tag=name)
            write_jsonl(logs, RESULTS / f"noveleval_{name}_windows.jsonl")
            s = summarize_logs(logs)
            per_seed.append({"seed": seed, "nDCG@1": ev["nDCG@1"], "nDCG@5": ev["nDCG@5"], "nDCG@10": ev["nDCG@10"],
                             "per_query_ndcg10": ev["per_query"]["10"], "seconds": time.time() - t0, **s})
            print(f"[noveleval] {name}: nDCG@1/5/10 = {ev['nDCG@1']*100:.2f}/{ev['nDCG@5']*100:.2f}/{ev['nDCG@10']*100:.2f} "
                  f"well-formed {s['well_formed_rate']*100:.0f}% missing-ids {s['missing_ids_total']} dup {s['duplicate_ids_total']}")
        agg = {}
        for k in ("nDCG@1", "nDCG@5", "nDCG@10"):
            vals = [r[k] for r in per_seed]
            agg[k] = {"mean": statistics.mean(vals), "std": statistics.pstdev(vals) if len(vals) > 1 else 0.0}
        report["orders"][order] = {"runs": per_seed, "aggregate": agg}
        print(f"[noveleval] {order}: mean nDCG@1/5/10 = {agg['nDCG@1']['mean']*100:.2f}/{agg['nDCG@5']['mean']*100:.2f}/{agg['nDCG@10']['mean']*100:.2f}")
    jdump(report, RESULTS / f"noveleval_{args.tag}.json")


if __name__ == "__main__":
    main()
