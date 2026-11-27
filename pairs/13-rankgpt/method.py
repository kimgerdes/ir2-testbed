"""RankGPT (Sun et al., EMNLP 2023) on the IR 2 testbed.

Instructional permutation generation with sliding windows (w=20, s=10) over the BM25 top-50 (ArguAna: top-20, the
query's own document removed first), using RankZephyr-7B in 4-bit as the open-weights stand-in for GPT-4, or one of
the MiniLM-L6 cross-encoders distilled from ChatGPT's permutations. Writes runs/<dataset>.json in the testbed format:
the re-ranked candidates on top, the untouched BM25 tail below, 100 documents per query.

Usage, from the testbed root:
  python pairs/13-rankgpt/method.py --system rankzephyr [--datasets scifact nfcorpus arguana] [--limit N]
  python pairs/13-rankgpt/method.py --system minilm_chatgpt
  python testbed.py score pairs/13-rankgpt/runs --method "RankGPT (RankZephyr-7B 4-bit; BM25 top-50, ArguAna top-20)" --pair 13-rankgpt --against ce

The implementation lives in the rankrepro package: the copy in reproduction/src, or our research repository
(../../../rankgpt-repro, or RANKGPT_REPRO) when it is present. Every LLM window is cached in
cache/<system>_<dataset>.jsonl: interrupted runs resume, and the submitted runs replay without a GPU.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
TESTBED = HERE.parents[1]
REPRO = Path(os.environ.get("RANKGPT_REPRO", TESTBED.parent / "rankgpt-repro"))
# rankrepro: our research repository when it is next to the testbed (its hf_cache holds the models on our machines),
# otherwise the copy in reproduction/src that ships with this folder.
for cand in (REPRO / "src", HERE / "reproduction" / "src"):
    if (cand / "rankrepro").is_dir():
        sys.path.insert(0, str(cand))
        break
sys.path.insert(0, str(TESTBED))
import testbed  # noqa: E402  (the shared script: data, baseline runs, doc texts)

DEPTH = {"scifact": 50, "nfcorpus": 50, "arguana": 20}
STUDENTS = {"minilm_chatgpt": "models/minilm_chatgpt", "minilm_marco_binary": "models/minilm_marco_binary"}


def candidates(name):
    corpus, queries, _ = testbed.load(name)
    texts = dict(zip(corpus, testbed.doc_texts(corpus)))
    run = testbed.baseline_run(name, "bm25")
    out = {}
    for q, text in queries.items():
        ranked = sorted(run[q], key=run[q].get, reverse=True)
        if name == "arguana":
            ranked = [d for d in ranked if d != q]
        out[q] = (" ".join(text.split()), [{"docid": d, "text": texts[d]} for d in ranked[: testbed.K]])
    return out


def write_run(order, cands, path):
    run = {}
    for q, (_, hits) in cands.items():
        docs = list(order.get(q, [])) + [h["docid"] for h in hits if h["docid"] not in set(order.get(q, []))]
        run[q] = {d: float(len(docs) - i) for i, d in enumerate(docs)}
    path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(run, open(path, "w"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", default="rankzephyr", choices=["rankzephyr"] + list(STUDENTS))
    ap.add_argument("--datasets", nargs="*", default=testbed.DATASETS)
    ap.add_argument("--model", default="castorini/rank_zephyr_7b_v1_full")
    ap.add_argument("--quant", default="4bit", choices=["4bit", "8bit", "none"])
    ap.add_argument("--limit", type=int, default=None, help="first N queries only (the shared scorer needs all of them)")
    ap.add_argument("--out", default=str(HERE / "runs"))
    args = ap.parse_args()

    if args.system == "rankzephyr":
        from rankrepro.llm_backend import HFChatLLM
        from rankrepro.listwise import WindowCache, sliding_window_rerank, summarize_logs
        llm = HFChatLLM(args.model, quant=args.quant, device_map={"": 0})
    else:
        from rankrepro.crossencoder import CrossEncoderScorer
        if not (REPRO / STUDENTS[args.system]).is_dir():
            sys.exit(f"{args.system}: the distilled students (175 MB) are not part of this folder. They are trained by "
                     "scripts/05_distill.py of our research repository; reproduce.ipynb shows their testbed scores.")
        scorer = CrossEncoderScorer(REPRO / STUDENTS[args.system])

    for name in args.datasets:
        cands = candidates(name)
        qids = list(cands)[: args.limit] if args.limit else list(cands)
        depth, order, t0 = DEPTH[name], {}, time.time()
        if args.system == "rankzephyr":
            cache = WindowCache(HERE / "cache" / f"{args.system}_{name}.jsonl")
            logs = []
            for i, q in enumerate(qids, 1):
                query, hits = cands[q]
                new_hits, lg = sliding_window_rerank(llm, query, hits[:depth], window=20, step=10, style="zephyr",
                                                     cache=cache, model_tag=f"{args.model}|{args.quant}|zephyr")
                order[q] = [h["docid"] for h in new_hits]
                logs += lg
                if i % 20 == 0 or i == len(qids):
                    s = summarize_logs(logs)
                    print(f"{name}: {i}/{len(qids)} queries, well-formed {s['well_formed_rate']*100:.0f}%, "
                          f"{(time.time() - t0) / i:.1f}s/query", flush=True)
        else:
            for q in qids:
                query, hits = cands[q]
                s = scorer.score(query, [h["text"] for h in hits[:depth]])
                order[q] = [hits[i]["docid"] for i in sorted(range(len(s)), key=lambda i: -s[i])]
        write_run(order, {q: cands[q] for q in qids}, Path(args.out) / f"{name}.json")
        print(f"{name}: wrote {len(qids)} queries ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
