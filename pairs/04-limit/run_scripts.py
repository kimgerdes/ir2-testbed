"""Runs the whole pair-04 pipeline, or some of its steps, in order. From the repo root:

  python pairs/04-limit/run_scripts.py                      every step (about 1 h on one GPU, 16 GB for Qwen3-8B)
  python pairs/04-limit/run_scripts.py testbed analysis     only these steps
  python pairs/04-limit/run_scripts.py --dry-run            print the commands without running them

Steps:
  testbed         method.py (GTE-ModernColBERT, then Qwen3-Embedding-0.6B) and testbed.py score -> results*.json
  analysis        per-query error analysis of both against bge-small (needs the testbed runs)
  limit           BM25, bge-small, Qwen3-0.6B, Qwen3-8B and ColBERT on LIMIT-small and LIMIT, then the ColBERT diagnosis
  free-embedding  critical n for d = 4..45 (Table 6) and the smallest d on real qrels, seeds 42 (official), 1 and 2
  notebook        executes reproduce.ipynb on outputs/
Every step reuses what earlier runs cached, so a stopped run resumes where it stopped."""

import argparse
import subprocess
import sys
from pathlib import Path

PAIR = Path(__file__).resolve().parent
ROOT = PAIR.parents[1]
PY = sys.executable
S = f"pairs/{PAIR.name}/scripts"
METHODS = {
    "colbert": ("LIMIT (GTE-ModernColBERT, MaxSim)", "runs", "results.json"),
    "qwen3": ("LIMIT (Qwen3-Embedding-0.6B)", "runs/qwen3", "results_qwen3.json")
}
CRITICAL_N_D = [4, 8, 12, 16, 20, 24, 30, 45]
EXTRA_SEEDS, EXTRA_SEED_D = [1, 2], [4, 8, 12, 16, 20]
QRELS = ["scifact", "nfcorpus", "arguana", "limit-small", "limit"]


def steps():
    """{step: [(command, working directory)]}"""
    pair = f"pairs/{PAIR.name}"
    testbed = []
    for system, (label, runs, out) in METHODS.items():
        testbed.append(([PY, f"{pair}/method.py"] + ([] if system == "colbert" else [system]), ROOT))
        testbed.append(([PY, "testbed.py", "score", f"{pair}/{runs}", "--method", label, "--pair", PAIR.name,
                         "--against", "bge", "--out", f"{pair}/{out}"], ROOT))
    limit = []
    for split in ["limit-small", "limit"]:
        limit.append(([PY, f"{S}/limit_eval.py", "--split", split, "--systems",
                       "bm25", "bge", "qwen3", "qwen3-8b", "colbert"], ROOT))
        limit.append(([PY, f"{S}/colbert_diagnosis.py", "--split", split], ROOT))
    free = [([PY, f"{S}/free_embedding.py", "critical-n", "--d", str(d)], ROOT) for d in CRITICAL_N_D]
    free += [([PY, f"{S}/free_embedding.py", "critical-n", "--d", str(d), "--seed", str(s)], ROOT)
             for s in EXTRA_SEEDS for d in EXTRA_SEED_D]
    free += [([PY, f"{S}/free_embedding.py", "min-d", "--qrels", q, "--seed", str(s)], ROOT)
             for s in [42] + EXTRA_SEEDS for q in QRELS]
    return {"testbed": testbed,
            "analysis": [([PY, f"{S}/analysis.py"], ROOT), ([PY, f"{S}/analysis.py", "qwen3"], ROOT)],
            "limit": limit,
            "free-embedding": free,
            "notebook": [([PY, "-m", "jupyter", "execute", "--inplace", "reproduce.ipynb"], PAIR)]}


def main():
    all_steps = steps()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("steps", nargs="*", metavar="step",
                        help=f"steps to run, always in the order above (default: all): {', '.join(all_steps)}")
    parser.add_argument("--dry-run", action="store_true", help="print the commands without running them")
    args = parser.parse_args()
    unknown = set(args.steps) - set(all_steps)
    if unknown:
        parser.error(f"unknown step(s) {', '.join(sorted(unknown))}; choose from {', '.join(all_steps)}")
    chosen = [s for s in all_steps if not args.steps or s in args.steps]
    for name in chosen:
        print(f"== {name}", flush=True)
        for cmd, cwd in all_steps[name]:
            print("  " + " ".join(c if " " not in c else f'"{c}"' for c in ["python"] + cmd[1:]), flush=True)
            if not args.dry_run:
                subprocess.run(cmd, cwd=cwd, check=True)


if __name__ == "__main__":
    main()
