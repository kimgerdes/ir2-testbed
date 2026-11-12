"""Free-embedding optimisation (LIMIT paper, Section 4), ported from google-deepmind/limit
code/free_embedding_experiment.py (Apache 2.0) to PyTorch so that it runs on any CUDA GPU, Windows included.

Same algorithm and hyper-parameters as the official script.
Changes: the accuracy check runs on the GPU (the official one loops over queries in Python), and besides the
paper's synthetic qrels the qrels can come from a real dataset, with a variable number of relevant documents.

  python pairs/04-limit/scripts/free_embedding.py critical-n --d 4 [--k 2]        paper's search: the largest n that d can fit (Table 6)
  python pairs/04-limit/scripts/free_embedding.py min-d --qrels scifact           smallest d that fits a dataset's test qrels exactly
  python pairs/04-limit/scripts/free_embedding.py fit --qrels scifact --d 8       one fit (min-d reuses it: run a grid of these in parallel)
Results: outputs/free_embedding/<setting>/<n or d>.json, plus summary.json for a search; --seed S (not 42) adds _seedS."""
import argparse
import itertools
import json
import math
import time

import numpy as np
import torch

import limitlib

DEFAULTS = dict(lr=0.01, temperature=0.1, max_iters=100_000, patience=1000, min_delta=1e-5, log_every=50, seed=42)
QRELS = ["scifact", "nfcorpus", "arguana", "limit-small", "limit"]


def synthetic_qrels(n, k):
    """The paper's setting: one query for every k-subset of n documents."""
    combos = list(itertools.combinations(range(n), k))
    rel = torch.zeros(len(combos), n, dtype=torch.bool)
    rows = torch.arange(len(combos)).repeat_interleave(k)
    rel[rows, torch.tensor(combos).reshape(-1)] = True
    return rel


def dataset_qrels(name):
    """A dataset's test qrels as a [queries, documents] relevance matrix over its whole corpus (grade > 0)."""
    corpus, _, qrels = limitlib.load_any(name)
    col = {d: j for j, d in enumerate(corpus)}
    pairs = [(i, col[d]) for i, rel in enumerate(qrels.values()) for d, g in rel.items() if g > 0 and d in col]
    rel = torch.zeros(len(qrels), len(corpus), dtype=torch.bool)
    rows, cols = zip(*pairs)
    rel[list(rows), list(cols)] = True
    return rel[rel.any(dim=1)]


def accuracy(Q, D, rel, k_q, kmax):
    """Share of relevant documents found in each query's top-|relevant| (the official metric), on the GPU."""
    top = (Q @ D.T).topk(kmax, dim=1).indices
    hit = rel.gather(1, top) & (torch.arange(kmax, device=rel.device)[None, :] < k_q[:, None])
    return float(hit.sum() / k_q.sum())


def fit(rel, d, lr, temperature, max_iters, patience, min_delta, log_every, seed):
    dev = limitlib.device()
    q, n = rel.shape
    g = torch.Generator().manual_seed(seed)
    Q = torch.randn(q, d, generator=g)
    D = torch.randn(n, d, generator=g)
    Q = (Q / Q.norm(dim=1, keepdim=True)).to(dev).requires_grad_()
    D = (D / D.norm(dim=1, keepdim=True)).to(dev).requires_grad_()
    rel = rel.to(dev)
    mask = rel.float()
    num_pos = mask.sum()
    k_q = rel.sum(dim=1)
    kmax = int(k_q.max())
    opt = torch.optim.Adam([Q, D], lr=lr)
    best, wait, max_acc, losses, accs = math.inf, 0, 0.0, [], []
    t0, i = time.time(), 0
    for i in range(max_iters):
        loss = -(torch.log_softmax(Q @ D.T / temperature, dim=1) * mask).sum() / num_pos
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        with torch.no_grad():
            Q.div_(Q.norm(dim=1, keepdim=True))
            D.div_(D.norm(dim=1, keepdim=True))
        value = loss.item()
        if value < best - min_delta:
            best, wait = value, 0
        else:
            wait += 1
            if wait >= patience:
                break
        if i % log_every == 0 or i == max_iters - 1:
            with torch.no_grad():
                acc = accuracy(Q, D, rel, k_q, kmax)
            losses.append(value)
            accs.append(acc)
            max_acc = max(max_acc, acc)
            if acc >= 1.0:
                break
    with torch.no_grad():
        final = accuracy(Q, D, rel, k_q, kmax)
    keep = np.linspace(0, len(losses) - 1, min(len(losses), 100)).astype(int).tolist() if losses else []
    return {"success": max(max_acc, final) >= 1.0, "max_accuracy": max(max_acc, final), "final_accuracy": final,
            "final_loss": losses[-1] if losses else None, "iterations": i + 1, "seconds": round(time.time() - t0, 1),
            "device": dev, "losses": [losses[j] for j in keep], "accuracies": [accs[j] for j in keep]}


def cached_fit(path, params, rel_fn):
    """Runs one fit unless the same parameters were already run (by this process or by a parallel grid job)."""
    if path.exists():
        old = json.load(open(path, encoding="utf-8"))
        if old.get("parameters") == params:
            return old
    rel = rel_fn()
    hp = {k: params[k] for k in DEFAULTS}
    result = {"parameters": params, "n_queries": rel.shape[0], "n_docs": rel.shape[1],
              "n_relevant": int(rel.sum()), **fit(rel, params["d"], **hp)}
    limitlib.save_json(result, path)
    status = "fits" if result["success"] else "fails"
    print(f"{path.parent.name} d={params['d']} n={rel.shape[1]} q={rel.shape[0]}: {status} "
          f"(accuracy {result['max_accuracy']:.4f}, {result['iterations']} steps, {result['seconds']}s)", flush=True)
    return result


def folder_of(setting, args):
    """outputs/free_embedding/<setting>, with a _seed<s> suffix for any seed but the official 42."""
    return limitlib.OUT / "free_embedding" / (setting if args.seed == DEFAULTS["seed"] else f"{setting}_seed{args.seed}")


def hyper(args):
    return {k: getattr(args, k) for k in DEFAULTS}


def qrels_fit(args, d, rel_cache={}):
    params = {"qrels": args.qrels, "d": d, **hyper(args)}
    path = folder_of(args.qrels, args) / f"d{d}.json"
    def rel_fn():
        if args.qrels not in rel_cache:
            rel_cache[args.qrels] = dataset_qrels(args.qrels)
        return rel_cache[args.qrels]
    return cached_fit(path, params, rel_fn)


def cmd_fit(args):
    qrels_fit(args, args.d)


def cmd_min_d(args):
    """Doubling then binary search for the smallest d whose fit reaches 100 % (assumes monotonicity in d)."""
    ok = lambda d: qrels_fit(args, d)["success"]
    lo, hi = 0, 1 # invariant: lo fails (or is 0), hi is the candidate
    while not ok(hi):
        lo, hi = hi, hi * 2
        if hi > args.d_max:
            hi = None
            break
    if hi is not None:
        while hi - lo > 1:
            mid = (lo + hi) // 2
            lo, hi = (lo, mid) if ok(mid) else (mid, hi)
    folder = folder_of(args.qrels, args)
    runs = sorted((r for r in (json.load(open(p, encoding="utf-8")) for p in folder.glob("d*.json"))
                   if r["parameters"] == {"qrels": args.qrels, "d": r["parameters"]["d"], **hyper(args)}),
                  key=lambda r: r["parameters"]["d"])
    summary = {"qrels": args.qrels, "min_d": hi, "d_max": args.d_max, "hyper": hyper(args),
               "n_queries": runs[0]["n_queries"], "n_docs": runs[0]["n_docs"], "n_relevant": runs[0]["n_relevant"],
               "tried": {r["parameters"]["d"]: {"success": r["success"], "accuracy": r["max_accuracy"]} for r in runs}}
    limitlib.save_json(summary, folder / "summary.json")
    print(f"{args.qrels}: smallest d that fits the qrels exactly = {hi}", flush=True)


def paper_fit(d):
    """The paper's cubic fit of critical n against d (Figure 2): a starting point for the search."""
    return -10.5322 + 4.0309 * d + 0.0520 * d ** 2 + 0.0037 * d ** 3


def cmd_critical_n(args):
    """The official three-phase search: gallop from the initial n, binary search, then sweep n-4..n+5."""
    setting = f"synthetic_k{args.k}_d{args.d}"
    folder = folder_of(setting, args)
    def ok(n):
        params = {"synthetic_k": args.k, "n": n, "d": args.d, **hyper(args)}
        return cached_fit(folder / f"n{n}.json", params, lambda: synthetic_qrels(n, args.k))["success"]
    n0 = args.initial_n or max(args.k + 1, round(paper_fit(args.d)))
    low = high = None
    if ok(n0):
        low, step = n0, 1
        while ok(low + step):
            low, step = low + step, step * 2
        high = low + step
    else:
        high, step = n0, 1
        while high > args.k:
            test = max(high - step, args.k)
            if ok(test):
                low = test
                break
            high, step = test, step * 2
    critical = low
    if low is not None:
        a, b = low, high
        while a <= b:
            mid = (a + b) // 2
            if ok(mid):
                critical, a = mid, mid + 1
            else:
                b = mid - 1
    sweep = {}
    if critical is not None and not args.no_sweep:
        for n in range(max(critical - 4, args.k), critical + 6):
            sweep[n] = ok(n)
    limitlib.save_json({"d": args.d, "k": args.k, "critical_n": critical, "initial_n": n0, "sweep": sweep,
                        "paper_fit": round(paper_fit(args.d), 1), "hyper": hyper(args)}, folder / "summary.json")
    print(f"d={args.d} k={args.k}: critical n = {critical} (paper's fit {paper_fit(args.d):.0f})", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name, fn in [("fit", cmd_fit), ("min-d", cmd_min_d), ("critical-n", cmd_critical_n)]:
        p = sub.add_parser(name)
        p.set_defaults(run=fn)
        for key, value in DEFAULTS.items():
            p.add_argument("--" + key.replace("_", "-"), dest=key, type=type(value), default=value)
        if name in ("fit", "min-d"):
            p.add_argument("--qrels", required=True, choices=QRELS)
        if name == "fit":
            p.add_argument("--d", type=int, required=True)
        if name == "min-d":
            p.add_argument("--d-max", type=int, default=1024)
        if name == "critical-n":
            p.add_argument("--d", type=int, required=True)
            p.add_argument("--k", type=int, default=2)
            p.add_argument("--initial-n", type=int)
            p.add_argument("--no-sweep", action="store_true", help="skip phase 3 (n-4 .. n+5)")
    args = parser.parse_args()
    torch.manual_seed(limitlib.SEED)
    args.run(args)
