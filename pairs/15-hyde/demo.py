import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import testbed
import hyde


def show_query(name, qid, top_k=5):
    corpus, queries, qrels = testbed.load(name)
    bge_run = testbed.baseline_run(name, "bge")
    hyde_run = json.load(open(HERE / "runs" / f"{name}.json"))
    gens = hyde.load_generations(name, "generations")

    print("=" * 70)
    print(f"[{name}] QID={qid} | Query: {queries[qid]}")
    print(f"Hypothetical Doc #1 (truncated): {' '.join(gens[qid][0].split()[:45])}...")
    gold = [d for d, g in qrels[qid].items() if g > 0]
    print(f"Relevant documents ({len(gold)}): {', '.join(gold[:3])}{' ...' if len(gold) > 3 else ''}")
    print("-" * 70)

    bge_top = sorted(bge_run[qid], key=bge_run[qid].get, reverse=True)
    hyde_top = sorted(hyde_run[qid], key=hyde_run[qid].get, reverse=True)
    if name == "arguana":
        bge_top = [d for d in bge_top if d != qid]
        hyde_top = [d for d in hyde_top if d != qid]

    print(f"{'Rank':<5} | {'BGE-small (Baseline)':<32} | {'HyDE + BGE-small':<32}")
    print("-" * 70)
    for i in range(top_k):
        bd, hd = bge_top[i], hyde_top[i]
        b_rel = "✓" if qrels[qid].get(bd, 0) > 0 else "✗"
        h_rel = "✓" if qrels[qid].get(hd, 0) > 0 else "✗"
        b_title = (corpus[bd]["text"] if name == "arguana" else corpus[bd]["title"] or corpus[bd]["text"])[:30].replace("\n", " ")
        h_title = (corpus[hd]["text"] if name == "arguana" else corpus[hd]["title"] or corpus[hd]["text"])[:30].replace("\n", " ")
        print(f"#{i+1:<4} | {b_rel} {b_title:<30} | {h_rel} {h_title:<30}")


def find_examples(name):
    _, _, qrels = testbed.load(name)
    bge_run = testbed.baseline_run(name, "bge")
    hyde_run = json.load(open(HERE / "runs" / f"{name}.json"))
    qids, b_nd, _ = testbed.score_run(bge_run, name)
    _, h_nd, _ = testbed.score_run(hyde_run, name)
    diff = h_nd - b_nd
    hero_idx = int(diff.argmax())
    fail_idx = int(diff.argmin())
    print(f"\n>>> {name}: HERO qid={qids[hero_idx]} (delta={diff[hero_idx]:+.3f}), FAIL qid={qids[fail_idx]} (delta={diff[fail_idx]:+.3f})")
    return qids[hero_idx], qids[fail_idx]


if __name__ == "__main__":
    if len(sys.argv) == 3:
        show_query(sys.argv[1], sys.argv[2])
    else:
        hero_qid, _ = find_examples("nfcorpus")
        show_query("nfcorpus", hero_qid)
        _, fail_qid = find_examples("arguana")
        show_query("arguana", fail_qid)
