"""Session demo for pair 13 (RankGPT): one query, three rankings side by side - BM25 top-5, the testbed's cross-encoder
`ce` top-5 and RankZephyr-7B top-5 - plus the raw permutation RankZephyr wrote for its top window.

Usage, from the testbed root (the environment of method.py, see README.md):
  python pairs/13-rankgpt/demo.py hero | fail | wild              # the queries of demo.md: cached windows, no GPU needed
  python pairs/13-rankgpt/demo.py --dataset scifact --qid 1278    # any test query (cached once method.py has run)
  python pairs/13-rankgpt/demo.py --dataset scifact --query "a claim typed by the room"     # one live query
  python pairs/13-rankgpt/demo.py --dataset scifact --interactive                           # several live queries
Live = BM25 over the corpus, ce on its top-100 and one RankZephyr window on its top-20. The BM25 index and the 7B model
are loaded once; after that each query costs one LLM window.
"""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import method    # noqa: E402  (puts the testbed and rankrepro on sys.path)
import testbed   # noqa: E402
from rankrepro.listwise import sliding_window_rerank, WindowCache   # noqa: E402

SHORTCUTS = {"hero": ("scifact", "1278"), "fail": ("scifact", "569"), "wild": ("nfcorpus", "PLAIN-977")}
MODEL, QUANT = "castorini/rank_zephyr_7b_v1_full", "4bit"
CE = "cross-encoder/ms-marco-MiniLM-L-6-v2"
WIDTH = 58
_LLM = None

# When both models are already in the local Hugging Face cache the demo must not depend on the room's network.
# (Our research copy of rankrepro sets HF_HOME to its model cache, the copy in reproduction/ leaves it to the default;
# huggingface_hub is not imported yet, so the switch still takes effect.)
_hub = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
if all((_hub / ("models--" + m.replace("/", "--"))).is_dir() for m in (MODEL, CE)):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")


def llm():
    """RankZephyr, created once. Lazy: the weights are loaded only when a window is not in the cache."""
    global _LLM
    if _LLM is None:
        from rankrepro.llm_backend import HFChatLLM
        _LLM = HFChatLLM(MODEL, quant=QUANT, device_map={"": 0})
    return _LLM


def ndcg10(qrels_q, ranking):
    import pytrec_eval
    run = {"q": {d: float(len(ranking) - i) for i, d in enumerate(ranking)}}
    return pytrec_eval.RelevanceEvaluator({"q": qrels_q}, {"ndcg_cut_10"}).evaluate(run)["q"]["ndcg_cut_10"]


def cell(docid, corpus, qrels_q):
    title = " ".join((corpus[docid]["title"] or corpus[docid]["text"]).split())
    mark = "*" if qrels_q.get(docid, 0) > 0 else " "
    return f"{mark} {title[:WIDTH - 3]}" + ("..." if len(title) > WIDTH - 3 else "")


def rankzephyr(query, hits, cache_path):
    model, t0 = llm(), time.time()          # the tokenizer load is not part of the ranking time
    new_hits, logs = sliding_window_rerank(model, query, hits, window=20, step=10, style="zephyr",
                                           cache=WindowCache(cache_path), model_tag=f"{MODEL}|{QUANT}|zephyr")
    top = [w for w in logs if w.get("start") == 0] or logs[-1:]
    return [h["docid"] for h in new_hits], top[0], time.time() - t0


def show(query, corpus, qrels_q, bm25, ce, rz, window, seconds):
    print()
    print("QUERY:", query)
    print()
    if qrels_q:
        print(f"nDCG@10   BM25 {100 * ndcg10(qrels_q, bm25):.1f}   cross-encoder {100 * ndcg10(qrels_q, ce):.1f}"
              f"   RankZephyr {100 * ndcg10(qrels_q, rz):.1f}      (* = relevant in the qrels)")
        print()
    print(f"{'':4}{'BM25':<{WIDTH}}  {'cross-encoder (ce)':<{WIDTH}}  RankZephyr-7B")
    print("-" * (3 * WIDTH + 8))
    for i in range(5):
        print(f"{i + 1:>2}  {cell(bm25[i], corpus, qrels_q):<{WIDTH}}  {cell(ce[i], corpus, qrels_q):<{WIDTH}}  "
              f"{cell(rz[i], corpus, qrels_q)}")
    print()
    rel = [i + 1 for i, d in enumerate(window.get("docids_in", [])) if qrels_q.get(d, 0) > 0]
    where = "cached" if window.get("cached") else f"generated in {window.get('seconds', 0):.1f} s"
    print(f"RankZephyr top window ({window.get('n')} passages, {window.get('prompt_tokens')} prompt tokens, {where};"
          f" relevant passages: {rel or 'n/a'}; total {seconds:.1f} s):")
    print("   ", window.get("response"))


class Live:
    """One corpus kept in memory for free-text queries. BM25 is built exactly as in testbed.bm25_search (stop words,
    Porter stemmer, k1=1.2, b=0.75), but once, so that a second query does not pay for the index again."""

    def __init__(self, dataset):
        import nltk
        from nltk.corpus import stopwords
        from nltk.stem import PorterStemmer
        from rank_bm25 import BM25Okapi
        from sentence_transformers import CrossEncoder
        from sklearn.feature_extraction.text import CountVectorizer
        t0 = time.time()
        self.corpus, _, _ = testbed.load(dataset)
        self.ids, self.texts = list(self.corpus), testbed.doc_texts(self.corpus)
        try:
            sw = set(stopwords.words("english"))
        except LookupError:                      # first use on this machine: needs the network once
            nltk.download("stopwords", quiet=True)
            sw = set(stopwords.words("english"))
        porter, analyze = PorterStemmer(), CountVectorizer().build_analyzer()
        self.prep = lambda t: [porter.stem(x) for x in analyze(t) if x not in sw]
        self.bm25 = BM25Okapi([self.prep(t) for t in self.texts], k1=1.2, b=0.75)
        self.ce = CrossEncoder(CE, device=testbed.device())
        print(f"{dataset}: BM25 index over {len(self.ids)} documents and ce ready in {time.time() - t0:.0f} s", flush=True)

    def rank(self, text):
        """BM25 top-100 and the ce order of those 100, as lists of document ids."""
        s = self.bm25.get_scores(self.prep(text))
        top = [int(j) for j in np.argsort(-s)[: testbed.K]]
        scores = self.ce.predict([[text, self.texts[j]] for j in top], batch_size=64, show_progress_bar=False)
        return [self.ids[j] for j in top], [self.ids[top[i]] for i in np.argsort(-np.asarray(scores))]

    def ask(self, text):
        bm25, ce = self.rank(text)
        texts = dict(zip(self.ids, self.texts))
        hits = [{"docid": d, "text": texts[d]} for d in bm25[:20]]
        rz, window, seconds = rankzephyr(" ".join(text.split()), hits, HERE / "cache" / "rankzephyr_live.jsonl")
        show(text, self.corpus, {}, bm25, ce, rz, window, seconds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("shortcut", nargs="?", choices=list(SHORTCUTS))
    ap.add_argument("--dataset", default="scifact", choices=testbed.DATASETS)
    ap.add_argument("--qid", help="a test query id of the dataset")
    ap.add_argument("--query", help="free text: live BM25 + ce + one RankZephyr window on the BM25 top-20")
    ap.add_argument("--interactive", action="store_true", help="live queries typed one after the other (empty line to stop)")
    args = ap.parse_args()
    if args.shortcut:
        args.dataset, args.qid = SHORTCUTS[args.shortcut]

    if args.query or args.interactive:
        live = Live(args.dataset)
        if args.query:
            live.ask(args.query)
        while args.interactive:
            text = input("query> ").strip()
            if not text:
                break
            live.ask(text)
        return

    corpus, queries, qrels = testbed.load(args.dataset)
    if args.qid not in queries:
        sys.exit(f"{args.qid!r} is not a test query of {args.dataset}; e.g. {list(queries)[:3]}")
    query, hits = method.candidates(args.dataset)[args.qid]
    bm25 = [h["docid"] for h in hits]
    ce_run = testbed.baseline_run(args.dataset, "ce")[args.qid]
    ce = sorted(ce_run, key=ce_run.get, reverse=True)
    rz, window, seconds = rankzephyr(query, hits[: method.DEPTH[args.dataset]],
                                     HERE / "cache" / f"rankzephyr_{args.dataset}.jsonl")
    show(query, corpus, qrels[args.qid], bm25, ce, rz + [d for d in bm25 if d not in set(rz)], window, seconds)


if __name__ == "__main__":
    main()
