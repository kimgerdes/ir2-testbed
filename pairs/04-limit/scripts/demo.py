"""Live demo: one query, the top 5 of BM25, bge-small and GTE-ModernColBERT side by side, on precomputed document
embeddings (made by limit_eval.py and method.py); only the query is encoded live. ✓ marks a relevant document.

  python pairs/04-limit/scripts/demo.py "Who likes Quokkas?"                      LIMIT-small (default)
  python pairs/04-limit/scripts/demo.py --dataset scifact "Aspirin reduces ..."   a testbed dataset
  python pairs/04-limit/scripts/demo.py --dataset scifact                         the live demo: load once, then type queries
  python pairs/04-limit/scripts/demo.py --suggest                                 hero and failure candidates from outputs/
On LIMIT, any "Who likes X?" query is judged by reading the documents, so a wildcard from the room is judged too."""
import argparse
import difflib
import json
import os
import re
import warnings

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error") # no load reports or deprecation notes on the demo screen
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
warnings.filterwarnings("ignore")

import numpy as np

import limitlib
from limitlib import testbed

SHOW = 5


def wildcard(name, text):
    """X of a "Who likes X?" query on LIMIT, else None."""
    m = re.fullmatch(r"\s*who likes (.+?)\??\s*", text, flags=re.I)
    return m.group(1).strip() if name in limitlib.LIMIT_SPLITS and m else None


def judge(name, corpus, queries, qrels, text):
    """Relevant documents of a query: the dataset's labels when the query is one of its own, else for LIMIT the
    people whose list of likes contains X (a wildcard is judged too), else unknown (None)."""
    for q, t in queries.items():
        if t.strip().lower() == text.strip().lower():
            return set(qrels[q])
    x = wildcard(name, text)
    if x is not None:
        return {d for d, doc in corpus.items() if norm(x) in {norm(i) for i in likes(doc["text"])}}
    return None


def norm(item):
    """Lower case, last word without its plural s: 'Quokkas' and 'quokka' are the same item."""
    words = item.lower().split()
    if words and len(words[-1]) > 3 and words[-1].endswith("s") and not words[-1].endswith("ss"):
        words[-1] = words[-1][:-1]
    return " ".join(words)


def likes(text):
    """'Ann Lee likes A, B, C and D.' -> {'a', 'b', 'c', 'd'}."""
    items = [a.strip().lower() for a in text.split(" likes ", 1)[-1].rstrip(".").split(",")]
    last = items.pop()
    head, sep, tail = last.rpartition(" and ")
    return set(items) | ({head.strip(), tail.strip()} if sep else {last})


def close_items(corpus, x, n=5):
    """Items of the corpus close to X, for a wildcard that nobody likes: those sharing a word, then spelling."""
    items = sorted(set().union(*(likes(d["text"]) for d in corpus.values())))
    words = set(norm(x).split())
    sharing = [i for i in items if words & set(norm(w) for w in i.split())]
    return list(dict.fromkeys(sharing + difflib.get_close_matches(x.lower(), items, n=n, cutoff=0.6)))[:n]


def ranking(scores, ids):
    return [ids[j] for j in np.argsort(-scores, kind="stable")]


def searcher(name):
    """Loads the index and both models once; returns a function that answers one query in about a second."""
    import torch
    import transformers
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer
    from rank_bm25 import BM25Okapi
    from sklearn.feature_extraction.text import CountVectorizer
    transformers.utils.logging.disable_progress_bar()
    corpus, queries, qrels = limitlib.load_any(name)
    ids = list(corpus)
    sw, porter, analyze = set(stopwords.words("english")), PorterStemmer(), CountVectorizer().build_analyzer()
    prep = lambda t: [porter.stem(x) for x in analyze(t) if x not in sw]     # the testbed's BM25, as in the spring
    bm25 = BM25Okapi([prep(t) for t in limitlib.texts_of(name, corpus)], k1=1.2, b=0.75)
    D, _ = limitlib.dense_embeddings(limitlib.BGE, name, corpus, queries)
    bge = limitlib.load_dense(limitlib.BGE)
    Dc, _ = limitlib.colbert_embeddings(name, corpus, queries)
    colbert = limitlib.load_colbert("default", name)

    def search(text):
        q = bge.encode([limitlib.query_prefix(limitlib.BGE, name) + text])
        qc = [torch.from_numpy(np.asarray(colbert.encode([text], is_query=True, show_progress_bar=False)[0])).half()]
        columns = {"BM25": ranking(bm25.get_scores(prep(text)), ids),
                   "bge-small (1 vector)": ranking(limitlib.cosine_scores(q, D)[0], ids),
                   "GTE-ModernColBERT (MaxSim)": ranking(limitlib.maxsim_scores(qc, Dc)[0], ids)}
        rel = judge(name, corpus, queries, qrels, text)
        x = wildcard(name, text)
        if rel == set() and x is not None:
            print(f"\nNobody in {name} likes '{x}'. Close items: {', '.join(i.title() for i in close_items(corpus, x))}")
            return
        show(name, corpus, text, columns, rel)
    return search


def show(name, corpus, text, columns, rel):
    """Top 5 of each system side by side (✓: relevant), then where each system ranks the relevant documents."""
    print(f"\nQuery: {text}    ({name}; relevant: {'unknown' if rel is None else len(rel)})\n")
    width = 34
    print("     " + "".join(f"{c:<{width}}" for c in columns))
    for r in range(SHOW):
        cells = []
        for col in columns.values():
            d = col[r]
            mark = "" if rel is None else ("✓ " if d in rel else "  ")
            label = d if name in limitlib.LIMIT_SPLITS else (corpus[d]["title"] or corpus[d]["text"])
            cells.append(f"{mark}{label}"[:width - 2].ljust(width))
        print(f"{r + 1:>3}  " + "".join(cells))
    if rel:
        print("\nfound in top 2:     " + "   ".join(f"{c} {len(set(col[:2]) & rel)}/{min(2, len(rel))}"
                                               for c, col in columns.items()))
        print("relevant ranked at: " + "   ".join(f"{c} {', '.join(str(col.index(d) + 1) for d in sorted(rel, key=col.index))}"
                                               for c, col in columns.items()))


def suggest():
    """Hero: ColBERT finds both people, bge finds neither. Failure: the queries where ColBERT misses someone."""
    folder = limitlib.OUT / "limit" / "limit-small"
    res = {s: json.load(open(folder / f"{s}.json", encoding="utf-8")) for s in ["bm25", "bge", "colbert"]}
    _, queries, _ = limitlib.load_limit("limit-small")
    qids = res["bge"]["query_ids"]
    r2 = {s: np.array(res[s]["systems"][s]["per_query_recall@2"]) for s in res}
    hero = [queries[qids[i]] for i in np.where((r2["colbert"] == 1) & (r2["bge"] == 0))[0][:5]]
    fail = [queries[qids[i]] for i in np.where(r2["colbert"] < 1)[0]]
    print("Hero candidates (ColBERT 2/2, bge 0/2):", *hero, sep="\n  ")
    print("Failure candidates (ColBERT misses someone in the top 2):", *fail, sep="\n  ")
    print("Testbed failures: python pairs/04-limit/scripts/analysis.py writes outputs/error_analysis_colbert.md "
          "(largest losses against bge-small).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("query", nargs="?")
    parser.add_argument("--dataset", default="limit-small", choices=limitlib.LIMIT_SPLITS + testbed.DATASETS)
    parser.add_argument("--suggest", action="store_true")
    args = parser.parse_args()
    if args.suggest:
        suggest()
    elif args.query:
        searcher(args.dataset)(args.query)
    else:                                   # the live demo: load once, then one query per line
        search = searcher(args.dataset)
        try:
            while (text := input("\nquery (empty line to quit)> ").strip()):
                search(text)
        except (EOFError, KeyboardInterrupt): # Ctrl-Z / Ctrl-C quit quietly
            print()
