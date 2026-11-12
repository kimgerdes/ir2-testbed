"""Run once before the talk (online): builds every index the demo uses, so nothing is trained or indexed live.
BM25 (the testbed's preprocessing), MiniLM and static document embeddings for the testbed corpora.
Usage: python pairs/6-static-embeddings/demo/prepare_demo.py   ->  demo/cache/"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
import testbed

CACHE = HERE / "cache"
CACHE.mkdir(exist_ok=True)
MODELS = {"minilm": "sentence-transformers/all-MiniLM-L6-v2", "static": "sentence-transformers/static-retrieval-mrl-en-v1"}

from sentence_transformers import SentenceTransformer
encoders = {k: SentenceTransformer(v, device=testbed.device()) for k, v in MODELS.items()}
for name in ["scifact", "nfcorpus"]:
    corpus, queries, qrels = testbed.load(name)
    texts = testbed.doc_texts(corpus)
    for k, m in encoders.items():  # exactly the testbed's dense_search document side
        D = m.encode(texts, batch_size=1024 if k == "static" else 64, normalize_embeddings=True, convert_to_numpy=True)
        np.save(CACHE / f"{name}_{k}.npy", D.astype(np.float32))
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer
    from rank_bm25 import BM25Okapi
    from sklearn.feature_extraction.text import CountVectorizer
    nltk.download("stopwords", quiet=True)
    sw, porter, analyze = set(stopwords.words("english")), PorterStemmer(), CountVectorizer().build_analyzer()
    bm25 = BM25Okapi([[porter.stem(x) for x in analyze(t) if x not in sw] for t in texts], k1=1.2, b=0.75)
    pickle.dump(bm25, open(CACHE / f"{name}_bm25.pkl", "wb"))
    json.dump({"ids": list(corpus), "titles": [d["title"] for d in corpus.values()], "texts": texts,
               "queries": queries, "qrels": qrels}, open(CACHE / f"{name}_meta.json", "w"))
    print(f"{name}: {len(texts)} documents indexed for BM25, MiniLM and static")
for k, m in encoders.items():   # also warm the CPU copies in the HF cache
    SentenceTransformer(MODELS[k], device="cpu")
(CACHE / "READY").write_text("ok\n")
