"""What each system costs on a laptop CPU (run on an idle machine; torch's default thread count).
Per system: query latency at batch size 1 (median and p95 over 100 SciFact queries; encoding, then a flat
dot-product search over the 5,183 documents), corpus encoding throughput, parameters, dims, index bytes/doc.
Usage: python pairs/6-static-embeddings/latency.py  ->  results_variants/latency.json"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))
import testbed
from variants import VARIANTS

N_QUERIES, N_DOCS_TRANSFORMER = 100, 256
SYSTEMS = {  # name -> (label, sentence-transformers id or None, query prefix, prompt_name)
    "bm25": ("BM25", None, "", None),
    "minilm": ("all-MiniLM-L6-v2", "sentence-transformers/all-MiniLM-L6-v2", "", None),
    "bge": ("bge-small-en-v1.5", "BAAI/bge-small-en-v1.5", testbed.BGE_QUERY, None),
    "official": ("static-retrieval-mrl-en-v1", "sentence-transformers/static-retrieval-mrl-en-v1", "", None),
}
for name in ["glove", "mpnet", "contriever", "e5", "qwen3", "m2v_base", "potion8m", "potion_ret32m"]:
    label, _, _, kw = VARIANTS[name]
    SYSTEMS[name] = (label, kw["st"], kw.get("qp", ""), kw.get("prompt_name"))

corpus, queries, _ = testbed.load("scifact")
docs = testbed.doc_texts(corpus)
qtexts = list(queries.values())[:N_QUERIES]
out = {"torch_threads": torch.get_num_threads(), "n_docs": len(docs), "systems": {}}


def timed(fn, reps):
    times = []
    for x in reps:
        t0 = time.perf_counter()
        fn(x)
        times.append(time.perf_counter() - t0)
    return 1e3 * np.median(times), 1e3 * np.percentile(times, 95)


OUT_PATH = HERE / "results_variants" / "latency.json"
(HERE / "results_variants").mkdir(exist_ok=True)
for name, (label, st, qp, prompt_name) in SYSTEMS.items():
    if name in out["systems"]:
        continue
    print(f"== {label}", flush=True)
    if st is None:  # BM25 with the testbed's own preprocessing and scorer
        import nltk
        from nltk.corpus import stopwords
        from nltk.stem import PorterStemmer
        from rank_bm25 import BM25Okapi
        from sklearn.feature_extraction.text import CountVectorizer
        nltk.download("stopwords", quiet=True)
        sw, porter, analyze = set(stopwords.words("english")), PorterStemmer(), CountVectorizer().build_analyzer()
        pre = lambda t: [porter.stem(x) for x in analyze(t) if x not in sw]
        t0 = time.perf_counter(); bm25 = BM25Okapi([pre(t) for t in docs], k1=1.2, b=0.75); index_s = time.perf_counter() - t0
        med, p95 = timed(lambda q: np.argsort(-bm25.get_scores(pre(q)))[:100], qtexts)
        out["systems"][name] = {"label": label, "query_ms_median": med, "query_ms_p95": p95, "index_s": index_s,
                                "params": 0, "dims": None, "index_bytes_per_doc": None}
        json.dump(out, open(OUT_PATH, "w"), indent=1)
        continue
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(st, device="cpu")
    params = sum(p.numel() for p in m.parameters())
    static = name == "official" or VARIANTS.get(name, ("", 0, ""))[2].startswith("static")
    sample = docs if static else docs[:N_DOCS_TRANSFORMER]
    m.encode(sample[:32], batch_size=32)  # warm-up
    t0 = time.perf_counter(); D = m.encode(sample, batch_size=1024 if static else 32, normalize_embeddings=True); enc_s = time.perf_counter() - t0
    # Same-size matrix for the search timing. The search runs in torch, in the same thread pool as the encoder:
    # mixing torch (OpenMP) with a NumPy/OpenBLAS matmul costs ~100 ms of thread hand-off per query on this machine.
    D_full = torch.from_numpy(np.repeat(D, int(np.ceil(len(docs) / len(D))), axis=0)[: len(docs)])
    m.encode([qp + qtexts[0]], prompt_name=prompt_name)  # warm-up
    search = lambda q: torch.topk(D_full @ m.encode([qp + q], prompt_name=prompt_name, normalize_embeddings=True, convert_to_tensor=True)[0].float(), 100)
    med, p95 = timed(search, qtexts)
    enc_only, _ = timed(lambda q: m.encode([qp + q], prompt_name=prompt_name), qtexts[:30])
    out["systems"][name] = {"label": label, "query_ms_median": med, "query_ms_p95": p95, "query_encode_ms_median": enc_only,
                            "corpus_docs_per_s": len(sample) / enc_s, "corpus_encode_s_estimate": len(docs) / (len(sample) / enc_s),
                            "params": params, "dims": int(D.shape[1]), "index_bytes_per_doc": int(D.shape[1]) * 4}
    print(f"   query {med:.2f} ms (encode {enc_only:.2f} ms)   corpus {len(docs) / (len(sample) / enc_s):.1f} s   {params / 1e6:.1f}M params", flush=True)
    del m
    json.dump(out, open(OUT_PATH, "w"), indent=1)   # saved after every system
