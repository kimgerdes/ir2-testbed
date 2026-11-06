"""Beyond the official row: one model per era, the static family, Matryoshka truncation and a hybrid,
all on the testbed, all scored by testbed.py against `minilm` (the spring bi-encoder).
Usage: python pairs/6-static-embeddings/variants.py [names...]      (default: all; finished ones are skipped)
Writes runs_variants/<name>/<dataset>.json (local) and results_variants/<name>.json (committed).
Every model gets its own documented prompt and similarity, so no baseline is handicapped."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))
import testbed

OUR_TRAINED = HERE / "models" / "static-mrl"          # from: python pairs/6-static-embeddings/train_static.py

# name -> (label, year, family, loader kwargs)
VARIANTS = {
    # one model per era (bm25, minilm and bge are testbed baselines: not re-run)
    "glove":        ("GloVe 6B-300d average", 2014, "static (word vectors)", dict(st="sentence-transformers/average_word_embeddings_glove.6B.300d", bs=128)),
    "mpnet":        ("all-mpnet-base-v2", 2021, "transformer", dict(st="sentence-transformers/all-mpnet-base-v2")),
    "contriever":   ("Contriever (MS MARCO)", 2021, "transformer", dict(st="facebook/contriever-msmarco", normalize=False)),  # trained with dot product
    "e5":           ("e5-small-v2", 2023, "transformer", dict(st="intfloat/e5-small-v2", qp="query: ", dp="passage: ")),
    "qwen3":        ("Qwen3-Embedding-0.6B", 2025, "LLM-based", dict(st="Qwen/Qwen3-Embedding-0.6B", prompt_name="query", bs=16, fp16=True)),
    # the static family
    "m2v_base":     ("Model2Vec M2V_base_output", 2024, "static (distilled)", dict(st="minishlab/M2V_base_output", bs=1024)),
    "potion8m":     ("Model2Vec potion-base-8M", 2024, "static (distilled + Tokenlearn)", dict(st="minishlab/potion-base-8M", bs=1024)),
    "potion_ret32m": ("Model2Vec potion-retrieval-32M", 2025, "static (distilled + Tokenlearn)", dict(st="minishlab/potion-retrieval-32M", bs=1024)),
    "ours_m2v_bge": ("ours: Model2Vec distil of bge-small", 2026, "static (this pair)", dict(distill="BAAI/bge-small-en-v1.5")),
    "ours_trained": ("ours: trained static (MS MARCO, 10 s)", 2026, "static (this pair)", dict(ours=str(OUR_TRAINED))),
    # Matryoshka: the official model truncated (384 dims = MiniLM's index size)
    **{f"static_d{d}": (f"static-retrieval-mrl-en-v1 @ {d} dims", 2025, "static (contrastive)",
                        dict(st="sentence-transformers/static-retrieval-mrl-en-v1", bs=1024, truncate=d)) for d in (512, 384, 256, 128, 64)},
    # the spring hybrid with a static dense half: RRF(BM25 + static) instead of RRF(BM25 + MiniLM)
    "rrf_bm25_static": ("RRF (BM25 + static)", 2025, "hybrid", dict(rrf=True)),
}


def search(encode_q, encode_d, corpus, queries, normalize=True):
    """testbed.dense_search, generalised to any encoder and to dot product (Contriever)."""
    ids, qids = list(corpus), list(queries)
    D = np.asarray(encode_d(testbed.doc_texts(corpus)), dtype=np.float32)
    Q = np.asarray(encode_q([queries[q] for q in qids]), dtype=np.float32)
    if normalize:
        D /= np.linalg.norm(D, axis=1, keepdims=True) + 1e-12
        Q /= np.linalg.norm(Q, axis=1, keepdims=True) + 1e-12
    S = Q @ D.T
    top = np.argsort(-S, axis=1)[:, : testbed.K + 1]
    return {q: {ids[j]: float(S[i, j]) for j in top[i]} for i, q in enumerate(qids)}


def encoders(kw):
    """-> (encode_queries, encode_docs) for one variant."""
    if "st" in kw:
        from sentence_transformers import SentenceTransformer
        m = SentenceTransformer(kw["st"], device=testbed.device(), truncate_dim=kw.get("truncate"),
                                model_kwargs={"torch_dtype": torch.float16} if kw.get("fp16") else None)
        bs = kw.get("bs", 64)
        qp, dp = kw.get("qp", ""), kw.get("dp", "")
        pn = kw.get("prompt_name")
        eq = lambda t: m.encode([qp + x for x in t], batch_size=bs, prompt_name=pn, show_progress_bar=False)
        ed = lambda t: m.encode([dp + x for x in t], batch_size=bs, show_progress_bar=False)
        return eq, ed
    from static_emb.static_model import StaticEmbedder
    if "distill" in kw:
        from static_emb.distill import distill
        m = distill(kw["distill"], pca_dims=256, device=testbed.device())
    else:
        if not Path(kw["ours"]).exists():
            sys.exit(f"{kw['ours']} missing: run train_static.py first")
        m = StaticEmbedder.load(kw["ours"])
    enc = lambda t: m.encode(t, normalize=True)
    return enc, enc


def run_variant(name):
    label, _, _, kw = VARIANTS[name]
    out_runs = HERE / "runs_variants" / name
    if kw.get("rrf"):
        for ds in testbed.DATASETS:
            static_run = json.load(open(HERE / "runs" / f"{ds}.json"))  # the official row's run
            testbed.save_run(testbed.rrf([testbed.baseline_run(ds, "bm25"), static_run]), out_runs / f"{ds}.json")
    else:
        eq, ed = encoders(kw)
        for ds in testbed.DATASETS:
            corpus, queries, _ = testbed.load(ds)
            run = search(eq, ed, corpus, queries, normalize=kw.get("normalize", True))
            testbed.save_run(run, out_runs / f"{ds}.json")
        del eq, ed
        torch.cuda.empty_cache()
    (HERE / "results_variants").mkdir(exist_ok=True)
    against = "rrf" if kw.get("rrf") else "minilm"
    testbed.cmd_score(argparse.Namespace(run_dir=str(out_runs), method=label, pair="6-static-embeddings",
                                         against=against, out=str(HERE / "results_variants" / f"{name}.json")))


if __name__ == "__main__":
    names = sys.argv[1:] or list(VARIANTS)
    for name in names:
        if (HERE / "results_variants" / f"{name}.json").exists():
            print(f"{name}: done, skipped")
            continue
        print(f"== {name}", flush=True)
        run_variant(name)
