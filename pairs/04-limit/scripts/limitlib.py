"""Shared code for pair 04 (LIMIT): the LIMIT data, the three encoders, exact MaxSim and Recall@k.
Every script of the pair folder imports it; it imports the class testbed for the BEIR data and BM25."""

import json
import os
import sys
import urllib.request
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1] # the pair folder (this file is in its scripts/)
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
import testbed

OUT = HERE / "outputs" # small JSON results: committed
CACHE = HERE / "cache" # embeddings and runs: local only (see .gitignore)
SEED = 42              # the seed of the paper's code (free-embedding script and dataset generator)

# The LIMIT data at a fixed commit of github.com/google-deepmind/limit.
LIMIT_COMMIT = "90c5dd3a7c3c0d68bc6cc7a0c5b0f4ca124dfdcb"
LIMIT_URL = "https://raw.githubusercontent.com/google-deepmind/limit/" + LIMIT_COMMIT + "/data/{}/{}.jsonl"
LIMIT_SPLITS = ["limit-small", "limit"]

BGE = "BAAI/bge-small-en-v1.5"
QWEN = "Qwen/Qwen3-Embedding-0.6B"
QWEN8 = "Qwen/Qwen3-Embedding-8B" # the model of the paper's Tables 4-5
COLBERT = "lightonai/GTE-ModernColBERT-v1"

# Qwen3-Embedding is Matryoshka-trained down to 32 dimensions.
QWEN_DIMS = {QWEN: [32, 64, 128, 256, 512, 1024], QWEN8: [32, 64, 128, 256, 512, 1024, 2048, 4096]}

# Qwen3-Embedding wants a one-line task instruction on the query side (its model card); none on documents.
QWEN_TASKS = {
    "limit-small": "Given a web search query, retrieve relevant passages that answer the query",
    "limit": "Given a web search query, retrieve relevant passages that answer the query",
    "scifact": "Given a scientific claim, retrieve documents that support or refute the claim",
    "nfcorpus": "Given a question, retrieve relevant documents that best answer the question",
    "arguana": "Given a claim, find documents that refute the claim"
}

# ColBERT query length in tokens: the model default (48) except ArguAna, whose queries are whole arguments.
COLBERT_QUERY_LENGTH = {"arguana": 256}
COLBERT_DOC_LENGTH = 512 # covers the testbed's 200-word documents and every LIMIT document

# Query-side settings for the ColBERT diagnosis. "paper-era" is the model's configuration when the paper was written
# (revision 78d50a1, Aug 2025): query length 32 and no do_query_expansion key, so PyLate pads every query with [MASK]
# tokens to 32; the model's authors switched expansion off on 2025-09-09 (revision 506af20).
COLBERT_VARIANTS = {"default": {}, "paper-era": {"query_length": 32, "do_query_expansion": True}, "qlen32": {"query_length": 32, "do_query_expansion": False}}


def device():
    return testbed.device()


def load_limit(split):
    """Downloads a LIMIT split once into the testbed's data/ folder. Same shapes as testbed.load:
    corpus {id: {"title", "text"}}, queries {id: text}, qrels {query id: {doc id: 1}}."""
    folder = testbed.DATA / split
    folder.mkdir(parents=True, exist_ok=True)
    for part in ["corpus", "queries", "qrels"]:
        path = folder / f"{part}.jsonl"
        if not path.exists():
            print(f"downloading {split}/{part}", file=sys.stderr)
            urllib.request.urlretrieve(LIMIT_URL.format(split, part), path)
    corpus, queries, qrels = {}, {}, {}
    with open(folder / "corpus.jsonl", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            corpus[d["_id"]] = {"title": d.get("title", ""), "text": d["text"]}
    with open(folder / "queries.jsonl", encoding="utf-8") as f:
        for line in f:
            q = json.loads(line)
            queries[q["_id"]] = q["text"]
    with open(folder / "qrels.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["score"] > 0:
                qrels.setdefault(r["query-id"], {})[r["corpus-id"]] = int(r["score"])
    return corpus, {q: queries[q] for q in qrels}, qrels


def load_any(name):
    return load_limit(name) if name in LIMIT_SPLITS else testbed.load(name)


def texts_of(name, corpus):
    """Document texts: the testbed's 200-word spring text; LIMIT documents are shorter and kept whole."""
    if name in LIMIT_SPLITS:
        return [d["text"] for d in corpus.values()]
    return testbed.doc_texts(corpus)


def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path) # never leaves a half-written file behind when a job is killed


# ========== single-vector encoders ==========

def load_dense(model_name):
    import torch
    from sentence_transformers import SentenceTransformer
    kwargs = {}
    if model_name == QWEN8:
        kwargs = {"torch_dtype": torch.bfloat16} # 8B: the dtype it was trained in, half the memory of float32
    elif model_name == QWEN and device() == "cuda":
        kwargs = {"torch_dtype": torch.float16}
    return SentenceTransformer(model_name, device=device(), model_kwargs=kwargs)


def query_prefix(model_name, name):
    if model_name == BGE:
        return testbed.BGE_QUERY
    if model_name in (QWEN, QWEN8):
        return f"Instruct: {QWEN_TASKS[name]}\nQuery:"
    return ""


def dense_embeddings(model_name, name, corpus, queries, batch_size=32):
    """Raw (not normalised) document and query embeddings, cached as .npy: truncation needs the raw vectors."""
    tag = model_name.split("/")[-1]
    dpath, qpath = CACHE / name / f"{tag}.docs.npy", CACHE / name / f"{tag}.queries.npy"
    if dpath.exists() and qpath.exists():
        return np.load(dpath), np.load(qpath)
    model = load_dense(model_name)
    D = model.encode(texts_of(name, corpus), batch_size=batch_size, convert_to_numpy=True, show_progress_bar=False)
    prefix = query_prefix(model_name, name)
    Q = model.encode([prefix + t for t in queries.values()], batch_size=batch_size, convert_to_numpy=True,
                     show_progress_bar=False)
    dpath.parent.mkdir(parents=True, exist_ok=True)
    np.save(dpath, D.astype(np.float32))
    np.save(qpath, Q.astype(np.float32))
    return np.load(dpath), np.load(qpath)


def cosine_scores(Q, D, dim=None):
    """Cosine similarity of the first `dim` coordinates (Matryoshka truncation), computed in chunks of queries."""
    Q, D = Q[:, :dim], D[:, :dim]
    Q = Q / np.linalg.norm(Q, axis=1, keepdims=True)
    D = D / np.linalg.norm(D, axis=1, keepdims=True)
    return (Q @ D.T).astype(np.float32)



# ========== multi-vector (ColBERT) encoder ==========

def load_colbert(variant="default", name=None):
    from pylate import models
    model = models.ColBERT(model_name_or_path=COLBERT, device=device(), document_length=COLBERT_DOC_LENGTH,
                           **COLBERT_VARIANTS[variant])
    if variant == "default":
        model.query_length = COLBERT_QUERY_LENGTH.get(name, model.query_length)
    return model


def colbert_embeddings(name, corpus, queries, variant="default", batch_size=32):
    """Token embeddings of every document and query, cached with torch.save (lists of fp16 tensors). The variants
    change only the query side, so they share the document embeddings."""
    import torch
    dpath = CACHE / name / "colbert.docs.pt"
    qpath = CACHE / name / ("colbert.queries.pt" if variant == "default" else f"colbert-{variant}.queries.pt")
    if dpath.exists() and qpath.exists():
        return torch.load(dpath), torch.load(qpath)
    model = load_colbert(variant, name)
    encode = lambda texts, is_query: [torch.from_numpy(np.asarray(x)).half() for x in model.encode(
        texts, is_query=is_query, batch_size=batch_size, convert_to_numpy=True, show_progress_bar=False)]
    dpath.parent.mkdir(parents=True, exist_ok=True)
    if not dpath.exists():
        torch.save(encode(texts_of(name, corpus), False), dpath)
    torch.save(encode(list(queries.values()), True), qpath)
    return torch.load(dpath), torch.load(qpath)


def _pad(tensors, order, with_first_token):
    """Stacks variable-length token matrices. Documents are padded with copies of their own first token, which
    cannot change a max over their tokens; queries with zero vectors, which add 0 to the sum over query tokens."""
    import torch
    length = max(tensors[i].shape[0] for i in order)
    out = torch.zeros(len(order), length, tensors[order[0]].shape[1], dtype=tensors[order[0]].dtype)
    for row, i in enumerate(order):
        t = tensors[i]
        out[row, :t.shape[0]] = t
        if with_first_token:
            out[row, t.shape[0]:] = t[0]
    return out


def maxsim_scores(Q, D, budget=2 ** 28):
    """Exact late-interaction scores: for every query and document, the sum over query tokens of the best
    dot product with any document token. Q, D: lists of [tokens, dim] tensors. Returns [n queries, n docs]."""
    import torch
    dev = device()
    dtype = torch.float16 if dev == "cuda" else torch.float32
    order = sorted(range(len(D)), key=lambda i: D[i].shape[0]) # similar lengths together: little padding
    Qp = _pad(Q, list(range(len(Q))), with_first_token=False).to(dev, dtype)
    nq, lq, _ = Qp.shape
    scores = torch.empty(nq, len(D), dtype=torch.float32)
    qb = max(1, min(nq, budget // (lq * 512 * 64))) # query batch so one doc chunk stays small
    start = 0
    while start < len(order):
        chunk = max(1, budget // (qb * lq * D[order[start]].shape[0]))
        longest = D[order[min(start + chunk, len(order)) - 1]].shape[0]
        chunk = max(1, budget // (qb * lq * longest)) # sized by the longest document it holds
        idx = order[start:start + chunk]
        Dc = _pad(D, idx, with_first_token=True).to(dev, dtype) # [c, ld, h]
        c, ld, h = Dc.shape
        flat = Dc.reshape(c * ld, h).T
        for b in range(0, nq, qb):
            q = Qp[b:b + qb]
            s = (q.reshape(-1, h) @ flat).reshape(q.shape[0], lq, c, ld).amax(dim=3).sum(dim=1)
            scores[b:b + qb, idx] = s.float().cpu()
        start += chunk
    return scores.numpy()



# ========== runs and metrics ==========

def to_run(scores, query_ids, doc_ids, k=testbed.K + 1):
    """Top-k documents per query as a testbed run {query id: {doc id: score}}."""
    import torch
    S = torch.from_numpy(np.ascontiguousarray(scores))
    k = min(k, S.shape[1])
    vals, idx = S.topk(k, dim=1)
    return {q: {doc_ids[j]: float(v) for j, v in zip(idx[i].tolist(), vals[i].tolist())}
            for i, q in enumerate(query_ids)}


def recall_at(run, qrels, ks=(2, 10, 20, 100)):
    """Recall@k as in MTEB and the LIMIT paper: share of a query's relevant documents in its top k.
    Returns {k: per-query array} in the order of qrels."""
    out = {k: [] for k in ks}
    for q, rel in qrels.items():
        ranked = sorted(run[q], key=run[q].get, reverse=True)
        for k in ks:
            out[k].append(len(set(ranked[:k]) & set(rel)) / len(rel))
    return {k: np.array(v) for k, v in out.items()}
