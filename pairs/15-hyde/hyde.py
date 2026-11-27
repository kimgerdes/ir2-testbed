"""HyDE helpers shared by reproduce.ipynb, method.py and demo.py."""
import gzip
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent


def load_generations(name, folder="generations"):
    """{query id: [hypothetical documents]}, the cache written by generate.py."""
    with gzip.open(HERE / folder / f"{name}.jsonl.gz", "rt") as f:
        return {r["qid"]: r["docs"] for r in map(json.loads, f)}


def embed_generations(model, queries, generations, normalize=False):
    """Embeddings of every query's hypothetical documents: an array (queries, documents per query, dimensions)."""
    docs = [d for q in queries for d in generations[q]]
    E = model.encode(docs, batch_size=32, normalize_embeddings=normalize, convert_to_numpy=True)
    return E.reshape(len(queries), -1, E.shape[1])


def hyde_vectors(E, Q=None, n=8):
    """One vector per query: the mean of its first n hypothetical documents and, as in the paper, of the query Q."""
    if Q is None:
        return E[:, :n].mean(axis=1)
    return (E[:, :n].sum(axis=1) + Q) / (n + 1)


def search(Q, D, qids, ids, k=101):
    """Dot-product ranking as a testbed run: {query id: {document id: score}}."""
    S = Q @ D.T
    top = np.argsort(-S, axis=1)[:, :k]
    return {q: {ids[j]: float(S[i, j]) for j in top[i]} for i, q in enumerate(qids)}
