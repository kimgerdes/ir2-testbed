"""The 10-minute demo: one query, three systems side by side on precomputed indexes, CPU only.
BM25 | all-MiniLM-L6-v2 (the spring bi-encoder) | static-retrieval-mrl-en-v1. Nothing is trained or indexed live.
Usage: python pairs/6-static-embeddings/demo/prepare_demo.py   (once, online)
       python pairs/6-static-embeddings/demo/demo_app.py       (offline; opens http://127.0.0.1:7860)"""
import json
import os
import pickle
import re
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
import gradio as gr
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

HERE = Path(__file__).resolve().parent
CACHE = HERE / "cache"
DEMO = json.load(open(HERE / "demo_queries.json"))
torch.set_grad_enabled(False)

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer

SW, PORTER, ANALYZE = set(stopwords.words("english")), PorterStemmer(), CountVectorizer().build_analyzer()
pre = lambda t: [PORTER.stem(x) for x in ANALYZE(t) if x not in SW]

models = {"minilm": SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu"),
          "static": SentenceTransformer("sentence-transformers/static-retrieval-mrl-en-v1", device="cpu")}
DATA = {}
for name in ["scifact", "nfcorpus"]:
    meta = json.load(open(CACHE / f"{name}_meta.json"))
    DATA[name] = {"meta": meta, "bm25": pickle.load(open(CACHE / f"{name}_bm25.pkl", "rb")),
                  "minilm": torch.from_numpy(np.load(CACHE / f"{name}_minilm.npy")), "static": torch.from_numpy(np.load(CACHE / f"{name}_static.npy")),
                  "qid_by_text": {t.strip().lower(): q for q, t in meta["queries"].items()}}
for m in models.values():  # warm-up, so the first query is not slower
    m.encode(["warm up"])

SYSTEMS = [("bm25", "BM25"), ("minilm", "all-MiniLM-L6-v2"),
           ("static", "static-retrieval-mrl-en-v1 (this paper)")]


def retrieve(system, query, data):
    t0 = time.perf_counter()
    if system == "bm25":
        scores = data["bm25"].get_scores(pre(query))
    else:
        # torch end to end: a NumPy matmul after a torch encode costs ~100 ms of thread-pool hand-off on CPU
        q = models[system].encode([query], normalize_embeddings=True, convert_to_tensor=True)[0]
        top = torch.topk(data[system] @ q, 100).indices.numpy()
        return top, 1e3 * (time.perf_counter() - t0)
    top = np.argsort(-scores)[:100]
    return top, 1e3 * (time.perf_counter() - t0)


def highlight(text, query):
    stems = set(pre(query))
    return " ".join(f"**{w}**" if (s := pre(w)) and s[0] in stems else w for w in text.split())


def search(query, dataset):
    data = DATA[dataset]
    meta = data["meta"]
    qid = data["qid_by_text"].get(query.strip().lower())
    rel = meta["qrels"].get(qid, {}) if qid else {}
    outs = []
    for system, label in SYSTEMS:
        top, ms = retrieve(system, query, data)
        first = next((r for r, j in enumerate(top, 1) if rel.get(meta["ids"][j], 0) > 0), None)
        lines = [f"### {label}", f"**{ms:.1f} ms** on CPU" + (f" · first relevant at rank **{first or '> 100'}**" if rel else ""), ""]
        for r, j in enumerate(top[:4], 1):
            mark = "<b style='color:#2e7d32'>✓ relevant</b>" if rel.get(meta["ids"][j], 0) > 0 else ""
            snippet = meta["texts"][j][:220] + "…"
            lines.append(f"{r}. {mark} {highlight(snippet, query)}")
        outs.append("\n\n".join(lines))
    info = f"Test query `{qid}` · {len(rel)} relevant document(s) in the qrels" if rel else "Free query: no relevance labels, judge for yourself"
    return [info] + outs


def selected(query):
    """Highlights the preset whose query is in the box (none for a free query)."""
    return [gr.update(variant="primary" if query.strip() == v["query"] else "secondary") for v in DEMO.values()]


with gr.Blocks(title="Static embeddings: live demo") as app:
    gr.Markdown("## Static embeddings vs the spring pipeline: same query, precomputed indexes, CPU only")
    with gr.Row():
        query = gr.Textbox(label="Query (type your own and press Enter, or click a demo query below)", value=DEMO["hero"]["query"], scale=5)
        dataset = gr.Radio(["scifact", "nfcorpus"], value=DEMO["hero"].get("dataset", "scifact"), label="Testbed corpus", scale=1)
        go = gr.Button("Search", variant="secondary", scale=1)
    with gr.Row():
        buttons = {k: gr.Button(f"{k.capitalize()}: {v['query'][:45]}…", variant="primary" if k == "hero" else "secondary")
                   for k, v in DEMO.items()}
    info = gr.Markdown()
    with gr.Row():
        cols = [gr.Markdown() for _ in SYSTEMS]
    btns = list(buttons.values())
    for trigger in (go.click, query.submit):
        trigger(search, [query, dataset], [info] + cols).then(selected, query, btns)
    for k, b in buttons.items():
        b.click(lambda k=k: (DEMO[k]["query"], DEMO[k].get("dataset", "scifact")), None, [query, dataset]).then(
            selected, query, btns).then(search, [query, dataset], [info] + cols)

if __name__ == "__main__":
    app.launch(server_name="127.0.0.1", server_port=int(os.environ.get("GRADIO_SERVER_PORT", 7860)))
