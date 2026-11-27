"""Local demo page: BGE-small next to HyDE + BGE-small on cached queries, and on a typed SciFact claim with --live.
python pairs/15-hyde/demo_web.py [--live], then open http://127.0.0.1:8765. Illustration only: nothing here is scored."""
import json
import sys
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse
import numpy as np
from sentence_transformers import SentenceTransformer

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))

import testbed
import hyde
from prompts import PROMPTS

INDEX = HERE / "runs" / "demo"
model = SentenceTransformer("BAAI/bge-small-en-v1.5", device=testbed.device())
load_generations = lru_cache(hyde.load_generations)
data = {}
live = None
if "--live" in sys.argv:
    try:
        import mlx.core as mx
        from mlx_lm import batch_generate, load
        from mlx_lm.sample_utils import make_sampler
        live = load("mlx-community/Qwen2.5-3B-Instruct-8bit")
    except ImportError:
        print("mlx-lm is missing: live generation is off")


def dataset(name):
    if name not in data:
        corpus, queries, qrels = testbed.load(name)
        path = INDEX / f"{name}.npy"
        if not path.exists():
            INDEX.mkdir(parents=True, exist_ok=True)
            np.save(path, model.encode(testbed.doc_texts(corpus), batch_size=64, normalize_embeddings=True, show_progress_bar=True))
        data[name] = corpus, queries, qrels, list(corpus), np.load(path)
    return data[name]


def rank(name, vec, qid):
    corpus, _, qrels, ids, D = dataset(name)
    rows = []
    for j in np.argsort(-(D @ vec))[:11]:
        d = ids[j]
        if d != qid:
            title = corpus[d]["text"] if name == "arguana" else corpus[d]["title"] or corpus[d]["text"]
            rows.append({"id": d, "score": float(D[j] @ vec), "title": title[:160], "rel": qrels[qid].get(d, 0) > 0 if qid else None})
    return rows[:10]


def run(name, text, docs, qid):
    Q = model.encode([testbed.BGE_QUERY + text], normalize_embeddings=True)
    E = model.encode(docs, normalize_embeddings=True)[None]
    ranks = []
    for n in range(len(docs) + 1):
        V = hyde.hyde_vectors(E, Q, n)[0]
        ranks.append(rank(name, V / np.linalg.norm(V), qid))
    return {"query": text, "docs": docs, "base": ranks[0], "hyde": ranks}


def generate(text):
    mx.random.seed(0)
    llm, tok = live
    prompt = tok.apply_chat_template([{"role": "user", "content": PROMPTS["scifact"].format(text)}], add_generation_prompt=True)
    texts = batch_generate(llm, tok, [prompt] * 8, max_tokens=512, sampler=make_sampler(temp=0.7, top_p=1.0)).texts
    return [t.strip() for t in texts]


class Handler(BaseHTTPRequestHandler):
    def reply(self, body, kind="application/json"):
        raw = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        url = urlparse(self.path)
        p = {k: v[0] for k, v in parse_qs(url.query).items()}
        if url.path == "/":
            self.reply((HERE / "demo.html").read_bytes(), "text/html; charset=utf-8")
        elif url.path == "/api/config":
            self.reply({"live": live is not None})
        elif url.path == "/api/queries":
            queries = dataset(p["dataset"])[1]
            self.reply([{"id": q, "text": t} for q, t in queries.items() if p["q"].lower() in t.lower()][:20])
        elif url.path == "/api/run":
            queries = dataset(p["dataset"])[1]
            self.reply(run(p["dataset"], queries[p["qid"]], load_generations(p["dataset"], p["cache"])[p["qid"]], p["qid"]))
        else:
            self.send_error(404)

    def do_POST(self):
        if live is None:
            return self.send_error(400)
        text = json.loads(self.rfile.read(int(self.headers["Content-Length"])))["text"]
        self.reply(run("scifact", text, generate(text), None))


if __name__ == "__main__":
    for name in testbed.DATASETS:
        dataset(name)
    print("http://127.0.0.1:8765")
    HTTPServer(("127.0.0.1", 8765), Handler).serve_forever()
