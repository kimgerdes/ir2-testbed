"""Rank1 demo in the browser, on SciFact: BM25 → the spring cross-encoder → Rank1, side by side, with Rank1's reasoning.

Once, from the repo root, on a Mac with Apple silicon (Rank1 runs live with MLX):
    pip install -r requirements.txt mlx-lm
    python pairs/14-rank1/demo.py --prepare     # BM25 index of SciFact, downloads the cross-encoder and Rank1-1.5B
Then:
    python pairs/14-rank1/demo.py               # opens http://127.0.0.1:8014

Hero and failure are precomputed (demo/queries.json, from our Kaggle runs T0, T1-3b and T2-3b on the BM25 top 20).
A typed claim runs live: BM25 top 10 → cross-encoder → Rank1-1.5B (4-bit) answers each abstract at once,
then reasons in full on each of them, and the page shows the reasoning as it is written."""
import argparse
import json
import math
import pickle
import queue
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
import testbed

WEB = HERE / "demo"
QUERIES = WEB / "queries.json"
INDEX = WEB / "index" / "scifact_bm25.pkl"
OUTPUTS = HERE / "outputs"
PORT = 8014
CE_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
LIVE_MODEL = "jhu-clsp/rank1-1.5b"
RUN_DEPTH, LIVE_DEPTH, MAX_THINK = 20, 10, 1024
PREPARED = [("hero", "Hero", "238"), ("fail", "Failure", "268")]
INSTRUCTION = "Determine if the following passage is relevant to the query. Answer only with 'true' or 'false'.\n"


def prompt(query, passage):
    return INSTRUCTION + f"Query: {query}\nPassage: {passage}\n<think>"


def analyzer():
    # the testbed's BM25 preprocessing, so that live ranks match the runs
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer
    from sklearn.feature_extraction.text import CountVectorizer
    nltk.download("stopwords", quiet=True)
    sw, porter, analyze = set(stopwords.words("english")), PorterStemmer(), CountVectorizer().build_analyzer()
    return lambda t: [porter.stem(x) for x in analyze(t) if x not in sw]


def read_jsonl(path):
    out = {}
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        out[(r["qid"], r["did"])] = r
    return out


def build_index():
    from rank_bm25 import BM25Okapi
    corpus, _, _ = testbed.load("scifact")
    pre = analyzer()
    print("BM25 index of SciFact", flush=True)
    bm25 = BM25Okapi([pre(t) for t in testbed.doc_texts(corpus)], k1=1.2, b=0.75)
    INDEX.parent.mkdir(parents=True, exist_ok=True)
    pickle.dump({"ids": list(corpus), "bm25": bm25}, open(INDEX, "wb"))


def build_queries():
    # hero and failure from our Kaggle runs: the cross-encoder (T0) and Rank1-3B with (T1) and without (T2) reasoning
    corpus, queries, qrels = testbed.load("scifact")
    index, pre = pickle.load(open(INDEX, "rb")), analyzer()
    ids, bm25, text_of = index["ids"], index["bm25"], dict(zip(corpus, testbed.doc_texts(corpus)))
    ce = read_jsonl(OUTPUTS / "results" / "T0" / "scifact.jsonl")
    r1 = read_jsonl(OUTPUTS / "results" / "T1-3b" / "scifact.jsonl")
    r2 = read_jsonl(OUTPUTS / "results" / "T2-3b" / "scifact.jsonl")
    prepared = []
    for key, label, qid in PREPARED:
        s = bm25.get_scores(pre(queries[qid]))
        top = [ids[j] for j in sorted(range(len(ids)), key=lambda j: -s[j])[:RUN_DEPTH]]
        if set(top) != {d for q, d in r1 if q == qid}:
            print(f"  query {qid}: BM25 top {RUN_DEPTH} differs from the run", flush=True)
        docs = [{"id": d, "title": corpus[d]["title"], "text": text_of[d], "labelled": d in qrels[qid],
                 "bm25": round(float(s[ids.index(d)]), 3), "ce": round(ce[(qid, d)]["score"], 4),
                 "r1": round(r1[(qid, d)]["score"], 6), "r2": round(r2[(qid, d)]["score"], 6),
                 "reasoning": r1[(qid, d)]["text"].strip(), "tokens": r1[(qid, d)]["ntok"]} for d in top]
        prepared.append({"key": key, "label": label, "qid": qid, "claim": queries[qid], "docs": docs})
        order = lambda f: sorted(range(len(docs)), key=lambda i: -docs[i][f])
        for i, d in enumerate(docs):
            if d["labelled"]:
                print(f"  {label:9} {qid}: BM25 {i + 1}, cross-encoder {order('ce').index(i) + 1}, "
                      f"Rank1-3B {order('r1').index(i) + 1}, without reasoning {order('r2').index(i) + 1}", flush=True)
    json.dump({"queries": prepared}, open(QUERIES, "w", encoding="utf-8"), ensure_ascii=False)
    print(f"wrote {QUERIES.relative_to(ROOT)}", flush=True)


def prepare():
    build_index()
    if (OUTPUTS / "results").exists():
        build_queries()
    print("downloading the cross-encoder and Rank1", flush=True)
    from sentence_transformers import CrossEncoder
    CrossEncoder(CE_MODEL)
    try:
        from mlx_lm import load
        load(LIVE_MODEL)
    except ImportError:
        print("mlx-lm is not installed (Mac with Apple silicon only): the precomputed queries work, live claims do not", flush=True)
    print("ready: python pairs/14-rank1/demo.py", flush=True)


class Job:
    def __init__(self, claim):
        self.claim, self.events, self.cancel = claim, queue.Queue(), threading.Event()


class Engine(threading.Thread):
    """All model work runs in this one thread: MLX streams belong to the thread that created them."""

    def __init__(self):
        super().__init__(daemon=True)
        self.jobs, self.status, self.current = queue.Queue(), "loading", None

    def load(self):
        corpus, queries, qrels = testbed.load("scifact")
        index = pickle.load(open(INDEX, "rb"))
        self.ids, self.bm25, self.pre = index["ids"], index["bm25"], analyzer()
        self.corpus, self.text = corpus, dict(zip(corpus, testbed.doc_texts(corpus)))
        self.labels = {" ".join(queries[q].lower().split()): set(qrels[q]) for q in queries}
        from sentence_transformers import CrossEncoder
        self.ce = CrossEncoder(CE_MODEL, device="cpu")
        import mlx.core as mx
        import mlx.nn as nn
        from mlx_lm import load, stream_generate
        self.mx, self.stream_generate = mx, stream_generate
        self.model, self.tok = load(LIVE_MODEL)
        nn.quantize(self.model, group_size=64, bits=4)
        mx.eval(self.model.parameters())
        self.true_false = (self.tok.encode(" true")[0], self.tok.encode(" false")[0])

    def run(self):
        try:
            self.load()
            self.status = "ready"
        except Exception as e:
            self.status = f"unavailable: {type(e).__name__}: {e}"
            return
        while True:
            job = self.jobs.get()
            try:
                for event in self.live(job):
                    if job.cancel.is_set():
                        break
                    job.events.put(event)
            except Exception as e:
                job.events.put({"type": "error", "message": f"{type(e).__name__}: {e}"})
            job.events.put(None)

    def submit(self, claim):
        if self.current:
            self.current.cancel.set()
        self.current = Job(claim)
        self.jobs.put(self.current)
        return self.current

    def p_true(self, logprobs):
        t, f = (math.exp(logprobs[i].item()) for i in self.true_false)
        return t / (t + f)

    def answer(self, text):
        # one forward pass after </think>: P(true) of the next token
        logits = self.model(self.mx.array(self.tok.encode(text))[None])[0, -1]
        return self.p_true(logits - self.mx.logsumexp(logits))

    def reason(self, job, p):
        # yields the reasoning as it is written, then (P(true), tokens, capped)
        text, n, closed = "", 0, False
        for r in self.stream_generate(self.model, self.tok, self.tok.encode(p), max_tokens=MAX_THINK):
            if job.cancel.is_set():
                return
            n += 1
            if closed:
                yield self.p_true(r.logprobs), n, False
                return
            text += r.text
            closed = text.rstrip().endswith("</think>")
            if not closed:
                yield r.text
        # token cap: cut at the last full sentence and close the block, as the authors do
        cut = max(text.rfind("."), text.rfind("!"), text.rfind("?"))
        yield self.answer(f"{p}\n{(text[:cut + 1] if cut != -1 else text).strip()}\n</think>"), n, True

    def live(self, job):
        t0 = time.time()
        claim = job.claim
        s = self.bm25.get_scores(self.pre(claim))
        top = [self.ids[j] for j in sorted(range(len(self.ids)), key=lambda j: -s[j])[:LIVE_DEPTH]]
        labelled = self.labels.get(" ".join(claim.lower().split()), set())
        ce = self.ce.predict([[claim, self.text[d]] for d in top], show_progress_bar=False)
        docs = [{"id": d, "title": self.corpus[d]["title"], "text": self.text[d], "labelled": d in labelled,
                 "bm25": round(float(s[self.ids.index(d)]), 3), "ce": round(float(c), 4)} for d, c in zip(top, ce)]
        yield {"type": "candidates", "claim": claim, "known": bool(labelled), "docs": docs}
        yield {"type": "r2", "scores": {d: round(self.answer(prompt(claim, self.text[d]) + "\n\n</think>"), 6) for d in top}}
        for d in top:
            yield {"type": "start", "id": d}
            started = time.time()
            for out in self.reason(job, prompt(claim, self.text[d])):
                if isinstance(out, str):
                    yield {"type": "text", "id": d, "text": out}
                else:
                    p, n, capped = out
                    yield {"type": "verdict", "id": d, "r1": round(p, 6), "tokens": n, "capped": capped,
                           "seconds": round(time.time() - started, 1)}
        yield {"type": "done", "seconds": round(time.time() - t0, 1)}


class Handler(BaseHTTPRequestHandler):
    engine, prepared = None, {}
    static = {"/": ("index.html", "text/html"), "/style.css": ("style.css", "text/css"),
              "/app.js": ("app.js", "application/javascript")}

    def log_message(self, *args):
        pass

    def send(self, body, kind="application/json", code=200):
        body = body if isinstance(body, bytes) else json.dumps(body, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", f"{kind}; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        url = urlparse(self.path)
        if url.path in self.static:
            name, kind = self.static[url.path]
            return self.send((WEB / name).read_bytes(), kind)
        font = WEB / "fonts" / Path(url.path).name
        if url.path.startswith("/fonts/") and font.suffix == ".woff2" and font.exists():
            return self.send(font.read_bytes(), "font/woff2")
        if url.path == "/api/state":
            return self.send({"queries": [{k: q[k] for k in ("key", "label", "qid", "claim")} for q in self.prepared.values()],
                              "live": self.engine.status, "model": LIVE_MODEL.split("/")[-1]})
        if url.path.startswith("/api/query/") and url.path[11:] in self.prepared:
            return self.send(self.prepared[url.path[11:]])
        if url.path == "/api/live":
            claim = " ".join(parse_qs(url.query).get("q", [""])[0].split())
            if not claim or self.engine.status != "ready":
                return self.send({"error": "live model not ready"}, code=503)
            return self.stream(self.engine.submit(claim))
        self.send({"error": "not found"}, code=404)

    def stream(self, job):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            while True:
                event = job.events.get()
                if event is None:
                    break
                self.wfile.write(f"data: {json.dumps(event, ensure_ascii=False)}\n\n".encode())
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            job.cancel.set()


def serve(port, browser):
    if not QUERIES.exists() or not INDEX.exists():
        sys.exit("run python pairs/14-rank1/demo.py --prepare first")
    Handler.prepared = {q["key"]: q for q in json.load(open(QUERIES, encoding="utf-8"))["queries"]}
    Handler.engine = Engine()
    Handler.engine.start()
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    url = f"http://127.0.0.1:{port}"
    print(f"demo on {url} (the live model loads in the background); Ctrl+C to stop", flush=True)
    if browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepare", action="store_true", help="build the BM25 index and demo/queries.json, download the models")
    ap.add_argument("--port", type=int, default=PORT)
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args()
    prepare() if a.prepare else serve(a.port, not a.no_browser)
