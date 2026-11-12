"""Checks on method.py, on a small part of the data. Run: python test_method.py"""
import numpy as np

from encoder import encode, load_model, load_tokenizer
from method import embed_dataset, search, testbed

QWEN = "Qwen/Qwen2.5-0.5B"


def test_search():
    docs = np.array([[1.0, 0.0], [0.0, 1.0], [0.6, 0.8]])
    queries = np.array([[0.0, 1.0]])
    run = search(docs, queries, ["a", "b", "c"], ["q"])
    assert list(run["q"]) == ["b", "c", "a"], run
    assert abs(run["q"]["c"] - 0.8) < 1e-6
    print("ok search ranks by cosine")


def test_format_with_the_shared_scorer():
    # random embeddings on the full dataset: the shared script must accept our run, with a bad score
    rng = np.random.default_rng(0)
    for name in testbed.DATASETS:
        corpus, queries, _ = testbed.load(name)
        docs = rng.normal(size=(len(corpus), 16))
        qs = rng.normal(size=(len(queries), 16))
        run = search(docs, qs, list(corpus), list(queries))
        assert all(len(r) == testbed.K + 1 for r in run.values())
        _, ndcg, recall = testbed.score_run(run, name)
        assert ndcg.mean() < 0.1, ndcg.mean()
        print(f"ok {name}: shared scorer accepts the run, random nDCG@10 {ndcg.mean():.3f}, Recall@100 {recall.mean():.3f}")


def test_small_run(model, tok):
    # a text must be its own best match when it is both the query and a document
    corpus, _, _ = testbed.load("scifact")
    small = dict(list(corpus.items())[:30])
    texts = testbed.doc_texts(small)
    docs = encode(model, tok, texts, "bi", max_length=512)
    qs = encode(model, tok, texts[:3], "bi", max_length=512)
    run = search(docs, qs, list(small), ["q0", "q1", "q2"])
    for i, q in enumerate(["q0", "q1", "q2"]):
        assert next(iter(run[q])) == list(small)[i]
    print("ok a text finds itself first")

    queries = {"q0": "Vitamin D lowers the risk of cancer.", "q1": "Aspirin helps the heart."}
    d, q = embed_dataset(model, tok, "scifact", "bi", "mean", small, queries)
    assert d.shape == (30, model.config.hidden_size) and q.shape == (2, model.config.hidden_size)
    assert np.isfinite(d).all() and np.isfinite(q).all()
    assert abs(np.linalg.norm(q[0]) - 1) < 1e-3
    print("ok embed_dataset gives normalized embeddings")


if __name__ == "__main__":
    test_search()
    test_format_with_the_shared_scorer()
    tok = load_tokenizer(QWEN)
    test_small_run(load_model(QWEN), tok)
    print("all checks passed")
