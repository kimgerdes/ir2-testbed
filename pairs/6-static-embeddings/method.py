"""Static embeddings on the shared testbed: the paper's released static retriever replaces the spring bi-encoder.
Writes one run per dataset to runs/, then:
python testbed.py score pairs/6-static-embeddings/runs --method "static-retrieval-mrl-en-v1 (static, 2025)" \
    --pair 6-static-embeddings --against minilm"""
import sys
from pathlib import Path
from sentence_transformers import SentenceTransformer

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
import testbed

# The encoder is a lookup table + mean (no attention): no query prefix, no length limit to set.
model = SentenceTransformer("sentence-transformers/static-retrieval-mrl-en-v1", device=testbed.device())
for name in testbed.DATASETS:
    corpus, queries, qrels = testbed.load(name)      # never tune on qrels: they are the test labels
    run = testbed.dense_search(model, corpus, queries, batch_size=1024)
    testbed.save_run(run, HERE / "runs" / f"{name}.json")
