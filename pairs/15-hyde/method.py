import sys
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import testbed
import hyde

MODEL_NAME = "BAAI/bge-small-en-v1.5"
RUNS_DIR = HERE / "runs"


def main():
    model = SentenceTransformer(MODEL_NAME, device=testbed.device())
    for name in testbed.DATASETS:
        corpus, queries, _ = testbed.load(name)
        ids, qids = list(corpus), list(queries)

        D = model.encode(
            testbed.doc_texts(corpus),
            batch_size=64,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

        Q = model.encode(
            [testbed.BGE_QUERY + queries[q] for q in qids],
            batch_size=64,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

        gens = hyde.load_generations(name, folder="generations")
        E = hyde.embed_generations(model, qids, gens, normalize=True)

        V = hyde.hyde_vectors(E, Q, n=8)
        V = V / np.linalg.norm(V, axis=1, keepdims=True)

        run = hyde.search(V, D, qids, ids)
        testbed.save_run(run, RUNS_DIR / f"{name}.json")
        print(f"saved {name} run to {RUNS_DIR / f'{name}.json'}")


if __name__ == "__main__":
    main()
