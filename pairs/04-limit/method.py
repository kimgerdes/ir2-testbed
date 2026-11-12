"""Our testbed row: GTE-ModernColBERT (multi-vector, exact MaxSim), the escape from the single-vector bound that
LIMIT points to, against bge-small, the closest single-vector baseline. Writes one run per dataset, then:
python testbed.py score pairs/04-limit/runs --method "LIMIT (GTE-ModernColBERT, MaxSim)" --pair 04-limit --against bge

  python pairs/04-limit/method.py:       GTE-ModernColBERT           -> runs/<dataset>.json        (results.json)
  python pairs/04-limit/method.py qwen3: Qwen3-Embedding-0.6B, 1024d -> runs/qwen3/<dataset>.json  (README only)"""

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts")) # scripts/limitlib.py
import limitlib
from limitlib import testbed

system = sys.argv[1] if len(sys.argv) > 1 else "colbert"
assert system in ("colbert", "qwen3"), "usage: method.py [colbert|qwen3]"
out = limitlib.HERE / "runs" / ("" if system == "colbert" else system)

for name in testbed.DATASETS:
    corpus, queries, qrels = testbed.load(name) # we never tune on qrels: they are the test labels
    if system == "colbert":
        D, Q = limitlib.colbert_embeddings(name, corpus, queries)
        scores = limitlib.maxsim_scores(Q, D)
    else:
        D, Q = limitlib.dense_embeddings(limitlib.QWEN, name, corpus, queries, batch_size=16)
        scores = limitlib.cosine_scores(Q, D)
    testbed.save_run(limitlib.to_run(scores, list(queries), list(corpus)), out / f"{name}.json")
    print(f"{name}: wrote {out / f'{name}.json'}", flush=True)