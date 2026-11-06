"""Our own static retriever, trained from random init with the blog's recipe at 1/40 scale (MS MARCO triplets only,
MNRL + Matryoshka, batch 2048, lr 0.2, 1 epoch; ~2 min data prep + ~10 s of training on an RTX 5050). Corpus and qrels of the
testbed are never seen. Usage: python pairs/6-static-embeddings/train_static.py  ->  models/static-mrl/"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from static_emb.train import TrainConfig, train

model, history = train(TrainConfig(seed=12))          # every hyper-parameter is annotated in static_emb/train.py
model.save(str(HERE / "models" / "static-mrl"))
print(f"trained on {history['n_train']:,} triplets in {history['train_s']:.0f}s")
