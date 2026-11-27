"""Paths, seeds and small IO helpers shared by all scripts.

In the research repository, importing this module before transformers points HF_HOME at its hf_cache directory.
In this copy (pairs/13-rankgpt/reproduction) HF_HOME is left to Hugging Face's default: set it to reuse a model
cache.
"""
import json
import os
import pathlib
import random

ROOT = pathlib.Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW = DATA / "raw"
PROC = DATA / "processed"
RUNS = ROOT / "runs"
RESULTS = ROOT / "results"
CACHE = ROOT / "cache"
MODELS = ROOT / "models"
HF_CACHE = ROOT / "hf_cache"

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("PYTHONUTF8", "1")

BEIR_SETS = ("scifact", "nfcorpus", "arguana")
DOC_MAX_WORDS = 200          # testbed convention: documents are cut to 200 words
BM25_K1, BM25_B = 0.9, 0.4   # pyserini defaults used by RankGPT


def ensure_dirs():
    for d in (PROC, RUNS, RESULTS, CACHE, MODELS):
        d.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int):
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def jdump(obj, path, indent=2):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=indent)


def jload(path):
    with open(path, encoding="utf8") as f:
        return json.load(f)


def read_jsonl(path):
    with open(path, encoding="utf8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(rows, path):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def normalize_ws(s: str) -> str:
    return " ".join((s or "").split())


def truncate_words(s: str, max_words):
    if max_words is None:
        return s
    return " ".join(s.split()[: int(max_words)])
