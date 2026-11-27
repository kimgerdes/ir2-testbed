"""Writes 8 hypothetical documents per test query with Qwen2.5-3B-Instruct (8-bit, MLX, Apple silicon).
Sampling as in the paper's code: temperature 0.7, top-p 1, at most 512 tokens.
Output: generations/<dataset>.jsonl.gz, one line per query: {"qid": ..., "docs": [8 texts]}.
Stopping and restarting continues where it stopped. Needs: pip install mlx-lm
"python generate.py plain" is the second run: system message prompts.PLAIN, at most 256 tokens, generations_plain/."""
import gzip
import json
import sys
from pathlib import Path
import mlx.core as mx
from mlx_lm import batch_generate, load
from mlx_lm.sample_utils import make_sampler

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
import testbed
from prompts import PLAIN, PROMPTS

N, BATCH = 8, 8                          # documents per query, queries per batch
plain = sys.argv[1:] == ["plain"]
system = [{"role": "system", "content": PLAIN}] if plain else []
out = HERE / ("generations_plain" if plain else "generations")
model, tok = load("mlx-community/Qwen2.5-3B-Instruct-8bit")
sampler = make_sampler(temp=0.7, top_p=1.0)
mx.random.seed(0)
out.mkdir(exist_ok=True)
for name in testbed.DATASETS:
    corpus, queries, qrels = testbed.load(name)
    path = out / f"{name}.jsonl.gz"
    done = {json.loads(line)["qid"] for line in gzip.open(path, "rt")} if path.exists() else set()
    todo = [q for q in queries if q not in done]
    for i in range(0, len(todo), BATCH):
        qids = todo[i:i + BATCH]
        prompts = [tok.apply_chat_template(system + [{"role": "user", "content": PROMPTS[name].format(queries[q])}],
                                           add_generation_prompt=True) for q in qids for _ in range(N)]
        texts = batch_generate(model, tok, prompts, max_tokens=256 if plain else 512, sampler=sampler).texts
        with gzip.open(path, "at") as f:
            for j, q in enumerate(qids):
                f.write(json.dumps({"qid": q, "docs": [t.strip() for t in texts[j * N:(j + 1) * N]]}) + "\n")
        print(name, len(done) + i + len(qids), "of", len(queries), flush=True)
