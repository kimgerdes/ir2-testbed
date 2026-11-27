# Rank1: Test-Time Compute for Reranking in Information Retrieval (Weller et al., COLM 2025)

Pair 14: Matteo Lombardi, Sukhjot Kaur. Method: `jhu-clsp/rank1-3b` (released checkpoint, no training) reasons about each query–document pair of the BM25 top-20 and scores it with P(true); positions 21–100 keep the BM25 order, so Recall@100 is BM25's. Compared with `ce`, and with the same cross-encoder on our top-20 (`vs_ce_top20` in `results.json`).

How to run, from the repo root on a GPU (Python 3.10, about 5 h on one T4):

```
pip install vllm==0.7.2 transformers==4.48.2
python pairs/14-rank1/method.py
python testbed.py score pairs/14-rank1/runs --method "Rank1-3B (reasoning), BM25 top-20" --pair 14-rank1 --against ce
```

Other runs of the analysis: `--model jhu-clsp/rank1-0.5b` or `rank1-1.5b`, `--no_reasoning`, `--authors_prompt` (the paper's dataset prompts), `--out <dir>`. `runs/pairs/` keeps P(true) and the reasoning of every pair.

- `reproduce.ipynb`: Rank1-7B on BRIGHT Biology and TheoremQA-theorems (paper, Table 1), with and without reasoning, authors' prompt vs raw query (Kaggle, 2 × T4, about 14 h).
- `analysis.ipynb`: all testbed runs: model size, reasoning, prompt, cost, score polarization, error analysis, and why the paper's TheoremQA first stage has 78 queries where BRIGHT now has 76.
- `slides.pdf`: the presentation.
- `demo.py`, `demo/`: the demo in the browser (queries in `demo.md`). Hero and failure are precomputed in `demo/queries.json`; typed claims run live, on a Mac with Apple silicon (MLX):

```
pip install -r requirements.txt mlx-lm
python pairs/14-rank1/demo.py --prepare && python pairs/14-rank1/demo.py
```

| Dataset | Rank1-3B | ce (top-100) | Δ nDCG@10 [95% CI] | p | Δ vs ce on our top-20 | p |
|---|:-:|:-:|:-:|:-:|:-:|:-:|
| SciFact | 0.729 | 0.688 | +0.041 [+0.007, +0.076] | 0.021 | +0.039 | 0.009 |
| NFCorpus | 0.338 | 0.347 | −0.009 [−0.023, +0.005] | 0.22 | −0.004 | 0.50 |
| ArguAna | 0.344 | 0.422 | −0.078 [−0.099, −0.058] | < 0.001 | −0.116 | < 0.001 |