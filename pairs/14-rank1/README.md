# Rank1: Test-Time Compute for Reranking in Information Retrieval (Weller et al., COLM 2025) — pair 14: Matteo Lombardi, Sukhjot Kaur

Method: `jhu-clsp/rank1-3b` (released checkpoint, no training) reasons about each query–document pair of the BM25 top-20 and scores it with P(true); BM25 positions 21–100 follow, so Recall@100 is BM25's. Compared with `ce`; `results.json` also has `vs_ce_top20`, the same cross-encoder on our top-20 and texts.
Run from the repo root on a GPU (Python 3.10, about 5 h on one T4): `pip install vllm==0.7.2 transformers==4.48.2`, then `python pairs/14-rank1/method.py` and `python testbed.py score pairs/14-rank1/runs --method "Rank1-3B (reasoning), BM25 top-20" --pair 14-rank1 --against ce` (= `results.json`). `runs/pairs/` keeps P(true) and the reasoning of every pair.
Other runs of the analysis: `--model jhu-clsp/rank1-0.5b` or `rank1-1.5b`, `--no_reasoning`, `--authors_prompt` (the dataset prompts of the paper), `--out`.
`reproduce.ipynb`: Rank1-7B on BRIGHT Biology and TheoremQA theorems (Table 1), with and without reasoning and the authors' prompt. `analysis.ipynb`: model size, reasoning, prompt, cost, score polarization, error analysis, and why the paper's BRIGHT first stage has 78 queries where BRIGHT now has 76.

| Dataset | Rank1-3B | ce (top-100) | Δ nDCG@10 [95% CI] | p | Δ vs ce on our top-20 | p |
|---|:-:|:-:|:-:|:-:|:-:|:-:|
| SciFact / NFCorpus / ArguAna | 0.729 / 0.338 / 0.344 | 0.688 / 0.347 / 0.422 | +0.041 [+0.007, +0.076] / −0.009 [−0.023, +0.005] / −0.078 [−0.099, −0.058] | 0.021 / 0.22 / < 0.001 | +0.039 / −0.004 / −0.116 | 0.009 / 0.50 / < 0.001 |

AI use: Claude (Anthropic) helped write code, notebooks, README and slides; we ran and checked all results ourselves (details in the report).
