# Cross-hardware replication: Colab against the laptop

nDCG@10 ×100. Laptop: RTX 4060 8 GB; Colab: one A100 session, 24–25 November 2026. Same code, data, prompts, models and quantisation; only the GPU differs. The laptop runs stay the submitted results.

Replicated runs: 32 of 32; identical query by query: 14.

**RankZephyr testbed runs**

| run | queries | laptop | Colab | Colab − laptop [95% CI], p | queries with the same nDCG@10 | windows with the same answer |
|---|---|---|---|---|---|---|
| scifact top-50 | 300 | 73.98 | 73.92 | -0.06 [-0.40, +0.28], p = 0.74 | 296 of 300 | 1086 of 1200 |
| nfcorpus top-50 | 323 | 36.09 | 36.06 | -0.03 [-0.33, +0.25], p = 0.83 | 312 of 323 | 1194 of 1292 |
| arguana top-20 | 1406 | 50.40 | 50.42 | +0.03 [-0.05, +0.11], p = 0.5 | 1398 of 1406 | 1341 of 1406 |

**RankZephyr NovelEval, Table 3**

| run | queries | laptop | Colab | Colab − laptop [95% CI], p | queries with the same nDCG@10 | windows with the same answer |
|---|---|---|---|---|---|---|
| shuffled seed0 | 21 | 89.97 | 89.97 | identical | 21 of 21 | 17 of 21 |
| shuffled seed1 | 21 | 89.42 | 89.42 | identical | 21 of 21 | 20 of 21 |
| shuffled seed2 | 21 | 88.12 | 88.12 | identical | 21 of 21 | 21 of 21 |
| bm25 | 21 | 88.89 | 88.89 | identical | 21 of 21 | 21 of 21 |
| released | 21 | 89.46 | 89.46 | identical | 21 of 21 | 21 of 21 |

**cross-encoder controls**

| run | queries | laptop | Colab | Colab − laptop [95% CI], p | queries with the same nDCG@10 | windows with the same answer |
|---|---|---|---|---|---|---|
| scifact top-50 | 300 | 68.53 | 68.53 | identical | 300 of 300 |  |
| scifact top-100 | 300 | 68.81 | 68.81 | identical | 300 of 300 |  |
| nfcorpus top-50 | 323 | 34.61 | 34.61 | +0.00 [+0.00, +0.00], p = 0.32 | 322 of 323 |  |
| nfcorpus top-100 | 323 | 34.66 | 34.67 | +0.00 [+0.00, +0.00], p = 0.32 | 322 of 323 |  |
| arguana top-20 | 1406 | 45.98 | 45.99 | +0.01 [-0.06, +0.07], p = 0.85 | 1399 of 1406 |  |
| arguana top-50 | 1406 | 43.61 | 43.57 | -0.05 [-0.11, +0.00], p = 0.13 | 1401 of 1406 |  |
| arguana top-100 | 1406 | 42.23 | 42.21 | -0.03 [-0.07, +0.00], p = 0.23 | 1402 of 1406 |  |
| noveleval | 21 | 77.08 | 77.08 | identical | 21 of 21 |  |

**distillation, Table 7: minilm_chatgpt**

| run | queries | laptop | Colab | Colab − laptop [95% CI], p | queries with the same nDCG@10 | windows with the same answer |
|---|---|---|---|---|---|---|
| scifact | 300 | 64.48 | 65.20 | +0.73 [-0.18, +1.68], p = 0.13 | 251 of 300 |  |
| nfcorpus | 323 | 33.62 | 33.27 | -0.36 [-0.78, +0.04], p = 0.084 | 190 of 323 |  |
| arguana | 1406 | 27.88 | 28.00 | +0.12 [-0.51, +0.74], p = 0.71 | 1020 of 1406 |  |
| noveleval | 21 | 72.47 | 75.89 | +3.42 [+1.29, +5.90], p = 0.012 | 7 of 21 |  |

**distillation, Table 7: minilm_marco_binary**

| run | queries | laptop | Colab | Colab − laptop [95% CI], p | queries with the same nDCG@10 | windows with the same answer |
|---|---|---|---|---|---|---|
| scifact | 300 | 21.62 | 16.86 | -4.77 [-6.88, -2.76], p = 7.2e-06 | 196 of 300 |  |
| nfcorpus | 323 | 22.42 | 18.78 | -3.64 [-4.82, -2.49], p = 2.5e-09 | 135 of 323 |  |
| arguana | 1406 | 2.55 | 2.23 | -0.32 [-0.75, +0.11], p = 0.14 | 1294 of 1406 |  |
| noveleval | 21 | 42.13 | 45.02 | +2.89 [-0.86, +6.21], p = 0.14 | 0 of 21 |  |

**Flan-T5 prompting, Table 4, SciFact 100 queries**

| run | queries | laptop | Colab | Colab − laptop [95% CI], p | queries with the same nDCG@10 | windows with the same answer |
|---|---|---|---|---|---|---|
| pointwise.yes_no | 100 | 51.50 | 51.50 | identical | 100 of 100 |  |
| pointwise.qlm | 100 | 70.17 | 70.19 | +0.02 [+0.00, +0.05], p = 0.32 | 99 of 100 |  |
| listwise.likelihood | 100 | 66.79 | 66.98 | +0.19 [+0.00, +0.51], p = 0.17 | 98 of 100 |  |
| listwise.generation | 100 | 68.32 | 68.32 | identical | 100 of 100 |  |
| setwise.heapsort | 100 | 63.21 | 63.21 | identical | 100 of 100 |  |
| setwise.bubblesort | 100 | 70.22 | 70.22 | identical | 100 of 100 |  |
| pairwise.heapsort | 100 | 74.71 | 74.71 | identical | 100 of 100 |  |
| pairwise.bubblesort | 100 | 75.90 | 75.90 | identical | 100 of 100 |  |

**Course testbed scorer, `testbed.py score --against ce`**

| run | dataset | laptop nDCG@10 | Colab nDCG@10 | laptop − `ce` [95% CI], p | Colab − `ce` [95% CI], p |
|---|---|---|---|---|---|
| RankZephyr (top-50, ArguAna top-20) | scifact | 0.740 | 0.739 | +0.052 [+0.023, +0.081], p = 0.0006 | +0.051 [+0.023, +0.080], p = 0.00057 |
| RankZephyr (top-50, ArguAna top-20) | nfcorpus | 0.361 | 0.361 | +0.014 [+0.002, +0.026], p = 0.026 | +0.013 [+0.002, +0.025], p = 0.023 |
| RankZephyr (top-50, ArguAna top-20) | arguana | 0.504 | 0.504 | +0.082 [+0.062, +0.100], p = 6.8e-17 | +0.082 [+0.062, +0.101], p = 5.3e-17 |
| cross-encoder on BM25 top-100 | scifact | 0.688 | 0.688 | -0.000 [-0.000, -0.000], p = 6.2e-07 | -0.000 [-0.000, -0.000], p = 6.2e-07 |
| cross-encoder on BM25 top-100 | nfcorpus | 0.347 | 0.347 | -0.001 [-0.001, +0.000], p = 0.14 | -0.000 [-0.001, +0.000], p = 0.15 |
| cross-encoder on BM25 top-100 | arguana | 0.422 | 0.422 | -0.000 [-0.001, +0.001], p = 0.78 | -0.000 [-0.001, +0.000], p = 0.38 |
| student distilled from ChatGPT | scifact | 0.645 | 0.652 | -0.043 [-0.071, -0.016], p = 0.0022 | -0.036 [-0.064, -0.009], p = 0.011 |
| student distilled from ChatGPT | nfcorpus | 0.336 | 0.333 | -0.011 [-0.021, -0.002], p = 0.024 | -0.014 [-0.024, -0.005], p = 0.003 |
| student distilled from ChatGPT | arguana | 0.279 | 0.280 | -0.144 [-0.163, -0.125], p = 7.4e-48 | -0.142 [-0.161, -0.124], p = 8e-47 |
| student trained on MS MARCO labels | scifact | 0.216 | 0.169 | -0.472 [-0.517, -0.427], p = 6.5e-60 | -0.519 [-0.569, -0.469], p = 2e-59 |
| student trained on MS MARCO labels | nfcorpus | 0.224 | 0.188 | -0.123 [-0.143, -0.103], p = 4.8e-27 | -0.159 [-0.182, -0.137], p = 3.4e-35 |
| student trained on MS MARCO labels | arguana | 0.025 | 0.022 | -0.397 [-0.417, -0.377], p = 7.9e-227 | -0.400 [-0.420, -0.380], p = 1.4e-228 |
