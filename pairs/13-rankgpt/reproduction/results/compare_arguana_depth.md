# arguana: paired comparison of RankZephyr, ce_top20, ce_top50

| run | name | queries | nDCG@10 | window / step | passage words | windows / query | prompt tokens | s / query | vs BM25 | vs ce |
|---|---|---|---|---|---|---|---|---|---|---|
| RankZephyr | rankzephyr_w20s10_top20 | 1406 | 50.40 | 20 / 10 | fitted to context | 1.0 | 3961 | 12.6 | -1.03 [-2.22, +0.15], p=0.0932 | +8.17 [+6.29, +10.04], p=5.55e-17 |
| ce_top20 | ce_minilm_top20 | 1406 | 45.98 | - / - | fitted to context | nan | 0 | 0.0 | -5.45 [-7.22, -3.71], p=2.1e-09 | +3.75 [+3.03, +4.44], p=1.6e-24 |
| ce_top50 | ce_minilm_top50 | 1406 | 43.61 | - / - | fitted to context | nan | 0 | 0.1 | -7.82 [-9.69, -5.95], p=9.25e-16 | +1.38 [+1.07, +1.70], p=8.62e-17 |

| difference | queries | nDCG@10 points [95% CI], p |
|---|---|---|
| ce_top50 − ce_top20 | 1406 | -2.37 [-2.98, -1.74], p=1.08e-13 |
| ce_top50 − RankZephyr | 1406 | -6.79 [-8.62, -4.95], p=7.47e-13 |
| ce_top20 − RankZephyr | 1406 | -4.42 [-6.13, -2.70], p=5.37e-07 |

**Mixed quantisation** {'RankZephyr': '4bit', 'ce_top20': None, 'ce_top50': None}: a difference between runs with different quantisation is not a clean comparison.
