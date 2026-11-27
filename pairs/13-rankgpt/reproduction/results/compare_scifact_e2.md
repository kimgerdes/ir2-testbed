# scifact: paired comparison of A, B, C

| run | name | queries | nDCG@10 | window / step | passage words | windows / query | prompt tokens | s / query | vs BM25 | vs ce |
|---|---|---|---|---|---|---|---|---|---|---|
| A | rankzephyr_colab_w20s10_top50 | 300 | 73.92 | 20 / 10 | fitted to context | 4.0 | 3965 | 20.5 | +6.31 [+3.56, +9.10], p=1.35e-05 | +5.11 [+2.27, +7.96], p=0.000565 |
| B | rankzephyr_abl_w10s5_top50 | 300 | 74.37 | 10 / 5 | fitted to context | 9.0 | 3482 | 24.2 | +6.75 [+4.62, +9.00], p=4.08e-09 | +5.56 [+2.80, +8.36], p=0.000101 |
| C | rankzephyr_abl_pw100_w10s5_top50 | 300 | 74.95 | 10 / 5 | 100 | 9.0 | 2039 | 21.9 | +7.34 [+4.90, +9.87], p=2.04e-08 | +6.14 [+3.36, +9.02], p=2.61e-05 |

| difference | queries | nDCG@10 points [95% CI], p |
|---|---|---|
| C − B | 300 | +0.58 [-1.14, +2.32], p=0.507 |
| C − A | 300 | +1.03 [-0.20, +2.28], p=0.103 |
| B − A | 300 | +0.45 [-1.44, +2.37], p=0.649 |
