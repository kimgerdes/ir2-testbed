# nfcorpus: paired comparison of top50, top100

| run | name | queries | nDCG@10 | window / step | passage words | windows / query | prompt tokens | s / query | vs BM25 | vs ce |
|---|---|---|---|---|---|---|---|---|---|---|
| top50 | rankzephyr_colab_w20s10_top50 | 323 | 36.06 | 20 / 10 | fitted to context | 4.0 | 3966 | 21.7 | +3.99 [+2.77, +5.24], p=1.59e-09 | +1.40 [+0.25, +2.55], p=0.0187 |
| top100 | rankzephyr_depth_w20s10_top100 | 323 | 36.16 | 20 / 10 | fitted to context | 9.0 | 3966 | 48.5 | +4.09 [+2.82, +5.42], p=2.75e-09 | +1.50 [+0.31, +2.74], p=0.0167 |

| difference | queries | nDCG@10 points [95% CI], p |
|---|---|---|
| top100 − top50 | 323 | +0.10 [-0.60, +0.82], p=0.776 |
