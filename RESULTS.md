# Shared results table

Built by `build_table.py` from `baselines/scores.json` and `pairs/*/results.json`: do not edit by hand.
A difference is in bold only when its p-value is below 0.05 and its confidence interval excludes zero.
Pair 00 is the instructor's worked example from the model presentation.

## SciFact

| Method | nDCG@10 | Recall@100 | Compared with | Δ nDCG@10 [95 % CI] | p |
|---|:-:|:-:|---|:-:|:-:|
| *BM25* | 0.676 | 0.918 | | | |
| *all-MiniLM-L6-v2* | 0.646 | 0.925 | | | |
| *bge-small-en-v1.5* | 0.710 | 0.948 | | | |
| *RRF (BM25 + MiniLM)* | 0.718 | 0.948 | | | |
| *BM25 top-100 + cross-encoder* | 0.688 | 0.918 | | | |
| Beyond Abstracts (ModernBERT, both_random) ([00-beyond-abstracts](pairs/00-beyond-abstracts/)) | 0.717 | 0.982 | bge-small-en-v1.5 | +0.007 [-0.020, +0.034] | 0.626 |
| static-retrieval-mrl-en-v1 (static, 2025) ([6-static-embeddings](pairs/6-static-embeddings/)) | 0.591 | 0.901 | all-MiniLM-L6-v2 | **-0.054 [-0.088, -0.020]** | 0.002 |

## NFCorpus

| Method | nDCG@10 | Recall@100 | Compared with | Δ nDCG@10 [95 % CI] | p |
|---|:-:|:-:|---|:-:|:-:|
| *BM25* | 0.321 | 0.246 | | | |
| *all-MiniLM-L6-v2* | 0.316 | 0.311 | | | |
| *bge-small-en-v1.5* | 0.340 | 0.304 | | | |
| *RRF (BM25 + MiniLM)* | 0.342 | 0.310 | | | |
| *BM25 top-100 + cross-encoder* | 0.347 | 0.246 | | | |
| Beyond Abstracts (ModernBERT, both_random) ([00-beyond-abstracts](pairs/00-beyond-abstracts/)) | 0.313 | 0.324 | bge-small-en-v1.5 | **-0.027 [-0.047, -0.008]** | 0.006 |
| static-retrieval-mrl-en-v1 (static, 2025) ([6-static-embeddings](pairs/6-static-embeddings/)) | 0.301 | 0.280 | all-MiniLM-L6-v2 | -0.015 [-0.034, +0.004] | 0.115 |

## ArguAna

| Method | nDCG@10 | Recall@100 | Compared with | Δ nDCG@10 [95 % CI] | p |
|---|:-:|:-:|---|:-:|:-:|
| *BM25* | 0.516 | 0.969 | | | |
| *all-MiniLM-L6-v2* | 0.503 | 0.977 | | | |
| *bge-small-en-v1.5* | 0.606 | 0.986 | | | |
| *RRF (BM25 + MiniLM)* | 0.549 | 0.985 | | | |
| *BM25 top-100 + cross-encoder* | 0.422 | 0.969 | | | |
| Beyond Abstracts (ModernBERT, both_random) ([00-beyond-abstracts](pairs/00-beyond-abstracts/)) | 0.461 | 0.983 | bge-small-en-v1.5 | **-0.146 [-0.165, -0.126]** | < 0.001 |
| static-retrieval-mrl-en-v1 (static, 2025) ([6-static-embeddings](pairs/6-static-embeddings/)) | 0.453 | 0.967 | all-MiniLM-L6-v2 | **-0.049 [-0.067, -0.032]** | < 0.001 |
