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
| LLM2Vec (Qwen2.5-0.5B) ([02-llm2vec](pairs/02-llm2vec/)) | 0.553 | 0.883 | bge-small-en-v1.5 | **-0.157 [-0.197, -0.117]** | < 0.001 |
| LIMIT (GTE-ModernColBERT, MaxSim) ([04-limit](pairs/04-limit/)) | 0.757 | 0.963 | bge-small-en-v1.5 | **+0.047 [+0.018, +0.076]** | 0.002 |
| gpl_BGE (eval_base) vs bge ([10_GPL](pairs/10-GPL/)) | 0.511 | 0.859 | bge-small-en-v1.5 | **-0.198 [-0.237, -0.160]** | < 0.001 |
| BGE-M3 (All) ([11-bge-m3](pairs/11-bge-m3/)) | 0.692 | 0.914 | RRF (BM25 + MiniLM) | -0.026 [-0.054, +0.003] | 0.078 |
| mat2vec word2vec + SIF (paper 12) ([12-mat2vec](pairs/12-mat2vec/)) | 0.414 | 0.808 | BM25 | **-0.262 [-0.307, -0.216]** | < 0.001 |
| RankGPT (RankZephyr-7B 4-bit; BM25 top-50, ArguAna top-20) ([13-rankgpt](pairs/13-rankgpt/)) | 0.740 | 0.918 | BM25 top-100 + cross-encoder | **+0.052 [+0.023, +0.081]** | < 0.001 |
| Rank1-3B (reasoning), BM25 top-20 ([14-rank1](pairs/14-rank1/)) | 0.729 | 0.918 | BM25 top-100 + cross-encoder | **+0.041 [+0.007, +0.076]** | 0.021 |
| HyDE + BGE-small (Qwen2.5-3B, N=8) ([15-hyde](pairs/15-hyde/)) | 0.742 | 0.972 | bge-small-en-v1.5 | **+0.032 [+0.010, +0.054]** | 0.005 |
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
| LLM2Vec (Qwen2.5-0.5B) ([02-llm2vec](pairs/02-llm2vec/)) | 0.188 | 0.223 | bge-small-en-v1.5 | **-0.152 [-0.178, -0.127]** | < 0.001 |
| LIMIT (GTE-ModernColBERT, MaxSim) ([04-limit](pairs/04-limit/)) | 0.375 | 0.322 | bge-small-en-v1.5 | **+0.035 [+0.017, +0.053]** | < 0.001 |
| gpl_BGE (eval_base) vs bge ([10_GPL](pairs/10-GPL/)) | 0.303 | 0.275 | bge-small-en-v1.5 | **-0.038 [-0.054, -0.021]** | < 0.001 |
| BGE-M3 (All) ([11-bge-m3](pairs/11-bge-m3/)) | 0.336 | 0.289 | RRF (BM25 + MiniLM) | -0.006 [-0.022, +0.010] | 0.470 |
| mat2vec word2vec + SIF (paper 12) ([12-mat2vec](pairs/12-mat2vec/)) | 0.219 | 0.208 | BM25 | **-0.102 [-0.125, -0.079]** | < 0.001 |
| RankGPT (RankZephyr-7B 4-bit; BM25 top-50, ArguAna top-20) ([13-rankgpt](pairs/13-rankgpt/)) | 0.361 | 0.246 | BM25 top-100 + cross-encoder | **+0.014 [+0.002, +0.026]** | 0.026 |
| Rank1-3B (reasoning), BM25 top-20 ([14-rank1](pairs/14-rank1/)) | 0.338 | 0.246 | BM25 top-100 + cross-encoder | -0.009 [-0.023, +0.005] | 0.220 |
| HyDE + BGE-small (Qwen2.5-3B, N=8) ([15-hyde](pairs/15-hyde/)) | 0.361 | 0.328 | bge-small-en-v1.5 | **+0.021 [+0.006, +0.036]** | 0.006 |
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
| LLM2Vec (Qwen2.5-0.5B) ([02-llm2vec](pairs/02-llm2vec/)) | 0.452 | 0.956 | bge-small-en-v1.5 | **-0.154 [-0.172, -0.134]** | < 0.001 |
| LIMIT (GTE-ModernColBERT, MaxSim) ([04-limit](pairs/04-limit/)) | 0.431 | 0.935 | bge-small-en-v1.5 | **-0.175 [-0.195, -0.155]** | < 0.001 |
| gpl_BGE (eval_base) vs bge ([10_GPL](pairs/10-GPL/)) | 0.360 | 0.900 | bge-small-en-v1.5 | **-0.246 [-0.266, -0.225]** | < 0.001 |
| BGE-M3 (All) ([11-bge-m3](pairs/11-bge-m3/)) | 0.527 | 0.980 | RRF (BM25 + MiniLM) | **-0.022 [-0.036, -0.008]** | 0.002 |
| mat2vec word2vec + SIF (paper 12) ([12-mat2vec](pairs/12-mat2vec/)) | 0.385 | 0.912 | BM25 | **-0.131 [-0.148, -0.112]** | < 0.001 |
| RankGPT (RankZephyr-7B 4-bit; BM25 top-50, ArguAna top-20) ([13-rankgpt](pairs/13-rankgpt/)) | 0.504 | 0.969 | BM25 top-100 + cross-encoder | **+0.082 [+0.062, +0.100]** | < 0.001 |
| Rank1-3B (reasoning), BM25 top-20 ([14-rank1](pairs/14-rank1/)) | 0.344 | 0.969 | BM25 top-100 + cross-encoder | **-0.078 [-0.099, -0.058]** | < 0.001 |
| HyDE + BGE-small (Qwen2.5-3B, N=8) ([15-hyde](pairs/15-hyde/)) | 0.477 | 0.969 | bge-small-en-v1.5 | **-0.129 [-0.145, -0.114]** | < 0.001 |
| static-retrieval-mrl-en-v1 (static, 2025) ([6-static-embeddings](pairs/6-static-embeddings/)) | 0.453 | 0.967 | all-MiniLM-L6-v2 | **-0.049 [-0.067, -0.032]** | < 0.001 |
