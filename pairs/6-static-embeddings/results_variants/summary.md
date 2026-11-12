Δ vs `minilm` (RRF row: vs `rrf`); bold = Holm-corrected p < 0.05 and CI excluding 0. Holm families: the official row (3 tests, pre-specified) and the exploratory rest (66 tests).

| System | Year | Family | SciFact | NFCorpus | ArguAna | mean | CPU ms/query |
|---|---|---|---|---|---|---|---|
| Qwen3-Embedding-0.6B | 2025 | LLM-based | 0.686 (+0.041, p<sub>Holm</sub>=0.22) | 0.356 (**+0.040**, p<sub>Holm</sub>=0.00061) | 0.675 (**+0.172**, p<sub>Holm</sub>=8.4e-78) | 0.573 | 508.4 |
| bge-small-en-v1.5 | 2023 | transformer | 0.710 | 0.340 | 0.606 | 0.552 | 12.8 |
| BM25 | 1994 | sparse lexical | 0.676 | 0.321 | 0.516 | 0.504 | 6.0 |
| RRF (BM25 + static) | 2025 | hybrid | 0.673 (**-0.045**, p<sub>Holm</sub>=0.0017) | 0.325 (-0.018, p<sub>Holm</sub>=0.11) | 0.503 (**-0.046**, p<sub>Holm</sub>=4.4e-12) | 0.500 | – |
| Contriever (MS MARCO) | 2021 | transformer | 0.672 (+0.027, p<sub>Holm</sub>=0.83) | 0.331 (+0.015, p<sub>Holm</sub>=0.82) | 0.494 (-0.009, p<sub>Holm</sub>=1) | 0.499 | 30.6 |
| all-MiniLM-L6-v2 (spring) | 2021 | transformer | 0.646 | 0.316 | 0.503 | 0.488 | 6.4 |
| all-mpnet-base-v2 | 2021 | transformer | 0.648 (+0.002, p<sub>Holm</sub>=1) | 0.333 (+0.017, p<sub>Holm</sub>=0.6) | 0.471 (**-0.031**, p<sub>Holm</sub>=0.02) | 0.484 | 34.2 |
| e5-small-v2 | 2023 | transformer | 0.680 (+0.035, p<sub>Holm</sub>=0.42) | 0.326 (+0.010, p<sub>Holm</sub>=1) | 0.442 (**-0.061**, p<sub>Holm</sub>=1.1e-09) | 0.483 | 12.3 |
| Model2Vec potion-retrieval-32M | 2025 | static (distilled + Tokenlearn) | 0.625 (-0.021, p<sub>Holm</sub>=1) | 0.312 (-0.004, p<sub>Holm</sub>=1) | 0.456 (**-0.047**, p<sub>Holm</sub>=2.7e-05) | 0.464 | 0.4 |
| static-retrieval-mrl-en-v1 (official row) | 2025 | static (contrastive) | 0.591 (**-0.054**, p<sub>Holm</sub>=0.0045) | 0.301 (-0.015, p<sub>Holm</sub>=0.12) | 0.453 (**-0.049**, p<sub>Holm</sub>=1.9e-07) | 0.449 | 1.4 |
| static-retrieval-mrl-en-v1 @ 512 dims | 2025 | static (contrastive) | 0.581 (**-0.064**, p<sub>Holm</sub>=0.019) | 0.298 (-0.018, p<sub>Holm</sub>=0.83) | 0.450 (**-0.053**, p<sub>Holm</sub>=1.5e-07) | 0.443 | – |
| static-retrieval-mrl-en-v1 @ 384 dims | 2025 | static (contrastive) | 0.580 (**-0.065**, p<sub>Holm</sub>=0.013) | 0.294 (-0.022, p<sub>Holm</sub>=0.39) | 0.446 (**-0.056**, p<sub>Holm</sub>=3.5e-08) | 0.440 | – |
| static-retrieval-mrl-en-v1 @ 256 dims | 2025 | static (contrastive) | 0.557 (**-0.088**, p<sub>Holm</sub>=0.00022) | 0.288 (**-0.028**, p<sub>Holm</sub>=0.038) | 0.435 (**-0.068**, p<sub>Holm</sub>=1.6e-11) | 0.427 | – |
| ours: trained static (MS MARCO, 10 s) | 2026 | static (this pair) | 0.580 (**-0.066**, p<sub>Holm</sub>=0.037) | 0.274 (**-0.042**, p<sub>Holm</sub>=0.00019) | 0.393 (**-0.110**, p<sub>Holm</sub>=2.5e-25) | 0.416 | – |
| static-retrieval-mrl-en-v1 @ 128 dims | 2025 | static (contrastive) | 0.535 (**-0.110**, p<sub>Holm</sub>=2.6e-06) | 0.260 (**-0.056**, p<sub>Holm</sub>=1.1e-08) | 0.411 (**-0.092**, p<sub>Holm</sub>=1.6e-20) | 0.402 | – |
| Model2Vec potion-base-8M | 2024 | static (distilled + Tokenlearn) | 0.523 (**-0.123**, p<sub>Holm</sub>=6.1e-07) | 0.247 (**-0.069**, p<sub>Holm</sub>=4.3e-10) | 0.436 (**-0.067**, p<sub>Holm</sub>=5.1e-10) | 0.402 | 0.4 |
| static-retrieval-mrl-en-v1 @ 64 dims | 2025 | static (contrastive) | 0.463 (**-0.183**, p<sub>Holm</sub>=3.9e-13) | 0.227 (**-0.089**, p<sub>Holm</sub>=3.5e-15) | 0.370 (**-0.133**, p<sub>Holm</sub>=1.3e-36) | 0.353 | – |
| Model2Vec M2V_base_output | 2024 | static (distilled) | 0.455 (**-0.191**, p<sub>Holm</sub>=1.8e-13) | 0.203 (**-0.113**, p<sub>Holm</sub>=3.4e-20) | 0.317 (**-0.186**, p<sub>Holm</sub>=3.3e-64) | 0.325 | 0.4 |
| ours: Model2Vec distil of bge-small | 2026 | static (this pair) | 0.422 (**-0.224**, p<sub>Holm</sub>=1.5e-18) | 0.188 (**-0.129**, p<sub>Holm</sub>=1.6e-20) | 0.323 (**-0.180**, p<sub>Holm</sub>=5.6e-60) | 0.311 | – |
| GloVe 6B-300d average | 2014 | static (word vectors) | 0.310 (**-0.336**, p<sub>Holm</sub>=6.2e-30) | 0.137 (**-0.179**, p<sub>Holm</sub>=5.5e-27) | 0.386 (**-0.117**, p<sub>Holm</sub>=3.3e-27) | 0.278 | 0.5 |
