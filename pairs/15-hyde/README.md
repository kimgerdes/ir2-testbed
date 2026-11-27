# HyDE: Precise Zero-Shot Dense Retrieval without Relevance Labels (Gao et al., ACL 2023)

Pair 15: Xuanru Dai and Juan Kenichi Sutan, session of Friday 27 November. An LLM (Qwen2.5-3B-Instruct, 8-bit, MLX) writes hypothetical documents per query; the retriever searches with the mean of their embeddings and the query embedding. The generations are committed in `generations/` and `generations_plain/` (14 MB) on purpose: every row depends on them and regenerating takes about 6 hours on Apple silicon (`generate.py`, `generate.py plain`). `reproduce.ipynb` follows the paper (full documents, dot product); `method.py`, `run_variants.py` (writes `results_variants/`), `demo.py` and `demo_web.py` (see `demo.md`) follow the testbed (200 words, cosine), so the two Contriever scores differ; the notebook section "Same encoder, two protocols" shows all four combinations. `results.json` was written on Xuanru's machine; a re-run on Kenichi's is deterministic and differs by at most 0.001 nDCG@10 (ArguAna 0.4760 against 0.4766).

AI assistance: Claude (Anthropic) helped write the code and run the experiments. We chose the experiments, checked the numbers and wrote the conclusions. Also declared in the report.

```
pip install -r pairs/15-hyde/requirements.txt && jupyter nbconvert --to notebook --execute --inplace pairs/15-hyde/reproduce.ipynb
python pairs/15-hyde/method.py && python testbed.py score pairs/15-hyde/runs --method "HyDE + BGE-small (Qwen2.5-3B, N=8)" --pair 15-hyde --against bge
```
