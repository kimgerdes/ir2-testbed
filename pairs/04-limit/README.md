# On the Theoretical Limitations of Embedding-Based Retrieval, Weller et al., ICLR 2026
**Pair:** Zakaria Soualah Mohammed, Wissam Houria Amar<br>**Method:** GTE-ModernColBERT (many vectors per text, exact MaxSim) against bge-small (one 384d vector)<br>**Reproduction:** the free-embedding experiment (Table 6, then on real test qrels), BM25, bge-small, Qwen3-Embedding-0.6B/8B and ColBERT on LIMIT (Tables 4–5)
## How to Run
From the repository root, do the following:
```
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu126 && pip install -r pairs/04-limit/requirements.txt # install PyTorch for your GPU's CUDA version and requirements
python pairs/04-limit/method.py # run ColBERT on the testbed datasets
python testbed.py score pairs/04-limit/runs --method "LIMIT (GTE-ModernColBERT, MaxSim)" --pair 04-limit --against bge # testbed row
python pairs/04-limit/run_scripts.py # testbed on Qwen, analysis, notebook, etc.
```