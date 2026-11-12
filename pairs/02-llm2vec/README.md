# LLM2Vec: Large Language Models Are Secretly Powerful Text Encoders

Pair: Baptiste and Raphael. Method: Qwen2.5-0.5B with bidirectional attention, then MNTP and unsupervised SimCSE with LoRA on the testbed documents (no queries, no labels), mean pooling. We compare with `bge`, the closest dense baseline. The ablation, the 1.5B model, the encoders (ModernBERT, Ettin) and our check of the paper's Table 11 are in `reproduce.ipynb`, the error analysis in `analysis.ipynb`, all scores in `results/`. One full recipe takes about 6 h on a Mac M5 with 24 GB (PyTorch MPS).

```
mamba env create -f environment.yml && mamba activate ir2-llm2vec
cd pairs/02-llm2vec && python prepare_corpus.py
python train.py --model Qwen/Qwen2.5-0.5B --step mntp --corpus data/train.txt --out outputs/q05/mntp
python train.py --model Qwen/Qwen2.5-0.5B --step simcse --init outputs/q05/mntp --corpus data/train.txt --out outputs/q05/mntp-simcse
python method.py --name full --mode bi --adapters outputs/q05/mntp outputs/q05/mntp-simcse
python ../../testbed.py score runs/full --method "LLM2Vec (Qwen2.5-0.5B)" --pair 02-llm2vec --against bge --out results.json
python demo.py --prepare && python demo.py
```

AI use: we used Claude Code as an assistant to write parts of the code, the notebooks and the slides. We ran and checked all results ourselves. Details in the report.
