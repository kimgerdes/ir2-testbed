# Static embeddings are back (Aarsen, HF blog 2025; Model2Vec 2024): pair 6, Martin Leiva · Vladimir Herrera Nativi
Method: `static-retrieval-mrl-en-v1`, a token lookup table + mean pooling (no attention), replaces the spring bi-encoder; compared with `minilm`.
```
pip install -r pairs/6-static-embeddings/requirements.txt
python pairs/6-static-embeddings/method.py && python testbed.py score pairs/6-static-embeddings/runs --method "static-retrieval-mrl-en-v1 (static, 2025)" --pair 6-static-embeddings --against minilm
python pairs/6-static-embeddings/train_static.py && python pairs/6-static-embeddings/variants.py    # era sweep, Model2Vec, Matryoshka, RRF -> results_variants/
python pairs/6-static-embeddings/latency.py && python pairs/6-static-embeddings/analysis.py        # CPU cost (idle machine), Holm, error analysis -> results_variants/, figures/
python pairs/6-static-embeddings/demo/prepare_demo.py && python pairs/6-static-embeddings/demo/demo_app.py   # demo, precomputed indexes
```
`reproduce.ipynb`: the blog's NanoBEIR table, CPU speed-up and training recipe on 1 of its 13 datasets (MS MARCO). `static_emb/`: our from-scratch implementation (tests and 3-seed study in the companion repo). AI assistance (Claude) declared in the report.
