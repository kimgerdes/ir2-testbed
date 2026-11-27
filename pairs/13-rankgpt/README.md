# RankGPT: Is ChatGPT Good at Search? LLMs as re-ranking agents (Sun et al., EMNLP 2023)

Pair: Zening and Said. Method: listwise re-ranking, with no training on the test sets. The LLM reads the query and 20 numbered passages and writes a permutation (`[2] > [3] > [1]`); windows of 20 with step 10 slide back to front over the BM25 top-50 (ArguAna: top-20, the query's own document removed), while `ce` re-ranks the top-100. GPT-4 is replaced by RankZephyr-7B (open weights, fine-tuned on RankGPT permutations) in 4-bit on an 8 GB laptop GPU; to fit 20 passages into its 4,096 tokens, each passage is cut to about 100 words. Every LLM window is cached in `cache/`, so the commands below need no GPU. `reproduce.ipynb` replays Table 3 (NovelEval) from the cached windows and shows our small-scale Table 7 (permutation distillation, from the saved scores of our two students; it does not retrain them), using `reproduction/`, a copy of our research code and data.

```
pip install -r requirements.txt -r pairs/13-rankgpt/requirements.txt    # from the testbed root; CUDA only for new windows
python pairs/13-rankgpt/method.py                                       # runs/<dataset>.json from the cached windows (a few minutes)
python testbed.py score pairs/13-rankgpt/runs --method "RankGPT (RankZephyr-7B 4-bit; BM25 top-50, ArguAna top-20)" --pair 13-rankgpt --against ce
jupyter nbconvert --to notebook --execute --inplace pairs/13-rankgpt/reproduce.ipynb   # Tables 3 and 7, Colab check; no GPU
python pairs/13-rankgpt/demo.py hero                                    # the queries of demo.md; --interactive for live ones (GPU)
```

- Results (`results.json`, nDCG@10, paired test over all test queries): SciFact 0.740 vs `ce` 0.688, +0.052 [+0.023, +0.081], p = 0.0006; NFCorpus 0.361 vs 0.347, +0.014 [+0.002, +0.026], p = 0.026; ArguAna 0.504 vs 0.422, +0.082 [+0.062, +0.100], p = 7e-17. The paper's gpt-3.5 / gpt-4 reach 0.704 / 0.750 on SciFact and 0.356 / 0.385 on NFCorpus (BM25 top-100; gpt-4 re-ranks gpt-3.5's top-30).
- ArguAna: RankZephyr stays about 1 point below BM25 (0.516; −0.012 [−0.024, +0.000], p = 0.054). `ce` loses more the deeper it re-ranks there, so about half of the +0.082 is depth: against `ce` on the same top-20 the gain is +0.044 [+0.027, +0.061], p = 5e-7.
- Cross-hardware check: we ran the whole reproduction a second time on a Colab A100, with the same code, data, prompts, models and 4-bit quantisation. The Colab row (`results_colab.json`, same scorer) is 0.739 / 0.361 / 0.504, within 0.001 of the laptop run, with nearly the same gains over `ce`. Section 4 of `reproduce.ipynb` compares all 32 runs; `results_overview.md` gathers every result of both machines.
- AI use: we used Claude Code and Codex to help build the code and run automated tests, had several agents cross-check the results, and used agents to help write the documentation and render the slides; details in the report.
