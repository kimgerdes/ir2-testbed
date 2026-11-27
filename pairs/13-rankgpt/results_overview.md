# Pair 13 (RankGPT): all results, laptop and Colab

Every experiment of sections 1 and 2 ran twice, with the same code, data, prompts, model weights and 4-bit
quantisation: on a laptop GPU (RTX 4060, 8 GB; the submitted runs, `results.json`) and in one Colab session on an A100
(24–25 November 2026; `results_colab.json`). The extensions of sections 3.3 and 3.4 ran on the A100 only. Statistics:
paired t-test and 10,000-sample bootstrap 95% confidence interval over all test queries. Result files are in
`reproduction/results/`, except `results.json` and `results_colab.json` in this folder.

## 1. The testbed row (Track A)

RankZephyr-7B in 4-bit re-ranks the testbed's BM25 top-50 (ArguAna: top-20, the query's own document removed) with
windows of 20 and step 10; `ce` re-ranks the BM25 top-100. nDCG@10 from the shared `testbed.py score --against ce`.

| dataset | queries | BM25 | `ce` | laptop | Colab | laptop − `ce` [95% CI], p | Colab − `ce` [95% CI], p |
|---|---|---|---|---|---|---|---|
| SciFact | 300 | 0.676 | 0.688 | 0.740 | 0.739 | +0.052 [+0.023, +0.081], 0.0006 | +0.051 [+0.023, +0.080], 0.0006 |
| NFCorpus | 323 | 0.321 | 0.347 | 0.361 | 0.361 | +0.014 [+0.002, +0.026], 0.026 | +0.013 [+0.002, +0.025], 0.023 |
| ArguAna | 1,406 | 0.516 | 0.422 | 0.504 | 0.504 | +0.082 [+0.062, +0.100], 7e-17 | +0.082 [+0.062, +0.101], 5e-17 |

- Recall@100 hardly moves (0.918, 0.246, 0.969; at most 0.0001 from `ce`): both systems re-order the BM25 top-100.
- The paper (nDCG@10 ×100, BM25 top-100, BM25 / gpt-3.5 / gpt-4): SciFact 67.89 / 70.43 / 74.95, NFCorpus
  30.75 / 35.62 / 38.47. On both machines our SciFact and NFCorpus results lie between gpt-3.5 and gpt-4.
- ArguAna: RankZephyr stays about 1 point below BM25 (laptop, shared scorer: −0.012 [−0.024, +0.000], p = 0.054). About
  half of its gain over `ce` comes from depth (section 3.2).
- Output quality, laptop and Colab alike: well-formed permutations in 99.9%, 100% and 99.6% of the windows; windows
  that left out at least one id: 1.7%, 1.0%, 0.7% on the laptop and 1.8%, 1.4%, 0.7% on Colab.
- Cost of the 3,898 windows: 45.2, 46.5 and 12.7 s per query on the laptop (about 13 hours), 20.5, 21.7 and 5.4 s on
  the A100 (about 6 hours).

## 2. Reproduction of the paper

### 2.1 NovelEval (Table 3)

21 questions collected in June 2023, after the training data of the paper's GPT models, 20 judged passages each; nDCG ×100. On the A100, all five
RankZephyr runs and the `ce` run are identical to the laptop's, query by query.

| system | input order | nDCG@1 | nDCG@5 | nDCG@10 |
|---|---|---|---|---|
| BM25 (paper) | | 33.33 | 45.96 | 55.77 |
| gpt-3.5-turbo (paper) | shuffled | 76.19 | 74.15 | 75.71 |
| gpt-4 (paper) | shuffled | 85.71 | 87.49 | 90.45 |
| monoT5-3B (paper) | | 83.33 | 78.38 | 84.62 |
| RankZephyr bf16 (RankZephyr paper) | released | 92.86 | 86.15 | 89.34 |
| `ce`, MiniLM-L6 (ours) | | 73.81 | 72.94 | 77.08 |
| RankZephyr 4-bit (ours) | released | 92.86 | 88.25 | 89.46 |
| RankZephyr 4-bit (ours) | BM25 | 83.33 | 85.22 | 88.89 |
| RankZephyr 4-bit (ours) | shuffled, seeds 0–2, mean ± sd | 87.30 ± 2.97 | 87.09 ± 1.74 | 89.17 ± 0.78 |

The open 7B model comes within 1.3 points of gpt-4 (89.17 against 90.45; 21 questions, so we read this as close, not
as equal), far above gpt-3.5 (75.71). RankZephyr's base model (Mistral-7B, September 2023) is newer than NovelEval, so
for RankZephyr we cannot rule out exposure to related text. With the released order, our 4-bit run matches the bf16
number of the RankZephyr paper (89.46 against 89.34).

### 2.2 Permutation distillation (Tables 7 and 13, small scale)

MiniLM-L6 students (22M parameters), RankNet loss, the paper's 10K MS MARCO queries, 2 epochs; they re-rank the BM25
top-50 (NovelEval: its 20 passages). nDCG@10 ×100, from the shared scorer on the three testbed sets and from our
evaluation on NovelEval. The students were trained a second time on the A100.

| model | machine | SciFact | NFCorpus | ArguAna | NovelEval |
|---|---|---|---|---|---|
| `ce` (MiniLM-L6 trained on all of MS MARCO) | both | 68.8 | 34.7 | 42.2 | 77.1 |
| student trained on ChatGPT's permutations | laptop | 64.5 | 33.6 | 27.9 | 72.5 |
| | Colab | 65.2 | 33.3 | 28.0 | 75.9 |
| student trained on the MS MARCO labels of the same queries | laptop | 21.6 | 22.4 | 2.6 | 42.1 |
| | Colab | 16.9 | 18.8 | 2.2 | 45.0 |

With the same model and the same 10K queries, ChatGPT's permutations give a usable re-ranker while our student trained
on the binary labels collapses. The paper's Table 13 shows the same contrast with a DeBERTa-large student (435M parameters): 74.18 / 38.48
from the permutations against 35.35 / 33.94 from the labels (SciFact / NFCorpus). Unlike that student, ours stays below
`ce`: −0.043 / −0.011 / −0.144 on the laptop and −0.036 / −0.014 / −0.142 on Colab (shared scorer, all p < 0.05).
Training again on another GPU moves the permutation student by less than one point on the testbed sets (3.4 points on
NovelEval) and the label student by up to 4.8 points; the gap between the two students holds on both machines.

### 2.3 Instruction types (Table 4), with Flan-T5-large

SciFact, first 100 test queries, BM25 top-50; on these queries BM25 = 71.65 and `ce` = 71.41. Reference: the Setwise
paper (Zhuang et al., SIGIR 2024), Table 3, Flan-T5-large on all SciFact queries, BM25 top-100 (BM25 = 67.9).
nDCG@10 ×100.

| method | laptop | Colab | Setwise paper | LLM calls per query |
|---|---|---|---|---|
| pointwise.yes_no (relevance generation) | 51.50 | 51.50 | 50.4 | 2 (batches of 32) |
| pointwise.qlm (query generation) | 70.17 | 70.19 | 64.4 | 2 (batches of 32) |
| listwise.generation (permutation generation, windows of 4) | 68.32 | 68.32 | 65.0 | 120 |
| listwise.likelihood | 66.79 | 66.98 | 63.9 | 120 |
| setwise.heapsort | 63.21 | 63.21 | 62.0 | 46 |
| setwise.bubblesort | 70.22 | 70.22 | 63.6 | 151 |
| pairwise.heapsort | 74.71 | 74.71 | 67.1 | 132 |
| pairwise.bubblesort | 75.90 | 75.90 | 70.0 | 339 |

The two best methods (pairwise) and the two worst (setwise.heapsort, yes/no relevance generation) are the same as in
the Setwise paper; in between, setwise.bubblesort ranks third here and sixth there (Spearman 0.83 over the eight
methods). Our values are 1–8 points higher, partly because the 100-query subset is easier (BM25 71.65 against 67.9) and
the depth differs (top-50 against top-100). Relevance generation falls 20 points below BM25 (17.5 in the Setwise paper). Only the pairwise methods score above BM25
and `ce`, and only pairwise.bubblesort significantly (+4.25 over BM25, p = 0.002; +4.44 over `ce`, p = 0.04), at 339
calls per query. With eight methods tested, only the comparison with BM25 survives a Bonferroni correction
(0.05 / 8). The two machines give the same order.

## 3. Analyses and extensions

### 3.1 How much of each passage RankZephyr reads

RankZephyr has a 4,096-token context, so each window's 20 passages are cut to a common word cap that makes the prompt
fit (the rank_llm procedure). Laptop runs (`truncation.json`):

| dataset | word cap per passage, median (range) | passages cut | words per candidate | words the model saw |
|---|---|---|---|---|
| SciFact | 102 (86–139) | 97.0% | 179 | 102 (57%) |
| NFCorpus | 112 (91–137) | 94.9% | 184 | 110 (60%) |
| ArguAna | 133 (54–195) | 67.4% | 159 | 124 (78%) |

### 3.2 ArguAna: depth against method (laptop)

`ce` loses more the deeper it re-ranks on ArguAna (nDCG@10 ×100: top-20 45.98, top-50 43.61, top-100 42.23), so the
comparison with the testbed's `ce` (top-100) mixes method and depth. At the same depth, RankZephyr − `ce` top-20 =
+4.42 [+2.70, +6.13] points, p = 5e-7 (`compare_arguana_depth.md`), about half of the +8.2 of section 1.

### 3.3 Window size and passage length, SciFact (A100 only)

Does the cut of section 3.1 cost accuracy? Three runs on the same BM25 top-50 (`compare_scifact_e2.md`), nDCG@10 ×100:

| run | window / step | passages | windows per query | prompt tokens | words the model saw | s per query | nDCG@10 |
|---|---|---|---|---|---|---|---|
| A (the submitted setting) | 20 / 10 | cut to fit the context | 4 | 3,965 | 57.1% | 20.5 | 73.92 |
| B | 10 / 5 | cut to fit the context | 9 | 3,482 | 99.9% | 24.2 | 74.37 |
| C | 10 / 5 | cut to 100 words | 9 | 2,039 | 55.7% | 21.9 | 74.95 |

B − A = +0.45 [−1.44, +2.37], p = 0.65; C − A = +1.03 [−0.20, +2.28], p = 0.10; C − B = +0.58 [−1.14, +2.32],
p = 0.51. None of the three differences is significant, even before a correction for multiple comparisons. C − B
isolates the cut (same windows) and shows no measurable cost, although its interval does not exclude a loss of about
one point; B − A also changes the window size. The cut can still remove the evidence of single queries, as in the
failure query of `demo.md`. B and C reuse 18 windows of their first two queries from a laptop test run.

### 3.4 Re-ranking depth, NFCorpus (A100 only)

| depth | windows per query | s per query | nDCG@10 | − `ce` [95% CI], p |
|---|---|---|---|---|
| top-50 (the submitted setting) | 4 | 21.7 | 36.06 | +1.40 [+0.25, +2.55], p = 0.019 |
| top-100 (the paper's setting) | 9 | 48.5 | 36.16 | +1.50 [+0.31, +2.74], p = 0.017 |

top-100 − top-50 = +0.10 [−0.60, +0.82] points, p = 0.78 (`compare_nfcorpus_e4.md`): 2.25 times the windows and the
time for no measurable gain over our top-50 on NFCorpus.

## 4. Cross-hardware validation

The code, data, prompts, model weights and quantisation were the same; the GPU, the operating system (Windows against
Linux) and with them the CUDA kernels differed. Full table: `replication_summary.md`,
also printed by section 4 of `reproduce.ipynb`.

- RankZephyr's 4-bit generation is not bit-identical across GPUs. Given the same 20 prompts (5 SciFact queries), 17
  answers were identical and 4 of the 5 queries got the same nDCG@10 (`generation_check_scifact_4bit_float16.json`).
  In a full run, a different answer changes the input of the next windows:

| run | queries | laptop | Colab | Colab − laptop [95% CI], p | queries with the same nDCG@10 | windows with the same answer |
|---|---|---|---|---|---|---|
| SciFact top-50 | 300 | 73.98 | 73.92 | −0.06 [−0.40, +0.28], p = 0.74 | 296 | 1,086 of 1,200 |
| NFCorpus top-50 | 323 | 36.09 | 36.06 | −0.03 [−0.33, +0.25], p = 0.83 | 312 | 1,194 of 1,292 |
| ArguAna top-20 | 1,406 | 50.40 | 50.42 | +0.03 [−0.05, +0.11], p = 0.50 | 1,398 | 1,341 of 1,406 |

- The Colab session repeated 32 laptop runs (sections 1 and 2, and the `ce` controls of section 3.2); 14 are identical
  query by query: the five NovelEval runs, six of the eight Flan-T5 methods, and `ce` on SciFact (top-50, top-100) and
  NovelEval. The other `ce` runs differ by at most
  0.05 points (p ≥ 0.13), the two other Flan-T5 methods by 0.02 and 0.19. The students, the only models trained again,
  are the only runs that move by more than a point (section 2.2).
- The change of machine alters single LLM answers, and the inference runs differ by at most 0.2 nDCG@10 points between the
  machines, and every comparison of sections 1 and 2 keeps its sign and its significance. The laptop runs stay the
  submitted row; `results_colab.json` is the same row from the A100.
