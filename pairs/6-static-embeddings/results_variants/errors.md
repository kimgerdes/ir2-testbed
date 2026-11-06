# Error analysis: static-retrieval-mrl-en-v1 vs MiniLM and BM25 (per-query nDCG@10)

## scifact

static beats MiniLM on 49 queries, loses on 98, ties on 153. Spearman(query length, Δ) = 0.04 (p = 0.52). Mean Δ with a negation word: -0.095 (n = 35) vs -0.049; with digits: -0.079 (n = 114) vs -0.039 (Mann-Whitney p: negation 0.41446961902038826, digits 0.5028290410896674). Per-query Pearson r: static~BM25 0.69, static~MiniLM 0.72, MiniLM~BM25 0.69.

### Biggest wins vs MiniLM

| Δ vs MiniLM | Δ vs BM25 | first relevant rank (static / MiniLM / BM25) | query |
|---|---|---|---|
| +1.00 | +1.00 | 1 / 16 / 11 | Individuals with low serum vitamin D concentrations have increased risk of multiple sclerosis. |
| +1.00 | +0.61 | 1 / 36 / 5 | In transgenic mice harboring green florescent protein under the control of the Sox2 promoter, less than ten percent of the cells with green florescent colocalize with cell proliferation markers. |
| +1.00 | +0.37 | 1 / 13 / 2 | Homozygous deletion of murine Sbds gene from osterix-expressing mesenchymal stem and progenitor cells (MPCs) prevents oxidative stress. |
| +1.00 | +0.00 | 1 / 24 / 1 | Macrolides protect against myocardial infarction. |
| +0.70 | +0.00 | 1 / 9 / 1 | Macrolides have no protective effect against myocardial infarction. |
| +0.68 | +0.37 | 1 / 8 / 2 | In mice, P. chabaudi parasites are able to proliferate faster early in infection when inoculated at lower numbers than when inoculated at high numbers. |
| +0.57 | +0.00 | 1 / 4 / 1 | In adult tissue, most T cells are memory T cells. |
| +0.57 | +0.00 | 1 / 4 / 1 | Rapid phosphotransfer rates govern fidelity in two component systems |

### Biggest losses vs MiniLM

| Δ vs MiniLM | Δ vs BM25 | first relevant rank (static / MiniLM / BM25) | query |
|---|---|---|---|
| -1.00 | -0.43 | 24 / 1 / 4 | Breast cancer development is determined exclusively by genetic factors. |
| -1.00 | -1.00 | 13 / 1 / 1 | Deletion of αvβ8 does not result in a spontaneous inflammatory phenotype. |
| -1.00 | -1.00 | 21 / 1 / 1 | Enhanced early production of inflammatory chemokines improves viral control in the lung. |
| -1.00 | +0.00 | 77 / 1 / 42 | Silencing of Bcl2 is important for the maintenance and progression of tumors. |
| -1.00 | -0.63 | 42 / 1 / 2 | Lymphadenopathy is observed in knockin mouse lacking the SHP-2 MAPK pathway. |
| -0.70 | -0.01 | 9 / 1 / 8 | The treatment of cancer patients with co-IR blockade precipitates adverse autoimmune events. |
| -0.68 | +0.03 | 8 / 1 / 10 | The treatment of cancer patients with co-IR blockade does not cause any adverse autoimmune events. |
| -0.68 | -0.68 | 8 / 1 / 1 | The locus rs647161 is associated with colorectal carcinoma. |

## nfcorpus

static beats MiniLM on 100 queries, loses on 130, ties on 93. Spearman(query length, Δ) = -0.21 (p = 0.00016). Mean Δ with a negation word: -0.057 (n = 2) vs -0.015; with digits: -0.067 (n = 6) vs -0.014 (Mann-Whitney p: negation 0.5109890552380116, digits 0.4111795160548146). Per-query Pearson r: static~BM25 0.86, static~MiniLM 0.84, MiniLM~BM25 0.81.

### Biggest wins vs MiniLM

| Δ vs MiniLM | Δ vs BM25 | first relevant rank (static / MiniLM / BM25) | query |
|---|---|---|---|
| +1.00 | +0.00 | 1 / >100 / 1 | kohlrabi |
| +0.81 | +0.34 | 1 / >100 / 1 | rickets |
| +0.49 | -0.05 | 1 / 7 / 1 | cadaverine |
| +0.46 | +0.46 | 1 / 20 / >100 | Zoloft |
| +0.44 | -0.27 | 2 / 9 / 1 | grapes |
| +0.42 | +0.42 | 2 / 53 / >100 | bagels |
| +0.41 | +0.11 | 1 / 13 / 1 | Fish Fog |
| +0.39 | +0.00 | 1 / 1 / 1 | Panama |

### Biggest losses vs MiniLM

| Δ vs MiniLM | Δ vs BM25 | first relevant rank (static / MiniLM / BM25) | query |
|---|---|---|---|
| -0.68 | -0.03 | 3 / 1 / 1 | Chernobyl |
| -0.52 | +0.07 | 2 / 1 / 8 | low-carb diets |
| -0.41 | +0.00 | 15 / 1 / 48 | prenatal vitamins |
| -0.41 | -0.15 | 15 / 1 / 3 | Heart Disease Starts in Childhood |
| -0.41 | +0.15 | 1 / 1 / 1 | cocaine |
| -0.39 | +0.00 | 59 / 5 / >100 | myelopathy |
| -0.38 | +0.04 | 1 / 1 / 1 | milk |
| -0.36 | -0.25 | 3 / 1 / 1 | Stool Size and Breast Cancer Risk |

## arguana

static beats MiniLM on 416 queries, loses on 598, ties on 392. Spearman(query length, Δ) = 0.01 (p = 0.62). Mean Δ with a negation word: -0.044 (n = 1134) vs -0.072; with digits: -0.046 (n = 1179) vs -0.066 (Mann-Whitney p: negation 0.26008423148591076, digits 0.7436798136140412). Per-query Pearson r: static~BM25 0.65, static~MiniLM 0.55, MiniLM~BM25 0.53.

