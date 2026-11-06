# Demo queries (testbed SciFact / NFCorpus: BM25 | MiniLM, the spring bi-encoder | static-retrieval-mrl-en-v1)

- **Hero**: "Individuals with low serum vitamin D concentrations have increased risk of multiple sclerosis." (SciFact). The labelled abstract, *Past exposure to sun, skin phenotype, and risk of multiple sclerosis*, never names vitamin D in its title. It is rank 1 for the static model, 11 for BM25 and 16 for MiniLM: contrastive training put the vectors of *vitamin d* and *sun exposure* close together, which BM25 cannot do and MiniLM did not need to learn.
- **Failure**: "The locus rs647161 is associated with colorectal carcinoma." (SciFact). Rank 1 for BM25 and MiniLM, 8 for static. The tokenizer cuts `rs647161` into `rs ##64 ##7 ##16 ##1`: five generic digit pieces whose fixed vectors carry no identity once they are averaged with the rest of the claim. BM25 matches the exact identifier, and MiniLM's attention composes the pieces. Static models lose exactly where one rare exact token decides relevance.
- **Wildcard**: proposed on the spot by the discussants or the room. Backups: "Zoloft" (NFCorpus: static rank 1, MiniLM 20, BM25 > 100: a brand name whose tokens the static table links to depression), then the negation pair "Macrolides protect against myocardial infarction." / "Macrolides have no protective effect against myocardial infarction.": the static model returns the same abstract first for both, and SciFact counts it relevant for both.

Demo queries were chosen from the test queries after the runs, for presentation only: nothing was tuned on them.

Fallback recording: link (2 minutes, to be added)
