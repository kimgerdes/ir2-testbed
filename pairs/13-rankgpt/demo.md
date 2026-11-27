# Demo queries (pair 13: RankGPT with RankZephyr-7B)

`python pairs/13-rankgpt/demo.py hero|fail|wild` prints, for one query, the BM25 top-5, the cross-encoder `ce` top-5 and
the RankZephyr top-5 side by side (`*` = labelled relevant), each system's nDCG@10, and the raw permutation RankZephyr
wrote for its top window of 20 passages. The three queries below replay cached windows (no GPU, under a second).
`--dataset scifact --interactive` takes queries typed by the room: BM25 over the corpus and `ce` on its top-100 (ready
in about 10 s, then 1 s per query), then one RankZephyr window on the BM25 top-20 (the 7B model loads once in about
40 s, then about 14 s per query on an RTX 4060 laptop; it needs the GPU, and no network once the models are cached). We start it before the talk, so the index and the model are ready before the demo.

- **Hero** (SciFact, query 1278): "The treatment of cancer patients with co-IR blockade does not cause any adverse
  autoimmune events." BM25 and `ce` both put *In vitro characterization of the anti-PD-1 antibody nivolumab* first and
  leave the labelled abstract, *Is autoimmunity the Achilles' heel of cancer immunotherapy?*, at rank 10 (nDCG@10 28.9
  for both). RankZephyr moves it to rank 1 (nDCG@10 100): its permutation starts `[10] > [1] > [9] > ...`. The abstract
  is the evidence against the claim ("immunotoxicity and autoimmunity are emerging as the Achilles' heel of
  immunotherapy") and never says "co-IR" or "blockade", so the match is not lexical (co-inhibitory-receptor blockade is
  checkpoint immunotherapy); RankZephyr also ranks the 20 passages against each other instead of scoring each alone.
- **Failure** (SciFact, query 569): "In adult tissue, most T cells are memory T cells." The labelled abstract, *Early
  life compartmentalization of human T cell differentiation and regulatory function in mucosal and lymphoid tissues*, is
  rank 1 for BM25 and `ce` (nDCG@10 100). RankZephyr drops it to rank 14 (nDCG@10 0) and fills the top with reviews on
  memory T cells. We trace the failure to the prompt: to fit 20 passages into RankZephyr's 4096-token
  context every passage is cut (here to 122 words), and the sentence that supports the claim, "Whereas adult tissues
  contain a predominance of memory T cells", starts at word 124 (title included). The model only saw an introduction
  about T cells in infants, while BM25 and `ce` read the 200-word text. Over all of SciFact the cut costs no
  measurable nDCG@10 (`results_overview.md`, section 3.3), but here it removed the supporting sentence.
- **Wildcard**: proposed on the spot by the discussants or the room, typed into
  `python pairs/13-rankgpt/demo.py --dataset scifact --interactive`. Backup (NFCorpus, query PLAIN-977): "crib death".
  BM25 and `ce` follow the word "death" (*Apoptosis induced by death receptors*, sudden cardiac death) and have no
  relevant document in their top 10 (nDCG@10 0 for both); the labelled *Relation of beta-casomorphin to apnea in sudden
  infant death syndrome* is at BM25 rank 15. RankZephyr puts it first (nDCG@10 46.9), although the passage never says
  "crib" (crib death is another name for SIDS).

Fallback recording (two minutes, the three queries above): https://drive.google.com/file/d/1_SHRiI6ghgVw7jIwnCq4ncJhER6KRtAo/view?usp=drive_link
