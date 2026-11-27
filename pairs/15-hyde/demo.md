# Demo Queries (Pair 15: HyDE)

Run after `python pairs/15-hyde/method.py`, which writes `runs/`:

    python pairs/15-hyde/demo.py nfcorpus PLAIN-817
    python pairs/15-hyde/demo.py arguana test-health-ppelfhwbpba-pro02a
    python pairs/15-hyde/demo.py scifact 1

## 1. Hero Query (`nfcorpus` / `PLAIN-817`, nDCG@10 +1.000)
- **Selection**: the largest nDCG@10 gain of HyDE + BGE-small over BGE-small on NFCorpus.
- **Query**: `canker sores`
- **Why HyDE wins**: Lay medical query vs. technical abstracts. BGE-small alone confuses "sores" with "muscle soreness" (0 relevant in top 5). Qwen2.5-3B bridges the vocabulary gap by generating medical terms (*aphthous ulcers*, *oral mucosal lesions*), bringing the relevant paper (`MED-4841`) to rank #1.

## 2. Failure Query (`arguana` / `test-health-ppelfhwbpba-pro02a`, nDCG@10 -1.000)
- **Selection**: the largest nDCG@10 loss of HyDE + BGE-small against BGE-small on ArguAna.
- **Query**: Long argument on personhood at birth and partial-birth abortion.
- **Why HyDE fails**: BGE-small already ranks the true counter-argument #1. The 416-word average hypothetical response adds generic meta-phrasing (*"several counterarguments"*, *"Definition of Personhood"*), diluting the specific argument and pulling generic abortion documents ahead. Length explains most of it for this query: with the plain cache (116 words on average, no formatting) the counter-argument is at rank 2 instead of 19, still below BGE-small alone at rank 1. Over all ArguAna queries HyDE + BGE-small still loses 0.069 with the plain cache (0.129 with the first cache), as BGE-small is already a strong ranker.

## 3. Wildcard Query (Backup: `scifact` / `1`, nDCG@10 +1.000)
- Ready to run any query ID proposed live from `scifact`, `nfcorpus`, or `arguana` via `python pairs/15-hyde/demo.py <dataset> <qid>`.

## Live Page
- The talk uses `python pairs/15-hyde/demo_web.py` (open http://127.0.0.1:8765): the three queries above as buttons, a search over all test queries, an N slider from 0 to 8 and the two caches. The first start encodes the three corpora (about 9 minutes, kept in `runs/demo/`). With `--live` and `mlx-lm`, a typed SciFact claim gets 8 new hypothetical documents (sampled, seed 0). The page is an illustration and nothing on it is scored.
- If the page fails, the three `demo.py` commands above print the same rankings (first cache, top 5) in the terminal.
