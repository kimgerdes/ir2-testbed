"""Why does a model rank a document where it does? Run after prepare_demo.py.
The static model's vector is a mean of token vectors, so its cosine with a document splits exactly
into one term per query token (and per document token). MiniLM's does not; we print its top results instead.
Usage: python pairs/6-static-embeddings/demo/explain_query.py scifact "Individuals with low serum vitamin D ..."
"""
import json, sys
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
CACHE = str(Path(__file__).resolve().parent / "cache") + "/"
ds, query = sys.argv[1], sys.argv[2]
m = json.load(open(CACHE + f"{ds}_meta.json")); ids = m["ids"]; idx = {d: i for i, d in enumerate(ids)}
qid = [k for k, t in m["queries"].items() if t == query][0]; rel = {d for d, g in m["qrels"][qid].items() if g > 0}
S = SentenceTransformer("sentence-transformers/static-retrieval-mrl-en-v1", device="cpu")
L = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
emb, tok = S[0].embedding.weight.detach().float().numpy(), S[0].tokenizer
def toks(t): return tok.encode(t, add_special_tokens=False)
def vec(t):
    e = toks(t); v = emb[e.ids].mean(0); return v / np.linalg.norm(v), e
qv, qe = vec(query)
assert np.allclose(qv, S.encode([query], normalize_embeddings=True)[0], atol=1e-4)
Ds, Dm = np.load(CACHE + f"{ds}_static.npy"), np.load(CACHE + f"{ds}_minilm.npy")
lm = L.encode([query], normalize_embeddings=True)[0]
rs, rm = np.argsort(-(Ds @ qv)), np.argsort(-(Dm @ lm))
rank = lambda order, j: int(np.where(order == j)[0][0]) + 1
for d in rel:
    j = idx[d]; text = m["texts"][j]; dv, de = vec(text)
    print(f"\n=== relevant {d}: static rank {rank(rs, j)}, MiniLM rank {rank(rm, j)}; cos static {Ds[j]@qv:.3f}, MiniLM {Dm[j]@lm:.3f}")
    print("TEXT:", text[:400], "...")
    n_lm = len(L.tokenizer(text)["input_ids"]); print(f"MiniLM tokens: {n_lm} (max_seq_length {L.max_seq_length})")
    # query-token contributions: score = sum_t (e_t . dv) / (n * |mean|)
    qmean = emb[qe.ids].mean(0); c = (emb[qe.ids] @ dv) / (len(qe.ids) * np.linalg.norm(qmean))
    print("QUERY TOKEN CONTRIBUTIONS:", sorted(zip(qe.tokens, np.round(c, 3)), key=lambda x: -x[1]))
    dmean = emb[de.ids].mean(0); cd = (emb[de.ids] @ qv) / (len(de.ids) * np.linalg.norm(dmean))
    agg = {}
    for t, x in zip(de.tokens, cd): agg[t] = agg.get(t, 0) + x
    print("DOC TOKENS ADDING MOST:", [(t, round(float(x), 3)) for t, x in sorted(agg.items(), key=lambda x: -x[1])[:15]])
for name, order in [("STATIC", rs), ("MINILM", rm)]:
    print(f"\n{name} TOP 5:")
    for j in order[:5]: print(" ", "REL" if ids[j] in rel else "   ", m["texts"][j][:110])
