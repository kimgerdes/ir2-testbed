"""Cross-encoder scoring (MS MARCO MiniLM, or a distilled student) and candidate re-ranking."""
import time

import torch

from .common import RUNS, RESULTS, jdump
from .bm25 import write_trec_run
from .evaluate import evaluate_run


class CrossEncoderScorer:
    def __init__(self, model_name_or_path, max_length=512, batch_size=64, device=None, fp16=True):
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        self.name = str(model_name_or_path)
        self.tok = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, num_labels=1)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device).eval()
        self.fp16 = fp16 and self.device == "cuda"
        self.max_length = max_length
        self.batch_size = batch_size

    @torch.no_grad()
    def score(self, query, texts):
        scores = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            enc = self.tok([query] * len(batch), batch, padding=True, truncation=True,
                           max_length=self.max_length, return_tensors="pt").to(self.device)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.fp16):
                logits = self.model(**enc).logits
            scores.extend(logits.view(-1).float().cpu().tolist())
        return scores


def rerank_items(scorer, items, hits):
    """items: candidates (list of {qid, query, hits:[{docid,text,...}]}). The top `hits` candidates are reordered by
    cross-encoder score; the remaining first-stage candidates keep their order below them.
    Returns run {qid: [(docid, score)]} and seconds per query."""
    run, t0 = {}, time.time()
    for it in items:
        top = it["hits"][:hits]
        rest = it["hits"][hits:]
        s = scorer.score(it["query"], [h["text"] for h in top]) if top else []
        order = sorted(range(len(top)), key=lambda i: -s[i])
        ranked = [(top[i]["docid"], float(len(top) + len(rest) - r)) for r, i in enumerate(order)]
        ranked += [(h["docid"], float(len(rest) - r)) for r, h in enumerate(rest)]
        run[it["qid"]] = ranked
    return run, (time.time() - t0) / max(1, len(items))


def evaluate_and_save(run, qrels, dataset, tag, hits, model_name, extra=None):
    ev = evaluate_run(run, qrels)
    write_trec_run(run, RUNS / dataset / f"{tag}_top{hits}.trec", tag=tag)
    out = {"dataset": dataset, "tag": tag, "model": model_name, "hits": hits, "n_queries": ev["n_queries"],
           "nDCG@1": ev["nDCG@1"], "nDCG@5": ev["nDCG@5"], "nDCG@10": ev["nDCG@10"],
           "per_query_ndcg10": ev["per_query"]["10"], "extra": extra or {}}
    jdump(out, RESULTS / f"{dataset}_{tag}_top{hits}.json")
    print(f"[{dataset}] {tag} top{hits}: nDCG@1/5/10 = {ev['nDCG@1']*100:.2f}/{ev['nDCG@5']*100:.2f}/{ev['nDCG@10']*100:.2f}  (n={ev['n_queries']})")
    return out
