"""Loaders for BEIR (scifact / nfcorpus / arguana), NovelEval and the MS MARCO 10K distillation set."""
import csv
from collections import defaultdict

from .common import RAW, PROC, normalize_ws, truncate_words, jload, read_jsonl


# ---------------------------------------------------------------- BEIR ----
def load_qrels_tsv(path):
    qrels = defaultdict(dict)
    with open(path, encoding="utf8") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)
        if header[0] != "query-id":          # file without header
            qrels[header[0]][header[1]] = int(header[2])
        for q, d, s in reader:
            qrels[q][d] = int(s)
    return dict(qrels)


def load_beir(name, split="test"):
    """Returns corpus {docid: {title, text}}, queries {qid: text} (only judged queries), qrels {qid: {docid: rel}}."""
    d = RAW / "beir" / name
    corpus = {}
    for j in read_jsonl(d / "corpus.jsonl"):
        corpus[j["_id"]] = {"title": j.get("title", "") or "", "text": j.get("text", "") or ""}
    qrels = load_qrels_tsv(d / "qrels" / f"{split}.tsv")
    queries = {}
    for j in read_jsonl(d / "queries.jsonl"):
        if j["_id"] in qrels:
            queries[j["_id"]] = normalize_ws(j["text"])
    return corpus, queries, qrels


def doc_to_text(doc, max_words=None):
    """'title text' with whitespace normalised and (optionally) cut to max_words words."""
    title = normalize_ws(doc.get("title", ""))
    text = normalize_ws(doc.get("text", ""))
    full = f"{title} {text}".strip() if title else text
    return truncate_words(full, max_words)


def load_candidates(name, hits=None):
    """Candidates produced by scripts/01_prepare_data.py: list of {qid, query, hits:[{docid, score, rank, text}]}."""
    path = PROC / name / "candidates.jsonl"
    items = list(read_jsonl(path))
    if hits is not None:
        for it in items:
            it["hits"] = it["hits"][:hits]
    return items


# ----------------------------------------------------------- NovelEval ----
def load_noveleval():
    """Returns queries {qid: text}, qrels {qid: {docid: rel}}, candidates {qid: [{docid, text, rank}]} in the released order."""
    d = RAW / "noveleval"
    data = jload(d / "data.json")
    queries, cands = {}, {}
    for item in data:
        qid = str(item["hits"][0]["qid"])
        queries[qid] = normalize_ws(item["query"])
        cands[qid] = [{"docid": h["docid"], "text": normalize_ws(h["content"]), "rank": h["rank"]} for h in item["hits"]]
    qrels = defaultdict(dict)
    with open(d / "qrels.txt", encoding="utf8") as f:
        for line in f:
            q, _, doc, rel = line.split()
            qrels[q][doc] = int(rel)
    return queries, dict(qrels), cands


# --------------------------------------------------------- MS MARCO 10K ----
def load_marco10k():
    """Returns (records, permutations). records[i] = {query, query_id, positive_passages, retrieved_passages(20)}; permutations[i] = ChatGPT string."""
    d = RAW / "marco10k"
    records = list(read_jsonl(d / "marco-train-10k.jsonl"))
    perms = jload(d / "marco-train-10k-gpt3.5.json")
    assert len(records) == len(perms), (len(records), len(perms))
    return records, perms
