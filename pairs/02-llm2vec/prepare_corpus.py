"""Download the 3 testbed corpora from BEIR and write data/train.txt for train.py.

We train only on the documents, never on the queries or the qrels. In ArguAna the
queries are also documents of the corpus, so we remove them from the training text.
Run: python prepare_corpus.py
"""
import io
import json
import os
import urllib.request
import zipfile

URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip"
# number of documents from the course page, to be sure we have the same data
DATASETS = {"scifact": 5183, "nfcorpus": 3633, "arguana": 8674}


def read_docs(name):
    path = os.path.join("data", name, "corpus.jsonl")
    if not os.path.exists(path):
        print("download", name, flush=True)
        with urllib.request.urlopen(URL.format(name)) as r:
            zipfile.ZipFile(io.BytesIO(r.read())).extractall("data")
    with open(path) as f:
        docs = [json.loads(line) for line in f]
    assert len(docs) == DATASETS[name], f"{name}: {len(docs)} docs"
    return docs


def read_query_texts(name):
    with open(os.path.join("data", name, "queries.jsonl")) as f:
        return {" ".join(json.loads(line)["text"].split()) for line in f}


def main():
    texts = []
    for name in DATASETS:
        docs = read_docs(name)
        queries = read_query_texts(name)
        kept = [d for d in docs if " ".join(d["text"].split()) not in queries]
        print(f"{name}: {len(docs)} docs, {len(docs) - len(kept)} removed because they are queries")
        texts += [" ".join((d["title"] + " " + d["text"]).split()) for d in kept]
    # same text 2 times in a SimCSE batch would be a wrong negative
    unique = list(dict.fromkeys(t for t in texts if t))
    with open(os.path.join("data", "train.txt"), "w") as f:
        f.write("\n".join(unique) + "\n")
    print(f"{len(texts)} docs, {len(unique)} after removing duplicates and empty ones")


if __name__ == "__main__":
    main()
