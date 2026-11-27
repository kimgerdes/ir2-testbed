"""Instructional permutation generation (RankGPT) with sliding windows, output parsing and a per-window cache."""
import hashlib
import json
import pathlib
import re

from .common import truncate_words

RANKGPT_SYSTEM = "You are RankGPT, an intelligent assistant that can rank passages based on their relevancy to the query."
RANKLLM_SYSTEM = "You are RankLLM, an intelligent assistant that can rank passages based on their relevancy to the query"


def clean_passage(text, max_words=None, replace_brackets=True):
    text = (text or "").replace("Title: Content: ", "").strip()
    if replace_brackets:                      # rank_llm convention: [n] inside passages -> (n)
        text = re.sub(r"\[(\d+)\]", r"(\1)", text)
    try:
        from ftfy import fix_text
        text = fix_text(text)
    except ImportError:
        pass
    text = " ".join(text.split())
    return truncate_words(text, max_words)


def build_messages(query, passages, style="zephyr"):
    """passages: list of cleaned strings. Returns a chat message list."""
    num = len(passages)
    if style == "rankgpt":                    # Appendix A.5 of the paper (multi-turn)
        msgs = [{"role": "system", "content": RANKGPT_SYSTEM},
                {"role": "user", "content": f"I will provide you with {num} passages, each indicated by number identifier []. \nRank the passages based on their relevance to query: {query}."},
                {"role": "assistant", "content": "Okay, please provide the passages."}]
        for i, p in enumerate(passages, 1):
            msgs.append({"role": "user", "content": f"[{i}] {p}"})
            msgs.append({"role": "assistant", "content": f"Received passage [{i}]."})
        msgs.append({"role": "user", "content": f"Search Query: {query}. \nRank the {num} passages above based on their relevance to the search query. The passages should be listed in descending order using identifiers. The most relevant passages should be listed first. The output format should be [] > [], e.g., [1] > [2]. Only response the ranking results, do not say any word or explain."})
        return msgs
    if style == "zephyr":                     # rank_llm/rank_zephyr_template.yaml (single turn)
        body = "".join(f"[{i}] {p}\n" for i, p in enumerate(passages, 1))
        user = (f"I will provide you with {num} passages, each indicated by a numerical identifier []. Rank the passages based on their relevance to the search query: {query}.\n"
                + body +
                f"Search Query: {query}.\nRank the {num} passages above based on their relevance to the search query. All the passages should be included and listed using identifiers, in descending order of relevance. The output format should be [] > [], e.g., [2] > [1], Answer concisely and directly and only respond with the ranking results, do not say any word or explain.")
        return [{"role": "system", "content": RANKLLM_SYSTEM}, {"role": "user", "content": user}]
    if style == "plain":                      # single-turn RankGPT wording for generic instruct models
        body = "".join(f"[{i}] {p}\n" for i, p in enumerate(passages, 1))
        user = (f"I will provide you with {num} passages, each indicated by number identifier []. Rank the passages based on their relevance to query: {query}.\n"
                + body +
                f"Search Query: {query}.\nRank the {num} passages above based on their relevance to the search query. The passages should be listed in descending order using identifiers. The most relevant passages should be listed first. The output format should be [] > [], e.g., [1] > [2]. Only response the ranking results, do not say any word or explain.")
        return [{"role": "system", "content": RANKGPT_SYSTEM}, {"role": "user", "content": user}]
    raise ValueError(style)


WELL_FORMED = re.compile(r"^\s*\[\d+\](\s*>\s*\[\d+\])*\s*$")


def parse_permutation(response, n):
    """Returns (perm, stats). perm is a list of 0-based indices covering all n passages: the ids the model produced
    (deduplicated, out-of-range dropped) followed by the missing ids in their original order (RankGPT, App. F fn. 9)."""
    ids = [int(x) for x in re.findall(r"\[(\d+)\]", response)]
    if not ids:                               # fall back to RankGPT's digit scan
        ids = [int(x) for x in re.sub(r"\D", " ", response).split()]
    seen, uniq, dup, oor = set(), [], 0, 0
    for i in ids:
        if i in seen:
            dup += 1
            continue
        seen.add(i)
        if 1 <= i <= n:
            uniq.append(i - 1)
        else:
            oor += 1
    present = set(uniq)
    missing = [i for i in range(n) if i not in present]
    stats = {"n_ids": len(ids), "duplicates": dup, "out_of_range": oor, "missing": len(missing),
             "well_formed": bool(WELL_FORMED.match(response.strip())), "empty": len(ids) == 0}
    return uniq + missing, stats


class WindowCache:
    """Append-only JSONL cache keyed by a hash of (model tag, prompt)."""

    def __init__(self, path):
        self.path = pathlib.Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.mem = {}
        if self.path.exists():
            with open(self.path, encoding="utf8") as f:
                for line in f:
                    try:
                        r = json.loads(line)
                        self.mem[r["key"]] = r
                    except json.JSONDecodeError:
                        pass

    @staticmethod
    def key(model_tag, prompt_text):
        return hashlib.sha1((model_tag + "\n" + prompt_text).encode("utf8")).hexdigest()

    def get(self, key):
        return self.mem.get(key)

    def put(self, record):
        self.mem[record["key"]] = record
        with open(self.path, "a", encoding="utf8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def max_new_tokens_for(llm, n):
    """Length of the full identifier list in model tokens, plus slack."""
    return llm.count_tokens(" > ".join(f"[{i + 1}]" for i in range(n))) + 16


def fit_to_context(llm, query, texts, style, context_limit, reserve):
    """Shorten every passage to the same word cap so that the rendered prompt has at most context_limit - reserve
    tokens (rank_llm's fit_prompt_to_context). Returns (texts, cap or None)."""
    budget = context_limit - reserve
    if llm.count_tokens(llm.render(build_messages(query, texts, style))) <= budget:
        return texts, None
    lo, hi = 8, max(len(t.split()) for t in texts)
    best = lo
    while lo <= hi:                                   # binary search for the largest cap that fits
        mid = (lo + hi) // 2
        cut = [truncate_words(t, mid) for t in texts]
        if llm.count_tokens(llm.render(build_messages(query, cut, style))) <= budget:
            best, lo = mid, mid + 1
        else:
            hi = mid - 1
    return [truncate_words(t, best) for t in texts], best


def rerank_window(llm, query, texts, style, cache=None, model_tag=None, max_new_tokens=None, context_limit=None):
    """One LLM call over texts (already cleaned). Returns (perm, record)."""
    max_new = max_new_tokens or max_new_tokens_for(llm, len(texts))
    cap = None
    if context_limit:
        texts, cap = fit_to_context(llm, query, texts, style, context_limit, max_new + 8)
    messages = build_messages(query, texts, style)
    prompt_text = llm.render(messages)
    key = WindowCache.key(model_tag or llm.name, prompt_text)
    rec = cache.get(key) if cache else None
    if rec is None:
        out = llm.generate(messages, max_new_tokens=max_new)
        rec = {"key": key, "model": model_tag or llm.name, "n": len(texts), "response": out["text"],
               "prompt_tokens": out["prompt_tokens"], "gen_tokens": out["gen_tokens"], "seconds": out["seconds"],
               "words_cap": cap, "cached": False}
        if cache:
            cache.put(rec)
    else:
        rec = dict(rec, cached=True)
    perm, stats = parse_permutation(rec["response"], len(texts))
    rec["perm"] = perm
    rec["stats"] = stats
    return perm, rec


def sliding_window_rerank(llm, query, hits, window=20, step=10, style="zephyr", passage_words=None, cache=None,
                          model_tag=None, rank_start=0, rank_end=None, context_limit=4096):
    """RankGPT sliding windows (back to front). hits: list of dicts with docid and text.
    context_limit: passages are shortened so that each prompt fits (RankZephyr was trained with a 4096-token context).
    Returns (reordered hits, list of per-window records)."""
    items = list(hits)
    rank_end = len(items) if rank_end is None else min(rank_end, len(items))
    logs = []
    if rank_end - rank_start < 2:                       # nothing to rank
        return items, logs
    end_pos, start_pos = rank_end, max(rank_end - window, rank_start)
    while True:
        segment = items[start_pos:end_pos]
        texts = [clean_passage(h["text"], passage_words) for h in segment]
        perm, rec = rerank_window(llm, query, texts, style, cache, model_tag, context_limit=context_limit)
        items[start_pos:end_pos] = [segment[i] for i in perm]
        rec = dict(rec, start=start_pos, end=end_pos, docids_in=[h["docid"] for h in segment])
        rec.pop("prompt", None)
        logs.append(rec)
        if start_pos == rank_start:
            break
        end_pos -= step
        start_pos -= step
        if start_pos < rank_start:                      # always finish with a window at the top of the list
            start_pos = rank_start
            end_pos = min(rank_start + window, rank_end)
    return items, logs


def summarize_logs(logs):
    n = len(logs)
    if n == 0:
        return {}
    s = {"windows": n,
         "windows_cached": sum(1 for r in logs if r.get("cached")),
         "well_formed_rate": sum(1 for r in logs if r["stats"]["well_formed"]) / n,
         "windows_with_missing": sum(1 for r in logs if r["stats"]["missing"] > 0) / n,
         "missing_ids_total": sum(r["stats"]["missing"] for r in logs),
         "duplicate_ids_total": sum(r["stats"]["duplicates"] for r in logs),
         "out_of_range_total": sum(r["stats"]["out_of_range"] for r in logs),
         "empty_responses": sum(1 for r in logs if r["stats"]["empty"]),
         "avg_prompt_tokens": sum(r["prompt_tokens"] for r in logs) / n,
         "avg_gen_tokens": sum(r["gen_tokens"] for r in logs) / n,
         "avg_seconds_uncached": None}
    unc = [r["seconds"] for r in logs if not r.get("cached")]
    if unc:
        s["avg_seconds_uncached"] = sum(unc) / len(unc)
    return s
