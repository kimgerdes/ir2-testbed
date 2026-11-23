"""Rank1 (Weller et al., 2025) on the testbed: it reranks the BM25 top-20, positions 21-100 keep the BM25 order.
Writes one run per dataset to runs/, then:
python testbed.py score pairs/14-rank1/runs --method "Rank1-3B (reasoning), BM25 top-20" --pair 14-rank1 --against ce
The other runs of the analysis: --model jhu-clsp/rank1-0.5b or rank1-1.5b, --no_reasoning, --authors_prompt, --out."""
import sys, json, math, argparse
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
import testbed

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="jhu-clsp/rank1-3b")
ap.add_argument("--no_reasoning", action="store_true")
ap.add_argument("--authors_prompt", action="store_true")
ap.add_argument("--top", type=int, default=20)
ap.add_argument("--out", default=str(HERE / "runs"))
a = ap.parse_args()

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

INSTRUCTION = "Determine if the following passage is relevant to the query. Answer only with 'true' or 'false'.\n"
# authors' prompts (prompts.py of the official repository)
PROMPTS = {
    "scifact": "Claim: FILL_QUERY_HERE\n\nA relevant passage would provide evidence that either **supports** or **refutes** this claim. A passage with any information on any related subpart should be relevant.",
    "nfcorpus": "Topic: FILL_QUERY_HERE\n\nGiven the above topic, I need to learn about all aspects of it. It does not need to be directly relevant, only tangentially informational. Please mark as relevant any passages with even weak connections. I need to learn fast for my job, which means I need to understand each part individually.\n\nAgain remember, any connection means relevant even if indirect. So if it is not addressed, that is okay -- it does not need to be explicitly. \n\nFind me passages with any type of connection, including weak connections!!!!",
    "arguana": "I am looking to write an essay and need to find counterarguments against this statement:\n\nFILL_QUERY_HERE\n\nDoes this passage have any counterargument or evidence that could be used to help me?\n\n",
}
MAX_LEN, MAX_NEW, CHUNK = 4096, 2048, 2000

tok = AutoTokenizer.from_pretrained(a.model)
TRUE = tok(" true", add_special_tokens=False).input_ids[0]
FALSE = tok(" false", add_special_tokens=False).input_ids[0]
llm = LLM(model=a.model, max_model_len=MAX_LEN, gpu_memory_utilization=0.85, dtype="float16", seed=0)
reason = SamplingParams(temperature=0, max_tokens=MAX_NEW, logprobs=20, stop=["</think> true", "</think> false"],
                        skip_special_tokens=False)
answer = SamplingParams(temperature=0, max_tokens=1, logprobs=20, allowed_token_ids=[TRUE, FALSE], skip_special_tokens=False)


def prompt(query, passage, template):
    if template:
        query = template.replace("FILL_QUERY_HERE", query)
    p = INSTRUCTION + f"Query: {query}\nPassage: {passage}\n<think>"
    extra = len(tok(p).input_ids) - (MAX_LEN - 64)
    if extra > 0:                                    # cuts the passage only if the prompt would not fit
        ids = tok(passage, add_special_tokens=False).input_ids
        p = INSTRUCTION + f"Query: {query}\nPassage: {tok.decode(ids[:max(0, len(ids) - extra - 16)])}\n<think>"
    return p


def p_true(logprobs):
    t, f = math.exp(logprobs[TRUE].logprob), math.exp(logprobs[FALSE].logprob)
    return t / (t + f)


def forced(prompts, texts):
    # as _fix_incomplete_responses in rank1.py: cut at the last full sentence, close </think>, force true or false
    closed = []
    for p, t in zip(prompts, texts):
        t = t.rstrip()
        if not t.endswith((".", "!", "?")):
            cut = max(t.rfind("."), t.rfind("!"), t.rfind("?"))
            t = t[:cut + 1] if cut != -1 else t
        closed.append(f"{p}\n{t.strip()}\n</think>")
    outs = llm.generate(closed, answer, use_tqdm=False)
    return [p_true(o.outputs[0].logprobs[-1]) if o.outputs[0].logprobs else 0.5 for o in outs]


def score(prompts):
    # per pair: (p(true), generated tokens, reasoning)
    if a.no_reasoning:
        return [(s, 1, None) for s in forced(prompts, [""] * len(prompts))]
    res, todo = [], []
    for i, o in enumerate(llm.generate(prompts, reason, use_tqdm=False)):
        g = o.outputs[0]
        last = g.logprobs[-1] if g.logprobs else {}
        res.append((p_true(last) if TRUE in last and FALSE in last else None, len(g.token_ids), g.text))
        if res[-1][0] is None:                       # token cap hit or no answer: the answer is forced
            todo.append(i)
    for i, s in zip(todo, forced([prompts[i] for i in todo], [res[i][2] for i in todo])):
        res[i] = (s,) + res[i][1:]
    return res


out = Path(a.out)
for name in testbed.DATASETS:
    corpus, queries, _ = testbed.load(name)
    texts = dict(zip(corpus, testbed.doc_texts(corpus)))
    bm25 = testbed.baseline_run(name, "bm25")
    template = PROMPTS[name] if a.authors_prompt else None
    order = {}
    for q in queries:
        ranked = sorted(bm25[q], key=bm25[q].get, reverse=True)
        order[q] = [d for d in ranked if not (name == "arguana" and d == q)][:testbed.K]

    # one line per pair with the reasoning, written per chunk so that a stopped run resumes
    cache = out / "pairs" / f"{name}.jsonl"
    cache.parent.mkdir(parents=True, exist_ok=True)
    done = {}
    if cache.exists():
        for line in open(cache):
            r = json.loads(line)
            done[(r["qid"], r["did"])] = r["score"]
    todo = [(q, d) for q in queries for d in order[q][:a.top] if (q, d) not in done]
    for i in range(0, len(todo), CHUNK):
        chunk = todo[i:i + CHUNK]
        res = score([prompt(queries[q], texts[d], template) for q, d in chunk])
        with open(cache, "a") as f:
            for (q, d), (s, n, text) in zip(chunk, res):
                done[(q, d)] = s
                f.write(json.dumps({"qid": q, "did": d, "score": s, "tokens": n, "reasoning": text}, ensure_ascii=False) + "\n")
        print(f"{name}: {i + len(chunk)}/{len(todo)} pairs", flush=True)

    # run file in bm25 order, so that tied scores keep the bm25 order; positions 21-100 below the reranked ones
    run = {}
    for q in queries:
        run[q] = {d: done[(q, d)] for d in order[q][:a.top]}
        low = min(run[q].values())
        run[q].update({d: low - 1 - i for i, d in enumerate(order[q][a.top:])})
    testbed.save_run(run, out / f"{name}.json")
    print(f"{name}: wrote {out / f'{name}.json'}", flush=True)
