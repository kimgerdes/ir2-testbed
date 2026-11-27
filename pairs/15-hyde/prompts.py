"""Prompts for the hypothetical documents, fixed before any test score was computed.
SciFact and ArguAna are copied from the paper's code (texttron/hyde, src/hyde/promptor.py).
The paper has no NFCorpus prompt: that one is ours, modelled on the paper's TREC-NEWS prompt."""

PROMPTS = {
    "scifact": """Please write a scientific paper passage to support/refute the claim.
Claim: {}
Passage:""",
    "arguana": """Please write a counter argument for the passage.
Passage: {}
Counter Argument:""",
    "nfcorpus": """Please write a medical paper passage about the topic.
Topic: {}
Passage:""",
}

# Second run, added after the first results (see reproduce.ipynb, "After the fact"): the same prompts,
# with this system message and at most 256 tokens, written to generations_plain/.
PLAIN = "Answer with the passage only: plain text, one paragraph, no heading, no list, no introduction."
