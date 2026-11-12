"""Checks on the attention masks and the pooling. Run: python test_encoder.py [model]"""
import sys

import numpy as np
import torch

from encoder import DEVICE, encode, load_model, load_tokenizer, make_mask, pool, tokenize

NAME = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-0.5B"
TEXT = "The cat often sits by the window, dreaming of chasing birds in the warm sunshine."


def hidden(model, ids, attn):
    with torch.no_grad():
        return model(input_ids=ids, attention_mask=attn).last_hidden_state.float()


def test_pooling():
    h = torch.tensor([[[1.0], [3.0], [100.0]]])
    m = torch.tensor([[1, 1, 0]])
    assert pool(h, m, "mean").item() == 2.0
    assert abs(pool(h, m, "weighted_mean").item() - (1 + 2 * 3) / 3) < 1e-6
    assert pool(h, m, "last").item() == 3.0
    print("ok pooling ignores padding")


def test_future_token(model, tok):
    # we change one token at position 10: position < 10 must not move in uni mode
    a = tok([TEXT], return_tensors="pt")["input_ids"].to(DEVICE)
    b = a.clone()
    b[0, 10] += 7
    attn = torch.ones_like(a)
    uni = (hidden(model, a, make_mask(attn, "uni")) - hidden(model, b, make_mask(attn, "uni")))[0, :10].abs().max()
    bi = (hidden(model, a, make_mask(attn, "bi")) - hidden(model, b, make_mask(attn, "bi")))[0, :10].abs().max()
    assert uni == 0, f"uni sees the future: {uni}"
    assert bi > 0, "bi does not see the future"
    print(f"ok future token: uni diff {uni:.1f}, bi diff {bi:.1f}")

    # not a check of our code, we keep it to show the problem in the report
    default = (hidden(model, a, attn) - hidden(model, b, attn))[0, :10].abs().max()
    print(f"info default transformers mask on {DEVICE}: diff {default:.1f} (0 means causal)")


def test_padding(model, tok):
    # same text alone or next to a longer text must give the same embedding
    for mode in ["uni", "bi"]:
        alone = encode(model, tok, [TEXT], mode)[0]
        batch = encode(model, tok, [TEXT, TEXT + " " + TEXT], mode)[0]
        assert np.dot(alone, batch) > 0.999, mode
    print("ok padding does not change the embedding")


def test_instruction(tok):
    inst = "Given a claim, find documents that refute the claim:"
    batch = tokenize(tok, [TEXT], inst)
    n_inst = len(tok(inst)["input_ids"])
    assert batch["pool_mask"][0, :n_inst].sum() == 0
    assert batch["pool_mask"][0, n_inst:].sum() == batch["attention_mask"][0, n_inst:].sum()
    assert len(tokenize(tok, [TEXT * 100], inst, max_length=64)["input_ids"][0]) == 64
    print("ok instruction is in the input but not in the pooling")


def test_special_tokens(tok):
    # decoders: no pooling on BOS (Llama adds one, Qwen adds nothing)
    ids = tok(TEXT)["input_ids"]
    n_special = sum(i in tok.all_special_ids for i in ids)
    assert tokenize(tok, [TEXT])["pool_mask"].sum() == len(ids)
    assert tokenize(tok, [TEXT], pool_special=False)["pool_mask"].sum() == len(ids) - n_special
    print(f"ok special tokens out of the pooling for decoders ({n_special} here)")


def test_order(model, tok):
    texts = ["short text", TEXT, "a text of medium size here"]
    together = encode(model, tok, texts, "bi", batch_size=2)
    for i, t in enumerate(texts):
        assert np.dot(together[i], encode(model, tok, [t], "bi")[0]) > 0.999
    print("ok encode keeps the order of the texts")


if __name__ == "__main__":
    tok = load_tokenizer(NAME)
    model = load_model(NAME).eval()
    test_pooling()
    test_future_token(model, tok)
    test_padding(model, tok)
    test_instruction(tok)
    test_special_tokens(tok)
    test_order(model, tok)
    print("all checks passed")
