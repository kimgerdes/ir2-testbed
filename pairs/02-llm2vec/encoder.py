"""Load a model and turn texts into embeddings.

A model runs in one of 3 modes:
  uni     decoder with causal attention (the normal LLM)
  bi      decoder with bidirectional attention (step 1 of LLM2Vec)
  encoder a real encoder like ModernBERT, it is already bidirectional
"""
import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModel, AutoTokenizer

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
PAD_MULTIPLE = 64  # on MPS many different batch lengths fill the memory until it crashes


def bi_mask(attn):
    # every token can see every token, except padding
    B, L = attn.shape
    return attn[:, None, None, :].expand(B, 1, L, L).bool()


def causal_mask(attn):
    # a token can only see itself and the tokens before it
    L = attn.shape[1]
    tril = torch.tril(torch.ones(L, L, dtype=torch.bool, device=attn.device))
    return tril[None, None] & bi_mask(attn)


def make_mask(attn, mode):
    # We always give an explicit 4D mask for decoders. On MPS with bfloat16 the
    # default mask of transformers is not causal (see test_encoder.py).
    if mode == "encoder":
        return attn
    return bi_mask(attn) if mode == "bi" else causal_mask(attn)


def pool(hidden, pool_mask, how="mean"):
    # pool_mask is 1 on the tokens to use. Padding is on the right.
    hidden = hidden.float()
    if how == "mean":
        weights = pool_mask.float()
    elif how == "weighted_mean":
        # later tokens count more: 1, 2, 3, ...
        weights = pool_mask.float().cumsum(dim=1) * pool_mask.float()
    elif how == "last":
        last = pool_mask.sum(dim=1) - 1
        return hidden[torch.arange(hidden.shape[0]), last]
    else:
        raise ValueError(how)
    weights = weights / weights.sum(dim=1, keepdim=True)
    return (hidden * weights.unsqueeze(-1)).sum(dim=1)


def load_tokenizer(name):
    tok = AutoTokenizer.from_pretrained(name)
    # offline and without the files, transformers gives an empty tokenizer and no error
    assert len(tok) > 1000, f"tokenizer of {name} is empty, download it first"
    if tok.pad_token is None:  # Llama has no padding token, the mask hides it anyway
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    return tok


def load_model(name, adapters=(), attention_dropout=0.0):
    # sdpa is needed: with eager a boolean 4D mask does not mask anything
    kwargs = {"attention_dropout": attention_dropout} if attention_dropout else {}
    model = AutoModel.from_pretrained(name, dtype=torch.bfloat16, attn_implementation="sdpa", **kwargs)
    # each trained LoRA is merged in the model before the next one, like the paper
    for path in adapters:
        model = PeftModel.from_pretrained(model, path).merge_and_unload()
    model.config.use_cache = False
    return model.to(DEVICE)


def tokenize(tok, texts, instruction=None, max_length=256, pad_multiple=None, pool_special=True):
    # The instruction goes in the input but not in the pooling (like the paper).
    # The paper also does not pool on the BOS token of decoders (pool_special=False).
    # Qwen adds no special token, so it only changes something for Llama.
    inst = tok(instruction)["input_ids"] if instruction else []
    add_special = not instruction
    special = set(tok.all_special_ids)
    features = []
    pool_masks = []
    for text in texts:
        ids = tok(text, add_special_tokens=add_special, truncation=True, max_length=max_length - len(inst))["input_ids"]
        features.append({"input_ids": inst + ids})
        pool_masks.append([0] * len(inst) + [int(pool_special or i not in special) for i in ids])
    batch = tok.pad(features, return_tensors="pt", padding=True, pad_to_multiple_of=pad_multiple)
    L = batch["input_ids"].shape[1]
    batch["pool_mask"] = torch.tensor([m + [0] * (L - len(m)) for m in pool_masks])
    return batch


def forward(model, batch, mode, how="mean"):
    ids = batch["input_ids"].to(DEVICE)
    attn = batch["attention_mask"].to(DEVICE)
    hidden = model(input_ids=ids, attention_mask=make_mask(attn, mode)).last_hidden_state
    return pool(hidden, batch["pool_mask"].to(DEVICE), how)


@torch.no_grad()
def encode(model, tok, texts, mode, instruction=None, how="mean", batch_size=32, max_length=256):
    """Return normalized embeddings as a numpy array, in the same order as texts."""
    model.eval()
    # sort by length so a batch has less padding, it is faster
    order = np.argsort([len(t) for t in texts])
    out = [None] * len(texts)
    for start in range(0, len(texts), batch_size):
        idx = order[start:start + batch_size]
        batch = tokenize(tok, [texts[i] for i in idx], instruction, max_length, PAD_MULTIPLE, pool_special=mode == "encoder")
        emb = forward(model, batch, mode, how)
        emb = torch.nn.functional.normalize(emb, dim=1).cpu().numpy()
        for i, e in zip(idx, emb):
            out[i] = e
    return np.stack(out)
