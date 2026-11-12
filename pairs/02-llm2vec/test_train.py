"""Checks on the training code, on very small runs. Run: python test_train.py"""
import json
import os
import tempfile

import numpy as np
import torch
from safetensors.torch import load_file

from encoder import DEVICE, encode, forward, load_model, load_tokenizer, tokenize
from train import (
    add_lora,
    build_model,
    contrastive_loss,
    enable_checkpointing,
    get_args,
    head_loss,
    mask_tokens,
    next_token_targets,
    train,
)

QWEN = "Qwen/Qwen2.5-0.5B"
BERT = "answerdotai/ModernBERT-base"
TEXTS = [
    "She loves to travel in summer, especially to cold destinations.",
    "The cat often sits by the window, dreaming of chasing birds.",
    "He reads books every night, finding solace in fiction.",
    "She paints landscapes on weekends with vibrant colors.",
    "Vitamin D deficiency is common in people who stay inside.",
    "The court decided that the contract was not valid.",
    "Regular exercise lowers the risk of heart disease in adults.",
    "The new bridge will connect the two sides of the river.",
]


def run(tmp, name, **kw):
    out = os.path.join(tmp, name)
    corpus = os.path.join(tmp, "corpus.txt")
    with open(corpus, "w") as f:
        f.write("\n".join(TEXTS))
    args = {"model": QWEN, "corpus": corpus, "out": out, "steps": 4, "batch_size": 4, "max_length": 24, "print_every": 1000}
    args.update(kw)
    argv = []
    for k, v in args.items():
        argv += [f"--{k}"] + ([str(x) for x in v] if isinstance(v, list) else [str(v)])
    train(get_args(argv))
    return out


def losses(out):
    with open(os.path.join(out, "log.jsonl")) as f:
        return [json.loads(line)["loss"] for line in f]


def test_mask_tokens():
    tok = load_tokenizer(QWEN)
    torch.manual_seed(0)
    ids = torch.randint(100, 5000, (64, 128))
    attn = torch.ones_like(ids)
    attn[:, 100:] = 0  # padding
    inputs, labels = mask_tokens(ids, attn, tok, prob=0.2)
    masked = labels != -100
    assert not masked[:, 0].any(), "first token can not be predicted"
    assert not masked[:, 100:].any(), "padding must not be masked"
    assert (labels[masked] == ids[masked]).all()
    assert (inputs[~masked] == ids[~masked]).all(), "only chosen tokens can change"
    share = masked[:, 1:100].float().mean().item()
    blank = (inputs[masked] == tok.convert_tokens_to_ids("_")).float().mean().item()
    same = (inputs[masked] == ids[masked]).float().mean().item()
    assert 0.17 < share < 0.23, share
    assert 0.75 < blank < 0.85, blank
    assert 0.06 < same < 0.14, same
    print(f"ok masking: {share:.2f} of tokens, {blank:.2f} blank, {same:.2f} unchanged")


def test_shift():
    # the value of hidden is its position, so we can see which position is used
    hidden = torch.arange(6.0)[None, :, None]
    labels = torch.tensor([[-100, -100, 11, -100, 44, -100]])
    h, targets = next_token_targets(hidden, labels)
    assert h.squeeze(-1).tolist() == [1.0, 3.0], "must use position i-1"
    assert targets.tolist() == [11, 44]
    print("ok masked token is predicted from the previous position")


def test_head_loss():
    # chunks must give the same loss as the normal cross entropy
    head = torch.nn.Linear(8, 50)
    for n in [1, 255, 256, 257, 700]:
        h = torch.randn(n, 8)
        targets = torch.randint(50, (n,))
        expected = torch.nn.functional.cross_entropy(head(h), targets)
        assert abs(head_loss(head, h, targets).item() - expected.item()) < 1e-4, n
    print("ok loss in chunks is the same as the normal loss")


def test_contrastive_loss():
    a = torch.nn.functional.normalize(torch.randn(16, 64), dim=1)
    assert contrastive_loss(a, a).item() < 0.5, "same views: low loss"
    b = torch.randn(16, 64)
    assert abs(contrastive_loss(a, b, scale=1.0).item() - np.log(16)) < 0.2, "random views: about log(batch)"
    print("ok contrastive loss")


def test_only_lora_is_trainable():
    for step in ["mntp", "simcse"]:
        args = get_args(["--model", QWEN, "--step", step, "--corpus", "x", "--out", "x"])
        model, lm_head = build_model(args, resume=False)
        names = [n for n, p in model.named_parameters() if p.requires_grad]
        assert names and all("lora_" in n for n in names)
        assert lm_head is None or not any(p.requires_grad for p in lm_head.parameters())
        del model, lm_head
    print("ok only LoRA weights are trainable")


def grads_of_simcse(use_checkpoint):
    tok = load_tokenizer(QWEN)
    batch = tokenize(tok, TEXTS[:4], max_length=24)
    torch.manual_seed(0)
    model = add_lora(load_model(QWEN, attention_dropout=0.3))
    # lora_B starts at 0: we move it so lora_A also gets gradients
    for n, p in model.named_parameters():
        if "lora_B" in n:
            torch.nn.init.normal_(p, std=1e-3)
    if use_checkpoint:
        enable_checkpointing(model)
    model.to(DEVICE).train()
    torch.manual_seed(1)
    contrastive_loss(forward(model, batch, "bi"), forward(model, batch, "bi")).backward()
    return torch.cat([p.grad.flatten().cpu().double() for p in model.parameters() if p.requires_grad])


def test_checkpoint_gradients():
    # with dropout, checkpointing must give the same gradients as no checkpointing
    a, b = grads_of_simcse(False), grads_of_simcse(True)
    cos = torch.nn.functional.cosine_similarity(a, b, dim=0).item()
    diff = (a - b).abs().max().item()
    assert cos > 0.999, f"wrong gradients with checkpointing, cosine {cos:.4f}"
    print(f"ok checkpointing keeps the dropout, gradient cosine {cos:.5f}, max diff {diff:.6f}")


def test_lora_trains(tmp):
    out = run(tmp, "mntp", step="mntp")
    weights = load_file(os.path.join(out, "adapter_model.safetensors"))
    assert len(weights) > 0 and all("lora_" in k for k in weights)
    assert all(k.startswith("base_model.model.layers.") for k in weights), list(weights)[0]
    assert all(v.dtype == torch.float32 for v in weights.values()), "LoRA should stay float32"
    assert any(v.abs().sum() > 0 for k, v in weights.items() if "lora_B" in k), "lora_B must have moved from 0"
    print(f"ok MNTP saves only LoRA weights, losses {losses(out)}")
    return out


def test_merge_changes_embeddings(mntp_out):
    tok = load_tokenizer(QWEN)
    base = encode(load_model(QWEN), tok, TEXTS[:2], "bi")
    merged = encode(load_model(QWEN, [mntp_out]), tok, TEXTS[:2], "bi")
    diff = np.abs(base - merged).max()
    assert diff > 0, "merged LoRA has no effect"
    print(f"ok merged LoRA changes the embeddings, max diff {diff:.5f}")


def test_learning(tmp):
    # a small corpus and a big lr: the loss must go down, if not the training is broken
    for name, kw in [
        ("learn-mntp", {"step": "mntp", "lr": 1e-3, "mask_prob": 0.5}),
        ("learn-simcse", {"step": "simcse", "lr": 1e-3}),
    ]:
        out = run(tmp, name, steps=30, batch_size=8, **kw)
        l = losses(out)
        first, last = np.mean(l[:5]), np.mean(l[-5:])
        assert last < first, f"{name}: loss {first:.3f} -> {last:.3f}"
        print(f"ok {name}: loss {first:.3f} -> {last:.3f}")


def test_resume(tmp):
    straight = run(tmp, "straight", step="simcse", steps=6)
    resumed = run(tmp, "resumed", step="simcse", steps=3)
    run(tmp, "resumed", step="simcse", steps=6)
    a = load_file(os.path.join(straight, "adapter_model.safetensors"))
    b = load_file(os.path.join(resumed, "adapter_model.safetensors"))
    diff = max((a[k] - b[k]).abs().max().item() for k in a)
    assert losses(resumed) == losses(straight), (losses(resumed), losses(straight))
    assert diff < 1e-5, diff
    print(f"ok a resumed run is the same as a normal run, max diff {diff}")


def test_dropout_views():
    # simcse needs 2 different views in train mode, for the decoder and for the encoder
    for name, mode in [(QWEN, "bi"), (BERT, "encoder")]:
        tok = load_tokenizer(name)
        model = load_model(name, attention_dropout=0.3).train()
        batch = tokenize(tok, TEXTS[:4], max_length=24)
        with torch.no_grad():
            a, b = forward(model, batch, mode), forward(model, batch, mode)
        cos = torch.nn.functional.cosine_similarity(a, b, dim=1).mean().item()
        assert cos < 0.9999, f"{name}: dropout does nothing"
        print(f"ok {name}: two views have cosine {cos:.4f}")
        del model


def test_encoder_mode(tmp):
    out = run(tmp, "bert", model=BERT, step="simcse", mode="encoder")
    tok = load_tokenizer(BERT)
    emb = encode(load_model(BERT, [out]), tok, TEXTS[:2], "encoder")
    assert emb.shape == (2, 768) and np.isfinite(emb).all()
    print(f"ok SimCSE on ModernBERT, losses {losses(out)}")


def test_chain_and_uni(mntp_out, tmp):
    chain = run(tmp, "chain", step="simcse", init=[mntp_out])
    uni = run(tmp, "uni", step="simcse", mode="uni")
    tok = load_tokenizer(QWEN)
    # a new LoRA starts at 0, so on top of MNTP it must not change the embeddings yet
    merged = load_model(QWEN, [mntp_out])
    before = encode(merged, tok, TEXTS[:2], "bi")
    after = encode(add_lora(merged).to(DEVICE), tok, TEXTS[:2], "bi")
    assert np.abs(before - after).max() == 0
    emb = encode(load_model(QWEN, [mntp_out, chain]), tok, TEXTS[:2], "bi")
    assert np.isfinite(emb).all() and np.abs(emb - before).max() > 0
    assert np.isfinite(encode(load_model(QWEN, [uni]), tok, TEXTS[:2], "uni")).all()
    print("ok MNTP then SimCSE, and SimCSE with causal attention")


if __name__ == "__main__":
    print("device", DEVICE)
    test_mask_tokens()
    test_shift()
    test_head_loss()
    test_contrastive_loss()
    test_only_lora_is_trainable()
    test_checkpoint_gradients()
    with tempfile.TemporaryDirectory() as tmp:
        mntp_out = test_lora_trains(tmp)
        test_merge_changes_embeddings(mntp_out)
        test_dropout_views()
        test_encoder_mode(tmp)
        test_chain_and_uni(mntp_out, tmp)
        test_learning(tmp)
        test_resume(tmp)
    print("all checks passed")
