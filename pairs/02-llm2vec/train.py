"""Train one step of LLM2Vec with LoRA: MNTP or SimCSE.

MNTP on a decoder (always bidirectional):
  python train.py --model Qwen/Qwen2.5-0.5B --step mntp --corpus data/train.txt --out outputs/q05/mntp

SimCSE after MNTP, then SimCSE without MNTP, or without bidirectional attention:
  python train.py --model Qwen/Qwen2.5-0.5B --step simcse --init outputs/q05/mntp --corpus data/train.txt --out outputs/q05/mntp-simcse
  python train.py --model Qwen/Qwen2.5-0.5B --step simcse --corpus data/train.txt --out outputs/q05/bi-simcse
  python train.py --model Qwen/Qwen2.5-0.5B --step simcse --mode uni --corpus data/train.txt --out outputs/q05/uni-simcse

SimCSE on an encoder:
  python train.py --model answerdotai/ModernBERT-base --step simcse --mode encoder --corpus data/train.txt --out outputs/modernbert/simcse

The corpus is a text file with one document per line. If the run stops, the same
command continues from the last save.
"""
import argparse
import gc
import json
import os
import random
import time

import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model
from torch.utils.checkpoint import checkpoint
from transformers import AutoModelForCausalLM

from encoder import DEVICE, bi_mask, forward, load_model, load_tokenizer, tokenize

LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
DECODER_TARGETS = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
LORA_TARGETS = {"qwen2": DECODER_TARGETS, "modernbert": ["Wqkv", "Wo", "Wi"]}
DEFAULT_LR = {"mntp": 5e-5, "simcse": 3e-5}
HEAD_CHUNK = 256


def mask_tokens(ids, attn, tok, prob=0.2):
    """BERT masking: of the chosen tokens 80% become "_", 10% random, 10% stay the same."""
    can_mask = attn.bool() & ~torch.isin(ids, torch.tensor(tok.all_special_ids))
    can_mask[:, 0] = False  # there is no previous token to predict the first one
    masked = (torch.rand(ids.shape) < prob) & can_mask
    labels = ids.clone()
    labels[~masked] = -100

    inputs = ids.clone()
    r = torch.rand(ids.shape)
    inputs[masked & (r < 0.8)] = tok.convert_tokens_to_ids("_")
    use_random = masked & (r >= 0.8) & (r < 0.9)
    inputs[use_random] = torch.randint(len(tok), ids.shape)[use_random]
    return inputs, labels


def next_token_targets(hidden, labels):
    # main idea of MNTP: the masked token at position i is predicted from position i-1
    keep = labels[:, 1:] != -100
    return hidden[:, :-1][keep], labels[:, 1:][keep]


def mntp_loss(model, lm_head, tok, batch, prob):
    inputs, labels = mask_tokens(batch["input_ids"], batch["attention_mask"], tok, prob)
    attn = batch["attention_mask"].to(DEVICE)
    hidden = model(input_ids=inputs.to(DEVICE), attention_mask=bi_mask(attn)).last_hidden_state
    # lm_head only on the masked tokens, on all tokens it takes a lot of memory
    h, targets = next_token_targets(hidden, labels.to(DEVICE))
    return head_loss(lm_head, h, targets)


def head_loss(lm_head, h, targets):
    # The number of masked tokens changes at each step. With a new shape each time, MPS
    # memory grows until the crash, so lm_head always gets chunks of the same size.
    total = 0.0
    for start in range(0, len(h), HEAD_CHUNK):
        part = h[start:start + HEAD_CHUNK]
        missing = HEAD_CHUNK - len(part)
        part = F.pad(part, (0, 0, 0, missing))
        part_targets = F.pad(targets[start:start + HEAD_CHUNK], (0, missing), value=-100)
        total = total + F.cross_entropy(lm_head(part).float(), part_targets, ignore_index=-100, reduction="sum")
    return total / len(targets)


def contrastive_loss(a, b, scale=20.0):
    # a[i] and b[i] are the same text, the other texts of the batch are negatives
    a = F.normalize(a, dim=1)
    b = F.normalize(b, dim=1)
    return F.cross_entropy(a @ b.T * scale, torch.arange(a.shape[0], device=a.device))


def simcse_loss(model, batch, mode, scale):
    # 2 passes on the same batch: only the dropout makes them different
    return contrastive_loss(forward(model, batch, mode), forward(model, batch, mode), scale)


def add_lora(base, dropout=LORA_DROPOUT):
    config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=dropout,
        target_modules=LORA_TARGETS[base.config.model_type],
        bias="none",
    )
    return get_peft_model(base, config)


def checkpoint_with_mps_rng(function, *args, **kwargs):
    # torch checkpoint saves the random state of cuda but not of mps. Without this
    # the dropout is different in the backward pass and the gradients are wrong
    # (see test_checkpoint_gradients in test_train.py)
    state = torch.mps.get_rng_state() if DEVICE == "mps" else None

    def run(*a, **k):
        if state is not None:
            torch.mps.set_rng_state(state)
        return function(*a, **k)

    return checkpoint(run, *args, use_reentrant=False, **kwargs)


def enable_checkpointing(model):
    # paper uses it too. It is slower but we do not have enough memory without it
    model.get_base_model()._set_gradient_checkpointing(True, checkpoint_with_mps_rng)


def build_model(args, resume):
    """Return the model with its LoRA, and the lm_head for MNTP (None for SimCSE)."""
    lm_head = None
    if args.step == "mntp":
        full = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, attn_implementation="sdpa")
        full.config.use_cache = False
        base = full.model
        lm_head = full.lm_head.requires_grad_(False).to(DEVICE)
    else:
        base = load_model(args.model, args.init, attention_dropout=args.dropout)

    model = PeftModel.from_pretrained(base, args.out, is_trainable=True) if resume else add_lora(base)
    enable_checkpointing(model)
    return model.to(DEVICE).train(), lm_head


def free_memory():
    # gives the memory kept by Metal back to the Mac, the pool stays small during long runs
    gc.collect()
    if DEVICE == "mps":
        torch.mps.empty_cache()


def memory_gb():
    return round(torch.mps.driver_allocated_memory() / 1e9, 1) if DEVICE == "mps" else None


def clean_log(path, start):
    # the steps after the last save will be done again, so we remove them from the log
    rows = []
    if os.path.exists(path):
        with open(path) as f:
            rows = [line for line in f if json.loads(line)["step"] <= start]
    with open(path, "w") as f:
        f.writelines(rows)


def save(model, opt, step, out):
    model.save_pretrained(out)
    torch.save({"step": step, "optimizer": opt.state_dict()}, os.path.join(out, "state.pt"))


def train(args):
    os.makedirs(args.out, exist_ok=True)
    with open(args.corpus) as f:
        corpus = [line.strip() for line in f if line.strip()]
    state_path = os.path.join(args.out, "state.pt")
    resume = os.path.exists(state_path)
    with open(os.path.join(args.out, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    tok = load_tokenizer(args.model)
    torch.manual_seed(args.seed)  # the first LoRA weights are random
    model, lm_head = build_model(args, resume)
    params = [p for p in model.parameters() if p.requires_grad]
    # the HF Trainer of the authors decays the lr very little in 1000 steps, we keep it constant
    opt = torch.optim.AdamW(params, lr=args.lr or DEFAULT_LR[args.step], weight_decay=0.0)
    start = 0
    if resume:
        state = torch.load(state_path, map_location=DEVICE)
        opt.load_state_dict(state["optimizer"])
        start = state["step"]
        print(f"resume from step {start}", flush=True)

    log_path = os.path.join(args.out, "log.jsonl")
    clean_log(log_path, start)
    with open(log_path, "a") as log:
        for step in range(start, args.steps):
            # seed depends on the step, so a resumed run does the same as a normal run
            rng = random.Random(args.seed + step)
            torch.manual_seed(args.seed + step)
            t0 = time.time()
            batch = tokenize(tok, rng.sample(corpus, args.batch_size), max_length=args.max_length)
            if args.step == "mntp":
                loss = mntp_loss(model, lm_head, tok, batch, args.mask_prob)
            else:
                loss = simcse_loss(model, batch, args.mode, args.scale)
            assert torch.isfinite(loss), f"loss is {loss} at step {step}"
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            opt.zero_grad()
            loss_value = round(loss.item(), 4)
            del loss
            free_memory()

            row = {"step": step + 1, "loss": loss_value, "seconds": round(time.time() - t0, 2), "memory_gb": memory_gb()}
            log.write(json.dumps(row) + "\n")
            log.flush()
            if (step + 1) % args.print_every == 0:
                print(row, flush=True)
            if (step + 1) % args.save_every == 0:
                save(model, opt, step + 1, args.out)
    save(model, opt, args.steps, args.out)


def get_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--step", required=True, choices=["mntp", "simcse"])
    p.add_argument("--corpus", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--init", nargs="*", default=[], help="LoRA folders to merge before training")
    p.add_argument("--mode", default="bi", choices=["uni", "bi", "encoder"], help="only for simcse")
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_length", type=int, default=128)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--mask_prob", type=float, default=0.2)
    p.add_argument("--dropout", type=float, default=0.3, help="simcse attention dropout")
    p.add_argument("--scale", type=float, default=20.0, help="simcse loss scale")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save_every", type=int, default=100)
    p.add_argument("--print_every", type=int, default=10)
    return p.parse_args(argv)


if __name__ == "__main__":
    train(get_args())
