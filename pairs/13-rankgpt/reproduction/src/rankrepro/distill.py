"""Permutation distillation (RankNet) of a MiniLM cross-encoder from ChatGPT permutations or MS MARCO labels."""
import math
import random
import time

import torch
from torch.utils.data import Dataset, DataLoader

from .listwise import parse_permutation


def build_examples(records, perms, teacher="chatgpt", label_mode="binary", list_size=20):
    """Returns list of (query, [passage texts], [labels]) where larger label = should rank higher."""
    examples, stats = [], {"n": 0, "perm_missing_total": 0, "perm_malformed": 0, "pos_in_retrieved": 0}
    for rec, perm in zip(records, perms):
        retrieved = rec["retrieved_passages"][:list_size]
        if teacher == "chatgpt":
            order, st = parse_permutation(perm, len(retrieved))
            stats["perm_missing_total"] += st["missing"]
            stats["perm_malformed"] += 0 if st["well_formed"] else 1
            texts = [retrieved[i]["text"] for i in order]
            labels = [1.0 / (r + 1) for r in range(len(texts))]          # official code: y_true = 1/(i+1)
        elif teacher == "marco":
            pos_ids = {p["docid"] for p in rec["positive_passages"]}
            pos = rec["positive_passages"][0]["text"]
            negs = [p["text"] for p in retrieved if p["docid"] not in pos_ids][: list_size - 1]
            if any(p["docid"] in pos_ids for p in retrieved):
                stats["pos_in_retrieved"] += 1
            texts = [pos] + negs
            if label_mode == "binary":
                labels = [1.0] + [0.0] * len(negs)
            else:                                                        # 'bm25chain': official label=True branch
                labels = [1.0 / (r + 1) for r in range(len(texts))]
        else:
            raise ValueError(teacher)
        examples.append((rec["query"], texts, labels))
        stats["n"] += 1
    return examples, stats


class ListDataset(Dataset):
    def __init__(self, examples):
        self.ex = examples

    def __len__(self):
        return len(self.ex)

    def __getitem__(self, i):
        return self.ex[i]


def make_collate(tok, max_length):
    def collate(batch):
        qs, ps, ys, sizes = [], [], [], []
        for q, texts, labels in batch:
            qs += [q] * len(texts); ps += texts; ys += labels; sizes.append(len(texts))
        enc = tok(qs, ps, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        return enc, torch.tensor(ys, dtype=torch.float32), sizes
    return collate


def ranknet_loss(scores, labels, sizes):
    """Pairwise logistic loss over pairs (i, j) within a query with label_i > label_j (mean over pairs)."""
    losses, off = [], 0
    for n in sizes:
        s, y = scores[off:off + n], labels[off:off + n]
        off += n
        diff_y = y[:, None] - y[None, :]
        mask = diff_y > 0
        if mask.any():
            diff_s = (s[:, None] - s[None, :])[mask]
            losses.append(torch.nn.functional.binary_cross_entropy_with_logits(diff_s, torch.ones_like(diff_s), reduction="none"))
    if not losses:
        return scores.sum() * 0
    return torch.cat(losses).mean()


def train_student(student_name, examples, save_dir, epochs=2, lr=5e-5, batch_queries=8, grad_accum=4, max_length=256,
                  seed=42, fp16=True, log_every=20, device=None):
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    torch.manual_seed(seed); random.seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(student_name)
    model = AutoModelForSequenceClassification.from_pretrained(student_name, num_labels=1).to(device)
    loader = DataLoader(ListDataset(examples), batch_size=batch_queries, shuffle=True, collate_fn=make_collate(tok, max_length))
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    scaler = torch.amp.GradScaler("cuda", enabled=(fp16 and device == "cuda"))
    history, step, t0 = [], 0, time.time()
    model.train()
    for ep in range(epochs):
        run_loss, n_seen = 0.0, 0
        for it, (enc, y, sizes) in enumerate(loader):
            enc = enc.to(device); y = y.to(device)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(fp16 and device == "cuda")):
                logits = model(**enc).logits.view(-1).float()
            loss = ranknet_loss(logits, y, sizes) / grad_accum
            scaler.scale(loss).backward()
            run_loss += loss.item() * grad_accum; n_seen += 1
            if (it + 1) % grad_accum == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
                step += 1
                if step % log_every == 0:
                    history.append({"epoch": ep, "step": step, "loss": run_loss / n_seen, "seconds": time.time() - t0})
                    print(f"  epoch {ep} step {step} loss {run_loss / n_seen:.4f} ({time.time() - t0:.0f}s)", flush=True)
                    run_loss, n_seen = 0.0, 0
    save_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(save_dir); tok.save_pretrained(save_dir)
    return model, tok, history
