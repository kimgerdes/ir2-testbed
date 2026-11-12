"""Training a static embedding model from random init (blog §Training Details).

A plain PyTorch loop that reproduces what the blog's SentenceTransformerTrainer
setup does, made explicit:

  model     : StaticEmbedder(bert-base-uncased tokenizer, d=1024), E ~ N(0, 1)
  data      : (query, positive, hard negative) triplets, no-duplicates batches
  loss      : MatryoshkaLoss(MNRL), dims (1024, 512, 256, 128, 64, 32)
  optimiser : AdamW, lr 0.2, linear warmup (10 %) then linear decay, 1 epoch, bf16
"""
from __future__ import annotations

import json
import math
import time
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass, field

import numpy as np
import torch
from tokenizers import Tokenizer

from static_emb.losses import matryoshka_loss, mnrl_loss
from static_emb.static_model import StaticEmbedder


@dataclass
class TrainConfig:
    tokenizer_name: str = "google-bert/bert-base-uncased"  # blog §English Retrieval
    embedding_dim: int = 1024  # blog §Model Inspiration
    dataset: str = "sentence-transformers/msmarco-co-condenser-margin-mse-sym-mnrl-mean-v1"  # blog: "msmarco", triplet subset
    dataset_subset: str = "triplet"
    eval_holdout: int = 10_000  # blog: train_test_split(test_size=10_000, seed=12)
    max_train_samples: int | None = None  # reduced scale knob (Deviations #1)
    epochs: int = 1  # blog §Training Arguments
    batch_size: int = 2048  # blog §Training Arguments
    lr: float = 0.2  # blog: "much larger than with normal embedding model training"
    warmup_ratio: float = 0.1  # blog
    weight_decay: float = 0.0  # HF Trainer default (blog does not set it)
    max_grad_norm: float = 1.0  # HF Trainer default (blog does not set it)
    bf16: bool = True  # blog
    scale: float = 20.0  # MNRL default scale (temperature 0.05)
    matryoshka_dims: tuple[int, ...] | None = (1024, 512, 256, 128, 64, 32)  # blog final script; None = MNRL only (control)
    seed: int = 12  # blog: random.seed(12)
    log_every: int = 10
    eval_every: int = 0  # steps between NanoBEIR evaluations (0 = only at the end)
    extra: dict = field(default_factory=dict)


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
class PackedColumn:
    """A tokenised text column stored CSR-style: one flat id array + row offsets."""

    def __init__(self, ids: list[list[int]]):
        lengths = np.fromiter((len(x) for x in ids), dtype=np.int64, count=len(ids))
        self.starts = np.zeros(len(ids) + 1, dtype=np.int64)
        np.cumsum(lengths, out=self.starts[1:])
        self.flat = np.fromiter((t for x in ids for t in x), dtype=np.int64, count=int(lengths.sum()))

    def gather(self, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Rows -> (flat ids, offsets) ready for EmbeddingBag."""
        s, e = self.starts[rows], self.starts[rows + 1]
        lengths = e - s
        offsets = np.zeros(len(rows), dtype=np.int64)
        np.cumsum(lengths[:-1], out=offsets[1:])
        # Vectorised multi-slice gather: position j of row r reads flat[s_r + j].
        idx = np.repeat(s - offsets, lengths) + np.arange(lengths.sum())
        return self.flat[idx], offsets


def tokenize_columns(tokenizer: Tokenizer, columns: dict[str, list[str]]) -> dict[str, PackedColumn]:
    tokenizer.no_padding()
    tokenizer.no_truncation()
    return {
        name: PackedColumn([e.ids for e in tokenizer.encode_batch(texts, add_special_tokens=False)])
        for name, texts in columns.items()
    }


def no_duplicates_batches(row_keys: list[tuple], batch_size: int, rng: np.random.Generator) -> Iterator[np.ndarray]:
    """Shuffle, then fill each batch skipping rows whose texts (any column) are
    already in it; skipped rows wait for a later batch (sentence-transformers
    BatchSamplers.NO_DUPLICATES, blog §Training Arguments). A duplicate text in
    the batch would be an in-batch 'negative' identical to a positive, i.e. a
    false negative for MNRL.

    row_keys[i] = tuple of hashes of the texts of row i.
    """
    remaining = dict.fromkeys(rng.permutation(len(row_keys)).tolist())
    while remaining:
        batch, seen = [], set()
        for idx in remaining:
            keys = row_keys[idx]
            if seen.isdisjoint(keys):
                batch.append(idx)
                seen.update(keys)
                if len(batch) == batch_size:
                    break
        for idx in batch:
            del remaining[idx]
        yield np.asarray(batch)


def load_triplets(cfg: TrainConfig) -> dict[str, list[str]]:
    from datasets import load_dataset

    ds = load_dataset(cfg.dataset, cfg.dataset_subset, split="train")
    ds = ds.train_test_split(test_size=cfg.eval_holdout, seed=cfg.seed)["train"]
    if cfg.max_train_samples:
        ds = ds.select(range(min(cfg.max_train_samples, len(ds))))
    return {c: list(ds[c]) for c in ds.column_names}


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------
def linear_warmup_decay(step: int, warmup: int, total: int) -> float:
    """LR multiplier of HF `get_linear_schedule_with_warmup`."""
    if step < warmup:
        return step / max(1, warmup)
    return max(0.0, (total - step) / max(1, total - warmup))


def train(
    cfg: TrainConfig,
    columns: dict[str, list[str]] | None = None,
    eval_fn: Callable[[StaticEmbedder], dict] | None = None,
    device: str | None = None,
) -> tuple[StaticEmbedder, dict]:
    """Train and return (model, history). ``columns`` overrides dataset loading
    (anchor column first, then positive, then optional negative)."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    t0 = time.time()
    columns = columns if columns is not None else load_triplets(cfg)
    names = list(columns)
    tokenizer = Tokenizer.from_pretrained(cfg.tokenizer_name)
    packed = tokenize_columns(tokenizer, columns)
    row_keys = [tuple(hash(t) for t in texts) for texts in zip(*columns.values())]
    n_rows = len(row_keys)
    t_data = time.time() - t0

    model = StaticEmbedder(Tokenizer.from_pretrained(cfg.tokenizer_name), embedding_dim=cfg.embedding_dim).to(device)
    # torch AdamW = HF Trainer default "adamw_torch" (betas 0.9/0.999, eps 1e-8).
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps_per_epoch = math.ceil(n_rows / cfg.batch_size)
    total = steps_per_epoch * cfg.epochs
    warmup = math.ceil(cfg.warmup_ratio * total)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: linear_warmup_decay(s, warmup, total))

    def loss_fn(*embs: torch.Tensor) -> torch.Tensor:
        if cfg.matryoshka_dims:
            return matryoshka_loss(*embs, dims=cfg.matryoshka_dims, scale=cfg.scale)
        return mnrl_loss(*embs, scale=cfg.scale)

    history: dict = {"config": asdict(cfg), "n_train": n_rows, "steps": total, "loss": [], "eval": [], "data_prep_s": t_data}
    if eval_fn is not None:
        history["eval"].append({"step": 0, **eval_fn(model)})
        model.train()

    step, t_train = 0, time.time()
    for _ in range(cfg.epochs):
        for rows in no_duplicates_batches(row_keys, cfg.batch_size, rng):
            # One EmbeddingBag call for all columns of the batch, then split.
            gathered = [packed[c].gather(rows) for c in names]
            shift = np.cumsum([0] + [len(ids) for ids, _ in gathered[:-1]])
            ids = torch.from_numpy(np.concatenate([g[0] for g in gathered])).to(device, non_blocking=True)
            offsets = torch.from_numpy(np.concatenate([g[1] + s for g, s in zip(gathered, shift)])).to(device)
            with torch.autocast(device_type=device.split(":")[0], dtype=torch.bfloat16, enabled=cfg.bf16):
                embs = model(ids, offsets).split(len(rows))
                loss = loss_fn(*embs)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % cfg.log_every == 0 or step == 1 or step == total:
                history["loss"].append({"step": step, "loss": loss.item(), "lr": sched.get_last_lr()[0]})
                print(f"step {step:4d}/{total}  loss {loss.item():.4f}  lr {sched.get_last_lr()[0]:.4f}", flush=True)
            if eval_fn is not None and cfg.eval_every and step % cfg.eval_every == 0 and step != total:
                history["eval"].append({"step": step, **eval_fn(model)})
                model.train()
    history["train_s"] = time.time() - t_train
    model.eval()
    if eval_fn is not None:
        history["eval"].append({"step": step, **eval_fn(model)})
    return model, history


def save_history(history: dict, path: str) -> None:
    with open(path, "w") as f:
        json.dump(history, f, indent=1)
