"""Model2Vec distillation, from scratch.

Model2Vec docs, "How Model2Vec works": "for every individual token in the
selected base model, we run a forward pass through the base model. This creates
our initial static embeddings." Then PCA, then SIF weighting with token
probabilities estimated by Zipf's law. Inference = mean of token vectors.

Steps, as numbered in this file:
  (M1) E_raw[t] = mean_pool( Transformer([CLS] t [SEP]) )      for each token t
  (M2) E_pca    = PCA_k(E_raw)                                   k = 256 by default
  (M3) p(t)     = (1 / (rank(t) + 2)) / Z                        Zipf's law, rank = token id order
       w(t)     = a / (a + p(t))                                 SIF, a = 1e-4
  (M4) E[t]     = w(t) * E_pca[t]
  (M5) at inference: e(x) = normalize( mean_i E[t_i] )
"""
from __future__ import annotations

import re

import numpy as np
import torch
from tokenizers import Tokenizer
from transformers import AutoModel, AutoTokenizer

from static_emb.static_model import StaticEmbedder

# Tokens removed before distillation (model2vec default `token_remove_pattern`).
DEFAULT_REMOVE_PATTERN = r"\[unused\d+\]"


def _kept_token_ids(tokenizer: Tokenizer, hf_tok, token_remove_pattern: str | None) -> np.ndarray:
    """Vocabulary used for distillation, in id order (position = Zipf rank).

    Removes tokens matching ``token_remove_pattern`` ([unusedN]) and the special
    tokens other than [PAD]/[UNK] ([CLS], [SEP], [MASK]): they never appear
    when encoding with add_special_tokens=False. Not stated in the docs;
    taken from model2vec ``tokenizer/tokenizer.py::clean_and_create_vocabulary``
    (see README Deviations #2). It changes every Zipf rank by 3.
    """
    id_to_tok = {i: t for t, i in tokenizer.get_vocab().items()}
    pattern = re.compile(token_remove_pattern) if token_remove_pattern else None
    special = set(hf_tok.all_special_tokens) - {hf_tok.pad_token, hf_tok.unk_token}
    return np.array([
        i for i in range(tokenizer.get_vocab_size())
        if id_to_tok[i] not in special and not (pattern and pattern.fullmatch(id_to_tok[i]))
    ])


@torch.inference_mode()
def token_forward_embeddings(
    model: torch.nn.Module,
    token_ids: np.ndarray,
    cls_id: int | None,
    sep_id: int | None,
    batch_size: int = 1024,
    device: str = "cpu",
) -> np.ndarray:
    """(M1): one forward pass per vocabulary token.

    Each token t is fed as the 3-token sequence [CLS] t [SEP] (model2vec
    `turn_tokens_into_ids`: [*bos_ids, id, *eos_ids]), and the output hidden
    states of all positions are mean-pooled (model2vec default PoolingMode.MEAN).
    All sequences have the same length, so no padding or mask is needed.

    Returns: (len(token_ids), hidden) float32 array.
    """
    model = model.to(device).eval()
    out = []
    for start in range(0, len(token_ids), batch_size):
        chunk = torch.as_tensor(token_ids[start : start + batch_size], dtype=torch.long)[:, None]  # (B, 1)
        parts = [chunk]
        if cls_id is not None:
            parts.insert(0, torch.full_like(chunk, cls_id))
        if sep_id is not None:
            parts.append(torch.full_like(chunk, sep_id))
        seq = torch.cat(parts, dim=1).to(device)  # (B, 3) = [CLS] t [SEP]
        kwargs = {"input_ids": seq, "attention_mask": torch.ones_like(seq)}
        if "token_type_ids" in model.forward.__code__.co_varnames:
            kwargs["token_type_ids"] = torch.zeros_like(seq)
        hidden = model(**kwargs).last_hidden_state.float()  # (B, L, h)
        out.append(hidden.mean(dim=1).cpu())
    return torch.cat(out).numpy()


def pca(X: np.ndarray, k: int) -> np.ndarray:
    """(M2): project X (n, h) onto its top-k principal components -> (n, k).

    Plain PCA via eigendecomposition of the covariance; no whitening. The
    Model2Vec docs note PCA helps "even if we don't reduce dimensionality, as it
    normalizes the embedding space" (centres it and makes the axes decorrelated,
    sorted by variance). Sign convention: the largest-|score| entry of every
    component is positive (same as model2vec / sklearn, so outputs match).
    """
    Xc = X - X.mean(axis=0)
    cov = (Xc.T @ Xc) / (X.shape[0] - 1)
    eigvals, eigvecs = np.linalg.eigh(cov)  # ascending order
    components = eigvecs[:, ::-1][:, :k]  # (h, k), largest variance first
    scores = Xc @ components
    signs = np.sign(scores[np.argmax(np.abs(scores), axis=0), np.arange(k)])
    return scores * signs


def zipf_sif_weights(n_tokens: int, sif_coefficient: float = 1e-4) -> np.ndarray:
    """(M3): SIF weights w = a / (a + p) with p from Zipf's law.

    Tokenizer vocabularies are (roughly) sorted by frequency, so the rank of a
    token approximates its frequency: p(rank) ∝ 1/rank (Zipf). model2vec uses
    1/(rank+2) for 0-based ranks. Frequent tokens ("the", ".") get w ≈ 0,
    rare content tokens get w ≈ 1. This is the static-embedding version of IDF.
    """
    inv_rank = 1.0 / np.arange(2, n_tokens + 2)
    p = inv_rank / inv_rank.sum()
    return sif_coefficient / (sif_coefficient + p)


def distill(
    model_name: str,
    pca_dims: int | None = 256,
    sif_coefficient: float | None = 1e-4,
    device: str | None = None,
    batch_size: int = 1024,
    token_remove_pattern: str | None = DEFAULT_REMOVE_PATTERN,
    raw_embeddings: np.ndarray | None = None,
) -> StaticEmbedder:
    """Distil a (sentence) transformer into a StaticEmbedder.

    ``pca_dims=None`` / ``sif_coefficient=None`` switch off (M2) / (M3-4), for
    the ablation. ``raw_embeddings`` lets the ablation reuse one (M1) pass.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    hf_tok = AutoTokenizer.from_pretrained(model_name)
    tokenizer = Tokenizer.from_pretrained(model_name)

    V = tokenizer.get_vocab_size()
    kept = _kept_token_ids(tokenizer, hf_tok, token_remove_pattern)

    if raw_embeddings is None:
        model = AutoModel.from_pretrained(model_name)
        raw_embeddings = token_forward_embeddings(
            model, kept, hf_tok.cls_token_id, hf_tok.sep_token_id, batch_size=batch_size, device=device
        )
    E = raw_embeddings
    if pca_dims is not None:
        E = pca(E, min(pca_dims, E.shape[1]))
    if sif_coefficient is not None:
        # model2vec computes the weights before PCA and applies them after it;
        # PCA is a per-row linear map plus centring, so the order only matters
        # in that the weights must not influence the PCA fit.
        E = E * zipf_sif_weights(len(kept), sif_coefficient)[:, None]

    # Removed tokens ([unusedN]) never come out of the tokenizer on real text;
    # they get zero rows so that the table still matches the tokenizer ids.
    W = np.zeros((V, E.shape[1]), dtype=np.float32)
    W[kept] = E
    return StaticEmbedder(tokenizer, weights=W, normalize=True, skip_unk=True)  # (M5)


def raw_token_embeddings(model_name: str, device: str | None = None, batch_size: int = 1024) -> np.ndarray:
    """Run only (M1), e.g. to share one forward pass across ablation variants."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    hf_tok = AutoTokenizer.from_pretrained(model_name)
    tokenizer = Tokenizer.from_pretrained(model_name)
    kept = _kept_token_ids(tokenizer, hf_tok, DEFAULT_REMOVE_PATTERN)
    model = AutoModel.from_pretrained(model_name)
    return token_forward_embeddings(model, kept, hf_tok.cls_token_id, hf_tok.sep_token_id, batch_size, device)
