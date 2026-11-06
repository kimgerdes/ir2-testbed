"""Training objectives of the blog recipe (§Loss Function Selection).

(L1) MultipleNegativesRankingLoss (MNRL), a.k.a. InfoNCE with in-batch negatives:
     "Given an anchor (e.g. a question), assign the highest similarity to the
     corresponding positive (i.e. answer) out of all positives and negatives
     (e.g. all answers) in the batch."

         S_ij = s * cos(q_i, c_j),    c = [d_1^+, ..., d_B^+, d_1^-, ..., d_B^-]
         L    = (1/B) * sum_i  -log( exp(S_ii) / sum_j exp(S_ij) )

     with scale s = 20 (sentence-transformers default = temperature 0.05).

(L2) MatryoshkaLoss: the same loss on the first m dimensions, summed over m:

         L_MRL = sum_{m in M} w_m * L( q[:, :m], c[:, :m] ),  M = {1024, 512, ..., 32}, w_m = 1

     so that the first m coordinates on their own already form a good embedding.
"""
from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F


def mnrl_loss(
    anchors: torch.Tensor,
    positives: torch.Tensor,
    negatives: torch.Tensor | None = None,
    scale: float = 20.0,
) -> torch.Tensor:
    """(L1). anchors, positives, negatives: (B, d). Returns a scalar.

    Row i of the score matrix holds anchor i against every candidate in the
    batch; the correct candidate is column i (its own positive), so this is a
    B-way (or 2B-way with hard negatives) classification with labels 0..B-1.
    """
    candidates = positives if negatives is None else torch.cat([positives, negatives], dim=0)
    scores = scale * F.normalize(anchors, dim=-1) @ F.normalize(candidates, dim=-1).T  # (B, B or 2B)
    labels = torch.arange(anchors.shape[0], device=anchors.device)
    return F.cross_entropy(scores, labels)


def matryoshka_loss(
    anchors: torch.Tensor,
    positives: torch.Tensor,
    negatives: torch.Tensor | None = None,
    dims: Sequence[int] = (1024, 512, 256, 128, 64, 32),
    weights: Sequence[float] | None = None,
    scale: float = 20.0,
) -> torch.Tensor:
    """(L2). Sum of MNRL over prefix truncations ``x[:, :m]``.

    The embeddings are computed once and only sliced, which for a static model
    is the whole cost saving. Truncate first, then normalise: the cosine in
    ``mnrl_loss`` renormalises each prefix.
    """
    weights = [1.0] * len(dims) if weights is None else weights
    total = anchors.new_zeros(())
    for m, w in zip(dims, weights):
        neg = None if negatives is None else negatives[:, :m]
        total = total + w * mnrl_loss(anchors[:, :m], positives[:, :m], neg, scale=scale)
    return total
