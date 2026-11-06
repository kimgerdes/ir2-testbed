"""The static encoder: a token lookup table followed by mean pooling.

Blog §Model Inspiration: "It's as simple as using a torch EmbeddingBag, which is
nothing more than an efficient Embedding (i.e. a lookup table for embeddings)
with mean pooling." This replaces the Transformer encoder of the usual
tokenizer -> encoder -> pooler pipeline (blog §Modern Embeddings).

Sentence embedding of a text with tokens t_1..t_n:

    e(x) = (1/n) * sum_i E[t_i]          E in R^{V x d}

No attention and no position embeddings: the same token always contributes the
same vector, so 'bank' in 'river bank' and 'bank account' gets one vector.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from torch import nn


class StaticEmbedder(nn.Module):
    """Tokenizer + ``nn.EmbeddingBag(mode="mean")``.

    Args:
        tokenizer: a Hugging Face ``tokenizers.Tokenizer`` (fast, Rust).
        embedding_dim: d, used for random init when ``weights`` is None.
        weights: optional (V, d) initial matrix E (e.g. from Model2Vec distillation).
        normalize: L2-normalise sentence embeddings in ``encode`` (Model2Vec does;
            for training it does not matter because the loss uses cosine similarity).
        max_length: optional cap on tokens per text (None = unlimited, the blog's
            setting: static models have no positional limit, blog §Next Steps).
        skip_unk: drop [UNK] tokens before pooling (model2vec ``StaticModel.tokenize``
            does this; sentence-transformers' StaticEmbedding does not).
    """

    def __init__(
        self,
        tokenizer: Tokenizer,
        embedding_dim: int | None = None,
        weights: torch.Tensor | np.ndarray | None = None,
        normalize: bool = False,
        max_length: int | None = None,
        skip_unk: bool = False,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        # Static models need no padding: texts are packed into one flat id vector
        # with offsets (that is what EmbeddingBag consumes).
        self.tokenizer.no_padding()
        self.tokenizer.no_truncation()
        self.normalize = normalize
        self.max_length = max_length
        self.skip_unk = skip_unk
        unk = getattr(tokenizer.model, "unk_token", None)
        self.unk_id = tokenizer.token_to_id(unk) if (skip_unk and unk) else None
        vocab_size = tokenizer.get_vocab_size()
        if weights is not None:
            weights = torch.as_tensor(np.asarray(weights), dtype=torch.float32)
            assert weights.shape[0] == vocab_size, (weights.shape, vocab_size)
            self.embedding = nn.EmbeddingBag.from_pretrained(weights, freeze=False, mode="mean")
        else:
            assert embedding_dim is not None
            # Random init: nn.EmbeddingBag default, N(0, 1). The blog found random
            # init works best when training on lots of data (§Model Inspiration).
            self.embedding = nn.EmbeddingBag(vocab_size, embedding_dim, mode="mean")

    @property
    def dim(self) -> int:
        return self.embedding.embedding_dim

    def tokenize(self, texts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (input_ids, offsets): flat LongTensor of all token ids and the
        start position of each text. No [CLS]/[SEP] (special tokens carry no
        meaning without attention) and no padding."""
        encodings = self.tokenizer.encode_batch(texts, add_special_tokens=False)
        ids = [e.ids for e in encodings]
        if self.unk_id is not None:
            ids = [[t for t in x if t != self.unk_id] for x in ids]
        if self.max_length:
            ids = [x[: self.max_length] for x in ids]
        lengths = np.fromiter((len(x) for x in ids), dtype=np.int64, count=len(ids))
        offsets = np.zeros(len(ids), dtype=np.int64)
        np.cumsum(lengths[:-1], out=offsets[1:])
        flat = np.fromiter((t for x in ids for t in x), dtype=np.int64, count=int(lengths.sum()))
        return torch.from_numpy(flat), torch.from_numpy(offsets)

    def forward(self, input_ids: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        """(N_tokens,), (B,) -> (B, d) mean of the token rows of each text.
        A text with zero tokens maps to the zero vector."""
        return self.embedding(input_ids, offsets)

    @torch.inference_mode()
    def encode(
        self,
        texts: list[str],
        batch_size: int = 4096,
        truncate_dim: int | None = None,
        normalize: bool | None = None,
        convert_to_numpy: bool = True,
    ):
        """texts -> (len(texts), d or truncate_dim) embeddings."""
        normalize = self.normalize if normalize is None else normalize
        device = self.embedding.weight.device
        out = []
        for start in range(0, len(texts), batch_size):
            ids, offsets = self.tokenize(texts[start : start + batch_size])
            emb = self(ids.to(device), offsets.to(device))
            if truncate_dim is not None:
                emb = emb[:, :truncate_dim]  # Matryoshka truncation: keep a prefix
            if normalize:
                emb = F.normalize(emb, dim=-1)
            out.append(emb)
        emb = torch.cat(out) if out else torch.empty(0, truncate_dim or self.dim)
        return emb.float().cpu().numpy() if convert_to_numpy else emb

    # --- persistence -------------------------------------------------------
    def save(self, path: str) -> None:
        import os

        os.makedirs(path, exist_ok=True)
        self.tokenizer.save(os.path.join(path, "tokenizer.json"))
        np.save(os.path.join(path, "embeddings.npy"), self.embedding.weight.detach().float().cpu().numpy())
        with open(os.path.join(path, "config.txt"), "w") as f:
            f.write(f"normalize={self.normalize}\nskip_unk={self.skip_unk}\n")

    @classmethod
    def load(cls, path: str) -> "StaticEmbedder":
        import os

        tok = Tokenizer.from_file(os.path.join(path, "tokenizer.json"))
        W = np.load(os.path.join(path, "embeddings.npy"))
        cfg = open(os.path.join(path, "config.txt")).read()
        return cls(tok, weights=W, normalize="normalize=True" in cfg, skip_unk="skip_unk=True" in cfg)
