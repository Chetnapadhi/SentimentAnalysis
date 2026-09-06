"""Emoji encoder for the emoji-aware models.

Handles:
- emoji vocabulary (emoji_to_id / id_to_emoji)
- trainable embedding table (random or pretrained initialization)
- aggregation over multiple emojis (mean pooling by default)

The emoji vocabulary is built from the training split's extracted emojis so
that no test information leaks into the vocabulary. Unknown emojis at
eval/inference map to an UNK token.
"""

from __future__ import annotations

import json
from collections import Counter

import torch
import torch.nn as nn


class EmojiEncoder(nn.Module):
    """Trainable emoji embedding with mean aggregation."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 32,
        unk_id: int = 0,
        max_emojis: int = 8,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.unk_id = unk_id
        self.max_emojis = max_emojis

        # Random init (trainable). Use a modest scale for stability.
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=None)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.1)

    def forward(self, emoji_ids: torch.Tensor, emoji_masks: torch.Tensor) -> torch.Tensor:
        """Aggregate emoji embeddings via mean pooling.

        Parameters
        ----------
        emoji_ids : (B, max_emojis) long tensor of emoji token ids.
        emoji_masks : (B, max_emojis) float mask (1 = valid, 0 = pad).

        Returns
        -------
        (B, embedding_dim) mean-pooled emoji representation.
        """
        emb = self.embedding(emoji_ids)  # (B, M, D)
        mask = emoji_masks.unsqueeze(-1).float()  # (B, M, 1)
        summed = (emb * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        return summed / counts  # (B, D)


# ---------------------------------------------------------------------------
# Vocabulary helpers
# ---------------------------------------------------------------------------

def build_emoji_vocab(emoji_lists: list[list[str]], min_freq: int = 1) -> dict:
    """Build a deterministic emoji vocabulary from training emojis only.

    Returns dict with keys: ``emoji_to_id``, ``id_to_emoji``, ``vocab_size``,
    ``unk_id``, ``counts``.
    """
    counter = Counter()
    for emojis in emoji_lists:
        counter.update(emojis)

    # id 0 = UNK, then most-common emojis
    emoji_to_id: dict[str, int] = {"<UNK>": 0}
    id_to_emoji: dict[int, str] = {0: "<UNK>"}
    for emoji_char, count in counter.most_common():
        if count >= min_freq:
            eid = len(emoji_to_id)
            emoji_to_id[emoji_char] = eid
            id_to_emoji[eid] = emoji_char

    return {
        "emoji_to_id": emoji_to_id,
        "id_to_emoji": id_to_emoji,
        "vocab_size": len(emoji_to_id),
        "unk_id": 0,
        "counts": dict(counter.most_common(50)),
    }


def encode_emoji_list(emojis: list[str], emoji_to_id: dict, max_emojis: int = 8) -> tuple[list[int], list[float]]:
    """Encode a list of emojis into padded ids + mask."""
    unk = emoji_to_id.get("<UNK>", 0)
    ids = [emoji_to_id.get(e, unk) for e in emojis[:max_emojis]]
    mask = [1.0] * len(ids)
    pad_len = max_emojis - len(ids)
    if pad_len > 0:
        ids += [0] * pad_len
        mask += [0.0] * pad_len
    return ids, mask


def save_vocab(vocab: dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)


def load_vocab(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
