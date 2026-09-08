"""E5 — gated fusion model with pretrained emoji embeddings.

E5 = frozen BERT text (768-d) + pretrained frozen emoji embedding (32-d)
   -> mean-pooled emoji aggregation -> 32-d emoji representation
   -> learned 32-d gate g = sigmoid(Wg · [text; emoji] + bg) -> shape [B, 32]
   -> gated_emoji = g * emoji_rep -> shape [B, 32]
   -> fused = concat(text, gated_emoji) -> shape [B, 800]
   -> MLP head Linear(800->256) -> ReLU -> Dropout(0.3) -> Linear(256->3)

Controlled-experiment constraints:
- Text branch receives ONLY ``text_without_emoji``.
- Emoji branch receives ONLY ``emoji_list``.
- ``original_text`` is never passed to BERT.
- Emoji embeddings are pretrained (from TweetEval) and frozen.
- Gate values are 32-dimensional and reported as interpretability signals, not causal explanations.
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel


class GatedFusionModel(nn.Module):
    """Text + emoji with 32-dimensional gated fusion (E5).

    The 32-d gate learns per-feature modulation of emoji representations.
    A gate value near 1 means the model relies on emoji signal in that dimension;
    near 0 means the model suppresses it.
    """

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        text_dim: int = 768,
        emoji_dim: int = 32,
        num_labels: int = 3,
        dropout: float = 0.3,
        classifier_hidden: int = 256,
        freeze_encoder: bool = True,
        emoji_vocab_size: int = 1,
        max_emojis: int = 8,
        pretrained_emoji_path: str = "",
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.text_dim = text_dim
        self.emoji_dim = emoji_dim
        self.fused_dim = text_dim + emoji_dim  # 800

        # Text branch: frozen BERT
        config = AutoConfig.from_pretrained(model_name, num_labels=num_labels)
        self.encoder = AutoModel.from_pretrained(model_name, config=config)
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        # Emoji branch: pretrained frozen embedding
        self.emoji_encoder = nn.Embedding(
            num_embeddings=emoji_vocab_size,
            embedding_dim=emoji_dim,
            padding_idx=0,
        )
        if pretrained_emoji_path and os.path.exists(pretrained_emoji_path):
            self._load_pretrained_emoji(pretrained_emoji_path, emoji_vocab_size, emoji_dim)
        else:
            nn.init.normal_(self.emoji_encoder.weight, mean=0.0, std=0.1)

        # Freeze emoji embeddings
        for param in self.emoji_encoder.parameters():
            param.requires_grad = False

        # 32-dimensional Gate: Linear(800 -> 32) followed by Sigmoid
        self.gate = nn.Sequential(
            nn.Linear(text_dim + emoji_dim, emoji_dim),
            nn.Sigmoid(),
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.fused_dim, classifier_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden, num_labels),
        )

        # Storage for gate values and intermediate tensors (for analysis and verification)
        self._last_gate_values: torch.Tensor | None = None
        self.last_text_rep: torch.Tensor | None = None
        self.last_emoji_rep: torch.Tensor | None = None
        self.last_gated_emoji: torch.Tensor | None = None
        self.last_fused: torch.Tensor | None = None

    def _load_pretrained_emoji(
        self, path: str, vocab_size: int, emoji_dim: int
    ) -> None:
        """Load pretrained emoji embedding weights."""
        artifact = torch.load(path, map_location="cpu")
        pretrained_weight = artifact["emoji_embedding"]
        assert pretrained_weight.shape == (vocab_size, emoji_dim), (
            f"Shape mismatch: expected ({vocab_size}, {emoji_dim}), "
            f"got {pretrained_weight.shape}"
        )
        self.emoji_encoder.weight.data.copy_(pretrained_weight)

    def _emoji_mean_pool(
        self, emoji_ids: torch.Tensor, emoji_masks: torch.Tensor
    ) -> torch.Tensor:
        """Mean-pool emoji embeddings with explicit zero-emoji safety.

        Args:
            emoji_ids: (B, M) emoji token IDs.
            emoji_masks: (B, M) float mask (1=valid, 0=pad).

        Returns:
            (B, emoji_dim) mean-pooled emoji representation.
        """
        emb = self.emoji_encoder(emoji_ids)       # (B, M, D)
        mask = emoji_masks.unsqueeze(-1).float()  # (B, M, 1)
        valid_counts = mask.sum(dim=1)            # (B, 1)
        has_emoji = valid_counts > 0              # (B, 1) bool

        summed = (emb * mask).sum(dim=1)          # (B, D)
        # For examples with no valid emojis, return clean zero vector without division or NaN
        emoji_rep = torch.where(
            has_emoji,
            summed / valid_counts.clamp(min=1.0),
            torch.zeros_like(summed),
        )
        return emoji_rep  # (B, D)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        emoji_ids: torch.Tensor,
        emoji_masks: torch.Tensor,
    ) -> torch.Tensor:
        """End-to-end forward (encodes text from tokens)."""
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        last_hidden = outputs.last_hidden_state
        mask = attention_mask.unsqueeze(-1).float()
        text_rep = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)

        return self._gated_forward(text_rep, emoji_ids, emoji_masks)

    def forward_from_cache(
        self,
        text_rep: torch.Tensor,
        emoji_ids: torch.Tensor,
        emoji_masks: torch.Tensor,
    ) -> torch.Tensor:
        """Forward from precomputed text embeddings (skips BERT)."""
        return self._gated_forward(text_rep, emoji_ids, emoji_masks)

    def _gated_forward(
        self,
        text_rep: torch.Tensor,
        emoji_ids: torch.Tensor,
        emoji_masks: torch.Tensor,
    ) -> torch.Tensor:
        """Core 32-dimensional gated fusion logic."""
        emoji_rep = self._emoji_mean_pool(emoji_ids, emoji_masks)  # (B, 32)

        # Gate: g = sigmoid(Wg · [text; emoji] + bg) -> shape (B, 32)
        gate_input = torch.cat([text_rep, emoji_rep], dim=-1)      # (B, 800)
        g = self.gate(gate_input)                                   # (B, 32)
        self._last_gate_values = g.detach()

        # Modulate emoji element-wise with 32-d gate, then concatenate
        gated_emoji = g * emoji_rep                                 # (B, 32)
        fused = torch.cat([text_rep, gated_emoji], dim=-1)          # (B, 800)

        self.last_text_rep = text_rep.detach()
        self.last_emoji_rep = emoji_rep.detach()
        self.last_gated_emoji = gated_emoji.detach()
        self.last_fused = fused.detach()

        return self.classifier(fused)

    @property
    def last_gate_values(self) -> torch.Tensor | None:
        """Return gate values from the last forward pass (shape: [B, 32])."""
        return self._last_gate_values
