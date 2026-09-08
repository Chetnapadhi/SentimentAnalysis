"""Attention-based fusion model for E3 and E4.

E3 = frozen BERT text (768-d) + randomly-initialized emoji embedding (32-d)
   -> text-conditioned attention fusion -> 32-d emoji context
   -> concatenation -> 800-d fused representation
   -> MLP head Linear(800->256) -> ReLU -> Dropout(0.3) -> Linear(256->3)

E4 = same architecture but emoji embeddings are pretrained (from TweetEval)
     and frozen.

Controlled-experiment constraints:
- Text branch receives ONLY ``text_without_emoji``.
- Emoji branch receives ONLY ``emoji_list``.
- ``original_text`` is never passed to BERT.
- Attention dropout is strictly 0 (no dropout on attention weights).
- Zero-emoji examples use a learned fallback parameter ``emoji_absent`` ([32]).
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel


class AttentionFusionModel(nn.Module):
    """Text + emoji with text-conditioned attention fusion (E3/E4).

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier for the text encoder.
    text_dim : int
        Dimensionality of text representation (768 for BERT-base).
    emoji_dim : int
        Dimensionality of emoji embeddings.
    num_labels : int
        Number of output classes.
    dropout : float
        Dropout rate for the classifier head (specified 0.3).
    classifier_hidden : int
        Hidden layer size in the MLP classifier.
    freeze_encoder : bool
        Whether to freeze the text encoder weights.
    emoji_vocab_size : int
        Size of the emoji vocabulary (including UNK at index 0).
    max_emojis : int
        Maximum number of emojis per example.
    init_mode : str
        ``'random'`` for E3 (random emoji init, trainable) or
        ``'pretrained'`` for E4 (load pretrained, frozen).
    pretrained_emoji_path : str
        Path to pretrained emoji embedding artifact. Required when
        ``init_mode='pretrained'``.
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
        init_mode: str = "random",
        pretrained_emoji_path: str = "",
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.text_dim = text_dim
        self.emoji_dim = emoji_dim
        self.init_mode = init_mode
        self.fused_dim = text_dim + emoji_dim  # 800

        # Text branch: frozen BERT
        config = AutoConfig.from_pretrained(model_name, num_labels=num_labels)
        self.encoder = AutoModel.from_pretrained(model_name, config=config)
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        # Emoji branch
        self.emoji_encoder = nn.Embedding(
            num_embeddings=emoji_vocab_size,
            embedding_dim=emoji_dim,
            padding_idx=0,
        )

        if init_mode == "pretrained" and pretrained_emoji_path:
            self._load_pretrained_emoji(pretrained_emoji_path, emoji_vocab_size, emoji_dim)
            for param in self.emoji_encoder.parameters():
                param.requires_grad = False
        else:
            # Random initialization (E3)
            nn.init.normal_(self.emoji_encoder.weight, mean=0.0, std=0.1)

        # Attention projections (linear, no bias)
        self.Wq = nn.Linear(text_dim, emoji_dim, bias=False)  # Q = Wq(text)
        self.Wk = nn.Linear(emoji_dim, emoji_dim, bias=False)  # K = Wk(emoji)
        self.Wv = nn.Linear(emoji_dim, emoji_dim, bias=False)  # V = Wv(emoji)

        # Learned fallback representation for examples with zero emojis: shape [32]
        self.emoji_absent = nn.Parameter(torch.empty(emoji_dim))
        nn.init.normal_(self.emoji_absent, mean=0.0, std=0.1)

        # Classification head: Dropout ONLY here (rate = 0.3)
        self.classifier = nn.Sequential(
            nn.Linear(self.fused_dim, classifier_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden, num_labels),
        )

        # Intermediate activations storage for inspection and smoke testing
        self.last_text_rep: torch.Tensor | None = None
        self.last_emoji_emb: torch.Tensor | None = None
        self.last_Q: torch.Tensor | None = None
        self.last_K: torch.Tensor | None = None
        self.last_V: torch.Tensor | None = None
        self.last_attention_weights: torch.Tensor | None = None
        self.last_emoji_context: torch.Tensor | None = None
        self.last_fused: torch.Tensor | None = None

    def _load_pretrained_emoji(
        self, path: str, vocab_size: int, emoji_dim: int
    ) -> None:
        """Load pretrained emoji embedding weights."""
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Pretrained emoji embedding not found: {path}"
            )
        artifact = torch.load(path, map_location="cpu")
        pretrained_weight = artifact["emoji_embedding"]  # (V, D)
        assert pretrained_weight.shape == (vocab_size, emoji_dim), (
            f"Shape mismatch: expected ({vocab_size}, {emoji_dim}), "
            f"got {pretrained_weight.shape}"
        )
        self.emoji_encoder.weight.data.copy_(pretrained_weight)

    def _attention_fusion(
        self,
        text_rep: torch.Tensor,
        emoji_ids: torch.Tensor,
        emoji_masks: torch.Tensor,
    ) -> torch.Tensor:
        """Compute emoji context via text-conditioned scaled dot-product attention.

        Args:
            text_rep: (B, text_dim) text representation.
            emoji_ids: (B, M) emoji token IDs.
            emoji_masks: (B, M) float mask (1=valid, 0=pad).

        Returns:
            emoji_context: (B, emoji_dim) attention-weighted or fallback representation.
        """
        B = text_rep.size(0)
        emoji_emb = self.emoji_encoder(emoji_ids)  # (B, M, D)

        Q = self.Wq(text_rep)          # (B, D)
        K = self.Wk(emoji_emb)         # (B, M, D)
        V = self.Wv(emoji_emb)         # (B, M, D)

        # Identify examples with at least one valid emoji
        valid_counts = emoji_masks.sum(dim=-1)  # (B,)
        has_emoji = valid_counts > 0            # (B,) bool

        # Scaled dot-product: Q @ K^T / sqrt(d_k)
        scores = torch.einsum("bd,bmd->bm", Q, K) / (self.emoji_dim ** 0.5)  # (B, M)
        scores = scores.masked_fill(emoji_masks == 0, float("-inf"))

        # Attention weights calculation:
        # 1. Padded positions must be exactly 0.
        # 2. For examples with zero valid emojis, do NOT perform softmax over all -inf.
        # 3. NO attention dropout is applied.
        alpha = torch.zeros_like(scores)  # (B, M)
        if has_emoji.any():
            alpha[has_emoji] = torch.softmax(scores[has_emoji], dim=-1)

        # Store intermediate tensors for verification
        self.last_text_rep = text_rep.detach()
        self.last_emoji_emb = emoji_emb.detach()
        self.last_Q = Q.detach()
        self.last_K = K.detach()
        self.last_V = V.detach()
        self.last_attention_weights = alpha.detach()

        # Context computation:
        # Non-empty: weighted sum of V
        context = torch.einsum("bm,bmd->bd", alpha, V)  # (B, D)

        # Zero-emoji: directly use the learned fallback parameter
        fallback = self.emoji_absent.unsqueeze(0).expand(B, -1)  # (B, D)
        emoji_context = torch.where(has_emoji.unsqueeze(-1), context, fallback)

        self.last_emoji_context = emoji_context.detach()
        return emoji_context

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

        emoji_context = self._attention_fusion(text_rep, emoji_ids, emoji_masks)
        fused = torch.cat([text_rep, emoji_context], dim=-1)
        self.last_fused = fused.detach()
        return self.classifier(fused)

    def forward_from_cache(
        self,
        text_rep: torch.Tensor,
        emoji_ids: torch.Tensor,
        emoji_masks: torch.Tensor,
    ) -> torch.Tensor:
        """Forward from precomputed text embeddings (skips BERT)."""
        emoji_context = self._attention_fusion(text_rep, emoji_ids, emoji_masks)
        fused = torch.cat([text_rep, emoji_context], dim=-1)
        self.last_fused = fused.detach()
        return self.classifier(fused)


# Backward-compatible alias for existing imports
E3AttentionFusionModel = AttentionFusionModel