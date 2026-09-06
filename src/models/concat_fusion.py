"""E1 — text + random emoji embedding, concatenation fusion.

E1 = E0 frozen BERT text representation (768-d, mean pool over text_without_emoji)
   + randomly-initialized, trainable emoji embedding (32-d, mean pool over emoji_list)
   -> concatenation -> 800-d fused representation
   -> MLP head Linear(800->256) -> ReLU -> Dropout(0.3) -> Linear(256->3)

Critical controlled-experiment constraints:
- The text branch receives ONLY ``text_without_emoji`` (no emojis).
- The emoji branch receives ONLY ``emoji_list``.
- ``original_text`` is never passed to BERT.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel

from src.embeddings.emoji_encoder import EmojiEncoder


class ConcatFusionModel(nn.Module):
    """Text + emoji embeddings with concatenation fusion (E1/E2)."""

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

        # Emoji branch: random trainable embedding
        self.emoji_encoder = EmojiEncoder(
            vocab_size=emoji_vocab_size,
            embedding_dim=emoji_dim,
            unk_id=0,
            max_emojis=max_emojis,
        )

        # Classification head (same MLP shape as E0 but input = fused 800)
        self.classifier = nn.Sequential(
            nn.Linear(self.fused_dim, classifier_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden, num_labels),
        )

    def forward(self, input_ids, attention_mask, emoji_ids, emoji_masks):
        # Text branch
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        last_hidden = outputs.last_hidden_state  # (B, L, H)
        mask = attention_mask.unsqueeze(-1).float()  # (B, L, 1)
        summed = (last_hidden * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        text_rep = summed / counts  # (B, 768)

        # Emoji branch
        emoji_rep = self.emoji_encoder(emoji_ids, emoji_masks)  # (B, 32)

        # Fusion: concatenation
        fused = torch.cat([text_rep, emoji_rep], dim=-1)  # (B, 800)
        logits = self.classifier(fused)
        return logits
