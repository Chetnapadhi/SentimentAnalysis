"""E0 — text-only sentiment classifier.

Architecture:
- Frozen pretrained BERT encoder (bert-base-uncased)
- Mean pooling of the [CLS]-style token representations
- Trainable MLP classification head -> 3 classes (Bearish/Neutral/Bullish)

Input is ``text_without_emoji`` — emojis are explicitly removed so the model
cannot see them, keeping E0 a clean text-only baseline.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel


class TextOnlyModel(nn.Module):
    """Frozen-BERT + pooling + MLP head for text-only sentiment."""

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        hidden_size: int = 768,
        num_labels: int = 3,
        dropout: float = 0.1,
        freeze_encoder: bool = True,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.hidden_size = hidden_size
        self.num_labels = num_labels

        config = AutoConfig.from_pretrained(model_name, num_labels=num_labels)
        self.encoder = AutoModel.from_pretrained(model_name, config=config)

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_labels),
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        # Mean pooling over non-pad tokens
        last_hidden = outputs.last_hidden_state  # (B, L, H)
        mask = attention_mask.unsqueeze(-1).float()  # (B, L, 1)
        summed = (last_hidden * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        pooled = summed / counts  # (B, H)
        logits = self.classifier(pooled)
        return logits
