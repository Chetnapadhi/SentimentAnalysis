"""Loss functions with class weighting."""

from __future__ import annotations

import torch
import torch.nn as nn


def class_weights_from_counts(counts: dict[int, int]) -> torch.Tensor:
    """Inverse-frequency class weights normalized so mean weight = 1.

    Parameters
    ----------
    counts : dict mapping class id -> frequency.

    Returns
    -------
    torch.Tensor of length n_classes.
    """
    classes = sorted(counts.keys())
    n_classes = len(classes)
    total = sum(counts.values())
    weights = {c: total / (n_classes * counts[c]) for c in classes}
    return torch.tensor([weights[c] for c in classes], dtype=torch.float32)


def build_weighted_cross_entropy(counts: dict[int, int]) -> nn.Module:
    """Build a class-weighted cross-entropy loss."""
    weights = class_weights_from_counts(counts)
    return nn.CrossEntropyLoss(weight=weights)
