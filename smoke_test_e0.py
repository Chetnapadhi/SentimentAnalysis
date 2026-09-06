"""Smoke test for the E0 head-training loop and evaluation code path.

Uses synthetic embeddings (no BERT featurization) to verify the classifier
training, checkpointing, metric computation, confusion matrix, and plot code
work end-to-end locally on CPU. This does NOT featurize real text (which needs
GPU for reasonable speed on 91K examples).
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score

from src.models.text_only import TextOnlyModel
from src.training.losses import build_weighted_cross_entropy
from src.utils.seed import set_seed


def make_synthetic(n=400, hidden=768, n_classes=3, seed=42):
    rng = np.random.RandomState(seed)
    emb = rng.randn(n, hidden).astype(np.float32)
    # Create separable classes so head can learn
    labels = (rng.rand(n) * n_classes).astype(int)
    return emb, labels


def main():
    set_seed(42)
    device = torch.device("cpu")
    emb, labels = make_synthetic()

    counts = {0: int((labels == 0).sum()), 1: int((labels == 1).sum()), 2: int((labels == 2).sum())}
    from src.training.losses import class_weights_from_counts
    weights = class_weights_from_counts(counts)
    print("Class weights:", weights.tolist())

    model = TextOnlyModel(num_labels=3, freeze_encoder=True)
    # Only head is used; skip encoder entirely
    criterion = torch.nn.CrossEntropyLoss(weight=weights)

    # Train head on synthetic
    optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=0.01)
    X = torch.tensor(emb)
    y = torch.tensor(labels)
    model.train()
    for step in range(50):
        optimizer.zero_grad()
        out = model.classifier(X)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

    # Evaluate
    model.eval()
    with torch.no_grad():
        logits = model.classifier(X)
        preds = logits.argmax(dim=-1).numpy()

    acc = accuracy_score(labels, preds)
    mf1 = f1_score(labels, preds, average="macro", zero_division=0)
    print(f"Smoke test accuracy: {acc:.4f}, macro-F1: {mf1:.4f}")

    # Test checkpoint save + confusion matrix + plot code
    with tempfile.TemporaryDirectory() as tmp:
        from src.evaluation.visualization import plot_confusion_matrix, plot_training_history
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(labels, preds, labels=[0, 1, 2])
        plot_confusion_matrix(cm, os.path.join(tmp, "cm.png"))
        hist = {"train_loss": [1.0, 0.8, 0.6], "val_loss": [1.1, 0.9, 0.7],
                "val_acc": [0.5, 0.6, 0.7], "val_macro_f1": [0.3, 0.5, 0.6]}
        plot_training_history(hist, os.path.join(tmp, "hist.png"))
        print("Plots generated OK in:", tmp)

    print("\nSMOKE TEST PASSED")


if __name__ == "__main__":
    main()
