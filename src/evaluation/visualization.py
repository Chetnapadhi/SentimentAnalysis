"""Visualization utilities for E0 (confusion matrix, training history)."""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

LABEL_NAMES = ["Bearish", "Neutral", "Bullish"]


def plot_confusion_matrix(cm: np.ndarray, save_path: str) -> None:
    """Plot a labeled 3x3 confusion matrix (and normalized version)."""
    cm = np.asarray(cm)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", ax=axes[0],
        xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES,
    )
    axes[0].set_title("Confusion Matrix (counts)")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Actual")

    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9)
    sns.heatmap(
        cm_norm, annot=True, fmt=".2f", cmap="Blues", ax=axes[1],
        xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES,
    )
    axes[1].set_title("Confusion Matrix (normalized)")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("Actual")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved confusion matrix -> {save_path}")


def plot_training_history(history: dict, save_path: str) -> None:
    """Plot training loss and validation metrics vs epoch."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    epochs = list(range(1, len(history["train_loss"]) + 1))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(epochs, history["train_loss"], marker="o", label="Train Loss")
    axes[0].plot(epochs, history["val_loss"], marker="o", label="Val Loss")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(epochs, history["val_acc"], marker="o", color="green")
    axes[1].set_title("Validation Accuracy")
    axes[1].set_xlabel("Epoch")

    axes[2].plot(epochs, history["val_macro_f1"], marker="o", color="purple")
    axes[2].set_title("Validation Macro-F1")
    axes[2].set_xlabel("Epoch")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved training history -> {save_path}")
