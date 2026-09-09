"""Per-class metrics (Bearish, Neutral, Bullish) and confusion matrix visualization."""

from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.analysis.data_loader import (
    LABEL_NAMES,
    get_project_root,
    load_all_confusion_matrices,
    load_all_metrics,
)


def extract_per_class_metrics() -> pd.DataFrame:
    """Extract Precision, Recall, F1 for Bearish, Neutral, and Bullish across all models."""
    metrics = load_all_metrics()
    rows = []

    for exp in ["E0", "E1", "E2", "E3", "E4", "E5"]:
        m = metrics[exp]
        rep = m.get("classification_report", m.get("report", m.get("per_class", {})))

        for cls_name in ["Bearish", "Neutral", "Bullish"]:
            c_data = rep.get(cls_name, {})
            p = c_data.get("precision", 0.0)
            r = c_data.get("recall", 0.0)
            f1 = c_data.get("f1-score", c_data.get("f1", 0.0))
            supp = c_data.get("support", 0)

            rows.append({
                "experiment": exp,
                "class_name": cls_name,
                "precision": float(p),
                "recall": float(r),
                "f1_score": float(f1),
                "support": int(supp),
            })

    return pd.DataFrame(rows)


def plot_per_class_metrics(per_class_df: pd.DataFrame, out_path: Path) -> None:
    """Create a 3-panel comparison plot showing F1, Precision, Recall across classes."""
    sns.set_theme(style="whitegrid", font_scale=1.05)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharey=True)

    classes = ["Bearish", "Neutral", "Bullish"]
    palette = {"Bearish": "#c94c4c", "Neutral": "#8d99ae", "Bullish": "#2b9348"}

    for idx, metric in enumerate(["f1_score", "precision", "recall"]):
        ax = axes[idx]
        metric_title = "F1-Score" if metric == "f1_score" else metric.capitalize()

        sns.barplot(
            data=per_class_df,
            x="experiment",
            y=metric,
            hue="class_name",
            palette=palette,
            ax=ax,
            alpha=0.9,
        )
        ax.set_title(f"Class-Wise {metric_title}", fontsize=13, fontweight="bold", pad=10)
        ax.set_xlabel("Experiment", fontweight="bold")
        ax.set_ylabel(metric_title if idx == 0 else "", fontweight="bold")
        ax.set_ylim(0.2, 0.75)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y*100:.0f}%"))

        # Add data labels
        for container in ax.containers:
            ax.bar_label(container, fmt="%.2f", padding=2, fontsize=8, rotation=90)

        if idx == 2:
            ax.legend(title="Sentiment Class", loc="upper left", framealpha=0.95)
        else:
            ax.legend_.remove()

    plt.suptitle("Per-Class Sentiment Performance (Bearish, Neutral, Bullish) Across E0–E5", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved per-class comparison plot -> {out_path}")


def plot_all_confusion_matrices(cms: dict[str, np.ndarray], out_path: Path) -> None:
    """Plot a unified 2x3 grid of normalized confusion matrices for E0 through E5."""
    sns.set_theme(style="white", font_scale=1.0)
    fig, axes = plt.subplots(2, 3, figsize=(17, 10))
    axes = axes.flatten()

    class_names = ["Bearish", "Neutral", "Bullish"]
    experiments = ["E0", "E1", "E2", "E3", "E4", "E5"]

    titles = {
        "E0": "E0: Text-Only Baseline",
        "E1": "E1: Concat + Random",
        "E2": "E2: Concat + Pretrained",
        "E3": "E3: Attention + Random",
        "E4": "E4: Attention + Pretrained",
        "E5": "E5: Gated + Pretrained (32-d)",
    }

    for idx, exp in enumerate(experiments):
        ax = axes[idx]
        cm = cms[exp]

        # Calculate row-normalized percentage
        cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9)

        # Create combined label string: Count \n (Percentage%)
        annot = np.empty_like(cm, dtype=object)
        for i in range(3):
            for j in range(3):
                annot[i, j] = f"{cm[i, j]:,}\n({cm_norm[i, j]*100:.1f}%)"

        sns.heatmap(
            cm_norm,
            annot=annot,
            fmt="",
            cmap="Blues",
            cbar=(idx == 2 or idx == 5),
            xticklabels=class_names,
            yticklabels=class_names,
            ax=ax,
            vmin=0.2,
            vmax=0.7,
            linewidths=1.0,
            linecolor="white",
        )

        ax.set_title(titles[exp], fontsize=12, fontweight="bold", pad=8)
        ax.set_xlabel("Predicted Label" if idx >= 3 else "", fontweight="bold")
        ax.set_ylabel("Actual Label" if idx % 3 == 0 else "", fontweight="bold")

    plt.suptitle("Normalized Confusion Matrices Across All Six Experiments (Row-Normalized)", fontsize=16, fontweight="bold", y=0.98)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved all confusion matrices plot -> {out_path}")
