"""Master E0–E5 comparison table, rankings, and baseline delta calculations."""

from __future__ import annotations

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.analysis.data_loader import get_project_root, load_all_metrics

EXP_DESCRIPTIONS = {
    "E0": "Text-only baseline (Frozen BERT)",
    "E1": "Random emoji embedding + Concat",
    "E2": "Pretrained emoji embedding + Concat",
    "E3": "Random emoji embedding + Attention",
    "E4": "Pretrained emoji embedding + Attention",
    "E5": "Pretrained emoji embedding + Gated Fusion (32-d)",
}

FUSION_TYPES = {
    "E0": "None (Text Only)",
    "E1": "Concatenation (Mean Pool)",
    "E2": "Concatenation (Mean Pool)",
    "E3": "Attention (Text-Conditioned)",
    "E4": "Attention (Text-Conditioned)",
    "E5": "Gated Modulation (32-d Gate)",
}

EMOJI_INITS = {
    "E0": "None",
    "E1": "Random (Trainable)",
    "E2": "TweetEval (Frozen)",
    "E3": "Random (Trainable)",
    "E4": "TweetEval (Frozen)",
    "E5": "TweetEval (Frozen)",
}


def build_master_comparison() -> pd.DataFrame:
    """Build master comparison dataframe across all six experiments."""
    metrics = load_all_metrics()

    rows = []
    for exp in ["E0", "E1", "E2", "E3", "E4", "E5"]:
        m = metrics[exp]

        # Extract accuracy
        acc = m.get("accuracy", m.get("test_accuracy"))
        # Extract macro metrics
        macro_p = m.get("macro_precision", m.get("test_macro_precision"))
        macro_r = m.get("macro_recall", m.get("test_macro_recall"))
        macro_f1 = m.get("macro_f1", m.get("test_macro_f1"))

        best_ep = m.get("best_epoch")
        best_val_f1 = m.get("best_val_macro_f1")
        train_time = m.get("training_time_sec")

        ds_sizes = m.get("dataset_sizes", {"train": 91121, "validation": 20676, "test": 11966})
        train_size = ds_sizes.get("train", 91121)
        val_size = ds_sizes.get("validation", 20676)
        test_size = ds_sizes.get("test", 11966)

        rows.append({
            "experiment": exp,
            "description": EXP_DESCRIPTIONS[exp],
            "fusion_type": FUSION_TYPES[exp],
            "emoji_init": EMOJI_INITS[exp],
            "accuracy": float(acc),
            "macro_precision": float(macro_p),
            "macro_recall": float(macro_r),
            "macro_f1": float(macro_f1),
            "best_epoch": int(best_ep) if best_ep is not None else None,
            "best_val_macro_f1": float(best_val_f1) if best_val_f1 is not None else None,
            "training_time_sec": float(train_time) if train_time is not None else None,
            "train_size": train_size,
            "val_size": val_size,
            "test_size": test_size,
        })

    df = pd.DataFrame(rows)
    return df


def build_model_rankings(master_df: pd.DataFrame) -> pd.DataFrame:
    """Rank models by Macro F1 and Accuracy and compute deltas relative to E0."""
    e0_row = master_df[master_df["experiment"] == "E0"].iloc[0]
    e0_acc = e0_row["accuracy"]
    e0_f1 = e0_row["macro_f1"]

    df = master_df.copy()

    # Calculate absolute percentage-point difference: (Metric - E0) * 100
    df["delta_acc_pp"] = (df["accuracy"] - e0_acc) * 100
    df["delta_f1_pp"] = (df["macro_f1"] - e0_f1) * 100

    # Calculate relative percentage improvement: ((Metric - E0) / E0) * 100
    df["rel_improvement_acc_pct"] = ((df["accuracy"] - e0_acc) / e0_acc) * 100
    df["rel_improvement_f1_pct"] = ((df["macro_f1"] - e0_f1) / e0_f1) * 100

    # Rankings
    df["rank_by_f1"] = df["macro_f1"].rank(ascending=False, method="min").astype(int)
    df["rank_by_acc"] = df["accuracy"].rank(ascending=False, method="min").astype(int)

    df = df.sort_values(by="rank_by_f1").reset_index(drop=True)
    return df


def plot_master_performance(rankings_df: pd.DataFrame, out_path: Path) -> None:
    """Create performance comparison bar chart with E0 reference line."""
    sns.set_theme(style="whitegrid", font_scale=1.1)
    fig, ax = plt.subplots(figsize=(12, 6))

    # Order by experiment name E0-E5 for clear progression
    plot_df = rankings_df.sort_values("experiment").copy()
    x = np.arange(len(plot_df))
    width = 0.35

    # Acc and Macro F1 bars
    bars1 = ax.bar(x - width/2, plot_df["accuracy"] * 100, width, label="Accuracy (%)", color="#2b5c8f", alpha=0.9)
    bars2 = ax.bar(x + width/2, plot_df["macro_f1"] * 100, width, label="Macro F1 (%)", color="#e27c38", alpha=0.9)

    # Reference lines for E0 baseline
    e0_acc = plot_df[plot_df["experiment"] == "E0"]["accuracy"].values[0] * 100
    e0_f1 = plot_df[plot_df["experiment"] == "E0"]["macro_f1"].values[0] * 100

    ax.axhline(e0_acc, color="#2b5c8f", linestyle="--", linewidth=1.2, alpha=0.7, label=f"E0 Accuracy ({e0_acc:.2f}%)")
    ax.axhline(e0_f1, color="#e27c38", linestyle="--", linewidth=1.2, alpha=0.7, label=f"E0 Macro F1 ({e0_f1:.2f}%)")

    # Annotate bars with values
    for bar in bars1:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.5, f"{yval:.2f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

    for bar in bars2:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.5, f"{yval:.2f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xlabel("Experiment", fontweight="bold", labelpad=10)
    ax.set_ylabel("Score (%)", fontweight="bold", labelpad=10)
    ax.set_title("Master Performance Comparison Across All Six Experiments (E0–E5)", fontsize=14, fontweight="bold", pad=15)
    ax.set_xticks(x)
    labels = [f"{row['experiment']}\n({row['fusion_type'].split()[0]})" for _, row in plot_df.iterrows()]
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(40, 60)
    ax.legend(loc="lower right", frameon=True, framealpha=0.95)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved performance comparison plot -> {out_path}")
