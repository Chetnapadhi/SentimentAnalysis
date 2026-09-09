"""Analysis of model performance stratified by emoji frequency and sentiment distribution."""

from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from src.analysis.data_loader import LABEL_NAMES


def evaluate_slice(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute accuracy and macro metrics on a subset."""
    if len(y_true) == 0:
        return {"accuracy": 0.0, "macro_precision": 0.0, "macro_recall": 0.0, "macro_f1": 0.0, "count": 0}
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "count": len(y_true),
    }


def analyze_emoji_presence(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze performance on Single Emoji vs Multiple Emojis."""
    rows = []
    models = ["e0", "e1", "e2", "e3", "e4", "e5"]
    model_names = {
        "e0": "E0 (Text-only)",
        "e1": "E1 (Concat Random)",
        "e2": "E2 (Concat Pretrained)",
        "e3": "E3 (Attention Random)",
        "e4": "E4 (Attention Pretrained)",
        "e5": "E5 (Gated Pretrained)",
    }

    subsets = {
        "All Test Examples": df,
        "Single Emoji (num_emojis == 1)": df[df["num_emojis"] == 1],
        "Multiple Emojis (num_emojis >= 2)": df[df["num_emojis"] >= 2],
    }

    for subset_name, sub_df in subsets.items():
        y_true = sub_df["label"].to_numpy()
        for m in models:
            y_pred = sub_df[f"{m}_pred"].to_numpy()
            metrics = evaluate_slice(y_true, y_pred)
            rows.append({
                "subset": subset_name,
                "model_key": m.upper(),
                "model_name": model_names[m],
                "sample_count": metrics["count"],
                "accuracy": metrics["accuracy"],
                "macro_precision": metrics["macro_precision"],
                "macro_recall": metrics["macro_recall"],
                "macro_f1": metrics["macro_f1"],
            })

    res_df = pd.DataFrame(rows)
    return res_df


def analyze_by_emoji_count(df: pd.DataFrame) -> pd.DataFrame:
    """Breakdown performance by discrete emoji counts: 1, 2, 3, 4+."""
    rows = []
    models = ["e0", "e1", "e2", "e3", "e4", "e5"]

    df = df.copy()
    df["emoji_bin"] = df["num_emojis"].apply(
        lambda n: "1 emoji" if n == 1 else ("2 emojis" if n == 2 else ("3 emojis" if n == 3 else "4+ emojis"))
    )

    for bin_name in ["1 emoji", "2 emojis", "3 emojis", "4+ emojis"]:
        sub_df = df[df["emoji_bin"] == bin_name]
        y_true = sub_df["label"].to_numpy()

        for m in models:
            y_pred = sub_df[f"{m}_pred"].to_numpy()
            metrics = evaluate_slice(y_true, y_pred)
            rows.append({
                "emoji_bin": bin_name,
                "model": m.upper(),
                "sample_count": metrics["count"],
                "accuracy": metrics["accuracy"],
                "macro_precision": metrics["macro_precision"],
                "macro_recall": metrics["macro_recall"],
                "macro_f1": metrics["macro_f1"],
            })

    return pd.DataFrame(rows)


def analyze_sentiment_by_emoji_count(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate sentiment class distribution across different emoji counts."""
    df = df.copy()
    df["emoji_bin"] = df["num_emojis"].apply(
        lambda n: "1 emoji" if n == 1 else ("2 emojis" if n == 2 else ("3 emojis" if n == 3 else "4+ emojis"))
    )

    rows = []
    for bin_name in ["1 emoji", "2 emojis", "3 emojis", "4+ emojis", "All"]:
        sub = df if bin_name == "All" else df[df["emoji_bin"] == bin_name]
        total = len(sub)
        counts = sub["label"].value_counts()
        bearish_pct = (counts.get(0, 0) / total) * 100
        neutral_pct = (counts.get(1, 0) / total) * 100
        bullish_pct = (counts.get(2, 0) / total) * 100

        rows.append({
            "emoji_bin": bin_name,
            "total_count": total,
            "bearish_count": counts.get(0, 0),
            "neutral_count": counts.get(1, 0),
            "bullish_count": counts.get(2, 0),
            "bearish_pct": bearish_pct,
            "neutral_pct": neutral_pct,
            "bullish_pct": bullish_pct,
        })

    return pd.DataFrame(rows)


def plot_emoji_presence_comparison(presence_df: pd.DataFrame, out_path: Path) -> None:
    """Plot grouped bar chart comparing Macro F1 on Single vs Multiple Emojis."""
    sns.set_theme(style="whitegrid", font_scale=1.05)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5), sharey=True)

    single_df = presence_df[presence_df["subset"] == "Single Emoji (num_emojis == 1)"].sort_values("model_key")
    multi_df = presence_df[presence_df["subset"] == "Multiple Emojis (num_emojis >= 2)"].sort_values("model_key")

    colors = ["#7f7f7f", "#bcbd22", "#17becf", "#1f77b4", "#9467bd", "#2ca02c"]

    # 1. Single Emoji Plot
    bars1 = ax1.bar(single_df["model_key"], single_df["macro_f1"] * 100, color=colors, alpha=0.9)
    e0_single_f1 = single_df[single_df["model_key"] == "E0"]["macro_f1"].values[0] * 100
    ax1.axhline(e0_single_f1, color="#7f7f7f", linestyle="--", linewidth=1.2, label=f"E0 Baseline ({e0_single_f1:.2f}%)")

    for bar in bars1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 0.3, f"{yval:.2f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

    count_single = single_df["sample_count"].iloc[0]
    ax1.set_title(f"Single Emoji Tweets (N = {count_single:,}, 69.8%)", fontsize=12, fontweight="bold", pad=8)
    ax1.set_xlabel("Experiment", fontweight="bold")
    ax1.set_ylabel("Macro F1 (%)", fontweight="bold")
    ax1.set_ylim(40, 60)
    ax1.legend(loc="lower right", framealpha=0.95)

    # 2. Multiple Emojis Plot
    bars2 = ax2.bar(multi_df["model_key"], multi_df["macro_f1"] * 100, color=colors, alpha=0.9)
    e0_multi_f1 = multi_df[multi_df["model_key"] == "E0"]["macro_f1"].values[0] * 100
    ax2.axhline(e0_multi_f1, color="#7f7f7f", linestyle="--", linewidth=1.2, label=f"E0 Baseline ({e0_multi_f1:.2f}%)")

    for bar in bars2:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 0.3, f"{yval:.2f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

    count_multi = multi_df["sample_count"].iloc[0]
    ax2.set_title(f"Multiple Emoji Tweets (N = {count_multi:,}, 30.2%)", fontsize=12, fontweight="bold", pad=8)
    ax2.set_xlabel("Experiment", fontweight="bold")
    ax2.set_ylim(40, 60)
    ax2.legend(loc="lower right", framealpha=0.95)

    plt.suptitle("Model Macro-F1 Stratified by Emoji Frequency: Single vs Multiple Emojis", fontsize=14, fontweight="bold", y=1.0)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved emoji presence comparison plot -> {out_path}")
