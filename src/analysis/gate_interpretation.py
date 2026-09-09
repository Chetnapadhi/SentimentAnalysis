"""Analysis and visualization of E5 32-dimensional gate activations."""

from __future__ import annotations

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.analysis.data_loader import get_project_root


def load_gate_data() -> dict:
    """Load gate_analysis.json from results/E5."""
    root = get_project_root()
    path = root / "results" / "E5" / "gate_analysis.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing gate analysis file at {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_gate_summary_table(gate_data: dict) -> pd.DataFrame:
    """Build summary table of gate statistics overall and per sentiment class."""
    rows = [
        {
            "subset": "Overall",
            "mean": gate_data["overall_mean"],
            "std": gate_data["overall_std"],
            "median": gate_data["overall_median"],
        },
        {
            "subset": "Bearish",
            "mean": gate_data["Bearish_mean"],
            "std": gate_data["Bearish_std"],
            "median": gate_data["Bearish_median"],
        },
        {
            "subset": "Neutral",
            "mean": gate_data["Neutral_mean"],
            "std": gate_data["Neutral_std"],
            "median": gate_data["Neutral_median"],
        },
        {
            "subset": "Bullish",
            "mean": gate_data["Bullish_mean"],
            "std": gate_data["Bullish_std"],
            "median": gate_data["Bullish_median"],
        },
    ]
    return pd.DataFrame(rows)


def plot_gate_analysis(gate_data: dict, out_path: Path) -> None:
    """Plot a 3-panel figure: 32-dim activation profile, per-class profiles, and value histogram."""
    sns.set_theme(style="whitegrid", font_scale=1.0)
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.1])

    # 1. Top Panel: 32-Dimensional Overall Gate Profile
    ax1 = fig.add_subplot(gs[0, :])
    dim_means = np.array(gate_data["dimension_means"])
    dims = np.arange(1, 33)

    # Color code by activation strength
    norm = plt.Normalize(vmin=dim_means.min(), vmax=dim_means.max())
    colors = plt.cm.viridis(norm(dim_means))

    bars = ax1.bar(dims, dim_means, color=colors, edgecolor="black", linewidth=0.5, alpha=0.9)
    ax1.axhline(gate_data["overall_mean"], color="red", linestyle="--", linewidth=1.5, label=f"Global Mean ({gate_data['overall_mean']:.4f})")
    ax1.set_xlabel("Gate Dimension index (1 to 32)", fontweight="bold", labelpad=8)
    ax1.set_ylabel("Mean Gate Value", fontweight="bold")
    ax1.set_title("E5 Gate Profile: Mean Activation Across 32 Emoji Embedding Dimensions", fontsize=13, fontweight="bold", pad=10)
    ax1.set_xticks(dims)
    ax1.set_ylim(0.1, 0.7)
    ax1.legend(loc="upper right", framealpha=0.95)

    # Add annotations for top and bottom dimensions
    top_dim = int(np.argmax(dim_means))
    bottom_dim = int(np.argmin(dim_means))
    ax1.annotate(f"Max: Dim {top_dim+1} ({dim_means[top_dim]:.3f})", xy=(top_dim+1, dim_means[top_dim]),
                 xytext=(top_dim+1, dim_means[top_dim]+0.06),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5),
                 fontsize=9, fontweight="bold", ha="center")
    ax1.annotate(f"Min: Dim {bottom_dim+1} ({dim_means[bottom_dim]:.3f})", xy=(bottom_dim+1, dim_means[bottom_dim]),
                 xytext=(bottom_dim+1, dim_means[bottom_dim]-0.06),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5),
                 fontsize=9, fontweight="bold", ha="center")

    # 2. Bottom Left Panel: Per-Class 32-d Comparison
    ax2 = fig.add_subplot(gs[1, 0])
    bearish_dims = np.array(gate_data["Bearish_dimension_means"])
    neutral_dims = np.array(gate_data["Neutral_dimension_means"])
    bullish_dims = np.array(gate_data["Bullish_dimension_means"])

    ax2.plot(dims, bearish_dims, marker="o", markersize=4, label="Bearish", color="#c94c4c", linewidth=1.5)
    ax2.plot(dims, neutral_dims, marker="s", markersize=4, label="Neutral", color="#8d99ae", linewidth=1.5)
    ax2.plot(dims, bullish_dims, marker="^", markersize=4, label="Bullish", color="#2b9348", linewidth=1.5)

    ax2.set_xlabel("Gate Dimension index (1 to 32)", fontweight="bold")
    ax2.set_ylabel("Mean Gate Value", fontweight="bold")
    ax2.set_title("Per-Class Gate Profiles Across Dimensions", fontsize=12, fontweight="bold", pad=10)
    ax2.set_xticks(range(1, 33, 2))
    ax2.set_ylim(0.1, 0.7)
    ax2.legend(loc="lower right", framealpha=0.95)

    # 3. Bottom Right Panel: Histogram Distribution of All Gate Values
    ax3 = fig.add_subplot(gs[1, 1])
    counts = np.array(gate_data["histogram_counts"])
    edges = np.array(gate_data["histogram_edges"])
    bin_centers = (edges[:-1] + edges[1:]) / 2
    bin_width = edges[1] - edges[0]

    ax3.bar(bin_centers, counts, width=bin_width*0.9, color="#4361ee", edgecolor="black", alpha=0.8)
    ax3.axvline(gate_data["overall_mean"], color="red", linestyle="--", linewidth=1.5, label=f"Mean ({gate_data['overall_mean']:.3f})")
    ax3.axvline(gate_data["overall_median"], color="green", linestyle=":", linewidth=1.5, label=f"Median ({gate_data['overall_median']:.3f})")

    ax3.set_xlabel("Gate Value g in [0, 1]", fontweight="bold")
    ax3.set_ylabel("Count (All Elements Across Test Set)", fontweight="bold")
    ax3.set_title("Distribution of All Gate Activations (N = 11,966 x 32)", fontsize=12, fontweight="bold", pad=10)
    ax3.set_xlim(0, 1)
    ax3.legend(loc="upper right", framealpha=0.95)

    plt.suptitle("E5 Gated Fusion Analysis (32-Dimensional Learned Gate)", fontsize=16, fontweight="bold", y=0.99)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved E5 gate analysis plot -> {out_path}")
