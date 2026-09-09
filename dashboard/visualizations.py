"""Consistent, accessible, high-contrast Plotly chart helpers for academic presentation."""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from dashboard.styles import (
    COLOR_BEARISH,
    COLOR_BULLISH,
    COLOR_NEUTRAL,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    BORDER_LIGHT,
)


def plot_probability_bars(probs: dict[str, float]) -> go.Figure:
    """Create an accessible, clean horizontal bar chart for predicted sentiment probabilities."""
    # Intuitive real-world sentiment labels with original names
    labels = ["Negative", "Neutral", "Positive"]
    # map from Bearish/Neutral/Bullish
    values = [
        probs.get("Bearish", 0.0) * 100,
        probs.get("Neutral", 0.0) * 100,
        probs.get("Bullish", 0.0) * 100,
    ]
    colors = [COLOR_BEARISH, COLOR_NEUTRAL, COLOR_BULLISH]

    fig = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker=dict(color=colors, line=dict(width=1, color=BORDER_LIGHT)),
            text=[f"<b>{v:.1f}%</b>" for v in values],
            textposition="inside",
            insidetextanchor="middle",
            textfont=dict(color="white", size=14, family="Plus Jakarta Sans, sans-serif"),
        )
    )

    fig.update_layout(
        font=dict(family="Plus Jakarta Sans, sans-serif", color=TEXT_PRIMARY, size=12),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        xaxis=dict(
            range=[0, 100],
            title="<b>Predicted Probability (%)</b>",
            showgrid=True,
            gridcolor=BORDER_LIGHT,
            tickfont=dict(color=TEXT_SECONDARY, size=11),
        ),
        yaxis=dict(
            autorange="reversed",
            tickfont=dict(color=TEXT_PRIMARY, size=13, family="Plus Jakarta Sans, sans-serif"),
        ),
        height=180,
        margin=dict(l=10, r=20, t=10, b=20),
    )
    return fig


def plot_metric_comparison_bars(
    master_df: pd.DataFrame,
    metric_col: str,
    metric_title: str,
) -> go.Figure:
    """Bar chart comparing E0-E5 across a selected metric with distinct highlight on E3."""
    df = master_df.copy()
    is_pct = "accuracy" in metric_col.lower() or "acc" in metric_col.lower()
    
    multiplier = 100 if is_pct or df[metric_col].max() <= 1.0 else 1.0
    values = df[metric_col] * multiplier
    
    # Palette: E3 is vibrant Emerald, E0 is neutral Slate, E1 is muted red, others are Steel Blue
    colors = []
    for exp in df["experiment"]:
        if exp == "E3":
            colors.append("#059669")  # Emerald 600
        elif exp == "E0":
            colors.append("#475569")  # Slate 600
        elif exp == "E1":
            colors.append("#DC2626")  # Red 600 (Degraded)
        else:
            colors.append("#2563EB")  # Blue 600

    fig = go.Figure(
        go.Bar(
            x=df["experiment"],
            y=values,
            marker=dict(color=colors, line=dict(width=1, color=BORDER_LIGHT)),
            text=[f"<b>{v:.2f}%</b>" if multiplier == 100 else f"<b>{v:.4f}</b>" for v in values],
            textposition="outside",
            textfont=dict(color=TEXT_PRIMARY, size=12),
        )
    )

    fig.update_layout(
        font=dict(family="Plus Jakarta Sans, sans-serif", color=TEXT_PRIMARY, size=12),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        title=dict(text=f"<b>Controlled Model Comparison: {metric_title}</b>", font=dict(size=15, color=TEXT_PRIMARY)),
        yaxis=dict(title=f"<b>{metric_title}</b>", showgrid=True, gridcolor=BORDER_LIGHT, tickfont=dict(color=TEXT_SECONDARY)),
        xaxis=dict(title="<b>Experiment</b>", tickfont=dict(color=TEXT_PRIMARY, size=12)),
        height=360,
        margin=dict(l=40, r=20, t=40, b=40),
    )
    return fig


def plot_interactive_confusion_matrix(cm_data: list[list[float]], exp_name: str) -> go.Figure:
    """Plot an interactive row-normalized confusion matrix heatmap with high text contrast."""
    labels = ["Negative", "Neutral", "Positive"]
    z_data = np.array(cm_data) * 100

    fig = px.imshow(
        z_data,
        labels=dict(x="Predicted Class", y="Actual Class", color="Rate (%)"),
        x=labels,
        y=labels,
        text_auto=".1f",
        color_continuous_scale="Blues",
        range_color=[0, 100],
    )

    fig.update_traces(
        textfont=dict(size=14, color="#0F172A"),
    )

    fig.update_layout(
        font=dict(family="Plus Jakarta Sans, sans-serif", color=TEXT_PRIMARY, size=12),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        title=dict(text=f"<b>Normalized Confusion Matrix ({exp_name})</b>", font=dict(size=15, color=TEXT_PRIMARY)),
        xaxis=dict(tickfont=dict(color=TEXT_PRIMARY, size=12), title=dict(font=dict(color=TEXT_PRIMARY))),
        yaxis=dict(tickfont=dict(color=TEXT_PRIMARY, size=12), title=dict(font=dict(color=TEXT_PRIMARY))),
        height=360,
        margin=dict(l=40, r=20, t=40, b=40),
    )
    return fig


def plot_emoji_scaling(count_df: pd.DataFrame) -> go.Figure:
    """Plot performance scaling curves by emoji count."""
    fig = go.Figure()
    models = ["E0", "E2", "E3", "E5"]
    palette = {"E0": "#475569", "E2": "#2563EB", "E3": "#059669", "E5": "#D97706"}

    for m in models:
        sub = count_df[count_df["model"] == m]
        is_key = m in ["E0", "E3"]
        fig.add_trace(
            go.Scatter(
                x=sub["emoji_bin"],
                y=sub["macro_f1"],
                mode="lines+markers",
                name=f"<b>{m}</b>",
                line=dict(color=palette.get(m, "#94A3B8"), width=3.5 if is_key else 2),
                marker=dict(size=9 if is_key else 7),
            )
        )

    fig.update_layout(
        font=dict(family="Plus Jakarta Sans, sans-serif", color=TEXT_PRIMARY, size=12),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        title=dict(text="<b>Macro F1 Scaling by Emoji Count (E0 vs Emoji Models)</b>", font=dict(size=15, color=TEXT_PRIMARY)),
        xaxis=dict(title="<b>Emoji Count</b>", tickfont=dict(color=TEXT_PRIMARY, size=12)),
        yaxis=dict(title="<b>Macro F1 Score</b>", showgrid=True, gridcolor=BORDER_LIGHT, tickfont=dict(color=TEXT_SECONDARY)),
        height=360,
        margin=dict(l=40, r=20, t=40, b=40),
        legend=dict(font=dict(color=TEXT_PRIMARY)),
    )
    return fig
