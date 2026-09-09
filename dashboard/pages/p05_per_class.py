"""Page 5: Per-Class Sentiment Performance."""

import pandas as pd
import plotly.express as px
import streamlit as st

from dashboard.components import render_page_header, render_research_alert
from dashboard.data import load_per_class_metrics
from dashboard.styles import (
    COLOR_BEARISH,
    COLOR_NEUTRAL,
    COLOR_BULLISH,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    BORDER_LIGHT,
)


def render():
    render_page_header(
        title="Per-Class Performance",
        subtitle="Evaluation broken down across Bearish (N=2,608), Neutral (N=4,191), and Bullish (N=5,167) classes.",
    )

    df = load_per_class_metrics()

    metric_type = st.radio("Select Metric to View:", ["F1-score", "Recall", "Precision"], horizontal=True)
    col_map = {"F1-score": "f1_score", "Recall": "recall", "Precision": "precision"}
    val_col = col_map[metric_type]

    # Grouped Bar Chart with High Contrast Styling
    fig = px.bar(
        df,
        x="experiment",
        y=val_col,
        color="class_name",
        barmode="group",
        color_discrete_map={"Bearish": COLOR_BEARISH, "Neutral": COLOR_NEUTRAL, "Bullish": COLOR_BULLISH},
        labels={"experiment": "Experiment", val_col: metric_type, "class_name": "Sentiment Class"},
        text_auto=".3f",
    )

    fig.update_layout(
        font=dict(family="Plus Jakarta Sans, sans-serif", color=TEXT_PRIMARY, size=12),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        margin=dict(l=40, r=20, t=30, b=40),
        yaxis=dict(title=f"<b>{metric_type}</b>", showgrid=True, gridcolor=BORDER_LIGHT, tickfont=dict(color=TEXT_SECONDARY)),
        xaxis=dict(title="<b>Experiment</b>", tickfont=dict(color=TEXT_PRIMARY, size=12)),
        height=380,
        legend=dict(font=dict(color=TEXT_PRIMARY)),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Per-Class Performance Table")
    pivot_f1 = df.pivot(index="experiment", columns="class_name", values="f1_score")[["Bearish", "Neutral", "Bullish"]]
    pivot_f1_disp = (pivot_f1 * 100).map("{:.2f}%".format)
    st.dataframe(pivot_f1_disp, use_container_width=True)

    st.markdown("### Architectural Trade-off: E3 vs E5")
    st.markdown(
        """
        While **E3 (Attention)** is the top-ranked model overall, per-class analysis uncovers a vital complementary dynamic:
        """
    )

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            """
            <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 10px; padding: 1.25rem;">
                <h4 style="color: #059669; margin-top: 0; font-size: 1.05rem;">E3 (Attention Fusion) Strengths</h4>
                <ul style="color: #334155; font-size: 0.92rem; line-height: 1.6;">
                    <li><strong>Dominates Neutral F1:</strong> <code>50.26%</code> vs E5 <code>44.56%</code> (+5.70 pp)</li>
                    <li><strong>Highest Bullish F1:</strong> <code>63.83%</code> vs E5 <code>62.41%</code> (+1.42 pp)</li>
                    <li><strong>Highest Overall Macro F1:</strong> <code>51.60%</code></li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            """
            <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 10px; padding: 1.25rem;">
                <h4 style="color: #2563EB; margin-top: 0; font-size: 1.05rem;">E5 (Gated Fusion) Strengths</h4>
                <ul style="color: #334155; font-size: 0.92rem; line-height: 1.6;">
                    <li><strong>Significantly higher Bearish Recall:</strong> <code>49.81%</code> vs E3 <code>34.28%</code> (<strong>+15.53 pp</strong>)</li>
                    <li><strong>Highest Bearish F1:</strong> <code>45.77%</code> vs E3 <code>40.72%</code> (+5.05 pp)</li>
                    <li>Superior sensitivity to extreme downside market signals.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    render_research_alert(
        "<strong>Trade-off Summary:</strong> E3 is more balanced across the full distribution, but E5's element-wise gating preserves strong signals for minority Bearish tweets that attention downweights.",
        alert_type="info",
    )
