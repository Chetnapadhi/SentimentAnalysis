"""Page 9: E5 32-Dimensional Gate Analysis."""

import plotly.graph_objects as go
import streamlit as st

from dashboard.components import render_page_header, render_research_alert
from dashboard.data import load_gate_raw_json, load_gate_summary
from dashboard.styles import (
    COLOR_BEARISH,
    COLOR_BULLISH,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    BORDER_LIGHT,
)


def render():
    render_page_header(
        title="E5 Gate Analysis",
        subtitle="Examining the internal feature-level modulation learned by the E5 gating architecture.",
    )

    render_research_alert(
        "<strong>Critical Interpretability Rule:</strong> The E5 gate contains element-wise sigmoid modulation factors in (0, 1)³². <strong>Do NOT say '41% of the prediction comes from emojis' or 'the model pays 41% attention to emojis'.</strong> Do NOT interpret the mean gate as causal attribution.",
        alert_type="warning",
    )

    summary_df = load_gate_summary()
    gate_data = load_gate_raw_json()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Gate Dimensionality", "32 Dimensions")
    with c2:
        st.metric("Overall Mean", f"{gate_data['overall_mean']:.4f}")
    with c3:
        st.metric("Overall Std", f"{gate_data['overall_std']:.4f}")
    with c4:
        st.metric("Overall Median", f"{gate_data['overall_median']:.4f}")

    st.markdown("### Activation Profile Across All 32 Dimensions")
    dims = list(range(32))
    overall_dim_means = gate_data["dimension_means"]
    bearish_dim_means = gate_data["Bearish_dimension_means"]
    neutral_dim_means = gate_data["Neutral_dimension_means"]
    bullish_dim_means = gate_data["Bullish_dimension_means"]

    fig = go.Figure()
    fig.add_trace(go.Bar(x=dims, y=overall_dim_means, name="Overall Mean", marker_color="#3B82F6"))
    fig.add_trace(go.Scatter(x=dims, y=bearish_dim_means, name="Bearish Mean", mode="lines+markers", line=dict(color=COLOR_BEARISH, width=2.5)))
    fig.add_trace(go.Scatter(x=dims, y=bullish_dim_means, name="Bullish Mean", mode="lines+markers", line=dict(color=COLOR_BULLISH, width=2.5)))

    fig.update_layout(
        font=dict(family="Plus Jakarta Sans, sans-serif", color=TEXT_PRIMARY, size=12),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        title=dict(text="<b>32-d Gate Activations per Dimension and Sentiment Class</b>", font=dict(size=14, color=TEXT_PRIMARY)),
        xaxis=dict(title="<b>Gate Dimension Index (0 - 31)</b>", dtick=2, tickfont=dict(color=TEXT_PRIMARY)),
        yaxis=dict(title="<b>Mean Sigmoid Activation (g)</b>", range=[0, 1.0], showgrid=True, gridcolor=BORDER_LIGHT, tickfont=dict(color=TEXT_SECONDARY)),
        height=380,
        margin=dict(l=40, r=20, t=30, b=40),
        legend=dict(font=dict(color=TEXT_PRIMARY)),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Summary Statistics Table")
    disp_summary = summary_df.copy()
    disp_summary.columns = ["Subset", "Mean Activation", "Std Deviation", "Median Activation"]
    st.dataframe(disp_summary, use_container_width=True, hide_index=True)
