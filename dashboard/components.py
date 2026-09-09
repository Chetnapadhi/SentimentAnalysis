"""Reusable UI widgets, page headers, metric cards, and research caveats."""

import streamlit as st
from dashboard.styles import (
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    BORDER_LIGHT,
    COLOR_BULLISH,
    COLOR_BEARISH,
)


def render_page_header(title: str, subtitle: str):
    """Render a prominent, high-contrast page header with consistent styling."""
    st.markdown(
        f"""
        <div style="margin-bottom: 1.25rem;">
            <div class="page-title">{title}</div>
            <div class="page-subtitle">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric_card(title: str, value: str, delta: str | None = None, is_positive: bool = True):
    """Render a clean card with high text contrast and clear labels."""
    delta_html = ""
    if delta:
        delta_class = "metric-delta-pos" if is_positive else "metric-delta-neg"
        arrow = "↑" if is_positive else "↓"
        delta_html = f'<div class="{delta_class}">{arrow} {delta}</div>'
    
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">{title}</div>
            <div class="metric-value">{value}</div>
            {delta_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_research_alert(message: str, alert_type: str = "info"):
    """Render an academic notice / caveat alert box with distinct border and background."""
    st.markdown(
        f"""
        <div class="research-alert {alert_type}">
            {message}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_pipeline_flow():
    """Render the research pipeline architecture flow with clean, high-contrast cards."""
    st.markdown(
        """
        <div class="pipeline-box">
            <div style="text-align: center; font-size: 0.78rem; font-weight: 700; color: #475569; margin-bottom: 1.1rem; text-transform: uppercase; letter-spacing: 0.8px;">
                End-to-End Neural Architecture Flow
            </div>
            <div style="display: flex; flex-wrap: wrap; justify-content: center; align-items: center; gap: 8px;">
                <span class="pipeline-step" style="background: #F1F5F9;">Input Tweet</span>
                <span class="pipeline-arrow">→</span>
                <span class="pipeline-step" style="background: #EFF6FF; border-color: #BFDBFE; color: #1E40AF;">
                    <strong>Extraction:</strong> Text & Emojis
                </span>
                <span class="pipeline-arrow">→</span>
                <div style="display: inline-flex; flex-direction: column; gap: 4px;">
                    <span class="pipeline-step" style="padding: 5px 10px; font-size: 0.8rem; background: #FFFFFF;">
                        <strong>Text:</strong> Frozen BERT (768-d)
                    </span>
                    <span class="pipeline-step" style="padding: 5px 10px; font-size: 0.8rem; background: #FFFFFF;">
                        <strong>Emojis:</strong> 32-d Vectors
                    </span>
                </div>
                <span class="pipeline-arrow">→</span>
                <span class="pipeline-step" style="background: #FEF3C7; border-color: #FDE68A; color: #92400E;">
                    <strong>Fusion:</strong> Attention / Gate (800-d)
                </span>
                <span class="pipeline-arrow">→</span>
                <span class="pipeline-step" style="background: #ECFDF5; border-color: #A7F3D0; color: #065F46; font-weight: 800;">
                    3-Class Prediction
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_footer():
    """Render standardized academic dashboard footer."""
    st.markdown(
        """
        <div class="research-footer">
            <strong>Emoji-Aware Sentiment Analysis</strong> &bull; Controlled E0–E5 Research Benchmark &bull; StockTwits Canonical Split
        </div>
        """,
        unsafe_allow_html=True,
    )
