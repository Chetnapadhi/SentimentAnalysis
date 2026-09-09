"""Page 7: Emoji Insights & Frequency Stratification."""

import streamlit as st

from dashboard.components import render_page_header, render_research_alert
from dashboard.data import load_emoji_count_breakdown, load_sentiment_by_emoji_count
from dashboard.visualizations import plot_emoji_scaling


def render():
    render_page_header(
        title="Emoji Insights & Frequency Stratification",
        subtitle="Examining how emoji density modulates sentiment expression and model accuracy.",
    )

    render_research_alert(
        "<strong>Critical Dataset Fact:</strong> Because every canonical test sample contains at least one emoji, an emoji-absent vs emoji-present test comparison is not available. We evaluate performance scaling across emoji density bins (1, 2, 3, 4+ emojis).",
        alert_type="warning",
    )

    count_df = load_emoji_count_breakdown()
    dist_df = load_sentiment_by_emoji_count()

    col1, col2 = st.columns([3, 2])
    with col1:
        st.markdown("### Performance Scaling Curve")
        fig = plot_emoji_scaling(count_df)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### Emoji Density Distribution")
        st.markdown(
            """
            <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 8px; padding: 1rem; color: #334155; font-size: 0.92rem; line-height: 1.6;">
                <div>&bull; <strong>1 Emoji:</strong> 8,357 tweets (<strong>69.8%</strong>)</div>
                <div>&bull; <strong>2 Emojis:</strong> 2,200 tweets (<strong>18.4%</strong>)</div>
                <div>&bull; <strong>3 Emojis:</strong> 797 tweets (<strong>6.7%</strong>)</div>
                <div>&bull; <strong>4+ Emojis:</strong> 612 tweets (<strong>5.1%</strong>)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("### Sentiment Skew by Emoji Density")
        st.markdown(
            """
            <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 8px; padding: 1rem; color: #334155; font-size: 0.92rem; line-height: 1.6;">
                As emoji count increases, the data exhibits natural <strong>sentiment skew</strong>:
                <ul style="margin-bottom: 0; padding-left: 1.2rem;">
                    <li><strong>1 emoji:</strong> 39.9% Bullish &bull; 37.1% Neutral &bull; 23.0% Bearish</li>
                    <li><strong>3 emojis:</strong> 51.9% Bullish &bull; 29.4% Neutral &bull; 18.7% Bearish</li>
                    <li><strong>4+ emojis:</strong> <strong>61.6% Bullish</strong> &bull; 19.3% Neutral &bull; 19.1% Bearish</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("### Incremental F1 Gain of E3 over Text-Only E0")

    # Table displaying scaling
    st.markdown(
        """
        | Emoji Density | E0 Macro F1 | E3 Macro F1 | Absolute $\Delta$ | Relative Gain |
        |---|---|---|---|---|
        | **1 Emoji** ($N=8,357$) | `45.95%` | `50.28%` | **+4.33 pp** | +9.4% |
        | **2 Emojis** ($N=2,200$) | `46.43%` | `52.11%` | **+5.68 pp** | +12.2% |
        | **3 Emojis** ($N=797$) | `43.76%` | `56.25%` | **+12.49 pp** | **+28.5%** |
        | **4+ Emojis** ($N=612$) | `46.62%` | `53.94%` | **+7.32 pp** | +15.7% |
        """
    )
    st.caption("Note: In the 4+ emoji bin, raw accuracy reaches 65.20% (E3) vs 55.88% (E0), partially driven by the increased natural prevalence of Bullish tweets.")
