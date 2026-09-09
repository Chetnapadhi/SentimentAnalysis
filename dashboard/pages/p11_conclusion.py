"""Page 11: Research Conclusions and Methodological Limitations."""

import streamlit as st
from dashboard.components import render_page_header, render_research_alert


def render():
    render_page_header(
        title="Research Conclusions",
        subtitle="Empirical findings, architectural insights, and methodological limitations for evaluation.",
    )

    st.markdown("### Key Findings")
    st.markdown(
        """
        <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 12px; padding: 1.5rem; color: #334155; line-height: 1.7;">
            <ol style="margin-bottom: 0; padding-left: 1.3rem;">
                <li style="margin-bottom: 0.6rem;">
                    <strong>Explicit emoji modeling improved over the text-only baseline in several configurations.</strong><br>
                    Adding emoji representations through text-conditioned attention (E3) or gated modulation (E5) captures non-redundant sentiment signals that frozen word representations alone fail to resolve.
                </li>
                <li style="margin-bottom: 0.6rem;">
                    <strong>E3 achieved the highest Macro-F1: 51.60% and Accuracy: 54.20%.</strong><br>
                    Text-conditioned attention fusion (E3) ranked first among all six evaluated models on the canonical test split.
                </li>
                <li style="margin-bottom: 0.6rem;">
                    <strong>E3 improved over E0 by:</strong><br>
                    &bull; <strong>+6.51 percentage points in Accuracy</strong> (47.69% ➔ 54.20%)<br>
                    &bull; <strong>+5.39 percentage points in Macro-F1</strong> (46.21% ➔ 51.60%)
                </li>
                <li>
                    <strong>Performance depends on both emoji representation and fusion mechanism:</strong><br>
                    Simply concatenating random emoji representations (E1) degrades accuracy below the text-only baseline (-2.86 pp F1). Effective fusion requires either structural attention (E3) or pretraining + gating (E2, E5).
                </li>
            </ol>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Methodological Scope & Limitations")

    render_research_alert(
        """
        <strong>1. StockTwits-Specific Benchmark:</strong> Findings reflect financial microblog dynamics on StockTwits, where emojis frequently represent contrarian or sardonic sentiment. Generalization to open-domain conversation requires further study.<br><br>
        <strong>2. Test Set Emoji Prevalence:</strong> All 11,966 canonical test examples contain at least one emoji. Consequently, this benchmark evaluates performance on emoji-bearing posts, rather than providing an emoji-free vs emoji-containing test split.<br><br>
        <strong>3. Partially Pretrained Embeddings:</strong> E2, E4, and E5 use partially pretrained emoji embeddings: 19 overlapping TweetEval emojis are pretrained and remaining StockTwits emojis are randomly initialized.<br><br>
        <strong>4. Significance Testing:</strong> No formal statistical significance testing was performed; metrics reflect deterministic single-seed (42) controlled evaluation across all 11,966 test examples.<br><br>
        <strong>5. Live Inference Scope:</strong> Arbitrary live predictions on user-submitted inputs do not have ground-truth labels and represent empirical model inference rather than verified truth.
        """,
        alert_type="info",
    )
