"""Page 1: Overview & Benchmark landing page."""

import streamlit as st
from dashboard.components import (
    render_page_header,
    render_metric_card,
    render_pipeline_flow,
    render_research_alert,
)


def render():
    render_page_header(
        title="Emoji-Aware Sentiment Analysis",
        subtitle="Controlled empirical study investigating whether explicit emoji modeling provides predictive signal beyond frozen text-only BERT embeddings.",
    )

    # Core Research Question Card
    st.markdown(
        """
        <div class="rq-card">
            <div style="font-size: 0.8rem; font-weight: 800; color: #2563EB; text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 0.4rem;">
                Core Research Question
            </div>
            <div style="font-size: 1.18rem; font-weight: 700; color: #0F172A; line-height: 1.45;">
                "For social-media sentiment analysis, does explicitly modeling emojis and fusing emoji representations with text representations improve sentiment classification compared with a text-only model?"
            </div>
            <div style="font-size: 0.88rem; color: #475569; margin-top: 0.5rem;">
                Evaluated on 11,966 canonical StockTwits financial microblog posts across 3 sentiment classes (Bearish, Neutral, Bullish).
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Dataset Summary Metrics
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_metric_card("Test Split (Evaluated)", "11,966", "Untouched Canonical Split", is_positive=True)
    with c2:
        render_metric_card("Training Split", "91,121", "Train-only Vocabulary", is_positive=True)
    with c3:
        render_metric_card("Validation Split", "20,676", "Checkpoint Selection", is_positive=True)
    with c4:
        render_metric_card("Controlled Architectures", "6 Models", "E0 through E5 Benchmark", is_positive=True)

    st.markdown("### The Controlled Study Matrix (E0 – E5)")
    st.markdown(
        "<p style='color: #475569; font-size: 0.94rem; margin-top: -0.3rem;'>All models use the exact same frozen BERT encoder (768-d), identical training splits, and fixed random seed (42). Only emoji initialization and fusion mechanics vary:</p>",
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            """
            <div class="exp-card">
                <div>
                    <span class="exp-badge badge-regular">E0 &bull; Baseline</span>
                    <h4 style="margin: 0.5rem 0 0.2rem 0; font-size: 1.1rem; color: #0F172A;">Text-Only Baseline</h4>
                    <p style="font-size: 0.86rem; color: #475569; line-height: 1.5;">
                        Frozen BERT (768-d) with mean pooling. Emojis are completely stripped so the model is emoji-blind.
                    </p>
                </div>
                <div style="margin-top: 0.9rem; padding-top: 0.7rem; border-top: 1px solid #E2E8F0; font-size: 0.92rem; color: #0F172A;">
                    <strong>Macro F1: 46.21%</strong> &bull; Acc: 47.69%
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
        st.markdown(
            """
            <div class="exp-card">
                <div>
                    <span class="exp-badge badge-regular">E1 &bull; Concatenation</span>
                    <h4 style="margin: 0.5rem 0 0.2rem 0; font-size: 1.1rem; color: #0F172A;">Random + Concat</h4>
                    <p style="font-size: 0.86rem; color: #475569; line-height: 1.5;">
                        Random 32-d emoji embeddings concatenated to BERT. Acts as noise, degrading performance below baseline.
                    </p>
                </div>
                <div style="margin-top: 0.9rem; padding-top: 0.7rem; border-top: 1px solid #E2E8F0; font-size: 0.92rem; color: #DC2626;">
                    <strong>Macro F1: 43.35%</strong> &bull; Acc: 44.83% (-2.86 pp)
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
            <div class="exp-card">
                <div>
                    <span class="exp-badge badge-regular">E2 &bull; Pretrained</span>
                    <h4 style="margin: 0.5rem 0 0.2rem 0; font-size: 1.1rem; color: #0F172A;">Pretrained + Concat</h4>
                    <p style="font-size: 0.86rem; color: #475569; line-height: 1.5;">
                        Partially pretrained (TweetEval) 32-d emoji embeddings concatenated to BERT. Regularizes naive concat.
                    </p>
                </div>
                <div style="margin-top: 0.9rem; padding-top: 0.7rem; border-top: 1px solid #E2E8F0; font-size: 0.92rem; color: #0F172A;">
                    <strong>Macro F1: 50.41%</strong> &bull; Acc: 52.05% (+4.19 pp)
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
        st.markdown(
            """
            <div class="exp-card best">
                <div>
                    <span class="exp-badge badge-best">★ RANK 1 &bull; BEST MODEL</span>
                    <h4 style="margin: 0.5rem 0 0.2rem 0; font-size: 1.1rem; color: #065F46;">Random + Attention (E3)</h4>
                    <p style="font-size: 0.86rem; color: #047857; line-height: 1.5;">
                        Text-conditioned scaled dot-product attention over trainable emoji sequence. Highest overall benchmark score.
                    </p>
                </div>
                <div style="margin-top: 0.9rem; padding-top: 0.7rem; border-top: 1px solid #A7F3D0; font-size: 0.95rem; font-weight: 800; color: #065F46;">
                    Macro F1: 51.60% &bull; Acc: 54.20% (+5.39 pp)
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            """
            <div class="exp-card">
                <div>
                    <span class="exp-badge badge-regular">E4 &bull; Pretrained</span>
                    <h4 style="margin: 0.5rem 0 0.2rem 0; font-size: 1.1rem; color: #0F172A;">Pretrained + Attention</h4>
                    <p style="font-size: 0.86rem; color: #475569; line-height: 1.5;">
                        Same attention architecture as E3 with frozen TweetEval emoji embeddings.
                    </p>
                </div>
                <div style="margin-top: 0.9rem; padding-top: 0.7rem; border-top: 1px solid #E2E8F0; font-size: 0.92rem; color: #0F172A;">
                    <strong>Macro F1: 50.53%</strong> &bull; Acc: 52.92% (+4.31 pp)
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
        st.markdown(
            """
            <div class="exp-card">
                <div>
                    <span class="exp-badge badge-regular">E5 &bull; Gated Fusion</span>
                    <h4 style="margin: 0.5rem 0 0.2rem 0; font-size: 1.1rem; color: #0F172A;">Pretrained + 32-d Gate</h4>
                    <p style="font-size: 0.86rem; color: #475569; line-height: 1.5;">
                        Learned 32-d sigmoid gate modulates emoji features. Achieves highest recall on Bearish sentiment.
                    </p>
                </div>
                <div style="margin-top: 0.9rem; padding-top: 0.7rem; border-top: 1px solid #E2E8F0; font-size: 0.92rem; color: #0F172A;">
                    <strong>Macro F1: 50.91%</strong> &bull; Acc: 52.33% (+4.70 pp)
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    render_pipeline_flow()

    render_research_alert(
        "<strong>Scientific Rigor Confirmation:</strong> All 6 models were trained on 91,121 examples, tuned on 20,676 validation examples, and evaluated on the untouched canonical test split (N = 11,966). All checkpoint weights and official metrics are permanently locked.",
        alert_type="info",
    )
