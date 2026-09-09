"""Page 2: Live Sentiment Analyzer with clear visual hierarchy, zero-emoji and UNK handling."""

import streamlit as st
from dashboard.components import render_page_header, render_research_alert
from dashboard.inference import run_live_inference, load_cached_emoji_vocab
from dashboard.visualizations import plot_probability_bars
from dashboard.styles import (
    COLOR_BEARISH,
    COLOR_BEARISH_BG,
    COLOR_NEUTRAL,
    COLOR_NEUTRAL_BG,
    COLOR_BULLISH,
    COLOR_BULLISH_BG,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    BORDER_LIGHT,
)


SAMPLE_PROMPTS = [
    "This stock is going to the moon 🚀🔥",
    "Earnings were terrible, dump everything 📉🐻",
    "Holding steady through the volatility, wait and see 🤔📊",
    "Company posted record revenue but margins compressed 🤷‍♂️",
    "Great job management, totally didn't lose half our money 🤡💸",
    "This company is fundamentally sound and cheap.",  # Zero-emoji sample
    "Target upgraded to 150 🪐✨",  # Out of vocab sample
]


def render():
    render_page_header(
        title="Live Sentiment Analyzer",
        subtitle="Real-world text & emoji sentiment inference using our benchmark-winning neural model (E3).",
    )

    # Input card container
    st.markdown(
        """
        <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 12px; padding: 1.4rem; box-shadow: 0 1px 3px rgba(0,0,0,0.03); margin-bottom: 1.25rem;">
        """,
        unsafe_allow_html=True,
    )

    col_sel, col_model = st.columns([3, 1])
    with col_sel:
        chosen_sample = st.selectbox(
            "Select a real-world example or type your own below:",
            ["(Custom input)"] + SAMPLE_PROMPTS,
            index=1,
        )
    with col_model:
        selected_model = st.selectbox(
            "Select Model:",
            ["E3 (Attention - Best)", "E0 (Text-only)", "E5 (Gated)"],
            index=0,
        )
        model_key = selected_model.split()[0]

    default_text = "" if chosen_sample == "(Custom input)" else chosen_sample
    user_input = st.text_area(
        "Enter sentence or social-media post:",
        value=default_text,
        height=85,
        placeholder="e.g., Target upgrade coming tomorrow morning 🚀🚀",
    )

    analyze_btn = st.button("Analyze Sentiment", type="primary", use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

    if analyze_btn or user_input:
        if not user_input.strip():
            st.warning("Please enter a sentence or social-media post.")
            return

        with st.spinner("Analyzing text & emojis..."):
            try:
                res = run_live_inference(user_input, model_name=model_key)
            except Exception as e:
                st.error(f"Inference error: {str(e)}")
                return

        # Vocabulary check for UNK
        vocab = load_cached_emoji_vocab()
        detected_emojis = res["emojis"]
        oov_emojis = [e for e in detected_emojis if e not in vocab["emoji_to_id"]]

        # Section 1: Prominent Result Card & Probabilities
        st.markdown("### Prediction & Confidence")
        col_res, col_chart = st.columns([1, 2])

        with col_res:
            label = res["pred_label"]
            short_label = res.get("short_label", label)
            conf = res["confidence"] * 100

            if label == "Bullish":
                b_color = COLOR_BULLISH
                bg_color = COLOR_BULLISH_BG
                sub_desc = "Positive Sentiment (Optimistic / Bullish)"
            elif label == "Bearish":
                b_color = COLOR_BEARISH
                bg_color = COLOR_BEARISH_BG
                sub_desc = "Negative Sentiment (Pessimistic / Bearish)"
            else:
                b_color = COLOR_NEUTRAL
                bg_color = COLOR_NEUTRAL_BG
                sub_desc = "Neutral Sentiment (Balanced / Ambivalent)"

            st.markdown(
                f"""
                <div style="background: {bg_color}; border: 2px solid {b_color}; border-radius: 12px; padding: 1.5rem; text-align: center; height: 100%; display: flex; flex-direction: column; justify-content: center;">
                    <div style="font-size: 0.78rem; color: {b_color}; font-weight: 800; text-transform: uppercase; letter-spacing: 0.8px;">PREDICTED SENTIMENT</div>
                    <div style="font-size: 2.2rem; font-weight: 800; color: {b_color}; margin: 0.2rem 0;">{short_label.upper()}</div>
                    <div style="font-size: 0.88rem; color: #475569; font-weight: 600;">{sub_desc}</div>
                    <div style="font-size: 1.05rem; color: #1E293B; font-weight: 700; margin-top: 0.4rem;">Confidence: {conf:.1f}%</div>
                    <div style="font-size: 0.76rem; color: #64748B; margin-top: 0.2rem;">Model: <strong>{model_key}</strong></div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col_chart:
            fig = plot_probability_bars(res["probs"])
            st.plotly_chart(fig, use_container_width=True)

        # Section 2: How Emojis Changed the Sentence
        st.markdown("### How Emojis Added Sentiment to the Text")
        c1, c2, c3 = st.columns([2, 1, 1])
        with c1:
            st.markdown(
                f"""
                <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 8px; padding: 1rem;">
                    <div style="font-size: 0.78rem; font-weight: 700; color: #475569; text-transform: uppercase;">Words Alone (Text Without Emojis)</div>
                    <div style="font-size: 0.95rem; font-weight: 600; color: #0F172A; margin-top: 0.3rem;">"{res['text_without_emoji']}"</div>
                    <div style="font-size: 0.78rem; color: #64748B; margin-top: 0.3rem;">What traditional text-only AI reads.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                f"""
                <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 8px; padding: 1rem;">
                    <div style="font-size: 0.78rem; font-weight: 700; color: #475569; text-transform: uppercase;">Detected Emojis</div>
                    <div style="font-size: 1.2rem; font-weight: 700; color: #0F172A; margin-top: 0.2rem;">{res['emojis'] if res['emojis'] else 'None'}</div>
                    <div style="font-size: 0.78rem; color: #64748B; margin-top: 0.3rem;">Count: <strong>{res['num_emojis']}</strong></div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with c3:
            status_text = "Standard" if not oov_emojis else "Contains &lt;UNK&gt;"
            st.markdown(
                f"""
                <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 8px; padding: 1rem;">
                    <div style="font-size: 0.78rem; font-weight: 700; color: #475569; text-transform: uppercase;">Emoji Embedding</div>
                    <div style="font-size: 0.95rem; font-weight: 700; color: #0F172A; margin-top: 0.3rem;">{status_text}</div>
                    <div style="font-size: 0.78rem; color: #64748B; margin-top: 0.3rem;">32-dimensional vector</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Explanatory Notes for Zero-Emoji and OOV
        if res["num_emojis"] == 0:
            render_research_alert(
                "<strong>Zero-Emoji Handling:</strong> Because no emojis were detected, the model did not execute attention over empty vectors. Instead, it routed via the learned <code>emoji_absent</code> fallback parameter [32-d] to produce a stable prediction.",
                alert_type="info",
            )
        elif oov_emojis:
            render_research_alert(
                f"<strong>Out-of-Vocabulary Emojis:</strong> The emoji(s) {oov_emojis} were not in the 1,610-token training vocabulary and were safely mapped to the <code>&lt;UNK&gt;</code> token (index 0).",
                alert_type="warning",
            )

        # Section 3: Model Interpretation
        st.markdown("### Model Interpretation")
        if model_key == "E3":
            st.markdown(
                """
                - **Text Branch:** 768-dimensional frozen BERT mean-pooled vector.
                - **Emoji Branch:** Trainable 32-dimensional emoji embeddings.
                - **Fusion Mechanism:** **Emoji-aware scaled dot-product attention.** The text acts as a query to dynamically decide how much weight to give each emoji.
                """
            )
        elif model_key == "E5":
            st.markdown(
                """
                - **Text Branch:** 768-dimensional frozen BERT mean-pooled vector.
                - **Emoji Branch:** Pretrained (TweetEval) 32-dimensional emoji embeddings.
                - **Fusion Mechanism:** **32-dimensional element-wise gated modulation.** A learned sigmoid gate acts like a volume knob, scaling each feature dimension individually.
                """
            )
        else:
            st.markdown(
                """
                - **Text Branch:** 768-dimensional frozen BERT mean-pooled vector.
                - **Emoji Branch:** None (Text-only baseline). Emojis are completely excluded.
                """
            )
