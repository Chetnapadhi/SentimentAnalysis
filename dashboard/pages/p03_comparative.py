"""Live comparison of text-only and emoji-aware emotion models."""

import plotly.graph_objects as go
import streamlit as st

from dashboard.components import render_page_header, render_research_alert
from dashboard.inference import run_emotion_inference
from dashboard.styles import BORDER_LIGHT, TEXT_PRIMARY


def render():
    render_page_header(
        title="Text-Only vs Emoji-Aware Emotion",
        subtitle="Compare the fine-tuned text baseline with text-conditioned emoji attention on the same sentence.",
    )
    dataset = st.radio(
        "Dataset label space",
        ["goemotions", "tweeteval"],
        format_func=lambda value: "GoEmotions Ekman-6" if value == "goemotions" else "TweetEval Emotion (4-class)",
        horizontal=True,
    )
    text = st.text_area("Sentence or social-media post", "I am excited but nervous about the result 🚀😰", height=90)
    if st.button("Compare emotion models", type="primary", use_container_width=True) or text:
        with st.spinner("Running both trained models..."):
            try:
                # The two trained models are compared as trained (mode="model");
                # the hybrid is shown alongside, not mixed into either.
                baseline = run_emotion_inference(text, dataset, "EM0", mode="model")
                emoji_model = run_emotion_inference(text, dataset, "EM3", mode="model")
                hybrid = run_emotion_inference(text, dataset, "EM3", mode="hybrid")
            except Exception as exc:
                st.error(f"Comparison unavailable: {exc}")
                return
        col1, col2, col3 = st.columns(3)
        for column, title, result, color in [
            (col1, "EM0 · Text-only", baseline, "#64748B"),
            (col2, "EM3 · Emoji attention", emoji_model, "#059669"),
            (col3, "EM3 + lexicon + sarcasm", hybrid, "#D97706"),
        ]:
            with column:
                st.markdown(
                    f"<div style='border:2px solid {color};border-radius:10px;padding:1rem;text-align:center;'>"
                    f"<div style='font-weight:800;color:{color};'>{title}</div>"
                    f"<div style='font-size:1.8rem;font-weight:800;color:#0F172A;margin:.4rem 0;'>{result['pred_label']}</div>"
                    f"<div style='color:#334155;'>Confidence: {result['confidence'] * 100:.1f}%</div></div>",
                    unsafe_allow_html=True,
                )
        labels = baseline["labels"]
        fig = go.Figure([
            go.Bar(name="EM0 text-only", x=labels, y=[baseline["probs"].get(label, 0) * 100 for label in labels], marker_color="#64748B"),
            go.Bar(name="EM3 emoji attention", x=labels, y=[emoji_model["probs"].get(label, 0) * 100 for label in labels], marker_color="#059669"),
            go.Bar(name="Hybrid", x=labels, y=[hybrid["probs"].get(label, 0) * 100 for label in labels], marker_color="#D97706"),
        ])
        if hybrid["sarcasm"]["is_sarcastic"]:
            st.warning(f"Sarcasm detected ({hybrid['sarcasm']['probability']:.0%}): "
                       + "; ".join(hybrid["sarcasm"]["reasons"]))
        fig.update_layout(barmode="group", yaxis=dict(title="Probability (%)", range=[0, 100], gridcolor=BORDER_LIGHT), font=dict(color=TEXT_PRIMARY), height=360)
        st.plotly_chart(fig, use_container_width=True)
        render_research_alert(
            "A difference between two live predictions is an illustration of model behavior, not proof that emojis caused the change. The controlled test-set results on the Emotion Study page are the valid evidence for research claims.",
            alert_type="info",
        )
