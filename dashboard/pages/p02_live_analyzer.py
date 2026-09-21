"""Live fine-grained emotion analyzer."""

import streamlit as st

from dashboard.components import render_page_header, render_research_alert
from dashboard.inference import run_emotion_inference
from dashboard.visualizations import plot_emotion_probability_bars


EXAMPLES = [
    "I cannot believe this happened, I am so angry 😤",
    "Everything is falling apart and I feel helpless 😭",
    "This is amazing news, I am so happy! 🎉🚀",
    "I am worried about what happens next 😰",
    "Wait, how did they reach that conclusion? 🤔",
]


def render():
    render_page_header(
        title="Live Emotion Analyzer",
        subtitle="Classify the emotion expressed by a sentence using the trained Phase 2 model, rather than a coarse positive or negative label.",
    )
    dataset = st.radio(
        "Emotion label space",
        ["goemotions", "tweeteval"],
        format_func=lambda value: "GoEmotions Ekman-6" if value == "goemotions" else "TweetEval Emotion (4-class)",
        horizontal=True,
    )
    model = st.selectbox(
        "Model",
        ["EM3", "EM0"],
        format_func=lambda value: {
            "EM3": "EM3: text-conditioned emoji attention",
            "EM0": "EM0: text-only baseline",
        }[value],
    )
    example = st.selectbox("Example sentence", ["Custom input"] + EXAMPLES)
    text = st.text_area(
        "Sentence or social-media post",
        value="" if example == "Custom input" else example,
        height=100,
        placeholder="Write a sentence such as: I am nervous about tomorrow's results 😰",
    )
    if st.button("Classify emotion", type="primary", use_container_width=True) or text:
        if not text.strip():
            st.warning("Enter a sentence first.")
            return
        with st.spinner("Reading the sentence and its emotional signal..."):
            try:
                result = run_emotion_inference(text, dataset=dataset, model_key=model)
            except Exception as exc:
                st.error(f"Emotion inference is unavailable: {exc}")
                return

        left, right = st.columns([1, 2])
        with left:
            st.markdown(
                f"<div style='background:#ECFDF5;border:2px solid #059669;border-radius:12px;padding:1.5rem;text-align:center;'>"
                f"<div style='font-size:.75rem;font-weight:800;color:#047857;text-transform:uppercase;'>Predicted emotion</div>"
                f"<div style='font-size:2.25rem;font-weight:800;color:#065F46;margin:.35rem 0;'>{result['pred_label']}</div>"
                f"<div style='font-weight:700;color:#334155;'>Confidence: {result['confidence'] * 100:.1f}%</div>"
                f"<div style='font-size:.8rem;color:#475569;margin-top:.5rem;'>{dataset} · {model}</div></div>",
                unsafe_allow_html=True,
            )
        with right:
            st.plotly_chart(plot_emotion_probability_bars(result["probs"]), use_container_width=True)

        st.markdown("### What the model received")
        col1, col2, col3 = st.columns(3)
        col1.metric("Detected emojis", result["num_emojis"])
        col2.metric("Label space", f"{len(result['labels'])} emotions")
        col3.metric("Backbone", "Twitter-RoBERTa")
        st.info(f"Text branch: `{result['text_without_emoji']}`\n\nEmoji branch: {result['emojis'] or 'No emojis detected'}")
        render_research_alert(
            "This is a model prediction, not a psychological diagnosis. The displayed class is limited to the selected dataset's label vocabulary; sarcasm, mixed emotions, and unseen contexts may be misclassified.",
            alert_type="warning",
        )
