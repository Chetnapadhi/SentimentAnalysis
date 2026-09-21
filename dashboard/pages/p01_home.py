"""Emotion-first project overview."""

import streamlit as st

from dashboard.components import render_metric_card, render_page_header, render_research_alert


def render():
    render_page_header(
        title="Emotion-Aware Social Text Analysis",
        subtitle="The teacher-directed project extension classifies the emotion expressed in a sentence, with emoji-aware modeling as an explicit research variable.",
    )

    st.markdown(
        """
        <div class="rq-card">
            <div style="font-size:.8rem;font-weight:800;color:#2563EB;text-transform:uppercase;letter-spacing:.8px;">Research question</div>
            <div style="font-size:1.2rem;font-weight:700;color:#0F172A;line-height:1.45;margin-top:.4rem;">
                Can a social-media model identify the specific emotion in a sentence, and do emojis provide useful signal beyond its words?
            </div>
            <div style="font-size:.9rem;color:#475569;margin-top:.55rem;">
                The main task is emotion classification: anger, disgust, fear, joy, sadness, surprise, or the smaller TweetEval emotion vocabulary.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_metric_card("GoEmotions", "6 classes", "Ekman single-label subset", is_positive=True)
    with c2:
        render_metric_card("TweetEval", "4 classes", "Social-media benchmark", is_positive=True)
    with c3:
        render_metric_card("Best TweetEval Accuracy", "79.9%", "EM0 fine-tuned", is_positive=True)
    with c4:
        render_metric_card("Best GoEmotions Accuracy", "77.4%", "EM0 fine-tuned", is_positive=True)

    st.markdown("### What each part of the project does")
    columns = st.columns(3)
    cards = [
        ("1 · Emotion classifier", "The live analyzer predicts a dataset-defined emotion instead of Positive, Negative, or Neutral."),
        ("2 · Emoji experiment", "EM3 uses text-conditioned attention to decide how much each emoji contributes to the emotion prediction."),
        ("3 · Controlled evidence", "The results page reports Macro F1, majority baselines, class-level scores, and emoji/no-emoji slices."),
    ]
    for column, (title, body) in zip(columns, cards):
        with column:
            st.markdown(
                f"<div class='exp-card'><h4 style='color:#0F172A;margin-top:0;'>{title}</h4><p>{body}</p></div>",
                unsafe_allow_html=True,
            )

    render_research_alert(
        "<strong>How to present the result:</strong> The fine-tuned text baseline is strongest overall on the reported datasets, while emoji-aware fusion is evaluated as a controlled research variable. Do not claim that emojis always improve accuracy.",
        alert_type="info",
    )
    render_research_alert(
        "<strong>Phase 1 is historical context:</strong> The original StockTwits Bearish/Neutral/Bullish benchmark remains available under Phase 1, but it is not the teacher's final emotion-classification task.",
        alert_type="warning",
    )
