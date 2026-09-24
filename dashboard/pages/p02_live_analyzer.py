"""Live fine-grained emotion analyzer (hybrid: model + emoji lexicon + sarcasm)."""

import pandas as pd
import streamlit as st

from dashboard.components import render_page_header, render_research_alert
from dashboard.inference import run_emotion_inference
from dashboard.visualizations import plot_emotion_probability_bars


EXAMPLES = [
    "I cannot believe this happened, I am so angry 😤",
    "Everything is falling apart and I feel helpless 😭",
    "This is amazing news, I am so happy! 🎉🚀",
    "I am worried about what happens next 😰",
    "I just love waiting 3 hours at the DMV 🙄🙄",
    "Great job management, losing half my money 🤡💸",
    "Oh wonderful, my flight got cancelled again 🙃",
    "😂😂😂😂",
    "I laughed so hard I cried 😂😭",
]


def _sarcasm_card(s: dict) -> str:
    pct = s["probability"] * 100
    if s["level"] == "detected":
        bg, border, fg, title = "#FEF3C7", "#D97706", "#92400E", "Sarcasm detected"
    elif s["level"] == "possible":
        bg, border, fg, title = "#FFFBEB", "#FBBF24", "#92400E", "Possible sarcasm (unconfirmed)"
    else:
        bg, border, fg, title = "#F1F5F9", "#94A3B8", "#334155", "No sarcasm detected"
    return (
        f"<div style='background:{bg};border:2px solid {border};border-radius:12px;padding:1rem 1.2rem;'>"
        f"<div style='font-size:.75rem;font-weight:800;color:{fg};text-transform:uppercase;'>{title}</div>"
        f"<div style='font-size:1.6rem;font-weight:800;color:{fg};margin:.2rem 0;'>{pct:.0f}%</div>"
        f"<div style='height:8px;background:#E2E8F0;border-radius:4px;overflow:hidden;'>"
        f"<div style='width:{pct:.0f}%;height:8px;background:{border};'></div></div></div>"
    )


def render():
    render_page_header(
        title="Live Emotion Analyzer",
        subtitle="The trained Phase 2 model, corrected by a hand-labelled emoji lexicon and a sarcasm detector.",
    )
    dataset = st.radio(
        "Emotion label space",
        ["goemotions", "tweeteval"],
        format_func=lambda value: "GoEmotions Ekman-6" if value == "goemotions" else "TweetEval Emotion (4-class)",
        horizontal=True,
    )
    col_a, col_b = st.columns(2)
    model = col_a.selectbox(
        "Model",
        ["EM3", "EM0"],
        format_func=lambda value: {
            "EM3": "EM3: text-conditioned emoji attention",
            "EM0": "EM0: text-only baseline",
        }[value],
    )
    mode = col_b.selectbox(
        "Prediction mode",
        ["hybrid", "model"],
        format_func=lambda value: {
            "hybrid": "Hybrid: model + emoji lexicon + sarcasm (recommended)",
            "model": "Model only (as trained)",
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
        with st.spinner("Reading the words, the emojis and the tone..."):
            try:
                result = run_emotion_inference(text, dataset=dataset, model_key=model, mode=mode)
            except Exception as exc:
                st.error(f"Emotion inference is unavailable: {exc}")
                return

        left, right = st.columns([1, 2])
        with left:
            changed = result["pred_label"] != result["model_label"]
            subline = (f"Model alone said <b>{result['model_label']}</b>"
                       if changed else "Model and hybrid agree")
            st.markdown(
                f"<div style='background:#ECFDF5;border:2px solid #059669;border-radius:12px;padding:1.5rem;text-align:center;'>"
                f"<div style='font-size:.75rem;font-weight:800;color:#047857;text-transform:uppercase;'>"
                f"{'Intended emotion' if result['sarcasm']['is_sarcastic'] else 'Predicted emotion'}</div>"
                f"<div style='font-size:2.25rem;font-weight:800;color:#065F46;margin:.35rem 0;'>{result['pred_label']}</div>"
                f"<div style='font-weight:700;color:#334155;'>Confidence: {result['confidence'] * 100:.1f}%</div>"
                f"<div style='font-size:.8rem;color:#475569;margin-top:.5rem;'>{subline}</div>"
                f"<div style='font-size:.75rem;color:#64748B;margin-top:.25rem;'>{dataset} · {model} · {mode}</div></div>",
                unsafe_allow_html=True,
            )
            if mode == "hybrid":
                st.markdown("<div style='height:.75rem'></div>", unsafe_allow_html=True)
                st.markdown(_sarcasm_card(result["sarcasm"]), unsafe_allow_html=True)
        with right:
            st.plotly_chart(plot_emotion_probability_bars(result["probs"]), use_container_width=True)

        if mode == "hybrid":
            level = result["sarcasm"]["level"]
            if level == "detected" and result["validated_label"] != result["pred_label"]:
                st.warning(
                    f"**Sarcasm.** Literally this reads as **{result['validated_label']}**, but the intended "
                    f"emotion is most likely **{result['pred_label']}**."
                )
            elif level == "detected":
                st.warning(f"**Sarcasm detected.** The emotion reading is **{result['pred_label']}** either way.")
            elif level == "possible":
                st.info(
                    "**Possibly sarcastic.** The irony classifier leans that way, but no word or emoji cue "
                    "confirms it, so the emotion shown is the literal reading."
                )
            if result["sarcasm"]["reasons"]:
                with st.expander("Why the sarcasm score is what it is", expanded=level == "detected"):
                    for reason in result["sarcasm"]["reasons"]:
                        st.markdown(f"- {reason}")
                    st.caption(f"Detector: {result['sarcasm']['backend']}")
            for note in result["notes"]:
                st.info(note)

        st.markdown("### Emojis in this sentence")
        breakdown = result["emoji_evidence"]["breakdown"]
        if breakdown:
            ev = result["emoji_evidence"]
            c1, c2, c3 = st.columns(3)
            c1.metric("Emoji occurrences", result["num_emojis"])
            c2.metric("Emoji evidence strength", f"{ev['strength'] * 100:.0f}%")
            c3.metric("Emojis point to", ev["top_label"].title() + (" (mixed)" if ev["mixed"] else ""))
            st.dataframe(
                pd.DataFrame([{
                    "Emoji": b["emoji"],
                    "Count": b["count"],
                    "Lexicon label": b["label"],
                    "Polarity": f"{b['polarity']:+.2f}",
                    "Sarcasm cue": f"{b['sarcasm_cue']:.2f}",
                    "Weight": b["weight"],
                    "Label source": b["source"],
                } for b in breakdown]),
                hide_index=True, use_container_width=True,
            )
        else:
            st.caption("No emojis detected — the prediction is the trained model's reading of the text.")

        with st.expander("What the trained model received"):
            st.markdown(f"Text branch: `{result['text_without_emoji']}`")
            st.markdown(f"Emoji branch: {' '.join(result['emojis']) or 'no emojis'}")
            st.markdown("Model-only probabilities: " + ", ".join(
                f"{k} {v * 100:.0f}%" for k, v in result["model_probs"].items()))

        render_research_alert(
            "This is a model prediction, not a psychological diagnosis. Labels are limited to the selected "
            "dataset's vocabulary. Sarcasm is detected by a pretrained irony classifier plus hand-written rules; "
            "subtle sarcasm without clear cues can still be missed.",
            alert_type="warning",
        )
