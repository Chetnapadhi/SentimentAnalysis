"""Page 3: Live Model Comparison across E0, E3, and E5."""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard.components import render_page_header, render_research_alert
from dashboard.inference import run_live_inference
from dashboard.styles import (
    COLOR_BEARISH,
    COLOR_NEUTRAL,
    COLOR_BULLISH,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    BORDER_LIGHT,
)


COMPARISON_PROMPTS = [
    "Great company, really love the management 🤡💀",  # Sarcasm / Irony
    "Breaking down support levels but volume is light 📉📈",  # Mixed signals
    "Bought the dip 🚀",  # Sarcasm / Flip
    "Another wonderful earnings report from our favorites 🗑️🚮",  # Negative emoji on positive text
]


def render():
    render_page_header(
        title="Live Model Comparison",
        subtitle="See how emojis flip or sharpen predictions: Words Alone (E0) vs Words + Emojis (E3, E5).",
    )

    chosen_preset = st.selectbox(
        "Choose an illustrative example (e.g. sarcasm, contrarian emojis) or enter custom text:",
        ["(Custom)"] + COMPARISON_PROMPTS,
        index=1,
    )

    initial_text = "" if chosen_preset == "(Custom)" else chosen_preset
    user_input = st.text_area(
        "Social-media post:",
        value=initial_text,
        height=80,
        placeholder="Enter text to observe how emoji modeling shifts predictions...",
    )

    run_comp = st.button("Compare Architectures", type="primary", use_container_width=True)

    if run_comp or user_input:
        if not user_input.strip():
            st.warning("Please enter a social-media post.")
            return

        with st.spinner("Running comparative inference..."):
            try:
                res_e0 = run_live_inference(user_input, model_name="E0")
                res_e3 = run_live_inference(user_input, model_name="E3")
                res_e5 = run_live_inference(user_input, model_name="E5")
            except Exception as e:
                st.error(f"Execution error: {str(e)}")
                return

        st.markdown(
            f"""
            <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 8px; padding: 0.75rem 1.2rem; margin: 1rem 0; font-size: 0.9rem; color: #334155;">
                <strong>Detected Emojis:</strong> {res_e3['emojis'] if res_e3['emojis'] else 'None'} &bull; 
                <strong>Stripped Text (Fed to BERT):</strong> <code>"{res_e3['text_without_emoji']}"</code>
            </div>
            """,
            unsafe_allow_html=True,
        )

        col1, col2, col3 = st.columns(3)
        models_data = [
            ("E0 (Words Only - Emoji Blind)", res_e0, "#475569", "#F8FAFC"),
            ("E3 (Words + Emojis via Attention)", res_e3, "#059669", "#ECFDF5"),
            ("E5 (Words + Emojis via Gate)", res_e5, "#2563EB", "#EFF6FF"),
        ]

        for col, (name, res, border_col, bg_col) in zip([col1, col2, col3], models_data):
            with col:
                lbl = res.get("short_label", res["pred_label"])
                conf = res["confidence"] * 100
                st.markdown(
                    f"""
                    <div style="background: {bg_col}; border: 2px solid {border_col}; border-radius: 12px; padding: 1.3rem; text-align: center; height: 100%;">
                        <div style="font-size: 0.75rem; font-weight: 800; color: {border_col}; text-transform: uppercase; letter-spacing: 0.7px;">{name}</div>
                        <div style="font-size: 1.9rem; font-weight: 800; color: #0F172A; margin: 0.4rem 0;">{lbl.upper()}</div>
                        <div style="font-size: 0.95rem; font-weight: 700; color: #334155;">Confidence: {conf:.1f}%</div>
                        <hr style="margin: 0.8rem 0; border: none; border-top: 1px solid #CBD5E1;">
                        <div style="font-size: 0.84rem; text-align: left; line-height: 1.6; color: #1E293B;">
                            <div>🔴 Negative: <strong>{res['probs']['Bearish']*100:.1f}%</strong></div>
                            <div>🟡 Neutral: <strong>{res['probs']['Neutral']*100:.1f}%</strong></div>
                            <div>🟢 Positive: <strong>{res['probs']['Bullish']*100:.1f}%</strong></div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### Comparative Probability Distribution")

        # Grouped bar chart comparing the 3 models with intuitive Positive/Neutral/Negative labels
        labels = ["Negative", "Neutral", "Positive"]
        key_map = {"Negative": "Bearish", "Neutral": "Neutral", "Positive": "Bullish"}
        fig = go.Figure()
        fig.add_trace(go.Bar(name="E0 (Words Only)", x=labels, y=[res_e0["probs"][key_map[l]]*100 for l in labels], marker_color="#64748B"))
        fig.add_trace(go.Bar(name="E3 (Words + Attention)", x=labels, y=[res_e3["probs"][key_map[l]]*100 for l in labels], marker_color="#059669"))
        fig.add_trace(go.Bar(name="E5 (Words + Gate)", x=labels, y=[res_e5["probs"][key_map[l]]*100 for l in labels], marker_color="#2563EB"))

        fig.update_layout(
            barmode="group",
            font=dict(family="Plus Jakarta Sans, sans-serif", color=TEXT_PRIMARY, size=12),
            plot_bgcolor="#FFFFFF",
            paper_bgcolor="#FFFFFF",
            margin=dict(l=40, r=20, t=30, b=40),
            yaxis=dict(title="<b>Probability (%)</b>", range=[0, 100], showgrid=True, gridcolor=BORDER_LIGHT),
            xaxis=dict(title="<b>Sentiment Class</b>"),
            height=340,
            legend=dict(font=dict(color=TEXT_PRIMARY)),
        )
        st.plotly_chart(fig, use_container_width=True)

        render_research_alert(
            "<strong>Real-World Insight:</strong> Notice how E0 (words only) often misinterprets sarcastic or ambivalent phrases because it only reads the words, whereas E3 and E5 look at the emojis to correctly detect the true positive or negative sentiment.",
            alert_type="info",
        )
