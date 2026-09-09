"""Page 6: Confusion Matrices across All 6 Experiments."""

import streamlit as st

from dashboard.components import render_page_header
from dashboard.data import load_confusion_matrix
from dashboard.visualizations import plot_interactive_confusion_matrix


def render():
    render_page_header(
        title="Normalized Confusion Matrices",
        subtitle="Inspect row-normalized error distributions showing where prediction mass falls for each actual ground-truth label.",
    )

    selected_exp = st.selectbox(
        "Select Model to Inspect:",
        ["E3 (Attention - Best)", "E5 (Gated)", "E4 (Attention Pretrained)", "E2 (Concat Pretrained)", "E0 (Text-only)", "E1 (Concat Random)"],
        index=0,
    )
    exp_key = selected_exp.split()[0]

    col_chart, col_notes = st.columns([3, 2])

    with col_chart:
        cm_data = load_confusion_matrix(exp_key)
        fig = plot_interactive_confusion_matrix(cm_data, exp_key)
        st.plotly_chart(fig, use_container_width=True)

    with col_notes:
        st.markdown("### Error Dynamics & Patterns")
        if exp_key in ["E0", "E1"]:
            st.markdown(
                f"""
                <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 8px; padding: 1.1rem; color: #334155; line-height: 1.6;">
                    <strong>Observations for {exp_key}:</strong>
                    <ul>
                        <li><strong>High Neutral Confusion:</strong> In E0, over 31.5% of Bearish and 32.7% of Bullish samples are misclassified as Neutral.</li>
                        <li><strong>Limited Discriminative Power:</strong> Without attention or pretraining, subtle signals are collapsed into the central Neutral cluster.</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )
        elif exp_key in ["E3", "E4"]:
            st.markdown(
                f"""
                <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 8px; padding: 1.1rem; color: #334155; line-height: 1.6;">
                    <strong>Observations for {exp_key}:</strong>
                    <ul>
                        <li><strong>Sharpened Bullish Sensitivity:</strong> Bullish recall reaches <strong>62.0% (E3)</strong> and <strong>67.2% (E4)</strong>.</li>
                        <li><strong>Reduced Neutral Leakage:</strong> Attention successfully pulls emotional expressions out of the Neutral misclassification trap.</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )
        elif exp_key == "E5":
            st.markdown(
                """
                <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 8px; padding: 1.1rem; color: #334155; line-height: 1.6;">
                    <strong>Observations for E5 (Gated):</strong>
                    <ul>
                        <li><strong>Highest Bearish Recall:</strong> Correctly classifies <strong>49.8%</strong> of actual Bearish tweets.</li>
                        <li><strong>Balanced Error Diagonal:</strong> Element-wise modulation prevents the model from ignoring the minority class.</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 8px; padding: 1.1rem; color: #334155; line-height: 1.6;">
                    <strong>Observations for {exp_key}:</strong>
                    <ul>
                        <li>Solid baseline improvement, but lacks the dynamic weighting achieved by attention and gating mechanisms.</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )
