"""Page 8: Error Analysis & Research Evidence."""

import pandas as pd
import streamlit as st

from dashboard.components import render_page_header, render_metric_card, render_research_alert
from dashboard.data import load_error_transition_cases, load_transition_summary


def render():
    render_page_header(
        title="Error Analysis & Transition Dynamics",
        subtitle="Auditing qualitative prediction transitions to verify what signals emojis rescue vs where regressions occur.",
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_metric_card("E0 Wrong ➔ E3 Correct", "2,494", "20.8% Attention Recovery", is_positive=True)
    with c2:
        render_metric_card("E0 Wrong ➔ E5 Correct", "2,049", "17.1% Gated Recovery", is_positive=True)
    with c3:
        render_metric_card("E0 Correct ➔ E3 Wrong", "1,715", "14.3% Attention Regression", is_positive=False)
    with c4:
        render_metric_card("E3 Net Gain over E0", "+779", "+6.51 pp Correctness Gain", is_positive=True)

    render_research_alert(
        "<strong>Methodological Caveat:</strong> These are prediction recoveries and regressions relative to E0; they should not be interpreted as direct causal evidence that emojis alone caused each change.",
        alert_type="info",
    )

    st.markdown("### Real Test Transition Examples")
    
    category_choice = st.selectbox(
        "Select Transition Pattern to Inspect:",
        [
            "E0 wrong ➔ E3 correct (Attention Recovery)",
            "E0 wrong ➔ E5 correct (Gated Recovery)",
            "E0 correct ➔ E3 wrong (Attention Regression)",
            "E0 correct ➔ E5 wrong (Gated Regression)",
        ],
    )

    cat_map = {
        "E0 wrong ➔ E3 correct (Attention Recovery)": "E0_wrong__E3_correct",
        "E0 wrong ➔ E5 correct (Gated Recovery)": "E0_wrong__E5_correct",
        "E0 correct ➔ E3 wrong (Attention Regression)": "E0_correct__E3_wrong",
        "E0 correct ➔ E5 wrong (Gated Regression)": "E0_correct__E5_wrong",
    }

    cases_df = load_error_transition_cases()
    sub_cases = cases_df[cases_df["category"] == cat_map[category_choice]]

    st.markdown(f"Displaying verified samples from `error_transition_cases.csv` (Total: {len(sub_cases)}):")

    for idx, row in sub_cases.iterrows():
        with st.expander(f"Tweet: \"{row['original_text'][:75]}...\" | Emojis: {row['emoji_list']}"):
            st.markdown(f"**Full Text:** `{row['original_text']}`")
            st.markdown(f"**Text Without Emoji:** `{row['text_without_emoji']}`")
            
            c_gt, c_e0, c_e3, c_e5 = st.columns(4)
            c_gt.markdown(f"**Ground Truth:**<br><span style='color: #0F172A; font-weight: 700;'>{row['true_label']}</span>", unsafe_allow_html=True)
            c_e0.markdown(f"**E0 Pred:**<br><span style='color: #475569; font-weight: 700;'>{row['e0_prediction']}</span>", unsafe_allow_html=True)
            c_e3.markdown(f"**E3 Pred:**<br><span style='color: #059669; font-weight: 700;'>{row['e3_prediction']}</span>", unsafe_allow_html=True)
            c_e5.markdown(f"**E5 Pred:**<br><span style='color: #2563EB; font-weight: 700;'>{row['e5_prediction']}</span>", unsafe_allow_html=True)
