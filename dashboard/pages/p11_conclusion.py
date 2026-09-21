"""Emotion-first research conclusions."""

import streamlit as st

from dashboard.components import render_page_header, render_research_alert


def render():
    render_page_header(
        title="Research Conclusions",
        subtitle="What the trained emotion models establish, and what they do not establish.",
    )

    st.markdown("### Main findings")
    st.markdown(
        """
        <div style="background:#FFFFFF;border:1px solid #E2E8F0;border-radius:12px;padding:1.5rem;color:#334155;line-height:1.7;">
            <ol style="margin-bottom:0;padding-left:1.3rem;">
                <li style="margin-bottom:.7rem;"><strong>Emotion classification is more informative than a single sentiment polarity.</strong><br>The Phase 2 models distinguish multiple emotion categories rather than collapsing every sentence into positive, negative, or neutral.</li>
                <li style="margin-bottom:.7rem;"><strong>Fine-tuning matters.</strong><br>Unfreezing the top two Twitter-RoBERTa blocks substantially improves the emotion results over the frozen controls.</li>
                <li style="margin-bottom:.7rem;"><strong>Emoji fusion is a research variable, not a guaranteed improvement.</strong><br>EM3 is strongest on the GoEmotions result set, but EM0 is strongest overall on both datasets' fine-tuned comparisons.</li>
                <li><strong>Macro F1 and the majority baseline are essential.</strong><br>Class imbalance makes raw accuracy alone insufficient, especially for GoEmotions where joy is the majority class.</li>
            </ol>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Limitations to state during evaluation")
    render_research_alert(
        "<strong>Label scope:</strong> GoEmotions is reduced to a single-label Ekman-6 subset by dropping neutral and ambiguous examples. TweetEval has four published classes: anger, joy, optimism, and sadness.<br><br><strong>Generalization:</strong> The two datasets have different domains, label definitions, and test distributions; their scores must not be ranked against each other.<br><br><strong>Emoji evidence:</strong> Emoji/no-emoji slices are descriptive because emoji presence is not randomly assigned. They do not prove causation.<br><br><strong>Evaluation:</strong> The reported runs use one seed and live predictions are demonstrations, not psychological diagnoses.",
        alert_type="info",
    )
