"""Emoji-Aware Sentiment Analysis — Streamlit Research Dashboard.

Streamlined 5-Section Architecture:
1. ⚡ Live Sentiment Analyzer (Real-World Demo: Positive/Neutral/Negative)
2. ⚖️ Word vs Emoji Comparison (E0 vs E3 vs E5 Side-by-Side)
3. 📊 Benchmark Results & Why 6 Models? (Tournament Leaderboard & Justifications)
4. 🔍 Deep-Dive Insights (Class breakdown, Emoji density scaling, Error transitions)
5. 🎓 Research Conclusions & Teacher FAQ (VIVA prep & Key Takeaways)
"""

from __future__ import annotations

import streamlit as st

from dashboard.components import render_footer
from dashboard.styles import CUSTOM_CSS

# 1. Page Configuration
st.set_page_config(
    page_title="Emoji-Aware Sentiment Analyzer",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 2. Inject High-Contrast Theme
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# 3. Import Page Modules
from dashboard.pages import (
    p01_home,
    p02_live_analyzer,
    p03_comparative,
    p04_experiments,
    p05_per_class,
    p06_confusion,
    p07_emoji_insights,
    p08_error_analysis,
    p09_gate_analysis,
    p10_architecture,
    p11_conclusion,
)

# Combined Section Renderers to reduce navigation clutter
def render_results_section():
    p04_experiments.render()
    st.markdown("---")
    st.markdown("### Architectural Deep-Dive")
    p10_architecture.render()

def render_insights_section():
    p05_per_class.render()
    st.markdown("---")
    p07_emoji_insights.render()
    st.markdown("---")
    p06_confusion.render()
    st.markdown("---")
    p08_error_analysis.render()
    st.markdown("---")
    p09_gate_analysis.render()

# 4. Streamlined 5-Tab Application Drawer
st.sidebar.markdown(
    """
    <div style="padding: 0.5rem 0.2rem 1rem 0.2rem;">
        <div style="font-size: 1.15rem; font-weight: 800; color: #FFFFFF; letter-spacing: -0.3px;">
            🔬 Emoji-Aware AI
        </div>
        <div style="font-size: 0.8rem; color: #94A3B8; font-weight: 500;">
            Sentiment Analysis Platform
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.markdown(
    """<div style="font-size: 0.72rem; font-weight: 700; color: #64748B; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 0.4rem;">Application Sections</div>""",
    unsafe_allow_html=True,
)

STREAMLINED_PAGES = {
    "⚡ Live Sentiment Analyzer": p02_live_analyzer.render,
    "⚖️ Word vs Emoji Comparison": p03_comparative.render,
    "🏠 Overview & Research Context": p01_home.render,
    "📊 Benchmark Results (E0–E5)": render_results_section,
    "🔍 Detailed Evaluation & Insights": render_insights_section,
    "📋 Research Conclusions": p11_conclusion.render,
}

page_selection = st.sidebar.radio(
    "Navigation Menu",
    list(STREAMLINED_PAGES.keys()),
    index=0,
    label_visibility="collapsed",
)

# 5. Render Active View
try:
    STREAMLINED_PAGES[page_selection]()
except Exception as e:
    st.error(f"Error rendering section '{page_selection}': {str(e)}")
    import traceback
    st.code(traceback.format_exc())

# 6. Global Footer
render_footer()
