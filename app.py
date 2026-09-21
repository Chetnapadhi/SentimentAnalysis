"""Emotion-aware social text analysis dashboard."""

from __future__ import annotations

import streamlit as st

from dashboard.components import render_footer
from dashboard.styles import CUSTOM_CSS

# 1. Page Configuration
st.set_page_config(
    page_title="Emotion-Aware Social Text Analyzer",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 2. Inject High-Contrast Theme
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# 3. Import Page Modules
from dashboard.pages import p01_home, p02_live_analyzer, p03_comparative, p11_conclusion, p12_emotion_results

# 4. Application Drawer
st.sidebar.markdown(
    """
    <div style="padding: 0.5rem 0.2rem 1rem 0.2rem;">
        <div style="font-size: 1.15rem; font-weight: 800; color: #FFFFFF; letter-spacing: -0.3px;">
            🧠 Emotion-Aware AI
        </div>
        <div style="font-size: 0.8rem; color: #94A3B8; font-weight: 500;">
            Fine-Grained Emotion Classification
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
    "⚡ Live Emotion Analyzer": p02_live_analyzer.render,
    "⚖️ Text vs Emoji Emotion": p03_comparative.render,
    "🏠 Emotion Project Overview": p01_home.render,
    "🧠 Emotion Study (Phase 2)": p12_emotion_results.render,
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
