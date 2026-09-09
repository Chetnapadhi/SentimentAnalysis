"""Centralized Design System and Theme Tokens for Emoji-Aware Sentiment Analysis Dashboard.

This module guarantees strict contrast ratios, academic elegance, and visual consistency.
Every font size, color, background, and component style is declared centrally.
"""

# ==============================================================================
# 1. CORE COLOR SYSTEM (WCAG AA/AAA Compliant)
# ==============================================================================

# Backgrounds
PAGE_BG = "#F8FAFC"             # Slate 50 (soft subtle neutral off-white)
CARD_BG = "#FFFFFF"             # Pure white for elevated surfaces
SIDEBAR_BG = "#0F172A"          # Slate 900 (deep professional navy/slate)
SIDEBAR_HOVER = "#1E293B"       # Slate 800
SIDEBAR_ACTIVE = "#2563EB"      # Blue 600

# Borders
BORDER_LIGHT = "#E2E8F0"        # Slate 200 (subtle crisp borders)
BORDER_MEDIUM = "#CBD5E1"       # Slate 300
BORDER_ACCENT = "#3B82F6"       # Blue 500

# High-Contrast Text Hierarchy
TEXT_PRIMARY = "#0F172A"        # Slate 900 (deep black-slate, extreme contrast on white)
TEXT_SECONDARY = "#334155"      # Slate 700 (solid dark neutral, perfectly readable)
TEXT_MUTED = "#475569"          # Slate 600 (dark enough for small labels/captions)
TEXT_INVERTED = "#FFFFFF"       # For dark containers (hero, sidebar, chips)
TEXT_INVERTED_MUTED = "#94A3B8" # Slate 400 for sidebar secondary text

# Primary Academic Brand Accents
ACCENT_PRIMARY = "#1D4ED8"      # Deep Royal Blue (academic/authoritative)
ACCENT_LIGHT = "#EFF6FF"        # Soft Blue surface
ACCENT_DARK = "#1E3A8A"

# Semantic Sentiment Palette (Used strictly where semantically meaningful)
COLOR_BEARISH = "#DC2626"       # Red 600 (Negative / Downside)
COLOR_BEARISH_BG = "#FEF2F2"    # Red 50
COLOR_BEARISH_BORDER = "#FCA5A5"# Red 300

COLOR_NEUTRAL = "#D97706"       # Amber 600 (Equilibrium / Unclear)
COLOR_NEUTRAL_BG = "#FFFBEB"    # Amber 50
COLOR_NEUTRAL_BORDER = "#FCD34D"# Amber 300

COLOR_BULLISH = "#059669"       # Emerald 600 (Positive / Upside)
COLOR_BULLISH_BG = "#ECFDF5"    # Emerald 50
COLOR_BULLISH_BORDER = "#6EE7B7"# Emerald 300

# ==============================================================================
# 2. GLOBAL CSS INJECTION
# ==============================================================================

CUSTOM_CSS = f"""
<style>
/* Font Stack: Clean, highly readable system font stack with crisp rendering */
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

html, body, [class*="css"], .stMarkdown, p, span, label, div {{
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
}}

/* Preserve Material Icons font so icon glyphs do NOT render as raw text like keyboard_double_arrow */
[data-testid*="Icon"], .material-symbols-rounded, [class*="material-symbols"], span[class*="icon"] {{
    font-family: 'Material Symbols Rounded', sans-serif !important;
}}

/* Prevent Streamlit sidebar collapse button icon from leaking raw text */
button[data-testid="stSidebarCollapseButton"], 
button[data-testid="stExpandSidebarButton"] {{
    overflow: hidden !important;
}}

code, pre, .font-mono {{
    font-family: 'JetBrains Mono', Consolas, "Courier New", monospace !important;
}}

/* Streamlit Header & Toolbar Fix: Remove upper black bar completely */
header[data-testid="stHeader"] {{
    background-color: transparent !important;
    background: transparent !important;
    color: transparent !important;
    height: 1.5rem !important;
    z-index: 1 !important;
}}

[data-testid="stHeader"] > * {{
    background-color: transparent !important;
}}

/* Fix sidebar collapse/expand icon display */
[data-testid="stSidebarCollapseButton"] button,
[data-testid="stExpandSidebarButton"] button,
[data-testid="collapsedControl"] {{
    font-family: 'Material Symbols Rounded', sans-serif !important;
}}

[data-testid="stSidebarCollapseButton"] span,
[data-testid="stExpandSidebarButton"] span {{
    font-family: 'Material Symbols Rounded', sans-serif !important;
}}

/* Overall page background */
.stApp {{
    background-color: {PAGE_BG} !important;
}}

/* Comfortable, balanced container width */
.block-container {{
    padding-top: 2rem !important;
    padding-bottom: 4rem !important;
    padding-left: 2.5rem !important;
    padding-right: 2.5rem !important;
    max-width: 1200px !important;
}}

/* ==============================================================================
   TYPOGRAPHY HIERARCHY (Strict Dark Text on Light BG)
   ============================================================================== */
h1, .page-title {{
    color: {TEXT_PRIMARY} !important;
    font-size: 2.1rem !important;
    font-weight: 800 !important;
    letter-spacing: -0.6px !important;
    margin-bottom: 0.35rem !important;
    line-height: 1.25 !important;
}}

.page-subtitle {{
    color: {TEXT_SECONDARY} !important;
    font-size: 1.02rem !important;
    font-weight: 500 !important;
    line-height: 1.55 !important;
    margin-bottom: 1.5rem !important;
}}

h2, .section-heading {{
    color: {TEXT_PRIMARY} !important;
    font-size: 1.45rem !important;
    font-weight: 700 !important;
    letter-spacing: -0.3px !important;
    margin-top: 1.5rem !important;
    margin-bottom: 0.6rem !important;
    border-bottom: 2px solid {BORDER_LIGHT};
    padding-bottom: 0.4rem;
}}

h3, .subsection-heading {{
    color: {TEXT_PRIMARY} !important;
    font-size: 1.15rem !important;
    font-weight: 700 !important;
    margin-top: 1.1rem !important;
    margin-bottom: 0.4rem !important;
}}

p, li {{
    color: {TEXT_SECONDARY} !important;
    font-size: 0.95rem !important;
    line-height: 1.6 !important;
}}

strong, b {{
    color: {TEXT_PRIMARY} !important;
    font-weight: 700 !important;
}}

/* ==============================================================================
   SIDEBAR STYLING (Polished Application Drawer)
   ============================================================================== */
[data-testid="stSidebar"] {{
    background-color: {SIDEBAR_BG} !important;
    border-right: 1px solid #1E293B !important;
    width: 305px !important;
}}

[data-testid="stSidebar"] * {{
    color: #F8FAFC !important;
}}

[data-testid="stSidebar"] hr {{
    border-color: #1E293B !important;
    margin: 1.2rem 0 !important;
}}

/* Hide ugly default radio circles */
[data-testid="stSidebar"] div[role="radiogroup"] > label > div:first-child {{
    display: none !important;
}}

/* Style navigation items as clean clickable cards */
[data-testid="stSidebar"] div[role="radiogroup"] > label {{
    background: transparent !important;
    border-radius: 8px !important;
    padding: 9px 14px !important;
    margin-bottom: 4px !important;
    cursor: pointer !important;
    border: 1px solid transparent !important;
    transition: all 0.15s ease-in-out !important;
}}

[data-testid="stSidebar"] div[role="radiogroup"] > label:hover {{
    background: rgba(255, 255, 255, 0.08) !important;
}}

/* Active state for navigation */
[data-testid="stSidebar"] div[role="radiogroup"] > label[data-checked="true"],
[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) {{
    background: #1E293B !important;
    border: 1px solid #38BDF8 !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.3) !important;
}}

[data-testid="stSidebar"] div[role="radiogroup"] > label[data-checked="true"] p,
[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) p {{
    color: #38BDF8 !important;
    font-weight: 700 !important;
}}

/* ==============================================================================
   METRIC & CONTENT CARDS
   ============================================================================== */
.metric-card {{
    background: {CARD_BG};
    border: 1px solid {BORDER_LIGHT};
    border-radius: 12px;
    padding: 1.15rem 1.35rem;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04), 0 1px 2px rgba(0, 0, 0, 0.02);
    transition: transform 0.15s ease, box-shadow 0.15s ease;
    height: 100%;
}}

.metric-card:hover {{
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.06), 0 2px 4px -2px rgba(0, 0, 0, 0.04);
    border-color: {BORDER_MEDIUM};
}}

.metric-title {{
    font-size: 0.78rem;
    font-weight: 700;
    text-transform: uppercase;
    color: {TEXT_MUTED};
    letter-spacing: 0.7px;
    margin-bottom: 0.3rem;
}}

.metric-value {{
    font-size: 1.85rem;
    font-weight: 800;
    color: {TEXT_PRIMARY};
    line-height: 1.2;
}}

.metric-delta-pos {{
    font-size: 0.85rem;
    font-weight: 700;
    color: {COLOR_BULLISH};
    margin-top: 0.35rem;
}}

.metric-delta-neg {{
    font-size: 0.85rem;
    font-weight: 700;
    color: {COLOR_BEARISH};
    margin-top: 0.35rem;
}}

/* Research Question Banner Card */
.rq-card {{
    background: #FFFFFF;
    border: 2px solid #3B82F6;
    border-radius: 12px;
    padding: 1.4rem 1.6rem;
    margin: 1.25rem 0;
    box-shadow: 0 4px 12px -2px rgba(59, 130, 246, 0.08);
}}

/* Experiment Matrix Cards */
.exp-card {{
    background: {CARD_BG};
    border: 1px solid {BORDER_LIGHT};
    border-radius: 12px;
    padding: 1.25rem;
    height: 100%;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    box-shadow: 0 1px 2px rgba(0,0,0,0.03);
}}

.exp-card.best {{
    border: 2px solid {COLOR_BULLISH};
    background: linear-gradient(180deg, #F0FDF4 0%, #FFFFFF 60%);
    box-shadow: 0 4px 12px -2px rgba(5, 150, 105, 0.15);
}}

.exp-badge {{
    display: inline-block;
    padding: 3px 9px;
    border-radius: 6px;
    font-size: 0.72rem;
    font-weight: 800;
    letter-spacing: 0.6px;
    text-transform: uppercase;
}}

.badge-best {{
    background: {COLOR_BULLISH};
    color: #FFFFFF;
}}

.badge-regular {{
    background: #F1F5F9;
    color: {TEXT_MUTED};
    border: 1px solid {BORDER_LIGHT};
}}

/* Research Alerts & Callouts */
.research-alert {{
    background: #FFFFFF;
    border-left: 4px solid #3B82F6;
    border-top: 1px solid {BORDER_LIGHT};
    border-right: 1px solid {BORDER_LIGHT};
    border-bottom: 1px solid {BORDER_LIGHT};
    padding: 1rem 1.3rem;
    border-radius: 0 8px 8px 0;
    margin: 1.2rem 0;
    font-size: 0.92rem;
    color: {TEXT_SECONDARY};
    line-height: 1.6;
}}

.research-alert.warning {{
    border-left-color: {COLOR_NEUTRAL};
    background: #FFFBEB;
    color: #78350F;
}}

.research-alert.info {{
    border-left-color: #2563EB;
    background: #EFF6FF;
    color: #1E3A8A;
}}

/* Pipeline Flow */
.pipeline-box {{
    background: #FFFFFF;
    border: 1px solid {BORDER_LIGHT};
    border-radius: 12px;
    padding: 1.35rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.03);
}}

.pipeline-step {{
    background: #F8FAFC;
    border: 1px solid {BORDER_LIGHT};
    padding: 8px 14px;
    border-radius: 8px;
    font-size: 0.88rem;
    font-weight: 600;
    color: {TEXT_PRIMARY};
}}

.pipeline-arrow {{
    color: {TEXT_MUTED};
    font-weight: 800;
    font-size: 1.1rem;
    margin: 0 6px;
}}

/* ==============================================================================
   FORM CONTROLS & BUTTONS
   ============================================================================== */
/* Text inputs & areas: strict light background with dark readable text */
.stTextArea textarea, .stTextInput input {{
    background-color: #FFFFFF !important;
    color: {TEXT_PRIMARY} !important;
    border: 1px solid {BORDER_MEDIUM} !important;
    border-radius: 8px !important;
    font-size: 0.95rem !important;
}}

.stTextArea textarea:focus, .stTextInput input:focus {{
    border-color: {ACCENT_PRIMARY} !important;
    box-shadow: 0 0 0 2px rgba(29, 78, 216, 0.15) !important;
}}

/* Selectbox labels and items */
.stSelectbox label, .stTextArea label {{
    color: {TEXT_PRIMARY} !important;
    font-weight: 600 !important;
    font-size: 0.92rem !important;
}}

/* Primary Buttons: Authoritative Deep Blue (Never Red for Actions) */
.stButton > button {{
    background-color: {ACCENT_PRIMARY} !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 700 !important;
    padding: 0.55rem 1.4rem !important;
    font-size: 0.95rem !important;
    transition: background-color 0.15s ease !important;
}}

.stButton > button:hover {{
    background-color: {ACCENT_DARK} !important;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important;
}}

/* Compact DataFrames */
[data-testid="stDataFrame"] {{
    border: 1px solid {BORDER_LIGHT} !important;
    border-radius: 8px !important;
}}

/* Footer */
.research-footer {{
    margin-top: 3.5rem;
    padding-top: 1.25rem;
    border-top: 1px solid {BORDER_LIGHT};
    text-align: center;
    color: {TEXT_MUTED};
    font-size: 0.85rem;
    font-weight: 500;
}}
</style>
"""
