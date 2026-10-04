"""
CSS injection for consistent, professional styling.

A single CSS block injected once via st.markdown to style metric cards,
tabs, buttons, tables, and general layout to match the blue/white
financial research tool aesthetic.
"""

from __future__ import annotations

import streamlit as st

from config import PALETTE


def inject_custom_css() -> None:
    """Inject the global custom stylesheet."""
    css = f"""
    <style>
        /* ── Google Font ── */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

        /* ── Base ── */
        html, body, [class*="css"] {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            color: {PALETTE.text};
        }}

        .main .block-container {{
            max-width: 1200px;
            padding-top: 1.5rem;
            padding-bottom: 2rem;
        }}

        /* ── Headings ── */
        h1 {{
            color: {PALETTE.primary};
            font-weight: 700;
            font-size: 1.6rem;
            letter-spacing: -0.01em;
            margin-bottom: 0.25rem;
        }}
        h2 {{
            color: {PALETTE.primary};
            font-weight: 600;
            font-size: 1.2rem;
            margin-top: 1.5rem;
            margin-bottom: 0.5rem;
        }}
        h3 {{
            color: {PALETTE.text};
            font-weight: 600;
            font-size: 1.05rem;
            margin-top: 1rem;
        }}

        /* ── Sidebar ── */
        [data-testid="stSidebar"] {{
            background-color: {PALETTE.surface};
            border-right: 1px solid {PALETTE.border};
        }}
        [data-testid="stSidebar"] h1 {{
            font-size: 1.1rem;
        }}

        /* ── Tabs ── */
        .stTabs [data-baseweb="tab-list"] {{
            gap: 0;
            border-bottom: 2px solid {PALETTE.border};
        }}
        .stTabs [data-baseweb="tab"] {{
            font-family: 'Inter', sans-serif;
            font-weight: 500;
            font-size: 0.85rem;
            color: {PALETTE.text_secondary};
            padding: 0.6rem 1.1rem;
            border-bottom: 2px solid transparent;
            margin-bottom: -2px;
        }}
        .stTabs [data-baseweb="tab"][aria-selected="true"] {{
            color: {PALETTE.primary};
            border-bottom-color: {PALETTE.primary};
            font-weight: 600;
        }}

        /* ── Metric card ── */
        .metric-card {{
            background: {PALETTE.background};
            border: 1px solid {PALETTE.border};
            border-radius: 6px;
            padding: 1rem 1.2rem;
            margin-bottom: 0.75rem;
        }}
        .metric-card .metric-label {{
            font-size: 0.75rem;
            color: {PALETTE.text_secondary};
            text-transform: uppercase;
            letter-spacing: 0.04em;
            font-weight: 600;
            margin-bottom: 0.25rem;
        }}
        .metric-card .metric-value {{
            font-size: 1.5rem;
            font-weight: 700;
            color: {PALETTE.primary};
            line-height: 1.2;
        }}
        .metric-card .metric-sub {{
            font-size: 0.8rem;
            color: {PALETTE.text_secondary};
            margin-top: 0.15rem;
        }}

        /* ── Streamlit default metrics override ── */
        [data-testid="stMetricValue"] {{
            font-family: 'Inter', sans-serif;
            font-weight: 700;
            color: {PALETTE.primary};
        }}
        [data-testid="stMetricLabel"] {{
            font-family: 'Inter', sans-serif;
            font-weight: 600;
            text-transform: uppercase;
            font-size: 0.72rem;
            letter-spacing: 0.04em;
            color: {PALETTE.text_secondary};
        }}

        /* ── Buttons ── */
        .stButton > button {{
            font-family: 'Inter', sans-serif;
            font-weight: 500;
            background-color: {PALETTE.primary};
            color: white;
            border: none;
            border-radius: 4px;
            padding: 0.45rem 1.2rem;
            font-size: 0.85rem;
            transition: background-color 0.15s ease;
        }}
        .stButton > button:hover {{
            background-color: {PALETTE.accent};
            color: white;
        }}

        /* ── Tables ── */
        .stDataFrame table {{
            font-family: 'Inter', sans-serif;
            font-size: 0.82rem;
        }}
        .stDataFrame th {{
            background-color: {PALETTE.surface};
            color: {PALETTE.primary};
            font-weight: 600;
            text-transform: uppercase;
            font-size: 0.72rem;
            letter-spacing: 0.03em;
        }}

        /* ── Disclaimer ── */
        .disclaimer {{
            background-color: {PALETTE.surface};
            border: 1px solid {PALETTE.border};
            border-radius: 4px;
            padding: 0.75rem 1rem;
            font-size: 0.78rem;
            color: {PALETTE.text_secondary};
            line-height: 1.5;
            margin-top: 1.5rem;
        }}

        /* ── Info box ── */
        .info-box {{
            background-color: {PALETTE.light_tint};
            border-left: 3px solid {PALETTE.accent};
            padding: 0.75rem 1rem;
            border-radius: 0 4px 4px 0;
            font-size: 0.85rem;
            color: {PALETTE.text};
            margin-bottom: 1rem;
        }}

        /* ── Warning box ── */
        .warning-box {{
            background-color: #FFF8E1;
            border-left: 3px solid #F0AD4E;
            padding: 0.75rem 1rem;
            border-radius: 0 4px 4px 0;
            font-size: 0.85rem;
            color: {PALETTE.text};
            margin-bottom: 1rem;
        }}

        /* ── Selectbox / inputs ── */
        .stSelectbox label, .stTextInput label, .stNumberInput label {{
            font-weight: 500;
            font-size: 0.82rem;
            color: {PALETTE.text};
        }}

        /* ── Download button ── */
        .stDownloadButton > button {{
            background-color: {PALETTE.background};
            color: {PALETTE.primary};
            border: 1px solid {PALETTE.border};
            font-weight: 500;
        }}
        .stDownloadButton > button:hover {{
            background-color: {PALETTE.light_tint};
            border-color: {PALETTE.accent};
        }}

        /* ── Divider ── */
        hr {{
            border: none;
            border-top: 1px solid {PALETTE.border};
            margin: 1.5rem 0;
        }}

        /* ── Spinner text ── */
        .stSpinner > div {{
            color: {PALETTE.text_secondary};
        }}

        /* ── Hide Streamlit branding ── */
        #MainMenu {{visibility: hidden;}}
        footer {{visibility: hidden;}}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)
