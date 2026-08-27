"""Shared warm espresso, caramel bronze & terracotta minimal design system."""

import streamlit as st

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"], .stMarkdown, .stText {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    letter-spacing: -0.01em;
    color: #f7f1eb;
}

/* Header & Title Styling */
h1 {
    font-weight: 700 !important;
    letter-spacing: -0.03em !important;
    color: #fbf8f5 !important;
    padding-bottom: 0.2rem !important;
}

h2, h3, h4 {
    font-weight: 600 !important;
    letter-spacing: -0.02em !important;
    color: #ead8c7 !important;
}

/* Modern Warm Card Containers */
.modern-card {
    background: #1c1412;
    border: 1px solid rgba(212, 163, 115, 0.12);
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 1rem;
    box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.35);
    transition: transform 0.15s ease, border-color 0.15s ease;
}

.modern-card:hover {
    border-color: rgba(212, 163, 115, 0.35);
}

/* Modern Warm Metric Cards */
.kpi-card {
    background: linear-gradient(145deg, #1e1614 0%, #17100e 100%);
    border: 1px solid rgba(212, 163, 115, 0.15);
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.25);
    transition: all 0.2s ease;
}

.kpi-card:hover {
    border-color: rgba(212, 163, 115, 0.4);
    transform: translateY(-2px);
}

.kpi-title {
    font-size: 0.78rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #a89f91;
    margin-bottom: 0.35rem;
}

.kpi-value {
    font-size: 1.75rem;
    font-weight: 700;
    color: #fbf8f5;
    line-height: 1.1;
    margin-bottom: 0.35rem;
}

.kpi-delta {
    font-size: 0.8rem;
    font-weight: 500;
    display: inline-flex;
    align-items: center;
    gap: 0.3rem;
}

.delta-positive {
    color: #81b29a;
}

.delta-neutral {
    color: #d4a373;
}

.delta-warning {
    color: #e07a5f;
}

/* Badges */
.badge {
    display: inline-flex;
    align-items: center;
    padding: 0.25rem 0.75rem;
    border-radius: 9999px;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}

.badge-bronze {
    background: rgba(212, 163, 115, 0.15);
    color: #d4a373;
    border: 1px solid rgba(212, 163, 115, 0.3);
}

.badge-terracotta {
    background: rgba(224, 122, 95, 0.15);
    color: #e07a5f;
    border: 1px solid rgba(224, 122, 95, 0.3);
}

.badge-sage {
    background: rgba(129, 178, 154, 0.15);
    color: #81b29a;
    border: 1px solid rgba(129, 178, 154, 0.3);
}

.badge-rust {
    background: rgba(231, 111, 81, 0.15);
    color: #e76f51;
    border: 1px solid rgba(231, 111, 81, 0.3);
}

/* Horizontal Navigation Bar Styling */
div[data-testid="stHorizontalBlock"]:has(div[data-testid="stRadio"]) {
    margin-bottom: 1.5rem;
}

div[data-testid="stRadio"] > div {
    display: flex;
    flex-direction: row;
    gap: 0.5rem;
    background: #17100e;
    padding: 0.35rem 0.5rem;
    border-radius: 12px;
    border: 1px solid rgba(212, 163, 115, 0.15);
    width: fit-content;
    margin-bottom: 1rem;
}

div[data-testid="stRadio"] label {
    background: transparent;
    padding: 0.45rem 1.1rem;
    border-radius: 8px;
    cursor: pointer;
    font-weight: 600;
    font-size: 0.88rem;
    color: #a89f91;
    transition: all 0.15s ease;
    margin: 0;
    border: 1px solid transparent;
}

div[data-testid="stRadio"] label:hover {
    color: #fbf8f5;
    background: rgba(212, 163, 115, 0.08);
}

div[data-testid="stRadio"] label[data-checked="true"],
div[data-testid="stRadio"] label:has(input:checked) {
    background: linear-gradient(135deg, #d4a373 0%, #c68b59 100%) !important;
    color: #120c0a !important;
    font-weight: 700 !important;
    box-shadow: 0 2px 8px rgba(212, 163, 115, 0.25) !important;
}

div[data-testid="stRadio"] input[type="radio"] {
    display: none;
}

/* Streamlit Inputs & Selectors in Warm Mocha */
div[data-baseweb="select"] > div {
    background-color: #1c1412 !important;
    border-color: rgba(212, 163, 115, 0.2) !important;
    color: #fbf8f5 !important;
    border-radius: 8px !important;
}

div[data-baseweb="select"] > div:hover {
    border-color: rgba(212, 163, 115, 0.5) !important;
}

/* Button in Warm Bronze */
.stButton > button {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
    border: 1px solid rgba(212, 163, 115, 0.4) !important;
    background: linear-gradient(180deg, #d4a373 0%, #b88358 100%) !important;
    color: #120c0a !important;
    transition: all 0.15s ease !important;
}

.stButton > button:hover {
    background: linear-gradient(180deg, #e0b080 0%, #d4a373 100%) !important;
    box-shadow: 0 4px 14px rgba(212, 163, 115, 0.35) !important;
    border-color: #e6ba95 !important;
}

/* Subtle sleek dividers */
hr {
    margin: 1.5rem 0 !important;
    border: none !important;
    height: 1px !important;
    background: rgba(212, 163, 115, 0.12) !important;
}

/* Hide default streamlit decorations */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""


def apply_custom_styles() -> None:
    """Inject custom warm espresso, caramel bronze & terracotta minimal styles."""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(
        family="Plus Jakarta Sans, sans-serif",
        color="#a89f91",
        size=12,
    ),
    xaxis=dict(
        gridcolor="rgba(212, 163, 115, 0.08)",
        linecolor="rgba(212, 163, 115, 0.15)",
        zerolinecolor="rgba(212, 163, 115, 0.15)",
        tickfont=dict(color="#a89f91"),
    ),
    yaxis=dict(
        gridcolor="rgba(212, 163, 115, 0.08)",
        linecolor="rgba(212, 163, 115, 0.15)",
        zerolinecolor="rgba(212, 163, 115, 0.15)",
        tickfont=dict(color="#a89f91"),
    ),
    margin=dict(l=40, r=40, t=50, b=40),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1,
        font=dict(color="#ead8c7", size=11),
    ),
    hovermode="x unified",
    hoverlabel=dict(
        bgcolor="#1c1412",
        bordercolor="rgba(212, 163, 115, 0.3)",
        font=dict(family="Plus Jakarta Sans", color="#fbf8f5", size=12),
    ),
)
