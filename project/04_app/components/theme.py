"""
Clean modern theme for the Streamlit app.
Streamlit-native approach: minimal custom CSS, professional and readable.
"""

import streamlit as st


def apply_theme() -> None:
    """Inject minimal, clean CSS."""
    st.markdown("""
<style>
[data-testid="stToolbar"] [data-testid="stToolbarActionButton"] { display: none !important; }
#MainMenu, footer { visibility: hidden; }
[data-testid="stSidebarNav"] { display: none !important; }

.block-container {
    padding-top: 1.2rem !important;
    padding-bottom: 2rem !important;
    max-width: 1200px !important;
}

[data-testid="stSidebar"] {
    background: #fafbfc !important;
    border-right: 1px solid #e5e7eb !important;
}

[data-testid="stMetric"] {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 12px;
    padding: 16px 20px !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}

[data-baseweb="tab-list"] {
    background: #f3f4f6 !important;
    border-radius: 10px !important;
    padding: 3px !important;
    border-bottom: none !important;
}
[data-baseweb="tab"] {
    border-radius: 8px !important;
    font-weight: 500 !important;
}
[aria-selected="true"][data-baseweb="tab"] {
    background: #2563eb !important;
    color: #ffffff !important;
}

[data-testid="stExpander"] {
    border: 1px solid #e5e7eb !important;
    border-radius: 10px !important;
    box-shadow: 0 1px 2px rgba(0,0,0,0.03);
}

[data-testid="stDataFrame"] {
    border-radius: 10px !important;
    border: 1px solid #e5e7eb !important;
}
</style>
""", unsafe_allow_html=True)


def sidebar_nav() -> None:
    """Sidebar navigation using native Streamlit components."""
    with st.sidebar:
        st.markdown("""
<div style='text-align:center;padding:8px 0 4px'>
  <div style='font-size:0.95rem;font-weight:800;color:#1e293b;letter-spacing:-0.3px'>
    Healthy Economies,<br>Healthy Communities
  </div>
  <div style='font-size:0.65rem;color:#6b7280;margin-top:4px;
    letter-spacing:1px;text-transform:uppercase'>
    Mastercard IGS &times; AUC 2026
  </div>
</div>
<hr style='border:none;border-top:1px solid #e5e7eb;margin:8px 0'>
""", unsafe_allow_html=True)

        st.page_link("app.py", label="Home", icon="🏠")
        st.page_link("pages/1_IGS_Landscape.py", label="1. IGS Landscape", icon="🗺️")
        st.page_link("pages/2_Delta_Deep_Dive.py", label="2. Delta Deep Dive", icon="🔍")
        st.page_link("pages/3_ML_Discovery.py", label="3. What Drives Turnaround?", icon="🧠")
        st.page_link("pages/4_The_Prescription.py", label="4. The Prescription", icon="🎯")
        st.page_link("pages/5_Small_Business_Solutions.py", label="5. Small Business Solutions", icon="💡")

        st.markdown("""
<hr style='border:none;border-top:1px solid #e5e7eb;margin:12px 0'>
<div style='font-size:0.65rem;color:#6b7280;line-height:1.6;text-align:center;padding:0 4px'>
  <div style='font-weight:600;color:#374151;margin-bottom:2px'>Jackson State University</div>
  Created by<br>
  <span style='color:#1e293b;font-weight:600'>Saurav Bhattarai &amp; Richa Pokhrel</span><br>
  Under <span style='color:#1e293b;font-weight:600'>Dr. Rocky Talchabhadel</span>
</div>
""", unsafe_allow_html=True)


def page_header(title: str, subtitle: str = "") -> None:
    """Clean page header."""
    st.markdown(f"## {title}")
    if subtitle:
        st.caption(subtitle)


def section_divider(label: str = "") -> None:
    """Thin divider with optional label."""
    if label:
        st.markdown(f"---")
        st.markdown(f"**{label}**")
    else:
        st.markdown("---")


def metric_card(label: str, value, delta=None, delta_label: str = "") -> None:
    """Wrapper for st.metric with consistent formatting."""
    if delta is not None:
        st.metric(label=label, value=value, delta=f"{delta:+.1f} {delta_label}" if isinstance(delta, (int, float)) else delta)
    else:
        st.metric(label=label, value=value)


def plotly_layout(**overrides) -> dict:
    """Standard Plotly layout for clean charts."""
    base = dict(
        paper_bgcolor='#ffffff',
        plot_bgcolor='#fafbfc',
        font=dict(color='#374151', family='Inter, system-ui, sans-serif'),
        title_font_color='#111827',
        margin=dict(l=40, r=40, t=50, b=40),
    )
    base.update(overrides)
    return base
