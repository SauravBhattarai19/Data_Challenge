"""
Page 1: The IGS Landscape — National Overview
Interactive choropleth of IGS scores, pillar breakdowns, sub-indicator drilldown.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st
import pandas as pd

st.set_page_config(page_title="IGS Landscape", layout="wide", page_icon="🗺️")

from components.theme import apply_theme, sidebar_nav, page_header, section_divider
from components.maps import make_national_choropleth
from components.charts import pillar_radar, county_igs_bars
from components.tables import lowest_igs_counties

apply_theme()
sidebar_nav()

from config import (
    IGS_NATIONAL, IGS_VULN_THRESHOLD,
    DELTA_COUNTY_FIPS, DELTA_COUNTY_NAMES,
    ECONOMY_SUBS, PLACE_SUBS, COMMUNITY_SUBS,
    IGS_SUB_TO_PILLAR,
)

# ── Load Data ────────────────────────────────────────────────────────────────
@st.cache_data
def load_national():
    return pd.read_parquet(IGS_NATIONAL)

df = load_national()

page_header(
    "🗺️ The IGS Landscape",
    "How do Inclusive Growth Scores vary across the United States? "
    "Explore by overall score or by Place, Community, and Economy pillars."
)

with st.expander("How to use this page", expanded=False):
    st.markdown(
        "This page gives you a **national view** of the Mastercard Inclusive Growth Score (IGS) "
        "across all US counties.\n\n"
        "- **Use the sidebar filters** to zoom into a specific state or focus only on "
        "counties that contain economically vulnerable tracts (IGS < 45).\n"
        "- **Explore the choropleth map** to see geographic patterns — hover over any county "
        "for its mean IGS score.\n"
        "- **Switch between pillars** (Economy, Place, Community) to see which dimension is "
        "driving county-level scores up or down.\n"
        "- **Bottom tables** rank the lowest-scoring counties nationally — useful starting "
        "points for identifying where intervention is most needed."
    )

# ── Filters ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Filters")
    states = sorted(df['state_name'].dropna().unique())
    selected_states = st.multiselect("Filter by state", states, default=[])
    show_below_only = st.checkbox(f"Show only counties with tracts below IGS {IGS_VULN_THRESHOLD}", value=False)

filtered = df.copy()
if selected_states:
    filtered = filtered[filtered['state_name'].isin(selected_states)]
if show_below_only:
    filtered = filtered[filtered['n_below_45'] > 0]

# ── KPIs ─────────────────────────────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)
with k1:
    st.metric("Counties Shown", f"{len(filtered):,}")
with k2:
    st.metric("Mean IGS", f"{filtered['igs_score'].mean():.1f}")
with k3:
    n_all_below = int((filtered['pct_below_45'] == 100).sum())
    st.metric("100% Below-45 Counties", f"{n_all_below:,}")
with k4:
    total_below = int(filtered['n_below_45'].sum())
    st.metric("Total Tracts Below 45", f"{total_below:,}")

section_divider()

# ── National Choropleth ──────────────────────────────────────────────────────
st.markdown("### National Map")

pillar_options = {
    'Overall IGS Score': 'igs_score',
    'Economy Pillar': 'igs_economy',
    'Place Pillar': 'igs_place',
    'Community Pillar': 'igs_community',
}
available_options = {k: v for k, v in pillar_options.items() if v in filtered.columns}

if available_options:
    map_col_label = st.radio(
        "Color map by:",
        list(available_options.keys()),
        horizontal=True,
    )
    map_col = available_options[map_col_label]

    fig = make_national_choropleth(
        filtered,
        color_col=map_col,
        color_label=map_col_label,
        title=f'{map_col_label} by County',
    )
    st.plotly_chart(fig, use_container_width=True)

section_divider()

# ── Pillar Breakdown ─────────────────────────────────────────────────────────
st.markdown("### Pillar & Sub-Indicator Breakdown")

tab_economy, tab_place, tab_community = st.tabs(["Economy", "Place", "Community"])

def show_sub_indicators(df_view, sub_list, pillar_name):
    available_subs = [s for s in sub_list if s in df_view.columns]
    if not available_subs:
        st.info(f"No {pillar_name} sub-indicator data available.")
        return

    means = df_view[available_subs].mean().sort_values(ascending=True)
    import plotly.graph_objects as go

    fig = go.Figure(go.Bar(
        y=means.index,
        x=means.values,
        orientation='h',
        marker_color='#2563eb',
        text=[f'{v:.1f}' for v in means.values],
        textposition='outside',
    ))
    fig.update_layout(
        title=f'{pillar_name} Sub-Indicators (County Mean)',
        xaxis=dict(title='Score (0-100)', range=[0, max(100, means.max() + 10)]),
        yaxis=dict(tickfont=dict(size=10)),
        height=max(250, len(available_subs) * 35 + 60),
        margin=dict(l=50, r=30, t=40, b=30),
        paper_bgcolor='#ffffff',
        plot_bgcolor='#fafbfc',
    )
    st.plotly_chart(fig, use_container_width=True)

    # Delta highlight
    delta_data = df_view[df_view['county_fips5'].isin(DELTA_COUNTY_FIPS)]
    if len(delta_data) > 0:
        delta_means = delta_data[available_subs].mean()
        national_means = df[available_subs].mean()
        comparison = pd.DataFrame({
            'Indicator': available_subs,
            'Delta Avg': [round(delta_means[s], 1) for s in available_subs],
            'National Avg': [round(national_means[s], 1) for s in available_subs],
            'Gap': [round(delta_means[s] - national_means[s], 1) for s in available_subs],
        })
        st.markdown("**MS Delta vs National:**")
        st.dataframe(comparison, use_container_width=True, hide_index=True)

with tab_economy:
    show_sub_indicators(filtered, ECONOMY_SUBS, "Economy")

with tab_place:
    show_sub_indicators(filtered, PLACE_SUBS, "Place")

with tab_community:
    show_sub_indicators(filtered, COMMUNITY_SUBS, "Community")

section_divider()

# ── Lowest IGS Counties Table ────────────────────────────────────────────────
st.markdown("### Most Vulnerable Counties (Lowest IGS)")
table = lowest_igs_counties(filtered, n=25)
st.dataframe(table, use_container_width=True, hide_index=True)

# ── Delta Spotlight ──────────────────────────────────────────────────────────
section_divider()
st.markdown("### Mississippi Delta Spotlight")

delta_counties = df[df['is_delta']].copy()
if len(delta_counties) > 0:
    delta_counties['county_name'] = delta_counties['county_fips5'].map(DELTA_COUNTY_NAMES)

    c1, c2 = st.columns([1, 1])
    with c1:
        fig = county_igs_bars(delta_counties)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        if all(c in delta_counties.columns for c in ['igs_economy', 'igs_place', 'igs_community']):
            radar_data = {
                'Economy': round(float(delta_counties['igs_economy'].mean()), 1),
                'Place': round(float(delta_counties['igs_place'].mean()), 1),
                'Community': round(float(delta_counties['igs_community'].mean()), 1),
            }
            national_radar = {
                'Economy': round(float(df['igs_economy'].mean()), 1),
                'Place': round(float(df['igs_place'].mean()), 1),
                'Community': round(float(df['igs_community'].mean()), 1),
            }
            fig = pillar_radar(radar_data, title='Delta vs National Pillars',
                              comparison=national_radar)
            st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        f"All 9 Delta counties have a mean IGS below {IGS_VULN_THRESHOLD}. "
        f"The Delta average ({delta_counties['igs_score'].mean():.1f}) is "
        f"**{df['igs_score'].mean() - delta_counties['igs_score'].mean():.1f} points** "
        f"below the national county average."
    )
