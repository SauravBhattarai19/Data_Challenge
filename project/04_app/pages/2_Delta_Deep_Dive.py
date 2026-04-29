"""
Page 2: Delta Deep Dive
County-level cards with all datasets, tract drill-down, Folium map, Delta vs national.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Delta Deep Dive", layout="wide", page_icon="🔍")

from components.theme import apply_theme, sidebar_nav, page_header, section_divider
from components.charts import pillar_radar, county_igs_bars
from components.tables import delta_county_summary

apply_theme()
sidebar_nav()

from config import (
    DELTA_PROFILE, DELTA_GEOJSON, PROCESSED,
    DELTA_COUNTY_FIPS, DELTA_COUNTY_NAMES,
    IGS_SUB_TO_PILLAR, IGS_VULN_THRESHOLD,
    ECONOMY_SUBS, PLACE_SUBS, COMMUNITY_SUBS,
    ZBP_PARQUET,
    FQHC_PARQUET,
)


@st.cache_data
def load_delta():
    if not DELTA_PROFILE.exists():
        return None
    return pd.read_parquet(DELTA_PROFILE)


@st.cache_data
def load_fqhc_sites():
    if not FQHC_PARQUET.exists():
        return None
    return pd.read_parquet(FQHC_PARQUET)


delta = load_delta()

if delta is None:
    st.error("Delta profile data is not available.")
    st.stop()

page_header(
    "🔍 The Delta Story",
    "A deep dive into the 9-county Mississippi Delta — every indicator, every dataset, "
    "from county summaries down to individual census tracts."
)

with st.expander("How to use this page", expanded=False):
    st.markdown(
        "This page lets you **explore every available data dimension** for the Mississippi Delta "
        "at both the county and census-tract level.\n\n"
        "- **County Overview table** at the top summarizes IGS scores and pillar values for "
        "all 9 Delta counties side by side.\n"
        "- **Select a county** from the dropdown to drill into that county's tracts, "
        "or keep 'All Delta' for the regional picture.\n"
        "- **IGS Pillar radar chart** shows how the selected county compares to the national "
        "average across Economy, Place, and Community.\n"
        "- **Tabs (Health Outcomes, Healthcare Access, Social Vulnerability, Business)** "
        "show the underlying dataset detail for the selected area.\n"
        "- **Census Tract table** ranks all tracts by IGS score so you can identify the "
        "most distressed neighborhoods.\n"
        "- **Tract Map** colors each census tract by your chosen indicator — "
        "toggle the FQHC overlay to see where healthcare clinics are located."
    )

# ── County Summary ───────────────────────────────────────────────────────────
st.markdown("### County Overview")

county_table = delta_county_summary(delta, DELTA_COUNTY_NAMES)
st.dataframe(county_table, use_container_width=True, hide_index=True)

# ── County Selector ──────────────────────────────────────────────────────────
section_divider()

county_options = ['All Delta'] + [f"{name} ({fips})" for fips, name in sorted(DELTA_COUNTY_NAMES.items(), key=lambda x: x[1])]

selected_county = st.selectbox("Select a county for detailed view", county_options)

if selected_county == 'All Delta':
    view = delta.copy()
    county_label = "All Delta"
else:
    fips = selected_county.split('(')[1].rstrip(')')
    view = delta[delta['county_fips5'] == fips].copy()
    county_label = DELTA_COUNTY_NAMES.get(fips, selected_county)

st.markdown(f"### {county_label} — Detailed Profile")
st.markdown(f"**{len(view)} census tracts**")

# ── KPIs for selected county ─────────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)

with k1:
    mean_igs = view['igs_score'].mean() if 'igs_score' in view.columns else 0
    nat_mean = delta.iloc[0].get('national_mean_igs_score', 50) if 'national_mean_igs_score' in delta.columns else 50
    st.metric("Mean IGS Score", f"{mean_igs:.1f}", delta=f"{mean_igs - nat_mean:+.1f} vs national")

with k2:
    if 'igs_economy' in view.columns:
        st.metric("Economy Pillar", f"{view['igs_economy'].mean():.1f}")
with k3:
    if 'igs_place' in view.columns:
        st.metric("Place Pillar", f"{view['igs_place'].mean():.1f}")
with k4:
    if 'igs_community' in view.columns:
        st.metric("Community Pillar", f"{view['igs_community'].mean():.1f}")

# ── Pillar Radar ─────────────────────────────────────────────────────────────
section_divider("IGS Pillars & Sub-Indicators")

c1, c2 = st.columns([1, 1])

with c1:
    pillars = {}
    national_pillars = {}
    for pillar, col in [('Economy', 'igs_economy'), ('Place', 'igs_place'), ('Community', 'igs_community')]:
        if col in view.columns:
            pillars[pillar] = round(float(view[col].mean()), 1)
            nat_col = f'national_mean_{col}'
            if nat_col in view.columns:
                national_pillars[pillar] = round(float(view[nat_col].iloc[0]), 1)

    if pillars:
        fig = pillar_radar(pillars, title=f'{county_label} Pillars',
                          comparison=national_pillars if national_pillars else None,
                          comp_label='National Average')
        st.plotly_chart(fig, use_container_width=True)

with c2:
    st.markdown("**Sub-Indicator Averages:**")

    for pillar_name, sub_list in [('Economy', ECONOMY_SUBS), ('Place', PLACE_SUBS), ('Community', COMMUNITY_SUBS)]:
        available = [s for s in sub_list if s in view.columns]
        if available:
            with st.expander(f"{pillar_name} ({len(available)} indicators)"):
                for sub in available:
                    val = view[sub].mean()
                    nat_col = f'national_mean_{sub}'
                    nat_val = view[nat_col].iloc[0] if nat_col in view.columns else None
                    gap_str = f" (gap: {val - nat_val:+.1f})" if nat_val is not None else ""
                    st.markdown(f"- **{sub}**: {val:.1f}{gap_str}")

# ── Health & Social Indicators ───────────────────────────────────────────────
section_divider("Health & Social Context")

tab_health, tab_access, tab_social, tab_biz = st.tabs(
    ["Health Outcomes", "Healthcare Access", "Social Vulnerability", "Business Infrastructure"]
)

with tab_health:
    health_cols = [c for c in view.columns if 'CrudePrev' in c]
    if health_cols:
        health_data = []
        for c in sorted(health_cols):
            val = view[c].mean()
            nat_col = f'national_mean_{c}'
            nat_val = view[nat_col].iloc[0] if nat_col in view.columns else None
            label = c.replace('_CrudePrev', '').replace('_', ' ')
            row = {'Condition': label, f'{county_label} Avg (%)': round(val, 1)}
            if nat_val is not None:
                row['National Avg (%)'] = round(nat_val, 1)
                row['Gap (pp)'] = round(val - nat_val, 1)
            health_data.append(row)
        st.dataframe(pd.DataFrame(health_data), use_container_width=True, hide_index=True)
    else:
        st.info("No CDC PLACES health data available in Delta profile.")

with tab_access:
    access_cols = {
        'pc_hpsa_score_max': 'Primary Care HPSA Score (0-26)',
        'mh_hpsa_score_max': 'Mental Health HPSA Score (0-26)',
        'in_pc_hpsa': 'In Primary Care Shortage Area',
        'in_mh_hpsa': 'In Mental Health Shortage Area',
        'in_mua': 'In Medically Underserved Area',
        'fqhc_count': 'FQHC Sites in County',
    }
    access_data = []
    for col, label in access_cols.items():
        if col in view.columns:
            val = view[col].mean()
            access_data.append({'Metric': label, 'Value': round(val, 1)})
    if access_data:
        st.dataframe(pd.DataFrame(access_data), use_container_width=True, hide_index=True)
    else:
        st.info("No healthcare access data available.")

with tab_social:
    svi_cols = {
        'RPL_THEMES': 'SVI Overall Percentile',
        'RPL_THEME1': 'SVI: Socioeconomic Status',
        'RPL_THEME2': 'SVI: Household Characteristics',
        'RPL_THEME3': 'SVI: Racial/Ethnic Minority',
        'RPL_THEME4': 'SVI: Housing & Transportation',
        'PovertyRate': 'Poverty Rate (%)',
    }
    svi_data = []
    for col, label in svi_cols.items():
        if col in view.columns:
            val = view[col].mean()
            svi_data.append({'Metric': label, f'{county_label}': round(val, 2)})
    if svi_data:
        st.dataframe(pd.DataFrame(svi_data), use_container_width=True, hide_index=True)

with tab_biz:
    biz_cols = [c for c in view.columns if c.startswith('biz_')]
    if biz_cols:
        biz_data = []
        for c in sorted(biz_cols):
            label = c.replace('biz_', '').replace('_', ' ').title()
            val = view[c].mean()
            biz_data.append({'Business Type': label, 'Count (County Avg)': round(val, 1)})
        st.dataframe(pd.DataFrame(biz_data), use_container_width=True, hide_index=True)
    else:
        st.info("No county-level business data available.")

    # ZIP-level business data
    if ZBP_PARQUET.exists():
        zbp = pd.read_parquet(ZBP_PARQUET)
        delta_zbp = zbp[zbp['is_delta']]
        if len(delta_zbp) > 0:
            st.markdown("**ZIP-Level Business Landscape (Delta ZIPs):**")
            zk1, zk2, zk3, zk4 = st.columns(4)
            with zk1:
                st.metric("Delta ZIP Establishments", f"{int(delta_zbp['est'].sum()):,}")
            with zk2:
                st.metric("Delta ZIP Employment", f"{int(delta_zbp['emp'].sum()):,}")
            with zk3:
                st.metric("Avg Sectors per ZIP", f"{delta_zbp['n_sectors'].mean():.1f}")
            with zk4:
                st.metric("Healthcare Establishments", f"{int(delta_zbp['healthcare_establishments'].sum()):,}")

            with st.expander("View Delta ZIP details"):
                zbp_display = delta_zbp[['zip', 'city', 'cty_name', 'est', 'emp',
                                         'small_biz_count', 'n_sectors',
                                         'healthcare_establishments']].copy()
                zbp_display.columns = ['ZIP', 'City', 'County', 'Establishments',
                                       'Employment', 'Small Biz (<20 emp)',
                                       'Sectors', 'Healthcare Est.']
                st.dataframe(zbp_display.sort_values('Employment', ascending=False),
                            use_container_width=True, hide_index=True)

# ── Tract-Level Detail ───────────────────────────────────────────────────────
section_divider("Census Tract Detail")

tract_display_cols = ['GEOID', 'county_name', 'igs_score']
if 'igs_economy' in view.columns:
    tract_display_cols.append('igs_economy')
if 'igs_place' in view.columns:
    tract_display_cols.append('igs_place')
if 'igs_community' in view.columns:
    tract_display_cols.append('igs_community')
if 'typology' in view.columns:
    tract_display_cols.append('typology')

available_display = [c for c in tract_display_cols if c in view.columns]
tract_table = view[available_display].sort_values('igs_score', ascending=True)

for c in ['igs_score', 'igs_economy', 'igs_place', 'igs_community']:
    if c in tract_table.columns:
        tract_table[c] = tract_table[c].round(1)

st.dataframe(tract_table, use_container_width=True, hide_index=True, height=400)

# ── Folium Map ───────────────────────────────────────────────────────────────
section_divider("Tract Map")

geojson_path = DELTA_GEOJSON if DELTA_GEOJSON.exists() else None
if geojson_path:
    from components.maps import make_delta_folium_map
    from streamlit_folium import st_folium

    map_color_options = {
        "IGS Score": "igs_score",
        "Economy Pillar": "igs_economy",
        "Place Pillar": "igs_place",
        "Community Pillar": "igs_community",
    }
    if "typology" in view.columns:
        map_color_options["Community typology"] = "typology"
    if "pc_hpsa_score_max" in view.columns:
        map_color_options["Primary care HPSA (max score, 0–26)"] = "pc_hpsa_score_max"
    map_color_options = {k: v for k, v in map_color_options.items() if v in view.columns}

    fqhc_raw = load_fqhc_sites()
    fqhc_delta = None
    if fqhc_raw is not None and "county_fips5" in fqhc_raw.columns:
        fqhc_delta = fqhc_raw[fqhc_raw["county_fips5"].isin(DELTA_COUNTY_FIPS)].copy()
        if selected_county != "All Delta":
            fqhc_delta = fqhc_delta[fqhc_delta["county_fips5"] == fips]
        if "lat" in fqhc_delta.columns and "lon" in fqhc_delta.columns:
            fqhc_delta = fqhc_delta.dropna(subset=["lat", "lon"])
        else:
            fqhc_delta = None

    show_fqhc = False
    if fqhc_delta is not None and len(fqhc_delta) > 0:
        show_fqhc = st.checkbox(
            f"Show FQHC sites ({len(fqhc_delta)} in this view)",
            value=True,
            key="delta_map_fqhc",
        )

    if map_color_options:
        map_metric = st.radio("Map color:", list(map_color_options.keys()), horizontal=True)
        col_name = map_color_options[map_metric]

        m = make_delta_folium_map(
            view,
            geojson_path=geojson_path,
            color_col=col_name,
            color_label=map_metric,
            fqhc_df=fqhc_delta if show_fqhc else None,
            show_fqhc=show_fqhc,
        )
        st_folium(m, use_container_width=True, height=500, returned_objects=[])
else:
    st.info("Tract-level map data is not available for this view.")
