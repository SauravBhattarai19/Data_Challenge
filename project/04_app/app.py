"""
Healthy Economies, Healthy Communities
Mastercard IGS × AUC 2026 Data Science Challenge

Home page — key statistics and narrative framing.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(
    page_title="Healthy Economies, Healthy Communities",
    layout="wide",
    page_icon="📊",
    initial_sidebar_state="expanded",
)

from components.theme import apply_theme, sidebar_nav
apply_theme()
sidebar_nav()

from config import (
    IGS_NATIONAL, PROCESSED, DELTA_COUNTY_FIPS,
    IGS_VULN_THRESHOLD, DELTA_COUNTY_NAMES, SBA_PARQUET,
    IGS_TRENDS_SUMMARY,
)

# ── Data Loading ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading data...")
def load_data():
    if not IGS_NATIONAL.exists():
        return None, None, None, None
    national = pd.read_parquet(IGS_NATIONAL)

    typ_path = PROCESSED / 'community_typology.parquet'
    typology = pd.read_parquet(typ_path) if typ_path.exists() else None

    sba = pd.read_parquet(SBA_PARQUET) if SBA_PARQUET.exists() else None

    trends = pd.read_parquet(IGS_TRENDS_SUMMARY) if IGS_TRENDS_SUMMARY.exists() else None

    return national, typology, sba, trends


national, typology, sba, trends = load_data()

if national is None:
    st.error("Data unavailable. Please contact the team.")
    st.stop()

# ── Compute statistics ───────────────────────────────────────────────────────
total_tracts     = int(national['n_tracts'].sum())
total_below_45   = int(national['n_below_45'].sum())
pct_below_45     = round(total_below_45 / total_tracts * 100, 1)
national_mean_igs = round(float(national['igs_score'].mean()), 1)

delta = national[national['is_delta']]
delta_mean_igs = round(float(delta['igs_score'].mean()), 1) if len(delta) > 0 else 0

# Typology stats
if typology is not None and 'typology' in typology.columns:
    n_stuck_national    = int((typology['typology'] == 'Stuck').sum())
    n_turnaround_national = int((typology['typology'] == 'Turnaround').sum())
    if 'county_fips5' in typology.columns:
        delta_typ = typology[typology['county_fips5'].isin(DELTA_COUNTY_FIPS)]
        if len(delta_typ) > 0:
            delta_stuck_pct  = round(float((delta_typ['typology'] == 'Stuck').mean() * 100))
            delta_turnaround = int((delta_typ['typology'] == 'Turnaround').sum())
            delta_total_typ  = len(delta_typ)
        else:
            delta_stuck_pct = delta_turnaround = delta_total_typ = 0
    else:
        delta_stuck_pct = delta_turnaround = delta_total_typ = 0
else:
    n_stuck_national = n_turnaround_national = 0
    delta_stuck_pct = delta_turnaround = delta_total_typ = 0

# ── Hero Section ─────────────────────────────────────────────────────────────
st.markdown("### Mastercard IGS × AUC 2026 Data Science Challenge")
st.markdown("# Healthy Economies, Healthy Communities")
st.markdown(
    f"A data-driven investigation across **{total_tracts:,} US census tracts** using "
    f"the **Mastercard Inclusive Growth Score** and **10 public datasets** to uncover "
    f"why some communities escape economic vulnerability while others remain stuck — "
    f"and what **targeted solutions** can change that."
)
st.caption("Jackson State University · Grand Finale April 30, 2026")

st.markdown(
    "**Live app:** [howtoimprove.streamlit.app](https://howtoimprove.streamlit.app) &nbsp;|&nbsp; "
    "**Code:** [github.com/SauravBhattarai19/Data_Challenge](https://github.com/SauravBhattarai19/Data_Challenge)"
)

st.divider()

# ── Key Statistics ───────────────────────────────────────────────────────────
st.markdown("### The Numbers That Matter")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="Tracts with IGS < 45",
        value=f"{total_below_45:,}",
        delta=f"{pct_below_45}% of all US tracts",
    )

with col2:
    st.metric(
        label="National Mean IGS",
        value=f"{national_mean_igs}",
    )

with col3:
    st.metric(
        label="MS Delta Mean IGS",
        value=f"{delta_mean_igs}",
        delta=f"{delta_mean_igs - national_mean_igs:+.1f} vs national",
    )

with col4:
    if typology is not None and delta_total_typ > 0:
        st.metric(
            label="Delta Tracts Still Stuck",
            value=f"{delta_stuck_pct}%",
            delta=f"Only {delta_turnaround} of {delta_total_typ} turned around",
        )
    else:
        st.metric(
            label="Delta Counties",
            value="9",
            delta="All below IGS 45",
        )

# ── IGS Trajectory ───────────────────────────────────────────────────────────
if trends is not None and len(trends) > 0:
    st.divider()
    st.markdown("### IGS Trajectory: National, Mississippi, and Delta")

    fig_t = go.Figure()
    fig_t.add_trace(go.Scatter(
        x=trends["year"], y=trends["nat_mean"],
        mode="lines+markers", name="National",
        line=dict(color="#2563eb"),
    ))
    fig_t.add_trace(go.Scatter(
        x=trends["year"], y=trends["ms_mean"],
        mode="lines+markers", name="Mississippi",
        line=dict(color="#ca8a04"),
    ))
    fig_t.add_trace(go.Scatter(
        x=trends["year"], y=trends["delta_mean"],
        mode="lines+markers", name="MS Delta (9 counties)",
        line=dict(color="#dc2626"),
    ))
    fig_t.add_hline(
        y=IGS_VULN_THRESHOLD,
        line_dash="dash", line_color="#64748b",
        annotation_text=f"Vulnerability threshold ({IGS_VULN_THRESHOLD})",
        annotation_position="bottom right",
    )
    fig_t.update_layout(
        xaxis_title="Year",
        yaxis_title="Mean IGS Score",
        hovermode="x unified",
        paper_bgcolor="#ffffff",
        plot_bgcolor="#fafbfc",
        height=420,
        margin=dict(l=48, r=24, t=24, b=48),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_t, use_container_width=True)

    ly  = int(trends["year"].max())
    row = trends[trends["year"] == ly]
    if len(row) > 0:
        n_ly = float(row["nat_mean"].iloc[0])
        d_ly = float(row["delta_mean"].iloc[0])
        st.caption(
            f"In **{ly}**, the Delta mean IGS ({d_ly:.1f}) was **{n_ly - d_ly:.1f} points** "
            f"below the national mean ({n_ly:.1f}) — and has remained persistently below the "
            f"vulnerability threshold of {IGS_VULN_THRESHOLD} throughout the study period."
        )

# ── Mississippi Small Business Context ───────────────────────────────────────
if sba is not None:
    ms = sba[sba['state_abbr'] == 'MS']
    if len(ms) > 0:
        ms_row = ms.iloc[0]
        st.divider()
        st.markdown("### Mississippi's Small Business Landscape (SBA 2025)")
        sb1, sb2, sb3, sb4 = st.columns(4)
        with sb1:
            val  = ms_row.get('n_small_businesses', None)
            rank = ms_row.get('n_small_businesses_rank', None)
            if val and not pd.isna(val):
                st.metric("Small Businesses", f"{val:,.0f}", delta=f"Rank: {rank:.0f}/51")
        with sb2:
            val  = ms_row.get('net_new_jobs', None)
            rank = ms_row.get('net_new_jobs_rank', None)
            if val and not pd.isna(val):
                st.metric("Net New Jobs (Small Biz)", f"{val:,.0f}", delta=f"Rank: {rank:.0f}/51")
        with sb3:
            val  = ms_row.get('women_owned_pct', None)
            rank = ms_row.get('women_owned_pct_rank', None)
            if val and not pd.isna(val):
                st.metric("Women-Owned (%)", f"{val:.1f}%", delta=f"Rank: {rank:.0f}/51")
        with sb4:
            val  = ms_row.get('sb_employment_share_pct', None)
            rank = ms_row.get('sb_employment_share_pct_rank', None)
            if val and not pd.isna(val):
                st.metric("Small Biz Employment Share", f"{val:.0f}%", delta=f"Rank: {rank:.0f}/51")

st.divider()

# ── The Thesis ───────────────────────────────────────────────────────────────
st.markdown("### Our Approach")

st.info(
    "**The Question:** Among communities with IGS below 45, what separates those "
    "that improved (\"turnaround\") from those that stayed stuck between 2017 and 2025?\n\n"
    "**The Method:** We trained an ensemble of machine learning models and used SHAP "
    "analysis across 89 granular features to identify the true drivers of community recovery.\n\n"
    "**The Finding:** Turnaround is predicted by **chronic disease burden, dental access, "
    "broadband connectivity, and commercial ecosystem diversity** — not by broad composite "
    "indices. This insight rewrites the investment playbook for stuck communities.\n\n"
    "**The Solution:** Data-driven, county-specific gap analysis that shows policymakers and "
    "investors exactly where to act first.",
    icon="💡",
)

# ── Navigation ───────────────────────────────────────────────────────────────
st.markdown("### Explore the Story")

c1, c2 = st.columns(2)
with c1:
    st.markdown("**🗺️ 1. IGS Landscape**")
    st.markdown("National overview of Inclusive Growth Scores by Place, Community, and Economy pillars.")

    st.markdown("")

    st.markdown("**🧠 3. What Drives Turnaround?**")
    st.markdown("SHAP feature importance, beeswarm plots, and domain breakdowns across 89 features.")

with c2:
    st.markdown("**🔍 2. The Delta Story**")
    st.markdown("9-county Mississippi Delta: every indicator, every dataset, from county summaries to individual tracts.")

    st.markdown("")

    st.markdown("**🎯 4. Investment Priority Matrix**")
    st.markdown("County-by-county gap analysis — see exactly which features to act on first, and which strengths to protect.")

st.divider()
