"""
Page 5: Small Business Solutions — The Action Plan
Translates gap analysis into concrete small-business interventions with IGS simulator.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Small Business Solutions", layout="wide", page_icon="💡")

from components.theme import apply_theme, sidebar_nav, page_header, section_divider
from components.charts import igs_simulator_gauge

apply_theme()
sidebar_nav()

from config import (
    DELTA_PROFILE, PROCESSED,
    DELTA_COUNTY_FIPS, DELTA_COUNTY_NAMES,
    IGS_SUB_TO_PILLAR, IGS_PILLAR_SIZES, IGS_VULN_THRESHOLD,
    SMALL_BIZ_SOLUTIONS, SBA_PARQUET,
)


@st.cache_data
def load_data():
    delta = pd.read_parquet(DELTA_PROFILE) if DELTA_PROFILE.exists() else None
    bm_path = PROCESSED / 'turnaround_benchmarks.parquet'
    benchmarks = pd.read_parquet(bm_path) if bm_path.exists() else None
    return delta, benchmarks


delta, benchmarks = load_data()

if delta is None:
    st.warning("Data not available. Run the build pipeline.")
    st.stop()

page_header(
    "💡 Small Business Solutions",
    "Every gap identified in The Prescription page maps to a concrete small-business "
    "intervention. These are businesses someone can start — not policy memos."
)

# ── Get turnaround targets ───────────────────────────────────────────────────
turnaround_targets = {}
if benchmarks is not None:
    turn_bm = benchmarks[benchmarks['typology'] == 'Turnaround']
    for _, row in turn_bm.iterrows():
        turnaround_targets[row['indicator']] = row['mean_2025']

# ── County Selector ──────────────────────────────────────────────────────────
county_options = {f"{name} ({fips})": fips for fips, name in sorted(DELTA_COUNTY_NAMES.items(), key=lambda x: x[1])}
selected = st.selectbox("Select a Delta county", list(county_options.keys()))
fips = county_options[selected]
county_name = DELTA_COUNTY_NAMES[fips]
county_data = delta[delta['county_fips5'] == fips]

if len(county_data) == 0:
    st.warning("No data for this county.")
    st.stop()

# ── Compute gaps ─────────────────────────────────────────────────────────────
gaps = []
for sub in IGS_SUB_TO_PILLAR:
    if sub not in county_data.columns:
        continue
    current = county_data[sub].mean()
    target = turnaround_targets.get(sub, np.nan)
    if np.isnan(target):
        continue
    gap = current - target
    gaps.append({
        'indicator': sub,
        'pillar': IGS_SUB_TO_PILLAR[sub],
        'current': current,
        'target': target,
        'gap': gap,
        'n_in_pillar': IGS_PILLAR_SIZES.get(sub, 5),
    })

gaps_df = pd.DataFrame(gaps).sort_values('gap') if gaps else pd.DataFrame()

current_igs = county_data['igs_score'].mean()

# ── Top Gaps with Solutions ──────────────────────────────────────────────────
st.markdown(f"### {county_name} County: Where to Invest")
st.markdown(f"Current mean IGS: **{current_igs:.1f}** | Threshold: **{IGS_VULN_THRESHOLD}**")

if len(gaps_df) > 0:
    negative_gaps = gaps_df[gaps_df['gap'] < 0].sort_values('gap')

    for _, row in negative_gaps.head(6).iterrows():
        indicator = row['indicator']
        solution = SMALL_BIZ_SOLUTIONS.get(indicator, None)

        gap_label = solution['gap_label'] if solution else indicator
        gap_val = row['gap']
        igs_impact = abs(gap_val) / row['n_in_pillar'] / 3

        st.markdown(f"#### {gap_label}")

        c1, c2 = st.columns([1, 1])

        with c1:
            st.markdown(
                f"- **Current:** {row['current']:.1f}  \n"
                f"- **Turnaround Target:** {row['target']:.1f}  \n"
                f"- **Gap:** {gap_val:+.1f}  \n"
                f"- **Pillar:** {row['pillar']}  \n"
                f"- **Potential IGS lift:** ~{igs_impact:.1f} points"
            )

        with c2:
            if solution:
                st.markdown("**Small businesses that can close this gap:**")
                for biz in solution['businesses']:
                    st.markdown(f"- {biz}")
                st.caption(f"🔗 Mastercard link: {solution['mastercard_link']}")
            else:
                st.markdown("*(No specific business mapping for this indicator)*")

        st.markdown("---")

# ── IGS Investment Simulator ─────────────────────────────────────────────────
section_divider()
st.markdown("### IGS Investment Simulator")
st.markdown(
    "Use the sliders below to simulate sub-indicator improvements. "
    "The IGS impact is computed using Mastercard's pillar algebra: "
    "ΔIGS = (Δsub / n_in_pillar) / 3."
)

total_igs_delta = 0.0
adjustments = {}

# Show sliders only for the top gaps
slider_indicators = []
if len(gaps_df) > 0:
    slider_indicators = gaps_df[gaps_df['gap'] < 0].sort_values('gap').head(8)['indicator'].tolist()

if slider_indicators:
    cols = st.columns(2)
    for i, indicator in enumerate(slider_indicators):
        row = gaps_df[gaps_df['indicator'] == indicator].iloc[0]
        current_val = row['current']
        gap_val = row['gap']
        max_improvement = min(abs(gap_val) * 1.5, 100 - current_val)

        solution = SMALL_BIZ_SOLUTIONS.get(indicator, {})
        label = solution.get('gap_label', indicator) if solution else indicator

        with cols[i % 2]:
            improvement = st.slider(
                f"{label}",
                min_value=0.0,
                max_value=round(max_improvement, 0),
                value=0.0,
                step=1.0,
                help=f"Current: {current_val:.1f}, Target: {row['target']:.1f}, Gap: {gap_val:+.1f}",
                key=f"slider_{indicator}",
            )
            if improvement > 0:
                n_in_pillar = row['n_in_pillar']
                igs_change = improvement / n_in_pillar / 3
                total_igs_delta += igs_change
                adjustments[indicator] = {
                    'improvement': improvement,
                    'igs_change': igs_change,
                    'pillar': row['pillar'],
                }

    # ── Show result ──────────────────────────────────────────────────────────
    section_divider()

    projected_igs = current_igs + total_igs_delta

    c1, c2 = st.columns([1, 1])

    with c1:
        fig = igs_simulator_gauge(current_igs, projected_igs)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown(f"**Current IGS:** {current_igs:.1f}")
        st.markdown(f"**Projected IGS:** {projected_igs:.1f}")
        st.markdown(f"**Total improvement:** {total_igs_delta:+.1f} points")

        if projected_igs >= IGS_VULN_THRESHOLD:
            st.success(
                f"With these improvements, {county_name} would cross the "
                f"IGS {IGS_VULN_THRESHOLD} threshold!",
                icon="🎉",
            )
        else:
            remaining = IGS_VULN_THRESHOLD - projected_igs
            st.warning(
                f"Still {remaining:.1f} points below the threshold. "
                f"Additional interventions needed.",
                icon="📊",
            )

        if adjustments:
            st.markdown("**Breakdown:**")
            for ind, adj in adjustments.items():
                solution = SMALL_BIZ_SOLUTIONS.get(ind, {})
                label = solution.get('gap_label', ind) if solution else ind
                st.markdown(
                    f"- {label}: +{adj['improvement']:.0f} → IGS +{adj['igs_change']:.2f} "
                    f"({adj['pillar']} pillar)"
                )

else:
    st.info("No gap data available. Run the full analysis pipeline.")

# ── The Mastercard Thesis ────────────────────────────────────────────────────
section_divider()
st.markdown("### The Mastercard Connection")
st.info(
    "**Every small business loan that creates a healthcare SME directly improves "
    "the Economy pillar** (through Small Business Loans Score and Commercial Diversity Score) "
    "**AND indirectly reduces health burden** (through better chronic disease management → "
    "higher labor participation → higher income → higher spending).\n\n"
    "This is the virtuous cycle:\n\n"
    "**Mastercard loan → Healthcare small business → Managed chronic disease → "
    "Higher labor participation → Higher income → Higher spending → "
    "IGS improvement across ALL three pillars.**\n\n"
    "The Inclusive Growth Score is not just a measurement tool — it is a roadmap "
    "for where Mastercard's investments can have the greatest impact.",
    icon="💳",
)

st.markdown(
    f"For **{county_name} County**, the data shows that the highest-impact investments are in "
    f"the indicators where the gap to turnaround is largest. Small businesses that address "
    f"**broadband access**, **commercial diversity**, and **healthcare services** are the "
    f"most promising paths to crossing IGS {IGS_VULN_THRESHOLD}."
)

# ── Mississippi SBA Context ──────────────────────────────────────────────────
if SBA_PARQUET.exists():
    sba = pd.read_parquet(SBA_PARQUET)
    ms = sba[sba['state_abbr'] == 'MS']
    if len(ms) > 0:
        section_divider()
        st.markdown("### Mississippi's Small Business Reality (SBA 2025)")

        ms_row = ms.iloc[0]
        metrics = [
            ('n_small_businesses', 'Small Businesses'),
            ('sb_employment_share_pct', 'SB Employment Share (%)'),
            ('establishments_opened', 'New Establishments Opened'),
            ('net_new_jobs', 'Net New Jobs (SB)'),
            ('women_owned_pct', 'Women-Owned (%)'),
            ('hispanic_owned_pct', 'Hispanic-Owned (%)'),
        ]
        sba_data = []
        for col, label in metrics:
            val = ms_row.get(col, None)
            rank = ms_row.get(f'{col}_rank', None)
            if val is not None and not pd.isna(val):
                sba_data.append({
                    'Metric': label,
                    'Mississippi': f"{val:,.0f}" if val > 100 else f"{val:.1f}%",
                    'Rank (of 51)': f"{rank:.0f}" if rank and not pd.isna(rank) else '-',
                })
        if sba_data:
            st.dataframe(pd.DataFrame(sba_data), use_container_width=True, hide_index=True)
            st.caption(
                "Source: U.S. Small Business Administration, State Small Business Statistics & Rankings 2025"
            )
