"""
Page 4: The Prescription — SHAP-Driven Gap Analysis
County/tract selector → gap analysis → priority ranking → IGS impact.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="The Prescription", layout="wide", page_icon="🎯")

from components.theme import apply_theme, sidebar_nav, page_header, section_divider
from components.charts import gap_bar_chart, pillar_radar

apply_theme()
sidebar_nav()

from config import (
    DELTA_PROFILE, PROCESSED,
    DELTA_COUNTY_FIPS, DELTA_COUNTY_NAMES,
    IGS_SUB_TO_PILLAR, IGS_PILLAR_SIZES, IGS_VULN_THRESHOLD,
    ECONOMY_SUBS, PLACE_SUBS, COMMUNITY_SUBS,
)


def _norm_shap_feature(name: str) -> str:
    s = str(name)
    for suf in ("_2025", "_2017"):
        if s.endswith(suf):
            return s[: -len(suf)]
    return s


@st.cache_data
def load_data():
    delta = pd.read_parquet(DELTA_PROFILE) if DELTA_PROFILE.exists() else None
    benchmarks_path = PROCESSED / "turnaround_benchmarks.parquet"
    benchmarks = pd.read_parquet(benchmarks_path) if benchmarks_path.exists() else None
    shap_path = PROCESSED / "shap_feature_summary.parquet"
    shap_summary = pd.read_parquet(shap_path) if shap_path.exists() else None
    return delta, benchmarks, shap_summary


delta, benchmarks, shap_summary = load_data()

if delta is None:
    st.warning("Delta profile not built. Run the build pipeline first.")
    st.stop()

page_header(
    "🎯 The Prescription",
    "For each Delta county, we compare current sub-indicator values to the turnaround benchmark — "
    "what communities that escaped IGS < 45 actually achieved. The gap tells us exactly where to invest."
)

# ── Get turnaround targets from benchmarks ───────────────────────────────────
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

st.markdown(f"### {county_name} County — Diagnosis")
st.markdown(f"**{len(county_data)} census tracts** | County FIPS: {fips}")

# ── County KPIs ──────────────────────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)
with k1:
    igs = county_data['igs_score'].mean()
    st.metric("Mean IGS Score", f"{igs:.1f}",
              delta=f"{igs - IGS_VULN_THRESHOLD:+.1f} from threshold")
with k2:
    if 'typology' in county_data.columns:
        n_stuck = int((county_data['typology'] == 'Stuck').sum())
        st.metric("Stuck Tracts", f"{n_stuck} / {len(county_data)}")
    else:
        n_below = int((county_data['igs_score'] < IGS_VULN_THRESHOLD).sum())
        st.metric("Below 45", f"{n_below} / {len(county_data)}")
with k3:
    if 'igs_economy' in county_data.columns:
        st.metric("Economy Pillar", f"{county_data['igs_economy'].mean():.1f}")
with k4:
    if 'igs_community' in county_data.columns:
        st.metric("Community Pillar", f"{county_data['igs_community'].mean():.1f}")

section_divider()

# ── Gap Analysis ─────────────────────────────────────────────────────────────
st.markdown("### Gap to Turnaround Target")
st.markdown(
    "Each bar shows how far this county's average is from what turnaround communities "
    "actually achieved. **Negative = below target** (needs improvement)."
)

all_subs = list(IGS_SUB_TO_PILLAR.keys())
gaps_data = []

for sub in all_subs:
    if sub not in county_data.columns:
        continue
    current = county_data[sub].mean()
    target = turnaround_targets.get(sub, np.nan)
    if np.isnan(target):
        continue
    gap = current - target
    pillar = IGS_SUB_TO_PILLAR[sub]
    n_in_pillar = IGS_PILLAR_SIZES.get(sub, 5)
    igs_impact = abs(gap) / n_in_pillar / 3  # approximate IGS impact if gap were closed

    gaps_data.append({
        'indicator': sub,
        'pillar': pillar,
        'current': round(current, 1),
        'target': round(target, 1),
        'gap': round(gap, 1),
        'igs_impact': round(igs_impact, 2),
    })

if gaps_data:
    gaps_df = pd.DataFrame(gaps_data).sort_values("gap")

    shap_exact = pd.Series(dtype=float)
    shap_by_base = pd.Series(dtype=float)
    if shap_summary is not None and "feature" in shap_summary.columns and "mean_abs_shap" in shap_summary.columns:
        shap_exact = shap_summary.groupby("feature")["mean_abs_shap"].max()
        tmp = shap_summary.assign(_base=shap_summary["feature"].map(_norm_shap_feature))
        shap_by_base = tmp.groupby("_base")["mean_abs_shap"].max()

    def _shap_weight(ind: str) -> float:
        if ind in shap_exact.index:
            return float(shap_exact[ind])
        nb = _norm_shap_feature(ind)
        if nb in shap_by_base.index:
            return float(shap_by_base[nb])
        return 0.0

    gaps_df["mean_abs_shap"] = gaps_df["indicator"].map(_shap_weight)
    gaps_df["priority_score"] = gaps_df["gap"].abs() * gaps_df["mean_abs_shap"]

    fig = gap_bar_chart(gaps_df, title=f"{county_name}: Gap to Turnaround Benchmark")
    st.plotly_chart(fig, use_container_width=True)

    # ── Priority Ranking ─────────────────────────────────────────────────────
    section_divider()
    st.markdown("### Priority ranking (SHAP-weighted)")
    st.markdown(
        "Among indicators **below** the turnaround benchmark, we rank by "
        "**|gap| × mean |SHAP|** so urgent gaps that also mattered most in the turnaround model rise to the top. "
        "Indicators with no SHAP match receive weight **0** and therefore score **0** regardless of gap size — "
        "see the full table below to review raw gaps for those indicators."
    )

    priority = gaps_df[gaps_df["gap"] < 0].sort_values("priority_score", ascending=False)

    if len(priority) > 0:
        for i, (_, row) in enumerate(priority.head(5).iterrows()):
            st.markdown(
                f"**{i + 1}. {row['indicator']}** ({row['pillar']} pillar)  \n"
                f"Current: {row['current']} → Target: {row['target']} → "
                f"Gap: **{row['gap']:+.1f}** → "
                f"Mean |SHAP|: **{row['mean_abs_shap']:.4f}** → "
                f"Priority score: **{row['priority_score']:.3f}** → "
                f"~{row['igs_impact']:.1f} pts IGS if closed (algebraic estimate)"
            )

        # Total potential — cap at a realistic ceiling since sub-indicators within
        # the same pillar are not independently additive (closing all at once
        # would over-count the pillar contribution).
        total_igs_gain = priority["igs_impact"].sum()
        st.success(
            f"**If all gaps were closed**, {county_name}'s IGS could increase by "
            f"approximately **{total_igs_gain:.1f} points** — potentially reaching "
            f"**{igs + total_igs_gain:.1f}** (threshold: {IGS_VULN_THRESHOLD}). "
            f"*Note: sub-indicators within the same pillar are not fully additive; "
            f"this is an upper-bound estimate.*",
            icon="📈",
        )
    else:
        st.success("This county meets or exceeds all turnaround benchmarks!")

    # ── Detailed Table ───────────────────────────────────────────────────────
    with st.expander("View full gap analysis table"):
        display = gaps_df[
            ["indicator", "pillar", "current", "target", "gap", "mean_abs_shap", "priority_score", "igs_impact"]
        ].copy()
        display.columns = [
            "Indicator",
            "Pillar",
            "Current",
            "Target",
            "Gap",
            "Mean |SHAP|",
            "Priority score",
            "IGS impact",
        ]
        st.dataframe(display, use_container_width=True, hide_index=True)

else:
    st.info("No sub-indicator data available for gap analysis. Ensure turnaround benchmarks are built.")

# ── Tract-Level View ─────────────────────────────────────────────────────────
section_divider()
st.markdown("### Individual Tract Gaps")

if 'prescription_top3' in county_data.columns:
    tract_cols = ['GEOID', 'igs_score', 'typology', 'prescription_top3']
    tract_cols = [c for c in tract_cols if c in county_data.columns]
    st.dataframe(county_data[tract_cols].sort_values('igs_score'), use_container_width=True, hide_index=True)
else:
    tract_cols = ['GEOID', 'igs_score']
    if 'typology' in county_data.columns:
        tract_cols.append('typology')

    sub_display = [s for s in all_subs if s in county_data.columns][:6]
    tract_cols += sub_display

    display = county_data[tract_cols].copy()
    for c in display.columns:
        if display[c].dtype in ['float64', 'float32']:
            display[c] = display[c].round(1)
    st.dataframe(display.sort_values('igs_score'), use_container_width=True, hide_index=True, height=400)
