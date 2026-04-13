"""
Page 3: What Drives Turnaround? — ML Framework & Results
Model comparison, SHAP dimensions, turnaround vs stuck profiles.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="What Drives Turnaround?", layout="wide", page_icon="🧠")

from components.theme import apply_theme, sidebar_nav, page_header, section_divider
from components.charts import (
    model_comparison_chart, shap_importance_chart,
    shap_dimension_donut, turnaround_vs_stuck_bars,
)

apply_theme()
sidebar_nav()

from config import PROCESSED, IGS_VULN_THRESHOLD, IGS_SUB_TO_PILLAR

# ── Load Data ────────────────────────────────────────────────────────────────
@st.cache_data
def load_ml_data():
    data = {}
    files = {
        'models': 'model_comparison.parquet',
        'shap_summary': 'shap_feature_summary.parquet',
        'shap_dims': 'shap_dimensions.parquet',
        'typology': 'community_typology.parquet',
        'benchmarks': 'turnaround_benchmarks.parquet',
    }
    for key, fname in files.items():
        path = PROCESSED / fname
        data[key] = pd.read_parquet(path) if path.exists() else None
    return data


data = load_ml_data()

page_header(
    "🧠 What Drives Turnaround?",
    f"Among ~25,000 census tracts with IGS below {IGS_VULN_THRESHOLD} in 2017, "
    f"what features predict crossing that threshold by 2025?"
)

# ── The Question ─────────────────────────────────────────────────────────────
st.info(
    f"**Study Design:** We identified all tracts with IGS < {IGS_VULN_THRESHOLD} in 2017 "
    f"(the at-risk population). The binary outcome: did the tract reach IGS ≥ {IGS_VULN_THRESHOLD} "
    f"by 2025 (\"turnaround\") or remain below (\"stuck\")? We trained 3 models and used "
    f"SHAP to explain which features matter most.",
    icon="🔬",
)

if data['typology'] is not None and 'typology' in data['typology'].columns:
    typ = data['typology']
    at_risk = typ[typ['igs_score_2017'] < IGS_VULN_THRESHOLD]
    n_turn = int((at_risk['typology'] == 'Turnaround').sum())
    n_stuck = int((at_risk['typology'] == 'Stuck').sum())
    n_total = len(at_risk)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("At-Risk Tracts (2017)", f"{n_total:,}")
    with c2:
        st.metric("Turnaround", f"{n_turn:,}", delta=f"{n_turn/n_total*100:.1f}%")
    with c3:
        st.metric("Stuck", f"{n_stuck:,}", delta=f"{n_stuck/n_total*100:.1f}%")

section_divider()

# ── Model Comparison ─────────────────────────────────────────────────────────
st.markdown("### Model Comparison")

if data['models'] is not None:
    fig = model_comparison_chart(data['models'])
    st.plotly_chart(fig, use_container_width=True)

    best = data['models'].sort_values('cv_auc_mean', ascending=False).iloc[0]
    st.success(
        f"**Best model: {best['model']}** with cross-validated AUC = {best['cv_auc_mean']:.3f} "
        f"(± {best['cv_auc_std']:.3f}). This means the model can reliably distinguish "
        f"turnaround from stuck communities.",
        icon="✅",
    )

    with st.expander("What do these models do?"):
        st.markdown(
            "- **Logistic Regression**: Linear baseline. Shows which features have "
            "the strongest linear relationship with turnaround.\n"
            "- **Random Forest**: Captures non-linear patterns and interactions between features. "
            "The SHAP analysis below uses this model.\n"
            "- **Gradient Boosting**: Sequential learning that focuses on hard-to-classify tracts. "
            "Often the highest AUC."
        )
else:
    st.warning("Model comparison data not available. Run the ML pipeline.")

section_divider()

# ── SHAP Feature Importance ──────────────────────────────────────────────────
st.markdown("### What Features Matter Most? (SHAP Analysis)")

if data['shap_summary'] is not None:
    fig = shap_importance_chart(data['shap_summary'], top_n=15)
    st.plotly_chart(fig, use_container_width=True)

    top3 = data['shap_summary'].head(3)['feature'].tolist()
    st.markdown(
        f"**Top 3 predictors of turnaround:** "
        f"{'  ·  '.join(top3)}"
    )

    with st.expander("View all feature importances"):
        display = data['shap_summary'][['feature', 'mean_abs_shap', 'pct_total', 'rank']].copy()
        display['mean_abs_shap'] = display['mean_abs_shap'].round(4)
        display['pct_total'] = display['pct_total'].round(1)
        display.columns = ['Feature', 'Mean |SHAP|', '% of Total', 'Rank']
        st.dataframe(display, use_container_width=True, hide_index=True)

section_divider()

# ── SHAP Dimensions ──────────────────────────────────────────────────────────
st.markdown("### Feature Dimensions (SHAP Clustering)")
st.markdown(
    "Features are automatically clustered into dimensions based on how their SHAP values "
    "co-vary across tracts. Features that influence turnaround together form a dimension."
)

if data['shap_dims'] is not None:
    c1, c2 = st.columns([1, 1])

    with c1:
        fig = shap_dimension_donut(data['shap_dims'])
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        for _, row in data['shap_dims'].iterrows():
            weight = row['weight_pct']
            name = row['dimension_name']
            features = row['features']
            top_feat = row['top_feature']
            st.markdown(f"**{name}** ({weight:.1f}% of predictive power)")
            st.caption(f"Top feature: {top_feat}")
            with st.expander(f"All features in {name}"):
                for f in features.split(', '):
                    st.markdown(f"- {f}")

section_divider()

# ── Turnaround vs Stuck Profile ─────────────────────────────────────────────
st.markdown("### Turnaround vs Stuck: Community Profiles")
st.markdown(
    "What did turnaround communities have in 2025 that stuck communities didn't? "
    "These benchmarks show the measurable differences."
)

if data['benchmarks'] is not None:
    key_indicators = [
        'igs_economy', 'igs_place', 'igs_community',
        'Internet Access Score', 'Commercial Diversity Score',
        'Health Insurance Coverage Score', 'Labor Market Engagement Index Score',
        'Small Business Loans Score', 'Female Above Poverty Score',
    ]
    available = [i for i in key_indicators if i in data['benchmarks']['indicator'].values]

    if available:
        fig = turnaround_vs_stuck_bars(data['benchmarks'], indicators=available)
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("Full benchmark data"):
        bm = data['benchmarks'].copy()
        for c in ['mean_2017', 'mean_2025', 'mean_delta']:
            if c in bm.columns:
                bm[c] = bm[c].round(2)
        st.dataframe(bm, use_container_width=True, hide_index=True, height=400)

# ── Key Insight ──────────────────────────────────────────────────────────────
section_divider()
st.markdown("### The Key Insight")
st.warning(
    "**It is not the number of doctors that predicts whether a community escapes "
    "economic vulnerability.** HPSA scores (healthcare shortage designations) have "
    "relatively low SHAP importance. Instead, the model reveals that turnaround is "
    "driven by **chronic disease burden** (diabetes, hypertension — the demand side), "
    "**broadband access** (Internet Access Score), and **commercial diversity** "
    "(number of distinct business types).\n\n"
    "The healthcare pathway is **indirect**: better healthcare → managed chronic disease → "
    "lower health burden → higher labor participation → higher income → IGS improvement. "
    "This means investing in **healthcare anchor small businesses** (pharmacies, clinics) "
    "addresses the root cause, while **broadband and business diversity** are the "
    "strongest direct levers.",
    icon="⚡",
)
