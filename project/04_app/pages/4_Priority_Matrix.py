"""
Page 4: Priority Matrix — Interactive 2×2 Investment Priority Matrix
Powered by SHAP feature importance × gap from turnaround tracts.
"""

import sys, io, contextlib, importlib, importlib.util
from pathlib import Path
from datetime import date

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="Priority Matrix", layout="wide", page_icon="🎯")

from components.theme import apply_theme, sidebar_nav, page_header
apply_theme()

from config import PROCESSED, DELTA_COUNTY_FIPS, DELTA_COUNTY_NAMES

# ── Import analysis module via importlib (filename starts with digit) ─────────
_analysis_path = (
    Path(__file__).resolve().parents[2] / "03_analysis" / "expanded_priority_matrix.py"
)
_spec = importlib.util.spec_from_file_location("expanded_priority_matrix", _analysis_path)
_mod  = importlib.util.module_from_spec(_spec)
with contextlib.redirect_stdout(io.StringIO()):
    _spec.loader.exec_module(_mod)

compute_gaps     = _mod.compute_gaps
get_ml_features  = _mod.get_ml_features
FEATURE_REGISTRY = _mod.FEATURE_REGISTRY
CATEGORY_COLORS  = _mod.CATEGORY_COLORS

# ── Load Data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df           = pd.read_parquet(PROCESSED / "expanded_model.parquet")
    shap_summary = pd.read_parquet(PROCESSED / "expanded_shap_summary.parquet")
    return df, shap_summary

df, shap_summary = load_data()

# Pre-compute ML feature list (fast — just column filtering)
with contextlib.redirect_stdout(io.StringIO()):
    features = get_ml_features(df)

# ── Cache gap computation per target ──────────────────────────────────────────
@st.cache_data
def cached_gaps(_df, _features, _shap_summary, target):
    with contextlib.redirect_stdout(io.StringIO()):
        return compute_gaps(_df, _features, _shap_summary, target)

# ── Build MS Delta county list (fixed 9 counties) ────────────────────────────
def build_delta_county_list():
    entries = []
    for fips, name in DELTA_COUNTY_NAMES.items():
        entries.append((f"{name} County, MS ({fips})", fips))
    entries.sort(key=lambda x: x[0])
    return entries

county_list = build_delta_county_list()

# All unique categories, registry-ordered
all_categories = list(dict.fromkeys(row[2] for row in FEATURE_REGISTRY))

# ── Sidebar ───────────────────────────────────────────────────────────────────
sidebar_nav()

with st.sidebar:
    st.markdown("---")
    st.markdown("**County Selector**")

    option_labels = (
        ["MS Delta Region (All 9 Delta Counties)"] + [lbl for lbl, _ in county_list]
    )
    option_values = ["delta"] + [fips for _, fips in county_list]

    selected_idx = st.selectbox(
        "Select County",
        range(len(option_labels)),
        format_func=lambda i: option_labels[i],
        index=0,
        key="county_selector",
    )
    target         = option_values[selected_idx]
    target_display = option_labels[selected_idx]

    with st.expander("⚙️ Display Settings"):
        top_n        = st.slider("Features to show",    10, 50,  30, 5)
        shap_thresh  = st.slider("SHAP threshold (%)",  1.0, 8.0, 3.0, 0.5)
        gap_thresh   = st.slider("Gap threshold",        5,  50,  20,  5)
        show_context = st.checkbox("Show demographic/context features", value=True)

    with st.expander("🔬 Filter Categories"):
        selected_cats = st.multiselect(
            "Include feature categories",
            all_categories,
            default=all_categories,
        )

# ── Page Header ───────────────────────────────────────────────────────────────
page_header(
    "Investment Priority Matrix",
    "Select any county to see what matters most — and how far it is from turnaround "
    "— powered by SHAP feature importance.",
)

with st.expander("How to use this page", expanded=False):
    st.markdown(
        "The Priority Matrix answers a single question: **where should this county invest first?**\n\n"
        "Each feature is plotted on two axes simultaneously:\n"
        "- **Vertical axis (SHAP importance):** how strongly this feature predicts national "
        "turnaround — fixed, model-derived.\n"
        "- **Horizontal axis (Gap from turnaround benchmark):** how far behind this county "
        "currently lags — county-specific.\n\n"
        "**Quadrant guide:**\n"
        "- **Act Now** (top-right): high importance + large gap → top investment priority.\n"
        "- **Protect** (top-left): high importance + small/no gap → preserve existing strength.\n"
        "- **Second Wave** (bottom-right): moderate importance + large gap → phase-2 priorities.\n"
        "- **Monitor** (bottom-left): lower importance + small gap → watch and wait.\n\n"
        "**Tips:**\n"
        "- Use the **County Selector** in the sidebar to switch between counties or view the full Delta region.\n"
        "- Adjust **SHAP / Gap thresholds** in Display Settings to raise or lower the quadrant boundaries.\n"
        "- Use **Filter Categories** to focus on a specific domain (e.g., only Health or only Business).\n"
        "- Hover over any dot in the scatter plot for feature name, SHAP share, and gap value."
    )

# ── Tract count guard ─────────────────────────────────────────────────────────
if target != "delta":
    n_tracts = int((df["county_fips5"] == target).sum())
    if n_tracts == 0:
        st.info(
            "No at-risk tracts (IGS < 45 in 2017) found for this county. "
            "Try a different county."
        )
        st.stop()
    elif n_tracts < 3:
        st.warning(
            f"Only {n_tracts} at-risk tract(s) in this county — results may be "
            f"less stable. Showing available data."
        )

# ── Compute gaps ──────────────────────────────────────────────────────────────
gap_df, _raw_label = cached_gaps(df, features, shap_summary, target)
target_label = target_display   # use full name from the selectbox

# ── Prominent county banner ───────────────────────────────────────────────────
_banner_color = "#2563eb" if target == "delta" else "#059669"
st.markdown(
    f"<div style='background:linear-gradient(90deg,{_banner_color}18 0%,{_banner_color}06 100%);"
    f"border-left:4px solid {_banner_color};border-radius:0 8px 8px 0;"
    f"padding:10px 18px;margin-bottom:4px'>"
    f"<span style='font-size:0.8rem;color:#6b7280;text-transform:uppercase;"
    f"letter-spacing:0.5px'>Analysing</span><br>"
    f"<span style='font-size:1.15rem;font-weight:700;color:#1e293b'>"
    f"{target_label}</span></div>",
    unsafe_allow_html=True,
)

# ── SHAP vs Gap methodology note ──────────────────────────────────────────────
with st.expander("ℹ️ How the axes work — SHAP Importance vs Gap from Turnaround"):
    st.markdown(f"""
**Y-axis — SHAP Importance — fixed for every county**

The Y-axis shows each feature's share of the model's total predictive power,
computed once on the full national dataset of {len(df):,} at-risk tracts.
It does **not** change when you switch counties.
*Dental Visit Rate is equally important as a national predictor whether you're
looking at Cook County, IL or Quitman County, MS.*

**X-axis — Gap from Turnaround — specific to the selected county**

The X-axis compares this county's average feature value to what tracts that
actually turned around look like nationally.
Switching counties re-computes every X-position:

| Gap | Meaning |
|---|---|
| **Large positive gap** (far right) | County is far behind turnaround tracts → needs investment |
| **Small / zero gap** | County is close to the turnaround benchmark |
| **Negative gap** (left of zero line) | County is already AHEAD of turnaround tracts → existing strength |

**What changes when you switch counties:**

| Element | Changes? |
|---|---|
| Dot Y-position (SHAP importance) | ✅ No — same for all counties |
| Quadrant threshold lines | ✅ No — set by sidebar sliders |
| Dot X-position (gap) | ♻️ Yes — recalculated per county |
| Quadrant assignment (ACT NOW / PROTECT / …) | ♻️ Yes — derived from the new gap |

The same feature can be in **ACT NOW** for one county and **PROTECT** for another,
because the gap differs — but its SHAP importance stays the same.
""")


# ── Helper: assign quadrant label ─────────────────────────────────────────────
def _quadrant(shap_pct_val, gap_val):
    if shap_pct_val >= shap_thresh and gap_val >= gap_thresh:
        return "🔴 ACT NOW"
    elif shap_pct_val >= shap_thresh and gap_val < gap_thresh:
        return "🟢 PROTECT"
    elif shap_pct_val < shap_thresh and gap_val >= gap_thresh:
        return "🟡 SECOND WAVE"
    else:
        return "⚪ MONITOR"


# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(
    ["📊 Priority Matrix", "📋 Priority Table", "ℹ️ How to Read This"]
)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1: Priority Matrix Chart
# ─────────────────────────────────────────────────────────────────────────────
with tab1:

    # Apply filters
    filtered = gap_df.head(top_n).copy()
    if not show_context:
        filtered = filtered[filtered["higher_good"].notna()]
    if selected_cats:
        filtered = filtered[filtered["category"].isin(selected_cats)]

    if filtered.empty:
        st.warning(
            "No features match the current filters. "
            "Adjust the settings in the sidebar."
        )
    else:
        # Axis ranges (pad generously so quadrant rects don't clip)
        x_min = min(float(filtered["gap"].min()) - 15.0, -70.0)
        x_max = max(float(filtered["gap"].max()) + 15.0, 110.0)
        y_max = max(float(filtered["shap_pct"].max()) * 1.30, float(shap_thresh) * 2.5)

        # Quadrant midpoints for annotation placement
        tr_mid_x = (gap_thresh + x_max) / 2
        tl_mid_x = (x_min + gap_thresh) / 2
        top_ann_y = shap_thresh + (y_max - shap_thresh) * 0.88
        bot_ann_y = shap_thresh * 0.45

        fig = go.Figure()

        # Quadrant background shapes + threshold lines
        fig.update_layout(
            shapes=[
                # Background rectangles
                dict(type="rect", xref="x", yref="y", layer="below",
                     x0=gap_thresh, x1=x_max, y0=shap_thresh, y1=y_max,
                     fillcolor="rgba(255,220,220,0.4)", line_width=0),
                dict(type="rect", xref="x", yref="y", layer="below",
                     x0=x_min, x1=gap_thresh, y0=shap_thresh, y1=y_max,
                     fillcolor="rgba(220,255,220,0.4)", line_width=0),
                dict(type="rect", xref="x", yref="y", layer="below",
                     x0=gap_thresh, x1=x_max, y0=0, y1=shap_thresh,
                     fillcolor="rgba(255,243,220,0.4)", line_width=0),
                dict(type="rect", xref="x", yref="y", layer="below",
                     x0=x_min, x1=gap_thresh, y0=0, y1=shap_thresh,
                     fillcolor="rgba(245,245,245,0.4)", line_width=0),
                # SHAP threshold (horizontal dashed)
                dict(type="line", xref="x", yref="y",
                     x0=x_min, x1=x_max, y0=shap_thresh, y1=shap_thresh,
                     line=dict(color="#999999", width=1.5, dash="dash")),
                # Gap threshold (vertical dashed)
                dict(type="line", xref="x", yref="y",
                     x0=gap_thresh, x1=gap_thresh, y0=0, y1=y_max,
                     line=dict(color="#999999", width=1.5, dash="dash")),
                # Zero line (vertical dotted)
                dict(type="line", xref="x", yref="y",
                     x0=0, x1=0, y0=0, y1=y_max,
                     line=dict(color="#cccccc", width=1.0, dash="dot")),
            ],
            annotations=[
                dict(x=tr_mid_x, y=top_ann_y, text="🔴 ACT NOW",
                     showarrow=False, xanchor="center",
                     font=dict(size=13, color="#c0392b"),
                     bgcolor="rgba(255,255,255,0.75)", borderpad=3),
                dict(x=tl_mid_x, y=top_ann_y, text="🟢 PROTECT",
                     showarrow=False, xanchor="center",
                     font=dict(size=13, color="#1e8449"),
                     bgcolor="rgba(255,255,255,0.75)", borderpad=3),
                dict(x=tr_mid_x, y=bot_ann_y, text="🟡 SECOND WAVE",
                     showarrow=False, xanchor="center",
                     font=dict(size=13, color="#e67e22"),
                     bgcolor="rgba(255,255,255,0.75)", borderpad=3),
                dict(x=tl_mid_x, y=bot_ann_y, text="⚪ MONITOR",
                     showarrow=False, xanchor="center",
                     font=dict(size=13, color="#888888"),
                     bgcolor="rgba(255,255,255,0.75)", borderpad=3),
            ],
        )

        # One trace per category (Plotly supports list of symbols per point)
        for cat in filtered["category"].unique():
            cat_df = filtered[filtered["category"] == cat].copy()
            color  = CATEGORY_COLORS.get(cat, "#888888")

            symbols = cat_df["higher_good"].apply(
                lambda h: "diamond" if h is None else "circle"
            ).tolist()

            def _dir_label(h):
                if h is True:  return "Higher = better"
                if h is False: return "Lower = better"
                return "Context / demographic"

            customdata = cat_df[["label", "target_raw", "turn_raw"]].copy()
            customdata["dir"] = cat_df["higher_good"].apply(_dir_label)
            cd = customdata.values  # shape (n, 4)

            fig.add_trace(go.Scatter(
                x=cat_df["gap"],
                y=cat_df["shap_pct"],
                mode="markers",
                marker=dict(
                    size=14,
                    color=color,
                    symbol=symbols,
                    line=dict(width=1.5, color="white"),
                ),
                name=cat,
                legendgroup=cat,
                customdata=cd,
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "Gap: %{x:+.1f}<br>"
                    "SHAP: %{y:.1f}%<br>"
                    "Target: %{customdata[1]:.2f}<br>"
                    "Turnaround avg: %{customdata[2]:.2f}<br>"
                    "Direction: %{customdata[3]}"
                    "<extra>" + cat + "</extra>"
                ),
            ))

        fig.update_layout(
            title=dict(
                text=f"Priority Matrix: {target_label}",
                font=dict(size=15, family="Inter, system-ui, sans-serif"),
            ),
            xaxis=dict(
                title="Gap from Turnaround  ◄ Strength  |  Behind ►<br>"
                      "(positive = needs improvement)",
                range=[x_min, x_max],
                zeroline=False,
                gridcolor="rgba(0,0,0,0.06)",
            ),
            yaxis=dict(
                title="SHAP Importance (% of total predictive power)",
                range=[0, y_max],
                zeroline=False,
                gridcolor="rgba(0,0,0,0.06)",
            ),
            height=600,
            legend=dict(
                orientation="v", x=1.02, y=1,
                xanchor="left", yanchor="top",
                bgcolor="rgba(255,255,255,0.9)",
                bordercolor="#e5e7eb", borderwidth=1,
                font=dict(size=11),
            ),
            plot_bgcolor="white",
            paper_bgcolor="white",
            margin=dict(l=60, r=180, t=80, b=80),
            font=dict(family="Inter, system-ui, sans-serif", color="#374151"),
            hovermode="closest",
        )

        st.plotly_chart(fig, use_container_width=True)

        # Quadrant metric cards
        filtered["_q"] = [
            _quadrant(s, g)
            for s, g in zip(filtered["shap_pct"], filtered["gap"])
        ]

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("🔴 ACT NOW",
                      f"{int((filtered['_q'] == '🔴 ACT NOW').sum())} features")
        with c2:
            st.metric("🟢 PROTECT",
                      f"{int((filtered['_q'] == '🟢 PROTECT').sum())} features")
        with c3:
            st.metric("🟡 SECOND WAVE",
                      f"{int((filtered['_q'] == '🟡 SECOND WAVE').sum())} features")
        with c4:
            st.metric("⚪ MONITOR",
                      f"{int((filtered['_q'] == '⚪ MONITOR').sum())} features")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2: Priority Table
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    # Category filter only — no top_n limit
    table_df = gap_df.copy()
    if selected_cats:
        table_df = table_df[table_df["category"].isin(selected_cats)]

    def _dir(h):
        if h is True:  return "Higher = better"
        if h is False: return "Lower = better"
        return "Context"

    table_df["Quadrant"]  = [
        _quadrant(s, g) for s, g in zip(table_df["shap_pct"], table_df["gap"])
    ]
    table_df["Direction"] = table_df["higher_good"].apply(_dir)

    display_df = table_df[[
        "Quadrant", "label", "category", "shap_pct", "gap",
        "target_raw", "turn_raw", "Direction",
    ]].copy()
    display_df.columns = [
        "Quadrant", "Feature", "Category", "SHAP %", "Gap",
        "Target Value", "Turnaround Value", "Direction",
    ]
    display_df["SHAP %"]           = display_df["SHAP %"].round(2)
    display_df["Gap"]              = display_df["Gap"].round(1)
    display_df["Target Value"]     = display_df["Target Value"].round(2)
    display_df["Turnaround Value"] = display_df["Turnaround Value"].round(2)

    _QUAD_BG = {
        "🔴 ACT NOW":    "background-color: #fde8e8",
        "🟢 PROTECT":    "background-color: #e8f8e8",
        "🟡 SECOND WAVE":"background-color: #fff8e8",
        "⚪ MONITOR":    "background-color: #f5f5f5",
    }

    def _color_rows(row):
        bg = _QUAD_BG.get(row["Quadrant"], "")
        return [bg] * len(row)

    st.dataframe(
        display_df.style.apply(_color_rows, axis=1),
        use_container_width=True,
        hide_index=True,
        height=500,
    )

    today_str = date.today().strftime("%Y%m%d")
    st.download_button(
        "⬇️ Download Priority Table",
        data=table_df.to_csv(index=False),
        file_name=f"priority_matrix_{target}_{today_str}.csv",
        mime="text/csv",
    )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3: How to Read This
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown(f"""
## How to Read the Priority Matrix

This chart has two axes:

**Y-axis — SHAP Importance**: How much the ML model relies on this feature to predict whether a
distressed tract will "turn around". Higher = more predictive power. Features above the dashed
line (≥ {shap_thresh:.1f}%) are the most influential.

**X-axis — Gap from Turnaround**: How far this county's value is from what tracts that actually
turned around look like.
- **Positive gap** = the county is BEHIND turnaround tracts (needs investment)
- **Negative gap** = the county is AHEAD of turnaround tracts (existing strength)

**The 4 Quadrants:**

| Quadrant | Meaning | Action |
|---|---|---|
| 🔴 ACT NOW | High importance + big gap | Top investment priority — data-proven lever with large deficit |
| 🟢 PROTECT | High importance + small/negative gap | Already strong on a key predictor — guard this |
| 🟡 SECOND WAVE | Lower importance + big gap | Real gap, but less predictive — plan for next phase |
| ⚪ MONITOR | Lower importance + small gap | Low urgency — watch but don't prioritize |

**Gap Direction:**
- ● Circle markers = Measurable lever (has a clear better/worse direction)
- ◆ Diamond markers = Demographic/context feature (gap is informational, not an intervention target)

**Important note on gap types:**
- IGS sub-indicator gaps compare the county's current score to what turnaround tracts ACHIEVED
  by 2025 — a true improvement target.
- Health, climate, and business gaps compare current conditions to what turnaround tracts
  currently look like — a snapshot comparison, not a change target.

**Adjusting the thresholds:**

Use the sidebar sliders to change the SHAP threshold (horizontal dashed line) and the gap
threshold (vertical dashed line):
- **SHAP threshold**: Lower it to surface more features in the top band; raise it to focus only
  on the strongest predictors.
- **Gap threshold**: Raise it to focus on the largest deficits; lower it to include moderate gaps.
- **Features to show**: Controls how many features (ranked by SHAP) are plotted. The Priority Table
  tab always shows all features regardless of this setting.
""")
