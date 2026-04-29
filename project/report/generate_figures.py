#!/usr/bin/env python3
"""
Generate publication-quality figures for the research report.

Run from the repo root:
    python project/report/generate_figures.py

Outputs: project/report/figures/fig{1-6}_*.pdf  (300 dpi, vector)
"""

import sys, io, contextlib, importlib, importlib.util, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib.cm import ScalarMappable
from matplotlib.gridspec import GridSpec

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "project" / "data_processed"
FIG_DIR   = Path(__file__).resolve().parent / "figures"
FIG_DIR.mkdir(exist_ok=True)

# ── Import analysis module (defines FEATURE_REGISTRY, compute_gaps, etc.) ────
_mod_path = ROOT / "project" / "03_analysis" / "expanded_priority_matrix.py"
_spec     = importlib.util.spec_from_file_location("epm", _mod_path)
_mod      = importlib.util.module_from_spec(_spec)
with contextlib.redirect_stdout(io.StringIO()):
    _spec.loader.exec_module(_mod)

FEATURE_REGISTRY  = _mod.FEATURE_REGISTRY
CATEGORY_COLORS   = _mod.CATEGORY_COLORS
compute_gaps      = _mod.compute_gaps
get_ml_features   = _mod.get_ml_features
DELTA_COUNTY_FIPS = _mod.DELTA_COUNTY_FIPS

label_lkp = {r[0]: r[1] for r in FEATURE_REGISTRY}
cat_lkp   = {r[0]: r[2] for r in FEATURE_REGISTRY}
dir_lkp   = {r[0]: r[3] for r in FEATURE_REGISTRY}

# ── Publication style ─────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":          "serif",
    "font.size":            9,
    "axes.labelsize":       9,
    "axes.titlesize":       10,
    "axes.titleweight":     "bold",
    "axes.titlepad":        6,
    "xtick.labelsize":      8,
    "ytick.labelsize":      8,
    "legend.fontsize":      8,
    "legend.framealpha":    0.9,
    "legend.edgecolor":     "#d1d5db",
    "figure.dpi":           120,
    "savefig.dpi":          300,
    "savefig.bbox":         "tight",
    "savefig.pad_inches":   0.05,
    "axes.spines.top":      False,
    "axes.spines.right":    False,
    "axes.linewidth":       0.8,
    "axes.grid":            True,
    "axes.grid.axis":       "x",
    "grid.color":           "#f0f0f0",
    "grid.linewidth":       0.6,
    "xtick.major.width":    0.8,
    "ytick.major.width":    0.8,
    "lines.linewidth":      1.8,
    "patch.linewidth":      0.8,
})

# Brand colours
C_NATIONAL  = "#374151"
C_MS        = "#b45309"
C_DELTA     = "#dc2626"
C_TURN      = "#059669"
C_STUCK     = "#dc2626"
C_THRESHOLD = "#6b7280"
C_BLUE      = "#2563eb"


# ─────────────────────────────────────────────────────────────────────────────
# Fig 1 · IGS Trajectory
# ─────────────────────────────────────────────────────────────────────────────
def fig1_igs_trajectory():
    t = pd.read_parquet(PROCESSED / "igs_trends_summary.parquet")

    fig, ax = plt.subplots(figsize=(6.5, 3.4))

    ax.fill_between(t["year"], 0, 45, color="#fee2e2", alpha=0.35, zorder=0)
    ax.axhline(45, color=C_THRESHOLD, lw=1.2, ls="--", zorder=1,
               label="Vulnerability threshold (IGS 45)")

    ax.plot(t["year"], t["nat_mean"],   color=C_NATIONAL, lw=2.0,
            marker="o", ms=4, label="National")
    ax.plot(t["year"], t["ms_mean"],    color=C_MS,       lw=2.0,
            marker="s", ms=4, label="Mississippi")
    ax.plot(t["year"], t["delta_mean"], color=C_DELTA,    lw=2.2,
            marker="^", ms=5, label="MS Delta (9 counties)")

    # End-point annotations
    last = t.iloc[-1]
    ax.annotate(f"{last['nat_mean']:.1f}",
                xy=(last["year"], last["nat_mean"]),
                xytext=(4, 0), textcoords="offset points",
                va="center", fontsize=7.5, color=C_NATIONAL, fontweight="bold")
    ax.annotate(f"{last['ms_mean']:.1f}",
                xy=(last["year"], last["ms_mean"]),
                xytext=(4, 2), textcoords="offset points",
                va="center", fontsize=7.5, color=C_MS, fontweight="bold")
    ax.annotate(f"{last['delta_mean']:.1f}",
                xy=(last["year"], last["delta_mean"]),
                xytext=(4, -2), textcoords="offset points",
                va="center", fontsize=7.5, color=C_DELTA, fontweight="bold")

    # "Below threshold" shading label
    ax.text(2017.15, 10, "Economically\nvulnerable zone\n(IGS < 45)",
            fontsize=7, color="#b91c1c", alpha=0.8, va="bottom")

    ax.set_xlim(2016.5, 2025.8)
    ax.set_ylim(28, 58)
    ax.set_xlabel("Year")
    ax.set_ylabel("Mean Inclusive Growth Score")
    ax.set_title("Persistent IGS Gap: MS Delta vs. Mississippi vs. Nation (2017–2025)")
    ax.legend(loc="upper right", frameon=True)
    ax.grid(axis="y", color="#f0f0f0", lw=0.6)
    ax.set_xticks(t["year"].tolist())

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig1_igs_trajectory.pdf")
    plt.close(fig)
    print("  ✓ fig1_igs_trajectory.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 2 · Study Population — two panels
# ─────────────────────────────────────────────────────────────────────────────
def fig2_study_population():
    em = pd.read_parquet(PROCESSED / "expanded_model.parquet",
                         columns=["turnaround", "igs_score_2017", "county_fips5",
                                  "igs_score_2025"])
    em["outcome"] = em["turnaround"].map({1: "Turnaround", 0: "Stuck"})
    em["is_delta"] = em["county_fips5"].isin(DELTA_COUNTY_FIPS)

    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.4))

    # ── Panel A: national counts (bar) ───────────────────────────────────────
    ax = axes[0]
    n_stuck   = (em["turnaround"] == 0).sum()
    n_turn    = (em["turnaround"] == 1).sum()
    d_stuck   = (em[em["is_delta"]]["turnaround"] == 0).sum()
    d_turn    = (em[em["is_delta"]]["turnaround"] == 1).sum()

    cats  = ["Turnaround\n(IGS ≥ 45 by 2025)", "Stuck\n(IGS < 45 by 2025)"]
    n_nat = [n_turn, n_stuck]
    n_dlt = [d_turn, d_stuck]

    x = np.arange(2)
    w = 0.35
    bars_nat = ax.bar(x - w/2, n_nat, w, color=[C_TURN, C_STUCK], alpha=0.85,
                      label="National at-risk", zorder=3)
    bars_dlt = ax.bar(x + w/2, n_dlt, w, color=[C_TURN, C_STUCK], alpha=0.5,
                      edgecolor="black", linewidth=0.8, label="MS Delta", zorder=3)

    for bar, val in zip(bars_nat, n_nat):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 150,
                f"{val:,}", ha="center", va="bottom", fontsize=7.5, fontweight="bold")
    for bar, val in zip(bars_dlt, n_dlt):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 150,
                f"{val}", ha="center", va="bottom", fontsize=7.5)

    ax.set_xticks(x)
    ax.set_xticklabels(cats, fontsize=8)
    ax.set_ylabel("Number of Census Tracts")
    ax.set_title("(a) Outcome Distribution")
    ax.legend(fontsize=7.5)
    ax.set_ylim(0, max(n_nat) * 1.18)
    ax.grid(axis="y")
    ax.set_axisbelow(True)

    # ── Panel B: IGS 2017 distribution by outcome (KDE) ──────────────────────
    ax = axes[1]
    from scipy.stats import gaussian_kde
    for outcome, color, ls in [("Turnaround", C_TURN, "-"), ("Stuck", C_STUCK, "--")]:
        vals = em[em["outcome"] == outcome]["igs_score_2017"].dropna().values
        kde  = gaussian_kde(vals, bw_method=0.2)
        xs   = np.linspace(vals.min(), vals.max(), 300)
        ys   = kde(xs)
        ax.plot(xs, ys, color=color, ls=ls, lw=2, label=outcome)
        ax.fill_between(xs, ys, alpha=0.12, color=color)
        ax.axvline(vals.mean(), color=color, lw=1.0, ls=":", alpha=0.8)

    ax.set_xlabel("IGS Score in 2017")
    ax.set_ylabel("Density")
    ax.set_title("(b) Baseline IGS Distribution by Outcome")
    ax.legend()
    ax.set_xlim(0, 45)
    ax.grid(axis="y")

    fig.suptitle("Study Population: 25,142 At-Risk US Census Tracts (IGS < 45 in 2017)",
                 fontsize=10, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig2_study_population.pdf")
    plt.close(fig)
    print("  ✓ fig2_study_population.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 3 · SHAP Feature Importance (top 20)
# ─────────────────────────────────────────────────────────────────────────────
def fig3_shap_importance():
    ss = pd.read_parquet(PROCESSED / "expanded_shap_summary.parquet")
    ss["label"]    = ss["feature"].map(label_lkp).fillna(ss["feature"])
    ss["category"] = ss["feature"].map(cat_lkp).fillna("Other")
    ss["color"]    = ss["category"].map(CATEGORY_COLORS).fillna("#888888")

    top = ss.head(20).sort_values("pct_total", ascending=True)

    fig, ax = plt.subplots(figsize=(6.5, 5.8))
    bars = ax.barh(top["label"], top["pct_total"],
                   color=top["color"].tolist(), height=0.72,
                   edgecolor="white", linewidth=0.4, zorder=3)

    for bar, val in zip(bars, top["pct_total"]):
        ax.text(bar.get_width() + 0.08, bar.get_y() + bar.get_height()/2,
                f"{val:.1f}%", va="center", fontsize=7.5)

    # Category legend
    seen_cats = dict.fromkeys(top["category"].tolist()[::-1])  # insertion order
    patches = [mpatches.Patch(color=CATEGORY_COLORS.get(c, "#888"), label=c)
               for c in seen_cats]
    ax.legend(handles=patches, loc="lower right", fontsize=7, ncol=1,
              framealpha=0.92, title="Domain", title_fontsize=7.5)

    ax.set_xlabel("Share of Total Predictive Power (% of Σ|SHAP|)")
    ax.set_title("Top 20 Predictors of Community Turnaround\n"
                 "(Mean |SHAP| as % of total — Random Forest, n = 5,000 tracts)")
    ax.set_xlim(0, top["pct_total"].max() * 1.22)
    ax.yaxis.set_tick_params(labelsize=8.5)
    ax.grid(axis="x", color="#ebebeb", lw=0.6)
    ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig3_shap_importance.pdf")
    plt.close(fig)
    print("  ✓ fig3_shap_importance.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 4 · SHAP Beeswarm (top 15)
# ─────────────────────────────────────────────────────────────────────────────
def _beeswarm_jitter(vals, max_spread=0.4, n_bins=60, seed=0):
    rng   = np.random.default_rng(seed)
    y_out = np.full(len(vals), np.nan)
    valid = ~np.isnan(vals)
    v     = vals[valid]
    if len(v) < 2:
        y_out[valid] = 0.0
        return y_out
    lo, hi = float(np.percentile(v, 0.5)), float(np.percentile(v, 99.5))
    if abs(hi - lo) < 1e-12:
        y_out[valid] = 0.0
        return y_out
    edges  = np.linspace(lo, hi, n_bins + 1)
    bidx   = np.clip(np.digitize(v, edges) - 1, 0, n_bins - 1)
    counts = np.bincount(bidx, minlength=n_bins)
    mx     = max(int(counts.max()), 1)
    offsets = np.zeros(len(v))
    for b in range(n_bins):
        m = bidx == b
        n = int(m.sum())
        if n == 0: continue
        spread = max_spread * (n / mx)
        pos    = np.linspace(-spread, spread, n) if n > 1 else np.zeros(1)
        offsets[m] = rng.permutation(pos)
    y_out[valid] = offsets
    return y_out


def fig4_shap_beeswarm():
    ss = pd.read_parquet(PROCESSED / "expanded_shap_summary.parquet")
    sv = pd.read_parquet(PROCESSED / "expanded_shap_values.parquet")
    fs = pd.read_parquet(PROCESSED / "expanded_shap_sample.parquet")

    ss["label"] = ss["feature"].map(label_lkp).fillna(ss["feature"])
    feats = ss["feature"].tolist()[:15]

    BEE_N = len(feats)
    fig, ax = plt.subplots(figsize=(6.5, 6.4))

    # RdBu_r colormap
    cmap   = plt.get_cmap("RdBu_r")
    norm_c = Normalize(vmin=0, vmax=100)

    for fi, feat in enumerate(feats):
        if feat not in sv.columns:
            continue
        y_base  = BEE_N - 1 - fi
        sv_vals = sv[feat].values.astype(float)
        y_jit   = _beeswarm_jitter(sv_vals, max_spread=0.38, seed=fi)
        y_vals  = y_base + y_jit

        # Normalise feature value for color
        fv = fs[feat].astype(float).values if feat in fs.columns else np.full(len(sv_vals), 50.0)
        p5, p95 = np.nanpercentile(fv, 5), np.nanpercentile(fv, 95)
        rng_w   = p95 - p5 if p95 != p5 else 1.0
        norm_fv = np.clip((fv - p5) / rng_w * 100, 0, 100)
        colors  = cmap(norm_c(norm_fv))

        ax.scatter(sv_vals, y_vals, c=colors, s=2.5, alpha=0.65,
                   linewidths=0, zorder=3, rasterized=True)

    ax.axvline(0, color="#9ca3af", lw=1.0, zorder=2)

    # Y-axis feature labels
    y_tick_text = []
    for feat in reversed(feats):
        rows = ss[ss["feature"] == feat]
        lbl  = rows.iloc[0]["label"] if not rows.empty else feat
        y_tick_text.append(lbl)
    ax.set_yticks(range(BEE_N))
    ax.set_yticklabels(y_tick_text, fontsize=8.5)
    ax.set_ylim(-0.7, BEE_N - 0.3)

    ax.set_xlabel("SHAP Value   ←  pushes toward Stuck  |  pushes toward Turnaround  →")
    ax.set_title("SHAP Beeswarm — Top 15 Predictors\n"
                 "(each dot = one census tract; color = feature value)")

    # Colorbar
    sm  = ScalarMappable(norm=norm_c, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02, shrink=0.25, aspect=12,
                        anchor=(0.0, 0.92))
    cbar.set_ticks([5, 95])
    cbar.set_ticklabels(["Low", "High"], fontsize=7.5)
    cbar.set_label("Feature\nvalue", fontsize=7.5, labelpad=2)

    ax.grid(axis="x", color="#ebebeb", lw=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig4_beeswarm.pdf", dpi=300)
    plt.close(fig)
    print("  ✓ fig4_beeswarm.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 5 · MS Delta Investment Priority Matrix
# ─────────────────────────────────────────────────────────────────────────────
def fig5_priority_matrix():
    em          = pd.read_parquet(PROCESSED / "expanded_model.parquet")
    shap_sum    = pd.read_parquet(PROCESSED / "expanded_shap_summary.parquet")
    with contextlib.redirect_stdout(io.StringIO()):
        features = get_ml_features(em)
        gap_df, _ = compute_gaps(em, features, shap_sum, "delta")

    SHAP_T, GAP_T = 3.0, 20.0

    top = gap_df.head(35).copy()
    top["label"]   = top["feature"].map(label_lkp).fillna(top["label"])
    top["color"]   = top["category"].map(CATEGORY_COLORS).fillna("#888888")

    def _q(s, g):
        if s >= SHAP_T and g >= GAP_T: return "ACT NOW"
        if s >= SHAP_T and g < GAP_T:  return "PROTECT"
        if s < SHAP_T  and g >= GAP_T: return "SECOND WAVE"
        return "MONITOR"
    top["quadrant"] = [_q(s, g) for s, g in zip(top["shap_pct"], top["gap"])]

    Q_COLORS = {"ACT NOW": "#dc2626", "PROTECT": "#059669",
                "SECOND WAVE": "#d97706", "MONITOR": "#9ca3af"}
    Q_ALPHA  = {"ACT NOW": 0.12, "PROTECT": 0.12,
                "SECOND WAVE": 0.08, "MONITOR": 0.06}

    fig, ax = plt.subplots(figsize=(6.5, 5.6))

    x_min = min(float(top["gap"].min()) - 12, -65)
    x_max = max(float(top["gap"].max()) + 12, 105)
    y_max = max(float(top["shap_pct"].max()) * 1.30, SHAP_T * 2.8)

    # Quadrant backgrounds
    for (x0, x1, y0, y1), q in [
        ((GAP_T, x_max, SHAP_T, y_max), "ACT NOW"),
        ((x_min, GAP_T, SHAP_T, y_max), "PROTECT"),
        ((GAP_T, x_max, 0, SHAP_T),     "SECOND WAVE"),
        ((x_min, GAP_T, 0, SHAP_T),     "MONITOR"),
    ]:
        ax.fill_between([x0, x1], y0, y1, color=Q_COLORS[q], alpha=Q_ALPHA[q], zorder=0)

    # Threshold lines
    ax.axhline(SHAP_T, color="#9ca3af", lw=1.1, ls="--", zorder=1)
    ax.axvline(GAP_T,  color="#9ca3af", lw=1.1, ls="--", zorder=1)
    ax.axvline(0,      color="#d1d5db", lw=0.8, ls=":",  zorder=1)

    # Quadrant labels
    tr_mx = (GAP_T + x_max) / 2;  tl_mx = (x_min + GAP_T) / 2
    top_y = SHAP_T + (y_max - SHAP_T) * 0.88; bot_y = SHAP_T * 0.38
    for txt, x, y, col in [
        ("ACT NOW",     tr_mx, top_y, "#991b1b"),
        ("PROTECT",     tl_mx, top_y, "#065f46"),
        ("SECOND WAVE", tr_mx, bot_y, "#92400e"),
        ("MONITOR",     tl_mx, bot_y, "#6b7280"),
    ]:
        ax.text(x, y, txt, ha="center", va="center", fontsize=8.5,
                fontweight="bold", color=col,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", alpha=0.75, lw=0))

    # Points (one scatter per category)
    from itertools import groupby
    for cat in top["category"].unique():
        sub  = top[top["category"] == cat]
        sym  = sub["higher_good"].apply(lambda h: "D" if h is None else "o").tolist()
        col  = CATEGORY_COLORS.get(cat, "#888")
        ax.scatter(sub["gap"], sub["shap_pct"], marker="o",
                   color=col, s=55, linewidths=0.8,
                   edgecolors="white", zorder=4, label=cat, alpha=0.88)

    # Label ACT-NOW dots
    act = top[top["quadrant"] == "ACT NOW"].nlargest(8, "shap_pct")
    for _, row in act.iterrows():
        ax.annotate(
            row["label"],
            xy=(row["gap"], row["shap_pct"]),
            xytext=(5, 3), textcoords="offset points",
            fontsize=6.5, color="#1e293b",
            arrowprops=dict(arrowstyle="-", color="#9ca3af", lw=0.6),
        )

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(0, y_max)
    ax.set_xlabel("Gap from Turnaround Benchmark  ←  Strength  |  Behind  →")
    ax.set_ylabel("SHAP Importance (% of total predictive power)")
    ax.set_title("Investment Priority Matrix — MS Delta Region\n"
                 "(SHAP importance is national; gap is specific to the Delta)")

    handles = [mpatches.Patch(color=CATEGORY_COLORS.get(c, "#888"), label=c)
               for c in top["category"].unique()]
    ax.legend(handles=handles, loc="upper left", fontsize=6.8, ncol=1,
              framealpha=0.9, title="Domain", title_fontsize=7)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig5_priority_matrix.pdf")
    plt.close(fig)
    print("  ✓ fig5_priority_matrix.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 6 · Feature Category × SHAP Importance + Delta gap bar
# ─────────────────────────────────────────────────────────────────────────────
def fig6_category_and_gap():
    ss       = pd.read_parquet(PROCESSED / "expanded_shap_summary.parquet")
    ss["category"] = ss["feature"].map(cat_lkp).fillna("Other")
    ss["color"]    = ss["category"].map(CATEGORY_COLORS).fillna("#888888")

    cat_agg = (ss.groupby("category")
                 .agg(pct_total=("pct_total", "sum"), n=("feature", "count"))
                 .reset_index()
                 .sort_values("pct_total", ascending=True))
    cat_agg["color"] = cat_agg["category"].map(CATEGORY_COLORS).fillna("#888")

    # Delta gap: load priority matrix data
    em       = pd.read_parquet(PROCESSED / "expanded_model.parquet")
    shap_sum = pd.read_parquet(PROCESSED / "expanded_shap_summary.parquet")
    with contextlib.redirect_stdout(io.StringIO()):
        features = get_ml_features(em)
        gap_df, _ = compute_gaps(em, features, shap_sum, "delta")

    gap_df["category"] = gap_df["feature"].map(cat_lkp).fillna("Other")
    cat_gap = (gap_df[gap_df["higher_good"].notna()]
               .groupby("category")["gap"]
               .mean()
               .reset_index()
               .rename(columns={"gap": "mean_gap"}))

    fig, axes = plt.subplots(1, 2, figsize=(6.5, 4.0))

    # ── Panel A: SHAP by category ─────────────────────────────────────────────
    ax = axes[0]
    bars = ax.barh(cat_agg["category"], cat_agg["pct_total"],
                   color=cat_agg["color"], height=0.68,
                   edgecolor="white", linewidth=0.4, zorder=3)
    for bar, row in zip(bars, cat_agg.itertuples()):
        ax.text(bar.get_width() + 0.15,
                bar.get_y() + bar.get_height()/2,
                f"{row.pct_total:.1f}%  (n={row.n})",
                va="center", fontsize=7)
    ax.set_xlabel("Cumulative SHAP (% of total)")
    ax.set_title("(a) SHAP Importance by Domain")
    ax.set_xlim(0, cat_agg["pct_total"].max() * 1.45)
    ax.grid(axis="x", color="#ebebeb", lw=0.6)
    ax.set_axisbelow(True)

    # ── Panel B: Mean gap per category for Delta ──────────────────────────────
    ax = axes[1]
    cat_gap_m = cat_gap.merge(cat_agg[["category", "color"]], on="category")
    cat_gap_m = cat_gap_m.sort_values("mean_gap", ascending=True)

    colors_bar = [C_TURN if g < 0 else C_STUCK for g in cat_gap_m["mean_gap"]]
    bars = ax.barh(cat_gap_m["category"], cat_gap_m["mean_gap"],
                   color=colors_bar, alpha=0.80, height=0.68,
                   edgecolor="white", linewidth=0.4, zorder=3)
    ax.axvline(0, color="#374151", lw=1.0, zorder=2)
    for bar, val in zip(bars, cat_gap_m["mean_gap"]):
        xpos = val + (1.5 if val >= 0 else -1.5)
        ha   = "left" if val >= 0 else "right"
        ax.text(xpos, bar.get_y() + bar.get_height()/2,
                f"{val:+.0f}", va="center", fontsize=7, ha=ha)
    ax.set_xlabel("Mean Gap from Turnaround Benchmark\n(positive = behind, needs investment)")
    ax.set_title("(b) MS Delta Gap by Domain")
    leg = [mpatches.Patch(color=C_STUCK, label="Behind (needs investment)"),
           mpatches.Patch(color=C_TURN,  label="Ahead (existing strength)")]
    ax.legend(handles=leg, fontsize=7, loc="lower right")
    ax.grid(axis="x", color="#ebebeb", lw=0.6)
    ax.set_axisbelow(True)

    fig.suptitle("Feature Domain Analysis — SHAP Importance & MS Delta Gaps",
                 fontsize=10, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig6_category_and_gap.pdf")
    plt.close(fig)
    print("  ✓ fig6_category_and_gap.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Saving figures to: {FIG_DIR}\n")
    fig1_igs_trajectory()
    fig2_study_population()
    fig3_shap_importance()
    fig4_shap_beeswarm()
    fig5_priority_matrix()
    fig6_category_and_gap()
    print(f"\nDone — {len(list(FIG_DIR.glob('*.pdf')))} figures generated.")
