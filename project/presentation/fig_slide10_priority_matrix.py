"""
Slide 10 (Revised) — SHAP × Gap Priority Matrix
=================================================
Answers: "What is the POINT of SHAP if the prescription only uses IGS gaps?"

For every key predictive feature, plots TWO dimensions simultaneously:
  Y-axis: SHAP importance — how much the ML model relies on this feature
  X-axis: Delta's gap from the turnaround level (normalized, higher = farther behind)

This produces a 2×2 Priority Matrix:
  TOP-RIGHT  (High SHAP + Large Gap) = "Act Now"  — root cause investments
  TOP-LEFT   (High SHAP + Small Gap) = "Protect these strengths"
  BOTTOM-RIGHT (Low SHAP + Large Gap) = "Second wave"
  BOTTOM-LEFT  (Low SHAP + Small Gap) = "Monitor"

The SHAP analysis directly determines WHICH gaps to prioritize.
Without SHAP, all IGS gaps look equal.  WITH SHAP, the order is data-proven.

Exported figure is the priority matrix panel only (axes title, no slide headline
or footer), consistent with other slide 9–10 figure styling.

Output: presentation/figures/slide10_priority_matrix.png  (300 DPI, 12×6.75)
Run:  python presentation/fig_slide10_priority_matrix.py
"""

import sys
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from adjustText import adjust_text

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
PROCESSED  = PROJECT_ROOT / 'data_processed'
OUT_DIR    = PROJECT_ROOT / 'presentation' / 'figures'
OUT_DIR.mkdir(parents=True, exist_ok=True)

from config import DELTA_COUNTY_FIPS

# ── Design ────────────────────────────────────────────────────────────────────
BG_COLOR   = '#ffffff'
TEXT_COLOR = '#1a1a2e'
DIM_COLOR  = '#555566'
BORDER_CLR = '#cccccc'

plt.rcParams.update({
    'font.family':      'DejaVu Sans',
    'figure.facecolor': BG_COLOR,
    'text.color':       TEXT_COLOR,
    'axes.facecolor':   BG_COLOR,
    'axes.edgecolor':   BORDER_CLR,
})

# ── Features: diagnose → prioritize → prescribe (not “disease list + random IGS”) ─
# - Health: 3 PLACES lines (inactivity, mobility, diabetes) = burden without 6× redundancy.
# - Business: CBP / tract anchors — food retail, pharmacy, sector mix; MWOB = inclusive economy.
# - Commute: Travel Time to Work (IGS) — labor-market access / jobs reach.
# - IGS + climate + connectivity round out Mastercard + NRI story.
# SHAP % from shap_feature_summary.parquet in load_data.
# (feature_col, plain label, domain, color, higher_is_better)
FEATURES = [
    ('biz_food_retail',                          'Food Retail Businesses',      'Business',     '#6B4226', True),
    ('biz_pharmacy',                             'Pharmacy Businesses',         'Business',     '#6B4226', True),
    ('biz_sector_diversity',                     'Sector Diversity (NAICS)',    'Business',     '#6B4226', True),
    ('Commercial Diversity Score_2017',          'Commercial Diversity',        'Commercial',   '#c8830a', True),
    ('Minority/Women Owned Businesses Score_2017', 'MWOB Score',               'Commercial',   '#c8830a', True),
    ('Travel Time to Work Score_2017',           'Travel Time to Work',         'Commute',      '#34495e', True),
    ('BUILDVALUE',                               'NRI Build Exposure ($)',        'Climate',      '#2c5282', True),
    ('igs_economy_2017',                         'IGS Economy (2017)',            'IGS',          '#b8860b', True),
    ('LPA_CrudePrev',                            'Physical Inactivity',         'Health',       '#c0392b', False),
    ('MOBILITY_CrudePrev',                       'Mobility Disability',         'Health',       '#e74c3c', False),
    ('DIABETES_CrudePrev',                       'Diabetes Rate',               'Health',       '#e67e22', False),
    ('Internet Access Score_2017',               'Internet Access',             'Connectivity', '#2980b9', True),
    ('Labor Market Engagement Index Score_2017', 'Labor Market',                'Connectivity', '#1abc9c', True),
]

# Thresholds for quadrant division
SHAP_THRESH = 3.5   # % of total SHAP
GAP_THRESH  = 28    # normalized gap units (0–100 scale)

# Quadrant styles
QUAD = {
    # (high_shap, large_gap)
    (True,  True):  ('#fde8e8', '#c0392b', 'ACT NOW\nRoot Cause + Score Lever'),
    (True,  False): ('#e8f8e8', '#1e8449', 'PROTECT\nExisting Strength'),
    (False, True):  ('#fff3e0', '#e67e22', 'SECOND WAVE\nAddress After Root Causes'),
    (False, False): ('#f5f5f5', '#888888', 'MONITOR\nLow Priority'),
}


def load_data():
    model_df = pd.read_parquet(PROCESSED / 'igs_improvement_model.parquet')
    model_df['county_fips5'] = model_df['GEOID'].str[:5]
    shap_df  = pd.read_parquet(PROCESSED / 'shap_feature_summary.parquet')
    delta_m  = model_df[model_df['county_fips5'].isin(DELTA_COUNTY_FIPS)]
    turn     = model_df[model_df['turnaround'] == 1]
    shap_lkp = shap_df.set_index('feature')['pct_total'].to_dict()

    rows = []
    for col, label, domain, color, higher_good in FEATURES:
        if col not in model_df.columns:
            continue
        p05 = model_df[col].quantile(0.05)
        p95 = model_df[col].quantile(0.95)
        rng = p95 - p05 if p95 != p05 else 1.0

        def norm(v):
            n = (v - p05) / rng * 100
            n = max(0.0, min(100.0, n))
            return n if higher_good else 100.0 - n

        t_n = norm(turn[col].mean())
        d_n = norm(delta_m[col].mean())
        gap = t_n - d_n   # positive = Delta is behind turnaround

        shap_pct = shap_lkp.get(col, 0)

        rows.append({
            'col':        col,
            'label':      label,
            'domain':     domain,
            'color':      color,
            'shap_pct':   shap_pct,
            'gap':        gap,
            't_raw':      turn[col].mean(),
            'd_raw':      delta_m[col].mean(),
            'high_shap':  shap_pct >= SHAP_THRESH,
            'large_gap':  gap >= GAP_THRESH,
        })
    return pd.DataFrame(rows)


def make_figure(data: pd.DataFrame):
    fig = plt.figure(figsize=(12, 6.75), facecolor=BG_COLOR)

    ax = fig.add_axes([0.10, 0.12, 0.86, 0.80])
    ax.set_facecolor(BG_COLOR)

    plot_df = data.reset_index(drop=True)

    # Quadrant backgrounds
    for (hs, lg), (bg_c, _, _) in QUAD.items():
        x0 = GAP_THRESH  if lg  else -15
        x1 = 105          if lg  else GAP_THRESH
        y0 = SHAP_THRESH  if hs  else 0
        y1 = 12           if hs  else SHAP_THRESH
        ax.fill_between([x0, x1], [y0, y0], [y1, y1],
                        color=bg_c, alpha=0.55, zorder=0)

    # Quadrant dividers
    ax.axhline(SHAP_THRESH, color='#999', lw=1.8, ls='--', zorder=1)
    ax.axvline(GAP_THRESH,  color='#999', lw=1.8, ls='--', zorder=1)

    # Quadrant labels
    quad_labels = {
        (True,  True):  (GAP_THRESH + 1,  SHAP_THRESH + 0.25, 'ACT NOW\n(Root Cause + Score Lever)',    '#c0392b', 12.0),
        (True,  False): (-12,              SHAP_THRESH + 0.25, 'PROTECT\n(Existing Strength)',           '#1e8449', 12.0),
        (False, True):  (GAP_THRESH + 1,  0.2,                 'SECOND WAVE\n(Address After Root Cause)','#e67e22', 11.0),
        (False, False): (-12,              0.2,                 'MONITOR',                               '#888',    11.0),
    }
    for (hs, lg), (xt, yt, txt, tc, fs) in quad_labels.items():
        ha = 'left' if lg else 'left'
        ax.text(xt, yt, txt, fontsize=fs, color=tc, fontweight='bold',
                va='bottom', ha='left', zorder=2, alpha=0.7,
                path_effects=[pe.withStroke(linewidth=2, foreground='white')])

    # Plot each feature; labels start at markers then adjust_text separates overlaps.
    label_font = 9.5
    texts = []
    for _, row in plot_df.iterrows():
        ax.scatter(
            row['gap'], row['shap_pct'],
            s=340, color=row['color'], zorder=5,
            edgecolors='white', linewidths=1.8,
        )
        if row['col'] == 'BUILDVALUE':
            pair = (
                f"{row['t_raw']/1e6:.0f}M vs {row['d_raw']/1e6:.0f}M $"
            )
        elif str(row['col']).startswith('biz_'):
            pair = f"{row['t_raw']:,.0f} vs {row['d_raw']:,.0f}"
        else:
            pair = f"{row['t_raw']:.1f} vs {row['d_raw']:.1f}"
        txt = f"{row['label']}\n({pair})"
        t = ax.text(
            row['gap'], row['shap_pct'], txt,
            fontsize=label_font, color=row['color'], fontweight='bold',
            ha='center', va='center', zorder=6,
            path_effects=[pe.withStroke(linewidth=2, foreground='white')],
        )
        texts.append(t)

    adjust_text(
        texts,
        x=plot_df['gap'].values,
        y=plot_df['shap_pct'].values,
        ax=ax,
        expand_points=(1.4, 1.55),
        expand_text=(1.25, 1.4),
        force_points=(0.35, 0.5),
        force_text=(0.45, 0.65),
        lim=800,
        arrowprops=dict(
            arrowstyle='-',
            color='#555555',
            lw=0.65,
            alpha=0.55,
            shrinkA=8,
            shrinkB=4,
        ),
    )

    # Threshold labels
    ax.text(GAP_THRESH - 1, 0.15,
            f'← Delta near target  |  Delta far behind →',
            ha='center', va='bottom', fontsize=10,
            color=DIM_COLOR, style='italic',
            transform=ax.get_xaxis_transform())
    ax.text(1, SHAP_THRESH + 0.06,
            f'SHAP threshold = {SHAP_THRESH}%  ↑ Higher model reliance',
            ha='left', va='bottom', fontsize=10,
            color=DIM_COLOR, style='italic')

    ax.set_xlim(-15, 105)
    ax.set_ylim(0, 12)
    ax.set_xlabel(
        'Delta Gap from Turnaround Level  (normalized 0–100, higher = Delta is further behind)',
        fontsize=12.5, labelpad=8,
    )
    ax.set_ylabel('SHAP Importance  (% of total predictive power)', fontsize=12.5)
    ax.tick_params(labelsize=11)
    for sp in ax.spines.values():
        sp.set_edgecolor(BORDER_CLR)

    ax.set_title(
        'Priority Matrix: SHAP Importance × Delta Gap — What to Fix First',
        fontsize=14, fontweight='bold', pad=8, loc='left',
    )

    # Domain legend
    domain_handles = []
    for col, label, domain, color, _ in FEATURES:
        if not any(h.get_label() == domain for h in domain_handles):
            domain_handles.append(
                mpatches.Patch(color=color, label=domain)
            )
    ax.legend(handles=domain_handles, fontsize=11, loc='upper left',
              facecolor=BG_COLOR, edgecolor=BORDER_CLR, framealpha=0.95)

    out = OUT_DIR / 'slide10_priority_matrix.png'
    plt.savefig(out, dpi=300, bbox_inches='tight',
                facecolor=BG_COLOR, edgecolor='none')
    plt.close()
    print(f'  Saved → {out}')


if __name__ == '__main__':
    print('=' * 60)
    print('  Slide 10 — SHAP × Gap Priority Matrix')
    print('=' * 60)
    data = load_data()
    print(f"\n  {'Feature':<28} {'SHAP%':>6} {'Gap':>7}  Quadrant")
    print('  ' + '-' * 60)
    for _, r in data.sort_values('shap_pct', ascending=False).iterrows():
        hs = '↑' if r['high_shap'] else ' '
        lg = '→' if r['large_gap'] else ' '
        q = 'ACT NOW' if r['high_shap'] and r['large_gap'] else \
            'PROTECT' if r['high_shap'] and not r['large_gap'] else \
            'SECOND WAVE' if not r['high_shap'] and r['large_gap'] else 'MONITOR'
        print(f"  {r['label']:<28} {r['shap_pct']:>6.2f} {r['gap']:>7.1f}  {hs}{lg} {q}")
    print()
    make_figure(data)
    print('  Done.')
