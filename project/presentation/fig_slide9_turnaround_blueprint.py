"""
Slide 9 Figure — The Turnaround Blueprint
==========================================
For each top SHAP-predicted feature, shows:
  🟢 What TURNAROUND communities had (national average)
  🔴 What STUCK communities had (national average)
  🔶 Where the MS DELTA is TODAY

Key finding: the Delta is not just "stuck" — it is BELOW stuck on almost
every single dimension. This directly motivates every investment priority.

All values normalised to 0–100 where 100 = best possible,
so all features share one axis and can be compared visually.

Data sources (current):
  data_processed/expanded_model.parquet      — 25K tracts × 89 features + outcome
  data_processed/expanded_shap_summary.parquet — per-feature SHAP stats + direction
  03_analysis/expanded_priority_matrix.py    — FEATURE_REGISTRY, CATEGORY_COLORS

Output: presentation/figures/slide9_turnaround_blueprint.png (300 DPI)
Run:    python presentation/fig_slide9_turnaround_blueprint.py
"""

import sys, io, contextlib, importlib.util
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
PROCESSED = PROJECT_ROOT / 'data_processed'
OUT_DIR   = PROJECT_ROOT / 'presentation' / 'figures'
OUT_DIR.mkdir(parents=True, exist_ok=True)

from config import DELTA_COUNTY_FIPS

# ── Load FEATURE_REGISTRY and CATEGORY_COLORS from analysis module ────────────
_spec = importlib.util.spec_from_file_location(
    'expanded_priority_matrix',
    PROJECT_ROOT / '03_analysis' / 'expanded_priority_matrix.py'
)
_mod = importlib.util.module_from_spec(_spec)
with contextlib.redirect_stdout(io.StringIO()):
    _spec.loader.exec_module(_mod)

FEATURE_REGISTRY = _mod.FEATURE_REGISTRY
CATEGORY_COLORS  = _mod.CATEGORY_COLORS
_reg = {row[0]: row for row in FEATURE_REGISTRY}   # col → (col, label, cat, hib)

# ── Features to show: top SHAP features, ordered by rank, excluding
#    non-actionable demographics (PCT_MINRTY, PCT_TractKids) and
#    BUILDVALUE (building exposure — ambiguous framing for a dumbbell chart)
SHOW_FEATURES = [
    'DENTAL_CrudePrev',
    'Commercial Diversity Score_2017',
    'LPA_CrudePrev',
    'MOBILITY_CrudePrev',
    'DIABETES_CrudePrev',
    'Internet Access Score_2017',
    'Labor Market Engagement Index Score_2017',
    'SLEEP_CrudePrev',
    'Acres of Park Land Score_2017',
    'STROKE_CrudePrev',
    'CSMOKING_CrudePrev',
    'PCT_UNEMP',
]

# ── Design ────────────────────────────────────────────────────────────────────
BG_COLOR   = '#ffffff'
BORDER_CLR = '#cccccc'
TEXT_COLOR = '#1a1a2e'
DIM_COLOR  = '#555566'
TURN_C     = '#1e8449'   # dark green — turnaround
STUCK_C    = '#c0392b'   # deep red   — stuck
DELTA_C    = '#7b3f00'   # dark brown — Delta (often below stuck)
TRACK_C    = '#e8e8e8'   # grey track

plt.rcParams.update({
    'font.family':      'DejaVu Sans',
    'figure.facecolor': BG_COLOR,
    'text.color':       TEXT_COLOR,
    'axes.facecolor':   BG_COLOR,
    'axes.edgecolor':   BORDER_CLR,
})


# ── Data loading ──────────────────────────────────────────────────────────────
def load_data() -> pd.DataFrame:
    em = pd.read_parquet(PROCESSED / 'expanded_model.parquet')
    ss = pd.read_parquet(PROCESSED / 'expanded_shap_summary.parquet')

    em['county_fips5'] = em['GEOID'].str[:5]
    turn  = em[em['turnaround'] == 1]
    stuck = em[em['turnaround'] == 0]
    delta = em[em['county_fips5'].isin(DELTA_COUNTY_FIPS)]

    shap_lkp = ss.set_index('feature')['pct_total'].to_dict()

    rows = []
    for col in SHOW_FEATURES:
        if col not in em.columns:
            print(f'  [skip] {col} not in expanded_model')
            continue
        if col not in _reg:
            print(f'  [skip] {col} not in FEATURE_REGISTRY')
            continue

        _, label, category, higher_good = _reg[col]
        color = CATEGORY_COLORS.get(category, '#888888')

        t_val = turn[col].mean()
        s_val = stuck[col].mean()
        d_val = delta[col].mean()

        # Normalise to 0–100 where 100 = BEST POSSIBLE
        # Scale on 5th–95th percentile of all at-risk tracts
        p05 = em[col].quantile(0.05)
        p95 = em[col].quantile(0.95)
        rng = p95 - p05 if p95 != p05 else 1.0

        def norm(v, hib=higher_good):
            n = np.clip((v - p05) / rng * 100, 0, 100)
            return n if hib else 100 - n

        rows.append({
            'col':       col,
            'label':     label,
            'category':  category,
            'color':     color,
            'hib':       higher_good,
            'shap_pct':  shap_lkp.get(col, 0.0),
            'turn_raw':  t_val,
            'stuck_raw': s_val,
            'delta_raw': d_val,
            'turn_norm': norm(t_val),
            'stuck_norm':norm(s_val),
            'delta_norm':norm(d_val),
        })

    return pd.DataFrame(rows)


# ── Figure ────────────────────────────────────────────────────────────────────
def make_figure(data: pd.DataFrame):
    # Sort: highest SHAP at top of figure
    plot_df = data.sort_values('shap_pct', ascending=True).reset_index(drop=True)
    n    = len(plot_df)
    y_pos = np.arange(n) * 1.6

    fig = plt.figure(figsize=(8, 5.6), facecolor=BG_COLOR)
    ax  = fig.add_axes([0.08, 0.13, 0.90, 0.83])
    ax.set_facecolor(BG_COLOR)

    for i, row in plot_df.iterrows():
        y = y_pos[i]
        t, s, d = row['turn_norm'], row['stuck_norm'], row['delta_norm']

        # ── Track bar ──────────────────────────────────────────────────────
        lo = min(t, s, d) - 3
        hi = max(t, s, d) + 3
        ax.barh(y, hi - lo, left=lo, height=0.55,
                color=TRACK_C, zorder=1, alpha=0.7)

        # ── Connecting lines ───────────────────────────────────────────────
        ax.plot([t, s], [y, y], color='#999999',
                lw=1.8, zorder=2, solid_capstyle='round')
        ax.plot([s, d], [y, y], color=DELTA_C,
                lw=1.5, ls='--', zorder=2, solid_capstyle='round', alpha=0.65)

        # ── Dots ───────────────────────────────────────────────────────────
        ax.scatter(t, y, s=220, color=TURN_C,  zorder=5,
                   edgecolors='white', linewidths=1.5)
        ax.scatter(s, y, s=220, color=STUCK_C, zorder=5,
                   edgecolors='white', linewidths=1.5)
        ax.scatter(d, y, s=260, color=DELTA_C, zorder=6,
                   marker='D', edgecolors='white', linewidths=1.5)

        # ── Raw value labels ───────────────────────────────────────────────
        def _fmt(v, col=row['col']):
            if 'Score' in col:
                return f'{v:.1f}'
            if col.endswith('CrudePrev'):
                return f'{v:.1f}%'
            if col == 'PCT_UNEMP':
                return f'{v:.1f}%'
            return f'{v:.1f}'

        # Offset direction: turnaround label goes away from stuck
        off_t = -3.5 if t > s else +3.5
        off_s = +3.5 if t > s else -3.5

        stroke = [pe.withStroke(linewidth=2, foreground='white')]

        ax.text(t + off_t, y, _fmt(row['turn_raw']),
                va='center', ha='right' if off_t < 0 else 'left',
                fontsize=8.5, color=TURN_C, fontweight='bold',
                path_effects=stroke)
        ax.text(s + off_s, y, _fmt(row['stuck_raw']),
                va='center', ha='left' if off_s > 0 else 'right',
                fontsize=8.5, color=STUCK_C, fontweight='bold',
                path_effects=stroke)
        ax.text(d - 1.5, y - 0.38, _fmt(row['delta_raw']),
                va='top', ha='center',
                fontsize=9.5, color=DELTA_C, fontweight='black',
                path_effects=[pe.withStroke(linewidth=2.5, foreground='white')])

        # ── SHAP % badge ───────────────────────────────────────────────────
        ax.text(102, y, f"{row['shap_pct']:.1f}%",
                va='center', ha='left',
                fontsize=9.5, color='#444', fontweight='bold')

    # ── Y-axis labels (coloured by domain) ────────────────────────────────
    ax.set_yticks(y_pos)
    ax.set_yticklabels(plot_df['label'], fontsize=12, fontweight='bold')
    for tick, (_, row) in zip(ax.get_yticklabels(), plot_df.iterrows()):
        tick.set_color(row['color'])

    # ── Axes decoration ────────────────────────────────────────────────────
    ax.set_xlim(-5, 115)
    ax.set_ylim(-0.9, y_pos[-1] + 0.9)
    ax.set_xlabel(
        '← Worse                 Normalised Score (0–100, higher = better)                 Better →',
        fontsize=11, labelpad=8
    )
    ax.tick_params(axis='x', labelsize=10)
    ax.tick_params(axis='y', left=False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.axvline(0,   color='#dddddd', lw=0.8, zorder=0)
    ax.axvline(100, color='#dddddd', lw=0.8, zorder=0)

    # SHAP % column header
    ax.text(102, y_pos[-1] + 0.75, 'SHAP %',
            va='center', ha='left',
            fontsize=9.5, color=DIM_COLOR, fontweight='bold')

    # ── Legend ────────────────────────────────────────────────────────────
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=TURN_C,
               markersize=9, label='Turnaround tracts'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=STUCK_C,
               markersize=9, label='Stuck tracts'),
        Line2D([0], [0], marker='D', color='w', markerfacecolor=DELTA_C,
               markersize=8, label='MS Delta (today)'),
    ]
    ax.legend(
        handles=legend_elements, fontsize=9,
        loc='upper center', bbox_to_anchor=(0.42, -0.16),
        ncol=3, columnspacing=1.2, handletextpad=0.4, borderaxespad=0.3,
        facecolor=BG_COLOR, edgecolor=BORDER_CLR,
        framealpha=0.95, borderpad=0.4,
    )

    ax.set_title(
        'What Separates Communities That Turn Around from Those That Stay Stuck?',
        fontsize=13.5, fontweight='bold', pad=8, loc='left'
    )

    out = OUT_DIR / 'slide9_turnaround_blueprint.png'
    plt.savefig(out, dpi=300, bbox_inches='tight',
                facecolor=BG_COLOR, edgecolor='none')
    plt.close()
    print(f'  Saved → {out}')


# ── CLI summary ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print('=' * 65)
    print('  Slide 9 — Turnaround Blueprint  (expanded model)')
    print('=' * 65)

    data = load_data()
    plot_df = data.sort_values('shap_pct', ascending=False).reset_index(drop=True)

    print(f"\n  {'Feature':<42} {'Turn':>8} {'Stuck':>8} {'Delta':>8}  {'SHAP%':>6}")
    print('  ' + '-' * 80)
    for _, r in plot_df.iterrows():
        arrow = '▲' if r['hib'] else '▼'
        print(
            f"  {r['label']:<42} "
            f"{r['turn_raw']:>8.2f} {r['stuck_raw']:>8.2f} "
            f"{r['delta_raw']:>8.2f}  {arrow}{r['shap_pct']:>5.1f}%"
        )

    print()
    make_figure(data)
    print('  Done.')
