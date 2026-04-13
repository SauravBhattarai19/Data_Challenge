"""
Slide 9 Figure — The Turnaround Blueprint
==========================================
The WOW slide. For each top SHAP-predicted feature, shows:
  🟢 What TURNAROUND communities had (national average)
  🔴 What STUCK communities had (national average)
  🔶 Where the MS DELTA is TODAY

Key finding: The Delta is not just "stuck" — it is BELOW stuck on every
single dimension. This directly motivates every prescription.

All values normalized to 0–100 where 100 = best possible,
so all features share one axis and can be compared visually.

Exported figure uses the dumbbell panel only (axes title, no slide headline or
footer), consistent with slide9_shap_framework.png styling.

Output: presentation/figures/slide9_turnaround_blueprint.png (300 DPI, 16×9)
Run:  python presentation/fig_slide9_turnaround_blueprint.py
"""

import sys
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
PROCESSED  = PROJECT_ROOT / 'data_processed'
OUT_DIR    = PROJECT_ROOT / 'presentation' / 'figures'
OUT_DIR.mkdir(parents=True, exist_ok=True)

from config import DELTA_COUNTY_FIPS

# ── Design ────────────────────────────────────────────────────────────────────
BG_COLOR   = '#ffffff'
PANEL_BG   = '#f5f6f8'
BORDER_CLR = '#cccccc'
TEXT_COLOR = '#1a1a2e'
DIM_COLOR  = '#555566'
TURN_C     = '#1e8449'    # dark green  — turnaround
STUCK_C    = '#c0392b'    # deep red    — stuck
DELTA_C    = '#7b3f00'    # dark brown  — Delta (even worse than stuck)
TRACK_C    = '#e8e8e8'    # grey track

plt.rcParams.update({
    'font.family':      'DejaVu Sans',
    'figure.facecolor': BG_COLOR,
    'text.color':       TEXT_COLOR,
    'axes.facecolor':   BG_COLOR,
    'axes.edgecolor':   BORDER_CLR,
})

# ── Features: burden + business anchors + commute + IGS + connectivity + NRI ─
# CBP biz_* columns + IGS MWOB / travel-time + prescription-aligned health/connectivity.
# SHAP % from shap_feature_summary.parquet.
# (column, plain label, domain_color, higher_is_better, unit)
FEATURES = [
    ('biz_food_retail',
     'Food Retail Businesses',            '#6B4226', True,  'establishments'),
    ('biz_pharmacy',
     'Pharmacy Businesses',                '#6B4226', True,  'establishments'),
    ('biz_sector_diversity',
     'Sector Diversity (NAICS)',          '#6B4226', True,  'distinct sectors'),
    ('Commercial Diversity Score_2017',
     'Commercial Diversity Score',        '#c8830a', True,  'score (0–100)'),
    ('Minority/Women Owned Businesses Score_2017',
     'MWOB Score',                        '#c8830a', True,  'score (0–100)'),
    ('Travel Time to Work Score_2017',
     'Travel Time to Work',               '#34495e', True,  'score (0–100)'),
    ('BUILDVALUE',
     'NRI Build Exposure ($)',            '#2c5282', True,  'exposure value ($)'),
    ('igs_economy_2017',
     'IGS Economy (2017)',                '#b8860b', True,  'score (0–100)'),
    ('LPA_CrudePrev',
     'Physical Inactivity',               '#c0392b', False, '% of adults'),
    ('MOBILITY_CrudePrev',
     'Mobility Disability',               '#e74c3c', False, '% of adults'),
    ('DIABETES_CrudePrev',
     'Diabetes Rate',                     '#e67e22', False, '% of adults'),
    ('Internet Access Score_2017',
     'Internet Access Score',             '#2980b9', True,  'score (0–100)'),
    ('Labor Market Engagement Index Score_2017',
     'Labor Market Score',                '#1abc9c', True,  'score (0–100)'),
]


def load_data():
    model_df = pd.read_parquet(PROCESSED / 'igs_improvement_model.parquet')
    model_df['county_fips5'] = model_df['GEOID'].str[:5]

    turn        = model_df[model_df['turnaround'] == 1]
    stuck       = model_df[model_df['turnaround'] == 0]
    delta_model = model_df[model_df['county_fips5'].isin(DELTA_COUNTY_FIPS)]

    shap_lkp = (
        pd.read_parquet(PROCESSED / 'shap_feature_summary.parquet')
        .set_index('feature')['pct_total'].to_dict()
    )

    rows = []
    for col, label, clr, higher_good, unit in FEATURES:
        t_val = turn[col].mean()
        s_val = stuck[col].mean()
        d_val = delta_model[col].mean()

        # Normalize to 0–100 where 100 = BEST
        # Use 5th–95th percentile of the full at-risk population for scaling
        p05 = model_df[col].quantile(0.05)
        p95 = model_df[col].quantile(0.95)
        rng = p95 - p05 if p95 != p05 else 1.0

        def norm(v):
            n = (v - p05) / rng * 100
            n = max(0.0, min(100.0, n))
            return 100 - n if not higher_good else n

        rows.append({
            'col':        col,
            'label':      label,
            'color':      clr,
            'unit':       unit,
            'higher_good': higher_good,
            'shap_pct':   shap_lkp.get(col, 0.0),
            'turn_raw':   t_val,
            'stuck_raw':  s_val,
            'delta_raw':  d_val,
            'turn_norm':  norm(t_val),
            'stuck_norm': norm(s_val),
            'delta_norm': norm(d_val),
            'gap_raw':    abs(t_val - d_val),
        })
    return pd.DataFrame(rows)


def make_figure(data: pd.DataFrame):
    fig = plt.figure(figsize=(8, 4.5), facecolor=BG_COLOR)

    ax = fig.add_axes([0.08, 0.08, 0.90, 0.88])
    ax.set_facecolor(BG_COLOR)

    # Highest SHAP at top of chart, lowest at bottom (barh: larger y → higher on figure).
    plot_df = data.sort_values('shap_pct', ascending=True).reset_index(drop=True)

    n = len(plot_df)
    y_pos = np.arange(n) * 1.5   # space rows

    for i, row in plot_df.iterrows():
        y = y_pos[i]
        t = row['turn_norm']
        s = row['stuck_norm']
        d = row['delta_norm']

        # Track bar (min to max range)
        lo, hi = min(t, s, d) - 3, max(t, s, d) + 3
        ax.barh(y, hi - lo, left=lo, height=0.55,
                color=TRACK_C, zorder=1, alpha=0.7)

        # Connecting line (Turnaround to Stuck)
        ax.plot([t, s], [y, y], color='#999999',
                lw=1.8, zorder=2, solid_capstyle='round')
        # Separate line to Delta (dashed, darker)
        ax.plot([s, d], [y, y], color=DELTA_C,
                lw=1.5, ls='--', zorder=2, solid_capstyle='round', alpha=0.6)

        # Markers
        ax.scatter(t, y, s=220, color=TURN_C,  zorder=5,
                   edgecolors='white', linewidths=1.5)
        ax.scatter(s, y, s=220, color=STUCK_C, zorder=5,
                   edgecolors='white', linewidths=1.5)
        ax.scatter(d, y, s=260, color=DELTA_C, zorder=6,
                   marker='D', edgecolors='white', linewidths=1.5)

        # Raw value labels (BUILDVALUE in millions for readability)
        def _fmt_raw(v):
            if row['col'] == 'BUILDVALUE':
                return f'{v/1e6:.0f}M'
            if str(row['col']).startswith('biz_'):
                return f'{v:,.0f}'
            return f'{v:.1f}'

        offset_t = -3.5 if t > s else +3.5
        offset_s = +3.5 if t > s else -3.5

        ax.text(t + offset_t, y, _fmt_raw(row['turn_raw']),
                va='center', ha='right' if offset_t < 0 else 'left',
                fontsize=8.5, color=TURN_C, fontweight='bold',
                path_effects=[pe.withStroke(linewidth=2, foreground='white')])
        ax.text(s + offset_s, y, _fmt_raw(row['stuck_raw']),
                va='center', ha='left' if offset_s > 0 else 'right',
                fontsize=8.5, color=STUCK_C, fontweight='bold',
                path_effects=[pe.withStroke(linewidth=2, foreground='white')])
        ax.text(d - 1.5, y - 0.35, _fmt_raw(row['delta_raw']),
                va='top', ha='center',
                fontsize=9.5, color=DELTA_C, fontweight='black',
                path_effects=[pe.withStroke(linewidth=2.5, foreground='white')])

        # SHAP % badge on far right
        ax.text(102, y, f'{row["shap_pct"]:.1f}%',
                va='center', ha='left',
                fontsize=9.5, color='#444', fontweight='bold')

    # Y-axis labels
    ax.set_yticks(y_pos)
    yticklabels = []
    for _, row in plot_df.iterrows():
        yticklabels.append(row['label'])
    ax.set_yticklabels(yticklabels, fontsize=12.5, fontweight='bold')

    # Color y-axis tick labels by domain
    for ytick, (_, row) in zip(ax.get_yticklabels(), plot_df.iterrows()):
        ytick.set_color(row['color'])

    ax.set_xlim(-5, 115)
    ax.set_ylim(-0.9, y_pos[-1] + 0.9)
    ax.set_xlabel('← Worse                     Normalized Score (0–100, higher = better)                     Better →',
                  fontsize=11.5, labelpad=8)
    ax.tick_params(axis='x', labelsize=11)
    ax.tick_params(axis='y', left=False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.axvline(0,   color='#dddddd', lw=0.8, zorder=0)
    ax.axvline(100, color='#dddddd', lw=0.8, zorder=0)

    # SHAP % header
    ax.text(102, y_pos[-1] + 0.65, 'SHAP %',
            va='center', ha='left', fontsize=9.5, color=DIM_COLOR, fontweight='bold')

    # Legend (short labels; single row to avoid covering markers)
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=TURN_C,
               markersize=9, label='Turnaround'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=STUCK_C,
               markersize=9, label='Stuck'),
        Line2D([0], [0], marker='D', color='w', markerfacecolor=DELTA_C,
               markersize=8, label='MS Delta'),
    ]
    ax.legend(
        handles=legend_elements, fontsize=9, loc='lower right',
        ncol=3, columnspacing=0.9, handletextpad=0.35, borderaxespad=0.25,
        facecolor=BG_COLOR, edgecolor=BORDER_CLR,
        framealpha=0.95, borderpad=0.35,
    )

    ax.set_title('What Separates Communities That Turn Around from Those That Stay Stuck?',
                 fontsize=14, fontweight='bold', pad=8, loc='left')

    out = OUT_DIR / 'slide9_turnaround_blueprint.png'
    plt.savefig(out, dpi=300, bbox_inches='tight',
                facecolor=BG_COLOR, edgecolor='none')
    plt.close()
    print(f'  Saved → {out}')


if __name__ == '__main__':
    print('=' * 60)
    print('  Slide 9 — Turnaround Blueprint')
    print('=' * 60)
    data = load_data()
    print()
    print(f"  {'Feature':<40} {'Turnaround':>12} {'Stuck':>8} {'Delta':>8}  {'Gap':>8}")
    print('  ' + '-' * 80)
    for _, r in data.iterrows():
        direction = '▼' if not r['higher_good'] else '▲'
        if r['col'] == 'BUILDVALUE':
            print(
                f"  {r['label']:<40} "
                f"{r['turn_raw']/1e6:>11.0f}M{r['stuck_raw']/1e6:>8.0f}M"
                f"{r['delta_raw']/1e6:>8.0f}M  {direction}{r['gap_raw']/1e6:>6.0f}M"
            )
        elif str(r['col']).startswith('biz_'):
            print(
                f"  {r['label']:<40} {r['turn_raw']:>12,.0f} {r['stuck_raw']:>8,.0f} "
                f"{r['delta_raw']:>8,.0f}  {direction}{r['gap_raw']:>8,.0f}"
            )
        else:
            print(f"  {r['label']:<40} {r['turn_raw']:>12.2f} {r['stuck_raw']:>8.2f} "
                  f"{r['delta_raw']:>8.2f}  {direction}{r['gap_raw']:>7.2f}")
    print()
    make_figure(data)
    print('  Done.')
