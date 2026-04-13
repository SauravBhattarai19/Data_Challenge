"""
Slide 7 Figure — Delta Chronic Disease Burden
==============================================
"Before We Show You What the Model Found — Look at This."

Produces a presentation-quality comparison of chronic disease prevalence
rates: Mississippi Delta vs National Average, with economic consequence chain.

Output: presentation/figures/slide7_health_burden.png  (300 DPI, 16×9)

Run:  python presentation/fig_slide7_health_burden.py
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
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
PROCESSED = PROJECT_ROOT / 'data_processed'
OUT_DIR   = PROJECT_ROOT / 'presentation' / 'figures'
OUT_DIR.mkdir(parents=True, exist_ok=True)

from config import DELTA_COUNTY_FIPS

# ── Design ────────────────────────────────────────────────────────────────────
BG_COLOR    = '#ffffff'
PANEL_COLOR = '#f5f6f8'
BORDER_CLR  = '#cccccc'
TEXT_COLOR  = '#1a1a2e'
DIM_COLOR   = '#555566'
NATIONAL_C  = '#1a5276'   # deep navy blue
DELTA_C     = '#c0392b'   # deep red
RATIO_C     = '#7b3f00'   # dark burnt-brown for ratio badges
CHAIN_C     = '#6c3483'   # purple for the economic chain
GOLD        = '#c8830a'

plt.rcParams.update({
    'font.family':      'DejaVu Sans',
    'figure.facecolor': BG_COLOR,
    'text.color':       TEXT_COLOR,
    'axes.labelcolor':  TEXT_COLOR,
    'xtick.color':      TEXT_COLOR,
    'ytick.color':      TEXT_COLOR,
    'axes.edgecolor':   BORDER_CLR,
    'axes.facecolor':   PANEL_COLOR,
})

# ── Conditions to show (sorted by ratio, most elevated first) ────────────────
# Labels intentionally plain English, not CDC column names
CONDITIONS = [
    ('STROKE_CrudePrev',    'Stroke'),
    ('MOBILITY_CrudePrev',  'Mobility Disability'),
    ('DIABETES_CrudePrev',  'Diabetes'),
    ('COPD_CrudePrev',      'COPD'),
    ('LPA_CrudePrev',       'Physical Inactivity'),
    ('CSMOKING_CrudePrev',  'Current Smoking'),
    ('BPHIGH_CrudePrev',    'Hypertension'),
    ('OBESITY_CrudePrev',   'Obesity'),
]


def load_data():
    df    = pd.read_parquet(PROCESSED / 'master_tract.parquet')
    delta = df[df['county_fips5'].isin(DELTA_COUNTY_FIPS)]

    rows = []
    for col, label in CONDITIONS:
        if col not in df.columns:
            continue
        nat = df[col].dropna().mean()
        dlt = delta[col].dropna().mean()
        rows.append({
            'col':      col,
            'label':    label,
            'national': nat,
            'delta':    dlt,
            'ratio':    dlt / nat if nat > 0 else 0,
            'gap':      dlt - nat,
        })
    return pd.DataFrame(rows).sort_values('ratio', ascending=True)   # bottom = highest ratio on horiz chart


def make_figure(data: pd.DataFrame):
    fig = plt.figure(figsize=(12, 6.75), facecolor=BG_COLOR)

    gs = gridspec.GridSpec(
        2, 2,
        figure=fig,
        left=0.03, right=0.985,
        top=0.90, bottom=0.06,
        wspace=0.06, hspace=0.38,
        width_ratios=[0.62, 0.38],
        height_ratios=[0.72, 0.28],
    )

    # ══════════════════════════════════════════════════════════════════════════
    # MAIN PANEL — Grouped horizontal bar chart
    # ══════════════════════════════════════════════════════════════════════════
    ax = fig.add_subplot(gs[0, 0])
    ax.set_facecolor(PANEL_COLOR)

    n      = len(data)
    y      = np.arange(n)
    height = 0.36

    bars_nat = ax.barh(y + height/2, data['national'], height,
                       color=NATIONAL_C, alpha=0.85,
                       edgecolor=BORDER_CLR, linewidth=0.5,
                       label='National Average')
    bars_dlt = ax.barh(y - height/2, data['delta'],    height,
                       color=DELTA_C,    alpha=0.88,
                       edgecolor=BORDER_CLR, linewidth=0.5,
                       label='MS Delta Average')

    # Value labels on bars
    for bar in bars_nat:
        w = bar.get_width()
        ax.text(w + 0.4, bar.get_y() + bar.get_height()/2,
                f'{w:.1f}%', va='center', ha='left',
                fontsize=10.5, color=NATIONAL_C, fontweight='bold')

    for bar in bars_dlt:
        w = bar.get_width()
        ax.text(w + 0.4, bar.get_y() + bar.get_height()/2,
                f'{w:.1f}%', va='center', ha='left',
                fontsize=10.5, color=DELTA_C, fontweight='bold')

    # Ratio badges on the right edge
    x_max = ax.get_xlim()[1]
    for i, (_, row) in enumerate(data.iterrows()):
        badge_x = data[['national', 'delta']].max().max() + 12
        ratio_txt = f'{row["ratio"]:.2f}×'
        clr = DELTA_C if row['ratio'] >= 1.5 else RATIO_C
        ax.text(badge_x, y[i], ratio_txt,
                va='center', ha='center', fontsize=12,
                fontweight='black', color=clr,
                path_effects=[pe.withStroke(linewidth=2.5, foreground='white')])

    # "× national" header label for ratio column
    badge_x_hdr = data[['national', 'delta']].max().max() + 12
    ax.text(badge_x_hdr, n + 0.15, '× nat.',
            va='center', ha='center', fontsize=10,
            color=DIM_COLOR, fontweight='bold')

    # Vertical divider before ratio column
    ax.axvline(data[['national', 'delta']].max().max() + 7,
               color=BORDER_CLR, lw=1.0, ls='--', zorder=0)

    # Y-axis condition labels
    ax.set_yticks(y)
    ax.set_yticklabels(data['label'], fontsize=13, fontweight='bold')
    ax.set_xlabel('Prevalence (% of adults)', fontsize=13)
    ax.tick_params(axis='x', labelsize=11)
    ax.set_xlim(0, data[['national', 'delta']].max().max() + 22)
    ax.set_ylim(-0.75, n - 0.25)
    ax.set_title('Mississippi Delta vs. National Average',
                 fontsize=15, fontweight='bold', pad=8, loc='left', color=TEXT_COLOR)

    # Legend
    leg = ax.legend(
        fontsize=12, loc='lower right',
        facecolor=BG_COLOR, edgecolor=BORDER_CLR,
        handles=[
            mpatches.Patch(color=NATIONAL_C, label='National Average'),
            mpatches.Patch(color=DELTA_C,    label='MS Delta Average'),
        ],
    )

    for sp in ax.spines.values():
        sp.set_edgecolor(BORDER_CLR)

    # ══════════════════════════════════════════════════════════════════════════
    # RIGHT PANEL — 4 Big KPI callouts
    # ══════════════════════════════════════════════════════════════════════════
    ax_kpi = fig.add_subplot(gs[0, 1])
    ax_kpi.set_facecolor(PANEL_COLOR)
    ax_kpi.axis('off')
    for sp in ax_kpi.spines.values():
        sp.set_visible(True)
        sp.set_edgecolor(BORDER_CLR)

    # Pick the 4 most impactful stats for callout
    by_ratio = data.sort_values('ratio', ascending=False)
    kpis = [
        (f"{by_ratio.iloc[0]['delta']:.1f}%",
         f"{by_ratio.iloc[0]['label'].upper()}",
         f"vs. {by_ratio.iloc[0]['national']:.1f}% nationally  ({by_ratio.iloc[0]['ratio']:.2f}× the rate)",
         DELTA_C),
        (f"{by_ratio.iloc[2]['delta']:.1f}%",
         f"{by_ratio.iloc[2]['label'].upper()}",
         f"vs. {by_ratio.iloc[2]['national']:.1f}% nationally  ({by_ratio.iloc[2]['ratio']:.2f}× the rate)",
         DELTA_C),
        (f"{by_ratio.iloc[6]['delta']:.1f}%",
         f"{by_ratio.iloc[6]['label'].upper()}",
         f"vs. {by_ratio.iloc[6]['national']:.1f}% nationally  ({by_ratio.iloc[6]['ratio']:.2f}× the rate)",
         '#a04000'),
        (f"{by_ratio.iloc[7]['delta']:.1f}%",
         f"{by_ratio.iloc[7]['label'].upper()}",
         f"vs. {by_ratio.iloc[7]['national']:.1f}% nationally  ({by_ratio.iloc[7]['ratio']:.2f}× the rate)",
         '#a04000'),
    ]

    row_h = 1.0 / len(kpis)
    for i, (big_val, condition, sub, clr) in enumerate(kpis):
        yc   = 1.0 - (i + 0.20) * row_h
        ycon = 1.0 - (i + 0.45) * row_h
        ysub = 1.0 - (i + 0.68) * row_h

        ax_kpi.text(0.08, yc, big_val,
                    transform=ax_kpi.transAxes,
                    fontsize=28, fontweight='black', color=clr, va='center')
        ax_kpi.text(0.08, ycon, condition,
                    transform=ax_kpi.transAxes,
                    fontsize=11.5, fontweight='bold', color=TEXT_COLOR, va='center')
        ax_kpi.text(0.08, ysub, sub,
                    transform=ax_kpi.transAxes,
                    fontsize=9.5, color=DIM_COLOR, va='center')

        if i < len(kpis) - 1:
            ax_kpi.axhline(1.0 - (i+1)*row_h + 0.01,
                           color=BORDER_CLR, lw=0.8, xmin=0.04, xmax=0.96)

    ax_kpi.set_title('The Numbers Behind the Burden',
                     fontsize=13, fontweight='bold', pad=6,
                     loc='left', color=TEXT_COLOR)

    # ══════════════════════════════════════════════════════════════════════════
    # BOTTOM PANEL — Economic consequence chain
    # ══════════════════════════════════════════════════════════════════════════
    ax_chain = fig.add_subplot(gs[1, :])
    ax_chain.set_facecolor('#fef9f0')   # very light warm tint
    ax_chain.axis('off')
    for sp in ax_chain.spines.values():
        sp.set_visible(True)
        sp.set_edgecolor('#e8d5b0')

    # Chain boxes
    chain_steps = [
        ('High Chronic\nDisease Burden',  DELTA_C),
        ('Workforce Cannot\nFully Participate', '#8e1c1c'),
        ('Labor Market\nEngagement Collapses', '#a04000'),
        ('Consumer Spending\nStays Low',  '#7b4f00'),
        ('Commercial Diversity\nShrinks', '#4a4000'),
        ('IGS Stays Below 45\n→ Community Stays Stuck', '#1a3a1a'),
    ]

    n_steps  = len(chain_steps)
    x_start  = 0.03
    x_gap    = (0.97 - x_start) / n_steps
    box_w    = x_gap * 0.72
    box_h    = 0.58
    y_center = 0.50

    for i, (text, clr) in enumerate(chain_steps):
        xc = x_start + x_gap * i + x_gap * 0.14
        # Box
        rect = mpatches.FancyBboxPatch(
            (xc, y_center - box_h/2), box_w, box_h,
            boxstyle='round,pad=0.02',
            facecolor=clr, edgecolor='white',
            linewidth=1.5, alpha=0.90,
            transform=ax_chain.transAxes, zorder=3,
        )
        ax_chain.add_patch(rect)
        ax_chain.text(xc + box_w/2, y_center, text,
                      transform=ax_chain.transAxes,
                      ha='center', va='center',
                      fontsize=10.5, fontweight='bold',
                      color='white', zorder=4,
                      path_effects=[pe.withStroke(linewidth=1.5, foreground=clr)])
        # Arrow to next
        if i < n_steps - 1:
            ax_chain.annotate(
                '', xy=(xc + box_w + x_gap * 0.14 + 0.002, y_center),
                xytext=(xc + box_w + 0.004, y_center),
                xycoords='axes fraction', textcoords='axes fraction',
                arrowprops=dict(arrowstyle='->', color=DIM_COLOR, lw=2.5),
                zorder=5,
            )

    # Chain label
    ax_chain.text(0.5, 0.04,
                  'The Economic Consequence Chain  —  '
                  'This is why disease burden, not provider shortage, predicts whether a community turns around',
                  transform=ax_chain.transAxes,
                  ha='center', va='bottom',
                  fontsize=10.5, color=DIM_COLOR, style='italic')

    # ── Main titles ───────────────────────────────────────────────────────────
    '''
    fig.text(0.5, 0.973,
             'The Mississippi Delta Is Not Just Economically Sick — It Is Chronically Sick',
             ha='center', va='top',
             fontsize=20, fontweight='black', color=TEXT_COLOR)
    fig.text(0.5, 0.948,
             'CDC PLACES 2025 tract-level prevalence  ·  '
             'All rates are % of adult population  ·  '
             'Delta = 9-county Mississippi Delta focus region',
             ha='center', va='top',
             fontsize=12, color=DIM_COLOR)

    fig.text(0.5, 0.018,
             'Source: CDC PLACES 2025  ·  Mastercard IGS 2025  ·  '
             'Jackson State University Data Challenge 2026',
             ha='center', fontsize=10, color='#888899')'''

    out = OUT_DIR / 'slide7_health_burden.png'
    plt.savefig(out, dpi=300, bbox_inches='tight',
                facecolor=BG_COLOR, edgecolor='none')
    plt.close()
    print(f'  Saved → {out}')


if __name__ == '__main__':
    print('=' * 55)
    print('  Slide 7 — Delta Chronic Disease Burden')
    print('=' * 55)
    data = load_data()
    print(f'  Loaded {len(data)} conditions')
    print()
    print(f"  {'Condition':<25} {'National':>10} {'Delta':>10} {'Ratio':>8}")
    print('  ' + '-' * 56)
    for _, row in data.sort_values('ratio', ascending=False).iterrows():
        print(f"  {row['label']:<25} {row['national']:>9.1f}% {row['delta']:>9.1f}%  {row['ratio']:>6.2f}×")
    print()
    make_figure(data)
    print('  Done.')
