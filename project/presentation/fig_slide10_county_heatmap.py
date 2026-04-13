"""
Slide 10 Figure — County-Level Gap Heat Map
============================================
"Every County Has a Different Profile."

For all 9 Delta counties × 14 IGS sub-indicators, shows the gap
between the county's current value and the national turnaround benchmark.

  Dark RED   = Far below the turnaround target (urgent gap)
  White      = At or near the turnaround target
  Dark GREEN = Above the turnaround target (strength)

Key message: Some gaps are UNIVERSAL (Internet Access, Labor Market).
Some are COUNTY-SPECIFIC. Each county needs its own prescription.

Exported figure is the heatmap panel only (axes title, no slide headline or footer),
consistent with slide 9 figure styling.

Output: presentation/figures/slide10_county_heatmap.png  (300 DPI, 16×9)
Run:  python presentation/fig_slide10_county_heatmap.py
"""

import sys
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
PROCESSED = PROJECT_ROOT / 'data_processed'
OUT_DIR   = PROJECT_ROOT / 'presentation' / 'figures'
OUT_DIR.mkdir(parents=True, exist_ok=True)

from config import DELTA_COUNTY_FIPS, DELTA_COUNTY_NAMES, IGS_SUB_TO_PILLAR

# ── Design ────────────────────────────────────────────────────────────────────
BG_COLOR   = '#ffffff'
PANEL_BG   = '#f5f6f8'
BORDER_CLR = '#cccccc'
TEXT_COLOR = '#1a1a2e'

# Diverging colormap: deep red → white → dark green
cmap_gap = mcolors.LinearSegmentedColormap.from_list('gap', [
    '#c0392b', '#e74c3c', '#f1948a', '#fdfefe',
    '#82e0aa', '#27ae60', '#1e8449',
])

# Short names for display
SHORT_NAMES = {
    'Small Business Loans Score':             'SB Loans',
    'Minority/Women Owned Businesses Score':  'Min/Women\nOwned Biz',
    'Commercial Diversity Score':             'Commercial\nDiversity',
    'New Businesses Score':                   'New\nBusinesses',
    'Spend Growth Score':                     'Spend\nGrowth',
    'Internet Access Score':                  'Internet\nAccess',
    'Affordable Housing Score':               'Affordable\nHousing',
    'Travel Time to Work Score':              'Travel\nTime',
    'Net Occupancy Score':                    'Net\nOccupancy',
    'Health Insurance Coverage Score':        'Health\nInsurance',
    'Labor Market Engagement Index Score':    'Labor\nMarket',
    'Female Above Poverty Score':             'Female\nAbove Poverty',
    'Personal Income Score':                  'Personal\nIncome',
    'Spending per Capita Score':              'Spending\nper Capita',
}

PILLAR_COLORS = {
    'Economy':   '#c8830a',
    'Place':     '#2980b9',
    'Community': '#8e44ad',
}

plt.rcParams.update({
    'font.family':      'DejaVu Sans',
    'figure.facecolor': BG_COLOR,
    'text.color':       TEXT_COLOR,
})


def load_data():
    delta = pd.read_parquet(PROCESSED / 'delta_profile.parquet')
    bench = pd.read_parquet(PROCESSED / 'turnaround_benchmarks.parquet')
    turn_targets = (bench[bench['typology'] == 'Turnaround']
                    .set_index('indicator')['mean_2025']
                    .to_dict())

    subs = [s for s in IGS_SUB_TO_PILLAR if s in delta.columns]

    # Build gap matrix: rows=counties, cols=sub-indicators
    counties_ordered = sorted(
        DELTA_COUNTY_FIPS,
        key=lambda f: delta[delta['county_fips5'] == f]['igs_score'].mean()
        if f in delta['county_fips5'].values else 999,
    )

    rows = []
    igs_scores = {}
    for fips in counties_ordered:
        cd = delta[delta['county_fips5'] == fips]
        if cd.empty:
            continue
        igs_scores[fips] = cd['igs_score'].mean()
        row = {'fips': fips, 'name': DELTA_COUNTY_NAMES[fips]}
        for s in subs:
            curr = cd[s].mean()
            tgt  = turn_targets.get(s, np.nan)
            row[s] = curr - tgt if not np.isnan(tgt) else np.nan
        rows.append(row)

    gap_df = pd.DataFrame(rows).set_index('fips')
    return gap_df, subs, igs_scores


def make_figure(gap_df, subs, igs_scores):
    n_counties = len(gap_df)
    n_subs     = len(subs)

    fig = plt.figure(figsize=(12, 6.75), facecolor=BG_COLOR)

    ax = fig.add_axes([0.07, 0.08, 0.86, 0.84])
    ax.set_facecolor(BG_COLOR)

    gap_matrix = gap_df[subs].values   # shape: (n_counties, n_subs)

    # Symmetric color scale centred at 0
    abs_max = min(np.nanpercentile(np.abs(gap_matrix), 95), 45)
    norm    = mcolors.TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max)

    im = ax.imshow(gap_matrix, cmap=cmap_gap, norm=norm, aspect='auto')

    # County names (y-axis) with IGS score
    county_labels = []
    for fips in gap_df.index:
        igs = igs_scores.get(fips, 0)
        county_labels.append(
            f"{gap_df.loc[fips, 'name']}  (IGS {igs:.0f})"
        )
    ax.set_yticks(range(n_counties))
    ax.set_yticklabels(county_labels, fontsize=12.5, fontweight='bold')

    # Color county labels by IGS (red = low, green = high)
    igs_vals = [igs_scores.get(f, 50) for f in gap_df.index]
    igs_norm = plt.Normalize(min(igs_vals) - 2, max(igs_vals) + 2)
    igs_cmap = plt.cm.RdYlGn
    for ytick, fips in zip(ax.get_yticklabels(), gap_df.index):
        clr = igs_cmap(igs_norm(igs_scores.get(fips, 50)))
        ytick.set_color(clr[:3])

    # Sub-indicator column headers (x-axis) — color by pillar
    ax.set_xticks(range(n_subs))
    ax.set_xticklabels(
        [SHORT_NAMES.get(s, s) for s in subs],
        fontsize=9.5, rotation=0, ha='center', va='top',
    )
    for xtick, sub in zip(ax.get_xticklabels(), subs):
        pillar = IGS_SUB_TO_PILLAR.get(sub, 'Economy')
        xtick.set_color(PILLAR_COLORS[pillar])
        xtick.set_fontweight('bold')

    # Gap values in each cell
    for r in range(n_counties):
        for c in range(n_subs):
            val = gap_matrix[r, c]
            if np.isnan(val):
                continue
            text_color = 'white' if abs(val) > abs_max * 0.55 else TEXT_COLOR
            ax.text(c, r, f'{val:+.0f}',
                    ha='center', va='center',
                    fontsize=8.5, fontweight='bold', color=text_color)

    # Grid lines
    for x in np.arange(-0.5, n_subs, 1):
        ax.axvline(x, color='white', lw=1.2, zorder=3)
    for y in np.arange(-0.5, n_counties, 1):
        ax.axhline(y, color='white', lw=1.2, zorder=3)

    # Pillar dividers (thicker lines between Economy/Place/Community)
    pillar_groups = {}
    for i, sub in enumerate(subs):
        p = IGS_SUB_TO_PILLAR.get(sub, 'Economy')
        pillar_groups.setdefault(p, []).append(i)

    prev_pillar = IGS_SUB_TO_PILLAR.get(subs[0], 'Economy')
    for i, sub in enumerate(subs[1:], 1):
        p = IGS_SUB_TO_PILLAR.get(sub, 'Economy')
        if p != prev_pillar:
            ax.axvline(i - 0.5, color='#333', lw=2.5, zorder=4)
        prev_pillar = p

    ax.set_xlim(-0.5, n_subs - 0.5)
    ax.set_ylim(n_counties - 0.5, -0.5)
    ax.tick_params(top=False, bottom=True, left=False)
    for sp in ax.spines.values():
        sp.set_edgecolor(BORDER_CLR)

    # Pillar labels under sub-indicator names (blended transform: x=data, y=axes).
    # y is in axes coords (0 = bottom of axes); small negatives only — not -1+.
    _pillar_y_axes = -0.11
    for pillar, indices in pillar_groups.items():
        mid_x = (indices[0] + indices[-1]) / 2
        ax.text(mid_x, _pillar_y_axes, pillar.upper(),
                ha='center', va='top',
                fontsize=11, fontweight='black',
                color=PILLAR_COLORS[pillar],
                transform=ax.get_xaxis_transform())

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, orientation='horizontal',
                        pad=0.12, fraction=0.03, aspect=40)
    cbar.set_label('Gap to Turnaround Benchmark  (points; negative = below target)',
                   fontsize=11)
    cbar.ax.tick_params(labelsize=10)

    ax.set_title('Gap to Turnaround Benchmark — Each Delta County × Each IGS Sub-Indicator',
                 fontsize=13.5, fontweight='bold', pad=36, loc='left')

    out = OUT_DIR / 'slide10_county_heatmap.png'
    plt.savefig(out, dpi=300, bbox_inches='tight',
                facecolor=BG_COLOR, edgecolor='none')
    plt.close()
    print(f'  Saved → {out}')


if __name__ == '__main__':
    print('=' * 55)
    print('  Slide 10 — County Gap Heat Map')
    print('=' * 55)
    gap_df, subs, igs_scores = load_data()
    print(f'  Counties: {len(gap_df)}')
    print(f'  Sub-indicators: {len(subs)}')

    # Print universal gaps
    print('\n  Universal gaps (all counties below target):')
    for sub in subs:
        col = gap_df[sub].dropna()
        if (col < -5).all():
            print(f'    {sub:<45}  mean gap = {col.mean():+.1f}')
    make_figure(gap_df, subs, igs_scores)
    print('  Done.')
