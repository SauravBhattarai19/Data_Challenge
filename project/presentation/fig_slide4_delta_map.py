"""
Slide 4 Figure — Mississippi Delta IGS Tract Map  (white background)
=====================================================================
Output: presentation/figures/slide4_delta_map.png  (300 DPI, 16×9)
Run:  python presentation/fig_slide4_delta_map.py
"""

import sys
from pathlib import Path

import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.colorbar import ColorbarBase
import matplotlib.gridspec as gridspec

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
PROCESSED  = PROJECT_ROOT / 'data_processed'
TIGER_PATH = Path('/data/hpc/disk1/5 Data Challenge/Data/TIGER/tl_2022_28_tract.zip')
OUT_DIR    = PROJECT_ROOT / 'presentation' / 'figures'
OUT_DIR.mkdir(parents=True, exist_ok=True)

from config import IGS_VULN_THRESHOLD, DELTA_COUNTY_FIPS, DELTA_COUNTY_NAMES

# ── Design ────────────────────────────────────────────────────────────────────
BG_COLOR    = '#ffffff'
PANEL_COLOR = '#f5f6f8'
BORDER_CLR  = '#cccccc'
MAP_BG      = '#ddeeff'      # light blue for the map background
TEXT_COLOR  = '#1a1a2e'
DIM_COLOR   = '#555566'
GOLD        = '#c8830a'
RED         = '#c0392b'
BLUE        = '#1a5276'
GREEN       = '#1e8449'
THRESHOLD   = IGS_VULN_THRESHOLD

plt.rcParams.update({
    'font.family':      'DejaVu Sans',
    'axes.facecolor':   PANEL_COLOR,
    'figure.facecolor': BG_COLOR,
    'text.color':       TEXT_COLOR,
    'axes.labelcolor':  TEXT_COLOR,
    'xtick.color':      TEXT_COLOR,
    'ytick.color':      TEXT_COLOR,
    'axes.edgecolor':   BORDER_CLR,
})

cmap = mcolors.LinearSegmentedColormap.from_list('igs', [
    (0.00, '#c0392b'), (0.20, '#e67e22'),
    (0.45, '#f39c12'), (0.50, '#f1c40f'),
    (0.60, '#27ae60'), (1.00, '#145a32'),
])
norm = mcolors.Normalize(vmin=0, vmax=100)

# Fine-tuned label nudges per county (dx, dy) in UTM metres
NUDGE = {
    '28011': (0,      8000),   # Bolivar
    '28027': (0,      8000),   # Coahoma
    '28053': (0,      5000),   # Humphreys
    '28055': (18000, -12000),  # Issaquena (tiny sliver)
    '28083': (0,      8000),   # Leflore
    '28119': (0,      8000),   # Quitman
    '28125': (12000,  5000),   # Sharkey
    '28133': (0,      8000),   # Sunflower
    '28151': (0,      8000),   # Washington
}


# ── Data loaders ──────────────────────────────────────────────────────────────
def load_data():
    gdf = gpd.read_file(TIGER_PATH)
    gdf['county_fips5'] = gdf['STATEFP'] + gdf['COUNTYFP']
    gdf = gdf[gdf['county_fips5'].isin(DELTA_COUNTY_FIPS)].copy()
    gdf['GEOID'] = gdf['STATEFP'] + gdf['COUNTYFP'] + gdf['TRACTCE']
    gdf = gdf.to_crs('EPSG:32615')

    master = pd.read_parquet(PROCESSED / 'master_tract.parquet')
    master = master[master['county_fips5'].isin(DELTA_COUNTY_FIPS)]

    gdf = gdf.merge(
        master[['GEOID', 'igs_score', 'igs_economy', 'igs_place', 'igs_community']],
        on='GEOID', how='left',
    )

    trends = pd.read_parquet(PROCESSED / 'igs_trends.parquet')
    cf = trends['county_fips5'].astype(str).str.strip().str.zfill(5)
    delta_trends = trends[cf.isin(DELTA_COUNTY_FIPS)].copy()
    delta_trends['county_fips5'] = cf[cf.isin(DELTA_COUNTY_FIPS)].values

    return gdf, master, delta_trends


# ── Figure ────────────────────────────────────────────────────────────────────
def make_figure(gdf, master, delta_trends):
    fig = plt.figure(figsize=(16, 9), facecolor=BG_COLOR)

    # GridSpec: map (left 55%) | 3 stacked panels (right 45%)
    gs = gridspec.GridSpec(
        3, 2,
        figure=fig,
        left=0.02, right=0.985,
        top=0.91, bottom=0.06,
        wspace=0.05, hspace=0.42,
        width_ratios=[0.55, 0.45],
        height_ratios=[0.44, 0.30, 0.26],
    )

    # ── Map ───────────────────────────────────────────────────────────────────
    ax_map = fig.add_subplot(gs[:, 0])
    ax_map.set_facecolor(MAP_BG)
    ax_map.axis('off')

    gdf.plot(ax=ax_map, column='igs_score', cmap=cmap, norm=norm,
             linewidth=0.3, edgecolor='#aaaaaa',
             missing_kwds={'color': '#dddddd', 'edgecolor': '#bbbbbb'})

    # County boundaries
    county_dissolved = gdf.dissolve(by='county_fips5').reset_index()
    county_dissolved.boundary.plot(ax=ax_map, linewidth=2.0,
                                   color=GOLD, zorder=5, alpha=0.9)

    # County labels
    county_igs = master.groupby('county_fips5')['igs_score'].mean()

    for _, row in county_dissolved.iterrows():
        fips  = row['county_fips5']
        name  = DELTA_COUNTY_NAMES.get(fips, fips)
        score = county_igs.get(fips, np.nan)
        if np.isnan(score):
            continue
        cx = row.geometry.centroid.x + NUDGE.get(fips, (0, 0))[0]
        cy = row.geometry.centroid.y + NUDGE.get(fips, (0, 0))[1]
        score_clr = cmap(norm(score))

        # County name
        ax_map.text(cx, cy + 3500, name,
                    fontsize=10, fontweight='bold', color=TEXT_COLOR,
                    ha='center', va='bottom', zorder=8,
                    path_effects=[pe.withStroke(linewidth=2.5, foreground='white')])
        # IGS score
        ax_map.text(cx, cy - 2500, f'IGS {score:.1f}',
                    fontsize=9.5, fontweight='black', color=score_clr,
                    ha='center', va='top', zorder=8,
                    path_effects=[pe.withStroke(linewidth=2.5, foreground='white')])

    # Warning banner (top of map)
    xlim = ax_map.get_xlim(); ylim = ax_map.get_ylim()
    ax_map.text(
        (xlim[0]+xlim[1])/2, ylim[1] - (ylim[1]-ylim[0])*0.025,
        '⚠  ALL 9 COUNTIES BELOW IGS 45  ⚠',
        ha='center', va='top', fontsize=12.5, fontweight='black',
        color=RED, zorder=10,
        path_effects=[pe.withStroke(linewidth=3, foreground='white')],
    )

    # Colorbar
    ax_cb = fig.add_axes([0.025, 0.12, 0.011, 0.52])
    cb = ColorbarBase(ax_cb, cmap=cmap, norm=norm, orientation='vertical')
    cb.ax.tick_params(labelsize=11, colors=TEXT_COLOR)
    cb.set_label('IGS Score', fontsize=12, labelpad=5, color=TEXT_COLOR)
    cb.outline.set_edgecolor(BORDER_CLR)
    cb.ax.axhline(y=THRESHOLD, color=RED, linewidth=2.5, linestyle='--', zorder=10)
    cb.ax.text(2.0, THRESHOLD, f'← {THRESHOLD}',
               transform=cb.ax.transData, va='center',
               fontsize=10, color=RED, fontweight='bold')

    # ══════════════════════════════════════════════════════════════════════════
    # PANEL A — County bar chart
    # ══════════════════════════════════════════════════════════════════════════
    ax_bar = fig.add_subplot(gs[0, 1])
    ax_bar.set_facecolor(PANEL_COLOR)

    county_scores = (
        county_igs.reset_index()
        .rename(columns={'igs_score': 'igs'})
        .assign(name=lambda x: x['county_fips5'].map(DELTA_COUNTY_NAMES))
        .sort_values('igs', ascending=True)
    )
    bar_colors = [cmap(norm(s)) for s in county_scores['igs']]

    bars = ax_bar.barh(county_scores['name'], county_scores['igs'],
                       color=bar_colors, edgecolor=BORDER_CLR,
                       linewidth=0.6, height=0.70)
    ax_bar.axvline(THRESHOLD, color=RED, lw=2.2, ls='--', zorder=5)
    ax_bar.text(THRESHOLD + 0.4, len(county_scores) - 0.55,
                f'Threshold = {THRESHOLD}',
                color=RED, fontsize=10, fontweight='bold', va='top')

    for bar, score in zip(bars, county_scores['igs']):
        ax_bar.text(score + 0.3, bar.get_y() + bar.get_height()/2,
                    f'{score:.1f}', va='center', fontsize=10,
                    color=TEXT_COLOR, fontweight='bold')

    ax_bar.set_xlim(0, 56)
    ax_bar.set_xlabel('Mean IGS Score', fontsize=12)
    ax_bar.tick_params(labelsize=11)
    for sp in ax_bar.spines.values():
        sp.set_edgecolor(BORDER_CLR)
    ax_bar.set_title('County IGS Rankings', fontsize=13,
                     fontweight='bold', pad=5, loc='left')

    # ══════════════════════════════════════════════════════════════════════════
    # PANEL B — Pillar comparison
    # ══════════════════════════════════════════════════════════════════════════
    ax_pil = fig.add_subplot(gs[1, 1])
    ax_pil.set_facecolor(PANEL_COLOR)

    pillar_delta = master.groupby('county_fips5')[
        ['igs_economy','igs_place','igs_community']].mean().mean()
    nat = pd.read_parquet(PROCESSED / 'igs_national.parquet')
    pillar_nat = nat[['igs_economy','igs_place','igs_community']].mean()

    labels = ['Economy', 'Place', 'Community']
    d_vals = [pillar_delta['igs_economy'], pillar_delta['igs_place'],
               pillar_delta['igs_community']]
    n_vals = [pillar_nat['igs_economy'],   pillar_nat['igs_place'],
               pillar_nat['igs_community']]

    x = np.arange(len(labels))
    w = 0.36
    br_nat  = ax_pil.bar(x - w/2, n_vals, w, color=BLUE,  alpha=0.82,
                          edgecolor=BORDER_CLR, lw=0.6, label='National Avg')
    br_dlt  = ax_pil.bar(x + w/2, d_vals, w, color=RED,   alpha=0.82,
                          edgecolor=BORDER_CLR, lw=0.6, label='MS Delta Avg')

    for bar in list(br_nat) + list(br_dlt):
        ax_pil.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.6,
                    f'{bar.get_height():.1f}', ha='center',
                    fontsize=10, fontweight='bold', color=TEXT_COLOR)

    ax_pil.axhline(THRESHOLD, color=RED, lw=1.8, ls='--', alpha=0.7)
    ax_pil.text(2.52, THRESHOLD + 0.8, f'{THRESHOLD}',
                color=RED, fontsize=9.5, fontweight='bold')
    ax_pil.set_xticks(x)
    ax_pil.set_xticklabels(labels, fontsize=12)
    ax_pil.set_ylim(0, 68)
    ax_pil.set_ylabel('Score', fontsize=12)
    ax_pil.tick_params(labelsize=11)
    ax_pil.legend(fontsize=10, loc='upper left',
                  facecolor=BG_COLOR, edgecolor=BORDER_CLR)
    for sp in ax_pil.spines.values():
        sp.set_edgecolor(BORDER_CLR)
    ax_pil.set_title('IGS Pillars: Delta vs. National', fontsize=13,
                      fontweight='bold', pad=5, loc='left')

    # ══════════════════════════════════════════════════════════════════════════
    # PANEL C — Trend sparkline
    # ══════════════════════════════════════════════════════════════════════════
    ax_tr = fig.add_subplot(gs[2, 1])
    ax_tr.set_facecolor(PANEL_COLOR)

    all_trends = pd.read_parquet(PROCESSED / 'igs_trends.parquet')
    nat_trend   = all_trends.groupby('year')['igs_score'].mean()
    delta_trend = delta_trends.groupby('year')['igs_score'].mean()
    years = sorted(set(nat_trend.index) & set(delta_trend.index))

    nat_y   = [nat_trend[y]   for y in years]
    delta_y = [delta_trend[y] for y in years]

    ax_tr.fill_between(years, delta_y, nat_y, alpha=0.15, color=RED)
    ax_tr.plot(years, nat_y,   color=BLUE, lw=2.2, marker='o', ms=4, label='National')
    ax_tr.plot(years, delta_y, color=RED,  lw=2.5, marker='o', ms=4, label='MS Delta')
    ax_tr.axhline(THRESHOLD, color=RED, lw=1.8, ls='--', alpha=0.7)

    # Annotate gap endpoints only (avoids clutter)
    g0 = nat_trend[years[0]]  - delta_trend[years[0]]
    gN = nat_trend[years[-1]] - delta_trend[years[-1]]
    mid0 = (nat_trend[years[0]]  + delta_trend[years[0]])  / 2
    midN = (nat_trend[years[-1]] + delta_trend[years[-1]]) / 2

    ax_tr.annotate(f'Gap\n−{g0:.1f}', xy=(years[0],  mid0),
                   xytext=(years[0]+0.25, mid0),
                   fontsize=9, color=GOLD, fontweight='bold', va='center')
    ax_tr.annotate(f'−{gN:.1f}', xy=(years[-1], midN),
                   xytext=(years[-1]-0.6, midN),
                   fontsize=9, color=GOLD, fontweight='bold', va='center')

    ax_tr.set_xlim(years[0]-0.4, years[-1]+0.4)
    ax_tr.set_ylim(30, 57)
    ax_tr.set_xticks(years)
    ax_tr.set_xticklabels([str(y) for y in years], fontsize=10, rotation=30, ha='right')
    ax_tr.set_ylabel('Mean IGS', fontsize=11)
    ax_tr.tick_params(labelsize=10)
    ax_tr.legend(fontsize=10, loc='lower right',
                 facecolor=BG_COLOR, edgecolor=BORDER_CLR, ncol=2)
    for sp in ax_tr.spines.values():
        sp.set_edgecolor(BORDER_CLR)
    ax_tr.set_title('IGS Trend 2017–2025', fontsize=13,
                     fontweight='bold', pad=5, loc='left')

    # ── Titles ────────────────────────────────────────────────────────────────
    '''fig.text(0.5, 0.975,
             'Mississippi Delta — Tract-Level IGS Analysis',
             ha='center', va='top', fontsize=21,
             fontweight='black', color=TEXT_COLOR)
    fig.text(0.5, 0.950,
             f'9 Counties · 59 Census Tracts · All below the economic '
             f'vulnerability threshold (IGS {THRESHOLD})',
             ha='center', va='top', fontsize=13, color=DIM_COLOR)

    fig.text(0.5, 0.020,
             'Source: Mastercard IGS 2025  ·  US Census Bureau TIGER 2022  ·  '
             'Jackson State University Data Challenge 2026',
             ha='center', fontsize=10, color='#888899') '''

    out = OUT_DIR / 'slide4_delta_map.png'
    plt.savefig(out, dpi=300, bbox_inches='tight',
                facecolor=BG_COLOR, edgecolor='none')
    plt.close()
    print(f'  Saved → {out}')


if __name__ == '__main__':
    print('=' * 55)
    print('  Slide 4 — MS Delta IGS Tract Map')
    print('=' * 55)
    gdf, master, delta_trends = load_data()
    print(f'  Delta tracts: {len(gdf)} ({gdf["igs_score"].notna().sum()} with IGS)')
    make_figure(gdf, master, delta_trends)
    print('  Done.')
