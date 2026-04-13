"""
Slide 4 Figure — Mississippi Delta IGS Tract Map
==================================================
"9 Counties. 59 Tracts. A Story in Numbers."

Produces a publication-quality tract-level choropleth of the 9-county
Mississippi Delta with per-county annotations, threshold callouts,
a pillar breakdown panel, and a trend sparkline.

Output: presentation/figures/slide4_delta_map.png  (300 DPI, 16×9)

Run:
    python presentation/fig_slide4_delta_map.py
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
from matplotlib.lines import Line2D
import matplotlib.gridspec as gridspec
import seaborn as sns

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
PROCESSED  = PROJECT_ROOT / 'data_processed'
TIGER_PATH = Path('/data/hpc/disk1/5 Data Challenge/Data/TIGER/tl_2022_28_tract.zip')
OUT_DIR    = PROJECT_ROOT / 'presentation' / 'figures'
OUT_DIR.mkdir(parents=True, exist_ok=True)

from config import (
    IGS_VULN_THRESHOLD, DELTA_COUNTY_FIPS, DELTA_COUNTY_NAMES,
)

# ── Design constants ───────────────────────────────────────────────────────────
BG_COLOR     = '#0d1117'
PANEL_COLOR  = '#161b22'
BORDER_COLOR = '#21262d'
TEXT_COLOR   = '#e6edf3'
DIM_COLOR    = '#8b949e'
ACCENT_GOLD  = '#f0c030'
ACCENT_RED   = '#e05050'
THRESHOLD    = IGS_VULN_THRESHOLD

# Same colormap as slide 3 — consistent visual language
cmap = mcolors.LinearSegmentedColormap.from_list(
    'igs_cmap',
    [
        (0.00, '#c0392b'),
        (0.15, '#e74c3c'),
        (0.30, '#e67e22'),
        (0.45, '#f1c40f'),
        (0.60, '#27ae60'),
        (0.80, '#1abc9c'),
        (1.00, '#0e6655'),
    ],
)
norm = mcolors.Normalize(vmin=0, vmax=100)

COUNTY_LABEL_OFFSETS = {
    # fips: (dx, dy) in map units for label nudge
    '28011': (0,      0),   # Bolivar
    '28027': (0,      0),   # Coahoma
    '28053': (0,      0),   # Humphreys
    '28055': (20000, -15000),  # Issaquena (tiny)
    '28083': (0,      0),   # Leflore
    '28119': (0,      0),   # Quitman
    '28125': (10000, -10000),  # Sharkey
    '28133': (0,      0),   # Sunflower
    '28151': (0,      0),   # Washington
}


def load_data():
    # ── Tract geometry ────────────────────────────────────────────────────────
    gdf = gpd.read_file(TIGER_PATH)
    gdf['county_fips5'] = gdf['STATEFP'] + gdf['COUNTYFP']
    gdf = gdf[gdf['county_fips5'].isin(DELTA_COUNTY_FIPS)].copy()
    gdf = gdf.rename(columns={'GEOID': 'GEOID_shp'})
    gdf['GEOID'] = gdf['STATEFP'] + gdf['COUNTYFP'] + gdf['TRACTCE']
    gdf = gdf.to_crs('EPSG:32615')  # UTM zone 15N — best for Mississippi

    # ── IGS + master tract ────────────────────────────────────────────────────
    master = pd.read_parquet(PROCESSED / 'master_tract.parquet')
    master = master[master['county_fips5'].isin(DELTA_COUNTY_FIPS)].copy()

    gdf = gdf.merge(
        master[['GEOID', 'igs_score', 'igs_economy', 'igs_place', 'igs_community']],
        on='GEOID', how='left',
    )

    # ── IGS trends for sparklines ──────────────────────────────────────────────
    trends = pd.read_parquet(PROCESSED / 'igs_trends.parquet')
    cf = trends['county_fips5'].astype(str).str.strip().str.zfill(5)
    delta_trends = trends[cf.isin(DELTA_COUNTY_FIPS)].copy()
    delta_trends['county_fips5'] = cf[cf.isin(DELTA_COUNTY_FIPS)].values
    return gdf, master, delta_trends


def county_centroids(gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    county_gdf = gdf.dissolve(by='county_fips5').reset_index()
    county_gdf['centroid_x'] = county_gdf.geometry.centroid.x
    county_gdf['centroid_y'] = county_gdf.geometry.centroid.y
    county_gdf['county_name'] = county_gdf['county_fips5'].map(DELTA_COUNTY_NAMES)
    return county_gdf[['county_fips5', 'county_name', 'centroid_x', 'centroid_y',
                        'geometry']]


def make_figure(gdf, master, delta_trends):
    fig = plt.figure(figsize=(20, 11.25), facecolor=BG_COLOR)
    fig.patch.set_facecolor(BG_COLOR)

    # ── Layout: map (left 56%) | panels (right 44%) ───────────────────────────
    gs = gridspec.GridSpec(
        3, 2,
        figure=fig,
        left=0.01, right=0.99,
        top=0.93, bottom=0.04,
        wspace=0.04, hspace=0.35,
        width_ratios=[0.56, 0.44],
        height_ratios=[0.42, 0.32, 0.26],
    )

    ax_map = fig.add_subplot(gs[:, 0])
    ax_map.set_facecolor(BG_COLOR)
    ax_map.axis('off')

    # ── Draw tracts ────────────────────────────────────────────────────────────
    gdf.plot(
        ax=ax_map,
        column='igs_score',
        cmap=cmap, norm=norm,
        linewidth=0.2,
        edgecolor='#1e2030',
        missing_kwds={'color': '#2a2a3e', 'edgecolor': '#2e3248'},
    )

    # ── County boundaries ──────────────────────────────────────────────────────
    county_gdf = county_centroids(gdf)
    county_gdf_geom = gdf.dissolve(by='county_fips5').reset_index()
    county_gdf_geom.boundary.plot(
        ax=ax_map, linewidth=1.8, color=ACCENT_GOLD, zorder=5, alpha=0.85,
    )

    # ── County labels (name + IGS) ─────────────────────────────────────────────
    county_igs = master.groupby('county_fips5')['igs_score'].mean()
    pillar_county = master.groupby('county_fips5')[
        ['igs_economy', 'igs_place', 'igs_community']
    ].mean()

    for _, row in county_gdf.iterrows():
        fips  = row['county_fips5']
        name  = row['county_name']
        cx    = row['centroid_x'] + COUNTY_LABEL_OFFSETS.get(fips, (0, 0))[0]
        cy    = row['centroid_y'] + COUNTY_LABEL_OFFSETS.get(fips, (0, 0))[1]
        score = county_igs.get(fips, np.nan)
        if np.isnan(score):
            continue

        score_color = cmap(norm(score))
        # County name
        ax_map.text(
            cx, cy + 4500, name,
            fontsize=8.5, fontweight='bold',
            color=TEXT_COLOR, ha='center', va='bottom', zorder=8,
            path_effects=[pe.withStroke(linewidth=2.5, foreground='#0d1117')],
        )
        # IGS score badge
        ax_map.text(
            cx, cy - 2000, f'IGS {score:.1f}',
            fontsize=8, fontweight='black',
            color=score_color, ha='center', va='top', zorder=8,
            path_effects=[pe.withStroke(linewidth=2.5, foreground='#0d1117')],
        )

    # ── "ALL BELOW THRESHOLD" banner ───────────────────────────────────────────
    xlim = ax_map.get_xlim(); ylim = ax_map.get_ylim()
    ax_map.text(
        (xlim[0] + xlim[1]) / 2, ylim[1] - (ylim[1] - ylim[0]) * 0.03,
        '⚠  ALL 9 COUNTIES BELOW IGS 45  ⚠',
        ha='center', va='top', fontsize=11, fontweight='black',
        color=ACCENT_RED, zorder=10,
        path_effects=[pe.withStroke(linewidth=3, foreground=BG_COLOR)],
    )

    # ── Colorbar (vertical, left of map) ──────────────────────────────────────
    from matplotlib.colorbar import ColorbarBase
    ax_cb = fig.add_axes([0.025, 0.10, 0.012, 0.55])
    cb = ColorbarBase(ax_cb, cmap=cmap, norm=norm, orientation='vertical')
    cb.ax.tick_params(colors=TEXT_COLOR, labelsize=8)
    cb.set_label('IGS Score', color=TEXT_COLOR, fontsize=9, labelpad=5)
    cb.outline.set_edgecolor(BORDER_COLOR)
    cb.ax.axhline(y=THRESHOLD, color=ACCENT_GOLD, linewidth=2.5,
                  linestyle='--', zorder=10)
    cb.ax.text(2.1, THRESHOLD, f'← {THRESHOLD}',
               transform=cb.ax.transData, va='center',
               fontsize=8, color=ACCENT_GOLD, fontweight='bold')

    # ══════════════════════════════════════════════════════════════════════════
    # RIGHT PANELS
    # ══════════════════════════════════════════════════════════════════════════

    # ── Panel A: County bar chart (ranked by IGS) ────────────────────────────
    ax_bar = fig.add_subplot(gs[0, 1])
    ax_bar.set_facecolor(PANEL_COLOR)

    county_scores = (
        county_igs
        .reset_index()
        .rename(columns={'igs_score': 'igs'})
        .assign(name=lambda x: x['county_fips5'].map(DELTA_COUNTY_NAMES))
        .sort_values('igs', ascending=True)
    )
    bar_colors = [cmap(norm(s)) for s in county_scores['igs']]

    bars = ax_bar.barh(
        county_scores['name'],
        county_scores['igs'],
        color=bar_colors,
        edgecolor=BORDER_COLOR,
        linewidth=0.6,
        height=0.72,
    )
    # Threshold line
    ax_bar.axvline(THRESHOLD, color=ACCENT_GOLD, lw=2, ls='--', zorder=5, alpha=0.9)
    ax_bar.text(
        THRESHOLD + 0.5, len(county_scores) - 0.5,
        f'  Threshold\n  = {THRESHOLD}',
        color=ACCENT_GOLD, fontsize=8, va='top', fontweight='bold',
    )
    # Value labels
    for bar, score in zip(bars, county_scores['igs']):
        ax_bar.text(
            score + 0.4, bar.get_y() + bar.get_height() / 2,
            f'{score:.1f}', va='center', fontsize=8.5,
            color=TEXT_COLOR, fontweight='bold',
        )
    ax_bar.set_xlim(0, 60)
    ax_bar.set_xlabel('Mean IGS Score', color=TEXT_COLOR, fontsize=9)
    ax_bar.tick_params(colors=TEXT_COLOR, labelsize=9)
    for sp in ax_bar.spines.values():
        sp.set_edgecolor(BORDER_COLOR)
    ax_bar.set_facecolor(PANEL_COLOR)
    ax_bar.set_title('County Rankings — All Below Threshold', color=TEXT_COLOR,
                     fontsize=10.5, fontweight='bold', pad=6, loc='left')
    ax_bar.patch.set_alpha(1)

    # ── Panel B: Pillar breakdown (grouped bars) ──────────────────────────────
    ax_pillar = fig.add_subplot(gs[1, 1])
    ax_pillar.set_facecolor(PANEL_COLOR)

    pillar_means = pillar_county.mean()
    national = pd.read_parquet(PROCESSED / 'igs_national.parquet')
    nat_means = national[['igs_economy', 'igs_place', 'igs_community']].mean()

    pillars     = ['Economy', 'Place', 'Community']
    delta_vals  = [pillar_means['igs_economy'],
                   pillar_means['igs_place'],
                   pillar_means['igs_community']]
    nat_vals    = [nat_means['igs_economy'],
                   nat_means['igs_place'],
                   nat_means['igs_community']]

    x     = np.arange(len(pillars))
    width = 0.35

    bars_nat  = ax_pillar.bar(x - width/2, nat_vals,  width,
                               color='#58a6ff', alpha=0.85,
                               edgecolor=BORDER_COLOR, label='National Avg',
                               linewidth=0.6)
    bars_dlt  = ax_pillar.bar(x + width/2, delta_vals, width,
                               color=[ACCENT_RED, '#e67e22', '#e67e22'],
                               alpha=0.9,
                               edgecolor=BORDER_COLOR, label='MS Delta Avg',
                               linewidth=0.6)

    # Value labels
    for bar in bars_nat:
        ax_pillar.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                       f'{bar.get_height():.1f}', ha='center', fontsize=8,
                       color='#58a6ff')
    for bar in bars_dlt:
        ax_pillar.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                       f'{bar.get_height():.1f}', ha='center', fontsize=8,
                       color=ACCENT_GOLD)

    ax_pillar.axhline(THRESHOLD, color=ACCENT_GOLD, lw=1.5, ls='--', alpha=0.7)
    ax_pillar.set_xticks(x)
    ax_pillar.set_xticklabels(pillars, color=TEXT_COLOR, fontsize=9.5)
    ax_pillar.set_ylim(0, 65)
    ax_pillar.set_ylabel('Score', color=TEXT_COLOR, fontsize=9)
    ax_pillar.tick_params(colors=TEXT_COLOR, labelsize=8.5)
    for sp in ax_pillar.spines.values():
        sp.set_edgecolor(BORDER_COLOR)
    legend = ax_pillar.legend(
        fontsize=8.5, facecolor=BG_COLOR,
        edgecolor=BORDER_COLOR, labelcolor=TEXT_COLOR,
        loc='upper right',
    )
    ax_pillar.set_title('IGS Pillars: Delta vs. National', color=TEXT_COLOR,
                        fontsize=10.5, fontweight='bold', pad=6, loc='left')

    # ── Panel C: IGS trend sparkline (2017–2025) ──────────────────────────────
    ax_trend = fig.add_subplot(gs[2, 1])
    ax_trend.set_facecolor(PANEL_COLOR)

    nat_trend   = (pd.read_parquet(PROCESSED / 'igs_trends.parquet')
                   .groupby('year')['igs_score'].mean())
    delta_trend = (delta_trends.groupby('year')['igs_score'].mean())

    years = sorted(set(nat_trend.index) & set(delta_trend.index))

    ax_trend.fill_between(
        years,
        [delta_trend[y] for y in years],
        [nat_trend[y]   for y in years],
        alpha=0.18, color=ACCENT_RED, label='The Gap',
    )
    ax_trend.plot(years, [nat_trend[y] for y in years],
                  color='#58a6ff', lw=2.2, marker='o', ms=4.5, label='National')
    ax_trend.plot(years, [delta_trend[y] for y in years],
                  color=ACCENT_RED, lw=2.5, marker='o', ms=4.5, label='MS Delta')
    ax_trend.axhline(THRESHOLD, color=ACCENT_GOLD, lw=1.8, ls='--', alpha=0.8)

    # Annotate 2017 gap
    gap_2017 = nat_trend[2017] - delta_trend[2017]
    gap_2025 = nat_trend[max(years)] - delta_trend[max(years)]
    ax_trend.annotate(
        f'Gap: −{gap_2017:.1f} pts\n(2017)',
        xy=(2017, (nat_trend[2017] + delta_trend[2017]) / 2),
        fontsize=7.5, color=ACCENT_GOLD, ha='left', va='center',
        xytext=(2017.3, (nat_trend[2017] + delta_trend[2017]) / 2),
    )
    ax_trend.annotate(
        f'−{gap_2025:.1f} pts\n(2025)',
        xy=(max(years), (nat_trend[max(years)] + delta_trend[max(years)]) / 2),
        fontsize=7.5, color=ACCENT_GOLD, ha='right', va='center',
        xytext=(max(years) - 0.3, (nat_trend[max(years)] + delta_trend[max(years)]) / 2),
    )

    ax_trend.set_xlim(min(years) - 0.3, max(years) + 0.3)
    ax_trend.set_ylim(28, 58)
    ax_trend.set_xticks(years)
    ax_trend.set_xticklabels(
        [str(y) for y in years],
        color=TEXT_COLOR, fontsize=8, rotation=30, ha='right',
    )
    ax_trend.tick_params(colors=TEXT_COLOR, labelsize=8)
    for sp in ax_trend.spines.values():
        sp.set_edgecolor(BORDER_COLOR)
    legend = ax_trend.legend(
        fontsize=8, facecolor=BG_COLOR,
        edgecolor=BORDER_COLOR, labelcolor=TEXT_COLOR,
        loc='upper left', ncol=3,
    )
    ax_trend.set_title('IGS Trend 2017–2025', color=TEXT_COLOR,
                       fontsize=10.5, fontweight='bold', pad=6, loc='left')

    # ── Main title ─────────────────────────────────────────────────────────────
    fig.text(
        0.5, 0.975,
        'Mississippi Delta — Tract-Level IGS Analysis',
        ha='center', va='top',
        fontsize=20, fontweight='black', color=TEXT_COLOR,
        path_effects=[pe.withStroke(linewidth=4, foreground=BG_COLOR)],
    )
    fig.text(
        0.5, 0.950,
        '9 Counties · 59 Census Tracts · Every single one below the economic '
        f'vulnerability threshold (IGS {THRESHOLD})',
        ha='center', va='top',
        fontsize=11, color=DIM_COLOR,
    )

    # ── Footer ─────────────────────────────────────────────────────────────────
    ''' fig.text(
        0.5, 0.012,
        'Source: Mastercard IGS 2025 · US Census Bureau TIGER 2022 '
        '(tl_2022_28_tract) · Jackson State University Data Challenge 2026',
        ha='center', fontsize=8.5, color='#484f58',
    ) '''

    out_path = OUT_DIR / 'slide4_delta_map.png'
    plt.savefig(out_path, dpi=300, bbox_inches='tight',
                facecolor=BG_COLOR, edgecolor='none')
    plt.close()
    print(f'\n  Saved → {out_path}')
    return out_path


if __name__ == '__main__':
    print('=' * 60)
    print('  Slide 4 — Mississippi Delta IGS Tract Map')
    print('=' * 60)
    print('\n  Loading spatial and IGS data...')
    gdf, master, delta_trends = load_data()
    print(f'  Delta tracts in TIGER: {len(gdf)}')
    print(f'  Delta tracts with IGS: {gdf["igs_score"].notna().sum()}')
    print('  Rendering figure...')
    make_figure(gdf, master, delta_trends)
    print('\n  Done.')
