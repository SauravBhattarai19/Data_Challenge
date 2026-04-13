"""
Slide 3 Figure — National IGS County Map  (white background)
=============================================================
Output: presentation/figures/slide3_national_map.png  (300 DPI, 16×9)
Run:  python presentation/fig_slide3_national_map.py
"""

import sys, urllib.request, zipfile, io
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
from scipy.stats import gaussian_kde

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
PROCESSED = PROJECT_ROOT / 'data_processed'
CACHE_DIR = PROJECT_ROOT / 'data_processed' / 'raw'
OUT_DIR   = PROJECT_ROOT / 'presentation' / 'figures'
OUT_DIR.mkdir(parents=True, exist_ok=True)

from config import IGS_VULN_THRESHOLD, DELTA_COUNTY_FIPS, DELTA_COUNTY_NAMES

# ── Design ────────────────────────────────────────────────────────────────────
EXCLUDE_STATES = {'02', '15', '72', '78', '60', '66', '69'}
BG_COLOR    = '#ffffff'
PANEL_COLOR = '#f5f6f8'
BORDER_CLR  = '#cccccc'
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


# ── Data loaders ──────────────────────────────────────────────────────────────
def download_us_counties(cache_dir: Path) -> Path:
    out_path = cache_dir / 'tl_2022_us_county'
    shp_path = out_path / 'tl_2022_us_county.shp'
    if shp_path.exists():
        print('  Shapefile cached.')
        return shp_path
    url = 'https://www2.census.gov/geo/tiger/TIGER2022/COUNTY/tl_2022_us_county.zip'
    print('  Downloading US county shapefile (~75 MB)...')
    out_path.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as r:
        zipfile.ZipFile(io.BytesIO(r.read())).extractall(out_path)
    print('  Done.')
    return shp_path

def load_county_geodata():
    gdf = gpd.read_file(download_us_counties(CACHE_DIR))
    gdf = gdf.rename(columns={'GEOID': 'county_fips5'})
    gdf = gdf[~gdf['STATEFP'].isin(EXCLUDE_STATES)].copy()
    return gdf[['county_fips5', 'STATEFP', 'geometry']].to_crs('EPSG:5070')

def load_igs_data():
    df = pd.read_parquet(PROCESSED / 'igs_national.parquet')
    return df[~df['state_fips'].isin(EXCLUDE_STATES)].copy()


# ── Figure ────────────────────────────────────────────────────────────────────
def make_figure(gdf, igs):
    merged = gdf.merge(
        igs[['county_fips5', 'igs_score', 'state_fips', 'is_delta']],
        on='county_fips5', how='left',
    )

    fig = plt.figure(figsize=(16, 9), facecolor=BG_COLOR)

    # ── Map (left 68%) ────────────────────────────────────────────────────────
    ax_map = fig.add_axes([0.01, 0.07, 0.67, 0.88])
    ax_map.set_facecolor('#eaf2fb')   # light blue ocean wash
    ax_map.axis('off')

    merged.plot(ax=ax_map, column='igs_score', cmap=cmap, norm=norm,
                linewidth=0.06, edgecolor='#bbbbbb',
                missing_kwds={'color': '#dddddd', 'edgecolor': '#cccccc'})

    # State boundaries
    states = merged.dissolve(by='state_fips').reset_index()
    states.boundary.plot(ax=ax_map, linewidth=0.5, color='#777788', zorder=3)

    # Delta highlight
    delta_geom = merged[merged['county_fips5'].isin(DELTA_COUNTY_FIPS)]
    if not delta_geom.empty:
        DELTA_BOX_CLR = '#7b3f00'   # dark burnt-brown — bold on white map

        delta_geom.plot(ax=ax_map, column='igs_score', cmap=cmap, norm=norm,
                        linewidth=2.5, edgecolor=DELTA_BOX_CLR, zorder=5)
        b = delta_geom.total_bounds
        pad = 130000
        ax_map.add_patch(mpatches.FancyBboxPatch(
            (b[0]-pad, b[1]-pad), (b[2]-b[0])+2*pad, (b[3]-b[1])+2*pad,
            boxstyle='round,pad=0', lw=3.0, edgecolor=DELTA_BOX_CLR,
            facecolor='none', zorder=6,
        ))
        cx = (b[0]+b[2])/2
        ax_map.annotate(
            'MS Delta\nIGS 37.9',
            xy=(cx, b[3]+pad*0.4), xytext=(cx-680000, b[3]+580000),
            fontsize=14, color=DELTA_BOX_CLR, fontweight='black',
            ha='center', va='bottom',
            arrowprops=dict(arrowstyle='->', color=DELTA_BOX_CLR, lw=2.2),
            zorder=10,
            path_effects=[pe.withStroke(linewidth=2, foreground='white')],
        )

    # ── Colorbar ──────────────────────────────────────────────────────────────
    ax_cb = fig.add_axes([0.025, 0.12, 0.013, 0.52])
    cb = ColorbarBase(ax_cb, cmap=cmap, norm=norm, orientation='vertical')
    cb.ax.tick_params(labelsize=11, colors=TEXT_COLOR)
    cb.set_label('IGS Score', fontsize=12, labelpad=6, color=TEXT_COLOR)
    cb.outline.set_edgecolor(BORDER_CLR)
    cb.ax.axhline(y=THRESHOLD, color=RED, linewidth=2.5, linestyle='--', zorder=10)
    cb.ax.text(2.0, THRESHOLD, f'← {THRESHOLD}',
               transform=cb.ax.transData, va='center',
               fontsize=10, color=RED, fontweight='bold')

    # ── Titles ────────────────────────────────────────────────────────────────
    fig.text(0.355, 0.975,
             'IGS Score by US County — Economic Vulnerability Map',
             ha='center', va='top', fontsize=20, fontweight='bold', color=TEXT_COLOR)
    fig.text(0.355, 0.948,
             f'Communities below IGS {THRESHOLD} are economically vulnerable  ·  '
             f'Mastercard Inclusive Growth Score 2025',
             ha='center', va='top', fontsize=13, color=DIM_COLOR)

    # ══════════════════════════════════════════════════════════════════════════
    # RIGHT PANELS (two stacked)
    # ══════════════════════════════════════════════════════════════════════════
    scores = igs['igs_score'].dropna()
    n_below   = int((scores < THRESHOLD).sum())
    n_total   = len(scores)
    nat_mean  = scores.mean()

    # ── KPI panel ─────────────────────────────────────────────────────────────
    kpi_ax = fig.add_axes([0.695, 0.50, 0.29, 0.455])
    kpi_ax.set_facecolor(PANEL_COLOR)
    kpi_ax.axis('off')
    for sp in kpi_ax.spines.values():
        sp.set_edgecolor(BORDER_CLR)
        sp.set_visible(True)

    kpis = [
        ('3,221',              'US Counties Analyzed',          BLUE),
        (f'{n_below:,}',       f'Counties Below IGS {THRESHOLD}', RED),
        (f'{n_below/n_total*100:.1f}%', 'Economically Vulnerable', '#a04000'),
        (f'{nat_mean:.1f}',    'National Mean IGS',             GREEN),
        ('37.9',               'MS Delta Mean IGS',             GOLD),
        ('−11.8 pts',          'Delta Gap vs. National',        GOLD),
    ]
    n = len(kpis)
    row_h = 1.0 / n
    for i, (val, lbl, clr) in enumerate(kpis):
        yc = 1.0 - (i + 0.38) * row_h
        yl = 1.0 - (i + 0.72) * row_h
        kpi_ax.text(0.07, yc, val, transform=kpi_ax.transAxes,
                    fontsize=22, fontweight='black', color=clr, va='center')
        kpi_ax.text(0.07, yl, lbl, transform=kpi_ax.transAxes,
                    fontsize=11, color=DIM_COLOR, va='center')
        if i < n - 1:
            kpi_ax.axhline(1.0 - (i+1)*row_h, color=BORDER_CLR,
                           lw=0.8, xmin=0.04, xmax=0.96)

    kpi_ax.set_title('Key Statistics', fontsize=13, fontweight='bold',
                     color=TEXT_COLOR, loc='left', pad=6)

    # ── Distribution panel ────────────────────────────────────────────────────
    ax_dist = fig.add_axes([0.695, 0.09, 0.29, 0.36])
    ax_dist.set_facecolor(PANEL_COLOR)

    x_lin = np.linspace(0, 100, 300)
    kde   = gaussian_kde(scores)(x_lin)

    ax_dist.fill_between(x_lin, kde, alpha=0.15, color=BLUE)
    ax_dist.plot(x_lin, kde, color=BLUE, lw=2, label='All counties')
    mask = x_lin < THRESHOLD
    ax_dist.fill_between(x_lin[mask], kde[mask], alpha=0.50, color=RED,
                         label=f'Vulnerable  (IGS < {THRESHOLD})')
    ax_dist.axvline(THRESHOLD, color=RED,  lw=2.2, ls='--', zorder=5)
    ax_dist.axvline(nat_mean,  color=BLUE, lw=1.8, ls=':',  zorder=5)

    ymax = kde.max()
    ax_dist.text(THRESHOLD + 1.2, ymax * 0.88, f'{THRESHOLD}',
                 color=RED, fontsize=11, fontweight='bold')
    ax_dist.text(nat_mean  + 1.2, ymax * 0.70, f'Nat. avg\n{nat_mean:.1f}',
                 color=BLUE, fontsize=10)

    ax_dist.set_xlim(0, 100)
    ax_dist.set_xlabel('IGS Score', fontsize=12)
    ax_dist.set_ylabel('Density', fontsize=12)
    ax_dist.tick_params(labelsize=11)
    ax_dist.set_title('County IGS Distribution', fontsize=13,
                       fontweight='bold', pad=5, loc='left')
    ax_dist.legend(fontsize=10, loc='upper left',
                   facecolor=BG_COLOR, edgecolor=BORDER_CLR)
    for sp in ax_dist.spines.values():
        sp.set_edgecolor(BORDER_CLR)

    # ── Footer ────────────────────────────────────────────────────────────────
    fig.text(0.5, 0.018,
             'Source: Mastercard IGS 2025  ·  US Census Bureau TIGER 2022  ·  '
             'Jackson State University Data Challenge 2026',
             ha='center', fontsize=10, color='#888899')

    out = OUT_DIR / 'slide3_national_map.png'
    plt.savefig(out, dpi=300, bbox_inches='tight',
                facecolor=BG_COLOR, edgecolor='none')
    plt.close()
    print(f'  Saved → {out}')


if __name__ == '__main__':
    print('=' * 55)
    print('  Slide 3 — National IGS County Map')
    print('=' * 55)
    gdf = load_county_geodata()
    print(f'  {len(gdf):,} counties loaded')
    igs = load_igs_data()
    print(f'  {len(igs):,} IGS records loaded')
    make_figure(gdf, igs)
    print('  Done.')
