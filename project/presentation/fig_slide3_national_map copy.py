"""
Slide 3 Figure — National IGS County Map
=========================================
"24,061 Tracts. 66% Stuck. This Is Not a Local Problem."

Produces a presentation-quality US county-level choropleth of IGS scores
with inset distribution histogram, Delta callout, and key statistics.

Output: presentation/figures/slide3_national_map.png  (300 DPI, 16×9)

Run:
    python presentation/fig_slide3_national_map.py
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
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import seaborn as sns
from scipy.stats import gaussian_kde

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
PROCESSED    = PROJECT_ROOT / 'data_processed'
CACHE_DIR    = PROJECT_ROOT / 'data_processed' / 'raw'
OUT_DIR      = PROJECT_ROOT / 'presentation' / 'figures'
OUT_DIR.mkdir(parents=True, exist_ok=True)

from config import IGS_VULN_THRESHOLD, DELTA_COUNTY_FIPS, DELTA_COUNTY_NAMES

# ── Constants ─────────────────────────────────────────────────────────────────
EXCLUDE_STATES = {'02', '15', '72', '78', '60', '66', '69'}  # AK, HI, territories
BG_COLOR       = '#0d1117'
PANEL_COLOR    = '#161b22'
TEXT_COLOR     = '#e6edf3'
ACCENT_GOLD    = '#f0c030'
ACCENT_RED     = '#e05050'
THRESHOLD      = IGS_VULN_THRESHOLD

# ── Custom colormap: red (0) → gold (45) → teal-green (100) ──────────────────
cmap_colors = [
    (0.0,   '#c0392b'),
    (0.15,  '#e74c3c'),
    (0.30,  '#e67e22'),
    (0.45,  '#f1c40f'),   # threshold at 45 → pivot = 0.45
    (0.60,  '#27ae60'),
    (0.80,  '#1abc9c'),
    (1.00,  '#0e6655'),
]
cmap = mcolors.LinearSegmentedColormap.from_list(
    'igs_cmap',
    [(v, mcolors.to_rgb(c)) for v, c in cmap_colors],
)
norm = mcolors.Normalize(vmin=0, vmax=100)


def download_us_counties(cache_dir: Path) -> Path:
    """Download Census TIGER 2022 US county shapefile (cached)."""
    out_path = cache_dir / 'tl_2022_us_county'
    shp_path = out_path / 'tl_2022_us_county.shp'
    if shp_path.exists():
        print('  US counties shapefile already cached.')
        return shp_path
    url = 'https://www2.census.gov/geo/tiger/TIGER2022/COUNTY/tl_2022_us_county.zip'
    print(f'  Downloading US counties shapefile (~75 MB)...')
    out_path.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as resp:
        zf = zipfile.ZipFile(io.BytesIO(resp.read()))
        zf.extractall(out_path)
    print('  Download complete.')
    return shp_path


def load_county_geodata() -> gpd.GeoDataFrame:
    shp_path = download_us_counties(CACHE_DIR)
    gdf = gpd.read_file(shp_path)
    gdf = gdf.rename(columns={'GEOID': 'county_fips5'})
    gdf = gdf[~gdf['STATEFP'].isin(EXCLUDE_STATES)].copy()
    gdf = gdf.to_crs('EPSG:5070')   # Albers Equal Area — best for CONUS
    return gdf[['county_fips5', 'geometry']]


def load_igs_data() -> pd.DataFrame:
    df = pd.read_parquet(PROCESSED / 'igs_national.parquet')
    df = df[~df['state_fips'].isin(EXCLUDE_STATES)].copy()
    return df


def make_figure(gdf: gpd.GeoDataFrame, igs: pd.DataFrame):
    merged = gdf.merge(igs[['county_fips5', 'igs_score', 'pct_below_45',
                             'is_delta', 'state_fips']],
                       on='county_fips5', how='left')

    # ── Canvas ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(20, 11.25), facecolor=BG_COLOR)   # 16:9 @ 20"
    fig.patch.set_facecolor(BG_COLOR)

    # ── Main map axis ─────────────────────────────────────────────────────────
    ax_map = fig.add_axes([0.0, 0.06, 0.72, 0.90])
    ax_map.set_facecolor(BG_COLOR)
    ax_map.axis('off')

    # Draw counties
    merged.plot(
        ax=ax_map,
        column='igs_score',
        cmap=cmap,
        norm=norm,
        linewidth=0.08,
        edgecolor='#2a2a3e',
        missing_kwds={'color': '#2a2a3e', 'edgecolor': '#3a3a4e'},
    )

    # State boundaries overlay
    states_gdf = merged.dissolve(by='state_fips').reset_index()
    states_gdf.boundary.plot(ax=ax_map, linewidth=0.4, color='#4a4a6a', zorder=3)

    # Highlight Delta counties with gold border
    delta_geom = merged[merged['county_fips5'].isin(DELTA_COUNTY_FIPS)]
    if not delta_geom.empty:
        delta_geom.plot(
            ax=ax_map,
            column='igs_score',
            cmap=cmap,
            norm=norm,
            linewidth=2.0,
            edgecolor=ACCENT_GOLD,
            zorder=5,
        )
        # Callout box around Delta
        bounds = delta_geom.total_bounds  # minx, miny, maxx, maxy
        pad = 120000  # metres in Albers
        rect = mpatches.FancyBboxPatch(
            (bounds[0] - pad, bounds[1] - pad),
            (bounds[2] - bounds[0]) + 2 * pad,
            (bounds[3] - bounds[1]) + 2 * pad,
            boxstyle='round,pad=0',
            linewidth=2.5,
            edgecolor=ACCENT_GOLD,
            facecolor='none',
            zorder=6,
        )
        ax_map.add_patch(rect)

        # Arrow + label pointing to Delta
        cx = (bounds[0] + bounds[2]) / 2
        cy = (bounds[1] + bounds[3]) / 2
        ax_map.annotate(
            'MS Delta\nIGS 37.9',
            xy=(cx, bounds[3] + pad * 0.5),
            xytext=(cx - 700000, bounds[3] + 600000),
            fontsize=11, color=ACCENT_GOLD, fontweight='bold',
            ha='center', va='bottom',
            arrowprops=dict(arrowstyle='->', color=ACCENT_GOLD, lw=1.8),
            zorder=10,
        )

    # ── Title ─────────────────────────────────────────────────────────────────
    fig.text(
        0.36, 0.975,
        'IGS Score by US County — Economic Vulnerability Map',
        ha='center', va='top',
        fontsize=18, fontweight='bold', color=TEXT_COLOR,
        path_effects=[pe.withStroke(linewidth=3, foreground=BG_COLOR)],
    )
    fig.text(
        0.36, 0.945,
        f'Counties below IGS {THRESHOLD} are economically vulnerable  ·  '
        f'Mastercard Inclusive Growth Score 2025',
        ha='center', va='top',
        fontsize=11, color='#8b949e',
    )

    # ── Colorbar ──────────────────────────────────────────────────────────────
    ax_cb = fig.add_axes([0.02, 0.10, 0.015, 0.55])
    cb = ColorbarBase(ax_cb, cmap=cmap, norm=norm, orientation='vertical')
    cb.ax.tick_params(colors=TEXT_COLOR, labelsize=9)
    cb.set_label('IGS Score', color=TEXT_COLOR, fontsize=10, labelpad=6)
    cb.outline.set_edgecolor('#3a3a4e')
    # Threshold marker
    cb.ax.axhline(y=THRESHOLD, color=ACCENT_GOLD, linewidth=2.5, linestyle='--', zorder=10)
    cb.ax.text(
        1.8, THRESHOLD, f'← {THRESHOLD} threshold',
        transform=cb.ax.transData, va='center',
        fontsize=8.5, color=ACCENT_GOLD, fontweight='bold',
    )

    # ── Right panel: stats + histogram ───────────────────────────────────────
    ax_stats = fig.add_axes([0.73, 0.55, 0.25, 0.38])
    ax_stats.set_facecolor(PANEL_COLOR)
    for sp in ax_stats.spines.values():
        sp.set_edgecolor('#30363d')

    # KDE distributions
    scores_all   = igs['igs_score'].dropna()
    scores_below = igs.loc[igs['igs_score'] < THRESHOLD, 'igs_score']
    scores_above = igs.loc[igs['igs_score'] >= THRESHOLD, 'igs_score']

    x = np.linspace(0, 100, 300)
    kde_all = gaussian_kde(scores_all)(x)

    ax_stats.fill_between(
        x, kde_all, alpha=0.15, color='#58a6ff',
    )
    ax_stats.plot(x, kde_all, color='#58a6ff', lw=1.5, label='All counties')

    # Shade below threshold in red
    mask = x < THRESHOLD
    ax_stats.fill_between(
        x[mask], kde_all[mask], alpha=0.45, color=ACCENT_RED,
        label=f'Vulnerable (IGS<{THRESHOLD})',
    )
    ax_stats.axvline(THRESHOLD, color=ACCENT_GOLD, lw=2, ls='--', zorder=5)
    ax_stats.axvline(scores_all.mean(), color='#58a6ff', lw=1.5, ls=':', zorder=5)

    ax_stats.set_xlim(0, 100)
    ax_stats.set_xlabel('IGS Score', color=TEXT_COLOR, fontsize=10)
    ax_stats.set_ylabel('Density', color=TEXT_COLOR, fontsize=10)
    ax_stats.tick_params(colors=TEXT_COLOR, labelsize=9)
    for sp in ax_stats.spines.values():
        sp.set_edgecolor('#30363d')
    ax_stats.set_title('County IGS Distribution', color=TEXT_COLOR,
                        fontsize=11, fontweight='bold', pad=6)

    # Threshold + mean labels
    ax_stats.text(THRESHOLD + 1, ax_stats.get_ylim()[1] * 0.9,
                  f'{THRESHOLD}', color=ACCENT_GOLD, fontsize=9, fontweight='bold')
    ax_stats.text(scores_all.mean() + 1, ax_stats.get_ylim()[1] * 0.75,
                  f'Mean\n{scores_all.mean():.1f}', color='#58a6ff', fontsize=8.5)

    legend = ax_stats.legend(
        fontsize=8.5, loc='upper left',
        facecolor=BG_COLOR, edgecolor='#30363d', labelcolor=TEXT_COLOR,
    )

    # ── Big KPI numbers ───────────────────────────────────────────────────────
    n_below = int((igs['igs_score'] < THRESHOLD).sum())
    n_total = len(igs)
    pct_below = n_below / n_total * 100
    nat_mean  = igs['igs_score'].mean()

    kpi_ax = fig.add_axes([0.73, 0.07, 0.25, 0.44])
    kpi_ax.set_facecolor(PANEL_COLOR)
    kpi_ax.axis('off')
    for sp in kpi_ax.spines.values():
        sp.set_edgecolor('#30363d')

    kpis = [
        ('3,221',       'US Counties Analyzed',   '#58a6ff'),
        (f'{n_below:,}', f'Counties Below IGS {THRESHOLD}', ACCENT_RED),
        (f'{pct_below:.1f}%', 'Economically Vulnerable', '#e67e22'),
        (f'{nat_mean:.1f}',   'National Mean IGS',       '#27ae60'),
        ('37.9',        'MS Delta Mean IGS',       ACCENT_GOLD),
        ('−11.8 pts',   'Delta Gap vs. National',  ACCENT_GOLD),
    ]

    for i, (val, lbl, clr) in enumerate(kpis):
        y = 0.90 - i * 0.155
        kpi_ax.text(0.06, y, val, transform=kpi_ax.transAxes,
                    fontsize=21, fontweight='black', color=clr, va='center')
        kpi_ax.text(0.06, y - 0.045, lbl, transform=kpi_ax.transAxes,
                    fontsize=9, color='#8b949e', va='center')

    # Divider line between KPIs
    for i in range(1, len(kpis)):
        y_line = 0.90 - i * 0.155 + 0.075
        kpi_ax.axhline(y=y_line, color='#21262d', lw=0.8,
                       xmin=0.04, xmax=0.96)

    kpi_ax.set_title('Key Statistics', color=TEXT_COLOR,
                     fontsize=11, fontweight='bold', pad=0,
                     loc='left', x=0.06)

    # ── Footer ────────────────────────────────────────────────────────────────
    fig.text(
        0.5, 0.012,
        'Source: Mastercard IGS 2025 · US Census Bureau TIGER 2022 · '
        'Jackson State University Data Challenge 2026',
        ha='center', fontsize=8.5, color='#484f58',
    )

    out_path = OUT_DIR / 'slide3_national_map.png'
    plt.savefig(out_path, dpi=300, bbox_inches='tight',
                facecolor=BG_COLOR, edgecolor='none')
    plt.close()
    print(f'\n  Saved → {out_path}')
    return out_path


if __name__ == '__main__':
    print('=' * 60)
    print('  Slide 3 — National IGS County Map')
    print('=' * 60)
    print('\n  Loading county geodata...')
    gdf = load_county_geodata()
    print(f'  Loaded {len(gdf):,} counties (CONUS)')
    print('  Loading IGS national data...')
    igs = load_igs_data()
    print(f'  Loaded {len(igs):,} county IGS records')
    print('  Rendering figure...')
    make_figure(gdf, igs)
    print('\n  Done.')
