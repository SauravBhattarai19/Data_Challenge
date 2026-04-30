"""
Conclusion Slide — 9-County Consensus Priority Matrix
======================================================
A 9-county × 10-feature heat map showing which investment priorities land
in the ACT NOW quadrant across every MS Delta county, revealing:

  • 5 features are ACT NOW for ALL 9 counties — the universal prescription
  • Commercial Diversity is the one differentiator (5/9 vs 4/9) — and the
    split perfectly tracks distress: most-distressed counties lack it,
    least-distressed counties are already at benchmark
  • 3 features are Phase 2 (SECOND WAVE) universally — real gaps, next step

Data sources:
  data_processed/expanded_model.parquet
  data_processed/expanded_shap_summary.parquet
  03_analysis/expanded_priority_matrix.py

Output: presentation/figures/slide_conclusion_consensus_matrix.png (300 DPI)
Run:    python presentation/fig_conclusion_consensus_matrix.py
"""

import sys, io, contextlib, importlib.util
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.lines import Line2D

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
PROCESSED = PROJECT_ROOT / 'data_processed'
OUT_DIR   = PROJECT_ROOT / 'presentation' / 'figures'
OUT_DIR.mkdir(parents=True, exist_ok=True)

_spec = importlib.util.spec_from_file_location(
    'expanded_priority_matrix',
    PROJECT_ROOT / '03_analysis' / 'expanded_priority_matrix.py'
)
_mod = importlib.util.module_from_spec(_spec)
with contextlib.redirect_stdout(io.StringIO()):
    _spec.loader.exec_module(_mod)

compute_gaps    = _mod.compute_gaps
get_ml_features = _mod.get_ml_features
CATEGORY_COLORS = _mod.CATEGORY_COLORS

from config import DELTA_COUNTY_FIPS, DELTA_COUNTY_NAMES

# ── Constants ──────────────────────────────────────────────────────────────────
SHAP_THRESH = 3.0
GAP_THRESH  = 20

# Feature order: SHAP-ranked, grouped for narrative clarity
FEAT_ORDER = [
    'DENTAL_CrudePrev',
    'Commercial Diversity Score_2017',
    'LPA_CrudePrev',
    'MOBILITY_CrudePrev',
    'DIABETES_CrudePrev',
    'Internet Access Score_2017',
    'Labor Market Engagement Index Score_2017',
    # ── Phase 2 below ────────────────────────────────
    'SLEEP_CrudePrev',
    'STROKE_CrudePrev',
    'CSMOKING_CrudePrev',
]
PHASE_BREAK = 7   # separator between index 6 and 7

Q_COLORS = {
    'ACT NOW':     '#dc2626',
    'PROTECT':     '#059669',
    'SECOND WAVE': '#f59e0b',
    'MONITOR':     '#d1d5db',
}
BG     = '#ffffff'
TEXT_C = '#1a1a2e'
DIM_C  = '#6b7280'
BORDER = '#e5e7eb'

plt.rcParams.update({'font.family': 'DejaVu Sans',
                     'figure.facecolor': BG, 'text.color': TEXT_C})


# ── Data loading ───────────────────────────────────────────────────────────────
def load_all_data():
    df           = pd.read_parquet(PROCESSED / 'expanded_model.parquet')
    shap_summary = pd.read_parquet(PROCESSED / 'expanded_shap_summary.parquet')

    df['county_fips5'] = df['GEOID'].str[:5]

    with contextlib.redirect_stdout(io.StringIO()):
        features = get_ml_features(df)

    # Per-county results
    results   = {}   # fips → {feature: quadrant}
    for fips in DELTA_COUNTY_FIPS:
        with contextlib.redirect_stdout(io.StringIO()):
            gdf, _ = compute_gaps(df, features, shap_summary, fips)
        gdf['quadrant'] = gdf.apply(
            lambda r: ('ACT NOW'     if r['shap_pct'] >= SHAP_THRESH and r['gap'] >= GAP_THRESH else
                       'PROTECT'     if r['shap_pct'] >= SHAP_THRESH and r['gap'] <  GAP_THRESH else
                       'SECOND WAVE' if r['shap_pct'] <  SHAP_THRESH and r['gap'] >= GAP_THRESH else
                       'MONITOR'), axis=1)
        results[fips] = gdf.set_index('feature')['quadrant'].to_dict()

    # Delta-wide for labels + bottom boxes
    with contextlib.redirect_stdout(io.StringIO()):
        dgdf, _ = compute_gaps(df, features, shap_summary, 'delta')
    delta_raw = dgdf.set_index('feature')['target_raw'].to_dict()
    turn_raw  = dgdf.set_index('feature')['turn_raw'].to_dict()
    shap_lkp  = dgdf.set_index('feature')['pct_total' if 'pct_total' in dgdf.columns else 'shap_pct'].to_dict()
    cat_lkp   = dgdf.set_index('feature')['category'].to_dict()
    label_lkp = dgdf.set_index('feature')['label'].to_dict()

    # Fallback: get shap_pct from shap_summary if not in dgdf
    if not shap_lkp or all(v == 0 for v in shap_lkp.values()):
        shap_lkp = shap_summary.set_index('feature')['pct_total'].to_dict()

    # County order: most distressed (lowest IGS) → least distressed
    county_mean_igs = (df[df['county_fips5'].isin(DELTA_COUNTY_FIPS)]
                       .groupby('county_fips5')['igs_score_2017'].mean()
                       .sort_values())
    county_order = county_mean_igs.index.tolist()

    return dict(results=results, delta_raw=delta_raw, turn_raw=turn_raw,
                shap_lkp=shap_lkp, cat_lkp=cat_lkp, label_lkp=label_lkp,
                county_order=county_order)


# ── Figure ─────────────────────────────────────────────────────────────────────
def make_figure(data):
    results      = data['results']
    delta_raw    = data['delta_raw']
    turn_raw     = data['turn_raw']
    shap_lkp     = data['shap_lkp']
    cat_lkp      = data['cat_lkp']
    label_lkp    = data['label_lkp']
    county_order = data['county_order']

    n_feats    = len(FEAT_ORDER)
    n_counties = len(county_order)

    fig = plt.figure(figsize=(15, 10), facecolor=BG)

    # ── Axes ──────────────────────────────────────────────────────────────────
    # SHAP bars + feature labels
    ax_s = fig.add_axes([0.07, 0.21, 0.17, 0.62])
    # Heat map
    ax_h = fig.add_axes([0.25, 0.21, 0.51, 0.62])
    # Right: X/9 counts + group annotations
    ax_r = fig.add_axes([0.77, 0.21, 0.22, 0.62])
    # Bottom boxes
    ax_b1 = fig.add_axes([0.04, 0.02, 0.29, 0.155])
    ax_b2 = fig.add_axes([0.355, 0.02, 0.29, 0.155])
    ax_b3 = fig.add_axes([0.67, 0.02, 0.295, 0.155])

    for ax in [ax_s, ax_h, ax_r, ax_b1, ax_b2, ax_b3]:
        ax.set_facecolor(BG)

    # ── Title ─────────────────────────────────────────────────────────────────
    fig.text(0.5, 0.972,
             'Two Root Causes. Universal Evidence.',
             ha='center', va='top', fontsize=20, fontweight='bold', color=TEXT_C)
    fig.text(0.5, 0.940,
             'Investment Priority Matrix — top 10 predictors across all 9 MS Delta counties  '
             '(columns sorted: most distressed → least distressed)',
             ha='center', va='top', fontsize=11.5, color=DIM_C)

    # ── Axis helpers ──────────────────────────────────────────────────────────
    y_pos = np.arange(n_feats)

    # ── Left: SHAP bar + feature labels ───────────────────────────────────────
    shap_vals       = [shap_lkp.get(f, 0)                                 for f in FEAT_ORDER]
    feat_cats       = [cat_lkp.get(f, 'Other')                            for f in FEAT_ORDER]
    feat_labels     = [label_lkp.get(f, f)                                for f in FEAT_ORDER]
    feat_colors     = [CATEGORY_COLORS.get(c, '#888888')                   for c in feat_cats]

    # Subtle group background bands (behind bars)
    group_bands = [
        (-0.5, 0.5,  '#fff7ed'),   # row 0  — Healthcare Access
        (0.5,  1.5,  '#f0fdf4'),   # row 1  — Commercial Diversity (differentiator)
        (1.5,  4.5,  '#fff1f2'),   # rows 2-4 — Chronic Disease Burden
        (4.5,  6.5,  '#eff6ff'),   # rows 5-6 — Economic Connectivity
        (6.5,  9.5,  '#fffbeb'),   # rows 7-9 — Phase 2 Priorities
    ]
    for y0, y1, bg_c in group_bands:
        ax_s.axhspan(y0, y1, facecolor=bg_c, alpha=0.7, zorder=0)
        ax_h.axhspan(y0, y1, facecolor=bg_c, alpha=0.35, zorder=0)

    ax_s.barh(y_pos, shap_vals, color=feat_colors, alpha=0.88,
              height=0.65, zorder=2)

    # SHAP % labels at bar ends
    for yi, sv in zip(y_pos, shap_vals):
        ax_s.text(sv + 0.08, yi, f'{sv:.1f}%',
                  va='center', ha='left', fontsize=8.5,
                  color=DIM_C, fontweight='bold', zorder=3)

    ax_s.set_yticks(y_pos)
    ax_s.set_yticklabels(feat_labels, fontsize=10.8, fontweight='bold')
    for tick, col in zip(ax_s.get_yticklabels(), feat_colors):
        tick.set_color(col)

    ax_s.invert_xaxis()
    ax_s.invert_yaxis()
    max_shap = max(shap_vals)
    ax_s.set_xlim(max_shap * 1.55, 0)
    ax_s.set_ylim(n_feats - 0.5, -0.5)
    ax_s.tick_params(axis='x', labelsize=8, colors=DIM_C, pad=2)
    ax_s.tick_params(axis='y', left=False, pad=5)
    ax_s.set_xlabel('SHAP importance (%)', fontsize=8.5, color=DIM_C, labelpad=3)

    # Phase separator in ax_s
    ax_s.axhline(PHASE_BREAK - 0.5, color='#374151', lw=1.5, ls='--', zorder=4)

    for sp in ax_s.spines.values():
        sp.set_visible(False)
    ax_s.axvline(0, color=BORDER, lw=0.8, zorder=1)

    # ── Heat map ───────────────────────────────────────────────────────────────
    for row_i, feat in enumerate(FEAT_ORDER):
        for col_j, fips in enumerate(county_order):
            q     = results.get(fips, {}).get(feat, 'MONITOR')
            color = Q_COLORS[q]
            ax_h.add_patch(Rectangle(
                (col_j - 0.5, row_i - 0.5), 1, 1,
                facecolor=color, zorder=2))
            ax_h.add_patch(Rectangle(
                (col_j - 0.5, row_i - 0.5), 1, 1,
                fill=False, edgecolor='white', linewidth=2.2, zorder=3))

    # County names: short, rotated, on TOP of axes
    short_names = [DELTA_COUNTY_NAMES.get(f, f).replace(' County', '') for f in county_order]
    ax_h.set_xticks(range(n_counties))
    ax_h.set_xticklabels(short_names, rotation=38, ha='left',
                          fontsize=11, fontweight='bold', color=TEXT_C)
    ax_h.xaxis.set_label_position('top')
    ax_h.xaxis.tick_top()
    ax_h.tick_params(axis='x', top=True, bottom=False, pad=3)

    ax_h.set_yticks([])
    ax_h.set_xlim(-0.5, n_counties - 0.5)
    ax_h.set_ylim(n_feats - 0.5, -0.5)

    # Phase separator line
    ax_h.axhline(PHASE_BREAK - 0.5, color='#374151', lw=2.0, ls='--', zorder=5)

    # Directional label above county axis
    ax_h.text(-0.48, -0.85, '← most distressed',
              ha='left', va='bottom', fontsize=8.5, color='#b91c1c',
              style='italic', fontweight='bold', clip_on=False)
    ax_h.text(n_counties - 0.52, -0.85, 'least distressed →',
              ha='right', va='bottom', fontsize=8.5, color='#15803d',
              style='italic', fontweight='bold', clip_on=False)

    # ── Dental callout ────────────────────────────────────────────────────────
    # Placed just below the Dental row (y=0.85), arrow points up to row 0
    dental_delta = delta_raw.get('DENTAL_CrudePrev', 6.1)
    dental_turn  = turn_raw.get('DENTAL_CrudePrev', 56.9)
    ax_h.annotate(
        f'Delta: {dental_delta:.1f}%   →   Turnaround: {dental_turn:.1f}%',
        xy=(n_counties / 2 - 0.5, 0),
        xytext=(n_counties / 2 - 0.5, 0.78),
        ha='center', va='top', fontsize=9.5,
        color='#991b1b', fontweight='bold',
        arrowprops=dict(arrowstyle='->', color='#991b1b', lw=1.6,
                        shrinkA=4, shrinkB=4),
        bbox=dict(boxstyle='round,pad=0.35', facecolor='#fef2f2',
                  edgecolor='#dc2626', linewidth=1.4),
        zorder=10,
    )

    # ── Commercial Diversity "DIFFERENTIATOR" tag ─────────────────────────────
    # Row 1 — put a small tag on the left of row 1 in the heat map
    ax_h.text(-0.48, 1,
              '★ DIFFERENTIATOR',
              va='center', ha='right', fontsize=8.5,
              color='#059669', fontweight='bold',
              path_effects=[pe.withStroke(linewidth=2.5, foreground='white')],
              clip_on=False)

    for sp in ax_h.spines.values():
        sp.set_visible(False)

    # ── Right axes: X/9 counts + group labels ──────────────────────────────────
    ax_r.set_xlim(0, 10)
    ax_r.set_ylim(n_feats - 0.5, -0.5)
    ax_r.set_xticks([])
    ax_r.set_yticks([])
    for sp in ax_r.spines.values():
        sp.set_visible(False)

    # Reproduce group bands
    for y0, y1, bg_c in group_bands:
        ax_r.axhspan(y0, y1, facecolor=bg_c, alpha=0.55, zorder=0)

    ax_r.axhline(PHASE_BREAK - 0.5, color='#374151', lw=1.5, ls='--', zorder=4)

    for row_i, feat in enumerate(FEAT_ORDER):
        qs        = [results.get(fips, {}).get(feat, 'MONITOR') for fips in county_order]
        n_act     = qs.count('ACT NOW')
        n_prot    = qs.count('PROTECT')
        n_sw      = qs.count('SECOND WAVE')

        if n_act == 9:
            badge_txt = '9/9'
            badge_col = '#dc2626'
            tag_txt   = 'UNIVERSAL'
            tag_col   = '#dc2626'
            tag_fw    = 'black'
        elif n_act == 8:
            badge_txt = '8/9'
            badge_col = '#dc2626'
            tag_txt   = '1 county at benchmark'
            tag_col   = '#6b7280'
            tag_fw    = 'normal'
        elif n_act == 5 and n_prot == 4:
            badge_txt = '5/9'
            badge_col = '#059669'
            tag_txt   = '4 counties PROTECT'
            tag_col   = '#059669'
            tag_fw    = 'bold'
        elif n_sw == 9:
            badge_txt = '9/9'
            badge_col = '#f59e0b'
            tag_txt   = 'PHASE 2'
            tag_col   = '#f59e0b'
            tag_fw    = 'bold'
        else:
            badge_txt = f'{n_act}/9'
            badge_col = DIM_C
            tag_txt   = ''
            tag_col   = DIM_C
            tag_fw    = 'normal'

        # Badge
        ax_r.text(0.6, row_i, badge_txt,
                  va='center', ha='center', fontsize=14,
                  color=badge_col, fontweight='black', zorder=3)
        # Tag
        if tag_txt:
            ax_r.text(1.5, row_i, tag_txt,
                      va='center', ha='left', fontsize=9,
                      color=tag_col, fontweight=tag_fw, zorder=3)

    # Group bracket labels (far right)
    bracket_defs = [
        (0,   0,   'Healthcare\nAccess',        '#e67e22'),
        (1,   1,   '← Differentiator',          '#059669'),
        (2,   4,   'Chronic Disease\nBurden',    '#c0392b'),
        (5,   6,   'Economic\nConnectivity',     '#2563eb'),
        (7,   9,   'Phase 2\nPriorities',        '#d97706'),
    ]
    BX = 6.6   # x-position of bracket
    for y_s, y_e, txt, col in bracket_defs:
        y_mid = (y_s + y_e) / 2
        # Bracket lines
        ax_r.plot([BX, BX], [y_s - 0.3, y_e + 0.3],
                  color=col, lw=2.2, solid_capstyle='round', clip_on=False)
        ax_r.plot([BX, BX + 0.25], [y_s - 0.3, y_s - 0.3],
                  color=col, lw=2.2, clip_on=False)
        ax_r.plot([BX, BX + 0.25], [y_e + 0.3, y_e + 0.3],
                  color=col, lw=2.2, clip_on=False)
        ax_r.text(BX + 0.45, y_mid, txt,
                  va='center', ha='left', fontsize=8.8,
                  color=col, fontweight='bold', clip_on=False,
                  linespacing=1.2)

    # ── Bottom boxes ──────────────────────────────────────────────────────────
    _box(ax_b1,
         icon='① Preventive Healthcare',
         icon_col='#ea580c',
         bg='#fff7ed',
         spine_col='#ea580c',
         stats_line=f"ALL 9 counties  ·  5 universal features",
         stat_col='#9a3412',
         detail_lines=[
             f"Delta dental rate:  {delta_raw.get('DENTAL_CrudePrev', 6.1):.1f}%  "
             f"→  Turnaround: {turn_raw.get('DENTAL_CrudePrev', 56.9):.1f}%",
             f"Delta inactivity:   {delta_raw.get('LPA_CrudePrev', 44.5):.1f}%  "
             f"→  Turnaround: {turn_raw.get('LPA_CrudePrev', 28.5):.1f}%",
         ],
         action_lines=[
             '▶  FQHCs  ·  Mobile dental units',
             '▶  Medicaid navigation programs',
         ])

    _box(ax_b2,
         icon='② Economic Connectivity',
         icon_col='#2563eb',
         bg='#eff6ff',
         spine_col='#2563eb',
         stats_line='8 – 9 of 9 counties  ·  2 universal features',
         stat_col='#1e40af',
         detail_lines=[
             f"Internet Access:   {delta_raw.get('Internet Access Score_2017', 10.5):.1f}"
             f"  →  {turn_raw.get('Internet Access Score_2017', 35.4):.1f}",
             f"Labor Market Eng.: {delta_raw.get('Labor Market Engagement Index Score_2017', 8.3):.1f}"
             f"  →  {turn_raw.get('Labor Market Engagement Index Score_2017', 32.6):.1f}",
         ],
         action_lines=[
             '▶  Rural broadband infrastructure',
             '▶  Workforce re-engagement programs',
         ])

    _box(ax_b3,
         icon='③ Commercial Ecosystem',
         icon_col='#b45309',
         bg='#fffbeb',
         spine_col='#b45309',
         stats_line='5/9 ACT NOW  ·  4/9 already at benchmark  ★',
         stat_col='#92400e',
         detail_lines=[
             f"Act Now counties:  {delta_raw.get('Commercial Diversity Score_2017', 27.4):.1f}"
             f"  →  {turn_raw.get('Commercial Diversity Score_2017', 42.3):.1f}",
             '4 counties PROTECT — most distressed need it most',
         ],
         action_lines=[
             '▶  Business incubators  ·  SBA lending',
             '▶  Mixed-use development incentives',
         ])

    # ── Legend ────────────────────────────────────────────────────────────────
    legend_handles = [
        mpatches.Patch(color=Q_COLORS['ACT NOW'],     label='ACT NOW — high importance + large gap'),
        mpatches.Patch(color=Q_COLORS['PROTECT'],     label='PROTECT — high importance + at/near benchmark'),
        mpatches.Patch(color=Q_COLORS['SECOND WAVE'], label='SECOND WAVE — lower importance + large gap'),
    ]
    fig.legend(handles=legend_handles, fontsize=9.5,
               loc='lower center', bbox_to_anchor=(0.5, 0.19),
               ncol=3, columnspacing=1.2, handlelength=1.4, handletextpad=0.6,
               facecolor=BG, edgecolor=BORDER, framealpha=0.95, borderpad=0.5)

    # ── URL footer ────────────────────────────────────────────────────────────
    fig.text(0.5, 0.004,
             "Explore your county's Priority Matrix in real time:  "
             "howtoimprove.streamlit.app",
             ha='center', va='bottom', fontsize=10.5,
             color='#2563eb', fontweight='bold')

    # ── Save ──────────────────────────────────────────────────────────────────
    out = OUT_DIR / 'slide_conclusion_consensus_matrix.png'
    plt.savefig(out, dpi=300, bbox_inches='tight', facecolor=BG, edgecolor='none')
    plt.close()
    print(f'  Saved → {out}')


def _box(ax, icon, icon_col, bg, spine_col, stats_line, stat_col,
         detail_lines, action_lines):
    """Draw a styled intervention callout box."""
    ax.set_facecolor(bg)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    for name, sp in ax.spines.items():
        if name == 'left':
            sp.set_color(spine_col)
            sp.set_linewidth(4.5)
            sp.set_visible(True)
        else:
            sp.set_visible(False)

    PAD = 0.06
    y = 0.91

    # Icon / title
    ax.text(PAD, y, icon, transform=ax.transAxes,
            fontsize=11.5, fontweight='bold', color=icon_col, va='top')
    y -= 0.19

    # Stats line
    ax.text(PAD, y, stats_line, transform=ax.transAxes,
            fontsize=9.5, fontweight='bold', color=stat_col, va='top')
    y -= 0.18

    # Detail lines (gap data)
    for line in detail_lines:
        ax.text(PAD, y, line, transform=ax.transAxes,
                fontsize=8.8, color='#374151', va='top',
                fontfamily='monospace')
        y -= 0.15

    y -= 0.03

    # Action lines
    for line in action_lines:
        ax.text(PAD, y, line, transform=ax.transAxes,
                fontsize=9, color='#374151', va='top', fontweight='bold')
        y -= 0.155


# ── CLI ────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print('=' * 65)
    print('  Conclusion Slide — 9-County Consensus Heat Map')
    print('=' * 65)

    data = load_all_data()

    # Print summary table
    print(f"\n  {'Feature':<42} {'ACT':>4} {'PROT':>5} {'2ND':>4}  Badge")
    print('  ' + '-' * 72)
    for feat in FEAT_ORDER:
        qs    = [data['results'].get(f, {}).get(feat, 'MONITOR')
                 for f in data['county_order']]
        n_act = qs.count('ACT NOW')
        n_p   = qs.count('PROTECT')
        n_sw  = qs.count('SECOND WAVE')
        lbl   = data['label_lkp'].get(feat, feat)[:40]
        badge = ('UNIVERSAL' if n_act == 9 else
                 f'{n_act}/9 ACT + {n_p}/9 PROT' if n_p else
                 'PHASE 2 ALL'  if n_sw == 9 else
                 f'mixed')
        print(f"  {lbl:<42} {n_act:>4} {n_p:>5} {n_sw:>4}  {badge}")

    print()
    make_figure(data)
    print('  Done.')
