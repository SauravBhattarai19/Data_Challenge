"""
Slide 10 (Quitman) — SHAP × Gap Priority Matrix for Quitman County
===================================================================
Quitman-only matrix: same layout philosophy as fig_slide10_priority_matrix —
panel only (no slide headline, footer, or side callouts), labels separated with
adjustText.

Plots non-model + IGS sub-indicators on one scatter:
  Y: SHAP %   X: gap from turnaround (normalized / IGS pts)

Output: presentation/figures/slide10_quitman_priority.png  (300 DPI, 12×6.75)
Run:  python presentation/fig_slide10_quitman_priority.py
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
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from adjustText import adjust_text

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
PROCESSED = PROJECT_ROOT / 'data_processed'
OUT_DIR   = PROJECT_ROOT / 'presentation' / 'figures'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Design ────────────────────────────────────────────────────────────────────
BG_COLOR   = '#ffffff'
TEXT_COLOR = '#1a1a2e'
DIM_COLOR  = '#555566'
BORDER_CLR = '#cccccc'
QUITMAN    = '28119'

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'figure.facecolor': BG_COLOR,
    'text.color': TEXT_COLOR,
    'axes.facecolor': BG_COLOR,
    'axes.edgecolor': BORDER_CLR,
})

SHAP_THRESH = 3.5
GAP_THRESH  = 28
# x-axis starts at −15; true gaps more negative (e.g. MWOB ahead of benchmark) would
# sit off-canvas — pin to this x for plotting only (matches IGS “above target” = −10).
GAP_PLOT_FLOOR = -10

DOMAIN_COLORS = {
    'Health':       '#c0392b',
    'Business':     '#6B4226',
    'Commercial':   '#c8830a',
    'Commute':      '#34495e',
    'IGS':          '#b8860b',
    'Climate':      '#2c5282',
    'Connectivity': '#2980b9',
}


def load_data():
    model_df = pd.read_parquet(PROCESSED / 'igs_improvement_model.parquet')
    model_df['county_fips5'] = model_df['GEOID'].str[:5]
    shap_df  = pd.read_parquet(PROCESSED / 'shap_feature_summary.parquet')
    bench    = pd.read_parquet(PROCESSED / 'turnaround_benchmarks.parquet')
    delta_p  = pd.read_parquet(PROCESSED / 'delta_profile.parquet')

    turn     = model_df[model_df['turnaround'] == 1]
    quitman  = model_df[model_df['county_fips5'] == QUITMAN]
    quit_dp  = delta_p[delta_p['county_fips5'] == QUITMAN]

    shap_lkp = shap_df.set_index('feature')['pct_total'].to_dict()
    turn_tgt = (bench[bench['typology'] == 'Turnaround']
                .set_index('indicator')['mean_2025']
                .to_dict())

    rows = []

    non_igs = [
        ('biz_food_retail',    'Food Retail Businesses', 'Business', True),
        ('biz_pharmacy',       'Pharmacy Businesses',    'Business', True),
        ('biz_sector_diversity', 'Sector Diversity (NAICS)', 'Business', True),
        ('Minority/Women Owned Businesses Score_2017', 'MWOB Score', 'Commercial', True),
        ('Travel Time to Work Score_2017', 'Travel Time to Work', 'Commute', True),
        ('LPA_CrudePrev',      'Physical Inactivity',    'Health',    False),
        ('MOBILITY_CrudePrev', 'Mobility Disability',    'Health',    False),
        ('DIABETES_CrudePrev', 'Diabetes Rate',          'Health',    False),
        ('BUILDVALUE',         'NRI Build Exposure ($)', 'Climate',   True),
        ('igs_economy_2017',   'IGS Economy (2017)',     'IGS',       True),
    ]
    for col, label, domain, hg in non_igs:
        if col not in model_df.columns or quitman.empty:
            continue
        p05 = model_df[col].quantile(0.05)
        p95 = model_df[col].quantile(0.95)
        rng = p95 - p05 if p95 != p05 else 1.0

        def norm(v, hg=hg, p05=p05, rng=rng):
            n = (v - p05) / rng * 100
            n = max(0.0, min(100.0, n))
            return n if hg else 100.0 - n

        t_n = norm(turn[col].mean())
        q_n = norm(quitman[col].mean())
        gap = t_n - q_n
        gap_plot = gap if gap >= GAP_PLOT_FLOOR else GAP_PLOT_FLOOR

        rows.append({
            'col':        col,
            'label':      label,
            'domain':     domain,
            'color':      DOMAIN_COLORS[domain],
            'shap_pct':   shap_lkp.get(col, 0),
            'gap_norm':   gap,
            'gap_plot':   gap_plot,
            't_raw':      turn[col].mean(),
            'q_raw':      quitman[col].mean(),
            'type':       'health/social',
            'marker':     'o',
        })

    igs_with_shap = [
        ('Commercial Diversity Score', 'Commercial Diversity', 'Commercial',   'D'),
        ('Internet Access Score',      'Internet Access',      'Connectivity', 'D'),
        ('Labor Market Engagement Index Score', 'Labor Market','Connectivity', 'D'),
        ('Female Above Poverty Score', 'Female Above Poverty', 'IGS',          'D'),
        ('Spend Growth Score',         'Spend Growth',         'IGS',          'D'),
    ]
    for sub_base, label, domain, marker in igs_with_shap:
        if sub_base not in quit_dp.columns:
            continue
        q_val = quit_dp[sub_base].mean()
        t_val = turn_tgt.get(sub_base, float('nan'))
        if np.isnan(t_val):
            continue
        gap_raw = t_val - q_val
        gap_norm = min(100, max(0, abs(gap_raw)))
        if q_val > t_val:
            gap_norm = -10

        shap_key = f'{sub_base}_2017'
        shap_pct = shap_lkp.get(shap_key, 0)

        rows.append({
            'col':      sub_base,
            'label':    label,
            'domain':   domain,
            'color':    DOMAIN_COLORS[domain],
            'shap_pct': shap_pct,
            'gap_norm': gap_norm,
            'gap_plot': gap_norm,
            't_raw':    t_val,
            'q_raw':    q_val,
            'type':     'IGS',
            'marker':   marker,
        })

    return pd.DataFrame(rows)


def _val_pair(row):
    unit = ' pts' if row['type'] == 'IGS' else ''
    if row.get('col') == 'BUILDVALUE':
        pair = f"{row['q_raw']/1e6:.0f}M vs {row['t_raw']/1e6:.0f}M $"
    elif str(row.get('col', '')).startswith('biz_'):
        pair = f"{row['q_raw']:,.0f} vs {row['t_raw']:,.0f}"
    else:
        pair = f"{row['q_raw']:.1f} vs {row['t_raw']:.1f}{unit}"
    return pair


def make_figure(data: pd.DataFrame):
    fig = plt.figure(figsize=(8, 6), facecolor=BG_COLOR)
    ax = fig.add_axes([0.10, 0.12, 0.86, 0.80])
    ax.set_facecolor(BG_COLOR)

    plot_df = data.reset_index(drop=True)

    quads = {
        (True,  True):  ('#fde8e8', '#c0392b', 'ACT NOW\n(Root Cause + Score Lever)'),
        (True,  False): ('#e8f8e8', '#1e8449', 'PROTECT\n(Existing Strength)'),
        (False, True):  ('#fff3e0', '#e67e22', 'SECOND WAVE\n(Address After Root Cause)'),
        (False, False): ('#f5f5f5', '#888888', 'MONITOR'),
    }
    for (hs, lg), (bg_c, _, _) in quads.items():
        x0 = GAP_THRESH if lg else -15
        x1 = 105        if lg else GAP_THRESH
        y0 = SHAP_THRESH if hs else 0
        y1 = 12         if hs else SHAP_THRESH
        ax.fill_between([x0, x1], [y0, y0], [y1, y1],
                        color=bg_c, alpha=0.55, zorder=0)

    ax.axhline(SHAP_THRESH, color='#999', lw=1.8, ls='--', zorder=1)
    ax.axvline(GAP_THRESH,  color='#999', lw=1.8, ls='--', zorder=1)

    quad_labels = {
        (True,  True):  (GAP_THRESH + 1,  SHAP_THRESH + 0.25, '#c0392b', 11.5),
        (True,  False): (-12,              SHAP_THRESH + 0.25, '#1e8449', 11.5),
        (False, True):  (GAP_THRESH + 1,  0.2,                 '#e67e22', 10.5),
        (False, False): (-12,              0.2,                 '#888888', 10.5),
    }
    for (hs, lg), (xt, yt, tc, fs) in quad_labels.items():
        _, _, txt = quads[(hs, lg)]
        ax.text(xt, yt, txt, fontsize=fs, color=tc, fontweight='bold',
                va='bottom', ha='left', zorder=2, alpha=0.7,
                path_effects=[pe.withStroke(linewidth=2, foreground='white')])

    label_font = 9.0
    texts = []
    for _, row in plot_df.iterrows():
        mk = row['marker']
        sz = 340 if mk == 'D' else 300
        ax.scatter(
            row['gap_plot'], row['shap_pct'],
            s=sz, color=row['color'], marker=mk, zorder=5,
            edgecolors='white', linewidths=1.8,
        )
        pair = _val_pair(row)
        t = ax.text(
            row['gap_plot'], row['shap_pct'],
            f"{row['label']}\n({pair})",
            fontsize=label_font, color=row['color'], fontweight='bold',
            ha='center', va='center', zorder=6,
            path_effects=[pe.withStroke(linewidth=2, foreground='white')],
        )
        texts.append(t)

    adjust_text(
        texts,
        x=plot_df['gap_plot'].values,
        y=plot_df['shap_pct'].values,
        ax=ax,
        expand_points=(1.45, 1.6),
        expand_text=(1.3, 1.45),
        force_points=(0.4, 0.55),
        force_text=(0.5, 0.7),
        lim=1000,
        arrowprops=dict(
            arrowstyle='-',
            color='#555555',
            lw=0.65,
            alpha=0.55,
            shrinkA=8,
            shrinkB=4,
        ),
    )

    ax.set_xlim(-15, 105)
    ax.set_ylim(0, 12)
    ax.set_xlabel(
        'Quitman gap from turnaround (normalized; higher = further behind)',
        fontsize=12, labelpad=8,
    )
    ax.set_ylabel('SHAP importance (% of total predictive power)', fontsize=12)
    ax.tick_params(labelsize=11)
    for sp in ax.spines.values():
        sp.set_edgecolor(BORDER_CLR)

    marker_handles = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#555555',
               markersize=9, label='Tract / county'),
        Line2D([0], [0], marker='D', color='w', markerfacecolor='#555555',
               markersize=8, label='IGS'),
    ]
    seen_dom = set()
    domain_handles = []
    for _, r in plot_df.iterrows():
        d = r['domain']
        if d not in seen_dom:
            seen_dom.add(d)
            domain_handles.append(
                mpatches.Patch(color=r['color'], label=d)
            )
    ax.legend(
        handles=marker_handles + domain_handles,
        fontsize=9, loc='upper left',
        facecolor=BG_COLOR, edgecolor=BORDER_CLR, framealpha=0.95,
        ncol=2, columnspacing=0.9,
    )

    out = OUT_DIR / 'slide10_quitman_priority.png'
    plt.savefig(out, dpi=300, bbox_inches='tight',
                facecolor=BG_COLOR, edgecolor='none')
    plt.close()
    print(f'  Saved → {out}')


if __name__ == '__main__':
    print('=' * 60)
    print('  Slide 10 — Quitman Priority Matrix')
    print('=' * 60)
    data = load_data()
    print(f'\n  {"Feature":<30} {"SHAP%":>6} {"Gap":>7}  {"Quadrant"}')
    print('  ' + '-' * 62)
    for _, r in data.sort_values('shap_pct', ascending=False).iterrows():
        hs = r['shap_pct'] >= SHAP_THRESH
        lg = r['gap_norm'] >= GAP_THRESH
        q = ('ACT NOW'     if hs and lg else
             'PROTECT'     if hs and not lg else
             'SECOND WAVE' if not hs and lg else 'MONITOR')
        print(f"  {r['label']:<30} {r['shap_pct']:>6.2f} {r['gap_norm']:>7.1f}  {q}")
    print()
    make_figure(data)
    print('  Done.')
