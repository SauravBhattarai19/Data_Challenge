"""
Slide 9 Figure — SHAP Framework: What Actually Drives Turnaround
================================================================
Manual conceptual grouping of SHAP feature importance by domain.
Each feature is assigned to ONE conceptual group; group totals are
the honest sum of individual mean |SHAP| values — not algorithmic clusters.
Single horizontal bar chart (wide left margin for domain labels).

Output: presentation/figures/slide9_shap_framework.png  (300 DPI, 16×9)
Run:  python presentation/fig_slide9_shap_framework.py
"""

import sys
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
PROCESSED = PROJECT_ROOT / 'data_processed'
OUT_DIR   = PROJECT_ROOT / 'presentation' / 'figures'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Design ────────────────────────────────────────────────────────────────────
BG_COLOR   = '#ffffff'
PANEL_BG   = '#f5f6f8'
BORDER_CLR = '#cccccc'
TEXT_COLOR = '#1a1a2e'
DIM_COLOR  = '#555566'

# One color per group — consistent with slide 7 where health = red
GROUP_COLORS = {
    'Health Burden\n& Engagement':       '#c0392b',   # deep red
    'Economic\nFoundation':              '#c8830a',   # gold
    'Social\nVulnerability':             '#8e44ad',   # purple
    'Climate &\nDisaster Exposure':      '#2980b9',   # steel blue
    'Connectivity\n& Labor Access':      '#1abc9c',   # teal
    'Healthcare\nProvider Supply':       '#27ae60',   # sage green
}

plt.rcParams.update({
    'font.family':      'DejaVu Sans',
    'figure.facecolor': BG_COLOR,
    'text.color':       TEXT_COLOR,
    'axes.labelcolor':  TEXT_COLOR,
    'xtick.color':      TEXT_COLOR,
    'ytick.color':      TEXT_COLOR,
    'axes.edgecolor':   BORDER_CLR,
    'axes.facecolor':   PANEL_BG,
})

# ── Manual conceptual grouping ────────────────────────────────────────────────
# Every feature assigned to exactly ONE group based on data source / meaning.
FEATURE_GROUPS = {
    # All CDC PLACES CrudePrev features → Health Burden & Engagement
    'Health Burden\n& Engagement': [
        'DENTAL_CrudePrev', 'LPA_CrudePrev', 'MOBILITY_CrudePrev',
        'DIABETES_CrudePrev', 'SLEEP_CrudePrev', 'STROKE_CrudePrev',
        'CSMOKING_CrudePrev', 'COLON_SCREEN_CrudePrev', 'MHLTH_CrudePrev',
        'OBESITY_CrudePrev', 'COPD_CrudePrev', 'BPHIGH_CrudePrev',
        'BINGE_CrudePrev', 'CHECKUP_CrudePrev', 'ACCESS2_CrudePrev',
        'MAMMOUSE_CrudePrev', 'CHD_CrudePrev', 'DEPRESSION_CrudePrev',
        'CASTHMA_CrudePrev',
    ],
    # IGS sub-indicators (2017 baselines)
    'Economic\nFoundation': [
        'Commercial Diversity Score_2017', 'igs_economy_2017',
        'Travel Time to Work Score_2017', 'Female Above Poverty Score_2017',
        'Personal Income Score_2017', 'Affordable Housing Score_2017',
        'Minority/Women Owned Businesses Score_2017',
        'Spending per Capita Score_2017', 'New Businesses Score_2017',
        'Small Business Loans Score_2017', 'Net Occupancy Score_2017',
        'igs_place_2017', 'igs_community_2017',
        'Health Insurance Coverage Score_2017', 'Spend Growth Score_2017',
    ],
    # SVI RPL_THEME features
    'Social\nVulnerability': [
        'RPL_THEME1', 'RPL_THEME2', 'RPL_THEME3', 'RPL_THEME4',
        'RPL_HVM', 'RPL_EBM',
    ],
    # FEMA NRI features
    'Climate &\nDisaster Exposure': [
        'BUILDVALUE', 'POPULATION', 'EAL_SCORE', 'RISK_SCORE',
        'IFLD_RISKS', 'TRND_RISKS', 'HWAV_RISKS', 'HRCN_RISKS', 'DRGT_RISKS',
    ],
    # Broadband, labor, poverty
    'Connectivity\n& Labor Access': [
        'Internet Access Score_2017', 'Labor Market Engagement Index Score_2017',
        'PovertyRate', 'LILATracts_1And10',
    ],
    # HPSA, FQHC, business infrastructure — the SUPPLY side
    'Healthcare\nProvider Supply': [
        'pc_hpsa_score_max', 'mh_hpsa_score_max', 'fqhc_count',
        'biz_food_retail', 'biz_pharmacy', 'biz_small_pct',
        'biz_total_all', 'biz_total_healthcare', 'biz_dental',
        'biz_home_health', 'biz_physician_office', 'biz_hospital',
        'biz_avg_emp_per_est', 'biz_sector_diversity',
        'biz_mental_health_svc', 'biz_small_under20', 'biz_total_emp',
    ],
}


def compute_groups(shap_df: pd.DataFrame) -> pd.DataFrame:
    shap_lookup = shap_df.set_index('feature')['mean_abs_shap'].to_dict()
    total_shap  = shap_df['mean_abs_shap'].sum()

    rows = []
    accounted = 0.0
    for group, features in FEATURE_GROUPS.items():
        group_shap = sum(shap_lookup.get(f, 0.0) for f in features)
        accounted += group_shap
        rows.append({
            'group':     group,
            'shap_sum':  group_shap,
            'pct':       group_shap / total_shap * 100,
            'n_features': sum(1 for f in features if f in shap_lookup),
        })

    df = pd.DataFrame(rows).sort_values('pct', ascending=False).reset_index(drop=True)
    print(f"  Accounted for: {accounted/total_shap*100:.1f}% of total SHAP mass")
    return df


def make_figure(groups: pd.DataFrame):
    fig = plt.figure(figsize=(8, 4.5), facecolor=BG_COLOR)

    # Single horizontal bar chart: wide left margin for long y-labels.
    ax_bar = fig.add_axes([0.26, 0.08, 0.69, 0.88])
    ax_bar.set_facecolor(PANEL_BG)

    bar_groups = groups.sort_values('pct', ascending=True)
    bar_labels = [g.replace('\n', ' ') for g in bar_groups['group']]
    bar_colors = [GROUP_COLORS[g] for g in bar_groups['group']]

    bars = ax_bar.barh(
        bar_labels, bar_groups['pct'],
        color=bar_colors, edgecolor=BORDER_CLR,
        linewidth=0.5, height=0.65,
    )
    for bar, pct in zip(bars, bar_groups['pct']):
        ax_bar.text(
            bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
            f'{pct:.1f}%', va='center', ha='left',
            fontsize=12, fontweight='black',
            color=bar.get_facecolor(),
            path_effects=[pe.withStroke(linewidth=2, foreground='white')],
        )

    ax_bar.set_xlim(0, 48)
    ax_bar.set_xlabel('% of Total Predictive Power (SHAP)', fontsize=12)
    ax_bar.tick_params(axis='y', labelsize=10.5)
    ax_bar.tick_params(axis='x', labelsize=11.5)
    ax_bar.set_title(
        'What Predicts Whether a Community Turns Around?\n'
        'Feature importance by conceptual domain (mean |SHAP|, grouped)',
        fontsize=14, fontweight='bold', pad=14, loc='left', color=TEXT_COLOR,
    )
    for sp in ax_bar.spines.values():
        sp.set_edgecolor(BORDER_CLR)

    out = OUT_DIR / 'slide9_shap_framework.png'
    plt.savefig(out, dpi=300, bbox_inches='tight',
                facecolor=BG_COLOR, edgecolor='none')
    plt.close()
    print(f'  Saved → {out}')


if __name__ == '__main__':
    print('=' * 60)
    print('  Slide 9 — SHAP Framework (manual conceptual grouping)')
    print('=' * 60)

    shap_df = pd.read_parquet(PROCESSED / 'shap_feature_summary.parquet')
    print(f'  Total features: {len(shap_df)}')

    groups = compute_groups(shap_df)
    print()
    print(f"  {'Group':<35} {'SHAP %':>8}  {'# Features':>11}")
    print('  ' + '-' * 58)
    for _, r in groups.iterrows():
        print(f"  {r['group'].replace(chr(10),' '):<35} {r['pct']:>7.1f}%  {r['n_features']:>11}")

    print()
    burden = groups.loc[groups['group']=='Health Burden\n& Engagement','pct'].values[0]
    supply = groups.loc[groups['group']=='Healthcare\nProvider Supply','pct'].values[0]
    print(f'  Health Burden ({burden:.1f}%) vs Healthcare Supply ({supply:.1f}%) = {burden/supply:.1f}× ratio')
    print()
    make_figure(groups)
    print('  Done.')

