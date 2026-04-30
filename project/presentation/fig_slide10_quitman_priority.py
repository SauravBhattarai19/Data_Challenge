"""
Slide 10 — SHAP × Gap Priority Matrix for Quitman County (28119)
=================================================================
Identical logic to the app's Page 4 (Investment Priority Matrix) —
uses compute_gaps() from expanded_priority_matrix.py so this figure
is always consistent with what the app shows.

Axes:
  Y: SHAP importance (% of total predictive power)
  X: normalised gap from turnaround benchmark
     positive = Quitman is behind; negative = Quitman is ahead

Quadrants (default thresholds, matching app):
  ACT NOW     — SHAP ≥ 3.0 % AND gap ≥ 20
  PROTECT     — SHAP ≥ 3.0 % AND gap <  20  (or negative = ahead)
  SECOND WAVE — SHAP <  3.0 % AND gap ≥ 20
  MONITOR     — SHAP <  3.0 % AND gap <  20

Data sources (current):
  data_processed/expanded_model.parquet
  data_processed/expanded_shap_summary.parquet
  03_analysis/expanded_priority_matrix.py   (compute_gaps, FEATURE_REGISTRY, CATEGORY_COLORS)

Output: presentation/figures/slide10_quitman_priority.png  (300 DPI)
Run:    python presentation/fig_slide10_quitman_priority.py
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
from matplotlib.lines import Line2D
from adjustText import adjust_text

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
PROCESSED = PROJECT_ROOT / 'data_processed'
OUT_DIR   = PROJECT_ROOT / 'presentation' / 'figures'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Load analysis module (filename starts with digit → importlib) ─────────────
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

# ── Settings ──────────────────────────────────────────────────────────────────
QUITMAN      = '28119'
SHAP_THRESH  = 3.0    # % — matches app default
GAP_THRESH   = 20     # normalised units — matches app default
TOP_N        = 22     # features to plot (by SHAP rank)
GAP_FLOOR    = -15    # x-axis left edge; features ahead of benchmark pin here

# Non-actionable categories to drop from the scatter (demographic context only)
SKIP_CATS = {'SVI: Demographic', 'Food/Demo: Age', 'Economic Context'}

BG_COLOR   = '#ffffff'
TEXT_COLOR = '#1a1a2e'
DIM_COLOR  = '#555566'
BORDER_CLR = '#cccccc'

plt.rcParams.update({
    'font.family':      'DejaVu Sans',
    'figure.facecolor': BG_COLOR,
    'text.color':       TEXT_COLOR,
    'axes.facecolor':   BG_COLOR,
    'axes.edgecolor':   BORDER_CLR,
})


# ── Data ──────────────────────────────────────────────────────────────────────
def load_data() -> pd.DataFrame:
    df           = pd.read_parquet(PROCESSED / 'expanded_model.parquet')
    shap_summary = pd.read_parquet(PROCESSED / 'expanded_shap_summary.parquet')

    with contextlib.redirect_stdout(io.StringIO()):
        features = get_ml_features(df)
        gap_df, _ = compute_gaps(df, features, shap_summary, QUITMAN)

    # Drop non-actionable demographic categories
    gap_df = gap_df[~gap_df['category'].isin(SKIP_CATS)].reset_index(drop=True)

    # Keep top N by SHAP, then clip gap to plot floor for display
    gap_df = gap_df.sort_values('shap_pct', ascending=False).head(TOP_N).copy()
    gap_df['gap_plot'] = gap_df['gap'].clip(lower=GAP_FLOOR)

    # Quadrant assignment
    gap_df['is_high_shap'] = gap_df['shap_pct'] >= SHAP_THRESH
    gap_df['is_large_gap'] = gap_df['gap']      >= GAP_THRESH
    gap_df['quadrant'] = gap_df.apply(
        lambda r: ('ACT NOW'     if r['is_high_shap'] and r['is_large_gap'] else
                   'PROTECT'     if r['is_high_shap'] and not r['is_large_gap'] else
                   'SECOND WAVE' if not r['is_high_shap'] and r['is_large_gap'] else
                   'MONITOR'),
        axis=1,
    )
    return gap_df


# ── Figure ────────────────────────────────────────────────────────────────────
def make_figure(data: pd.DataFrame):
    fig = plt.figure(figsize=(9, 6.2), facecolor=BG_COLOR)
    ax  = fig.add_axes([0.09, 0.10, 0.88, 0.82])
    ax.set_facecolor(BG_COLOR)

    shap_max = max(data['shap_pct'].max() * 1.12, SHAP_THRESH * 3.2)

    # ── Quadrant backgrounds ──────────────────────────────────────────────────
    quads = {
        ('high', 'large'): ('#fde8e8', '#c0392b', 'ACT NOW'),
        ('high', 'small'): ('#e8f8e8', '#1e8449', 'PROTECT'),
        ('low',  'large'): ('#fff3e0', '#e67e22', 'SECOND WAVE'),
        ('low',  'small'): ('#f5f5f5', '#888888', 'MONITOR'),
    }
    for (hs, lg), (bg_c, _, _) in quads.items():
        x0 = GAP_THRESH  if lg == 'large' else GAP_FLOOR
        x1 = 105         if lg == 'large' else GAP_THRESH
        y0 = SHAP_THRESH if hs == 'high'  else 0
        y1 = shap_max    if hs == 'high'  else SHAP_THRESH
        ax.fill_between([x0, x1], [y0, y0], [y1, y1],
                        color=bg_c, alpha=0.50, zorder=0)

    ax.axhline(SHAP_THRESH, color='#999', lw=1.8, ls='--', zorder=1)
    ax.axvline(GAP_THRESH,  color='#999', lw=1.8, ls='--', zorder=1)

    # Quadrant corner labels
    quad_labels = {
        ('high', 'large'): (GAP_THRESH + 1.5, SHAP_THRESH + 0.2, '#c0392b'),
        ('high', 'small'): (GAP_FLOOR + 0.5,  SHAP_THRESH + 0.2, '#1e8449'),
        ('low',  'large'): (GAP_THRESH + 1.5, 0.18,               '#e67e22'),
        ('low',  'small'): (GAP_FLOOR + 0.5,  0.18,               '#888888'),
    }
    for (hs, lg), (xt, yt, tc) in quad_labels.items():
        _, _, txt = quads[(hs, lg)]
        ax.text(xt, yt, txt, fontsize=10.5, color=tc, fontweight='bold',
                va='bottom', ha='left', zorder=2, alpha=0.75,
                path_effects=[pe.withStroke(linewidth=2, foreground='white')])

    # ── Scatter points + labels ───────────────────────────────────────────────
    texts = []
    for _, row in data.iterrows():
        ax.scatter(
            row['gap_plot'], row['shap_pct'],
            s=300, color=row['color'], zorder=5,
            edgecolors='white', linewidths=1.8,
        )

        # Value annotation: Quitman raw vs turnaround benchmark
        def _fmt(v, col=row['feature']):
            if col == 'BUILDVALUE':
                return f'${v/1e6:.0f}M'
            if col.endswith('CrudePrev') or col.startswith('PCT_'):
                return f'{v:.1f}%'
            if 'Score' in col:
                return f'{v:.1f}'
            if col.startswith('biz_'):
                return f'{v:,.0f}'
            return f'{v:.1f}'

        pair = f"{_fmt(row['target_raw'])} → {_fmt(row['turn_raw'])}"
        t = ax.text(
            row['gap_plot'], row['shap_pct'],
            f"{row['label']}\n({pair})",
            fontsize=8.8, color=row['color'], fontweight='bold',
            ha='center', va='center', zorder=6,
            path_effects=[pe.withStroke(linewidth=2, foreground='white')],
        )
        texts.append(t)

    adjust_text(
        texts,
        x=data['gap_plot'].values,
        y=data['shap_pct'].values,
        ax=ax,
        expand_points=(1.5, 1.6),
        expand_text=(1.3, 1.45),
        force_points=(0.4, 0.55),
        force_text=(0.5, 0.7),
        lim=1200,
        arrowprops=dict(
            arrowstyle='-', color='#666666',
            lw=0.6, alpha=0.55,
            shrinkA=8, shrinkB=4,
        ),
    )

    # ── Axes decoration ───────────────────────────────────────────────────────
    ax.set_xlim(GAP_FLOOR, 105)
    ax.set_ylim(0, shap_max)
    ax.set_xlabel(
        f'Gap from turnaround benchmark  (normalised 0–100; >{GAP_THRESH} = large gap)',
        fontsize=11.5, labelpad=8,
    )
    ax.set_ylabel('SHAP importance (% of total predictive power)', fontsize=11.5)
    ax.tick_params(labelsize=10)
    for sp in ax.spines.values():
        sp.set_edgecolor(BORDER_CLR)

    # "Ahead of benchmark" annotation on the left
    ax.text(GAP_FLOOR + 0.3, shap_max * 0.5,
            '← Ahead of\nbenchmark', fontsize=8.5,
            color=DIM_COLOR, ha='left', va='center', style='italic',
            rotation=90)

    # ── Legend — domain colors ────────────────────────────────────────────────
    seen, handles = set(), []
    for _, r in data.iterrows():
        cat = r['category']
        if cat not in seen:
            seen.add(cat)
            handles.append(mpatches.Patch(color=r['color'], label=cat))
    ax.legend(
        handles=handles, fontsize=8.5, loc='upper left',
        facecolor=BG_COLOR, edgecolor=BORDER_CLR, framealpha=0.95,
        ncol=2, columnspacing=0.9, handlelength=1.0,
    )

    ax.set_title(
        'Quitman County — Investment Priority Matrix\n'
        '(label format: county value → turnaround benchmark)',
        fontsize=12.5, fontweight='bold', pad=8, loc='left',
    )

    out = OUT_DIR / 'slide10_quitman_priority.png'
    plt.savefig(out, dpi=300, bbox_inches='tight',
                facecolor=BG_COLOR, edgecolor='none')
    plt.close()
    print(f'  Saved → {out}')


# ── CLI summary ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print('=' * 65)
    print('  Slide 10 — Quitman County Priority Matrix  (expanded model)')
    print('=' * 65)

    data = load_data()

    print(f"\n  {'Feature':<38} {'SHAP%':>6} {'Gap':>7}  Quadrant")
    print('  ' + '-' * 72)
    for _, r in data.sort_values('shap_pct', ascending=False).iterrows():
        print(
            f"  {r['label']:<38} "
            f"{r['shap_pct']:>6.1f} {r['gap']:>7.1f}  "
            f"{r['quadrant']}"
        )

    by_q = data.groupby('quadrant')['label'].apply(list)
    print('\n  Quadrant summary:')
    for q in ['ACT NOW', 'PROTECT', 'SECOND WAVE', 'MONITOR']:
        items = by_q.get(q, [])
        print(f'  {q:<12}: {len(items)}  {", ".join(items[:4])}{"…" if len(items) > 4 else ""}')

    print()
    make_figure(data)
    print('  Done.')
