"""
Chart components for the Streamlit app.
All charts use Plotly for consistency and interactivity.
"""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np


BLUE = '#2563eb'
RED = '#dc2626'
GREEN = '#059669'
ORANGE = '#f97316'
GRAY = '#6b7280'
LIGHT_BG = '#fafbfc'


def _base_layout(**overrides):
    base = dict(
        paper_bgcolor='#ffffff',
        plot_bgcolor=LIGHT_BG,
        font=dict(color='#374151', family='Inter, system-ui, sans-serif', size=12),
        margin=dict(l=50, r=30, t=50, b=40),
    )
    base.update(overrides)
    return base


def pillar_radar(data: dict, title: str = 'IGS Pillar Breakdown',
                 comparison: dict = None, comp_label: str = 'National Average') -> go.Figure:
    """
    Radar chart for Place / Community / Economy pillars.
    data = {'Place': 35, 'Community': 28, 'Economy': 40}
    comparison = optional second trace for national/turnaround avg.
    """
    categories = list(data.keys())
    values = list(data.values())
    values += [values[0]]
    categories += [categories[0]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values, theta=categories,
        fill='toself', fillcolor=f'rgba(37,99,235,0.15)',
        line=dict(color=BLUE, width=2),
        name='Selected',
    ))

    if comparison:
        comp_values = [comparison.get(c, 0) for c in categories[:-1]]
        comp_values += [comp_values[0]]
        fig.add_trace(go.Scatterpolar(
            r=comp_values, theta=categories,
            fill='toself', fillcolor='rgba(107,114,128,0.08)',
            line=dict(color=GRAY, width=1.5, dash='dash'),
            name=comp_label,
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(range=[0, 100], showticklabels=True, tickfont=dict(size=10)),
            bgcolor=LIGHT_BG,
        ),
        title=dict(text=title, font=dict(size=14)),
        showlegend=True,
        legend=dict(orientation='h', y=-0.1),
        **_base_layout(height=380),
    )
    return fig


def gap_bar_chart(gaps: pd.DataFrame, title: str = 'Gap to Turnaround Target') -> go.Figure:
    """
    Horizontal bar chart showing gap for each sub-indicator.
    gaps DataFrame must have columns: 'indicator', 'current', 'target', 'gap'
    """
    gaps = gaps.sort_values('gap', ascending=True)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=gaps['indicator'],
        x=gaps['gap'],
        orientation='h',
        marker_color=[RED if g < 0 else GREEN for g in gaps['gap']],
        text=[f'{g:+.1f}' for g in gaps['gap']],
        textposition='outside',
        textfont=dict(size=11),
    ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        xaxis_title='Gap (current - target)',
        yaxis=dict(tickfont=dict(size=11)),
        **_base_layout(height=max(300, len(gaps) * 35 + 80)),
    )
    return fig


def model_comparison_chart(models_df: pd.DataFrame) -> go.Figure:
    """
    Bar chart comparing ML model AUC scores.
    models_df has: model, train_auc, cv_auc_mean, cv_auc_std
    """
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name='Train AUC',
        x=models_df['model'],
        y=models_df['train_auc'],
        marker_color='#93c5fd',
        text=[f'{v:.3f}' for v in models_df['train_auc']],
        textposition='outside',
    ))
    fig.add_trace(go.Bar(
        name='CV AUC (5-fold)',
        x=models_df['model'],
        y=models_df['cv_auc_mean'],
        marker_color=BLUE,
        error_y=dict(type='data', array=models_df['cv_auc_std'].tolist(), visible=True),
        text=[f'{v:.3f}' for v in models_df['cv_auc_mean']],
        textposition='outside',
    ))

    fig.update_layout(
        title=dict(text='Model Comparison: Predicting IGS Turnaround', font=dict(size=14)),
        yaxis=dict(title='AUC', range=[0.5, 1.0]),
        barmode='group',
        **_base_layout(height=380),
    )
    return fig


def shap_importance_chart(summary_df: pd.DataFrame, top_n: int = 15) -> go.Figure:
    """
    Horizontal bar chart of top SHAP feature importances.
    summary_df has: feature, mean_abs_shap, mean_shap
    """
    top = summary_df.head(top_n).sort_values('mean_abs_shap', ascending=True)

    colors = [GREEN if s > 0 else RED for s in top['mean_shap']]

    fig = go.Figure(go.Bar(
        y=top['feature'],
        x=top['mean_abs_shap'],
        orientation='h',
        marker_color=colors,
        text=[f'{v:.4f}' for v in top['mean_abs_shap']],
        textposition='outside',
    ))

    fig.update_layout(
        title=dict(text=f'Top {top_n} Features by SHAP Importance', font=dict(size=14)),
        xaxis_title='Mean |SHAP value|',
        yaxis=dict(tickfont=dict(size=10)),
        **_base_layout(height=max(350, top_n * 28 + 80)),
    )
    fig.add_annotation(
        text="Green = pushes toward turnaround | Red = pushes toward stuck",
        xref='paper', yref='paper', x=0.5, y=-0.12,
        showarrow=False, font=dict(size=10, color=GRAY),
    )
    return fig


def shap_dimension_donut(dim_df: pd.DataFrame) -> go.Figure:
    """
    Donut chart showing SHAP dimension weights.
    dim_df has: dimension_name, weight_pct
    """
    colors = ['#2563eb', '#dc2626', '#059669', '#f97316', '#8b5cf6', '#06b6d4', '#d97706']

    fig = go.Figure(go.Pie(
        labels=dim_df['dimension_name'],
        values=dim_df['weight_pct'],
        hole=0.55,
        marker=dict(colors=colors[:len(dim_df)]),
        textinfo='label+percent',
        textposition='outside',
        textfont=dict(size=11),
    ))

    fig.update_layout(
        title=dict(text='What Matters Most? SHAP Dimension Weights', font=dict(size=14)),
        showlegend=False,
        **_base_layout(height=400),
    )
    return fig


def turnaround_vs_stuck_bars(benchmarks_df: pd.DataFrame,
                              indicators: list = None) -> go.Figure:
    """
    Grouped bar chart: turnaround mean vs stuck mean for key indicators.
    """
    if indicators is None:
        indicators = benchmarks_df['indicator'].unique()[:10]

    bm = benchmarks_df[benchmarks_df['indicator'].isin(indicators)]
    turn = bm[bm['typology'] == 'Turnaround'].set_index('indicator')
    stuck = bm[bm['typology'] == 'Stuck'].set_index('indicator')

    common = [i for i in indicators if i in turn.index and i in stuck.index]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name='Turnaround',
        x=common,
        y=[turn.loc[i, 'mean_2025'] for i in common],
        marker_color=GREEN,
    ))
    fig.add_trace(go.Bar(
        name='Stuck',
        x=common,
        y=[stuck.loc[i, 'mean_2025'] for i in common],
        marker_color=RED,
    ))

    fig.update_layout(
        title=dict(text='Turnaround vs Stuck: 2025 Indicator Values', font=dict(size=14)),
        barmode='group',
        xaxis=dict(tickangle=-35, tickfont=dict(size=9)),
        yaxis_title='Score',
        **_base_layout(height=420),
    )
    return fig


def igs_simulator_gauge(current_igs: float, projected_igs: float,
                         threshold: float = 45) -> go.Figure:
    """Gauge showing current vs projected IGS score."""
    fig = go.Figure(go.Indicator(
        mode='gauge+number+delta',
        value=projected_igs,
        delta=dict(reference=current_igs, valueformat='+.1f'),
        gauge=dict(
            axis=dict(range=[0, 100], tickwidth=1),
            bar=dict(color=BLUE),
            bgcolor=LIGHT_BG,
            steps=[
                dict(range=[0, threshold], color='#fee2e2'),
                dict(range=[threshold, 100], color='#dcfce7'),
            ],
            threshold=dict(
                line=dict(color=RED, width=3),
                thickness=0.8,
                value=threshold,
            ),
        ),
        title=dict(text='Projected IGS Score'),
    ))

    fig.update_layout(**_base_layout(height=280))
    return fig


def county_igs_bars(county_data: pd.DataFrame) -> go.Figure:
    """Bar chart of IGS scores for Delta counties."""
    df = county_data.sort_values('igs_score', ascending=True)

    colors = [RED if v < 45 else GREEN for v in df['igs_score']]

    fig = go.Figure(go.Bar(
        y=df['county_name'],
        x=df['igs_score'],
        orientation='h',
        marker_color=colors,
        text=[f'{v:.1f}' for v in df['igs_score']],
        textposition='outside',
    ))

    fig.add_vline(x=45, line_dash='dash', line_color=RED,
                  annotation_text='IGS = 45 Threshold')

    fig.update_layout(
        title=dict(text='Delta County IGS Scores', font=dict(size=14)),
        xaxis=dict(title='IGS Score', range=[0, max(65, df['igs_score'].max() + 5)]),
        **_base_layout(height=350),
    )
    return fig
