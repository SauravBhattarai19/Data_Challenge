"""
Table components for the Streamlit app.
Returns DataFrames ready for st.dataframe() with appropriate formatting.
"""

import pandas as pd
import numpy as np


def lowest_igs_counties(county_df: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    """Return the N counties with lowest mean IGS score."""
    cols = ['county_fips5', 'state_name', 'n_tracts', 'pct_below_45',
            'igs_score', 'igs_economy', 'igs_place', 'igs_community']
    cols = [c for c in cols if c in county_df.columns]

    result = county_df.nsmallest(n, 'igs_score')[cols].copy()
    rename = {
        'county_fips5': 'County FIPS',
        'state_name': 'State',
        'n_tracts': 'Tracts',
        'pct_below_45': '% Below 45',
        'igs_score': 'IGS Score',
        'igs_economy': 'Economy',
        'igs_place': 'Place',
        'igs_community': 'Community',
    }
    result = result.rename(columns={k: v for k, v in rename.items() if k in result.columns})

    for c in ['IGS Score', 'Economy', 'Place', 'Community']:
        if c in result.columns:
            result[c] = result[c].round(1)
    return result.reset_index(drop=True)


def delta_county_summary(delta_df: pd.DataFrame, delta_names: dict) -> pd.DataFrame:
    """Summary table for the 9 Delta counties."""
    if 'county_fips5' not in delta_df.columns:
        return pd.DataFrame()

    agg_spec = {
        'n_tracts': ('GEOID', 'count'),
        'igs_score': ('igs_score', 'mean'),
    }
    for col in ('igs_economy', 'igs_place', 'igs_community'):
        if col in delta_df.columns:
            agg_spec[col] = (col, 'mean')
    agg = delta_df.groupby('county_fips5').agg(**agg_spec).reset_index()

    agg['County'] = agg['county_fips5'].map(delta_names)
    agg = agg.sort_values('igs_score')

    for c in ['igs_score', 'igs_economy', 'igs_place', 'igs_community']:
        if c in agg.columns:
            agg[c] = agg[c].round(1)

    agg = agg.rename(columns={
        'n_tracts': 'Tracts',
        'igs_score': 'IGS',
        'igs_economy': 'Economy',
        'igs_place': 'Place',
        'igs_community': 'Community',
    })
    final_cols = ['County', 'Tracts', 'IGS'] + [
        c for c in ('Economy', 'Place', 'Community') if c in agg.columns
    ]
    return agg[final_cols]


def comparison_table(selected_values: dict, target_values: dict,
                     national_values: dict = None) -> pd.DataFrame:
    """
    Build a comparison table: indicator | current | target | gap | national
    All inputs are dicts: {indicator_name: value}
    """
    rows = []
    for ind in selected_values:
        current = selected_values[ind]
        target = target_values.get(ind, np.nan)
        gap = current - target if not np.isnan(target) else np.nan
        row = {
            'Indicator': ind,
            'Current': round(current, 1) if not np.isnan(current) else '-',
            'Turnaround Target': round(target, 1) if not np.isnan(target) else '-',
            'Gap': round(gap, 1) if not np.isnan(gap) else '-',
        }
        if national_values:
            nat = national_values.get(ind, np.nan)
            row['National Avg'] = round(nat, 1) if not np.isnan(nat) else '-'
        rows.append(row)

    return pd.DataFrame(rows)
