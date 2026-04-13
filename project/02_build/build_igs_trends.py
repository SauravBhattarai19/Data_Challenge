"""
Build IGS 2017–2025 time-series dataset with resilience context.

Joins each year's IGS economy score with static context from master_tract
to produce a per-tract, per-year trajectory dataset.

This enables the narrative: "MS Delta resilience stagnated 2017–2025 while
national averages improved."

Output: data_processed/igs_trends.parquet
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from config import (
    IGS_PARQUET, MASTER_TRACT, IGS_TRENDS_PARQUET,
    PROCESSED, DELTA_COUNTY_FIPS, DELTA_COUNTY_NAMES
)


def build_igs_trends():
    PROCESSED.mkdir(parents=True, exist_ok=True)

    print(f"Loading IGS all years: {IGS_PARQUET}")
    igs = pd.read_parquet(IGS_PARQUET)
    print(f"  Shape: {igs.shape}")
    print(f"  Years: {sorted(igs['year'].unique())}")

    print(f"\nLoading master tract: {MASTER_TRACT}")
    master = pd.read_parquet(MASTER_TRACT)
    # Select static context columns
    context_cols = [
        'GEOID', 'county_fips5', 'is_delta',
        'RISK_SCORE', 'RISK_RATNG',
        'RPL_THEMES',
        'in_pc_hpsa', 'in_mua',
        'pc_hpsa_score_max',
        'biz_total_healthcare',
        'POPULATION',
        'HWAV_RISKR', 'IFLD_RISKR',
    ]
    context_cols = [c for c in context_cols if c in master.columns]
    context = master[context_cols].copy()

    # ── Join IGS trends with static context ───────────────────────────────────
    trends = igs.merge(context, on='GEOID', how='left')

    # ── Add county name for Delta tracts ──────────────────────────────────────
    if 'county_fips5' in trends.columns:
        trends['county_name'] = trends['county_fips5'].map(DELTA_COUNTY_NAMES)
    if 'is_delta' not in trends.columns:
        if 'county_fips5' in trends.columns:
            trends['is_delta'] = trends['county_fips5'].isin(DELTA_COUNTY_FIPS)
        else:
            trends['is_delta'] = False

    print(f"\n  Trends shape: {trends.shape}")
    print(f"  Delta tracts across all years: {trends[trends['is_delta']].shape[0]:,}")

    # ── Compute year-over-year national and Delta averages ────────────────────
    national_avg = trends.groupby('year')['igs_score'].mean().reset_index()
    national_avg.columns = ['year', 'national_avg_igs']

    delta_avg = (
        trends[trends['is_delta']]
        .groupby('year')['igs_score']
        .mean()
        .reset_index()
    )
    delta_avg.columns = ['year', 'delta_avg_igs']

    print("\n  IGS Score Trend — National vs. MS Delta:")
    trend_summary = national_avg.merge(delta_avg, on='year', how='left')
    trend_summary['gap'] = trend_summary['national_avg_igs'] - trend_summary['delta_avg_igs']
    print(trend_summary.to_string(index=False))

    trends.to_parquet(IGS_TRENDS_PARQUET, index=False)
    print(f"\n  Saved -> {IGS_TRENDS_PARQUET}")
    return trends


if __name__ == "__main__":
    build_igs_trends()
