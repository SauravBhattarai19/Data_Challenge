"""
community_typology.py
---------------------
Stage 1 of the redesigned analytical pipeline.

Classifies all ~85,000 US census tracts into a 2x2 typology based on
their IGS trajectory from 2017 to 2025:

  +-----------------+-------------------+-------------------+
  |                 | IGS >= 45 in 2025 | IGS < 45 in 2025  |
  +-----------------+-------------------+-------------------+
  | IGS >= 45 (2017)| RESILIENT         | DECLINING         |
  | IGS < 45  (2017)| TURNAROUND        | STUCK             |
  +-----------------+-------------------+-------------------+

Why this matters:
  - Resilient tracts are BENCHMARKS (what "good enough" looks like)
  - Declining tracts are WARNINGS (what failure looks like from above)
  - Turnaround vs Stuck is the core ML question
  - Turnaround sub-indicator deltas become the empirical basis for the
    investment simulator (Stage 4) — replacing fabricated cost numbers

Outputs:
  data_processed/community_typology.parquet
    One row per tract: GEOID, igs_2017, igs_2025, delta_igs, typology,
    plus all 2017 sub-indicator baselines and 2025 outcomes.

  data_processed/turnaround_benchmarks.parquet
    Per sub-indicator: mean change in turnaround vs stuck tracts.
    This is what "realistic improvement" looks like — grounded in data.

  data_processed/typology_profiles.parquet
    Per typology group: mean of every feature (health, climate, social,
    business, etc.) — used for benchmark comparisons.

Run:
  python 03_analysis/community_typology.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import numpy as np
from config import (
    IGS_TRENDS_PARQUET, MASTER_TRACT, PROCESSED,
    DELTA_COUNTY_FIPS, DELTA_COUNTY_NAMES, IGS_VULN_THRESHOLD
)

THRESHOLD = IGS_VULN_THRESHOLD  # 45


def build_typology():
    print("=" * 65)
    print("  Stage 1: Community Typology (2x2 IGS Trajectory)")
    print("=" * 65)

    # ── Load IGS trends ──────────────────────────────────────────────────────
    trends = pd.read_parquet(IGS_TRENDS_PARQUET)
    print(f"  IGS trends: {len(trends):,} rows, {trends['GEOID'].nunique():,} tracts")
    print(f"  Years: {sorted(trends['year'].unique())}")

    # ── Extract 2017 and 2025 snapshots ──────────────────────────────────────
    # Sub-indicator columns we want for benchmarking
    SUB_COLS = [
        'Internet Access Score', 'Affordable Housing Score',
        'Travel Time to Work Score', 'Net Occupancy Score',
        'New Businesses Score', 'Small Business Loans Score',
        'Commercial Diversity Score', 'Minority/Women Owned Businesses Score',
        'Spend Growth Score',
        'Health Insurance Coverage Score', 'Labor Market Engagement Index Score',
        'Female Above Poverty Score', 'Personal Income Score',
        'Spending per Capita Score',
    ]
    pillar_cols = ['igs_score', 'igs_economy', 'igs_place', 'igs_community']
    keep_cols = ['GEOID'] + pillar_cols + [c for c in SUB_COLS if c in trends.columns]

    t17 = trends[trends['year'] == 2017][keep_cols].copy()
    t25 = trends[trends['year'] == 2025][keep_cols].copy()

    # Rename with suffix
    t17 = t17.rename(columns={c: f'{c}_2017' for c in t17.columns if c != 'GEOID'})
    t25 = t25.rename(columns={c: f'{c}_2025' for c in t25.columns if c != 'GEOID'})

    # Merge
    df = t17.merge(t25, on='GEOID', how='inner')
    print(f"  Tracts with both 2017 and 2025 data: {len(df):,}")

    # ── Classify into 2x2 typology ──────────────────────────────────────────
    below_2017 = df['igs_score_2017'] < THRESHOLD
    below_2025 = df['igs_score_2025'] < THRESHOLD

    conditions = [
        (~below_2017) & (~below_2025),  # Resilient
        (below_2017)  & (~below_2025),  # Turnaround
        (~below_2017) & (below_2025),   # Declining
        (below_2017)  & (below_2025),   # Stuck
    ]
    labels = ['Resilient', 'Turnaround', 'Declining', 'Stuck']
    df['typology'] = np.select(conditions, labels, default='Unknown')

    # Delta IGS
    df['delta_igs'] = df['igs_score_2025'] - df['igs_score_2017']

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n  {'Typology':<14} {'Count':>8} {'%':>7} {'Mean IGS 2017':>14} {'Mean IGS 2025':>14} {'Mean Delta':>11}")
    print("  " + "-" * 70)
    for label in labels:
        sub = df[df['typology'] == label]
        pct = len(sub) / len(df) * 100
        print(f"  {label:<14} {len(sub):>8,} {pct:>6.1f}% "
              f"{sub['igs_score_2017'].mean():>14.1f} "
              f"{sub['igs_score_2025'].mean():>14.1f} "
              f"{sub['delta_igs'].mean():>+11.1f}")

    # ── Merge with master_tract for feature profiles ─────────────────────────
    master = pd.read_parquet(MASTER_TRACT)
    # Drop IGS columns from master to avoid duplication
    master_feature_cols = [c for c in master.columns
                          if c not in pillar_cols and c != 'year']
    df = df.merge(master[master_feature_cols], on='GEOID', how='left')

    # ── Add Delta county context ─────────────────────────────────────────────
    if 'county_fips5' in df.columns:
        df['is_delta'] = df['county_fips5'].isin(DELTA_COUNTY_FIPS)
        delta = df[df['is_delta']]
        print(f"\n  MS Delta:")
        for label in labels:
            sub = delta[delta['typology'] == label]
            if len(sub) > 0:
                print(f"    {label}: {len(sub)} tracts ({len(sub)/len(delta)*100:.0f}%)")
        if len(delta) > 0:
            print(f"    Mean IGS 2017: {delta['igs_score_2017'].mean():.1f}")
            print(f"    Mean IGS 2025: {delta['igs_score_2025'].mean():.1f}")
            print(f"    Mean Delta: {delta['delta_igs'].mean():+.1f}")

    # ── Save typology ────────────────────────────────────────────────────────
    out_path = PROCESSED / 'community_typology.parquet'
    df.to_parquet(out_path, index=False)
    print(f"\n  Saved typology -> {out_path}  ({len(df):,} tracts)")

    return df


def build_turnaround_benchmarks(df: pd.DataFrame):
    """
    Compute what turnaround tracts ACTUALLY changed in each sub-indicator
    vs what stuck tracts changed. These deltas become the empirical basis
    for the investment simulator.
    """
    print("\n" + "=" * 65)
    print("  Turnaround Benchmarks: What Actually Changed?")
    print("=" * 65)

    SUB_COLS = [
        'Internet Access Score', 'Affordable Housing Score',
        'Travel Time to Work Score', 'Net Occupancy Score',
        'New Businesses Score', 'Small Business Loans Score',
        'Commercial Diversity Score', 'Minority/Women Owned Businesses Score',
        'Spend Growth Score',
        'Health Insurance Coverage Score', 'Labor Market Engagement Index Score',
        'Female Above Poverty Score', 'Personal Income Score',
        'Spending per Capita Score',
    ]

    # Also include pillar-level
    PILLAR_COLS = ['igs_economy', 'igs_place', 'igs_community']

    all_indicators = SUB_COLS + PILLAR_COLS
    rows = []

    print(f"\n  {'Indicator':<45} {'Turn. Delta':>13} {'Stuck Delta':>12} {'Gap':>8}")
    print("  " + "-" * 82)

    for ind in all_indicators:
        col_17 = f'{ind}_2017'
        col_25 = f'{ind}_2025'
        if col_17 not in df.columns or col_25 not in df.columns:
            continue

        for typ in ['Resilient', 'Turnaround', 'Declining', 'Stuck']:
            sub = df[df['typology'] == typ]
            delta = sub[col_25] - sub[col_17]
            rows.append({
                'indicator': ind,
                'typology': typ,
                'mean_2017': sub[col_17].mean(),
                'mean_2025': sub[col_25].mean(),
                'mean_delta': delta.mean(),
                'median_delta': delta.median(),
                'std_delta': delta.std(),
                'p25_delta': delta.quantile(0.25),
                'p75_delta': delta.quantile(0.75),
                'n_tracts': len(sub),
            })

        # Print turnaround vs stuck comparison
        turn = df[df['typology'] == 'Turnaround']
        stuck = df[df['typology'] == 'Stuck']
        t_delta = (turn[col_25] - turn[col_17]).mean()
        s_delta = (stuck[col_25] - stuck[col_17]).mean()
        gap = t_delta - s_delta
        print(f"  {ind:<45} {t_delta:>+13.2f} {s_delta:>+10.2f} {gap:>+8.2f}")

    benchmarks = pd.DataFrame(rows)
    out_path = PROCESSED / 'turnaround_benchmarks.parquet'
    benchmarks.to_parquet(out_path, index=False)
    print(f"\n  Saved benchmarks -> {out_path}  ({len(benchmarks)} rows)")

    return benchmarks


def build_typology_profiles(df: pd.DataFrame):
    """
    For each typology group, compute mean of every feature. These profiles
    let us say: "Resilient tracts have X diabetes rate; Stuck tracts have Y."
    """
    print("\n" + "=" * 65)
    print("  Typology Feature Profiles")
    print("=" * 65)

    # Features to profile (from master_tract)
    health_cols = [c for c in df.columns if 'CrudePrev' in c]
    climate_cols = [c for c in df.columns if c.endswith('_RISKS')]
    svi_cols = [c for c in df.columns if c.startswith('RPL_THEME')]
    hpsa_cols = ['pc_hpsa_score_max', 'mh_hpsa_score_max', 'fqhc_count']
    biz_cols = [c for c in df.columns if c.startswith('biz_')]
    other_cols = ['PovertyRate', 'POPULATION', 'BUILDVALUE', 'RISK_SCORE',
                  'RPL_EBM', 'RPL_HVM', 'RESL_SCORE', 'EAL_SCORE',
                  'LILATracts_1And10']

    profile_cols = (health_cols + climate_cols + svi_cols +
                    [c for c in hpsa_cols if c in df.columns] +
                    [c for c in biz_cols if c in df.columns] +
                    [c for c in other_cols if c in df.columns])

    profiles = df.groupby('typology')[profile_cols].agg(['mean', 'median', 'std']).reset_index()
    # Flatten multi-level columns
    profiles.columns = [f'{a}_{b}' if b else a for a, b in profiles.columns]

    out_path = PROCESSED / 'typology_profiles.parquet'
    profiles.to_parquet(out_path, index=False)
    print(f"  Saved profiles -> {out_path}")

    # Print key comparisons
    simple_means = df.groupby('typology')[profile_cols].mean()
    key_features = ['DIABETES_CrudePrev', 'DEPRESSION_CrudePrev', 'pc_hpsa_score_max',
                    'PovertyRate', 'RISK_SCORE', 'RPL_EBM', 'fqhc_count']
    key_features = [c for c in key_features if c in simple_means.columns]

    if key_features:
        print(f"\n  Key Feature Means by Typology:")
        print(simple_means[key_features].round(2).to_string())

    return profiles


def build_igs_improvement_model(df: pd.DataFrame):
    """
    Build the dataset used by Stage 2 ML. One row per tract that was below
    IGS=45 in 2017 (the at-risk population). Target: did they cross 45?
    Features: 2017 baselines + master_tract health/climate/social features.
    """
    print("\n" + "=" * 65)
    print("  ML-Ready Dataset: Turnaround vs Stuck")
    print("=" * 65)

    at_risk = df[df['igs_score_2017'] < THRESHOLD].copy()
    at_risk['turnaround'] = (at_risk['typology'] == 'Turnaround').astype(int)

    print(f"  At-risk tracts (IGS<{THRESHOLD} in 2017): {len(at_risk):,}")
    print(f"  Turnaround (crossed {THRESHOLD}): {at_risk['turnaround'].sum():,} "
          f"({at_risk['turnaround'].mean()*100:.1f}%)")
    print(f"  Stuck: {(1-at_risk['turnaround']).sum():,}")

    out_path = PROCESSED / 'igs_improvement_model.parquet'
    at_risk.to_parquet(out_path, index=False)
    print(f"  Saved -> {out_path}")

    return at_risk


if __name__ == "__main__":
    df = build_typology()
    build_turnaround_benchmarks(df)
    build_typology_profiles(df)
    build_igs_improvement_model(df)
    print("\n  Stage 1 complete.")
