"""
build_delta_profile.py
----------------------
Produces a comprehensive Delta tract-level dataset merging all available
data sources for the deep-dive pages. Also adds county-level aggregations.

Reads: master_tract.parquet, igs_latest.parquet, igs_trends.parquet,
       community_typology.parquet (if available), tract_prescriptions.parquet (if available)
Writes: delta_profile.parquet

This single parquet powers Pages 2, 4, and 5 of the app.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import numpy as np
from config import (
    MASTER_TRACT, IGS_LATEST_PARQUET, IGS_TRENDS_PARQUET,
    DELTA_PROFILE, PROCESSED,
    DELTA_COUNTY_FIPS, DELTA_COUNTY_NAMES,
    IGS_SUB_TO_PILLAR, IGS_VULN_THRESHOLD,
)


def build_delta_profile():
    print("=" * 65)
    print("  Building Comprehensive Delta Profile")
    print("=" * 65)

    # ── Load master tract and filter to Delta ────────────────────────────────
    master = pd.read_parquet(MASTER_TRACT)
    master['county_fips5'] = master['GEOID'].str[:5]
    delta = master[master['county_fips5'].isin(DELTA_COUNTY_FIPS)].copy()
    print(f"  Delta tracts from master_tract: {len(delta)}")

    # ── Merge IGS sub-indicators from igs_latest ─────────────────────────────
    igs = pd.read_parquet(IGS_LATEST_PARQUET)
    sub_indicator_cols = [c for c in igs.columns if c in IGS_SUB_TO_PILLAR]
    extra_igs_cols = [c for c in sub_indicator_cols if c not in delta.columns]
    if extra_igs_cols:
        delta = delta.merge(igs[['GEOID'] + extra_igs_cols], on='GEOID', how='left')
        print(f"  Merged {len(extra_igs_cols)} sub-indicator columns from IGS")

    # ── Add county name ──────────────────────────────────────────────────────
    delta['county_name'] = delta['county_fips5'].map(DELTA_COUNTY_NAMES)

    # ── Add typology if available ────────────────────────────────────────────
    typ_path = PROCESSED / 'community_typology.parquet'
    if typ_path.exists():
        typ = pd.read_parquet(typ_path)
        typ_cols = ['GEOID', 'typology', 'delta_igs', 'igs_score_2017', 'igs_score_2025']
        typ_cols = [c for c in typ_cols if c in typ.columns]
        # Also grab 2017 sub-indicator baselines
        baseline_cols = [c for c in typ.columns if c.endswith('_2017') and c not in typ_cols]
        typ_cols += baseline_cols
        typ_cols = list(dict.fromkeys(typ_cols))
        merge_cols = [c for c in typ_cols if c not in delta.columns or c == 'GEOID']
        if 'GEOID' in merge_cols:
            delta = delta.merge(typ[merge_cols], on='GEOID', how='left')
            print(f"  Merged typology columns: {[c for c in merge_cols if c != 'GEOID']}")

    # ── Add prescriptions if available ───────────────────────────────────────
    rx_path = PROCESSED / 'tract_prescriptions.parquet'
    if rx_path.exists():
        rx = pd.read_parquet(rx_path)
        rx_cols = [c for c in rx.columns if c not in delta.columns or c == 'GEOID']
        if 'GEOID' in rx_cols and len(rx_cols) > 1:
            delta = delta.merge(rx[rx_cols], on='GEOID', how='left')
            print(f"  Merged prescription columns: {len(rx_cols) - 1}")

    # ── Compute county-level means for comparison bars ───────────────────────
    numeric_cols = delta.select_dtypes(include='number').columns.tolist()
    county_means = delta.groupby('county_fips5')[numeric_cols].mean()
    county_means = county_means.add_prefix('county_mean_')
    county_means = county_means.reset_index()
    delta = delta.merge(county_means, on='county_fips5', how='left')

    # ── Add national means for comparison ────────────────────────────────────
    key_national_cols = ['igs_score', 'igs_economy', 'igs_place', 'igs_community']
    key_national_cols += [c for c in IGS_SUB_TO_PILLAR if c in master.columns]

    health_cols = [c for c in master.columns if 'CrudePrev' in c]
    key_national_cols += health_cols

    key_national_cols = [c for c in key_national_cols if c in master.columns]
    national_means = master[key_national_cols].mean()
    for c in key_national_cols:
        delta[f'national_mean_{c}'] = round(float(national_means[c]), 2)

    # ── Flag below-threshold ─────────────────────────────────────────────────
    if 'igs_score' in delta.columns:
        delta['below_45'] = (delta['igs_score'] < IGS_VULN_THRESHOLD).astype(int)

    delta.to_parquet(DELTA_PROFILE, index=False)
    print(f"\n  Saved delta_profile.parquet: {len(delta)} tracts, {len(delta.columns)} columns")

    # ── Summary ──────────────────────────────────────────────────────────────
    for county_fips, name in DELTA_COUNTY_NAMES.items():
        sub = delta[delta['county_fips5'] == county_fips]
        if len(sub) > 0:
            igs_mean = sub['igs_score'].mean() if 'igs_score' in sub.columns else 0
            n = len(sub)
            print(f"    {name:<12} {n:>3} tracts  IGS={igs_mean:.1f}")

    return delta


if __name__ == "__main__":
    build_delta_profile()
    print("\n  Done.")
