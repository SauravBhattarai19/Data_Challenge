"""
build_igs_national.py
---------------------
Produces a county-level IGS aggregation for the national overview page.

Reads: igs_latest.parquet (one row per tract, latest year)
Writes: igs_national.parquet (one row per county)

Columns: county_fips5, state_fips, n_tracts, n_below_45, pct_below_45,
         igs_score, igs_economy, igs_place, igs_community,
         + all 14 sub-indicator means (where available)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import numpy as np
from config import (
    IGS_LATEST_PARQUET, IGS_NATIONAL, IGS_VULN_THRESHOLD,
    DELTA_COUNTY_FIPS, DELTA_COUNTY_NAMES, IGS_SUB_TO_PILLAR,
)


# US state FIPS → name mapping (50 states + DC + territories)
STATE_FIPS_NAMES = {
    '01': 'Alabama', '02': 'Alaska', '04': 'Arizona', '05': 'Arkansas',
    '06': 'California', '08': 'Colorado', '09': 'Connecticut', '10': 'Delaware',
    '11': 'District of Columbia', '12': 'Florida', '13': 'Georgia', '15': 'Hawaii',
    '16': 'Idaho', '17': 'Illinois', '18': 'Indiana', '19': 'Iowa',
    '20': 'Kansas', '21': 'Kentucky', '22': 'Louisiana', '23': 'Maine',
    '24': 'Maryland', '25': 'Massachusetts', '26': 'Michigan', '27': 'Minnesota',
    '28': 'Mississippi', '29': 'Missouri', '30': 'Montana', '31': 'Nebraska',
    '32': 'Nevada', '33': 'New Hampshire', '34': 'New Jersey', '35': 'New Mexico',
    '36': 'New York', '37': 'North Carolina', '38': 'North Dakota', '39': 'Ohio',
    '40': 'Oklahoma', '41': 'Oregon', '42': 'Pennsylvania', '44': 'Rhode Island',
    '45': 'South Carolina', '46': 'South Dakota', '47': 'Tennessee', '48': 'Texas',
    '49': 'Utah', '50': 'Vermont', '51': 'Virginia', '53': 'Washington',
    '54': 'West Virginia', '55': 'Wisconsin', '56': 'Wyoming',
    '72': 'Puerto Rico', '78': 'Virgin Islands', '66': 'Guam',
    '69': 'Northern Mariana Islands', '60': 'American Samoa',
}


def build_igs_national():
    print("=" * 65)
    print("  Building National IGS County Aggregation")
    print("=" * 65)

    df = pd.read_parquet(IGS_LATEST_PARQUET)
    print(f"  Loaded {len(df):,} tracts from igs_latest.parquet")

    df['county_fips5'] = df['GEOID'].str[:5]
    df['state_fips'] = df['GEOID'].str[:2]

    pillar_cols = ['igs_score', 'igs_economy', 'igs_place', 'igs_community']
    sub_cols = [c for c in IGS_SUB_TO_PILLAR.keys() if c in df.columns]
    numeric_cols = [c for c in pillar_cols + sub_cols if c in df.columns]

    agg_dict = {c: 'mean' for c in numeric_cols}
    agg_dict['GEOID'] = 'count'

    county = df.groupby('county_fips5').agg(agg_dict).reset_index()
    county = county.rename(columns={'GEOID': 'n_tracts'})

    below_45 = df[df['igs_score'] < IGS_VULN_THRESHOLD].groupby('county_fips5')['GEOID'].count().reset_index()
    below_45.columns = ['county_fips5', 'n_below_45']
    county = county.merge(below_45, on='county_fips5', how='left')
    county['n_below_45'] = county['n_below_45'].fillna(0).astype(int)
    county['pct_below_45'] = (county['n_below_45'] / county['n_tracts'] * 100).round(1)

    county['state_fips'] = county['county_fips5'].str[:2]
    county['state_name'] = county['state_fips'].map(STATE_FIPS_NAMES).fillna('Unknown')
    county['is_delta'] = county['county_fips5'].isin(DELTA_COUNTY_FIPS)
    county['county_name'] = county['county_fips5'].map(DELTA_COUNTY_NAMES).fillna('')

    for c in numeric_cols:
        county[c] = county[c].round(2)

    county.to_parquet(IGS_NATIONAL, index=False)
    print(f"  Saved {len(county):,} counties -> {IGS_NATIONAL}")
    print(f"  Counties with 100% below-45 tracts: {(county['pct_below_45'] == 100).sum()}")
    print(f"  Mean national IGS: {county['igs_score'].mean():.1f}")

    delta = county[county['is_delta']]
    if len(delta) > 0:
        print(f"\n  Delta counties ({len(delta)}):")
        print(f"    Mean IGS: {delta['igs_score'].mean():.1f}")
        print(f"    Mean Economy: {delta['igs_economy'].mean():.1f}")
        print(f"    Mean Place: {delta['igs_place'].mean():.1f}")
        print(f"    Mean Community: {delta['igs_community'].mean():.1f}")

    return county


if __name__ == "__main__":
    build_igs_national()
    print("\n  Done.")
