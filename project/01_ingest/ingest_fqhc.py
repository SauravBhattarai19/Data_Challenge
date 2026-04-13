"""
Ingest HRSA Federally Qualified Health Centers (FQHC) site data -> Parquet.

Filters to active sites only and retains lat/lon for mapping.
Also saves county-level counts (number of FQHCs per county).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from config import FQHC_PATH, FQHC_PARQUET, PROCESSED_RAW, DELTA_COUNTY_FIPS

# Columns to keep
KEEP_COLS = [
    'Site Name',
    'Site Address',
    'Site City',
    'Site State Abbreviation',
    'Site Postal Code',
    'Geocoding Artifact Address Primary Y Coordinate',   # latitude
    'Geocoding Artifact Address Primary X Coordinate',   # longitude
    'Site Status Description',
    'Health Center Type',
    'State and County Federal Information Processing Standard Code',  # county FIPS int
]


def ingest_fqhc():
    print(f"Reading FQHC sites from {FQHC_PATH} ...")
    PROCESSED_RAW.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(FQHC_PATH, low_memory=False, encoding='latin-1')
    df.columns = df.columns.str.strip()
    print(f"  Raw shape: {df.shape}")

    # ── Filter active sites ────────────────────────────────────────────────────
    status_col = 'Site Status Description'
    if status_col in df.columns:
        df = df[df[status_col].str.strip() == 'Active']
        print(f"  Active sites: {len(df):,}")
    else:
        # Try alternate column name
        status_col = next((c for c in df.columns if 'status' in c.lower()), None)
        if status_col:
            df = df[df[status_col].str.strip() == 'Active']
            print(f"  Active sites (via {status_col}): {len(df):,}")

    # ── Keep available columns ─────────────────────────────────────────────────
    available = [c for c in KEEP_COLS if c in df.columns]
    df = df[available].copy()

    # ── Rename for convenience ─────────────────────────────────────────────────
    rename = {
        'Geocoding Artifact Address Primary Y Coordinate': 'lat',
        'Geocoding Artifact Address Primary X Coordinate': 'lon',
        'State and County Federal Information Processing Standard Code': 'county_fips5_int',
        'Site Name': 'site_name',
        'Site Address': 'address',
        'Site City': 'city',
        'Site State Abbreviation': 'state_abbr',
        'Site Postal Code': 'zip_code',
        'Site Status Description': 'status',
        'Health Center Type': 'hc_type',
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

    # ── Build county FIPS ──────────────────────────────────────────────────────
    if 'county_fips5_int' in df.columns:
        df['county_fips5'] = df['county_fips5_int'].astype(str).str.strip().str.zfill(5)
        df.drop(columns=['county_fips5_int'], inplace=True)

    # ── Coerce coordinates ─────────────────────────────────────────────────────
    for coord in ('lat', 'lon'):
        if coord in df.columns:
            df[coord] = pd.to_numeric(df[coord], errors='coerce')

    df.to_parquet(FQHC_PARQUET, index=False)
    print(f"  Saved site-level -> {FQHC_PARQUET}")

    # ── County-level count ─────────────────────────────────────────────────────
    if 'county_fips5' in df.columns:
        county_counts = (
            df.groupby('county_fips5')
              .size()
              .reset_index(name='fqhc_count')
        )
        county_path = FQHC_PARQUET.parent / 'fqhc_county_counts.parquet'
        county_counts.to_parquet(county_path, index=False)
        print(f"  Saved county counts -> {county_path}")

        # ── Verify MS Delta ────────────────────────────────────────────────────
        delta_fqhc = county_counts[county_counts['county_fips5'].isin(DELTA_COUNTY_FIPS)]
        print(f"\n  MS Delta FQHC counts:")
        print(delta_fqhc.sort_values('county_fips5').to_string(index=False))
        print(f"  Total Delta FQHCs: {delta_fqhc['fqhc_count'].sum()}")

    return df


if __name__ == "__main__":
    ingest_fqhc()
