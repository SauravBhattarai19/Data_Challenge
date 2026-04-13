"""
Ingest HRSA Medically Underserved Areas/Populations (MUA/MUP) -> county-level Parquet.

Key gotchas:
  - Filter 'MUA/P Status' == 'Designated'
  - All 9 MS Delta counties are county-level MUAs (IMU range 19.7–39.8)
  - County FIPS must be reconstructed from state/county sub-fields
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from config import MUA_PATH, MUA_PARQUET, PROCESSED_RAW, DELTA_COUNTY_FIPS


def ingest_mua():
    print(f"Reading MUA from {MUA_PATH} ...")
    PROCESSED_RAW.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(MUA_PATH, low_memory=False, encoding='latin-1')
    df.columns = df.columns.str.strip()
    print(f"  Raw shape: {df.shape}")
    print(f"  Columns: {list(df.columns[:15])}")

    # ── Filter active designations ─────────────────────────────────────────────
    # Prefer the Description column (not the Code column)
    status_col = next(
        (c for c in df.columns if 'status description' in c.lower()), None
    ) or next(
        (c for c in df.columns if 'status' in c.lower() and 'code' not in c.lower()), None
    )
    if status_col:
        df = df[df[status_col].str.strip() == 'Designated']
        print(f"  After status filter ({status_col}): {len(df):,} rows")

    # ── Build county FIPS ──────────────────────────────────────────────────────
    # Prefer the 5-digit combined column if present
    combined_col = next(
        (c for c in df.columns
         if ('state and county' in c.lower() or 'county federal' in c.lower())
         and 'fips' in c.lower()),
        None
    )
    state_col  = next((c for c in df.columns if 'primary state fips' in c.lower()), None) or \
                 next((c for c in df.columns if c.lower() == 'state fips code'), None)
    county_col = next((c for c in df.columns
                       if 'county or county equivalent federal' in c.lower()), None)

    if combined_col:
        df['county_fips5'] = df[combined_col].astype(str).str.strip().str.zfill(5)
        print(f"  Using combined FIPS column: {combined_col}")
    elif state_col and county_col:
        df['county_fips5'] = (
            df[state_col].astype(str).str.strip().str.zfill(2) +
            df[county_col].astype(str).str.strip().str.zfill(3)
        )
    else:
        # Last fallback: find any 5-digit FIPS-like column
        for c in df.columns:
            sample = df[c].dropna().astype(str).str.strip()
            if 'fips' in c.lower() and sample.str.len().median() >= 4:
                df['county_fips5'] = sample.str.zfill(5)
                print(f"  Using fallback FIPS column: {c}")
                break

    # ── IMU Score ─────────────────────────────────────────────────────────────
    imu_col = next((c for c in df.columns if 'imu' in c.lower()), None)
    mua_type_col = next((c for c in df.columns if 'type' in c.lower()), None)

    agg_dict = {'in_mua': ('county_fips5', 'count')}
    if imu_col:
        df[imu_col] = pd.to_numeric(df[imu_col], errors='coerce')
        agg_dict['imu_score_min'] = (imu_col, 'min')
        agg_dict['imu_score_mean'] = (imu_col, 'mean')

    agg = df.groupby('county_fips5').agg(**agg_dict).reset_index()
    agg['in_mua'] = agg['in_mua'] > 0

    print(f"  County-level output: {len(agg):,} counties")

    # ── Verify MS Delta ────────────────────────────────────────────────────────
    delta_mua = agg[agg['county_fips5'].isin(DELTA_COUNTY_FIPS)]
    print(f"\n  MS Delta counties in MUA: {len(delta_mua)}/9")
    if 'imu_score_min' in agg.columns:
        print(delta_mua[['county_fips5','in_mua','imu_score_min']].to_string(index=False))
    else:
        print(delta_mua[['county_fips5','in_mua']].to_string(index=False))

    agg.to_parquet(MUA_PARQUET, index=False)
    print(f"\n  Saved -> {MUA_PARQUET}")
    return agg


if __name__ == "__main__":
    ingest_mua()
