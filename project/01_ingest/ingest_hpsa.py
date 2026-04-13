"""
Ingest HRSA HPSA (Health Professional Shortage Area) data -> county-level Parquets.

Processes both Primary Care (PC) and Mental Health (MH) files.
Aggregates to one row per county FIPS with:
  - is_hpsa (bool)
  - max HPSA score
  - total shortage (shortage units)
  - count of HPSA designations

Key gotchas:
  - Filter HPSA Status == 'Designated' (removes withdrawn/proposed)
  - County FIPS must be reconstructed from State + County sub-fields
  - All 9 MS Delta counties are confirmed active PC HPSAs
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from config import (
    HPSA_PC_PATH, HPSA_MH_PATH,
    HPSA_PC_PARQUET, HPSA_MH_PARQUET,
    PROCESSED_RAW, DELTA_COUNTY_FIPS
)


def _build_county_fips(df: pd.DataFrame) -> pd.Series:
    """
    Reconstruct 5-digit county FIPS from HPSA data columns.
    Tries multiple known column-name patterns.
    """
    state_candidates = [
        'Primary State FIPS Code',
        'State FIPS Code',
        'State Fips',
        'STATFIPS',
    ]
    county_candidates = [
        'Primary County FIPS Code',
        'County FIPS Code',
        'County Fips',
        'CNTYFIPS',
    ]

    state_col  = next((c for c in state_candidates  if c in df.columns), None)
    county_col = next((c for c in county_candidates if c in df.columns), None)

    if state_col and county_col:
        state  = df[state_col].astype(str).str.strip().str.zfill(2)
        county = df[county_col].astype(str).str.strip().str.zfill(3)
        return state + county

    # Fallback: look for a pre-built FIPS-like column
    for c in df.columns:
        if 'fips' in c.lower() and df[c].astype(str).str.len().median() >= 4:
            return df[c].astype(str).str.strip().str.zfill(5)

    raise ValueError(f"Cannot build county FIPS. Available columns: {list(df.columns)}")


def _process_hpsa(path: Path, label: str) -> pd.DataFrame:
    print(f"  Reading {label} HPSA from {path.name} ...")
    df = pd.read_csv(path, low_memory=False, encoding='latin-1')
    df.columns = df.columns.str.strip()
    print(f"    Raw shape: {df.shape}")

    # ── Filter active designations ─────────────────────────────────────────────
    status_col = next(
        (c for c in df.columns if 'status' in c.lower() and 'hpsa' in c.lower()), None
    )
    if status_col:
        df = df[df[status_col].str.strip() == 'Designated']
        print(f"    After status filter: {len(df):,} rows")
    else:
        print(f"    WARNING: No HPSA Status column found. Columns: {list(df.columns[:10])}")

    # ── Build county FIPS ──────────────────────────────────────────────────────
    df['county_fips5'] = _build_county_fips(df)

    # ── Score column ───────────────────────────────────────────────────────────
    score_col = next(
        (c for c in df.columns if 'hpsa score' in c.lower() or 'score' in c.lower()),
        None
    )
    shortage_col = next(
        (c for c in df.columns if 'shortage' in c.lower()),
        None
    )
    geo_col = next(
        (c for c in df.columns if 'geographic' in c.lower() and 'type' in c.lower()),
        None
    )

    agg_dict = {'is_hpsa': ('county_fips5', 'count')}
    if score_col:
        df[score_col] = pd.to_numeric(df[score_col], errors='coerce')
        agg_dict['hpsa_score_max']  = (score_col, 'max')
        agg_dict['hpsa_score_mean'] = (score_col, 'mean')
    if shortage_col:
        df[shortage_col] = pd.to_numeric(df[shortage_col], errors='coerce')
        agg_dict['hpsa_shortage_total'] = (shortage_col, 'sum')

    agg = df.groupby('county_fips5').agg(**agg_dict).reset_index()
    agg['is_hpsa'] = agg['is_hpsa'] > 0

    # Rename generic score columns to be label-specific
    agg = agg.rename(columns={
        'hpsa_score_max':      f'{label}_hpsa_score_max',
        'hpsa_score_mean':     f'{label}_hpsa_score_mean',
        'hpsa_shortage_total': f'{label}_hpsa_shortage_total',
        'is_hpsa':             f'in_{label}_hpsa',
    })

    print(f"    County-level output: {len(agg):,} counties")
    return agg


def ingest_hpsa():
    PROCESSED_RAW.mkdir(parents=True, exist_ok=True)

    pc = _process_hpsa(HPSA_PC_PATH, 'pc')
    pc.to_parquet(HPSA_PC_PARQUET, index=False)
    print(f"  Saved PC -> {HPSA_PC_PARQUET}")

    mh = _process_hpsa(HPSA_MH_PATH, 'mh')
    mh.to_parquet(HPSA_MH_PARQUET, index=False)
    print(f"  Saved MH -> {HPSA_MH_PARQUET}")

    # ── Verify MS Delta presence ───────────────────────────────────────────────
    delta_pc = pc[pc['county_fips5'].isin(DELTA_COUNTY_FIPS)]
    print(f"\n  MS Delta counties in PC HPSA: {len(delta_pc)}/9")
    print(delta_pc[['county_fips5', 'in_pc_hpsa']].to_string(index=False))

    return pc, mh


if __name__ == "__main__":
    ingest_hpsa()
