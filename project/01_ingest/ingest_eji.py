"""
Ingest CDC Environmental Justice Index (EJI) -> Parquet.

Key gotcha: GEOID is int64 -> leading zero dropped -> must zfill(11).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import numpy as np
from config import EJI_PATH, EJI_PARQUET, PROCESSED_RAW

KEEP_COLS = [
    'GEOID',
    # Overall EJI
    'EJI',          # Environmental Justice Index (0–1)
    'RPL_EBM',      # Environmental Burden Module rank
    'RPL_SVM',      # Social Vulnerability Module rank
    'RPL_HVM',      # Health Vulnerability Module rank
    # Environmental burden components
    'RPL_EBM_THEME1',  # Air quality
    'RPL_EBM_THEME2',  # Potentially hazardous / toxic sites
    'RPL_EBM_THEME3',  # Built environment
    # Health vulnerability
    'RPL_HVM_THEME1',  # Health status
    'RPL_HVM_THEME2',  # Physical health
    'RPL_HVM_THEME3',  # Mental health
    # Social vulnerability
    'RPL_SVM_THEME1',  # Socioeconomic status
    'RPL_SVM_THEME2',  # Household characteristics
    'RPL_SVM_THEME3',  # Racial & ethnic minority status
    'RPL_SVM_THEME4',  # Residential characteristics
    # Location
    'STATEFP', 'COUNTYFP', 'STATEABBR', 'COUNTY',
]


def ingest_eji():
    print(f"Reading EJI from {EJI_PATH} ...")
    PROCESSED_RAW.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(EJI_PATH, low_memory=False)
    df.columns = df.columns.str.strip()
    print(f"  Raw shape: {df.shape}")

    # ── Build / clean GEOID ────────────────────────────────────────────────────
    df['GEOID'] = df['GEOID'].astype(str).str.strip().str.zfill(11)

    # ── Keep desired columns ───────────────────────────────────────────────────
    available = [c for c in KEEP_COLS if c in df.columns]
    missing   = [c for c in KEEP_COLS if c not in df.columns]
    if missing:
        print(f"  INFO: columns not found (may be named differently): {missing}")
    df = df[available]

    # ── Replace -999 sentinel ─────────────────────────────────────────────────
    df.replace(-999, np.nan, inplace=True)
    df.replace(-999.0, np.nan, inplace=True)

    df.to_parquet(EJI_PARQUET, index=False)
    print(f"  Saved -> {EJI_PARQUET}")
    print(f"  Shape: {df.shape}")
    return df


if __name__ == "__main__":
    ingest_eji()
