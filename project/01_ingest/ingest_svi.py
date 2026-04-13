"""
Ingest CDC/ATSDR Social Vulnerability Index 2022 -> Parquet.

Key gotchas:
  - FIPS stored as int64 -> leading zero dropped -> must zfill(11)
  - -999 is the missing-value sentinel -> replace with NaN
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import numpy as np
from config import SVI_PATH, SVI_PARQUET, PROCESSED_RAW

# SVI columns to retain
KEEP_COLS = [
    'FIPS',
    # Summary theme percentile ranks (core SVI outputs)
    'RPL_THEMES',   # Overall SVI (0–1, higher = more vulnerable)
    'RPL_THEME1',   # Socioeconomic
    'RPL_THEME2',   # Household characteristics
    'RPL_THEME3',   # Racial & ethnic minority status
    'RPL_THEME4',   # Housing type & transportation
    # Flags
    'F_TOTAL',      # Count of high-vulnerability flags (0–16)
    # Key component variables
    'E_TOTPOP',     # Total population
    'E_POV150',     # Persons below 150% poverty
    'E_UNEMP',      # Unemployed
    'E_NOHSDP',     # No high school diploma
    'E_UNINSUR',    # Uninsured
    'E_AGE65',      # Aged 65+
    'E_AGE17',      # Aged 17 and younger
    'E_DISABL',     # Persons with disability
    'E_SNGPNT',     # Single-parent households
    'E_MINRTY',     # Minority status
    'E_LIMENG',     # Limited English proficiency
    'E_MUNIT',      # Multi-unit structures
    'E_MOBILE',     # Mobile homes
    'E_CROWD',      # Crowding (>1 person per room)
    'E_NOVEH',      # No vehicle
    'E_GROUPQ',     # Group quarters
    # Location
    'STATE', 'COUNTY', 'LOCATION',
]


def ingest_svi():
    print(f"Reading SVI from {SVI_PATH} ...")
    PROCESSED_RAW.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(SVI_PATH, low_memory=False, dtype={'FIPS': str})
    print(f"  Raw shape: {df.shape}")

    # ── Build GEOID ────────────────────────────────────────────────────────────
    df['GEOID'] = df['FIPS'].astype(str).str.strip().str.zfill(11)

    # ── Keep desired columns ───────────────────────────────────────────────────
    available = [c for c in KEEP_COLS if c in df.columns]
    df = df[available + ['GEOID']]

    # ── Replace -999 sentinel with NaN ────────────────────────────────────────
    numeric_cols = df.select_dtypes(include='number').columns
    df[numeric_cols] = df[numeric_cols].replace(-999, np.nan)

    # ── Also replace -999.0 in float cols produced after read ─────────────────
    df.replace(-999.0, np.nan, inplace=True)

    df.to_parquet(SVI_PARQUET, index=False)
    print(f"  Saved -> {SVI_PARQUET}")
    print(f"  Shape: {df.shape}")
    print(f"  RPL_THEMES range: "
          f"{df['RPL_THEMES'].min():.3f} – {df['RPL_THEMES'].max():.3f}")
    return df


if __name__ == "__main__":
    ingest_svi()
