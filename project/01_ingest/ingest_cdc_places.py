"""
Ingest CDC PLACES 2025 census tract data -> Parquet.

Key gotchas:
  - TractFIPS is int64 -> must zfill(11)
  - Column name 'MOBILITY_CrudePrev\t' has trailing tab -> strip all col names
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from config import CDC_PLACES_PATH, PLACES_PARQUET, PROCESSED_RAW

# Columns to keep (stripped names, subset of the full ~40-col file)
KEEP_COLS = [
    'GEOID',
    # Core health outcomes
    'CASTHMA_CrudePrev',   # Asthma
    'CHD_CrudePrev',       # Coronary Heart Disease
    'DIABETES_CrudePrev',  # Diabetes
    'BPHIGH_CrudePrev',    # High Blood Pressure
    'COPD_CrudePrev',      # COPD
    'DEPRESSION_CrudePrev',# Depression
    'MHLTH_CrudePrev',     # Mental Health (poor days)
    'OBESITY_CrudePrev',   # Obesity
    'STROKE_CrudePrev',    # Stroke
    # Prevention / access indicators
    'ACCESS2_CrudePrev',   # Uninsured
    'CHECKUP_CrudePrev',   # Annual checkup
    'DENTAL_CrudePrev',    # Dental visit
    'MAMMOUSE_CrudePrev',  # Mammography use
    'COLON_SCREEN_CrudePrev', # Colorectal cancer screening
    # Unhealthy behaviors
    'CSMOKING_CrudePrev',  # Smoking
    'BINGE_CrudePrev',     # Binge drinking
    'LPA_CrudePrev',       # Physical inactivity
    'SLEEP_CrudePrev',     # Sleep <7 hrs
    # Mobility (col has trailing tab in raw — handled below)
    'MOBILITY_CrudePrev',
    # Location
    'StateAbbr', 'CountyName', 'TotalPopulation',
]


def ingest_cdc_places():
    print(f"Reading CDC PLACES from {CDC_PLACES_PATH} ...")
    PROCESSED_RAW.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(CDC_PLACES_PATH, low_memory=False)

    # CRITICAL: strip trailing whitespace / tabs from all column names
    df.columns = df.columns.str.strip()
    print(f"  Raw shape: {df.shape}")

    # ── Build GEOID ────────────────────────────────────────────────────────────
    df['GEOID'] = df['TractFIPS'].astype(str).str.zfill(11)

    # ── Keep only desired columns (those that exist) ───────────────────────────
    available = [c for c in KEEP_COLS if c in df.columns]
    missing   = [c for c in KEEP_COLS if c not in df.columns]
    if missing:
        print(f"  WARNING: expected columns not found: {missing}")
    df = df[available]

    # ── Coerce prevalence columns to float ────────────────────────────────────
    prev_cols = [c for c in df.columns if 'Prev' in c or 'Population' in c.lower()]
    for col in prev_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df.to_parquet(PLACES_PARQUET, index=False)
    print(f"  Saved -> {PLACES_PARQUET}")
    print(f"  Shape: {df.shape}")
    print(f"  Null rate on CASTHMA_CrudePrev: "
          f"{df['CASTHMA_CrudePrev'].isna().mean()*100:.1f}%")
    return df


if __name__ == "__main__":
    ingest_cdc_places()
