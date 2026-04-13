"""
Ingest IGS Excel file -> Parquet.

Sheet: "Compared to Urban-Rural"
Header structure: row 0 = pillar groups, row 1 = column names, row 2 = blank.
Data starts at row 3 (skiprows=3).
FIPS is column index 2, already an 11-digit string.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from config import IGS_PATH, IGS_PARQUET, IGS_LATEST_PARQUET, PROCESSED_RAW

SHEET = "Compared to Urban-Rural"


def ingest_igs():
    print(f"Reading IGS from {IGS_PATH} ...")
    PROCESSED_RAW.mkdir(parents=True, exist_ok=True)

    xl = pd.ExcelFile(IGS_PATH, engine="openpyxl")

    # ── Read the two header rows to recover column names ──────────────────────
    header_df = xl.parse(SHEET, header=None, nrows=2)
    col_names = header_df.iloc[1].tolist()   # row 1 = actual column labels

    # ── Read data (skip the 3 header rows) ────────────────────────────────────
    # Column index 2 is Census Tract FIPS -> force string to preserve leading zeros
    df = xl.parse(
        SHEET,
        header=None,
        skiprows=3,
        dtype={2: str},
    )
    df.columns = col_names

    print(f"  Raw shape: {df.shape}")
    print(f"  Columns (first 10): {list(df.columns[:10])}")

    # ── Rename key columns ────────────────────────────────────────────────────
    rename_map = {}
    for c in df.columns:
        cl = str(c).strip().lower()
        if 'fips' in cl or 'census tract' in cl:
            rename_map[c] = 'GEOID'
        elif cl == 'year':
            rename_map[c] = 'year'
        elif 'inclusive growth score' in cl:
            rename_map[c] = 'igs_score'
        elif cl == 'place':
            rename_map[c] = 'igs_place'
        elif cl == 'economy':
            rename_map[c] = 'igs_economy'
        elif cl == 'community':
            rename_map[c] = 'igs_community'
    df = df.rename(columns=rename_map)
    print(f"  Renamed columns applied. GEOID present: {'GEOID' in df.columns}")

    # ── Clean GEOID ───────────────────────────────────────────────────────────
    df['GEOID'] = df['GEOID'].astype(str).str.strip().str.zfill(11)

    # ── Filter valid years ────────────────────────────────────────────────────
    df['year'] = pd.to_numeric(df['year'], errors='coerce')
    df = df.dropna(subset=['year', 'GEOID'])
    df = df[df['year'].between(2017, 2025)]
    df['year'] = df['year'].astype(int)

    print(f"  Years present: {sorted(df['year'].unique())}")
    print(f"  Tracts per year:\n{df.groupby('year')['GEOID'].nunique()}")
    print(f"  Filtered shape: {df.shape}")

    # ── Save all years ────────────────────────────────────────────────────────
    df.to_parquet(IGS_PARQUET, index=False)
    print(f"  Saved all-years -> {IGS_PARQUET}")

    # ── Save latest year per tract ────────────────────────────────────────────
    latest_year = df.groupby('GEOID')['year'].transform('max')
    igs_latest = df[df['year'] == latest_year].copy()
    igs_latest.to_parquet(IGS_LATEST_PARQUET, index=False)
    print(f"  Saved latest-year snapshot -> {IGS_LATEST_PARQUET}")
    print(f"  Latest snapshot shape: {igs_latest.shape}")

    return df


if __name__ == "__main__":
    ingest_igs()
