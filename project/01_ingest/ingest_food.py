"""
Ingest USDA Food Access Research Atlas 2019 -> Parquet.

Key gotcha: CensusTract is int -> leading zero dropped -> zfill(11).
"""

import sys
import zipfile
import io
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from config import FOOD_ATLAS_ZIP, FOOD_PARQUET, PROCESSED_RAW

KEEP_COLS = [
    'GEOID',
    'LILATracts_1And10',   # 1-mile / 10-mile food desert flag (primary)
    'LILATracts_halfAnd10',
    'LILATracts_1And20',
    'PovertyRate',
    'MedianFamilyIncome',
    'LALOWI1_10',          # Low income + low access, 1 mile + 10 miles
    'lapop1_10',           # Population with low access (1+10)
    'TractKids',
    'TractSeniors',
    'TractWhite',
    'TractBlack',
    'TractHispanic',
    'TractSNAP',           # SNAP households
    'Urban',
]


def ingest_food():
    print(f"Reading Food Atlas from {FOOD_ATLAS_ZIP} ...")
    PROCESSED_RAW.mkdir(parents=True, exist_ok=True)

    # ── Read from zip ──────────────────────────────────────────────────────────
    with zipfile.ZipFile(FOOD_ATLAS_ZIP, 'r') as z:
        print(f"  Files in zip: {z.namelist()}")
        # Find the data file (xlsx or csv)
        data_files = [n for n in z.namelist()
                      if n.endswith('.csv') or n.endswith('.xlsx')]
        if not data_files:
            data_files = z.namelist()
        fname = data_files[0]
        print(f"  Reading: {fname}")
        with z.open(fname) as f:
            if fname.endswith('.xlsx'):
                df = pd.read_excel(f, dtype={'CensusTract': str})
            else:
                df = pd.read_csv(f, dtype={'CensusTract': str}, low_memory=False)

    df.columns = df.columns.str.strip()
    print(f"  Raw shape: {df.shape}")

    # ── Build GEOID ────────────────────────────────────────────────────────────
    tract_col = next(
        (c for c in df.columns if 'censustract' in c.lower() or c.lower() == 'tract'),
        None
    )
    if tract_col:
        df['GEOID'] = df[tract_col].astype(str).str.strip().str.zfill(11)
    else:
        raise ValueError(f"Cannot find tract FIPS column. Columns: {list(df.columns[:10])}")

    # ── Keep desired columns ───────────────────────────────────────────────────
    available = [c for c in KEEP_COLS if c in df.columns]
    missing   = [c for c in KEEP_COLS if c not in df.columns]
    if missing:
        print(f"  INFO: columns not found: {missing}")
    df = df[available]

    # ── Coerce types ───────────────────────────────────────────────────────────
    for col in df.columns:
        if col != 'GEOID':
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df.to_parquet(FOOD_PARQUET, index=False)
    print(f"  Saved -> {FOOD_PARQUET}")
    print(f"  Shape: {df.shape}")
    print(f"  Food desert tracts (LILATracts_1And10==1): "
          f"{df['LILATracts_1And10'].sum():.0f}" if 'LILATracts_1And10' in df.columns else "")
    return df


if __name__ == "__main__":
    ingest_food()
