"""
Ingest AHRF 2024-2025 county-level healthcare workforce data -> Parquet.

Extracts from multiple sub-CSVs within the zip archive:
  - HP: Physician counts (total, primary care, psychiatry, family medicine)
  - HF: Hospital counts and bed counts
  - EXP: Medicare FFS per-capita costs
  - POP: County population (for per-capita calculations)

Output: data_processed/raw/ahrf_county.parquet
  One row per county FIPS (~3,200 counties) with physician counts per capita,
  hospital beds, and Medicare spending.

Run:
  python 01_ingest/ingest_ahrf.py
"""

import sys
import zipfile
import io
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import numpy as np
from config import AHRF_ZIP_PATH, AHRF_PARQUET, PROCESSED_RAW

# ── Files inside the zip ─────────────────────────────────────────────────────
ZIP_PREFIX = 'NCHWA-2024-2025+AHRF+COUNTY+CSV/'

# ── Columns to extract from each sub-file ────────────────────────────────────
HP_COLS = {
    'fips_st_cnty': str,
    'phys_nf_prim_care_pc_exc_rsdt_23': float,  # Primary care physicians (exc residents)
    'md_nf_prim_care_pc_excl_rsdnt_23': float,   # MDs in primary care
    'md_nf_psych_23': float,                      # Psychiatrists
    'md_nf_fammed_gen_23': float,                 # Family medicine / general practice
}

HF_COLS = {
    'fips_st_cnty': str,
    'hosp_23': float,           # Total hospitals
    'hosp_beds_23': float,      # Total hospital beds
    'stgh_hosp_beds_23': float, # Short-term general hospital beds
    'critcl_access_hosp_23': float,  # Critical access hospitals
}

EXP_COLS = {
    'fips_st_cnty': str,
    'actl_per_cap_ffs_cost_23': float,     # Actual per-capita FFS cost
    'stdizd_per_cap_ffs_cost_23': float,   # Standardized per-capita FFS cost
    'medcr_ffs_inpat_covrd_stay_23': float,  # Medicare inpatient covered stays
}

POP_COLS = {
    'fips_st_cnty': str,
    'cnty_name_st_abbrev': str,
    'popn_est_24': float,  # 2024 population estimate
    'popn_23': float,      # 2023 population
}


def _read_sub_csv(zf: zipfile.ZipFile, filename: str, usecols: dict) -> pd.DataFrame:
    """Read a sub-CSV from the zip, selecting only needed columns."""
    full_path = ZIP_PREFIX + filename
    with zf.open(full_path) as f:
        df = pd.read_csv(
            io.TextIOWrapper(f, encoding='latin-1'),
            usecols=list(usecols.keys()),
            dtype={'fips_st_cnty': str},
            low_memory=False,
        )
    # Coerce numeric columns
    for col, dtype in usecols.items():
        if dtype == float and col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    print(f"    {filename}: {df.shape}")
    return df


def ingest_ahrf():
    print(f"Ingesting AHRF from: {AHRF_ZIP_PATH}")
    PROCESSED_RAW.mkdir(parents=True, exist_ok=True)

    if not AHRF_ZIP_PATH.exists():
        print(f"  ERROR: File not found: {AHRF_ZIP_PATH}")
        return None

    with zipfile.ZipFile(AHRF_ZIP_PATH) as zf:
        print("  Reading sub-files...")
        hp  = _read_sub_csv(zf, 'AHRF2025hp.csv',  HP_COLS)
        hf  = _read_sub_csv(zf, 'AHRF2025hf.csv',  HF_COLS)
        exp = _read_sub_csv(zf, 'AHRF2025exp.csv',  EXP_COLS)
        pop = _read_sub_csv(zf, 'AHRF2025pop.csv',  POP_COLS)

    # ── Merge all on county FIPS ─────────────────────────────────────────────
    df = pop.merge(hp, on='fips_st_cnty', how='left') \
            .merge(hf, on='fips_st_cnty', how='left') \
            .merge(exp, on='fips_st_cnty', how='left')
    print(f"  Merged shape: {df.shape}")

    # ── Compute per-capita metrics ───────────────────────────────────────────
    pop_col = 'popn_est_24'
    df[pop_col] = df[pop_col].fillna(df['popn_23'])  # fallback to 2023 if 2024 missing
    pop_safe = df[pop_col].replace(0, np.nan)

    df['ahrf_phys_per_10k'] = df['phys_nf_prim_care_pc_exc_rsdt_23'].add(
        df['md_nf_fammed_gen_23'].fillna(0)
    ) / pop_safe * 10_000

    df['ahrf_pc_phys_per_10k'] = df['phys_nf_prim_care_pc_exc_rsdt_23'] / pop_safe * 10_000
    df['ahrf_psych_per_10k'] = df['md_nf_psych_23'] / pop_safe * 10_000
    df['ahrf_beds_per_10k'] = df['hosp_beds_23'] / pop_safe * 10_000

    # ── Rename and clean ─────────────────────────────────────────────────────
    df = df.rename(columns={
        'fips_st_cnty': 'county_fips5',
        'cnty_name_st_abbrev': 'ahrf_county_name',
        'popn_est_24': 'ahrf_population',
    })
    df['county_fips5'] = df['county_fips5'].str.strip().str.zfill(5)

    # ── Select output columns ────────────────────────────────────────────────
    out_cols = [
        'county_fips5', 'ahrf_county_name', 'ahrf_population',
        'phys_nf_prim_care_pc_exc_rsdt_23', 'md_nf_psych_23',
        'md_nf_fammed_gen_23',
        'hosp_23', 'hosp_beds_23', 'critcl_access_hosp_23',
        'actl_per_cap_ffs_cost_23', 'stdizd_per_cap_ffs_cost_23',
        'ahrf_phys_per_10k', 'ahrf_pc_phys_per_10k',
        'ahrf_psych_per_10k', 'ahrf_beds_per_10k',
    ]
    out_cols = [c for c in out_cols if c in df.columns]
    df = df[out_cols]

    df.to_parquet(AHRF_PARQUET, index=False)
    print(f"  Saved -> {AHRF_PARQUET}")
    print(f"  Shape: {df.shape}")
    print(f"  Sample (first 5):")
    print(df.head().to_string())
    return df


if __name__ == "__main__":
    ingest_ahrf()
