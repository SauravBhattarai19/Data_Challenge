"""
Build the master tract-level dataset by joining all processed sources.

Join strategy:
  - Tract-level (11-digit GEOID): IGS, PLACES, SVI, NRI, EJI, Food Atlas
  - County-level (5-digit prefix): HPSA PC, HPSA MH, MUA, CBP, FQHC counts

Output: data_processed/master_tract.parquet (~84,000 rows × ~80 columns)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import numpy as np
from config import (
    PROCESSED_RAW, PROCESSED, MASTER_TRACT, DELTA_COUNTY_FIPS,
    IGS_LATEST_PARQUET, NRI_PARQUET, PLACES_PARQUET, SVI_PARQUET,
    EJI_PARQUET, HPSA_PC_PARQUET, HPSA_MH_PARQUET, MUA_PARQUET,
    FQHC_PARQUET, CBP_PARQUET, FOOD_PARQUET, AHRF_PARQUET
)

FQHC_COUNTY_PARQUET = PROCESSED_RAW / 'fqhc_county_counts.parquet'


def load_parquet_safe(path: Path, label: str) -> pd.DataFrame | None:
    if not path.exists():
        print(f"  WARNING: {label} parquet not found at {path}")
        return None
    df = pd.read_parquet(path)
    print(f"  Loaded {label}: {df.shape}")
    return df


def build_master_tract():
    PROCESSED.mkdir(parents=True, exist_ok=True)

    # ── Load all sources ───────────────────────────────────────────────────────
    igs    = load_parquet_safe(IGS_LATEST_PARQUET, 'IGS latest')
    nri    = load_parquet_safe(NRI_PARQUET,         'FEMA NRI')
    places = load_parquet_safe(PLACES_PARQUET,      'CDC PLACES')
    svi    = load_parquet_safe(SVI_PARQUET,         'SVI')
    eji    = load_parquet_safe(EJI_PARQUET,         'EJI')
    food   = load_parquet_safe(FOOD_PARQUET,        'Food Atlas')
    hpsa_pc = load_parquet_safe(HPSA_PC_PARQUET,   'HPSA PC')
    hpsa_mh = load_parquet_safe(HPSA_MH_PARQUET,   'HPSA MH')
    mua     = load_parquet_safe(MUA_PARQUET,        'MUA')
    cbp     = load_parquet_safe(CBP_PARQUET,        'CBP')
    fqhc_county = load_parquet_safe(FQHC_COUNTY_PARQUET, 'FQHC counts')
    ahrf = load_parquet_safe(AHRF_PARQUET, 'AHRF workforce')

    if igs is None:
        raise FileNotFoundError("IGS latest parquet is required. Run ingest_igs.py first.")

    # ── Select columns for each source ────────────────────────────────────────
    igs_cols = ['GEOID', 'year', 'igs_score', 'igs_economy', 'igs_place', 'igs_community']
    igs_cols = [c for c in igs_cols if c in igs.columns]

    # ── Start from IGS (broadest tract coverage) ──────────────────────────────
    master = igs[igs_cols].copy()
    print(f"\n  Starting master from IGS: {master.shape}")

    # ── Tract-level joins ──────────────────────────────────────────────────────
    def safe_merge(base, right, label, cols=None):
        if right is None:
            return base
        if cols:
            right = right[[c for c in cols if c in right.columns]]
        # Ensure no duplicate columns beyond GEOID
        overlap = [c for c in right.columns if c in base.columns and c != 'GEOID']
        if overlap:
            right = right.drop(columns=overlap)
        result = base.merge(right, on='GEOID', how='left')
        print(f"  After merge {label}: {result.shape}")
        return result

    if places is not None:
        places_cols = ['GEOID'] + [c for c in places.columns
                                    if c not in ('GEOID','StateAbbr','CountyName')]
        master = safe_merge(master, places[places_cols], 'PLACES')

    if svi is not None:
        svi_cols = ['GEOID', 'RPL_THEMES', 'RPL_THEME1', 'RPL_THEME2', 'RPL_THEME3',
                    'RPL_THEME4', 'F_TOTAL', 'E_TOTPOP', 'E_UNINSUR', 'E_POV150',
                    'STATE', 'COUNTY']
        master = safe_merge(master, svi, 'SVI', svi_cols)

    if nri is not None:
        nri_cols = ['GEOID', 'POPULATION', 'BUILDVALUE', 'AGRIVALUE',
                    'RISK_SCORE', 'RISK_RATNG',
                    'RESL_SCORE', 'RESL_RATNG', 'SOVI_SCORE', 'EAL_SCORE',
                    'HWAV_RISKS', 'HWAV_RISKR', 'HWAV_EALT',
                    'IFLD_RISKS', 'IFLD_RISKR', 'IFLD_EALT',
                    'HRCN_RISKS', 'HRCN_RISKR',
                    'TRND_RISKS', 'TRND_RISKR',
                    'CFLD_RISKS', 'CFLD_RISKR',
                    'DRGT_RISKS', 'DRGT_RISKR',
                    'WFIR_RISKS', 'WFIR_RISKR']
        master = safe_merge(master, nri, 'NRI', nri_cols)

    if eji is not None:
        eji_cols = ['GEOID', 'EJI', 'RPL_EBM', 'RPL_SVM', 'RPL_HVM', 'STATEABBR']
        master = safe_merge(master, eji, 'EJI', eji_cols)

    if food is not None:
        food_cols = ['GEOID', 'LILATracts_1And10', 'PovertyRate',
                     'MedianFamilyIncome', 'LALOWI1_10', 'TractSNAP', 'Urban']
        master = safe_merge(master, food, 'Food', food_cols)

    # ── County-level joins ─────────────────────────────────────────────────────
    master['county_fips5'] = master['GEOID'].str[:5]

    def safe_county_merge(base, right, label):
        if right is None:
            return base
        # Ensure no duplicate columns beyond county_fips5
        overlap = [c for c in right.columns if c in base.columns and c != 'county_fips5']
        if overlap:
            right = right.drop(columns=overlap)
        result = base.merge(right, on='county_fips5', how='left')
        print(f"  After county merge {label}: {result.shape}")
        return result

    master = safe_county_merge(master, hpsa_pc,     'HPSA PC')
    master = safe_county_merge(master, hpsa_mh,     'HPSA MH')
    master = safe_county_merge(master, mua,         'MUA')
    master = safe_county_merge(master, cbp,         'CBP')
    master = safe_county_merge(master, fqhc_county, 'FQHC county counts')
    master = safe_county_merge(master, ahrf,        'AHRF workforce')

    # ── Fill boolean flags ─────────────────────────────────────────────────────
    bool_cols = ['in_pc_hpsa', 'in_mh_hpsa', 'in_mua']
    for col in bool_cols:
        if col in master.columns:
            master[col] = master[col].fillna(False).astype(bool)

    # ── Add state abbreviation / county name helpers ───────────────────────────
    master['state_fips2'] = master['GEOID'].str[:2]
    master['is_delta']    = master['county_fips5'].isin(DELTA_COUNTY_FIPS)

    # ── Quality check ──────────────────────────────────────────────────────────
    print(f"\n  FINAL master shape: {master.shape}")
    print(f"  Null rates (key columns):")
    key_cols = ['igs_score', 'RISK_SCORE', 'RPL_THEMES']
    for col in key_cols:
        if col in master.columns:
            print(f"    {col}: {master[col].isna().mean()*100:.1f}%")

    delta_count = master['is_delta'].sum()
    print(f"  MS Delta tracts: {delta_count}")

    master.to_parquet(MASTER_TRACT, index=False)
    print(f"\n  Saved -> {MASTER_TRACT}")
    return master


if __name__ == "__main__":
    build_master_tract()
