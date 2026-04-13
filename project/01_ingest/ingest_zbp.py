"""
Ingest ZIP Business Patterns for Delta-area ZIP codes.

Reads: zbp23totals.zip (ZIP-level employment, payroll, establishments)
       zbp23detail.zip (ZIP-level industry breakdown)
Writes: zbp_delta_zips.parquet

Focuses on Mississippi ZIP codes in/near the Delta for granular analysis.
"""

import sys
import zipfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import numpy as np
from config import ZBP_TOTALS_ZIP, ZBP_DETAIL_ZIP, ZBP_PARQUET, PROCESSED_RAW


# Mississippi Delta ZIP codes (covering the 9-county region and nearby)
# These ZIPs fall within or overlap Bolivar, Coahoma, Humphreys, Issaquena,
# Leflore, Quitman, Sharkey, Sunflower, Washington counties
DELTA_ZIPS = [
    # Bolivar: Cleveland, Rosedale, Mound Bayou, Shelby, Merigold, Boyle, Benoit, Beulah
    '38732', '38769', '38762', '38774', '38759', '38730', '38725', '38726',
    # Coahoma: Clarksdale, Coahoma, Friars Point, Jonestown, Lula, Lyon
    '38614', '38617', '38631', '38639', '38644', '38645',
    # Humphreys: Belzoni, Isola, Louise
    '39038', '38754', '39097',
    # Issaquena: Mayersville, Valley Park
    '39113', '39177',
    # Leflore: Greenwood, Itta Bena, Sidon, Schlater, Morgan City
    '38930', '38941', '38954', '38952', '38946',
    # Quitman: Marks, Lambert, Crowder
    '38646', '38643', '38622',
    # Sharkey: Rolling Fork, Anguilla, Cary
    '39159', '38721', '39054',
    # Sunflower: Indianola, Ruleville, Moorhead, Sunflower, Drew, Inverness
    '38751', '38771', '38761', '38778', '38737', '38753',
    # Washington: Greenville, Leland, Hollandale, Avon, Glen Allan, Winterville
    '38701', '38702', '38703', '38756', '38748', '38723', '38744', '38782',
]
# Also include all MS ZIPs for state-level context — we'll filter after loading


def ingest_zbp():
    print(f"Reading ZIP Business Patterns ...")
    PROCESSED_RAW.mkdir(parents=True, exist_ok=True)

    # ── 1. ZBP Totals (employment, payroll, establishments per ZIP) ──────────
    print(f"\n  Reading ZBP totals from {ZBP_TOTALS_ZIP} ...")
    with zipfile.ZipFile(ZBP_TOTALS_ZIP, 'r') as z:
        txt_files = [n for n in z.namelist() if n.endswith('.txt')]
        fname = txt_files[0]
        print(f"    File: {fname}")
        with z.open(fname) as f:
            totals = pd.read_csv(f, dtype={'zip': str}, low_memory=False, encoding='latin-1')

    totals.columns = totals.columns.str.strip().str.lower()
    totals['zip'] = totals['zip'].astype(str).str.zfill(5)
    print(f"    Raw ZBP totals: {len(totals):,} ZIP codes")

    # Filter to Mississippi
    ms_totals = totals[totals['stabbr'] == 'MS'].copy()
    print(f"    Mississippi ZIPs: {len(ms_totals)}")

    # ── 2. ZBP Detail (industry breakdown per ZIP) ───────────────────────────
    print(f"\n  Reading ZBP detail from {ZBP_DETAIL_ZIP} ...")
    with zipfile.ZipFile(ZBP_DETAIL_ZIP, 'r') as z:
        txt_files = [n for n in z.namelist() if n.endswith('.txt')]
        fname = txt_files[0]
        print(f"    File: {fname}")
        with z.open(fname) as f:
            detail = pd.read_csv(f, dtype={'zip': str, 'naics': str},
                                low_memory=False, encoding='latin-1')

    detail.columns = detail.columns.str.strip().str.lower()
    detail['zip'] = detail['zip'].astype(str).str.zfill(5)

    # Filter to Mississippi
    ms_detail = detail[detail['stabbr'] == 'MS'].copy()
    print(f"    MS industry rows: {len(ms_detail):,}")

    # ── 3. Compute per-ZIP business metrics ──────────────────────────────────
    # Sector diversity: count distinct 2-digit NAICS sectors
    ms_detail['naics_clean'] = ms_detail['naics'].str.strip().str.replace('/', '', regex=False)
    sector_rows = ms_detail[ms_detail['naics_clean'].str.match(r'^\d{2}----$', na=False)]
    sector_diversity = sector_rows.groupby('zip')['naics_clean'].nunique().reset_index()
    sector_diversity.columns = ['zip', 'n_sectors']

    # Healthcare establishments (NAICS 62----)
    healthcare_rows = ms_detail[ms_detail['naics_clean'].str.startswith('62')]
    healthcare_6digit = healthcare_rows[healthcare_rows['naics_clean'].str.match(r'^\d{6}$', na=False)]
    healthcare_est = healthcare_6digit.groupby('zip')['est'].sum().reset_index()
    healthcare_est.columns = ['zip', 'healthcare_establishments']

    # Small businesses (from size bins in totals)
    for col in ['n<5', 'n5_9', 'n10_19']:
        if col in ms_totals.columns:
            ms_totals[col] = pd.to_numeric(
                ms_totals[col].astype(str).str.replace('N', '0'), errors='coerce'
            ).fillna(0)

    size_cols = ['n<5', 'n5_9', 'n10_19']
    available_size = [c for c in size_cols if c in ms_totals.columns]
    if available_size:
        ms_totals['small_biz_count'] = ms_totals[available_size].sum(axis=1)
    else:
        ms_totals['small_biz_count'] = 0

    ms_totals['emp'] = pd.to_numeric(ms_totals['emp'], errors='coerce').fillna(0)
    ms_totals['est'] = pd.to_numeric(ms_totals['est'], errors='coerce').fillna(0)
    ms_totals['ap'] = pd.to_numeric(ms_totals['ap'], errors='coerce').fillna(0)

    # ── 4. Merge everything ──────────────────────────────────────────────────
    result = ms_totals[['zip', 'name', 'city', 'cty_name', 'emp', 'est', 'ap', 'small_biz_count']].copy()
    result = result.merge(sector_diversity, on='zip', how='left')
    result = result.merge(healthcare_est, on='zip', how='left')

    result['n_sectors'] = result['n_sectors'].fillna(0).astype(int)
    result['healthcare_establishments'] = result['healthcare_establishments'].fillna(0).astype(int)
    result['small_biz_pct'] = np.where(
        result['est'] > 0,
        (result['small_biz_count'] / result['est'] * 100).round(1),
        0
    )
    result['avg_emp_per_est'] = np.where(
        result['est'] > 0,
        (result['emp'] / result['est']).round(1),
        0
    )
    result['annual_payroll_per_emp'] = np.where(
        result['emp'] > 0,
        (result['ap'] * 1000 / result['emp']).round(0),
        0
    )

    # Flag Delta ZIPs
    result['is_delta'] = result['zip'].isin(DELTA_ZIPS)

    result.to_parquet(ZBP_PARQUET, index=False)
    print(f"\n  Saved {len(result)} MS ZIPs -> {ZBP_PARQUET}")
    print(f"  Delta ZIPs found: {result['is_delta'].sum()}")

    # Delta summary
    delta_data = result[result['is_delta']]
    if len(delta_data) > 0:
        print(f"\n  Delta ZIP summary:")
        print(f"    Total establishments: {delta_data['est'].sum():,.0f}")
        print(f"    Total employment: {delta_data['emp'].sum():,.0f}")
        print(f"    Mean sectors per ZIP: {delta_data['n_sectors'].mean():.1f}")
        print(f"    Mean small biz %: {delta_data['small_biz_pct'].mean():.1f}%")
        print(f"    Healthcare establishments: {delta_data['healthcare_establishments'].sum():,.0f}")

    return result


if __name__ == "__main__":
    ingest_zbp()
