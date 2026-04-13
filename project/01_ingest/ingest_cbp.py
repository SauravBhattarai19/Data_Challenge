"""
Ingest Census County Business Patterns 2023 -> county-level healthcare/food business Parquet.

Reads cbp23co.txt from the zip archive.
Filters to relevant NAICS codes and pivots to one row per county.

Key gotchas:
  - Filter naics != '------' to exclude total rows
  - dtype must preserve leading zeros in FIPS fields
  - CBP uses 'emp_nf' noise flags; use 'emp' (midpoint estimate) for counts
"""

import sys
import zipfile
import io
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from config import CBP_COUNTY_ZIP, CBP_PARQUET, PROCESSED_RAW

# NAICS codes of interest (prefix match)
NAICS_FILTERS = {
    'pharmacy':          ['446110'],
    'physician_office':  ['621111', '621112', '6211'],
    'dental':            ['621210', '6212'],
    'home_health':       ['621610', '6216'],
    'hospital':          ['622'],
    'mental_health_svc': ['6222', '6223'],
    'food_retail':       ['445110', '445'],
    'total_healthcare':  ['621', '622', '623'],
}


def naics_match(code: str, prefixes: list[str]) -> bool:
    return any(code.startswith(p) for p in prefixes)


def ingest_cbp():
    print(f"Reading CBP county file from {CBP_COUNTY_ZIP} ...")
    PROCESSED_RAW.mkdir(parents=True, exist_ok=True)

    # ── Read from zip ──────────────────────────────────────────────────────────
    with zipfile.ZipFile(CBP_COUNTY_ZIP, 'r') as z:
        # Find the county-level txt file
        txt_files = [n for n in z.namelist() if n.endswith('.txt') and 'co' in n.lower()]
        print(f"  Files in zip: {z.namelist()[:5]}")
        fname = txt_files[0] if txt_files else z.namelist()[0]
        print(f"  Reading: {fname}")
        with z.open(fname) as f:
            df = pd.read_csv(
                f,
                dtype={'fipstate': str, 'fipscty': str, 'naics': str},
                low_memory=False,
                encoding='latin-1',
            )

    df.columns = df.columns.str.strip().str.lower()
    print(f"  Raw shape: {df.shape}")
    print(f"  Columns: {list(df.columns[:12])}")

    # ── Remove NAICS total rows ────────────────────────────────────────────────
    df = df[df['naics'].str.strip() != '------']
    df['naics'] = df['naics'].str.strip().str.replace('/', '', regex=False)

    # ── Build county FIPS ──────────────────────────────────────────────────────
    df['county_fips5'] = (
        df['fipstate'].str.zfill(2) + df['fipscty'].str.zfill(3)
    )

    # ── Establishment count column ─────────────────────────────────────────────
    est_col = 'est'  # establishments
    if est_col not in df.columns:
        # Try alternative names
        est_col = next((c for c in df.columns if 'est' in c), None)
    df[est_col] = pd.to_numeric(df[est_col], errors='coerce').fillna(0)

    # ── Build category flags and aggregate ────────────────────────────────────
    result_parts = []
    for cat, prefixes in NAICS_FILTERS.items():
        mask = df['naics'].apply(lambda x: naics_match(x, prefixes))
        sub = df[mask].groupby('county_fips5')[est_col].sum().reset_index()
        sub = sub.rename(columns={est_col: f'biz_{cat}'})
        result_parts.append(sub)

    # ── County-wide business metrics (beyond healthcare NAICS) ────────────────
    # Total establishments per county (all 6-digit NAICS)
    six_digit = df[df['naics'].str.match(r'^\d{6}$', na=False)]
    county_totals = six_digit.groupby('county_fips5').agg(
        biz_total_all=('est', 'sum'),
        biz_total_emp=('emp', 'sum') if 'emp' in df.columns else ('est', 'sum'),
    ).reset_index()

    if 'emp' in df.columns:
        emp_agg = six_digit.groupby('county_fips5')['emp'].sum().reset_index()
        emp_agg.columns = ['county_fips5', 'biz_total_emp']
        if 'biz_total_emp' not in county_totals.columns:
            county_totals = county_totals.merge(emp_agg, on='county_fips5', how='left')

    # Sector diversity: count distinct 2-digit NAICS sectors per county
    df['naics_2digit'] = df['naics'].str[:2]
    sector_counts = df[df['naics_2digit'].str.match(r'^\d{2}$', na=False)].groupby(
        'county_fips5'
    )['naics_2digit'].nunique().reset_index()
    sector_counts.columns = ['county_fips5', 'biz_sector_diversity']

    # Small businesses: establishments with <20 employees
    # Use size bin columns if available
    size_cols_small = ['n<5', 'n5_9', 'n10_19']
    available_small = [c for c in size_cols_small if c in six_digit.columns]
    if available_small:
        for c in available_small:
            six_digit[c] = pd.to_numeric(
                six_digit[c].astype(str).str.replace('N', '0'), errors='coerce'
            ).fillna(0)
        six_digit['_small'] = six_digit[available_small].sum(axis=1)
        small_biz = six_digit.groupby('county_fips5')['_small'].sum().reset_index()
        small_biz.columns = ['county_fips5', 'biz_small_under20']
    else:
        small_biz = pd.DataFrame({'county_fips5': df['county_fips5'].unique(), 'biz_small_under20': 0})

    result_parts.extend([county_totals, sector_counts, small_biz])

    # ── Merge all categories ───────────────────────────────────────────────────
    from functools import reduce
    all_counties = pd.DataFrame({'county_fips5': df['county_fips5'].unique()})
    agg = reduce(
        lambda left, right: pd.merge(left, right, on='county_fips5', how='left'),
        [all_counties] + result_parts
    )
    agg = agg.fillna(0)

    # Derived metrics
    agg['biz_small_pct'] = (agg['biz_small_under20'] / agg['biz_total_all'].clip(lower=1) * 100).round(1)
    agg['biz_avg_emp_per_est'] = (agg['biz_total_emp'] / agg['biz_total_all'].clip(lower=1)).round(1)

    print(f"  County-level output: {len(agg):,} counties")
    print(f"  Columns: {list(agg.columns)}")
    print(f"  New metrics: biz_total_all, biz_sector_diversity, biz_small_under20, biz_small_pct")

    agg.to_parquet(CBP_PARQUET, index=False)
    print(f"  Saved -> {CBP_PARQUET}")
    return agg


if __name__ == "__main__":
    ingest_cbp()
