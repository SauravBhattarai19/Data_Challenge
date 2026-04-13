"""
Ingest SBA State Small Business Rankings 2025.

Reads: state_statistics_rankings_2025.xlsx (Rankings sheet)
Writes: sba_state_rankings.parquet

Columns: state_name, state_abbr, n_small_businesses, sb_employment_share_pct,
         establishments_opened, net_new_jobs, women_owned_pct,
         hispanic_owned_pct, veteran_owned_pct, + rank columns
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from config import SBA_RANKINGS_PATH, SBA_PARQUET, PROCESSED_RAW


def ingest_sba():
    print(f"Reading SBA state rankings from {SBA_RANKINGS_PATH} ...")
    PROCESSED_RAW.mkdir(parents=True, exist_ok=True)

    df = pd.read_excel(SBA_RANKINGS_PATH, sheet_name='Rankings', header=0)
    print(f"  Raw shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")

    # The first column is the state name (may be 'Unnamed: 0' or similar)
    cols = list(df.columns)
    df = df.rename(columns={cols[0]: 'state_name'})

    # Drop any aggregate/total rows
    df = df[df['state_name'].notna()].copy()
    df['state_name'] = df['state_name'].str.strip()
    df = df[~df['state_name'].str.lower().isin(['total', 'united states', ''])]

    # Parse the paired rank/value columns
    # The Excel has: State | Rank | Value | Rank | Value | ... for 7 metrics
    metric_names = [
        'n_small_businesses',
        'sb_employment_share_pct',
        'establishments_opened',
        'net_new_jobs',
        'women_owned_pct',
        'hispanic_owned_pct',
        'veteran_owned_pct',
    ]

    result = pd.DataFrame({'state_name': df['state_name'].values})

    # Columns after 'state_name' come in pairs: rank, value
    data_cols = [c for c in df.columns if c != 'state_name']
    pair_idx = 0
    for i in range(0, len(data_cols), 2):
        if pair_idx >= len(metric_names):
            break
        metric = metric_names[pair_idx]
        rank_col = data_cols[i]
        val_col = data_cols[i + 1] if i + 1 < len(data_cols) else None

        result[f'{metric}_rank'] = pd.to_numeric(df[rank_col].values, errors='coerce')
        if val_col is not None:
            # Values may have commas or percent signs
            vals = df[val_col].astype(str).str.replace(',', '').str.replace('%', '').str.strip()
            result[metric] = pd.to_numeric(vals, errors='coerce')
        pair_idx += 1

    # Add state abbreviation mapping
    state_abbr_map = {
        'Alabama': 'AL', 'Alaska': 'AK', 'Arizona': 'AZ', 'Arkansas': 'AR',
        'California': 'CA', 'Colorado': 'CO', 'Connecticut': 'CT', 'Delaware': 'DE',
        'District of Columbia': 'DC', 'Florida': 'FL', 'Georgia': 'GA', 'Hawaii': 'HI',
        'Idaho': 'ID', 'Illinois': 'IL', 'Indiana': 'IN', 'Iowa': 'IA',
        'Kansas': 'KS', 'Kentucky': 'KY', 'Louisiana': 'LA', 'Maine': 'ME',
        'Maryland': 'MD', 'Massachusetts': 'MA', 'Michigan': 'MI', 'Minnesota': 'MN',
        'Mississippi': 'MS', 'Missouri': 'MO', 'Montana': 'MT', 'Nebraska': 'NE',
        'Nevada': 'NV', 'New Hampshire': 'NH', 'New Jersey': 'NJ', 'New Mexico': 'NM',
        'New York': 'NY', 'North Carolina': 'NC', 'North Dakota': 'ND', 'Ohio': 'OH',
        'Oklahoma': 'OK', 'Oregon': 'OR', 'Pennsylvania': 'PA', 'Rhode Island': 'RI',
        'South Carolina': 'SC', 'South Dakota': 'SD', 'Tennessee': 'TN', 'Texas': 'TX',
        'Utah': 'UT', 'Vermont': 'VT', 'Virginia': 'VA', 'Washington': 'WA',
        'West Virginia': 'WV', 'Wisconsin': 'WI', 'Wyoming': 'WY',
    }
    result['state_abbr'] = result['state_name'].map(state_abbr_map)

    print(f"  Parsed {len(result)} states")

    # Mississippi highlight
    ms = result[result['state_abbr'] == 'MS']
    if len(ms) > 0:
        ms_row = ms.iloc[0]
        print(f"\n  Mississippi:")
        for metric in metric_names:
            val = ms_row.get(metric, None)
            rank = ms_row.get(f'{metric}_rank', None)
            if val is not None and not pd.isna(val):
                print(f"    {metric}: {val:,.0f} (rank {rank:.0f})" if val > 100
                      else f"    {metric}: {val:.1f}% (rank {rank:.0f})")

    result.to_parquet(SBA_PARQUET, index=False)
    print(f"\n  Saved -> {SBA_PARQUET}")
    return result


if __name__ == "__main__":
    ingest_sba()
