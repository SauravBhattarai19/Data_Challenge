"""
Build pre-computed Delta GeoJSON for Folium choropleth map.

Requires Census TIGER 2022 tract shapefile: Data/TIGER/tl_2022_us_tract.zip

If TIGER not yet downloaded, prints download instructions and exits gracefully.

Output: data_processed/delta_full.parquet (tract-level with geometry)
        data_processed/delta_tracts.geojson (for Folium)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import json
from config import (
    TIGER_PATH, MASTER_TRACT, PROCESSED, DELTA_PARQUET,
    DELTA_COUNTY_FIPS, DELTA_COUNTY_NAMES
)

DELTA_GEOJSON = PROCESSED / 'delta_tracts.geojson'


def check_tiger():
    if not TIGER_PATH.exists():
        print("=" * 60)
        print("  TIGER Shapefile Not Found")
        print("=" * 60)
        print(f"  Expected: {TIGER_PATH}")
        print()
        print("  Download instructions:")
        print("  1. Go to: https://www.census.gov/geographies/mapping-files/time-series/geo/tiger-line-file.html")
        print("  2. Select: 2022 -> Web Interface -> Census Tracts -> Download All States")
        print("     OR download only Mississippi: tl_2022_28_tract.zip")
        print(f"  3. Save to: {TIGER_PATH.parent}/")
        print()
        print("  INTERIM: build_delta_geojson will be skipped.")
        print("  The app will use county-level Plotly choropleth instead of tract-level Folium map.")
        return False
    return True


def build_delta_geojson():
    PROCESSED.mkdir(parents=True, exist_ok=True)

    if not check_tiger():
        # Build a Delta subset parquet without geometry (still useful)
        if MASTER_TRACT.exists():
            ri = pd.read_parquet(MASTER_TRACT)
            ri['county_fips5'] = ri['GEOID'].str[:5]
            delta = ri[ri['county_fips5'].isin(DELTA_COUNTY_FIPS)].copy()
            delta['county_name'] = delta['county_fips5'].map(DELTA_COUNTY_NAMES)
            delta.to_parquet(DELTA_PARQUET, index=False)
            print(f"  Saved Delta tabular (no geometry) -> {DELTA_PARQUET}")
            print(f"  Shape: {delta.shape}")
        return None

    try:
        import geopandas as gpd
    except ImportError:
        print("  geopandas not installed. Run: pip install geopandas")
        return None

    print(f"Loading TIGER shapefile: {TIGER_PATH}")
    tracts = gpd.read_file(f"zip://{TIGER_PATH}")
    tracts['GEOID'] = tracts['GEOID'].astype(str).str.zfill(11)
    print(f"  All US tracts: {tracts.shape}")

    # ── Filter to MS Delta counties ────────────────────────────────────────────
    tracts['county_fips5'] = tracts['GEOID'].str[:5]
    delta_shp = tracts[tracts['county_fips5'].isin(DELTA_COUNTY_FIPS)].copy()
    print(f"  MS Delta tracts: {len(delta_shp)}")

    # ── Join with resilience index ─────────────────────────────────────────────
    if MASTER_TRACT.exists():
        mt = pd.read_parquet(MASTER_TRACT)
        delta_full = delta_shp.merge(
            mt.drop(columns=['county_fips5'], errors='ignore'),
            on='GEOID', how='left'
        )
    else:
        delta_full = delta_shp.copy()
        print("  WARNING: master_tract.parquet not found — geometry only")

    # ── Add county name ────────────────────────────────────────────────────────
    delta_full['county_name'] = delta_full['county_fips5'].map(DELTA_COUNTY_NAMES)

    # ── Save parquet (without geometry for tabular ops) ───────────────────────
    tabular = pd.DataFrame(delta_full.drop(columns='geometry'))
    tabular.to_parquet(DELTA_PARQUET, index=False)
    print(f"  Saved Delta tabular -> {DELTA_PARQUET}")

    # ── Save GeoJSON ───────────────────────────────────────────────────────────
    # Project to WGS84 for web map compatibility
    delta_full = delta_full.to_crs(epsg=4326)
    delta_full.to_file(DELTA_GEOJSON, driver='GeoJSON')
    print(f"  Saved Delta GeoJSON -> {DELTA_GEOJSON}")
    print(f"  GeoJSON size: {DELTA_GEOJSON.stat().st_size / 1024 / 1024:.1f} MB")

    return delta_full


if __name__ == "__main__":
    build_delta_geojson()
