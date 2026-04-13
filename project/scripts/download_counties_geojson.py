"""
One-time download of US county GeoJSON for offline choropleth (Page 1).

Writes: project/data_processed/counties.geojson
Same source as Plotly's geojson-counties-fips.json.
"""
import json
import sys
from pathlib import Path
from urllib.request import urlopen

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config import COUNTIES_GEOJSON, PROCESSED

URL = "https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json"


def main():
    PROCESSED.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {URL} ...")
    with urlopen(URL) as r:
        data = json.load(r)
    with open(COUNTIES_GEOJSON, "w", encoding="utf-8") as f:
        json.dump(data, f)
    print(f"Saved {COUNTIES_GEOJSON} ({COUNTIES_GEOJSON.stat().st_size / 1024 / 1024:.1f} MB)")


if __name__ == "__main__":
    main()
