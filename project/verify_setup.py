"""
Verification script — run this before starting ingestion.

Checks:
  1. All expected raw data files exist and are accessible
  2. Required Python packages are installed
  3. Output directories are writable
  4. Prints a summary with pass/fail for each check
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    DATA_ROOT, PROCESSED, PROCESSED_RAW,
    IGS_PATH, FEMA_NRI_PATH, CDC_PLACES_PATH, SVI_PATH, EJI_PATH,
    HPSA_PC_PATH, HPSA_MH_PATH, MUA_PATH, FQHC_PATH,
    CBP_COUNTY_ZIP, ZBP_DETAIL_ZIP, ZBP_TOTALS_ZIP,
    SBA_RANKINGS_PATH, FOOD_ATLAS_ZIP, TIGER_PATH, AHRF_ZIP_PATH,
)

CHECKS = []


def check(label: str, condition: bool, detail: str = ''):
    CHECKS.append((label, condition, detail))
    icon = 'PASS' if condition else 'FAIL'
    print(f"  [{icon}]  {label}" + (f"\n           {detail}" if detail and not condition else ''))


print("=" * 65)
print("  Healthy Economies, Healthy Communities")
print("  Setup Verification")
print("=" * 65)

# ── Python packages ─────────────────────────────────────────────────────────
print("\n[1] Python Packages")
pkgs = [
    ('pandas',      'pandas'),
    ('numpy',       'numpy'),
    ('scipy',       'scipy'),
    ('sklearn',     'scikit-learn'),
    ('statsmodels', 'statsmodels'),
    ('streamlit',   'streamlit'),
    ('plotly',      'plotly'),
    ('folium',      'folium'),
    ('openpyxl',    'openpyxl'),
    ('pyarrow',     'pyarrow'),
    ('tqdm',        'tqdm'),
    ('requests',    'requests'),
]

for mod, pkg in pkgs:
    try:
        __import__(mod)
        check(f"  {pkg}", True)
    except ImportError:
        check(f"  {pkg}", False, f"pip install {pkg}")

# shap and geopandas are optional but needed for analysis / maps
for mod, pkg, note in [
    ('shap',       'shap',       'needed for ml_discovery.py'),
    ('geopandas',  'geopandas',  'needed for TIGER tract map'),
]:
    try:
        __import__(mod)
        check(f"  {pkg} (optional)", True)
    except ImportError:
        check(f"  {pkg} (optional — {note})", False, f"pip install {pkg}")

# ── Tier 1: Critical files ───────────────────────────────────────────────────
print("\n[2] Tier 1 Data Files (Critical — ingest will fail without these)")
tier1 = [
    ("IGS Excel",    IGS_PATH),
    ("FEMA NRI DBF", FEMA_NRI_PATH),
    ("CDC PLACES",   CDC_PLACES_PATH),
    ("SVI CSV",      SVI_PATH),
    ("EJI CSV",      EJI_PATH),
]
for label, path in tier1:
    exists = path.exists()
    size   = f"{path.stat().st_size / 1024 / 1024:.0f} MB" if exists else "MISSING"
    check(f"  {label}", exists, f"Expected at: {path}  ({size})")

# ── Tier 2: Healthcare / HRSA files ─────────────────────────────────────────
print("\n[3] Tier 2 Healthcare Files")
tier2 = [
    ("HPSA Primary Care",   HPSA_PC_PATH),
    ("HPSA Mental Health",  HPSA_MH_PATH),
    ("MUA/MUP",             MUA_PATH),
    ("FQHC Sites",          FQHC_PATH),
    ("AHRF County Zip",     AHRF_ZIP_PATH),
]
for label, path in tier2:
    exists = path.exists()
    size   = f"{path.stat().st_size / 1024 / 1024:.0f} MB" if exists else "MISSING"
    check(f"  {label}", exists, f"Expected at: {path}  ({size})")

# ── Tier 3: Business / Economic files ───────────────────────────────────────
print("\n[4] Tier 3 Business & Economic Files")
tier3 = [
    ("CBP County Zip",      CBP_COUNTY_ZIP),
    ("ZBP Detail Zip",      ZBP_DETAIL_ZIP),
    ("ZBP Totals Zip",      ZBP_TOTALS_ZIP),
    ("SBA State Rankings",  SBA_RANKINGS_PATH),
    ("Food Atlas Zip",      FOOD_ATLAS_ZIP),
]
for label, path in tier3:
    exists = path.exists()
    size   = f"{path.stat().st_size / 1024 / 1024:.0f} MB" if exists else "MISSING"
    check(f"  {label}", exists, f"Expected at: {path}  ({size})")

# ── Optional: TIGER shapefile ────────────────────────────────────────────────
print("\n[5] Optional Files")
check(
    "  TIGER 2022 MS Tracts (optional — needed for Delta GeoJSON map)",
    TIGER_PATH.exists(),
    f"Download tl_2022_28_tract.zip from census.gov/geo/maps-data/data/tiger -> {TIGER_PATH}",
)

# ── Writability ──────────────────────────────────────────────────────────────
print("\n[6] Output Directories")
PROCESSED_RAW.mkdir(parents=True, exist_ok=True)
test_file = PROCESSED_RAW / '.write_test'
try:
    test_file.write_text('ok')
    test_file.unlink()
    check("  data_processed/raw/ is writable", True)
except Exception as e:
    check("  data_processed/raw/ is writable", False, str(e))

# ── DATA_ROOT sanity ─────────────────────────────────────────────────────────
print("\n[7] DATA_ROOT")
check(
    f"  DATA_ROOT exists ({DATA_ROOT})",
    DATA_ROOT.exists(),
    "Update DATA_ROOT in config.py to the correct absolute path on this machine",
)

# ── Summary ──────────────────────────────────────────────────────────────────
n_pass = sum(1 for _, ok, _ in CHECKS if ok)
n_fail = len(CHECKS) - n_pass
print(f"\n{'='*65}")
print(f"  Result: {n_pass} passed, {n_fail} failed")

critical_fail = [
    lbl for lbl, ok, _ in CHECKS
    if not ok and any(t in lbl for t in [
        'IGS', 'FEMA', 'PLACES', 'SVI', 'EJI',
        'pandas', 'numpy', 'DATA_ROOT',
    ])
]
if critical_fail:
    print(f"\n  CRITICAL failures (must fix before running ingestion):")
    for lbl in critical_fail:
        print(f"    - {lbl.strip()}")
    sys.exit(1)
elif n_fail:
    print("\n  Non-critical failures — some features (maps, ML, business data) may be limited.")
else:
    print("\n  All checks passed! Ready to run:")
    print("    conda activate data_challenge")
    print("    python 01_ingest/run_all_ingest.py")
