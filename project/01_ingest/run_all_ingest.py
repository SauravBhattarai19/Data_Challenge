"""
Master ingestion runner.

Runs all ingest scripts in order.
FEMA NRI is run first (slowest: 8-15 min) so it processes while others run.

Usage:
    cd project/
    python 01_ingest/run_all_ingest.py

    # Or run individual scripts:
    python 01_ingest/ingest_fema_nri.py
"""

import sys
import time
import traceback
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# ── Script registry ────────────────────────────────────────────────────────────
# Format: (module_name, function_name, description)
SCRIPTS = [
    # Run NRI first — slowest, benefits most from early start
    ('ingest_fema_nri',   'ingest_fema_nri',   'FEMA NRI (946 MB DBF — ~10 min)'),
    ('ingest_igs',        'ingest_igs',         'IGS 2017-2025 (237 MB Excel)'),
    ('ingest_cdc_places', 'ingest_cdc_places',  'CDC PLACES 2025'),
    ('ingest_svi',        'ingest_svi',         'CDC/ATSDR SVI 2022'),
    ('ingest_eji',        'ingest_eji',         'CDC EJI'),
    ('ingest_hpsa',       'ingest_hpsa',        'HRSA HPSA (PC + MH)'),
    ('ingest_mua',        'ingest_mua',         'HRSA MUA/MUP'),
    ('ingest_fqhc',       'ingest_fqhc',        'HRSA FQHC Sites'),
    ('ingest_cbp',        'ingest_cbp',         'Census CBP 2023 (county biz patterns)'),
    ('ingest_sba',        'ingest_sba',         'SBA State Small Business Rankings 2025'),
    ('ingest_zbp',        'ingest_zbp',         'Census ZBP 2023 (ZIP biz patterns — MS)'),
    ('ingest_food',       'ingest_food',        'USDA Food Atlas 2019'),
    ('ingest_ahrf',       'ingest_ahrf',        'HRSA AHRF 2024-2025 Healthcare Workforce'),
]


def run_all(skip_nri: bool = False):
    print("=" * 60)
    print("  Healthy Economies, Healthy Communities")
    print("  Phase 1: Data Ingestion")
    print("=" * 60)

    results = {}
    total_start = time.time()

    for module_name, func_name, description in SCRIPTS:
        if skip_nri and 'fema' in module_name:
            print(f"\n[SKIP] {description}")
            continue

        print(f"\n{'─'*60}")
        print(f"[START] {description}")
        t0 = time.time()
        try:
            import importlib
            mod = importlib.import_module(module_name)
            fn  = getattr(mod, func_name)
            fn()
            elapsed = time.time() - t0
            results[module_name] = ('OK', elapsed)
            print(f"[DONE]  {description} — {elapsed:.1f}s")
        except Exception as e:
            elapsed = time.time() - t0
            results[module_name] = ('FAIL', elapsed)
            print(f"[FAIL]  {description}")
            traceback.print_exc()

    # ── Summary ────────────────────────────────────────────────────────────────
    total_elapsed = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"  Ingestion Summary  (total: {total_elapsed/60:.1f} min)")
    print(f"{'='*60}")
    for mod, (status, secs) in results.items():
        icon = 'OK' if status == 'OK' else 'FAIL'
        print(f"  {icon} {mod:<30} {secs:>6.1f}s  [{status}]")

    failed = [m for m, (s, _) in results.items() if s == 'FAIL']
    if failed:
        print(f"\n  FAILED scripts: {failed}")
        sys.exit(1)
    else:
        print(f"\n  All ingestion scripts completed successfully.")
        print(f"  Parquet files written to: project/data_processed/raw/")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip-nri', action='store_true',
                        help='Skip FEMA NRI (use if parquet already exists)')
    args = parser.parse_args()
    run_all(skip_nri=args.skip_nri)
