"""
run_all_analysis.py
-------------------
Runs the analysis pipeline in the correct order.

Stage 1: Community Typology  (turnaround/stuck labels + benchmarks)
Stage 2: ML Discovery        (3-model ensemble + SHAP + dimension weights)
Stage 3: IGS Regression      (sub-indicator → IGS coefficients for simulator)

Prerequisites: master_tract.parquet and igs_trends.parquet must exist.
"""

import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import MASTER_TRACT, IGS_TRENDS_PARQUET


def main():
    start = time.time()
    print("\n" + "=" * 65)
    print("  ANALYSIS PIPELINE — Healthy Economies, Healthy Communities")
    print("=" * 65)

    if not MASTER_TRACT.exists():
        print(f"\n  ERROR: {MASTER_TRACT} not found.")
        print("  Run:  python 02_build/build_master_tract.py")
        return
    if not IGS_TRENDS_PARQUET.exists():
        print(f"\n  ERROR: {IGS_TRENDS_PARQUET} not found.")
        print("  Run:  python 02_build/build_igs_trends.py")
        return

    # Stage 1: Community Typology
    print("\n\n" + "=" * 65)
    print("  STAGE 1 / 3")
    print("=" * 65)
    from community_typology import (
        build_typology, build_turnaround_benchmarks,
        build_typology_profiles, build_igs_improvement_model
    )
    df = build_typology()
    build_turnaround_benchmarks(df)
    build_typology_profiles(df)
    build_igs_improvement_model(df)

    # Stage 2: ML Discovery + SHAP
    print("\n\n" + "=" * 65)
    print("  STAGE 2 / 3")
    print("=" * 65)
    from ml_discovery import (
        load_data, define_features, train_three_models,
        compute_shap_analysis, discover_vulnerability_dimensions,
        compute_shap_interactions, delta_analysis
    )
    ml_df = load_data()
    features = define_features(ml_df)
    rf, X, y, scaler, features = train_three_models(ml_df, features)
    shap_df, summary = compute_shap_analysis(rf, X, y, features)
    discover_vulnerability_dimensions(shap_df, summary, features)
    compute_shap_interactions(rf, X, features)
    delta_analysis(rf, X, features, ml_df)

    # Stage 3: IGS Regression (simulator coefficients)
    print("\n\n" + "=" * 65)
    print("  STAGE 3 / 3")
    print("=" * 65)
    try:
        from igs_regression_analysis import run_igs_regression
        run_igs_regression()
    except Exception as e:
        print(f"  WARNING: IGS regression failed ({e}). Simulator will use pillar algebra only.")

    elapsed = time.time() - start
    print("\n\n" + "=" * 65)
    print(f"  ALL STAGES COMPLETE  ({elapsed:.0f}s)")
    print("=" * 65)


if __name__ == "__main__":
    main()
