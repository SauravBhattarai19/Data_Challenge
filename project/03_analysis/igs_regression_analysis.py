"""
igs_regression_analysis.py
--------------------------
OLS Multiple Linear Regression: which community sub-indicators predict IGS score?

Provides the investment simulator with defensible, evidence-based coefficients.

Research question:
    A 1-point improvement in [sub-indicator] is associated with how many
    additional IGS score points for that census tract?

Method:
    Cross-section OLS regression on the most recent IGS year.
    Dependent variable : igs_score (0-100, Mastercard Inclusive Growth Score)
    Independent variables: IGS sub-indicators + external risk/health features

Output:
    data_processed/igs_regression_coefficients.parquet

Run:
    python 03_analysis/igs_regression_analysis.py
"""

import sys, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np

try:
    import statsmodels.api as sm
except ImportError:
    import subprocess
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'statsmodels'], check=True)
    import statsmodels.api as sm

from config import (
    PROCESSED, MASTER_TRACT, IGS_TRENDS_PARQUET,
    DELTA_COUNTY_FIPS
)


def run_igs_regression():
    print("=" * 60)
    print("  IGS Regression Analysis — Investment Simulator Coefficients")
    print("=" * 60)

    print("\nLoading data...")
    ri = pd.read_parquet(MASTER_TRACT)
    if 'county_fips5' not in ri.columns:
        ri['county_fips5'] = ri['GEOID'].str[:5]
    trends = pd.read_parquet(IGS_TRENDS_PARQUET)

    latest_year = int(trends['year'].max())
    igs_latest = trends[trends['year'] == latest_year].copy()
    print(f"Using IGS year: {latest_year}  |  tracts: {len(igs_latest):,}")

    IGS_SUB = [
        'Internet Access Score',
        'Commercial Diversity Score',
        'Health Insurance Coverage Score',
        'Labor Market Engagement Index Score',
        'Small Business Loans Score',
        'New Businesses Score',
        'Female Above Poverty Score',
        'Personal Income Score',
        'Affordable Housing Score',
        'Minority/Women Owned Businesses Score',
        'Spending per Capita Score',
        'Travel Time to Work Score',
    ]

    EXTERNAL = [
        'HWAV_RISKS',
        'IFLD_RISKS',
        'RPL_THEMES',
        'pc_hpsa_score_max',
        'DIABETES_CrudePrev',
        'BPHIGH_CrudePrev',
        'OBESITY_CrudePrev',
        'LPA_CrudePrev',
        'ACCESS2_CrudePrev',
    ]

    available_igs = [c for c in IGS_SUB if c in igs_latest.columns]
    available_ext = [c for c in EXTERNAL if c in ri.columns]
    print(f"IGS sub-indicators: {len(available_igs)}  |  External features: {len(available_ext)}")

    df = (
        igs_latest[['GEOID', 'igs_score'] + available_igs]
        .merge(ri[['GEOID', 'county_fips5'] + available_ext], on='GEOID', how='inner')
        .dropna(subset=['igs_score'])
    )
    print(f"Merged dataset: {len(df):,} tracts")

    all_features = available_igs + available_ext

    cov = df[all_features].notna().mean()
    all_features = [f for f in all_features if cov[f] >= 0.5]
    print(f"Features after coverage filter: {len(all_features)}")

    X = df[all_features].fillna(df[all_features].median())
    y = df['igs_score']

    print("\nFitting OLS regression...")
    X_sm = sm.add_constant(X)
    model = sm.OLS(y, X_sm).fit(cov_type='HC3')

    print(f"\n  R-squared = {model.rsquared:.4f}")
    print(f"  Adj R-sq  = {model.rsquared_adj:.4f}")
    print(f"  n         = {len(df):,}")

    ci = model.conf_int()
    results = pd.DataFrame({
        'feature':       all_features,
        'coeff_raw':     model.params.iloc[1:].values,
        'std_error':     model.bse.iloc[1:].values,
        'p_value':       model.pvalues.iloc[1:].values,
        'conf_int_low':  ci.iloc[1:, 0].values,
        'conf_int_high': ci.iloc[1:, 1].values,
        'significant':   (model.pvalues.iloc[1:].values < 0.05),
        'r_squared':     model.rsquared,
        'n_obs':         len(df),
    })

    delta_mask = df['county_fips5'].isin(DELTA_COUNTY_FIPS)
    for feat in all_features:
        results.loc[results['feature'] == feat, 'national_mean'] = float(df[feat].mean())
        if delta_mask.any():
            results.loc[results['feature'] == feat, 'delta_mean'] = float(df.loc[delta_mask, feat].mean())
        else:
            results.loc[results['feature'] == feat, 'delta_mean'] = np.nan

    results = results.sort_values('coeff_raw', ascending=False)

    print("\nTop positive predictors (significant):")
    top_pos = results[results['significant'] & (results['coeff_raw'] > 0)].head(8)
    for _, r in top_pos.iterrows():
        print(f"  {r['feature']:<45} coeff={r['coeff_raw']:+.3f}  p={r['p_value']:.4f}")

    out_path = PROCESSED / 'igs_regression_coefficients.parquet'
    results.to_parquet(out_path, index=False)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    run_igs_regression()
