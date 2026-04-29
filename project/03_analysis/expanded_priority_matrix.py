"""
expanded_priority_matrix.py
============================
Pipeline: 89-feature ML training → SHAP → 2×2 Investment Priority Matrix.

Every feature carries an explicit `higher_is_better` flag so gap direction is consistent:
  POSITIVE gap  =  area is WORSE / behind turnaround  (intervention needed)
  NEGATIVE gap  =  area is AHEAD of turnaround        (existing strength)

Outputs:
  data_processed/expanded_model.parquet           ML-ready feature dataset (25k at-risk tracts)
  data_processed/expanded_shap_summary.parquet    SHAP importance per feature (89 rows)
  data_processed/expanded_shap_values.parquet     Per-tract SHAP values (5,000 sample × 89)
  data_processed/expanded_shap_sample.parquet     Feature values for the SHAP sample
  data_processed/expanded_rf_model.pkl            Serialized Random Forest
  presentation/figures/priority_matrix_<target>.png
"""

import sys, warnings, argparse, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe

from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

try:
    import shap
except ImportError:
    import subprocess
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'shap'], check=True)
    import shap

try:
    from adjustText import adjust_text
    HAS_ADJUST_TEXT = True
except ImportError:
    HAS_ADJUST_TEXT = False

from config import PROCESSED, PROCESSED_RAW, DELTA_COUNTY_FIPS, DELTA_COUNTY_NAMES, IGS_VULN_THRESHOLD

# ─────────────────────────────────────────────────────────────────────────────
# FEATURE REGISTRY
# Each entry: (column_name, display_label, category, higher_is_better)
#
# higher_is_better=True  → gap = norm(turnaround) - norm(area)
#                           positive gap = area is BELOW turnaround = needs investment
# higher_is_better=False → gap = norm(area) - norm(turnaround)
#                           positive gap = area is WORSE (higher prevalence) than turnaround
# higher_is_better=None  → demographic/context feature, gap is directional but
#                           labeled as "context" not "lever" in the chart
# ─────────────────────────────────────────────────────────────────────────────

FEATURE_REGISTRY = [
    # ── IGS 2017 sub-indicators (all: higher percentile = better) ─────────────
    ('Internet Access Score_2017',                  'Internet Access',               'IGS: Place',      True),
    ('Affordable Housing Score_2017',               'Affordable Housing',            'IGS: Place',      True),
    ('Travel Time to Work Score_2017',              'Travel Time to Work',           'IGS: Place',      True),
    ('Net Occupancy Score_2017',                    'Net Occupancy',                 'IGS: Place',      True),
    ('Residential Real Estate Value Score_2017',    'Real Estate Value',             'IGS: Place',      True),
    ('Acres of Park Land Score_2017',               'Acres of Park Land',            'IGS: Place',      True),
    ('New Businesses Score_2017',                   'New Businesses',                'IGS: Economy',    True),
    ('Small Business Loans Score_2017',             'Small Business Loans',          'IGS: Economy',    True),
    ('Commercial Diversity Score_2017',             'Commercial Diversity',          'IGS: Economy',    True),
    ('Minority/Women Owned Businesses Score_2017',  'MWOB Score',                    'IGS: Economy',    True),
    ('Spend Growth Score_2017',                     'Spend Growth',                  'IGS: Economy',    True),
    ('Health Insurance Coverage Score_2017',        'Health Insurance',              'IGS: Community',  True),
    ('Labor Market Engagement Index Score_2017',    'Labor Market Engagement',       'IGS: Community',  True),
    ('Female Above Poverty Score_2017',             'Female Above Poverty',          'IGS: Community',  True),
    ('Personal Income Score_2017',                  'Personal Income',               'IGS: Community',  True),
    ('Spending per Capita Score_2017',              'Spending per Capita',           'IGS: Community',  True),
    ('Gini Coefficient Score_2017',                 'Gini Coefficient Score',        'IGS: Community',  True),
    ('Early Education Enrollment Score_2017',       'Early Education Enrollment',    'IGS: Community',  True),

    # ── CDC PLACES health (CrudePrev = % with condition; lower prevalence = better) ──
    ('DIABETES_CrudePrev',      'Diabetes Rate',               'Health: Chronic',    False),
    ('OBESITY_CrudePrev',       'Obesity Rate',                'Health: Chronic',    False),
    ('BPHIGH_CrudePrev',        'High Blood Pressure',         'Health: Chronic',    False),
    ('CHD_CrudePrev',           'Coronary Heart Disease',      'Health: Chronic',    False),
    ('STROKE_CrudePrev',        'Stroke Rate',                 'Health: Chronic',    False),
    ('COPD_CrudePrev',          'COPD Rate',                   'Health: Chronic',    False),
    ('CASTHMA_CrudePrev',       'Asthma Rate',                 'Health: Chronic',    False),
    ('DEPRESSION_CrudePrev',    'Depression Rate',             'Health: Chronic',    False),
    ('MHLTH_CrudePrev',         'Poor Mental Health Days',     'Health: Chronic',    False),
    ('MOBILITY_CrudePrev',      'Mobility Disability',         'Health: Chronic',    False),
    ('LPA_CrudePrev',           'Physical Inactivity',         'Health: Behavior',   False),
    ('CSMOKING_CrudePrev',      'Smoking Rate',                'Health: Behavior',   False),
    ('BINGE_CrudePrev',         'Binge Drinking Rate',         'Health: Behavior',   False),
    ('SLEEP_CrudePrev',         'Insufficient Sleep',          'Health: Behavior',   False),
    ('ACCESS2_CrudePrev',       'No Doctor Visit (Access)',    'Health: Access',     False),
    ('CHECKUP_CrudePrev',       'Annual Checkup Rate',         'Health: Access',     True),   # higher = better preventive care
    ('DENTAL_CrudePrev',        'Dental Visit Rate',           'Health: Access',     True),
    ('MAMMOUSE_CrudePrev',      'Mammography Use',             'Health: Access',     True),
    ('COLON_SCREEN_CrudePrev',  'Colorectal Screening',        'Health: Access',     True),

    # ── SVI granular (population-normalized %, all: lower = more disadvantaged) ──
    ('PCT_UNEMP',   'Unemployment Rate',           'SVI: Socioeconomic',  False),
    ('PCT_NOHSDP',  'No High School Diploma',      'SVI: Socioeconomic',  False),
    ('PCT_DISABL',  'Disability Rate',             'SVI: Household',      False),
    ('PCT_SNGPNT',  'Single Parent Rate',          'SVI: Household',      False),
    ('PCT_LIMENG',  'Limited English Proficiency', 'SVI: Minority',       False),
    ('PCT_MOBILE',  'Mobile Home Rate',            'SVI: Housing',        False),
    ('PCT_CROWD',   'Crowded Housing Rate',        'SVI: Housing',        False),
    ('PCT_NOVEH',   'No Vehicle Households',       'SVI: Housing',        False),
    # Demographic context (not better/worse, just composition)
    ('PCT_AGE65',   '% Age 65+',                  'SVI: Demographic',    None),
    ('PCT_AGE17',   '% Under Age 17',             'SVI: Demographic',    None),
    ('PCT_MINRTY',  '% Minority Population',      'SVI: Demographic',    None),
    ('PCT_MUNIT',   '% Multi-Unit Housing',       'SVI: Housing',        None),
    ('PCT_GROUPQ',  '% Group Quarters',           'SVI: Demographic',    None),

    # ── Food Atlas demographic (normalized) ──────────────────────────────────
    ('PCT_TractBlack',     '% Black Population',    'Food/Demo: Race',   None),
    ('PCT_TractHispanic',  '% Hispanic Population', 'Food/Demo: Race',   None),
    ('PCT_TractKids',      '% Children (<18)',      'Food/Demo: Age',    None),
    ('PCT_TractSeniors',   '% Seniors (65+)',       'Food/Demo: Age',    None),
    ('LILATracts_halfAnd10', 'Food Desert (0.5mi)', 'Food/Demo: Access', False),
    ('LILATracts_1And10',    'Food Desert (1mi)',   'Food/Demo: Access', False),
    ('MedianFamilyIncome',   'Median Family Income','Food/Demo: Econ',   True),

    # ── Business infrastructure (county-level; higher = better access) ────────
    ('biz_pharmacy',           'Pharmacies',                   'Business',  True),
    ('biz_physician_office',   'Physician Offices',            'Business',  True),
    ('biz_dental',             'Dental Offices',               'Business',  True),
    ('biz_home_health',        'Home Health Agencies',         'Business',  True),
    ('biz_hospital',           'Hospitals',                    'Business',  True),
    ('biz_mental_health_svc',  'Mental Health Services',       'Business',  True),
    ('biz_food_retail',        'Food Retail Businesses',       'Business',  True),
    ('biz_total_healthcare',   'Total Healthcare Businesses',  'Business',  True),
    ('biz_sector_diversity',   'Sector Diversity (NAICS)',     'Business',  True),
    ('biz_small_pct',          '% Small Businesses',          'Business',  True),

    # ── FEMA/climate risk (higher risk score = worse) ─────────────────────────
    ('RISK_SCORE',   'Overall Climate Risk',    'Climate',   False),
    ('EAL_SCORE',    'Expected Annual Loss',    'Climate',   False),
    ('HWAV_RISKS',   'Heat Wave Risk',          'Climate',   False),
    ('IFLD_RISKS',   'Inland Flood Risk',       'Climate',   False),
    ('HRCN_RISKS',   'Hurricane Risk',          'Climate',   False),
    ('TRND_RISKS',   'Tornado Risk',            'Climate',   False),
    ('DRGT_RISKS',   'Drought Risk',            'Climate',   False),
    ('RESL_SCORE',   'FEMA Resilience Score',   'Climate',   True),   # higher resilience = better
    ('BUILDVALUE',   'Building Exposure Value', 'Climate',   None),   # pure scale context

    # ── EJI composite burden (lower = less burden = better) ─────────────────
    ('RPL_EBM',   'Environmental Burden',         'EJI',   False),
    ('RPL_HVM',   'Health Vulnerability',         'EJI',   False),

    # ── Healthcare access ─────────────────────────────────────────────────────
    ('pc_hpsa_score_max',    'Primary Care Shortage Score',    'Healthcare Access',  False),  # higher HPSA = worse
    ('mh_hpsa_score_max',    'Mental Health Shortage Score',   'Healthcare Access',  False),
    ('fqhc_count',           'FQHC Count',                    'Healthcare Access',  True),
    ('in_pc_hpsa',           'In PC Shortage Area',           'Healthcare Access',  False),
    ('in_mh_hpsa',           'In MH Shortage Area',           'Healthcare Access',  False),
    ('in_mua',               'Medically Underserved Area',    'Healthcare Access',  False),
    ('imu_score_mean',        'IMU Underserved Score',        'Healthcare Access',  True),   # higher IMU = less underserved

    # ── AHRF county healthcare supply ─────────────────────────────────────────
    ('ahrf_pc_phys_per_10k',   'Primary Care MDs per 10k',    'Healthcare Supply',  True),
    ('ahrf_psych_per_10k',     'Psychiatrists per 10k',       'Healthcare Supply',  True),
    ('ahrf_beds_per_10k',      'Hospital Beds per 10k',       'Healthcare Supply',  True),

    # ── Other economic context ────────────────────────────────────────────────
    ('PovertyRate',      'Poverty Rate',         'Economic Context',  False),
    ('POPULATION',       'Population (scale)',   'Economic Context',  None),
]

# Color palette per category group
CATEGORY_COLORS = {
    'IGS: Place':       '#2980b9',
    'IGS: Economy':     '#c8830a',
    'IGS: Community':   '#27ae60',
    'Health: Chronic':  '#c0392b',
    'Health: Behavior': '#e74c3c',
    'Health: Access':   '#e67e22',
    'SVI: Socioeconomic':'#8e44ad',
    'SVI: Household':   '#9b59b6',
    'SVI: Minority':    '#6c3483',
    'SVI: Housing':     '#7d3c98',
    'SVI: Demographic': '#aab7b8',
    'Food/Demo: Race':  '#aab7b8',
    'Food/Demo: Age':   '#95a5a6',
    'Food/Demo: Access':'#7f8c8d',
    'Food/Demo: Econ':  '#27ae60',
    'Business':         '#6B4226',
    'Climate':          '#1a5276',
    'EJI':              '#117a65',
    'Healthcare Access':'#2471a3',
    'Healthcare Supply':'#1f618d',
    'Economic Context': '#7e5109',
}


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: Build expanded feature dataset
# ─────────────────────────────────────────────────────────────────────────────

def build_expanded_dataset():
    """
    Start from igs_improvement_model.parquet (base ML dataset with turnaround labels)
    and join all additional features: missing IGS 2017 sub-indicators, granular SVI
    population-normalized rates, and Food Atlas demographics.
    """
    print("=" * 65)
    print("  Building Expanded Feature Dataset")
    print("=" * 65)

    base = pd.read_parquet(PROCESSED / 'igs_improvement_model.parquet')
    print(f"  Base ML dataset: {len(base):,} tracts, {base.shape[1]} columns")

    # ── Join 4 missing IGS 2017 sub-indicators ───────────────────────────────
    igs_raw = pd.read_parquet(PROCESSED_RAW / 'igs_all_years.parquet')
    missing_igs = [
        'Residential Real Estate Value Score',
        'Acres of Park Land Score',
        'Gini Coefficient Score',
        'Early Education Enrollment Score',
    ]
    y2017 = igs_raw[igs_raw['year'] == 2017][['GEOID'] + missing_igs].copy()
    rename_map = {c: f'{c}_2017' for c in missing_igs}
    y2017 = y2017.rename(columns=rename_map)
    base = base.merge(y2017, on='GEOID', how='left')
    for new_col in rename_map.values():
        pct = base[new_col].notna().mean() * 100
        print(f"  + {new_col}: {pct:.1f}% coverage")

    # ── Build granular SVI population-normalized rates ────────────────────────
    svi_raw = pd.read_parquet(PROCESSED_RAW / 'svi.parquet')
    svi_e_cols = [
        'E_UNEMP', 'E_NOHSDP', 'E_AGE65', 'E_AGE17', 'E_DISABL',
        'E_SNGPNT', 'E_MINRTY', 'E_LIMENG', 'E_MUNIT', 'E_MOBILE',
        'E_CROWD', 'E_NOVEH', 'E_GROUPQ',
    ]
    svi_work = svi_raw[['GEOID', 'E_TOTPOP'] + svi_e_cols].copy()
    for col in svi_e_cols:
        pct_col = col.replace('E_', 'PCT_')
        svi_work[pct_col] = (
            svi_work[col] / svi_work['E_TOTPOP'].replace(0, np.nan)
        ) * 100
    pct_svi_cols = [c.replace('E_', 'PCT_') for c in svi_e_cols]
    base = base.merge(svi_work[['GEOID'] + pct_svi_cols], on='GEOID', how='left')
    # Drop the old composite SVI RPL_THEME columns (replaced by granular)
    drop_composites = [c for c in ['RPL_THEME1','RPL_THEME2','RPL_THEME3','RPL_THEME4',
                                    'RPL_THEMES','F_TOTAL','E_TOTPOP','E_UNINSUR','E_POV150']
                       if c in base.columns]
    base = base.drop(columns=drop_composites)
    print(f"  + {len(pct_svi_cols)} granular SVI PCT_ features (dropped {len(drop_composites)} composites)")

    # ── Join Food Atlas demographic columns ───────────────────────────────────
    food_raw = pd.read_parquet(PROCESSED_RAW / 'food_atlas.parquet')
    food_demo = food_raw[['GEOID', 'TractKids', 'TractSeniors',
                           'TractBlack', 'TractHispanic', 'TractWhite',
                           'LILATracts_halfAnd10', 'LILATracts_1And20',
                           'MedianFamilyIncome']].copy()
    # Normalize raw counts by SVI total population
    pop_ref = svi_raw[['GEOID', 'E_TOTPOP']].copy()
    food_demo = food_demo.merge(pop_ref, on='GEOID', how='left')
    for col in ['TractKids', 'TractSeniors', 'TractBlack', 'TractHispanic']:
        food_demo[f'PCT_{col}'] = (
            food_demo[col] / food_demo['E_TOTPOP'].replace(0, np.nan)
        ) * 100
    food_add_cols = ['PCT_TractKids', 'PCT_TractSeniors', 'PCT_TractBlack',
                     'PCT_TractHispanic', 'LILATracts_halfAnd10',
                     'LILATracts_1And20', 'MedianFamilyIncome']
    base = base.merge(food_demo[['GEOID'] + food_add_cols], on='GEOID', how='left')
    print(f"  + {len(food_add_cols)} Food Atlas demographic features")

    out_path = PROCESSED / 'expanded_model.parquet'
    base.to_parquet(out_path, index=False)
    print(f"\n  Saved → {out_path}  ({base.shape[0]:,} rows × {base.shape[1]} cols)")
    return base


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: Define ML features (registry → what actually exists in dataset)
# ─────────────────────────────────────────────────────────────────────────────

def get_ml_features(df):
    """
    Filter FEATURE_REGISTRY to columns that exist in df, excluding outcome/id columns.
    Returns list of feature column names.
    """
    exclude = {
        'GEOID', 'turnaround', 'typology', 'delta_igs', 'STATE', 'COUNTY',
        'county_fips5', 'state_fips2', 'is_delta', 'Urban', 'RISK_RATNG',
        'RESL_RATNG', 'HWAV_RISKR', 'IFLD_RISKR', 'HRCN_RISKR', 'TRND_RISKR',
        'DRGT_RISKR', 'WFIR_RISKR', 'ahrf_county_name',
    }
    # Columns ending in _2025 are outcome-period — exclude to prevent leakage
    registry_cols = {row[0] for row in FEATURE_REGISTRY}
    features = [
        c for c in df.columns
        if c in registry_cols
        and c not in exclude
        and not c.endswith('_2025')
    ]
    # Deduplicate, preserve registry order
    ordered = [row[0] for row in FEATURE_REGISTRY if row[0] in features]
    print(f"  ML feature count: {len(ordered)}")
    return ordered


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: Train 3-model ensemble + SHAP
# ─────────────────────────────────────────────────────────────────────────────

def train_and_shap(df, features):
    """
    Train LR / RF / HGB with 5-fold stratified CV.
    Compute SHAP on Random Forest (best explainability).
    Returns rf model, X matrix, SHAP summary DataFrame.
    """
    print("\n" + "=" * 65)
    print("  Training 3-Model Ensemble")
    print("=" * 65)

    X = df[features].fillna(df[features].median())
    y = df['turnaround']

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)

    lr  = LogisticRegression(max_iter=1000, C=1.0, random_state=42, n_jobs=-1)
    rf  = RandomForestClassifier(
        n_estimators=300, max_depth=12, min_samples_leaf=20,
        n_jobs=-1, random_state=42
    )
    hgb = HistGradientBoostingClassifier(
        max_iter=200, max_depth=5, learning_rate=0.1,
        min_samples_leaf=20, random_state=42
    )

    for name, model, Xm in [('Logistic Regression', lr, X_scaled),
                              ('Random Forest',       rf, X),
                              ('Gradient Boosting',   hgb, X)]:
        cv_auc = cross_val_score(model, Xm, y, cv=cv, scoring='roc_auc', n_jobs=-1)
        model.fit(Xm, y)
        train_auc = roc_auc_score(y, model.predict_proba(Xm)[:, 1])
        print(f"  {name:<22}  Train AUC={train_auc:.3f}  "
              f"CV AUC={cv_auc.mean():.3f} ± {cv_auc.std():.3f}")

    # SHAP on Random Forest — sample for speed (5000 tracts is plenty for stable estimates)
    shap_n = min(5000, len(X))
    X_shap = X.sample(shap_n, random_state=42)
    print(f"\n  Computing SHAP (TreeSHAP on RF, n={shap_n:,} sample)...")
    explainer = shap.TreeExplainer(rf)
    shap_vals = explainer.shap_values(X_shap)

    if isinstance(shap_vals, list):
        sv = shap_vals[1]
    elif isinstance(shap_vals, np.ndarray) and shap_vals.ndim == 3:
        sv = shap_vals[:, :, 1]
    else:
        sv = shap_vals

    summary = pd.DataFrame({
        'feature':         features,
        'mean_abs_shap':   np.abs(sv).mean(axis=0),
        'mean_shap':       sv.mean(axis=0),
    }).reset_index(drop=True).sort_values('mean_abs_shap', ascending=False)
    summary['rank'] = range(1, len(summary) + 1)
    summary['pct_total'] = (
        summary['mean_abs_shap'] / summary['mean_abs_shap'].sum() * 100
    )
    summary['direction'] = summary['mean_shap'].apply(
        lambda v: 'HELPS turnaround' if v > 0 else 'HURTS turnaround'
    )

    summary.to_parquet(PROCESSED / 'expanded_shap_summary.parquet', index=False)
    print(f"\n  TOP 25 FEATURES BY SHAP IMPORTANCE:")
    print(f"  {'Rank':>4} {'Feature':<50} {'SHAP%':>6}  {'Effect'}")
    print("  " + "-" * 80)
    for _, row in summary.head(25).iterrows():
        print(f"  {int(row['rank']):>4} {row['feature']:<50} "
              f"{row['pct_total']:>5.1f}%  {row['direction']}")

    # ── Save per-tract SHAP values + feature sample for beeswarm visualisation ─
    print(f"\n  Saving per-tract SHAP values ({shap_n:,} tracts × {len(features)} features)...")

    # sv has shape (shap_n, n_features) — SHAP contributions for turnaround class
    shap_vals_df = pd.DataFrame(sv, columns=features, index=X_shap.index)
    shap_vals_df.to_parquet(PROCESSED / 'expanded_shap_values.parquet', index=True)
    print(f"  Saved → expanded_shap_values.parquet  {shap_vals_df.shape}")

    # Feature values for the same sampled tracts + outcome label (for beeswarm colouring)
    shap_sample_df = X_shap.copy()
    shap_sample_df['turnaround'] = y.loc[X_shap.index]
    shap_sample_df.to_parquet(PROCESSED / 'expanded_shap_sample.parquet', index=True)
    print(f"  Saved → expanded_shap_sample.parquet  {shap_sample_df.shape}")

    # Save the trained RF model so SHAP can be recomputed without full retraining
    try:
        import joblib
        joblib.dump(rf, PROCESSED / 'expanded_rf_model.pkl')
        print(f"  Saved → expanded_rf_model.pkl")
    except Exception as _e:
        print(f"  Warning: could not save RF model ({_e})")

    return rf, X, y, summary


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: Compute directional gaps for a target geography
# ─────────────────────────────────────────────────────────────────────────────

def compute_gaps(df, features, shap_summary, target):
    """
    For every feature compute a SIGNED gap:
      positive  =  target area is WORSE than turnaround tracts  (needs investment)
      negative  =  target area is AHEAD of turnaround tracts    (existing strength)

    Gap is normalized to 0-100 scale using 5th-95th percentile of national distribution.

    target: 'delta' | county FIPS string (e.g. '28119')
    """
    # Resolve target mask
    df = df.copy()
    if 'county_fips5' not in df.columns:
        df['county_fips5'] = df['GEOID'].str[:5]

    if target == 'delta':
        mask = df['county_fips5'].isin(DELTA_COUNTY_FIPS)
        target_label = 'MS Delta Region'
    else:
        mask = df['county_fips5'] == str(target)
        name = DELTA_COUNTY_NAMES.get(str(target), f'County {target}')
        target_label = f'{name} County ({target})'

    target_df  = df[mask]
    turn_df    = df[df['turnaround'] == 1]
    shap_lkp   = shap_summary.set_index('feature')['pct_total'].to_dict()

    print(f"\n  Target: {target_label}  ({len(target_df):,} at-risk tracts)")
    print(f"  Turnaround reference: {len(turn_df):,} tracts")

    # Build direction lookup from registry
    direction_lkp = {row[0]: row[3] for row in FEATURE_REGISTRY}
    label_lkp     = {row[0]: row[1] for row in FEATURE_REGISTRY}
    cat_lkp       = {row[0]: row[2] for row in FEATURE_REGISTRY}

    rows = []
    for feat in features:
        if feat not in df.columns:
            continue
        higher_good = direction_lkp.get(feat, True)

        # Normalize using national 5th-95th percentile (cast booleans to float)
        col_vals = df[feat].astype(float)
        p05 = col_vals.quantile(0.05)
        p95 = col_vals.quantile(0.95)
        rng = p95 - p05 if p95 != p05 else 1.0

        def norm(v):
            n = (float(v) - p05) / rng * 100
            return float(np.clip(n, 0, 100))

        turn_mean   = float(turn_df[feat].astype(float).mean())
        target_mean = float(target_df[feat].astype(float).mean())

        if pd.isna(target_mean) or pd.isna(turn_mean):
            continue

        turn_norm   = norm(turn_mean)
        target_norm = norm(target_mean)

        if higher_good is True:
            # positive gap = target is BELOW turnaround = needs improvement
            gap = turn_norm - target_norm
        elif higher_good is False:
            # positive gap = target is ABOVE (worse) than turnaround = needs improvement
            gap = target_norm - turn_norm
        else:
            # Demographic context: compute direction-neutral distance
            # Positive = target has more of this than turnaround
            gap = target_norm - turn_norm

        rows.append({
            'feature':       feat,
            'label':         label_lkp.get(feat, feat),
            'category':      cat_lkp.get(feat, 'Other'),
            'higher_good':   higher_good,
            'shap_pct':      shap_lkp.get(feat, 0.0),
            'gap':           gap,
            'target_raw':    target_mean,
            'turn_raw':      turn_mean,
            'target_norm':   target_norm,
            'turn_norm':     turn_norm,
        })

    result = pd.DataFrame(rows).sort_values('shap_pct', ascending=False)
    result['color'] = result['category'].map(CATEGORY_COLORS).fillna('#888888')
    return result, target_label


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5: 2×2 Priority Matrix
# ─────────────────────────────────────────────────────────────────────────────

def plot_priority_matrix(gap_df, target_label, shap_thresh=3.0, gap_thresh=20,
                          top_n=25, out_dir=None):
    """
    2×2 Priority Matrix:
      Y-axis: SHAP importance (% of total predictive power)
      X-axis: Directional gap (positive = behind turnaround, negative = strength)

    Quadrants:
      TOP-RIGHT  : ACT NOW  — high importance AND behind turnaround
      TOP-LEFT   : PROTECT  — high importance AND already strong (ahead of turnaround)
      BOT-RIGHT  : SECOND WAVE — behind but lower ML importance
      BOT-LEFT   : MONITOR  — ahead and lower importance

    Only shows top_n features by SHAP for readability.
    Context-only (higher_good=None) features are shown as smaller diamonds.
    """
    if out_dir is None:
        out_dir = Path(__file__).resolve().parents[1] / 'presentation' / 'figures'
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Restrict to top_n by SHAP importance
    plot_df = gap_df.head(top_n).copy()

    BG   = '#ffffff'
    TEXT = '#1a1a2e'
    DIM  = '#666677'
    BORDER = '#cccccc'

    fig, ax = plt.subplots(figsize=(14, 8), facecolor=BG)
    ax.set_facecolor(BG)

    # ── Quadrant backgrounds ──────────────────────────────────────────────────
    quad_cfg = {
        # (high_shap, positive_gap): (bg_color, label_color, label_text)
        (True,  True):  ('#fde8e8', '#c0392b', 'ACT NOW\n(High importance + behind turnaround)'),
        (True,  False): ('#e8f8e8', '#1e8449', 'PROTECT\n(High importance + existing strength)'),
        (False, True):  ('#fff8e8', '#e67e22', 'SECOND WAVE\n(Low importance + behind turnaround)'),
        (False, False): ('#f5f5f5', '#888888', 'MONITOR\n(Low importance + existing strength)'),
    }
    x_lim = (-65, 105)
    y_max = plot_df['shap_pct'].max() * 1.25

    for (hs, pg), (bg_c, _, _) in quad_cfg.items():
        x0 = gap_thresh  if pg  else x_lim[0]
        x1 = x_lim[1]   if pg  else gap_thresh
        y0 = shap_thresh if hs  else 0
        y1 = y_max       if hs  else shap_thresh
        ax.fill_between([x0, x1], [y0, y0], [y1, y1], color=bg_c, alpha=0.5, zorder=0)

    ax.axhline(shap_thresh, color='#999', lw=1.5, ls='--', zorder=1)
    ax.axvline(gap_thresh,  color='#999', lw=1.5, ls='--', zorder=1)
    ax.axvline(0,           color='#ccc', lw=1.0, ls=':',  zorder=1)

    # Quadrant labels
    label_pos = {
        (True,  True):  (gap_thresh + 1,  shap_thresh + 0.15),
        (True,  False): (x_lim[0] + 1,    shap_thresh + 0.15),
        (False, True):  (gap_thresh + 1,   0.15),
        (False, False): (x_lim[0] + 1,     0.15),
    }
    for (hs, pg), (bg_c, tc, txt) in quad_cfg.items():
        xp, yp = label_pos[(hs, pg)]
        ax.text(xp, yp, txt, fontsize=9.5, color=tc, fontweight='bold',
                va='bottom', ha='left', zorder=2, alpha=0.75,
                path_effects=[pe.withStroke(linewidth=2.5, foreground=BG)])

    # ── Plot points ───────────────────────────────────────────────────────────
    texts = []
    for _, row in plot_df.iterrows():
        is_context = row['higher_good'] is None
        marker = 'D' if is_context else 'o'
        size   = 180 if is_context else 260
        alpha  = 0.65 if is_context else 0.92

        ax.scatter(row['gap'], row['shap_pct'],
                   s=size, color=row['color'], marker=marker,
                   alpha=alpha, zorder=5,
                   edgecolors='white', linewidths=1.5)

        # Format value pair label
        tgt  = row['target_raw']
        turn = row['turn_raw']
        cat  = row['category']
        if 'IGS' in cat:
            val_str = f"{tgt:.0f} vs {turn:.0f} pts"
        elif 'CrudePrev' in row['feature']:
            val_str = f"{tgt:.1f}% vs {turn:.1f}%"
        elif 'PCT_' in row['feature']:
            val_str = f"{tgt:.1f}% vs {turn:.1f}%"
        elif 'biz_' in row['feature']:
            val_str = f"{tgt:,.0f} vs {turn:,.0f}"
        elif row['feature'] == 'MedianFamilyIncome':
            val_str = f"${tgt:,.0f} vs ${turn:,.0f}"
        else:
            val_str = f"{tgt:.1f} vs {turn:.1f}"

        lbl = f"{row['label']}\n({val_str})"
        t = ax.text(row['gap'], row['shap_pct'], lbl,
                    fontsize=8.2, color=row['color'], fontweight='bold',
                    ha='center', va='center', zorder=6,
                    path_effects=[pe.withStroke(linewidth=2, foreground=BG)])
        texts.append(t)

    if HAS_ADJUST_TEXT:
        adjust_text(
            texts,
            x=plot_df['gap'].values,
            y=plot_df['shap_pct'].values,
            ax=ax,
            expand_points=(1.5, 1.6),
            expand_text=(1.3, 1.45),
            force_points=(0.4, 0.55),
            force_text=(0.5, 0.7),
            lim=1200,
            arrowprops=dict(arrowstyle='-', color='#666', lw=0.6, alpha=0.5,
                            shrinkA=8, shrinkB=4),
        )

    # ── Axis labels & formatting ──────────────────────────────────────────────
    ax.set_xlim(x_lim)
    ax.set_ylim(0, y_max)
    ax.set_xlabel(
        f'Gap from Turnaround  ◄ Strength (ahead)   |   Behind (needs investment) ►\n'
        f'(Positive = {target_label} is worse than turnaround tracts; '
        f'Negative = already stronger)',
        fontsize=11, labelpad=10, color=TEXT,
    )
    ax.set_ylabel('SHAP Importance (% of total predictive power)', fontsize=11, color=TEXT)
    ax.tick_params(labelsize=10, colors=TEXT)
    for sp in ax.spines.values():
        sp.set_edgecolor(BORDER)

    ax.set_title(
        f'Priority Matrix: {target_label}\n'
        f'SHAP Importance × Gap from Turnaround Tracts  '
        f'(Top {top_n} features by SHAP | SHAP threshold={shap_thresh}% | Gap threshold={gap_thresh})',
        fontsize=13, fontweight='bold', color=TEXT, pad=14,
    )

    # ── Legend ────────────────────────────────────────────────────────────────
    from matplotlib.lines import Line2D
    seen_cats = {}
    for _, row in plot_df.iterrows():
        c = row['category']
        if c not in seen_cats:
            seen_cats[c] = row['color']

    cat_handles = [
        mpatches.Patch(color=col, label=cat, alpha=0.85)
        for cat, col in seen_cats.items()
    ]
    marker_handles = [
        Line2D([0],[0], marker='o', color='w', markerfacecolor='#555', markersize=8,
               label='Measurable lever'),
        Line2D([0],[0], marker='D', color='w', markerfacecolor='#aaa', markersize=7,
               label='Context / demographic'),
    ]
    ax.legend(
        handles=marker_handles + cat_handles,
        fontsize=8, loc='upper left',
        bbox_to_anchor=(0.0, 1.0),
        facecolor=BG, edgecolor=BORDER, framealpha=0.95,
        ncol=2, columnspacing=0.8, handlelength=1.2,
    )

    slug = target_label.lower().replace(' ', '_').replace('/', '_').replace('(', '').replace(')', '')
    out_path = out_dir / f'priority_matrix_{slug}.png'
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor=BG)
    plt.close()
    print(f"\n  Figure saved → {out_path}")
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6: Print priority table (text output)
# ─────────────────────────────────────────────────────────────────────────────

def print_priority_table(gap_df, shap_thresh=3.0, gap_thresh=20):
    gap_df = gap_df.copy()
    gap_df['quadrant'] = gap_df.apply(
        lambda r: (
            'ACT NOW'     if r['shap_pct'] >= shap_thresh and r['gap'] >= gap_thresh else
            'PROTECT'     if r['shap_pct'] >= shap_thresh and r['gap'] <  gap_thresh else
            'SECOND WAVE' if r['shap_pct'] <  shap_thresh and r['gap'] >= gap_thresh else
            'MONITOR'
        ), axis=1
    )

    print(f"\n{'='*100}")
    print(f"  PRIORITY BREAKDOWN")
    print(f"{'='*100}")
    for quad in ['ACT NOW', 'PROTECT', 'SECOND WAVE', 'MONITOR']:
        sub = gap_df[gap_df['quadrant'] == quad].sort_values('shap_pct', ascending=False)
        if sub.empty:
            continue
        print(f"\n  ── {quad} ({'Invest here' if 'ACT' in quad else 'Guard this' if 'PROTECT' in quad else 'Plan next' if 'SECOND' in quad else 'Watch'}) ──")
        print(f"  {'Feature':<50} {'SHAP%':>6}  {'Gap':>7}  {'Target Raw':>12}  {'Turn. Raw':>10}  {'Direction'}")
        print(f"  {'-'*110}")
        for _, r in sub.head(20).iterrows():
            dir_label = ('+higher=better' if r['higher_good'] is True
                         else '-lower=better'  if r['higher_good'] is False
                         else 'context')
            print(f"  {r['label']:<50} {r['shap_pct']:>5.1f}%  {r['gap']:>+7.1f}  "
                  f"{r['target_raw']:>12.2f}  {r['turn_raw']:>10.2f}  {dir_label}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Expanded ML Priority Matrix')
    parser.add_argument('--target', default='delta',
                        help="'delta' or county FIPS (e.g. 28119)")
    parser.add_argument('--retrain', action='store_true',
                        help='Force retrain even if expanded_shap_summary.parquet exists')
    parser.add_argument('--shap_thresh', type=float, default=3.0,
                        help='SHAP %% threshold for top/bottom split (default 3.0)')
    parser.add_argument('--gap_thresh', type=float, default=20.0,
                        help='Gap threshold for left/right split (default 20)')
    parser.add_argument('--top_n', type=int, default=30,
                        help='Number of features to show in matrix (default 30)')
    args = parser.parse_args()

    shap_path    = PROCESSED / 'expanded_shap_summary.parquet'
    expanded_path = PROCESSED / 'expanded_model.parquet'

    # Build or load expanded dataset
    if not expanded_path.exists() or args.retrain:
        df = build_expanded_dataset()
    else:
        df = pd.read_parquet(expanded_path)
        print(f"  Loaded expanded dataset: {df.shape[0]:,} rows × {df.shape[1]} cols")

    features = get_ml_features(df)

    # Train or load SHAP
    if not shap_path.exists() or args.retrain:
        rf, X, y, shap_summary = train_and_shap(df, features)
    else:
        shap_summary = pd.read_parquet(shap_path)
        print(f"  Loaded SHAP summary: {len(shap_summary)} features")

    # Compute gaps for each target
    targets = args.target.split(',') if ',' in args.target else [args.target]
    for tgt in targets:
        tgt = tgt.strip()
        print(f"\n{'='*65}")
        print(f"  Computing gaps for target: {tgt}")
        gap_df, target_label = compute_gaps(df, features, shap_summary, tgt)
        print_priority_table(gap_df, args.shap_thresh, args.gap_thresh)
        plot_priority_matrix(gap_df, target_label,
                             shap_thresh=args.shap_thresh,
                             gap_thresh=args.gap_thresh,
                             top_n=args.top_n)

    print("\n  Done.")


if __name__ == '__main__':
    main()
