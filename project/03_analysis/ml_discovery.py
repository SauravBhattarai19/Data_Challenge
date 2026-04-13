"""
ml_discovery.py
---------------
Stage 2 of the redesigned analytical pipeline.

Three-model ensemble (Logistic Regression, Random Forest, Gradient Boosting) predicts
which at-risk tracts (IGS<45 in 2017) achieved turnaround by 2025.

Then computes SHAP values to:
  1. Rank features by actual predictive importance
  2. Cluster features into natural "vulnerability dimensions"
  3. Derive data-driven weights for each dimension (mean |SHAP|)

These SHAP-derived dimensions and weights replace the old literature-assumed
7-component Resilience Index weights. Every weight is now earned from data.

Outputs:
  data_processed/model_comparison.parquet    -- LR vs RF vs GBM AUCs
  data_processed/shap_values.parquet         -- per-tract SHAP values
  data_processed/shap_feature_summary.parquet -- feature importance + direction
  data_processed/shap_dimensions.parquet     -- clustered vulnerability dimensions
  data_processed/shap_derived_weights.json   -- dimension weights for RI

Run:
  python 03_analysis/ml_discovery.py
"""

import sys, warnings, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

try:
    import shap
except ImportError:
    import subprocess
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'shap'], check=True)
    import shap

from config import PROCESSED, DELTA_COUNTY_FIPS


def load_data():
    """Load the ML-ready dataset from Stage 1."""
    print("=" * 65)
    print("  Stage 2: ML Discovery + SHAP Vulnerability Dimensions")
    print("=" * 65)

    df = pd.read_parquet(PROCESSED / 'igs_improvement_model.parquet')
    print(f"  At-risk tracts: {len(df):,}")
    print(f"  Turnaround rate: {df['turnaround'].mean()*100:.1f}%")
    return df


def define_features(df):
    """
    Define the feature set. Uses 2017 IGS baselines + static health/climate/
    social features. All measured BEFORE the 2025 outcome -- no leakage.
    """
    # 2017 IGS sub-indicator baselines
    igs_2017_cols = [c for c in df.columns
                     if c.endswith('_2017') and c != 'igs_score_2017'
                     and 'Base' not in c and 'Tract' not in c]

    # Health features (static -- CDC PLACES cross-sectional)
    health_cols = [c for c in df.columns if 'CrudePrev' in c]

    # Healthcare access
    hpsa_cols = [c for c in ['pc_hpsa_score_max', 'mh_hpsa_score_max', 'fqhc_count']
                 if c in df.columns]

    # Social vulnerability
    svi_cols = [c for c in ['RPL_THEME1', 'RPL_THEME2', 'RPL_THEME3', 'RPL_THEME4']
                if c in df.columns]

    # Climate risk
    climate_cols = [c for c in ['RISK_SCORE', 'HWAV_RISKS', 'IFLD_RISKS',
                                'HRCN_RISKS', 'TRND_RISKS', 'DRGT_RISKS',
                                'EAL_SCORE']
                    if c in df.columns]

    # Environmental burden
    env_cols = [c for c in ['RPL_EBM', 'RPL_HVM'] if c in df.columns]

    # Business infrastructure
    biz_cols = [c for c in df.columns if c.startswith('biz_')]

    # Other context
    other_cols = [c for c in ['PovertyRate', 'LILATracts_1And10', 'POPULATION',
                              'BUILDVALUE']
                  if c in df.columns]

    all_features = (igs_2017_cols + health_cols + hpsa_cols + svi_cols +
                    climate_cols + env_cols + biz_cols + other_cols)
    # Deduplicate and filter to what exists
    all_features = list(dict.fromkeys(f for f in all_features if f in df.columns))

    print(f"  Feature count: {len(all_features)}")
    return all_features


def train_three_models(df, features):
    """
    Train LR, RF, Gradient Boosting on the same data with 5-fold CV.
    Returns models and comparison table.
    """
    print("\n" + "-" * 50)
    print("  Three-Model Comparison")
    print("-" * 50)

    X = df[features].fillna(df[features].median())
    y = df['turnaround']
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)

    lr  = LogisticRegression(max_iter=1000, C=1.0, random_state=42, n_jobs=-1)
    rf  = RandomForestClassifier(
        n_estimators=300, max_depth=12, min_samples_leaf=20,
        random_state=42, n_jobs=-1
    )
    # HistGradientBoostingClassifier is ~10x faster than GradientBoostingClassifier
    # and natively supports n_jobs parallelism
    xgb = HistGradientBoostingClassifier(
        max_iter=200, max_depth=5, learning_rate=0.1,
        min_samples_leaf=20, random_state=42
    )

    def _fit_and_cv(model, Xm, ym, cvm):
        cv_scores = cross_val_score(model, Xm, ym, cv=cvm, scoring='roc_auc', n_jobs=-1)
        model.fit(Xm, ym)
        train_auc = roc_auc_score(ym, model.predict_proba(Xm)[:, 1])
        return model, cv_scores, train_auc

    # Run all three models in parallel (uses threads; sklearn releases the GIL)
    results = Parallel(n_jobs=3, prefer='threads')(
        delayed(_fit_and_cv)(m, Xm, y, cv)
        for m, Xm in [(lr, X_scaled), (rf, X), (xgb, X)]
    )
    (lr, lr_cv, lr_train_auc), (rf, rf_cv, rf_train_auc), (xgb, xgb_cv, xgb_train_auc) = results

    print(f"  Logistic Regression:  Train AUC={lr_train_auc:.3f}  CV AUC={lr_cv.mean():.3f} +/- {lr_cv.std():.3f}")
    print(f"  Random Forest:        Train AUC={rf_train_auc:.3f}  CV AUC={rf_cv.mean():.3f} +/- {rf_cv.std():.3f}")
    print(f"  Gradient Boosting:    Train AUC={xgb_train_auc:.3f}  CV AUC={xgb_cv.mean():.3f} +/- {xgb_cv.std():.3f}")

    # ── Feature importance agreement ────────────────────────────────────────
    rf_fi = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False)
    lr_coefs = pd.Series(np.abs(lr.coef_[0]), index=features).sort_values(ascending=False)
    # HistGradientBoosting has no split-based feature_importances_; use permutation importance
    # on a subsample for speed (n_repeats=3, up to 3000 rows)
    _n = min(3000, len(X))
    _idx = np.random.default_rng(42).choice(len(X), _n, replace=False)
    _pi = permutation_importance(xgb, X.iloc[_idx], y.iloc[_idx],
                                 n_repeats=3, random_state=42, n_jobs=-1)
    xgb_fi = pd.Series(_pi.importances_mean, index=features).sort_values(ascending=False)

    top10_rf  = set(rf_fi.head(10).index)
    top10_xgb = set(xgb_fi.head(10).index)
    top10_lr  = set(lr_coefs.head(10).index)
    agreement_rf_xgb = len(top10_rf & top10_xgb)
    agreement_all = len(top10_rf & top10_xgb & top10_lr)
    print(f"\n  Top-10 feature agreement: RF-GBM={agreement_rf_xgb}/10, All-3={agreement_all}/10")

    # ── Save comparison ──────────────────────────────────────────────────────
    comparison = pd.DataFrame([
        {'model': 'Logistic Regression', 'train_auc': lr_train_auc,
         'cv_auc_mean': lr_cv.mean(), 'cv_auc_std': lr_cv.std(),
         'top5_features': ', '.join(lr_coefs.head(5).index)},
        {'model': 'Random Forest', 'train_auc': rf_train_auc,
         'cv_auc_mean': rf_cv.mean(), 'cv_auc_std': rf_cv.std(),
         'top5_features': ', '.join(rf_fi.head(5).index)},
        {'model': 'Gradient Boosting', 'train_auc': xgb_train_auc,
         'cv_auc_mean': xgb_cv.mean(), 'cv_auc_std': xgb_cv.std(),
         'top5_features': ', '.join(xgb_fi.head(5).index)},
    ])
    comparison.to_parquet(PROCESSED / 'model_comparison.parquet', index=False)
    print(f"\n  Saved model comparison -> model_comparison.parquet")

    return rf, X, y, scaler, features


def compute_shap_analysis(rf, X, y, features):
    """
    Compute SHAP values using TreeSHAP on the Random Forest.
    Returns SHAP values matrix and feature summary.
    """
    print("\n" + "-" * 50)
    print("  SHAP Analysis (TreeSHAP on Random Forest)")
    print("-" * 50)

    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X)

    # Handle different SHAP output formats:
    # - list of [class0_array, class1_array] (older shap)
    # - 3D array of shape (n_samples, n_features, 2) (newer shap)
    # - 2D array (regression or single-output)
    if isinstance(shap_values, list):
        sv = shap_values[1]  # SHAP for turnaround=1
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        sv = shap_values[:, :, 1]  # class 1 from 3D array
    else:
        sv = shap_values

    shap_df = pd.DataFrame(sv, columns=features, index=X.index)

    # ── Feature summary: mean |SHAP|, mean SHAP (direction), std ─────────
    summary = pd.DataFrame({
        'feature': features,
        'mean_abs_shap': np.abs(sv).mean(axis=0),
        'mean_shap': sv.mean(axis=0),
        'std_shap': sv.std(axis=0),
        'median_abs_shap': np.median(np.abs(sv), axis=0),
    }).sort_values('mean_abs_shap', ascending=False)
    summary['rank'] = range(1, len(summary) + 1)
    summary['pct_total'] = summary['mean_abs_shap'] / summary['mean_abs_shap'].sum() * 100

    print(f"\n  TOP 20 FEATURES BY MEAN |SHAP|:")
    print(f"  {'Rank':>4} {'Feature':<48} {'|SHAP|':>8} {'%Total':>7} {'Dir':>6}")
    print("  " + "-" * 75)
    for _, row in summary.head(20).iterrows():
        direction = '+' if row['mean_shap'] > 0 else '-'
        print(f"  {int(row['rank']):>4} {row['feature']:<48} "
              f"{row['mean_abs_shap']:>8.4f} {row['pct_total']:>6.1f}% {direction:>6}")

    # Save
    shap_df.to_parquet(PROCESSED / 'shap_values.parquet', index=False)
    summary.to_parquet(PROCESSED / 'shap_feature_summary.parquet', index=False)
    print(f"\n  Saved SHAP values -> shap_values.parquet ({shap_df.shape})")
    print(f"  Saved SHAP summary -> shap_feature_summary.parquet")

    return shap_df, summary


def discover_vulnerability_dimensions(shap_df, summary, features):
    """
    Cluster features into natural "vulnerability dimensions" based on how
    their SHAP values co-vary across tracts. Features that move together
    form a dimension.

    Weight each dimension by its total mean |SHAP| contribution.
    This produces DATA-DRIVEN weights instead of literature-assumed ones.
    """
    print("\n" + "-" * 50)
    print("  SHAP-Derived Vulnerability Dimensions")
    print("-" * 50)

    # Only cluster features with meaningful SHAP contribution
    # Use top features contributing to 95% of total |SHAP|
    cumulative = summary.sort_values('mean_abs_shap', ascending=False)['pct_total'].cumsum()
    n_keep = (cumulative <= 95).sum() + 1
    top_features = summary.sort_values('mean_abs_shap', ascending=False).head(n_keep)['feature'].tolist()
    print(f"  Features with 95% of SHAP mass: {len(top_features)}")

    # Compute correlation of SHAP values between features
    shap_corr = shap_df[top_features].corr(method='spearman').fillna(0)

    # Convert correlation to distance and cluster
    # distance = 1 - |correlation| (features that co-vary group together)
    dist_matrix = 1 - np.abs(shap_corr.values)
    np.fill_diagonal(dist_matrix, 0)
    dist_matrix = np.maximum(dist_matrix, 0)  # ensure non-negative

    # Hierarchical clustering
    condensed = squareform(dist_matrix, checks=False)
    Z = linkage(condensed, method='ward')

    # Cut into clusters -- use a distance threshold that produces 4-7 groups
    # Try different thresholds and pick the one giving 4-7 clusters
    best_n = 5
    best_t = 1.0
    for t in np.arange(0.5, 3.0, 0.1):
        labels = fcluster(Z, t=t, criterion='distance')
        n_clusters = len(set(labels))
        if 4 <= n_clusters <= 7:
            best_n = n_clusters
            best_t = t
            break

    cluster_labels = fcluster(Z, t=best_t, criterion='distance')
    print(f"  Clusters found: {best_n} (threshold={best_t:.1f})")

    # ── Assign semantic names to clusters based on their contents ─────────
    feature_clusters = pd.DataFrame({
        'feature': top_features,
        'cluster_id': cluster_labels,
        'mean_abs_shap': [summary[summary['feature'] == f]['mean_abs_shap'].values[0]
                          for f in top_features],
        'mean_shap': [summary[summary['feature'] == f]['mean_shap'].values[0]
                      for f in top_features],
    })

    # Name clusters by their DOMINANT feature category (weighted by mean |SHAP|)
    def name_cluster(cluster_features, cluster_shap_weights):
        """
        Assign a name based on which feature category accounts for the most
        SHAP weight in the cluster, not merely which categories are present.

        category_scores accumulates mean_abs_shap for each label bucket;
        the bucket with the highest total wins.
        """
        feat_weight = dict(zip(cluster_features, cluster_shap_weights))

        CLIMATE_FEATS = {'RISK_SCORE', 'HWAV_RISKS', 'IFLD_RISKS',
                         'HRCN_RISKS', 'TRND_RISKS', 'DRGT_RISKS', 'EAL_SCORE',
                         'BUILDVALUE', 'POPULATION'}
        ENV_FEATS     = {'RPL_EBM', 'RPL_HVM'}
        HPSA_FEATS    = {'pc_hpsa_score_max', 'mh_hpsa_score_max', 'fqhc_count'}
        OTHER_ECON    = {'PovertyRate', 'LILATracts_1And10'}

        scores = {
            'Health Burden':        0.0,
            'Social Vulnerability': 0.0,
            'Chronic Disease':      0.0,
            'Economic Baseline':    0.0,
            'Place Infrastructure': 0.0,
            'Community Capacity':   0.0,
            'Climate Risk':         0.0,
            'Environmental Burden': 0.0,
            'Healthcare Access':    0.0,
            'Business Infrastructure': 0.0,
        }

        for f, w in feat_weight.items():
            # Health — preventive/screening (no chronic disease)
            if 'CrudePrev' in f and not any(
                x in f for x in ['DIABETES', 'STROKE', 'OBESITY', 'COPD',
                                  'BPHIGH', 'CSMOKING', 'MHLTH', 'LPA',
                                  'MOBILITY', 'BINGE', 'ARTHRITIS', 'KIDNEY',
                                  'CASTHMA', 'CHD', 'DEPRESSION', 'TEETHLOST']):
                scores['Health Burden'] += w
            # Chronic disease (behavioural / cardiometabolic)
            if 'CrudePrev' in f and any(
                x in f for x in ['DIABETES', 'STROKE', 'OBESITY', 'COPD',
                                  'BPHIGH', 'CSMOKING', 'MHLTH', 'LPA',
                                  'MOBILITY', 'BINGE', 'ARTHRITIS', 'KIDNEY',
                                  'CASTHMA', 'CHD', 'DEPRESSION', 'TEETHLOST']):
                scores['Chronic Disease'] += w
            # SVI social vulnerability themes
            if 'RPL_THEME' in f:
                scores['Social Vulnerability'] += w
            # Climate / disaster risk
            if f in CLIMATE_FEATS:
                scores['Climate Risk'] += w
            # Environmental burden (EJI)
            if f in ENV_FEATS:
                scores['Environmental Burden'] += w
            # Healthcare access (HPSA / FQHC)
            if f in HPSA_FEATS or 'hpsa' in f.lower() or 'fqhc' in f.lower():
                scores['Healthcare Access'] += w
            # Business infrastructure
            if f.startswith('biz_'):
                scores['Business Infrastructure'] += w
            # IGS economic sub-indicators
            if '_2017' in f and any(
                x in f for x in ['Business', 'Commercial', 'Spend', 'economy',
                                  'Income', 'New Businesses']):
                scores['Economic Baseline'] += w
            # IGS place sub-indicators
            if '_2017' in f and any(
                x in f for x in ['Internet', 'Housing', 'Travel', 'Net Occ']):
                scores['Place Infrastructure'] += w
            # IGS community / labour sub-indicators
            if '_2017' in f and any(
                x in f for x in ['Insurance', 'Labor', 'Female', 'community']):
                scores['Community Capacity'] += w
            # Poverty / food-access context
            if f in OTHER_ECON:
                scores['Economic Baseline'] += w

        # Merge 'Chronic Disease' into a unified health label if it ties with
        # Health Burden, otherwise promote it to its own name for clarity
        top_cat = max(scores, key=scores.get)

        # Resolve ties: prefer more specific label
        if top_cat == 'Chronic Disease':
            return 'Chronic Disease Burden'
        if top_cat == 'Health Burden':
            # if Social Vulnerability is also significant (>25% of total), combine
            total = sum(scores.values()) or 1
            if scores['Social Vulnerability'] / total > 0.25:
                return 'Health & Social Vulnerability'
            return 'Health Burden'
        return top_cat

    # Build dimension summary
    dimensions = []
    for cid in sorted(feature_clusters['cluster_id'].unique()):
        cluster = feature_clusters[feature_clusters['cluster_id'] == cid]
        cluster_feats = cluster['feature'].tolist()
        total_shap = cluster['mean_abs_shap'].sum()
        top_feat = cluster.sort_values('mean_abs_shap', ascending=False).iloc[0]['feature']
        cluster_weights = cluster.set_index('feature')['mean_abs_shap'].to_dict()
        name = name_cluster(cluster_feats, [cluster_weights.get(f, 0.0) for f in cluster_feats])

        dimensions.append({
            'dimension_id': int(cid),
            'dimension_name': name,
            'n_features': len(cluster_feats),
            'features': ', '.join(cluster_feats),
            'top_feature': top_feat,
            'total_mean_abs_shap': total_shap,
            'features_list': cluster_feats,
        })

    dim_df = pd.DataFrame(dimensions)
    total_shap_all = dim_df['total_mean_abs_shap'].sum()
    dim_df['weight_pct'] = dim_df['total_mean_abs_shap'] / total_shap_all * 100
    dim_df['weight'] = dim_df['total_mean_abs_shap'] / total_shap_all
    dim_df = dim_df.sort_values('weight_pct', ascending=False)

    print(f"\n  {'Dimension':<25} {'#Feats':>6} {'Weight':>8} {'Top Feature':<40}")
    print("  " + "-" * 82)
    for _, row in dim_df.iterrows():
        print(f"  {row['dimension_name']:<25} {row['n_features']:>6} "
              f"{row['weight_pct']:>7.1f}% {row['top_feature']:<40}")

    # ── Save ─────────────────────────────────────────────────────────────────
    # Save dimension definitions (drop list column for parquet)
    dim_save = dim_df.drop(columns=['features_list'])
    dim_save.to_parquet(PROCESSED / 'shap_dimensions.parquet', index=False)

    # Save feature-to-cluster mapping
    feature_clusters.to_parquet(PROCESSED / 'shap_feature_clusters.parquet', index=False)

    # Save weights as JSON for app and further analysis
    weights = {}
    for _, row in dim_df.iterrows():
        weights[row['dimension_name']] = {
            'weight': round(float(row['weight']), 4),
            'features': row['features_list'],
            'top_feature': row['top_feature'],
        }
    with open(PROCESSED / 'shap_derived_weights.json', 'w') as f:
        json.dump(weights, f, indent=2)

    print(f"\n  Saved dimensions -> shap_dimensions.parquet")
    print(f"  Saved weights -> shap_derived_weights.json")

    return dim_df, feature_clusters


def compute_shap_interactions(rf, X, features):
    """Compute top SHAP interaction effects (which feature pairs matter together)."""
    print("\n" + "-" * 50)
    print("  SHAP Interaction Effects (top pairs)")
    print("-" * 50)

    # Sample for speed (interaction matrix is O(n * features^2))
    n_sample = min(2000, len(X))
    X_sample = X.sample(n_sample, random_state=42)

    explainer = shap.TreeExplainer(rf)
    interaction_values = explainer.shap_interaction_values(X_sample)

    if isinstance(interaction_values, list):
        iv = interaction_values[1]  # class 1
    elif isinstance(interaction_values, np.ndarray) and interaction_values.ndim == 4:
        iv = interaction_values[:, :, :, 1]  # class 1 from 4D
    else:
        iv = interaction_values

    # Mean absolute interaction strength for each pair — vectorized (no Python loop)
    mean_abs_iv = np.abs(iv).mean(axis=0)          # (n_feats, n_feats)
    i_idx, j_idx = np.triu_indices(len(features), k=1)
    pairs_df = pd.DataFrame({
        'feature_1': np.array(features)[i_idx],
        'feature_2': np.array(features)[j_idx],
        'interaction_strength': mean_abs_iv[i_idx, j_idx],
    }).sort_values('interaction_strength', ascending=False)

    print(f"\n  TOP 15 FEATURE INTERACTIONS:")
    print(f"  {'Feature 1':<35} {'Feature 2':<35} {'Strength':>10}")
    print("  " + "-" * 82)
    for _, row in pairs_df.head(15).iterrows():
        print(f"  {row['feature_1']:<35} {row['feature_2']:<35} "
              f"{row['interaction_strength']:>10.4f}")

    pairs_df.head(50).to_parquet(PROCESSED / 'shap_interaction_top.parquet', index=False)
    print(f"\n  Saved interactions -> shap_interaction_top.parquet")

    return pairs_df


def delta_analysis(rf, X, features, df):
    """Analyze Delta tracts specifically -- what does the model say about them?"""
    print("\n" + "-" * 50)
    print("  Delta Tract Predictions")
    print("-" * 50)

    if 'county_fips5' not in df.columns:
        print("  No county_fips5 -- skipping Delta analysis")
        return

    delta_mask = df['county_fips5'].isin(DELTA_COUNTY_FIPS)
    delta_idx = df[delta_mask].index.intersection(X.index)

    if len(delta_idx) == 0:
        print("  No Delta tracts in at-risk dataset")
        return

    X_delta = X.loc[delta_idx]
    probs = rf.predict_proba(X_delta)[:, 1]

    print(f"  Delta at-risk tracts: {len(X_delta)}")
    print(f"  Mean P(turnaround): {probs.mean():.3f}")
    print(f"  Max P(turnaround): {probs.max():.3f}")
    print(f"  Min P(turnaround): {probs.min():.3f}")
    print(f"  Actual turnaround rate: {df.loc[delta_idx, 'turnaround'].mean()*100:.1f}%")


if __name__ == "__main__":
    df = load_data()
    features = define_features(df)
    rf, X, y, scaler, features = train_three_models(df, features)
    shap_df, summary = compute_shap_analysis(rf, X, y, features)
    dim_df, feat_clusters = discover_vulnerability_dimensions(shap_df, summary, features)
    compute_shap_interactions(rf, X, features)
    delta_analysis(rf, X, features, df)
    print("\n  Stage 2 complete.")
