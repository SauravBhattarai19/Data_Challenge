# Healthy Economies, Healthy Communities

**Mastercard IGS x AUC 2026 Data Science Challenge**

A data pipeline and interactive dashboard that builds a **Climate-Health-Economy Resilience Index** across ~85,000 US census tracts, with a deep focus on the **Mississippi Delta** — identifying where small-business investment can generate the highest simultaneous return across economic, health, and climate dimensions.

---

## Problem

Climate change, healthcare access, and economic fragility converge in the same communities. Nationally, **4,497 census tracts (5.3%)** are triple-vulnerable — and 54% of Mississippi Delta tracts fall into this category. All 9 Delta counties score Grade F on compound risk, with a mean Resilience Index of 24.6/100 (vs. 45.4 national average).

## Approach

1. **Integrate** 13 public datasets at census-tract level (92 features)
2. **Classify** tracts into a 2x2 typology based on IGS trajectory (2017-2025): Resilient, Turnaround, Declining, Stuck
3. **Discover** what drives turnaround via 3-model ML ensemble + SHAP analysis
4. **Prescribe** targeted small-business interventions using empirical turnaround benchmarks and OLS regression coefficients

---

## Data Sources

| Source | Geography | Key Variables |
|--------|-----------|---------------|
| Mastercard IGS (2017-2025) | Census tract | IGS score, 3 pillars, 14 sub-indicators |
| FEMA National Risk Index | Census tract | Risk, resilience, 7 hazard-specific scores |
| CDC PLACES 2025 | Census tract | 20+ health outcomes (diabetes, asthma, hypertension, etc.) |
| CDC/ATSDR SVI 2022 | Census tract | 4 social vulnerability themes |
| CDC EJI | Census tract | Environmental justice, burden, health vulnerability |
| HRSA HPSA | County | Primary care & mental health shortage scores |
| HRSA MUA | County | Medically underserved area designations, IMU scores |
| HRSA FQHC | Site/County | Federally qualified health center locations |
| Census CBP 2023 | County | Business counts, employment, sector diversity |
| Census ZBP 2023 | ZIP | Small business, healthcare establishment counts |
| SBA State Rankings 2025 | State | Small business metrics and rankings |
| USDA Food Atlas 2019 | Census tract | Food desert flags, SNAP, demographics |
| HRSA AHRF 2024-2025 | County | Physician counts, hospital beds, Medicare costs |

---

## Project Structure

```
project/
├── config.py                  # Central configuration, thresholds, data paths
├── verify_setup.py            # Pre-flight data & dependency checker
├── 01_ingest/                 # Stage 1: Raw data -> Parquet (13 scripts)
│   ├── run_all_ingest.py      # Orchestrator
│   ├── ingest_igs.py          # Mastercard IGS (complex 3-row header)
│   ├── ingest_fema_nri.py     # FEMA NRI (custom DBF parser for 946MB file)
│   ├── ingest_cdc_places.py   # CDC health outcomes
│   ├── ingest_svi.py          # Social Vulnerability Index
│   ├── ingest_eji.py          # Environmental Justice Index
│   ├── ingest_hpsa.py         # Health Professional Shortage Areas
│   ├── ingest_mua.py          # Medically Underserved Areas
│   ├── ingest_fqhc.py         # Health center locations
│   ├── ingest_cbp.py          # County Business Patterns
│   ├── ingest_zbp.py          # ZIP Business Patterns
│   ├── ingest_sba.py          # SBA state rankings
│   ├── ingest_food.py         # USDA Food Access Atlas
│   └── ingest_ahrf.py         # Area Health Resources File
├── 02_build/                  # Stage 2: Join & compute
│   ├── build_master_tract.py  # 85K tracts x 92 columns
│   ├── build_igs_national.py  # County-level IGS aggregation
│   ├── build_igs_trends.py    # Tract-year trajectories (757K rows)
│   ├── build_delta_profile.py # Delta-specific enriched dataset
│   └── build_delta_geojson.py # GeoJSON for mapping (requires TIGER)
├── 03_analysis/               # Stage 3: ML + statistics
│   ├── run_all_analysis.py    # Orchestrator
│   ├── community_typology.py  # 2x2 typology + turnaround benchmarks
│   ├── ml_discovery.py        # LR/RF/GBM ensemble + SHAP
│   └── igs_regression_analysis.py  # OLS for investment simulator
├── 04_app/                    # Stage 4: Streamlit dashboard
│   ├── app.py                 # Home page
│   ├── pages/
│   │   ├── 1_IGS_Landscape.py        # National choropleth
│   │   ├── 2_Delta_Deep_Dive.py       # County/tract drilldown
│   │   ├── 3_ML_Discovery.py          # SHAP & model insights
│   │   ├── 4_The_Prescription.py      # Gap analysis & priorities
│   │   └── 5_Small_Business_Solutions.py  # Interventions & simulator
│   └── components/
│       ├── theme.py           # CSS styling & navigation
│       ├── charts.py          # Plotly chart builders
│       ├── maps.py            # Choropleth & Folium maps
│       └── tables.py          # Data tables
├── notebooks/
│   └── full_analysis.ipynb    # Exploratory analysis notebook
├── presentation/              # Presentation figure generators
│   └── figures/               # Generated PNG figures
├── reports/
│   ├── executive_summary.md   # Key findings & recommendations
│   └── technical_report.md    # Full methodology & results
└── scripts/
    └── download_counties_geojson.py  # Optional county boundaries
```

---

## Key Findings

- **Triple-vulnerable hotspot**: 54% of MS Delta tracts are simultaneously economically fragile, health-burdened, and climate-exposed
- **Health crisis**: Delta diabetes prevalence 21.4% (vs. 12.5% national), hypertension 52% (vs. 34.9%)
- **Healthcare desert**: All 9 Delta counties are designated HPSAs; HPSA scores up to 25/26
- **Turnaround is possible**: ~20-25% of at-risk tracts (IGS < 45 in 2017) achieved turnaround by 2025
- **SHAP-identified drivers**: Internet access, commercial diversity, health insurance coverage, and labor market engagement are the strongest predictors of turnaround
- **31 FQHCs** serve all 9 Delta counties — healthcare SMBs are resilience anchors

---

## Resilience Index

Five dimensions weighted by SHAP-derived importance:

| Dimension | Source | Weight |
|-----------|--------|--------|
| Economic Vitality | Mastercard IGS Economy pillar | 25% |
| Healthcare Access | HRSA HPSA (inverted) | 25% |
| Climate Risk | FEMA NRI (inverted) | 25% |
| Social Capital | CDC SVI | 15% |
| Business Infrastructure | Census CBP | 10% |

Validated against FEMA RESL_SCORE (r = 0.386, p < 0.001).

---

## Getting Started

### Prerequisites

- Python 3.10+
- Raw data files placed in `Data/` directory (see `verify_setup.py` for required files)

### Setup

```bash
# Option A: Conda
conda env create -f project/environment.yml
conda activate resilience-index

# Option B: Pip
pip install -r requirements.txt

# Verify data files exist
cd project/
python verify_setup.py
```

### Run the Pipeline

```bash
cd project/

# Stage 1: Ingest raw data (13 sources -> Parquet)
python 01_ingest/run_all_ingest.py

# Stage 2: Build integrated datasets
python 02_build/build_master_tract.py
python 02_build/build_igs_national.py
python 02_build/build_igs_trends.py
python 02_build/build_delta_profile.py
python 02_build/build_delta_geojson.py

# Stage 3: Analysis (typology + ML + regression)
python 03_analysis/run_all_analysis.py

# Stage 4: Launch dashboard
cd 04_app/
streamlit run app.py
```

Total pipeline runtime: ~30-45 minutes (FEMA NRI ingestion and SHAP computation are the bottlenecks).

---

## Dashboard Pages

| Page | Description |
|------|-------------|
| **Home** | Key statistics, typology breakdown, IGS trends |
| **IGS Landscape** | National county-level choropleth with state/sub-indicator filters |
| **Delta Deep Dive** | County and tract-level drilldown with radar charts, health outcomes, and FQHC maps |
| **What Drives Turnaround?** | ML model comparison, SHAP feature importance, vulnerability dimensions |
| **The Prescription** | Gap-to-turnaround analysis per county with projected IGS impact |
| **Small Business Solutions** | Concrete interventions mapped to IGS sub-indicators with investment simulator |

---

## ML Methodology

**Goal**: Predict which at-risk tracts (IGS < 45 in 2017) achieve turnaround by 2025.

- **Models**: Logistic Regression, Random Forest, Gradient Boosting (cross-validated AUC 0.68-0.74)
- **Features**: ~60 variables spanning IGS sub-indicators, health outcomes, climate risk, social vulnerability, and business infrastructure
- **SHAP analysis**: Identifies 5 vulnerability dimensions via hierarchical clustering of SHAP correlation matrix
- **Investment simulator**: OLS regression coefficients translate sub-indicator improvements into projected IGS gains

---

## Investment Framework

### Priority Tiers (MS Delta)

| Tier | Counties | RI Range | Strategy |
|------|----------|----------|----------|
| **Tier 1** (Critical) | Issaquena, Humphreys, Quitman, Sunflower, Sharkey | < 25 | Foundational infrastructure |
| **Tier 2** (High) | Leflore, Bolivar, Coahoma | 25-30 | Capacity building |
| **Tier 3** (Moderate) | Washington | 30-35 | Growth acceleration |

### Intervention Examples

Each IGS sub-indicator gap maps to concrete small-business opportunities:

- **Internet Access** gap -> Community broadband co-ops, digital literacy centers
- **Commercial Diversity** gap -> Business incubators, food trucks, e-commerce hubs
- **Health Insurance Coverage** gap -> ACA navigators, telehealth kiosks
- **Business Loans** gap -> CDFI partnerships, microloan programs

---

## Dependencies

Core: `pandas`, `numpy`, `scipy`, `scikit-learn`, `statsmodels`, `shap`
Visualization: `streamlit`, `plotly`, `folium`, `streamlit-folium`, `pydeck`
Data I/O: `openpyxl`, `pyarrow`, `dbfread`, `geopandas`

See [requirements.txt](requirements.txt) or [project/environment.yml](project/environment.yml) for full list.
