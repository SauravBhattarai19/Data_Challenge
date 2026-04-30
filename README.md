# Healthy Economies, Healthy Communities

**Mastercard IGS × AUC 2026 Data Science Challenge — Jackson State University**

> **Live app:** [howtoimprove.streamlit.app](https://howtoimprove.streamlit.app)  
> **Report:** `project/report/report.pdf` (9 pages, start here)

---

## The Question

Among the **25,142 US census tracts** with an Inclusive Growth Score (IGS) below 45 in 2017, only **34%** crossed that threshold by 2025. The Mississippi Delta's rate was just **24%**. What, precisely, separates communities that escaped from those that stayed stuck?

## The Answer

We trained a 3-model machine-learning ensemble on **89 granular features** from 10 public datasets and applied TreeSHAP to identify the exact drivers. The headline finding:

| Rank | Feature | SHAP Share | Delta vs. Turnaround |
|------|---------|-----------|----------------------|
| 1 | Dental Visit Rate | 9.2% | Delta: 6.1% · Turnaround avg: 56.9% |
| 2 | Commercial Diversity Score | 8.8% | Delta severely lagging |
| 3 | Residential Property Value | 6.7% | — |
| 4 | Physical Inactivity Rate | 6.4% | — |

**Dental access** is not a proxy for "health" alone — it predicts economic connectivity (r = 0.469 with internet access), workforce productivity, and healthcare system engagement. All 9 Delta counties are 100% HPSA-designated.

---

## App Structure (4 pages)

| Page | What it does |
|------|-------------|
| **Home** | Problem framing, key statistics, IGS trajectory 2017–2025 |
| **1 · IGS Landscape** | National county choropleth; filter by state or pillar |
| **2 · Delta Deep Dive** | County/tract drilldown, health tabs, Folium tract map with FQHC overlay |
| **3 · What Drives Turnaround?** | SHAP importance bar chart, beeswarm plot, category breakdown |
| **4 · Investment Priority Matrix** | 2×2 Act Now / Protect / Second Wave / Monitor per county |

---

## Model Performance

| Model | CV ROC-AUC | Role |
|-------|-----------|------|
| Logistic Regression | 0.854 ± 0.005 | Linear baseline |
| Random Forest | 0.864 ± 0.007 | TreeSHAP source |
| Gradient Boosting | 0.876 ± 0.006 | Sequential learning |

5-fold stratified cross-validation · 25,142 at-risk tracts · TreeSHAP on 5,000-tract sample

---

## Project Structure

```
project/
├── config.py                         # All paths, thresholds, constants
├── 01_ingest/                        # Raw data → Parquet (one script per source)
├── 02_build/                         # Dataset joins and feature engineering
├── 03_analysis/
│   └── expanded_priority_matrix.py   # ML pipeline, FEATURE_REGISTRY, compute_gaps()
├── 04_app/
│   ├── app.py                        # Home page
│   ├── pages/
│   │   ├── 1_IGS_Landscape.py
│   │   ├── 2_Delta_Deep_Dive.py
│   │   ├── 3_ML_Discovery.py
│   │   └── 4_Priority_Matrix.py
│   └── components/                   # theme, charts, maps, tables
├── data_processed/                   # Pre-computed Parquet files for the app
└── report/
    ├── report.pdf                    # Full 9-page write-up
    ├── report.tex
    └── generate_figures.py           # Reproduces all 6 publication figures
```

---

## Data Sources

| Dataset | Producer | Year | Geography |
|---------|----------|------|-----------|
| Mastercard IGS | Mastercard Economics Institute | 2017–2025 | Census tract |
| CDC PLACES | CDC / BRFSS | 2022 | Census tract |
| CDC SVI | ATSDR | 2022 | Census tract |
| USDA Food Atlas | USDA ERS | 2019 | Census tract |
| FEMA NRI | FEMA | 2023 | Census tract |
| CDC EJI | CDC ATSDR | 2022 | Census tract |
| HRSA HPSA / MUA | HRSA | 2024 | County |
| AHRF | HRSA BHPr | 2024–25 | County |
| Census CBP / ZBP | US Census Bureau | 2023 | County / ZIP |
| SBA Rankings | Small Business Administration | 2025 | State |

---

## Running Locally

```bash
pip install -r requirements.txt

# Launch the app (processed data already in data_processed/)
cd project/04_app
streamlit run app.py
```

To reproduce the full pipeline from raw data (requires all 10 source files in `Data/`):

```bash
cd project/
python 01_ingest/run_all_ingest.py
python 02_build/build_master_tract.py
python 02_build/build_igs_national.py
python 02_build/build_delta_profile.py
python 03_analysis/expanded_priority_matrix.py
```

Figures for the report:

```bash
python project/report/generate_figures.py
cd project/report && pdflatex report.tex
```
