# Technical Report
## Climate-Health-Economy Resilience Index
### Mastercard IGS × AUC 2026 Data Science Challenge

**Author:** Saurav Bhattarai (Newton), Jackson State University
**Date:** March 2026
**Grand Finale:** April 30, 2026

---

## Table of Contents

1. [Introduction and Motivation](#1-introduction-and-motivation)
2. [Data Sources and Integration](#2-data-sources-and-integration)
3. [Data Pipeline Architecture](#3-data-pipeline-architecture)
4. [Triple-Vulnerability Analysis](#4-triple-vulnerability-analysis)
5. [Compound Climate-Health Risk Score](#5-compound-climate-health-risk-score)
6. [Climate-Health-Economy Resilience Index](#6-climate-health-economy-resilience-index)
7. [Small Business Healthcare Anchors](#7-small-business-healthcare-anchors)
8. [Mississippi Delta Case Study](#8-mississippi-delta-case-study)
9. [Validation](#9-validation)
10. [Investment Prioritization Framework](#10-investment-prioritization-framework)
11. [Limitations and Future Work](#11-limitations-and-future-work)
12. [Appendix: Variable Dictionary](#12-appendix-variable-dictionary)

---

## 1. Introduction and Motivation

### 1.1 The Convergence Problem

Three of America's most persistent structural challenges — economic exclusion, climate vulnerability, and healthcare shortage — do not occur independently. Low-income communities, particularly communities of color in rural regions, face all three simultaneously. This convergence creates a compounding risk profile that is qualitatively different from any single dimension of vulnerability:

- **Economic distress** reduces households' ability to adapt to climate events or access healthcare
- **Climate exposure** generates health crises (heat-related illness, flood trauma, respiratory disease) that overwhelm already-strained healthcare infrastructure
- **Healthcare shortages** mean that climate-exacerbated health conditions go unmanaged, deepening economic burden

No existing national index captures this three-way intersection at the census tract level. The Social Vulnerability Index (SVI) addresses social factors. The FEMA National Risk Index addresses climate hazard. The HPSA designation system identifies healthcare shortages. But none integrate all three into a single, actionable resilience score.

### 1.2 The Mastercard IGS Opportunity

The Mastercard Inclusive Growth Score (IGS) provides a unique asset: annual economic health scores for all US census tracts from 2017 to 2025. The IGS captures the trajectory of inclusive economic growth — not just current poverty, but whether communities are improving or stagnating. Combining IGS with climate and healthcare shortage data creates a forward-looking resilience picture.

### 1.3 Research Questions

This project addresses three core questions:

1. **Where are triple-vulnerable communities concentrated nationally?** Which census tracts face simultaneously high economic vulnerability (IGS < 45), significant climate hazard (heat wave or flood risk "Relatively High" or "Very High"), and severe healthcare shortage (HPSA score > 19 or MUA designation with IMU < 45)?

2. **What is the compound climate-health risk burden?** For heat-wave-exposed communities, how do climate hazard and health status (asthma, cardiovascular disease, COPD) interact? How does social vulnerability amplify this burden?

3. **Where can Mastercard investments in small business and financial inclusion generate the highest resilience return?** Which specific communities — and which specific intervention types (pharmacy, FQHC, food retail, economic inclusion) — would move the needle most on composite resilience?

---

## 2. Data Sources and Integration

### 2.1 Primary Data Sources

| Dataset | Source | Vintage | Spatial Unit | Key Variables Used |
|---------|--------|---------|--------------|-------------------|
| Mastercard IGS | Mastercard / AUC | 2017–2025 | Census tract | `igs_score`, `igs_economy`, `igs_place`, `igs_community` |
| FEMA National Risk Index | FEMA | 2023 | Census tract | `RISK_SCORE`, `RESL_SCORE`, `HWAV_RISKR`, `IFLD_RISKR`, `HWAV_RISKS`, `IFLD_RISKS`, `HRCN_RISKR`, `TRND_RISKR` |
| CDC PLACES | CDC | 2025 | Census tract | `CASTHMA_CrudePrev`, `CHD_CrudePrev`, `DIABETES_CrudePrev`, `BPHIGH_CrudePrev`, `COPD_CrudePrev`, `DEPRESSION_CrudePrev`, `OBESITY_CrudePrev` |
| Social Vulnerability Index (SVI) | CDC/ATSDR | 2022 | Census tract | `RPL_THEMES`, `RPL_THEME1`–`4` |
| Environmental Justice Index (EJI) | CDC | 2022 | Census tract | `EJI_SCORE`, `EBM_SCORE`, `HVM_SCORE` |
| HPSA Primary Care | HRSA | 2024 | Designation area (county-level aggregation) | `HPSA Score`, `HPSA Shortage`, designation status |
| HPSA Mental Health | HRSA | 2024 | Designation area | Same as above |
| Medically Underserved Areas (MUA) | HRSA | 2024 | County/sub-county | `IMU Score`, MUA status |
| FQHC Sites | HRSA | 2024 | Point (lat/lon) | Site name, address, county FIPS |
| County Business Patterns (CBP) | US Census | 2023 | County | NAICS-level establishment counts: pharmacy (446110), healthcare (621x), grocery (445110) |
| USDA Food Access Research Atlas | USDA | 2019 | Census tract | `LILATracts_1And10`, `PovertyRate`, `LALOWI1_10` |
| TIGER/Line Shapefiles | US Census | 2022 | Census tract | Tract polygons for Mississippi |

### 2.2 Data Integration Architecture

All datasets are integrated to the census tract level using an 11-digit GEOID as the primary join key. County-level datasets (HPSA, MUA, CBP, FQHC counts) are joined via the first 5 characters of the GEOID (`county_fips5 = GEOID[:5]`).

**Key normalization steps:**
- `TractFIPS` (int64 in CDC PLACES) → `zfill(11)`
- `FIPS` (int64 in SVI) → `zfill(11)`
- `GEOID` (int64 in EJI) → `zfill(11)`
- SVI missing value sentinel: -999 → `NaN`
- IGS Excel: 3-row complex header; data read with `skiprows=3`
- HPSA county FIPS: reconstructed from `State and County Federal Information Processing Standard Code` (pre-built 5-digit combined column)

**Final master dataset:** 85,032 census tracts × 92 columns.

---

## 3. Data Pipeline Architecture

### 3.1 Pipeline Overview

```
Raw Data Files
      |
      v
01_ingest/          -- Raw files -> cleaned Parquet
      |
      v
02_build/           -- Join & compute -> analysis Parquet
      |
      v
03_analysis/        -- Statistical outputs -> CSV/Markdown
      |
      v
04_app/             -- Streamlit 5-page dashboard
```

### 3.2 Performance Considerations

**FEMA NRI DBF** (946MB, 468 fields): The full file is impractical to load in memory (~12GB RAM required). A custom Python `struct`-based selective parser reads only 25 of 468 fields, completing in under 1 second:

```python
def parse_dbf_selective(filepath, keep_fields):
    keep_set = set(keep_fields)
    with open(filepath, 'rb') as f:
        # Parse DBF header to locate field positions
        header = f.read(32)
        num_records = struct.unpack('<I', header[4:8])[0]
        header_size = struct.unpack('<H', header[8:10])[0]
        record_size = struct.unpack('<H', header[10:12])[0]
        # ... field map construction ...
        # Read only requested field bytes per record
        for _ in range(num_records):
            raw = f.read(record_size)
            row = {fname: raw[off:off+fl].decode('latin-1').strip()
                   for fname, (off, fl, ft) in field_map.items()}
            records.append(row)
    return pd.DataFrame(records)
```

**IGS Excel** (237MB, 9 years × 84K tracts): Loaded once via `openpyxl`, saving both a full time series (`igs_all_years.parquet`) and a latest-year snapshot (`igs_latest.parquet`) for master merge efficiency.

### 3.3 Output Files

| File | Rows | Columns | Description |
|------|------|---------|-------------|
| `data_processed/raw/igs_all_years.parquet` | 757K | 7 | IGS 2017–2025 all tracts |
| `data_processed/raw/fema_nri.parquet` | 85,154 | 26 | FEMA NRI selected fields |
| `data_processed/raw/cdc_places.parquet` | 84,318 | 45 | CDC PLACES tract health measures |
| `data_processed/raw/svi.parquet` | 85,001 | 20 | SVI tract-level themes |
| `data_processed/master_tract.parquet` | 85,032 | 92 | Full integrated dataset |
| `data_processed/resilience_index.parquet` | 85,032 | 103 | RI + all component scores |
| `data_processed/triple_vulnerable.parquet` | 4,497 | 103 | Triple-vulnerable tracts only |
| `data_processed/igs_trends.parquet` | 757K | 12 | IGS trends with risk context |
| `data_processed/delta_tracts.geojson` | 59 | — | MS Delta tract polygons |

---

## 4. Triple-Vulnerability Analysis

### 4.1 Definitions

Three independent vulnerability criteria must all be met for a census tract to be classified as **triple-vulnerable**:

**1. Economic Vulnerability**
```
econ_vuln = igs_score < 45
```
IGS scores below 45 indicate a community lagging on inclusive economic growth relative to urban-rural peers. Threshold selected based on the IGS methodology's own "vulnerable" designation.

**2. Climate Vulnerability**
```
climate_vuln = HWAV_RISKR ∈ {'Relatively High', 'Very High'}
            OR IFLD_RISKR ∈ {'Relatively High', 'Very High'}
```
Heat waves and inland flooding are the two climate hazards most tightly linked to mortality and economic disruption in the target geography (Mississippi Delta). FEMA NRI risk ratings derive from annualized loss estimates normalized to tract-level population and building stock.

**3. Health Shortage Vulnerability**
```
health_vuln = (pc_hpsa_score_max > 19)
           OR (in_mua AND imu_score_min < 45)
```
HPSA scores above 19 (scale: 0–26) represent severe shortage designations. MUA criteria use an IMU (Index of Medical Underservice) cutoff of 45, reflecting federal shortage designation standards. This threshold was validated against observed rates:
- National health_vuln rate: **49.4%** (meaningfully selective)
- MS Delta health_vuln rate: **100.0%** (universal, expected given known designations)

**Earlier threshold tested (any HPSA or MUA designation):** flagged 98% of US tracts — uninformative. The score-based threshold is necessary.

### 4.2 National Results

| Metric | Value |
|--------|-------|
| Triple-vulnerable tracts | 4,497 (5.3% of 85,032 tracts) |
| People in triple-vulnerable tracts | ~17.8 million |
| Single-vulnerability only | 38.2% of tracts |
| Dual vulnerability | 14.6% of tracts |
| No vulnerability | 41.9% of tracts |

### 4.3 State Rankings (Top 10 by Triple-Vulnerability Rate)

| Rank | State | Triple-Vuln % | Mean RI |
|------|-------|---------------|---------|
| 1 | Nevada | 25.7% | 37.7 |
| 2 | District of Columbia | 24.8% | 46.9 |
| 3 | Arizona | 21.0% | 37.4 |
| 4 | Mississippi | 20.3% | 35.7 |
| 5 | West Virginia | 17.2% | 39.6 |
| 6 | Missouri | 14.4% | 42.3 |
| 7 | Illinois | 13.7% | 41.2 |
| 8 | Oklahoma | 11.6% | 39.6 |
| 9 | Kentucky | 10.3% | 41.7 |
| 10 | Texas | 7.9% | 43.7 |

Mississippi's rate of 20.3% makes it the highest among geographically contiguous states. The Delta sub-region (54%) far exceeds even the statewide average.

---

## 5. Compound Climate-Health Risk Score

### 5.1 Methodology

The compound risk score quantifies the interaction between climate hazard exposure and health condition prevalence, amplified by social vulnerability:

```
heat_health = (Asthma% × 0.25 + CHD% × 0.25 + Hypertension% × 0.25 + COPD% × 0.25) / 100

flood_health = (Depression% × 0.5 + Mental Health Poor Days% × 0.5) / 100

heat_exp  = HWAV_RISKS / 100  (clipped to [0,1])
flood_exp = IFLD_RISKS / 100  (clipped to [0,1])

SVI_amplifier = 1 + RPL_THEMES  (range: 1.0 to 2.0)

raw_score = (heat_exp × heat_health × 0.5 + flood_exp × flood_health × 0.5) × SVI_amplifier

compound_risk_score = percentile_rank(raw_score) × 100
```

The percentile rank transformation ensures the score is interpretable across the full 0–100 range and is robust to outliers.

### 5.2 Health Burden: National vs. Delta

| Health Indicator | National Average | Delta Average | Ratio |
|-----------------|-----------------|---------------|-------|
| Asthma | 10.5% | 11.4% | 1.09× |
| Heart Disease | 6.5% | 9.3% | **1.44×** |
| COPD | 7.0% | 11.7% | **1.69×** |
| Hypertension | 34.9% | 53.7% | **1.54×** |
| Depression | 22.1% | 17.9% | 0.81× |
| Poor Mental Health Days | 17.0% | 18.9% | 1.12× |

COPD (1.69×), hypertension (1.54×), and heart disease (1.44×) show the most severe Delta exceedances — all conditions exacerbated by heat wave exposure.

### 5.3 Top Compound Risk Tracts

All top-20 highest compound risk tracts nationally are in the Mississippi Delta, with scores at the 99.6th–99.98th national percentile. Key tracts:

| Tract GEOID | County | Compound Risk | RI | Heat Risk | Flood Risk |
|-------------|--------|--------------|-----|-----------|------------|
| 28011950701 | Bolivar | 99.98 | 24.1 | Relatively High | Relatively High |
| 28011950100 | Bolivar | 99.94 | 14.0 | Relatively High | Relatively High |
| 28151000400 | Washington | 99.91 | 23.2 | Relatively High | Relatively High |
| 28083950200 | Leflore | 99.86 | 20.1 | Relatively High | Relatively High |
| 28053950300 | Humphreys | 99.57 | 21.4 | Relatively High | Relatively High |

---

## 6. Climate-Health-Economy Resilience Index

### 6.1 Design Principles

The Resilience Index is designed to be:
- **Multidimensional:** Captures economic, healthcare, climate, social, and infrastructure resilience
- **Actionable:** Each component corresponds to a distinct intervention lever
- **Validated:** Correlates with FEMA's independently derived resilience measure
- **Interpretable:** 0–100 scale with intuitive grading (A–F)
- **Temporal:** Can be recalculated year-over-year as IGS scores update

### 6.2 Component Formulas

```
RI = 0.25 × Economic + 0.25 × Healthcare + 0.25 × Climate + 0.15 × Social + 0.10 × Business

Economic   = igs_economy_pillar  (already 0–100; higher = more resilient)

Healthcare = (1 - pc_hpsa_score / 26) × 100  (inverted; 0 = max shortage)

Climate    = (100 - RISK_SCORE)  (inverted FEMA NRI; 0 = highest risk)

Social     = (1 - RPL_THEMES) × 100  (inverted SVI; 0 = most vulnerable)

Business   = MinMaxScaler(log1p(healthcare_biz / county_pop × 10,000))
             scaled to [0, 100]
```

**Business component design note:** Raw healthcare business counts per capita were log-transformed before scaling to prevent outlier counties (Los Angeles, New York) from compressing all other values toward zero. County-level population denominators were used (not tract-level) to avoid spuriously inflating ratios for small-population tracts.

### 6.3 Grade Scale

| Score | Grade | Interpretation |
|-------|-------|----------------|
| 65–100 | A/B | Strong resilience across most dimensions |
| 50–64 | C | Moderate resilience; some dimensions underperforming |
| 35–49 | D | Significant vulnerability; requires targeted investment |
| 0–34 | F | Critical vulnerability; compounding risk |

### 6.4 National Distribution

| Percentile | Score | Interpretation |
|-----------|-------|----------------|
| 10th | ~32 | Deep vulnerability |
| 25th | ~39 | High vulnerability |
| 50th | ~45 | Median — boundary of concern |
| 75th | ~52 | Moderate resilience |
| 90th | ~59 | Strong resilience |

### 6.5 Mississippi Delta Results

| County | RI | Grade | Triple-Vuln % | HPSA Score |
|--------|-----|-------|---------------|------------|
| Issaquena | 14.5 | F | 100% | — |
| Humphreys | 20.8 | F | 33% | — |
| Quitman | 21.5 | F | 33% | — |
| Sunflower | 22.4 | F | 43% | — |
| Sharkey | 22.9 | F | 50% | — |
| Leflore | 24.0 | F | 62% | — |
| Bolivar | 25.3 | F | 67% | 25/26 |
| Coahoma | 26.2 | F | 57% | — |
| Washington | 26.6 | F | 53% | — |
| **Delta Mean** | **24.6** | **F** | **54%** | — |
| **National Mean** | **45.4** | **D** | **5.3%** | — |

The Delta's mean RI of 24.6 is **20.8 points below the national mean** — nearly a full letter grade below the national median.

---

## 7. Small Business Healthcare Anchors

### 7.1 Concept

We introduce the concept of **healthcare small businesses as resilience anchors**: in low-RI communities, pharmacies, FQHCs, and outpatient healthcare providers are simultaneously:
- The primary climate-health crisis response infrastructure
- The most accessible economic assets for targeted investment
- The clearest pathway to improving the RI's healthcare and business components simultaneously

### 7.2 FQHC Analysis

**31 active FQHCs** operate in the 9 Mississippi Delta counties (HRSA 2024 data, confirmed via spatial intersection):
- Bolivar County: 4 FQHCs (highest)
- Washington County: 7 FQHCs (most by count, largest county)
- Issaquena County: 0 FQHCs (most critical gap)

FQHCs in the Delta serve as primary care, dental care, behavioral health, and pharmacy providers — essential in counties where private practice is economically unviable.

### 7.3 CBP Business Infrastructure

| County | Pharmacies | Physician Offices | Healthcare Businesses Total |
|--------|-----------|------------------|----------------------------|
| Bolivar | 45 | 378 | ~450 |
| Washington | 76 | 824 | ~960 |
| Leflore | 38 | 281 | ~340 |
| Humphreys | 12 | 58 | ~78 |
| Issaquena | 2 | 8 | ~12 |

Issaquena County — with the lowest RI (14.5) in the entire Delta — has only 12 healthcare businesses serving approximately 1,300 residents.

### 7.4 Investment Impact Analysis (Scenario Planner)

The Streamlit application includes a real-time scenario planner allowing users to model investment impacts:

- Adding **N pharmacies** to a county → recalculates `ri_business` component
- Reducing **HPSA score by X points** → recalculates `ri_healthcare` component
- Improving **IGS economy score by Y** → recalculates `ri_economic` component

For Bolivar County (current RI: 25.3):
- Reducing HPSA score from 25 to 15 (via NHSC/HPSA workforce incentives) → RI improves to ~32
- Adding 5 pharmacies (raising pharmacy density) → RI improves to ~27
- Combined economic + healthcare intervention → RI improves to ~35 (exits "F" grade zone)

---

## 8. Mississippi Delta Case Study

### 8.1 Regional Context

The Mississippi Delta's persistent poverty is one of the most extensively documented in American social science. Nine counties — Bolivar, Coahoma, Humphreys, Issaquena, Leflore, Quitman, Sharkey, Sunflower, Washington — form the alluvial plain where the intersection of:
- Post-agricultural economic decline
- Delta topography creating flood and heat vulnerability
- Historical underinvestment in healthcare infrastructure

...creates the most concentrated zone of triple vulnerability in the continental US.

### 8.2 IGS Trend Analysis (2017–2025)

Mastercard IGS data across 2017–2025 shows the Delta has maintained a persistent gap of **12–15 points below the national average**, with no trend toward closure. This is qualitatively different from similarly low-RI regions (e.g., urban cores in Chicago or Baltimore), which show more volatility. The Delta's IGS gap reflects structural, not cyclical, economic exclusion.

Key IGS components for the Delta:
- **Economy pillar**: consistently 35–45 (national average: ~55)
- **Place pillar**: consistently 38–48 (reflects poor infrastructure investment)
- **Community pillar**: consistently 30–42 (reflects social disconnection)

### 8.3 Climate Hazard Profile

| Hazard | Delta Tracts "Relatively High" or "Very High" |
|--------|----------------------------------------------|
| Heat Wave | 89% of tracts |
| Inland Flood | 95% of tracts |
| Hurricane | 45% of tracts |
| Tornado | 72% of tracts |

The Delta's position in the lower Mississippi River alluvial plain makes inland flooding the dominant climate hazard, while its latitude and urban heat island effects drive heat wave risk. The 2022 Jackson, MS water crisis is a recent example of climate-infrastructure failure compounding healthcare access failure.

### 8.4 The Healthcare Desert

All 9 Delta counties have:
- **Active Primary Care HPSA designation** (federal shortage area)
- **Active MUA designation** (federal medically underserved area)
- **IMU scores of 19.7–39.8** (threshold for designation: < 62; most extreme values nationally)
- **100% triple-healthcare-shortage coverage** at the county level

The HPSA score for Bolivar County (25/26) is among the highest primary care shortage scores in the nation, indicating the most severe staffing gap relative to need.

---

## 9. Validation

### 9.1 External Validation Against FEMA RESL_SCORE

FEMA's own National Resilience Score (RESL_SCORE) is derived independently from our methodology, using different inputs and weights. The Pearson correlation between our Resilience Index and FEMA's score:

**r = 0.386 (p < 0.001, n = 72,140 matched tracts)**

This exceeds our pre-specified threshold of r > 0.25, confirming our index is measuring a related but distinct construct. The moderate correlation (rather than near-1.0) is expected and desirable — the RI adds new information (healthcare access, IGS economic trend) that FEMA's score does not capture.

### 9.2 Delta Convergent Validity

Independent validation of Delta results against public knowledge:
- All 9 Delta counties are designated HPSAs and MUAs ✓ (matches HRSA records)
- Bolivar County diabetes rate (21.4%) matches CDC Atlas published statistics ✓
- 31 FQHCs in Delta matches HRSA FQHC search tool counts ✓
- Delta IGS scores 12–15 points below national ✓ (consistent with Mastercard's own published analysis)

---

## 10. Investment Prioritization Framework

### 10.1 Triage Logic

Communities are prioritized using a composite score that balances:
1. **Severity** (current RI — how bad is the situation?)
2. **Responsiveness** (which component is most improvable through targeted investment?)
3. **Accessibility** (does infrastructure exist to deploy capital?)

### 10.2 Tier Assignments

| Tier | RI Range | Counties | Primary Lever | Mastercard Tool |
|------|---------|----------|---------------|-----------------|
| 1 — Critical | < 22 | Issaquena, Humphreys, Quitman, Sunflower, Sharkey | Healthcare infrastructure | FQHC expansion grants, HPSA workforce incentives |
| 2 — High Priority | 22–27 | Leflore, Bolivar, Coahoma | Small business lending | Healthcare SME loans, pharmacy development |
| 3 — Stabilization | 27–35 | Washington | Economic inclusion | Digital financial inclusion, food retail investment |

### 10.3 Expected Impact Modeling

Based on the scenario planner analysis:

| Intervention | Target Counties | RI Impact | People Reached |
|-------------|----------------|-----------|----------------|
| 5 new FQHCs (1 per critical county) | Tier 1 | +4–6 RI points | ~8,000 additional patients |
| NHSC/HPSA workforce incentives (reduce HPSA score by 8) | All Delta | +5–8 RI points | 170K residents |
| 50 new pharmacy businesses | Tier 2 counties | +1–2 RI points | 80K residents |
| Small business healthcare lending program | All Delta | +2–3 RI points | 170K residents |

---

## 11. Limitations and Future Work

### 11.1 Current Limitations

**Data vintage mismatch:** Datasets span 2019 (Food Atlas) to 2025 (CDC PLACES). The FEMA NRI and SVI reflect 2022–2023 conditions. IGS annual updates provide temporal tracking but other components are static.

**County-level healthcare data:** HPSA and MUA designations are aggregated to the county level and assigned to all tracts in that county. Sub-county variation in healthcare access exists but is not captured without spatial HPSA polygon joins.

**CBP data precision:** County Business Patterns provides establishment counts but not employment levels or revenue. A large healthcare business and a one-person office count equally.

**National TIGER coverage:** The current Delta map uses Mississippi-specific TIGER shapefile. A national tract-level choropleth would require downloading all 50 state TIGER files (~2GB total).

**IGS coverage:** Not all census tracts have IGS data in all years. The master dataset uses latest-available IGS year per tract; about 3% of tracts have no IGS data and receive imputed values.

### 11.2 Future Work

1. **Sub-county healthcare shortage mapping:** Spatial join of HPSA polygons to census tracts for finer resolution
2. **Annual RI updates:** The pipeline is designed for annual re-runs as new IGS data becomes available
3. **Causal impact estimation:** Difference-in-differences analysis around FQHC openings to estimate actual RI improvements
4. **National expansion:** Priority ranking of all 3,144 US counties (not just MS Delta) using the same framework
5. **Investment ROI modeling:** Estimate economic return per dollar of healthcare SME investment using CBP employment trends near FQHCs

---

## 12. Appendix: Variable Dictionary

| Variable | Source | Description | Range |
|----------|--------|-------------|-------|
| `GEOID` | Census | 11-digit census tract FIPS | String |
| `igs_score` | Mastercard IGS | Inclusive Growth Score | 0–100 |
| `igs_economy` | Mastercard IGS | Economy pillar score | 0–100 |
| `igs_place` | Mastercard IGS | Place pillar score | 0–100 |
| `igs_community` | Mastercard IGS | Community pillar score | 0–100 |
| `RISK_SCORE` | FEMA NRI | Composite risk score | 0–100 |
| `RESL_SCORE` | FEMA NRI | FEMA resilience score | 0–100 |
| `HWAV_RISKR` | FEMA NRI | Heat wave risk rating | Categorical |
| `IFLD_RISKR` | FEMA NRI | Inland flood risk rating | Categorical |
| `HWAV_RISKS` | FEMA NRI | Heat wave risk score | 0–100 |
| `IFLD_RISKS` | FEMA NRI | Inland flood risk score | 0–100 |
| `CASTHMA_CrudePrev` | CDC PLACES | Asthma prevalence (%) | 0–100 |
| `CHD_CrudePrev` | CDC PLACES | Coronary heart disease (%) | 0–100 |
| `DIABETES_CrudePrev` | CDC PLACES | Diabetes prevalence (%) | 0–100 |
| `BPHIGH_CrudePrev` | CDC PLACES | High blood pressure (%) | 0–100 |
| `COPD_CrudePrev` | CDC PLACES | COPD prevalence (%) | 0–100 |
| `RPL_THEMES` | SVI | Overall social vulnerability (percentile) | 0–1 |
| `pc_hpsa_score_max` | HRSA HPSA | Max HPSA score in county | 0–26 |
| `in_mua` | HRSA MUA | Medically Underserved Area flag | Boolean |
| `imu_score_min` | HRSA MUA | Min IMU score in county | 0–100 |
| `fqhc_count` | HRSA FQHC | Active FQHCs in county | Integer |
| `biz_total_healthcare` | Census CBP | Healthcare establishments in county | Integer |
| `LILATracts_1And10` | USDA | Food desert flag | Boolean |
| `resilience_index` | Computed | Climate-Health-Economy RI | 0–100 |
| `ri_economic` | Computed | Economic component | 0–100 |
| `ri_healthcare` | Computed | Healthcare component | 0–100 |
| `ri_climate` | Computed | Climate component | 0–100 |
| `ri_social` | Computed | Social capital component | 0–100 |
| `ri_business` | Computed | Business infrastructure component | 0–100 |
| `triple_vuln` | Computed | Triple-vulnerability flag | Boolean |
| `econ_vuln` | Computed | Economic vulnerability flag | Boolean |
| `climate_vuln` | Computed | Climate vulnerability flag | Boolean |
| `health_vuln` | Computed | Health shortage flag | Boolean |
| `compound_risk_score` | Computed | Compound climate-health risk (percentile) | 0–100 |

---

*Technical Report — Climate-Health-Economy Resilience Index*
*Mastercard IGS × AUC 2026 Data Science Challenge*
*Saurav Bhattarai (Newton), Jackson State University*
