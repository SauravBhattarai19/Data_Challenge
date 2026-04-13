# Executive Summary
## Climate-Health-Economy Resilience Index
### Mastercard IGS × AUC 2026 Data Science Challenge

---

## The Problem

Across the United States, economic distress, climate exposure, and healthcare shortages rarely occur in isolation. When all three converge in a single community, the result is a compounding vulnerability that no single intervention can address — and that no existing public index fully captures.

**This project builds the first nationally comprehensive Climate-Health-Economy Resilience Index (RI)** — a 0–100 composite score integrating five dimensions of community resilience across all ~85,000 US census tracts. The index identifies communities where Mastercard's inclusive growth investments can generate the highest simultaneous return across economic, health, and climate dimensions.

---

## Key Findings

### National Scale
| Finding | Value |
|---------|-------|
| Triple-vulnerable tracts (economic + climate + healthcare) | **4,497 tracts (5.3%)** |
| People living in triple-vulnerable tracts | **17.8 million Americans** |
| National mean Resilience Index | **45.4 / 100** |
| States with highest triple-vulnerability rates | Nevada (25.7%), D.C. (24.8%), Arizona (21.0%), **Mississippi (20.3%)** |

### The Mississippi Delta: A National Epicenter

The Mississippi Delta — 9 counties along the alluvial plain — represents the most concentrated zone of triple vulnerability in the United States:

| Metric | MS Delta | National |
|--------|----------|----------|
| Mean Resilience Index | **24.6 / 100** | 45.4 / 100 |
| Triple-vulnerable tracts | **54%** | 5.3% |
| Healthcare shortage coverage | **100%** | 49.4% |
| County grades | **All 9: Grade F** | — |
| Compound risk score (national percentile) | **98–100th** | — |

Every single Delta county carries a primary care HPSA Score above 19 (maximum: 26). Every county is a federally designated Medically Underserved Area. Mean IGS scores have remained 12–15 points below the national average since 2017.

---

## The Index: What We Measure

The Resilience Index combines five independently validated components:

| Component | Weight | Data Source | What It Captures |
|-----------|--------|-------------|-----------------|
| **Economic** | 25% | Mastercard IGS Economy Pillar | Business climate, employment, GDP growth |
| **Healthcare Access** | 25% | HRSA HPSA Score (inverted) | Primary care shortage severity |
| **Climate Risk** | 25% | FEMA NRI Risk Score (inverted) | Heat wave, flood, hurricane, tornado exposure |
| **Social Capital** | 15% | CDC/ATSDR SVI RPL_THEMES | Poverty, race, housing, disability burden |
| **Business Infrastructure** | 10% | Census CBP 2023 (log-normalized) | Healthcare/pharmacy business density |

**Validation:** Our RI correlates with FEMA's independently derived RESL_SCORE at r = 0.386 (p < 0.001), exceeding the 0.25 pre-specified threshold.

---

## Case Study: Bolivar County, MS

Bolivar County exemplifies the Delta's compound crisis:

- **RI: 25.3/100** — bottom 1% nationally
- **HPSA Score: 25/26** — near-maximum healthcare shortage
- **Diabetes: 21.4%** vs. 12.5% national average
- **Hypertension: 52.0%** vs. 34.9% national average
- **Heat wave + inland flood risk: Relatively High** (FEMA)
- **31 FQHCs** serve all 9 Delta counties — but capacity is severely strained
- **Compound risk score: 99.98th national percentile**

---

## The Resilient Business Anchor Framework

Our analysis introduces a novel concept: **healthcare small businesses as resilience anchors**. In low-RI communities, pharmacies, FQHCs, and physician offices are often the first point of contact for climate-exacerbated health emergencies — and the most accessible economic assets for investment.

In the Mississippi Delta:
- Bolivar County has **45 pharmacies** and **378 physician offices** — serving 30,953 people across heat-exposed, flood-prone, food-desert tracts
- Every FQHC site in the Delta operates under capacity constraints tied to economic fragility, not lack of patient demand
- **Targeted small business lending** to expand pharmacy and FQHC capacity would raise RI scores by 3–8 points per county while directly addressing the healthcare-access component of triple vulnerability

---

## Investment Prioritization Framework

Based on the Resilience Index analysis, we identify three tiers of intervention:

**Tier 1 — Critical (RI < 25): Issaquena, Humphreys, Quitman, Sunflower, Sharkey**
Minimum viable intervention: mobile health units + FQHC expansion grants + HPSA workforce incentives

**Tier 2 — High Priority (RI 25–30): Leflore, Bolivar, Coahoma**
Intervention: small business healthcare loans + pharmacy infrastructure + food access investment

**Tier 3 — Stabilization (RI 30–35): Washington**
Intervention: economic inclusion programs + climate resilience planning + digital health access

---

## Tools Delivered

1. **Interactive Streamlit App** — 5-page dashboard with:
   - National choropleth: county-level RI across all 3,144 US counties
   - Delta Deep Dive: tract-level Folium map with FQHC site layer
   - Resilience Index explorer: radar charts, trend analysis, scenario planner
   - Community Profiles: automated report cards for all 9 Delta counties
   - Full methodology documentation

2. **Reproducible Pipeline** — 10+ data sources → Parquet → analysis → visualizations; runs end-to-end in ~15 minutes

3. **9 Community Profiles** — Data-driven investment briefs, one per Delta county

---

## Data Sources

Mastercard IGS 2017–2025 | FEMA NRI 2023 | CDC PLACES 2025 | CDC/ATSDR SVI 2022 | CDC EJI 2022 | HRSA HPSA/MUA 2024 | HRSA FQHC Sites 2024 | US Census CBP 2023 | USDA Food Atlas 2019 | Census TIGER/Line 2022

---

*Prepared by: Saurav Bhattarai (Newton)*
*Jackson State University — AUC Consortium*
*Mastercard IGS × AUC 2026 Data Science Challenge*
*Grand Finale: April 30, 2026*
