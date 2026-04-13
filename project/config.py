from pathlib import Path

# ── Root Paths ────────────────────────────────────────────────────────────────
DATA_ROOT = Path("/data/hpc/disk1/5 Data Challenge/Data")
PROJECT_ROOT = Path(__file__).parent
PROCESSED = PROJECT_ROOT / "data_processed"
PROCESSED_RAW = PROCESSED / "raw"

# ── Source File Paths ─────────────────────────────────────────────────────────
IGS_PATH          = DATA_ROOT / "IGS Mastercard" / "Inclusive Growth Score.xlsx"
FEMA_NRI_PATH     = DATA_ROOT / "Risk Index" / "Risk_Index_FEMA.dbf"
CDC_PLACES_PATH   = DATA_ROOT / "CDC Data" / "Census Track Data.csv"
SVI_PATH          = DATA_ROOT / "ASTDR SVI Data" / "SVI_2022_US.csv"
EJI_PATH          = DATA_ROOT / "Environment-Hazard Data" / "United States.csv"
HPSA_PC_PATH      = DATA_ROOT / "HRSA" / "Shortage Area" / "BCD_HPSA_FCT_DET_PC.csv"
HPSA_MH_PATH      = DATA_ROOT / "HRSA" / "Shortage Area" / "BCD_HPSA_FCT_DET_MH.csv"
MUA_PATH          = DATA_ROOT / "HRSA" / "Shortage Area" / "MUA_DET.csv"
FQHC_PATH         = DATA_ROOT / "HRSA" / "Health_Center_Service_Delivery_and_LookAlike_Sites.csv"
CBP_COUNTY_ZIP    = DATA_ROOT / "Small Business" / "cbp23co.zip"
ZBP_DETAIL_ZIP    = DATA_ROOT / "Small Business" / "zbp23detail.zip"
ZBP_TOTALS_ZIP    = DATA_ROOT / "Small Business" / "zbp23totals.zip"
SBA_RANKINGS_PATH = DATA_ROOT / "Small Business" / "state_statistics_rankings_2025.xlsx"
FOOD_ATLAS_ZIP    = DATA_ROOT / "Food" / "2019_Food_Access_Research_Atlas_Data.zip"
TIGER_PATH        = DATA_ROOT / "TIGER" / "tl_2022_28_tract.zip"
AHRF_ZIP_PATH     = DATA_ROOT / "HRSA" / "AHRF_2024-2025_CSV.zip"

# ── Processed Output Paths (raw ingest) ──────────────────────────────────────
IGS_PARQUET         = PROCESSED_RAW / "igs_all_years.parquet"
IGS_LATEST_PARQUET  = PROCESSED_RAW / "igs_latest.parquet"
NRI_PARQUET         = PROCESSED_RAW / "fema_nri.parquet"
PLACES_PARQUET      = PROCESSED_RAW / "cdc_places.parquet"
SVI_PARQUET         = PROCESSED_RAW / "svi.parquet"
EJI_PARQUET         = PROCESSED_RAW / "eji.parquet"
HPSA_PC_PARQUET     = PROCESSED_RAW / "hpsa_pc_county.parquet"
HPSA_MH_PARQUET     = PROCESSED_RAW / "hpsa_mh_county.parquet"
MUA_PARQUET         = PROCESSED_RAW / "mua_county.parquet"
FQHC_PARQUET        = PROCESSED_RAW / "fqhc_sites.parquet"
CBP_PARQUET         = PROCESSED_RAW / "cbp_naics_county.parquet"
SBA_PARQUET         = PROCESSED_RAW / "sba_state_rankings.parquet"
ZBP_PARQUET         = PROCESSED_RAW / "zbp_delta_zips.parquet"
FOOD_PARQUET        = PROCESSED_RAW / "food_atlas.parquet"
AHRF_PARQUET        = PROCESSED_RAW / "ahrf_county.parquet"

# ── Processed Output Paths (build stage) ─────────────────────────────────────
MASTER_TRACT        = PROCESSED / "master_tract.parquet"
IGS_TRENDS_PARQUET  = PROCESSED / "igs_trends.parquet"
IGS_NATIONAL        = PROCESSED / "igs_national.parquet"
DELTA_PROFILE       = PROCESSED / "delta_profile.parquet"
DELTA_PARQUET       = PROCESSED / "delta_full.parquet"
DELTA_GEOJSON       = PROCESSED / "delta_tracts.geojson"
# US county boundaries for Plotly choropleth (offline demo); download via scripts/download_counties_geojson.py
COUNTIES_GEOJSON    = PROCESSED / "counties.geojson"

# ── Mississippi Delta County FIPS ─────────────────────────────────────────────
DELTA_COUNTY_FIPS = [
    '28011','28027','28053','28055','28083',
    '28119','28125','28133','28151'
]
DELTA_COUNTY_NAMES = {
    '28011': 'Bolivar',
    '28027': 'Coahoma',
    '28053': 'Humphreys',
    '28055': 'Issaquena',
    '28083': 'Leflore',
    '28119': 'Quitman',
    '28125': 'Sharkey',
    '28133': 'Sunflower',
    '28151': 'Washington',
}

# ── Analysis Thresholds ───────────────────────────────────────────────────────
IGS_VULN_THRESHOLD = 45
HPSA_SCORE_MAX = 26

# ── IGS Pillar Structure (for algebraic investment simulator) ─────────────────
IGS_PILLAR_SIZES = {
    'Small Business Loans Score':            5,
    'Minority/Women Owned Businesses Score': 5,
    'Commercial Diversity Score':            5,
    'New Businesses Score':                  5,
    'Spend Growth Score':                    5,
    'Internet Access Score':                 4,
    'Affordable Housing Score':              4,
    'Travel Time to Work Score':             4,
    'Net Occupancy Score':                   4,
    'Health Insurance Coverage Score':       5,
    'Labor Market Engagement Index Score':   5,
    'Female Above Poverty Score':            5,
    'Personal Income Score':                 5,
    'Spending per Capita Score':             5,
}

IGS_SUB_TO_PILLAR = {
    'Small Business Loans Score':            'Economy',
    'Minority/Women Owned Businesses Score': 'Economy',
    'Commercial Diversity Score':            'Economy',
    'New Businesses Score':                  'Economy',
    'Spend Growth Score':                    'Economy',
    'Internet Access Score':                 'Place',
    'Affordable Housing Score':              'Place',
    'Travel Time to Work Score':             'Place',
    'Net Occupancy Score':                   'Place',
    'Health Insurance Coverage Score':       'Community',
    'Labor Market Engagement Index Score':   'Community',
    'Female Above Poverty Score':            'Community',
    'Personal Income Score':                 'Community',
    'Spending per Capita Score':             'Community',
}

# Sub-indicators grouped by pillar (convenience lists for the app)
ECONOMY_SUBS = [k for k, v in IGS_SUB_TO_PILLAR.items() if v == 'Economy']
PLACE_SUBS   = [k for k, v in IGS_SUB_TO_PILLAR.items() if v == 'Place']
COMMUNITY_SUBS = [k for k, v in IGS_SUB_TO_PILLAR.items() if v == 'Community']

# ── CBP NAICS Filters ─────────────────────────────────────────────────────────
NAICS_PHARMACY          = '446110'
NAICS_HEALTHCARE_PREFIX = '621'
NAICS_GROCERY           = '445110'

# ── Small-Business Solution Mapping ──────────────────────────────────────────
# Maps IGS sub-indicators to concrete small-business interventions
SMALL_BIZ_SOLUTIONS = {
    'Internet Access Score': {
        'gap_label': 'Broadband & Digital Access',
        'businesses': [
            'Community broadband co-ops',
            'Digital literacy training centers',
            'Co-working / WiFi hub spaces',
            'Mobile hotspot lending services',
        ],
        'mastercard_link': 'Digital-first small businesses expand the customer base for Mastercard payment networks.',
    },
    'Commercial Diversity Score': {
        'gap_label': 'Business Ecosystem Diversity',
        'businesses': [
            'Small business incubators / accelerators',
            'Mixed-use retail and service shops',
            'Food trucks and mobile vendors',
            'Local e-commerce fulfillment hubs',
        ],
        'mastercard_link': 'Each new business category increases transaction diversity in the Mastercard network.',
    },
    'Small Business Loans Score': {
        'gap_label': 'Access to Capital',
        'businesses': [
            'CDFI micro-lending offices',
            'SBA loan navigation services',
            'Peer-to-peer lending circles',
            'Financial literacy workshops',
        ],
        'mastercard_link': 'Mastercard small business lending tools directly improve this sub-indicator.',
    },
    'Health Insurance Coverage Score': {
        'gap_label': 'Insurance & Health Coverage',
        'businesses': [
            'ACA / Medicaid enrollment navigator offices',
            'Community health worker cooperatives',
            'Telehealth kiosk businesses',
            'Insurance brokerage for small employers',
        ],
        'mastercard_link': 'Higher coverage reduces emergency costs, freeing spending for local businesses.',
    },
    'Labor Market Engagement Index Score': {
        'gap_label': 'Workforce Participation',
        'businesses': [
            'Workforce training academies',
            'Staffing agencies for healthcare / trades',
            'Childcare centers (enables parent employment)',
            'Transportation micro-transit services',
        ],
        'mastercard_link': 'More workers employed means more transactions and inclusive growth.',
    },
    'Minority/Women Owned Businesses Score': {
        'gap_label': 'Inclusive Entrepreneurship',
        'businesses': [
            'Women-owned retail cooperatives',
            'Minority business development centers',
            'Micro-enterprise grant administration offices',
            'Culturally-specific food and service businesses',
        ],
        'mastercard_link': 'Mastercard\'s commitment to 50M small businesses includes equity-focused lending.',
    },
    'New Businesses Score': {
        'gap_label': 'New Business Formation',
        'businesses': [
            'Business registration & compliance services',
            'Shared commercial kitchen spaces',
            'Pop-up retail / maker spaces',
            'Franchise coaching services',
        ],
        'mastercard_link': 'Every new business is a new node in the Mastercard payment ecosystem.',
    },
    'Female Above Poverty Score': {
        'gap_label': 'Women\'s Economic Security',
        'businesses': [
            'Affordable childcare centers',
            'Women\'s financial planning services',
            'Home health aide businesses (predominantly female workforce)',
            'Online resale and craft marketplaces',
        ],
        'mastercard_link': 'Gender equity in economic participation drives inclusive growth scores.',
    },
    'Personal Income Score': {
        'gap_label': 'Income Growth',
        'businesses': [
            'Trade skill certification centers',
            'Remote work placement agencies',
            'Agricultural value-add processing (Delta context)',
            'Healthcare anchor businesses (pharmacies, clinics)',
        ],
        'mastercard_link': 'Higher incomes drive higher transaction volumes and consumer spending.',
    },
    'Spending per Capita Score': {
        'gap_label': 'Consumer Spending',
        'businesses': [
            'Local retail and grocery stores',
            'Restaurants and food service',
            'Entertainment and recreation venues',
            'E-commerce local delivery services',
        ],
        'mastercard_link': 'Spending per capita is measured directly through Mastercard transaction data.',
    },
    'Affordable Housing Score': {
        'gap_label': 'Housing Affordability',
        'businesses': [
            'Affordable housing development firms',
            'Home repair and renovation services',
            'Property management for affordable units',
            'Housing counseling agencies',
        ],
        'mastercard_link': 'Stable housing reduces financial stress and enables consistent economic participation.',
    },
    'Spend Growth Score': {
        'gap_label': 'Spending Momentum',
        'businesses': [
            'Buy-local marketing cooperatives',
            'Loyalty and rewards program administrators',
            'Small business POS modernization services',
            'Community event organizers (drives foot traffic)',
        ],
        'mastercard_link': 'Spend growth trend is derived from Mastercard anonymized transaction data.',
    },
    'Net Occupancy Score': {
        'gap_label': 'Occupied Housing & Commercial Space',
        'businesses': [
            'Property rehabilitation / vacant-lot conversion firms',
            'Affordable homeownership programs and land trusts',
            'Commercial space activation pop-ups',
            'Community development financial institutions (CDFIs)',
        ],
        'mastercard_link': 'Higher occupancy rates increase the density of potential Mastercard cardholders and merchants.',
    },
    'Travel Time to Work Score': {
        'gap_label': 'Commute Access & Transportation',
        'businesses': [
            'Micro-transit and ride-share cooperatives',
            'Employer shuttle services',
            'Bicycle and e-scooter rental shops',
            'Remote work co-working hubs (reduces commute need)',
        ],
        'mastercard_link': 'Shorter commutes increase time available for consumer spending and local economic activity.',
    },
}
