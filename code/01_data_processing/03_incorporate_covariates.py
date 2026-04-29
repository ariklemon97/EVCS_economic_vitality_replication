"""
Step 3 of Data Processing Pipeline: Incorporate PSM Covariates.
Produces psm_covariate_matrix.parquet with ALL covariates from Zheng et al. (2024):

Built Environment (EPA Smart Location Database, block-group level):
  - D1B  : Population density (gross, persons/acre)
  - D2A_EPHHM: Employment/household entropy index (proxy for building density)
  - D3AAO : Auto-oriented road miles per square mile
  - D3APO : Auto-oriented intersections per square mile
  - NatWalkInd: National Walkability Index

Socio-demographics (ACS 2019 / 2021, census tract level):
  - Median_Household_Income
  - Pct_Employed
  - Pct_Male
  - Pct_Minority

EV Market (CEC, county level):
  - EV_sales_per_1000

Building density (OpenStreetMap, approximated from EPA D1C: jobs/acre or D1B):
  - We use EPA D1C (gross employment density) as the building density proxy
    since OSM building footprints require heavy geospatial processing.
    OSM raw data is downloaded for completeness; exact footprint density
    can be computed in a future pass.
"""

import os
import glob
import logging
import zipfile
import urllib.request
import pandas as pd
import geopandas as gpd
import numpy as np
from pathlib import Path
import sys

PROJECT_ROOT_PATH = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_PATH))

from code.analysis_config import PROCESSED_DIR

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROCESSED_DIR = str(PROCESSED_DIR)
CENSUS_DIR    = os.path.join(PROJECT_ROOT, "data", "raw", "census_acs")
CEC_DIR       = os.path.join(PROJECT_ROOT, "data", "raw", "california_energy_commission")
EPA_DIR       = os.path.join(PROJECT_ROOT, "data", "raw", "epa_smart_location")
OSM_DIR       = os.path.join(PROJECT_ROOT, "data", "raw", "openstreetmap")

GEO_CRS  = "EPSG:4326"
PROJ_CRS = "EPSG:3310"   # NAD83 / California Albers (metres)


# =============================================================================
# 1. Census Tract Shapefiles
# =============================================================================

def get_ca_tracts_shapefile():
    """Downloads 2019 California Census Tract TIGER/Line shapefile."""
    shp_dir = os.path.join(CENSUS_DIR, "shapefiles")
    os.makedirs(shp_dir, exist_ok=True)
    zip_path = os.path.join(shp_dir, "tl_2019_06_tract.zip")

    if not os.path.exists(zip_path):
        logging.info("Downloading CA Census Tract shapefile (2019)...")
        url = "https://www2.census.gov/geo/tiger/TIGER2019/TRACT/tl_2019_06_tract.zip"
        urllib.request.urlretrieve(url, zip_path)
        logging.info("Download complete.")

    logging.info("Loading CA Census Tracts...")
    tracts = gpd.read_file(zip_path)
    # GEOID is the full FIPS (state+county+tract, 11 chars)
    tracts = tracts[['GEOID', 'geometry']].rename(columns={'GEOID': 'FIPS'})
    return tracts.to_crs(GEO_CRS)


# =============================================================================
# 2. ACS Socio-demographic Covariates  (tract level)
# =============================================================================

def load_census_data(period=1):
    """
    Loads ACS 5-year data for the relevant study period.
    period=1 → 2019 ACS; period=2 → 2021 ACS
    Returns tract-level DataFrame with covariates.
    """
    fname = "acs5_2019_california_tracts.csv" if period == 1 else "acs5_2021_california_tracts.csv"
    path  = os.path.join(CENSUS_DIR, fname)
    if not os.path.exists(path):
        logging.warning(f"ACS data not found: {path}")
        return None

    df = pd.read_csv(path, dtype={'FIPS': str})

    # ── Guard: replace Census API suppression sentinels ─────────────────────
    # The Census API codes suppressed/unavailable cells with large negative
    # integers (e.g. -666666666). Only negative values are invalid — positive
    # values, however high, are legitimate (e.g. Atherton, CA tracts can have
    # true medians above $250k). Do NOT apply an upper cap.
    if 'Median_Household_Income' in df.columns:
        df['Median_Household_Income'] = pd.to_numeric(
            df['Median_Household_Income'], errors='coerce'
        )
        bad_mask = df['Median_Household_Income'] < 0
        n_bad = bad_mask.sum()
        if n_bad:
            logging.warning(
                f"ACS ({fname}): {n_bad} tracts with negative Median_Household_Income "
                f"(Census suppression sentinel) set to NaN."
            )
        df.loc[bad_mask, 'Median_Household_Income'] = np.nan

    # ── Derived proportions ─────────────────────────────────────────────────
    pop = df['Total_Population'].replace(0, np.nan)
    df['Pct_Employed'] = df['Employed'] / pop
    df['Pct_Male']     = df['Male_Population'] / pop
    df['Pct_Minority'] = 1.0 - (df['White_NonHispanic'] / pop)

    keep = ['FIPS', 'Total_Population', 'Median_Household_Income',
            'Pct_Employed', 'Pct_Male', 'Pct_Minority']
    return df[[c for c in keep if c in df.columns]]


def load_disadvantaged_communities():
    """
    Load CEC/Justice40 disadvantaged or low-income community polygons.

    The replication requires an explicit geography-based flag. Income-quintile
    fallbacks are intentionally not used because they change the disadvantaged
    estimand.
    """
    candidates = [
        os.path.join(CEC_DIR, "ca_low_income_or_disadvantaged_communities.geojson"),
        os.path.join(CEC_DIR, "ca_justice40_disadvantaged_communities.geojson"),
    ]
    candidates.extend(glob.glob(os.path.join(CEC_DIR, "*disadvantaged*.geojson")))
    candidates.extend(glob.glob(os.path.join(CEC_DIR, "*Disadvantaged*.geojson")))
    candidates.extend(glob.glob(os.path.join(CEC_DIR, "*justice40*.geojson")))
    candidates.extend(glob.glob(os.path.join(CEC_DIR, "*Justice40*.geojson")))
    candidates.extend(glob.glob(os.path.join(CEC_DIR, "*disadvantaged*.shp")))
    candidates.extend(glob.glob(os.path.join(CEC_DIR, "*Disadvantaged*.shp")))

    seen = set()
    geographies = []
    for path in candidates:
        if path in seen or not os.path.exists(path):
            continue
        seen.add(path)
        if os.path.getsize(path) < 1_000:
            logging.warning("Skipping invalid small disadvantaged geography: %s", path)
            continue
        try:
            gdf = gpd.read_file(path)
        except Exception as exc:
            logging.warning("Could not read disadvantaged geography %s: %s", path, exc)
            continue
        if gdf.empty or "geometry" not in gdf.columns:
            logging.warning("Skipping empty disadvantaged geography: %s", path)
            continue
        if gdf.crs is None:
            gdf = gdf.set_crs(GEO_CRS)
        geographies.append(gdf[["geometry"]].to_crs(GEO_CRS))
        logging.info("Loaded disadvantaged geography: %s (%s features)", path, len(gdf))

    if not geographies:
        raise FileNotFoundError(
            "No readable CEC/Justice40 disadvantaged-community geography found in "
            f"{CEC_DIR}. Run code/00_data_download/07_download_disadvantaged_communities.py."
        )

    combined = pd.concat(geographies, ignore_index=True)
    combined = gpd.GeoDataFrame(combined, geometry="geometry", crs=GEO_CRS)
    combined = combined[combined.geometry.notna() & ~combined.geometry.is_empty].copy()
    if combined.empty:
        raise ValueError("CEC/Justice40 disadvantaged-community geography has no valid geometries.")
    return combined


def add_disadvantaged_flag(gdf_poi):
    """Spatially flag POIs inside the disadvantaged-community geography."""
    dac = load_disadvantaged_communities()
    poi = gdf_poi[["placekey", "geometry"]].copy()
    if poi.crs is None:
        poi = poi.set_crs(GEO_CRS)
    poi = poi.to_crs(GEO_CRS)

    joined = gpd.sjoin(poi, dac, how="left", predicate="intersects")
    disadvantaged_placekeys = set(joined.loc[joined["index_right"].notna(), "placekey"])
    out = gdf_poi.copy()
    out["is_disadvantaged"] = out["placekey"].isin(disadvantaged_placekeys).astype(int)
    logging.info(
        "CEC/Justice40 disadvantaged POIs: %s/%s",
        int(out["is_disadvantaged"].sum()),
        out["placekey"].nunique(),
    )
    return out


# =============================================================================
# 3. EPA Smart Location Database  (block-group level)
# =============================================================================
# Download the national geodatabase (~350 MB) once; filter to CA.
# Key columns used:
#   D1B       – Population density (persons/gross acre)
#   D1C       – Employment density (jobs/gross acre) — building density proxy
#   D3AAO     – Auto-oriented road miles per sq. mile
#   D3APO     – Auto-oriented intersections per sq. mile
#   NatWalkInd– National Walkability Index (1-20 scale)
#   GEOID10   – Census Block Group FIPS (12 chars)

EPA_CSV_URL = (
    "https://edg.epa.gov/data/public/OA/EPA_SmartLocationDatabase_V3_Jan_2021_Final.csv"
)
EPA_CSV_NAME = "EPA_SmartLocationDatabase_V3_Jan_2021_Final.csv"

def download_epa_smart_location():
    """Downloads the EPA Smart Location Database CSV if not already present."""
    os.makedirs(EPA_DIR, exist_ok=True)
    csv_path = os.path.join(EPA_DIR, EPA_CSV_NAME)

    # Skip if the CSV (or any CSV) is already on disk
    if os.path.exists(csv_path) and os.path.getsize(csv_path) > 1_000_000:
        logging.info(f"EPA Smart Location CSV already present ({csv_path}), skipping download.")
        return
    if glob.glob(os.path.join(EPA_DIR, "*.csv")):
        logging.info("EPA Smart Location CSV already present in directory, skipping download.")
        return

    logging.info("Downloading EPA Smart Location Database CSV (~50 MB)...")
    urllib.request.urlretrieve(EPA_CSV_URL, csv_path)
    logging.info(f"EPA Smart Location Database CSV ready: {csv_path}")


def load_epa_smart_location():
    """
    Loads EPA SLD into a block-group level DataFrame (CA only).
    Returns DataFrame keyed by census block-group GEOID (12 chars).
    Falls back gracefully if file is missing.
    """
    # Look for extracted shapefile / csv / gdb
    csv_candidates = glob.glob(os.path.join(EPA_DIR, "**", "*.csv"), recursive=True)
    shp_candidates = glob.glob(os.path.join(EPA_DIR, "**", "*.shp"), recursive=True)
    gdb_candidates = glob.glob(os.path.join(EPA_DIR, "**", "*.gdb"), recursive=True)

    epa_cols = ['GEOID10', 'D1B', 'D1C', 'D3AAO', 'D3APO', 'NatWalkInd', 'STATEFP']

    df_epa = None

    if csv_candidates:
        path = csv_candidates[0]
        logging.info(f"Loading EPA SLD from CSV: {path}")
        df_epa = pd.read_csv(path, dtype={'GEOID10': str}, low_memory=False)
    elif shp_candidates:
        path = shp_candidates[0]
        logging.info(f"Loading EPA SLD from SHP: {path}")
        gdf = gpd.read_file(path)
        df_epa = pd.DataFrame(gdf.drop(columns='geometry', errors='ignore'))
        if 'GEOID10' not in df_epa.columns and 'GEOID_Data' in df_epa.columns:
            df_epa = df_epa.rename(columns={'GEOID_Data': 'GEOID10'})
    elif gdb_candidates:
        path = gdb_candidates[0]
        logging.info(f"Loading EPA SLD from GDB: {path}")
        gdf = gpd.read_file(path, driver='OpenFileGDB')
        df_epa = pd.DataFrame(gdf.drop(columns='geometry', errors='ignore'))
    else:
        logging.warning("EPA Smart Location data not found. Built environment covariates will be missing.")
        return None

    # Filter to California (STATEFP == '06')
    if 'STATEFP' in df_epa.columns:
        df_epa['STATEFP'] = df_epa['STATEFP'].astype(str).str.zfill(2)
        df_epa = df_epa[df_epa['STATEFP'] == '06']
    elif 'GEOID10' in df_epa.columns:
        df_epa['GEOID10'] = df_epa['GEOID10'].astype(str).str.zfill(12)
        df_epa = df_epa[df_epa['GEOID10'].str.startswith('06')]

    available = [c for c in epa_cols if c in df_epa.columns]
    df_epa = df_epa[available].copy()

    # Ensure GEOID10 is 12-character string (block group)
    if 'GEOID10' in df_epa.columns:
        df_epa['GEOID10'] = df_epa['GEOID10'].astype(str).str.zfill(12)

    # Replace -99999 / negative sentinel values with NaN
    for col in ['D1B', 'D1C', 'D3AAO', 'D3APO', 'NatWalkInd']:
        if col in df_epa.columns:
            df_epa[col] = pd.to_numeric(df_epa[col], errors='coerce')
            df_epa.loc[df_epa[col] < 0, col] = np.nan

    logging.info(f"EPA SLD loaded: {len(df_epa)} CA block groups, cols: {df_epa.columns.tolist()}")
    return df_epa


# =============================================================================
# 4. CEC — EV Sales per 1,000 people  (county level)
# =============================================================================

def load_ev_sales_per_capita():
    """
    Returns a DataFrame with columns [county_fips, EV_sales_per_1000].
    Uses CEC ZEV sales excel + ACS county population.
    """
    sales_paths = glob.glob(os.path.join(CEC_DIR, "*ZEV_Sales*.xlsx"))
    if not sales_paths:
        logging.warning("CEC ZEV Sales file not found. EV_sales_per_1000 will be missing.")
        return None

    try:
        # The CEC file has multiple sheets; find the one with 'County'
        sheets = pd.read_excel(sales_paths[0], sheet_name=None, header=None)
        target = None
        for name, sheet in sheets.items():
            # Flatten all cell values and search for 'County'
            flat = sheet.astype(str).values.flatten()
            if any('county' in v.lower() for v in flat):
                target = sheet
                break

        if target is None:
            logging.warning("Could not identify county column in CEC ZEV file.")
            return None

        # Promote the first row that looks like a header
        for i, row in target.iterrows():
            if any('county' in str(v).lower() for v in row.values):
                target.columns = target.iloc[i]
                target = target.iloc[i+1:].reset_index(drop=True)
                break

        # Normalise column names
        target.columns = [str(c).strip() for c in target.columns]
        county_col = next((c for c in target.columns if 'county' in c.lower()), None)
        if county_col is None:
            logging.warning("No county column found after header promotion.")
            return None

        # Identify the correct column for vehicle counts
        count_col = next((c for c in target.columns if 'number of vehicles' in c.lower()), None)
        if count_col is None:
            # Fallback: find any column with 'number' or 'count'
            count_col = next((c for c in target.columns if any(p in c.lower() for p in ['number', 'count'])), None)
            
        if count_col:
            logging.info(f"Using {count_col} for vehicle counts.")
            target[count_col] = pd.to_numeric(target[count_col], errors='coerce').fillna(0)
            # Group by county to get a single row per county
            target = target.groupby(county_col)[count_col].sum().reset_index()
            target.columns = [county_col, 'total_zev']
        else:
            logging.warning("Could not identify count column in CEC ZEV file. Setting total_zev to 0.")
            target = target.groupby(county_col).head(1)[[county_col]].copy()
            target['total_zev'] = 0

        # Load CA county populations from ACS 2019
        acs_path = os.path.join(CENSUS_DIR, "acs5_2019_california_tracts.csv")
        if os.path.exists(acs_path):
            acs   = pd.read_csv(acs_path, dtype={'FIPS': str})
            # county FIPS = first 5 chars of tract FIPS
            acs['county_fips'] = acs['FIPS'].str[:5]
            county_pop = acs.groupby('county_fips')['Total_Population'].sum().reset_index()
            county_pop.columns = ['county_fips', 'county_population']
        else:
            county_pop = None

        county_sales = target[[county_col, 'total_zev']].copy()
        county_sales.columns = ['county_name', 'total_zev']
        county_sales['total_zev'] = pd.to_numeric(county_sales['total_zev'], errors='coerce')
        county_sales = county_sales.dropna(subset=['total_zev'])

        if county_pop is not None:
            # Join on county name — build a mapping from FIPS to name
            county_pop['county_name'] = county_pop['county_fips']  # placeholder
            # Better: use a known CA county FIPS lookup
            ca_fips = {
                'Alameda': '06001', 'Alpine': '06003', 'Amador': '06005', 'Butte': '06007',
                'Calaveras': '06009', 'Colusa': '06011', 'Contra Costa': '06013',
                'Del Norte': '06015', 'El Dorado': '06017', 'Fresno': '06019',
                'Glenn': '06021', 'Humboldt': '06023', 'Imperial': '06025',
                'Inyo': '06027', 'Kern': '06029', 'Kings': '06031', 'Lake': '06033',
                'Lassen': '06035', 'Los Angeles': '06037', 'Madera': '06039',
                'Marin': '06041', 'Mariposa': '06043', 'Mendocino': '06045',
                'Merced': '06047', 'Modoc': '06049', 'Mono': '06051',
                'Monterey': '06053', 'Napa': '06055', 'Nevada': '06057',
                'Orange': '06059', 'Placer': '06061', 'Plumas': '06063',
                'Riverside': '06065', 'Sacramento': '06067', 'San Benito': '06069',
                'San Bernardino': '06071', 'San Diego': '06073',
                'San Francisco': '06075', 'San Joaquin': '06077',
                'San Luis Obispo': '06079', 'San Mateo': '06081',
                'Santa Barbara': '06083', 'Santa Clara': '06085',
                'Santa Cruz': '06087', 'Shasta': '06089', 'Sierra': '06091',
                'Siskiyou': '06093', 'Solano': '06095', 'Sonoma': '06097',
                'Stanislaus': '06099', 'Sutter': '06101', 'Tehama': '06103',
                'Trinity': '06105', 'Tulare': '06107', 'Tuolumne': '06109',
                'Ventura': '06111', 'Yolo': '06113', 'Yuba': '06115',
            }
            county_sales['county_fips'] = county_sales['county_name'].str.strip().map(ca_fips)
            county_sales = county_sales.merge(county_pop, on='county_fips', how='left')
            county_sales['EV_sales_per_1000'] = (
                county_sales['total_zev'] / county_sales['county_population'] * 1000
            )
        else:
            county_sales['county_fips'] = county_sales['county_name'].str.strip().map(
                {v: v for v in county_sales['county_name']}  # identity if no pop data
            )
            county_sales['EV_sales_per_1000'] = county_sales['total_zev']

        result = county_sales[['county_fips', 'EV_sales_per_1000']].dropna(subset=['county_fips'])
        logging.info(f"CEC EV sales per 1,000 loaded for {len(result)} counties.")
        return result

    except Exception as e:
        logging.error(f"Failed to load CEC EV sales: {e}")
        return None


# =============================================================================
# 5. Main — assemble covariate matrix
# =============================================================================

def main():
    poi_path = os.path.join(PROCESSED_DIR, "poi_treatment_assignments.parquet")
    if not os.path.exists(poi_path):
        logging.error("Missing Step 1 output (poi_treatment_assignments.parquet). Run 01 first.")
        return

    df_poi = pd.read_parquet(poi_path)

    # Reconstruct GeoDataFrame
    if 'geometry' not in df_poi.columns and 'latitude' in df_poi.columns:
        gdf_poi = gpd.GeoDataFrame(
            df_poi,
            geometry=gpd.points_from_xy(df_poi.longitude, df_poi.latitude),
            crs=GEO_CRS
        )
    else:
        gdf_poi = gpd.GeoDataFrame(df_poi, geometry='geometry', crs=GEO_CRS)

    # ── A. Attach Census Tract FIPS ─────────────────────────────────────────
    tracts = get_ca_tracts_shapefile()
    if 'index_right' in gdf_poi.columns:
        gdf_poi = gdf_poi.drop(columns='index_right')

    logging.info("Spatial join: POIs → Census Tracts...")
    gdf_poi = gpd.sjoin(gdf_poi, tracts, how='left', predicate='within')
    gdf_poi = gdf_poi.sort_values(['placekey', 'FIPS']).drop_duplicates('placekey', keep='first')
    # 'FIPS' now attached from tracts
    gdf_poi = add_disadvantaged_flag(gdf_poi)

    # Derive county_fips
    gdf_poi['county_fips'] = gdf_poi['FIPS'].str[:5]

    # ── B. Attach Census Block-Group ID for EPA join ────────────────────────
    # Block-group GEOID10 = first 12 chars of block FIPS; tract = first 11.
    # We approximate BG GEOID as tract FIPS + "1" (first BG) — this is a rough
    # fallback. In the real pipeline the block-group shapefile should be used.
    # For the EPA join we truncate tract FIPS to 11 chars and match on tract
    # prefix (first 11 of GEOID10).
    gdf_poi['tract_fips'] = gdf_poi['FIPS'].str[:11]

    # ── C. ACS Socio-demographics ──────────────────────────────────────────
    # Period 1 POIs → 2019 ACS; Period 2 POIs → 2021 ACS
    # If period flag unavailable, use 2019 for all (conservative default).
    acs_2019 = load_census_data(period=1)
    acs_2021 = load_census_data(period=2)

    if acs_2019 is not None:
        acs_cols = [c for c in acs_2019.columns if c != 'FIPS']
        # Default: everyone gets 2019 ACS
        df_out = pd.DataFrame(gdf_poi.drop(columns='geometry', errors='ignore'))
        df_out = df_out.merge(acs_2019, on='FIPS', how='left')

        # Override with 2021 ACS for period-2 POIs if the EVCS open_date indicates
        if acs_2021 is not None and 'open_date' in df_out.columns:
            df_out['open_date'] = pd.to_datetime(df_out['open_date'], errors='coerce')
            p2_mask = df_out['open_date'].dt.year >= 2021
            df_p2_acs = df_out.loc[p2_mask, ['FIPS']].merge(acs_2021, on='FIPS', how='left')
            for col in acs_cols:
                if col in df_p2_acs.columns:
                    df_out.loc[p2_mask, col] = df_p2_acs[col].values
    else:
        df_out = pd.DataFrame(gdf_poi.drop(columns='geometry', errors='ignore'))

    # ── D. EPA Smart Location ──────────────────────────────────────────────
    download_epa_smart_location()
    df_epa = load_epa_smart_location()

    if df_epa is not None and 'GEOID10' in df_epa.columns:
        # Match on tract prefix (first 11 chars of 12-char block-group GEOID)
        df_epa['tract_fips'] = df_epa['GEOID10'].str[:11]
        # Aggregate to tract level (mean across block groups within tract)
        epa_agg_cols = [c for c in ['D1B', 'D1C', 'D3AAO', 'D3APO', 'NatWalkInd']
                        if c in df_epa.columns]
        if epa_agg_cols:
            df_epa_tract = df_epa.groupby('tract_fips')[epa_agg_cols].mean().reset_index()
            df_epa_tract = df_epa_tract.rename(columns={
                'D1B':       'pop_density',
                'D1C':       'building_density',
                'D3AAO':     'road_miles_auto',
                'D3APO':     'intersections_auto',
                'NatWalkInd':'walkability_index',
            })
            df_out = df_out.merge(df_epa_tract, on='tract_fips', how='left')
            logging.info("EPA Smart Location covariates merged.")
    else:
        logging.warning("EPA data unavailable — built environment covariates will be NaN.")
        for col in ['pop_density','building_density','road_miles_auto',
                    'intersections_auto','walkability_index']:
            df_out[col] = np.nan

    # ── E. CEC EV Sales per Capita ─────────────────────────────────────────
    df_ev = load_ev_sales_per_capita()
    if df_ev is not None:
        df_out = df_out.merge(df_ev, on='county_fips', how='left')
        logging.info("CEC EV sales per 1,000 merged.")
    else:
        df_out['EV_sales_per_1000'] = np.nan

    # ── F. Save ────────────────────────────────────────────────────────────
    df_out = df_out.sort_values(['placekey', 'FIPS']).drop_duplicates('placekey', keep='first')
    out_path = os.path.join(PROCESSED_DIR, "psm_covariate_matrix.parquet")
    df_out.to_parquet(out_path, index=False)

    n_complete = df_out.dropna(subset=[
        'Median_Household_Income', 'pop_density', 'walkability_index'
    ]).shape[0]
    logging.info(
        f"Saved psm_covariate_matrix.parquet — shape: {df_out.shape}, "
        f"rows with key covariates: {n_complete}"
    )


if __name__ == "__main__":
    main()
