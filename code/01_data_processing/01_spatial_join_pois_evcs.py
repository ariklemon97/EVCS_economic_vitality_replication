"""
Step 1 of Data Processing Pipeline: Spatial Joins and POI Extraction.
This script cleans the NREL EVCS data, filters the Global Places (POIs) to the target categories 
(Accommodation/Food, Retail, and Arts/Entertainment) in CA, and identifies Treated vs Control POIs 
by calculating the exact geospatial distance securely up to 500m via KDTree.
"""

import os
import glob
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import logging
import numpy as np
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from code.analysis_config import (
    BROAD_WINDOW,
    COMMERCIAL_ADJACENT_RADIUS_M,
    LOCAL_BUSINESS_CONTEXT_RADIUS_M,
    MIN_NEARBY_LOCAL_BUSINESSES,
    MIN_COMMERCIAL_POIS_NEAR_EVCS,
    RAW_DIR,
    PROCESSED_DIR,
    TARGET_NAICS_PREFIXES,
)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set paths
OUTPUT_DIR = str(PROCESSED_DIR)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# CRS Definitions
# EPSG 4326 is standard lat/lon. EPSG 3310 is NAD83 / California Albers (meters) for accurate spatial distance calc.
GEO_CRS = "EPSG:4326"
PROJ_CRS = "EPSG:3310"

def process_nrel_evcs():
    """Clean and process the NREL EVCS dataset."""
    logging.info("Processing NREL EVCS Data...")
    
    # Use glob to match the date-stamped afdc json file
    nrel_paths = glob.glob(os.path.join(RAW_DIR, "nrel_afdc", "afdc_california_ev_stations*.json"))
    nrel_path = nrel_paths[0] if nrel_paths else None
    
    if not nrel_path or not os.path.exists(nrel_path):
        logging.warning(f"NREL data not found at {nrel_path}. Please ensure downloads completed.")
        return None
        
    try:
        import json
        with open(nrel_path, 'r') as f:
            nrel_json = json.load(f)
            
        # Handle the structure of NREL API json
        # It's usually in 'fuel_stations' list
        data = nrel_json.get('fuel_stations', nrel_json.get('alt_fuel_stations', []))
        df_evcs = pd.DataFrame(data)
    except Exception as e:
        logging.error(f"Failed to read NREL JSON: {e}")
        return None

    # Filter to public EVCS in CA only
    df_evcs = df_evcs[(df_evcs['state'] == 'CA') & (df_evcs['access_code'] == 'public')]
    
    # Ensure open_date is temporal
    df_evcs['open_date'] = pd.to_datetime(df_evcs['open_date'], errors='coerce')
    
    # Create study period flags
    # Study Period 1: Opened in 2019
    # Study Period 2: Opened between Feb 2021 and June 2023
    df_evcs['period_1'] = df_evcs['open_date'].dt.year == 2019
    
    start_p2 = pd.to_datetime(f"{BROAD_WINDOW.p2_treatment_start}01", format="%Y%m%d")
    broad_end = str(BROAD_WINDOW.period_2_end)
    end_p2 = pd.Period(f"{broad_end[:4]}-{broad_end[4:]}", freq="M").end_time.normalize()
    df_evcs['period_2'] = (df_evcs['open_date'] >= start_p2) & (df_evcs['open_date'] <= end_p2)
    
    # Keep only those in either study period
    df_evcs = df_evcs[df_evcs['period_1'] | df_evcs['period_2']].copy()
    
    # Ensure columns exist, fill na with 0
    for col in ['ev_level2_evse_num', 'ev_dc_fast_num']:
        if col not in df_evcs.columns:
            df_evcs[col] = 0
        df_evcs[col] = df_evcs[col].fillna(0)
    
    df_evcs['total_ports'] = df_evcs['ev_level2_evse_num'] + df_evcs['ev_dc_fast_num']
    
    # Drop rows with no valid coordinates
    df_evcs = df_evcs.dropna(subset=['latitude', 'longitude'])
    
    # Create GeoDataFrame
    gdf_evcs = gpd.GeoDataFrame(
        df_evcs, 
        geometry=gpd.points_from_xy(df_evcs.longitude, df_evcs.latitude),
        crs=GEO_CRS
    )
    
    # Project to California Albers (meters)
    gdf_evcs = gdf_evcs.to_crs(PROJ_CRS)
    logging.info(f"Retained {len(gdf_evcs)} Public EVCS stations for analysis.")
    
    # Save the cleaned EVCS array
    output_path = os.path.join(OUTPUT_DIR, "clean_nrel_evcs.parquet")
    gdf_evcs.to_parquet(output_path)
    return gdf_evcs

def get_target_naics_codes():
    """Return NAICS prefixes for target POI Categories.
       - Accommodation and food services (NAICS 72)
       - Retail trade (NAICS 44, 45)
       - Arts, entertainment, and recreation (NAICS 71)
    """
    return TARGET_NAICS_PREFIXES

def process_poi_spatial(gdf_evcs):
    """
    Iterate over downloaded global_places chunks and extract California POIs
    that match target NAICS categories. Perform geospatial distance joins to EVCS.
    """
    logging.info("Scanning Global Places for CA POIs and assigning treatment...")
    
    places_dir = os.path.join(RAW_DIR, "dewey", "global_places")
    # Using glob to find all parquet chunks
    parquet_files = glob.glob(os.path.join(places_dir, "**", "*.parquet"), recursive=True)
    
    if not parquet_files:
        logging.warning("No Global Places parquet files found. Aborting spatial join.")
        return
        
    target_naics = get_target_naics_codes()
    
    # ── Pre-process EVCS for spatial join efficiency ──────────────────────────
    # Buffer once outside the loop. Reset index to ensure it's a unique integer index.
    gdf_evcs = gdf_evcs.reset_index(drop=True)
    gdf_evcs['geometry_buffered'] = gdf_evcs.geometry.buffer(COMMERCIAL_ADJACENT_RADIUS_M)
    gdf_evcs_buffer = gdf_evcs.set_geometry('geometry_buffered').copy()
    
    # Lists to store treated & control pools plus charger-level matches.
    # The pair table is needed downstream so port exposure can accumulate by
    # each charger's own opening month instead of turning all nearby ports on
    # at the earliest nearby charger opening.
    processed_dfs = []
    match_dfs = []
    
    for i, file_path in enumerate(parquet_files):
        try:
            # Note: We need specific columns to conserve RAM. Add as needed.
            # Dewey parquet schemas are typically entirely capitalized.
            cols = ['PLACEKEY', 'LOCATION_NAME', 'NAICS_CODE', 'LATITUDE', 'LONGITUDE', 'REGION']
            
            # Read snippet of parquet checking for headers
            df = pd.read_parquet(file_path, columns=cols)
            df.columns = [c.lower() for c in df.columns]
            
            # Filter exactly step-by-step
            # 1. State == CA
            df = df[df['region'] == 'CA']
            
            if df.empty:
                continue
                
            # 2. NAICS Code starting with 72, 44, 45, or 71
            df = df.dropna(subset=['naics_code'])
            df['naics_code_str'] = df['naics_code'].astype(str)
            df = df[df['naics_code_str'].str.startswith(target_naics)]
            
            # 3. Validity of Coordinates
            df = df.dropna(subset=['latitude', 'longitude'])
            
            # Transform to GeoPandas
            gdf_poi = gpd.GeoDataFrame(
                df, 
                geometry=gpd.points_from_xy(df.longitude, df.latitude),
                crs=GEO_CRS
            ).to_crs(PROJ_CRS)
            
            # --- Spatial Join ---
            # To emulate distance bins, we compute distance to nearest EVCS
            # Since an EVCS can be multiple, we want all EVCS within 500m. 
            # Dask/Sjoin nearest is great.
            
            # --- Correct Spatial Join: Density across Bins ---
            # Instead of finding the nearest, we find ALL EVCS within 500m for every POI
            joined = gpd.sjoin(gdf_poi, gdf_evcs_buffer, how='inner', predicate='within')
            
            if not joined.empty:
                # Calculate exact distance for all pairs
                # Using a list comprehension over native shapely geometries is the 
                # most bulletproof way to avoid all GeoPandas alignment and slicing bugs
                joined['dist_m'] = [
                    geom.distance(gdf_evcs.geometry.iloc[idx]) 
                    for geom, idx in zip(joined.geometry, joined['index_right'])
                ]
                
                # Assign each pair to a bin
                bins = [0, 100, 200, 300, 400, 500]
                labels = ['X0_100m', 'X100_200m', 'X200_300m', 'X300_400m', 'X400_500m']
                # astype(str) avoids the Pandas categorical unstack slice bug
                joined['bin'] = pd.cut(joined['dist_m'], bins=bins, labels=labels, include_lowest=True).astype(str)
                
                # Aggregate ports by POI and Bin
                joined['total_ports'] = pd.to_numeric(joined['total_ports'], errors='coerce').fillna(0)
                joined['ev_level2_evse_num'] = pd.to_numeric(joined['ev_level2_evse_num'], errors='coerce').fillna(0)
                joined['ev_dc_fast_num'] = pd.to_numeric(joined['ev_dc_fast_num'], errors='coerce').fillna(0)
                joined['open_date'] = pd.to_datetime(joined['open_date'], errors='coerce')

                pair_cols = [
                    'placekey',
                    'index_right',
                    'dist_m',
                    'bin',
                    'open_date',
                    'total_ports',
                    'ev_level2_evse_num',
                    'ev_dc_fast_num',
                ]
                if 'id' in joined.columns:
                    pair_cols.append('id')
                pair_matches = pd.DataFrame(joined[pair_cols]).rename(
                    columns={'index_right': 'evcs_index', 'id': 'evcs_id'}
                )
                match_dfs.append(pair_matches)
                
                # Pivot manually to avoid categorical KeyError / slice error
                pivot_ports = joined.groupby(['placekey', 'bin'])['total_ports'].sum().unstack(fill_value=0)
                
                # Ensure all 5 bins exist in the pivot
                for lab in labels:
                    if lab not in pivot_ports.columns:
                        pivot_ports[lab] = 0.0
                
                # Get the earliest open date among all chargers within 500m for this POI
                min_dates = joined.groupby('placekey')['open_date'].min()
                
                # Get the total level2 and dc_fast counts within 500m for this POI
                port_counts = joined.groupby('placekey')[['total_ports', 'ev_level2_evse_num', 'ev_dc_fast_num']].sum()
                
                # Merge back to the core POI chunk (gdf_poi)
                treated_ids = joined['placekey'].unique()
                treated_mask = gdf_poi['placekey'].isin(treated_ids)
                
                gdf_poi.loc[treated_mask, 'is_treated'] = 1
                gdf_poi.loc[~treated_mask, 'is_treated'] = 0
                gdf_poi.loc[treated_mask, 'commercial_adjacent_evcs'] = 1
                gdf_poi.loc[~treated_mask, 'commercial_adjacent_evcs'] = 0
                
                # Move the binned port counts and min date over
                gdf_poi = gdf_poi.merge(pivot_ports, on='placekey', how='left')
                gdf_poi = gdf_poi.merge(min_dates.rename('open_date'), on='placekey', how='left')
                gdf_poi = gdf_poi.merge(port_counts, on='placekey', how='left')
                
                # Fill NAs for controls
                for lab in labels:
                    gdf_poi[lab] = gdf_poi[lab].fillna(0)
                gdf_poi['total_ports'] = gdf_poi['total_ports'].fillna(0)
                gdf_poi['ev_level2_evse_num'] = gdf_poi['ev_level2_evse_num'].fillna(0)
                gdf_poi['ev_dc_fast_num'] = gdf_poi['ev_dc_fast_num'].fillna(0)
            else:
                gdf_poi['is_treated'] = 0
                gdf_poi['commercial_adjacent_evcs'] = 0
                for lab in labels:
                    gdf_poi[lab] = 0.0
                gdf_poi['open_date'] = pd.NaT
                gdf_poi['total_ports'] = 0.0
                gdf_poi['ev_level2_evse_num'] = 0.0
                gdf_poi['ev_dc_fast_num'] = 0.0

            # Rename back coordinates that might have been lost
            if 'latitude_left' in gdf_poi.columns:
                 gdf_poi = gdf_poi.rename(columns={'latitude_left': 'latitude', 'longitude_left': 'longitude'})

            # Drop geometry and append
            combined = gdf_poi.drop(columns=['geometry'], errors='ignore')
            processed_dfs.append(combined)
            
            # A single POI might match multiple EVCS. If memory gets tight, group by placekey here
            # But the paper models each EVCS installation. We preserve placekey-to-EVCS records.
            if (i+1) % 10 == 0:
                logging.info(f"Processed {i+1} parquet files from Global Places...")
                
        except Exception as e:
            logging.error(f"Error processing {file_path}: {e}")

    if processed_dfs:
        final_df = pd.concat(processed_dfs, ignore_index=True)
        final_df = add_local_business_context(final_df)
        out_path = os.path.join(OUTPUT_DIR, "poi_treatment_assignments.parquet")
        final_df.to_parquet(out_path)
        logging.info(f"Saved {len(final_df)} POI-EVCS assignments to {out_path}.")

        if match_dfs:
            matches = pd.concat(match_dfs, ignore_index=True)
            matches = matches.drop_duplicates(
                subset=['placekey', 'evcs_index', 'dist_m', 'bin', 'open_date']
            )
            commercial_counts = (
                matches.groupby('evcs_index')['placekey']
                .nunique()
                .rename('commercial_poi_count_500m')
            )
            matches = matches.merge(commercial_counts, on='evcs_index', how='left')
            matches['commercial_poi_count_500m'] = matches['commercial_poi_count_500m'].fillna(0).astype(int)
            matches['commercial_adjacent_evcs'] = (
                matches['commercial_poi_count_500m'] >= MIN_COMMERCIAL_POIS_NEAR_EVCS
            ).astype(int)
            matches['open_yyyymm'] = (
                matches['open_date'].dt.year * 100 + matches['open_date'].dt.month
            ).fillna(0).astype(int)
            match_path = os.path.join(OUTPUT_DIR, "poi_evcs_matches.parquet")
            matches.to_parquet(match_path)
            logging.info(f"Saved {len(matches)} POI-EVCS charger matches to {match_path}.")
    else:
        logging.warning("No POIs processed.")


def add_local_business_context(df: pd.DataFrame) -> pd.DataFrame:
    """
    Mark target POIs that are embedded in a local business context.

    Controls must be comparable local businesses, not isolated target-category
    POIs. We therefore require at least MIN_NEARBY_LOCAL_BUSINESSES other
    target POIs within LOCAL_BUSINESS_CONTEXT_RADIUS_M.
    """
    required = {"placekey", "latitude", "longitude"}
    if missing := required - set(df.columns):
        logging.warning("Cannot compute local business context; missing columns: %s", sorted(missing))
        out = df.copy()
        out["local_business_count_500m"] = 0
        out["local_business_context"] = 0
        return out

    from scipy.spatial import cKDTree

    poi = (
        df[["placekey", "latitude", "longitude"]]
        .dropna(subset=["latitude", "longitude"])
        .drop_duplicates("placekey")
        .copy()
    )
    if poi.empty:
        out = df.copy()
        out["local_business_count_500m"] = 0
        out["local_business_context"] = 0
        return out

    gdf = gpd.GeoDataFrame(
        poi,
        geometry=gpd.points_from_xy(poi.longitude, poi.latitude),
        crs=GEO_CRS,
    ).to_crs(PROJ_CRS)
    coords = np.column_stack([gdf.geometry.x.to_numpy(), gdf.geometry.y.to_numpy()])
    tree = cKDTree(coords)
    neighbors = tree.query_ball_point(coords, r=LOCAL_BUSINESS_CONTEXT_RADIUS_M)
    counts = np.array([max(len(idx) - 1, 0) for idx in neighbors], dtype=int)

    context = pd.DataFrame(
        {
            "placekey": gdf["placekey"].to_numpy(),
            "local_business_count_500m": counts,
            "local_business_context": (counts >= MIN_NEARBY_LOCAL_BUSINESSES).astype(int),
        }
    )
    out = df.merge(context, on="placekey", how="left")
    out["local_business_count_500m"] = out["local_business_count_500m"].fillna(0).astype(int)
    out["local_business_context"] = out["local_business_context"].fillna(0).astype(int)
    logging.info(
        "Local business context: %s/%s POIs have at least %s nearby target businesses within %sm.",
        out.loc[out["local_business_context"].eq(1), "placekey"].nunique(),
        out["placekey"].nunique(),
        MIN_NEARBY_LOCAL_BUSINESSES,
        LOCAL_BUSINESS_CONTEXT_RADIUS_M,
    )
    return out

if __name__ == "__main__":
    logging.info("Starting POI spatial join pipeline...")
    gdf_evcs = process_nrel_evcs()
    if gdf_evcs is not None:
        process_poi_spatial(gdf_evcs)
    logging.info("Pipeline Step 1 Completed.")
