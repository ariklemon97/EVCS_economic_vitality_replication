"""
04_spatial_competition.py

Tracks competitive response via SafeGraph Placekey interactions.
Goal: If this POI doesn't have an EVCS, but its competitor within a configured
radius does,
we generate a 'Treatment_Competitor' indicator to measure channel shifting/stealing.
"""

import argparse
import geopandas as gpd
import pandas as pd
import numpy as np
import logging
import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO)
OUTPUT_DIR = os.path.join(str(PROJECT_ROOT), "data", "processed")


def load_charger_level_matches() -> pd.DataFrame | None:
    """Load POI-to-EVCS matches so competitor ports activate by charger month."""
    match_path = os.path.join(OUTPUT_DIR, "poi_evcs_matches.parquet")
    if not os.path.exists(match_path):
        logging.warning("Missing poi_evcs_matches.parquet; using POI-level competitor timing fallback.")
        return None

    matches = pd.read_parquet(match_path)
    required = {
        "placekey",
        "open_yyyymm",
        "total_ports",
        "ev_level2_evse_num",
        "ev_dc_fast_num",
    }
    missing = required - set(matches.columns)
    if missing:
        logging.warning(
            "poi_evcs_matches.parquet is missing %s; using POI-level competitor timing fallback.",
            sorted(missing),
        )
        return None

    matches = matches.copy()
    for col in ["open_yyyymm", "total_ports", "ev_level2_evse_num", "ev_dc_fast_num"]:
        matches[col] = pd.to_numeric(matches[col], errors="coerce").fillna(0)
    matches["open_yyyymm"] = matches["open_yyyymm"].astype(int)
    if "commercial_adjacent_evcs" in matches.columns:
        matches["commercial_adjacent_evcs"] = pd.to_numeric(
            matches["commercial_adjacent_evcs"], errors="coerce"
        ).fillna(0).astype(int)
        matches = matches[matches["commercial_adjacent_evcs"] == 1].copy()
    else:
        matches["commercial_adjacent_evcs"] = 1

    matches = matches[matches["open_yyyymm"] > 0].copy()
    if matches.empty:
        logging.warning("No usable charger-level POI-EVCS matches; using POI-level fallback.")
        return None

    return matches


def spatial_competition(radius_m: int = 1000, output_suffix: str = ""):
    if radius_m <= 500:
        raise SystemExit(
            "Spatial competition radius must be greater than 500m so competitor exposure "
            "does not overlap the direct EVCS-proximity treatment radius."
        )

    logging.info("Reading assigned geographical POIs...")
    poi_path = os.path.join(OUTPUT_DIR, "poi_treatment_assignments.parquet")
    if not os.path.exists(poi_path):
        logging.error("Missing POI assignments!")
        return
        
    df = pd.read_parquet(poi_path)
    if 'latitude' not in df.columns:
        return
        
    if 'open_date' in df.columns:
        open_date = pd.to_datetime(df['open_date'], errors='coerce')
        df['treated_open_yyyymm'] = (open_date.dt.year * 100 + open_date.dt.month).fillna(0).astype(int)

    gdf = gpd.GeoDataFrame(
        df, 
        geometry=gpd.points_from_xy(df.longitude, df.latitude),
        crs="EPSG:4326"
    ).to_crs("EPSG:3310")
    
    logging.info("Constructing Competition Nodes by NAICS codes within %sm...", radius_m)
    
    # Pre-calculate 4-digit NAICS for both
    gdf['naics_4'] = gdf['naics_code_str'].astype(str).str[:4]
    if 'commercial_adjacent_evcs' not in gdf.columns:
        # Existing assignment files were already generated from target-business
        # POI-to-EVCS matches. Treat directly exposed POIs as commercial-adjacent
        # for backward compatibility until the spatial join is rebuilt.
        gdf['commercial_adjacent_evcs'] = gdf['is_treated']
    gdf['commercial_adjacent_evcs'] = pd.to_numeric(
        gdf['commercial_adjacent_evcs'], errors='coerce'
    ).fillna(0).astype(int)
    if 'local_business_context' not in gdf.columns:
        # Backward-compatible fallback until Step 1 is rebuilt.
        gdf['local_business_context'] = 1
    gdf['local_business_context'] = pd.to_numeric(
        gdf['local_business_context'], errors='coerce'
    ).fillna(0).astype(int)
    
    treated = gdf[
        (gdf['is_treated'] == 1)
        & (gdf['commercial_adjacent_evcs'] == 1)
        & (gdf['local_business_context'] == 1)
    ].copy()
    controls = gdf[(gdf['is_treated'] == 0) & (gdf['local_business_context'] == 1)].copy()
    
    # We'll map static competitive exposure metadata.
    controls['Treatment_Competitor'] = 0
    controls['competitor_open_yyyymm'] = 0
    controls['competitor_total_ports'] = 0.0
    controls['competitor_has_level2'] = 0
    controls['competitor_has_dc_fast'] = 0
    competitor_pairs = []
    charger_matches = load_charger_level_matches()
    
    from scipy.spatial import cKDTree
    
    # Iterate through unique NAICS-4 categories to bound the search
    unique_naics = treated['naics_4'].unique()
    total_found = 0
    
    for n4 in unique_naics:
        t_sub = treated[treated['naics_4'] == n4]
        c_sub = controls[controls['naics_4'] == n4]
        
        if t_sub.empty or c_sub.empty:
            continue
            
        # Build tree on treated locations
        tree = cKDTree(list(zip(t_sub.geometry.x, t_sub.geometry.y)))
        
        # Query controls in the same category
        # Find if any treated store is within the configured competition radius.
        control_points = list(zip(c_sub.geometry.x, c_sub.geometry.y))
        indices = tree.query_ball_point(control_points, r=radius_m)
        
        for control_idx, treated_matches in zip(c_sub.index, indices):
            if not treated_matches:
                continue
            matched = t_sub.iloc[treated_matches]
            control_geom = controls.loc[control_idx, 'geometry']
            distances = matched.geometry.distance(control_geom).values
            controls.loc[control_idx, 'Treatment_Competitor'] = 1
            controls.loc[control_idx, 'competitor_open_yyyymm'] = matched['treated_open_yyyymm'].replace(0, np.nan).min()
            controls.loc[control_idx, 'competitor_total_ports'] = matched['total_ports'].sum()
            controls.loc[control_idx, 'competitor_has_level2'] = int((matched['ev_level2_evse_num'] > 0).any())
            controls.loc[control_idx, 'competitor_has_dc_fast'] = int((matched['ev_dc_fast_num'] > 0).any())
            pair_frame = pd.DataFrame(
                {
                    'placekey': controls.loc[control_idx, 'placekey'],
                    'competitor_placekey': matched['placekey'].values,
                    'naics_4': n4,
                    'competitor_open_yyyymm': matched['treated_open_yyyymm'].values,
                    'competitor_total_ports': matched['total_ports'].values,
                    'competitor_level2_ports': matched['ev_level2_evse_num'].values,
                    'competitor_dc_fast_ports': matched['ev_dc_fast_num'].values,
                    'competitor_commercial_adjacent': matched['commercial_adjacent_evcs'].values,
                    'competitor_distance_m': distances,
                    'competition_radius_m': radius_m,
                }
            )
            competitor_pairs.append(pair_frame)
        
    total_found = controls['Treatment_Competitor'].sum()
    logging.info(f"Identified {total_found} Neighbor-of-Treated competitive locations.")
    
    # Merge result back to original df
    df_merged = pd.merge(
        df,
        controls[
            [
                'placekey',
                'Treatment_Competitor',
                'competitor_open_yyyymm',
                'competitor_total_ports',
                'competitor_has_level2',
                'competitor_has_dc_fast',
            ]
        ],
        on='placekey',
        how='left',
    )
    for col in ['Treatment_Competitor', 'competitor_has_level2', 'competitor_has_dc_fast']:
        df_merged[col] = df_merged[col].fillna(0)
    df_merged['competitor_total_ports'] = df_merged['competitor_total_ports'].fillna(0.0)
    df_merged['competitor_open_yyyymm'] = df_merged['competitor_open_yyyymm'].fillna(0)
    
    suffix = f"_{output_suffix}" if output_suffix else ""
    out_path = os.path.join(OUTPUT_DIR, f"poi_spatial_competition{suffix}.parquet")
    df_merged.to_parquet(out_path)
    logging.info(f"Saved competitive spatial map to {out_path}.")

    if competitor_pairs:
        pair_df = pd.concat(competitor_pairs, ignore_index=True)
        if charger_matches is not None:
            base_cols = [
                'placekey',
                'competitor_placekey',
                'naics_4',
                'competitor_distance_m',
                'competition_radius_m',
            ]
            pair_df = pair_df[base_cols].drop_duplicates()
            charger_cols = [
                'placekey',
                'evcs_index',
                'evcs_id',
                'open_yyyymm',
                'total_ports',
                'ev_level2_evse_num',
                'ev_dc_fast_num',
                'commercial_adjacent_evcs',
            ]
            charger_cols = [col for col in charger_cols if col in charger_matches.columns]
            pair_df = pair_df.merge(
                charger_matches[charger_cols].rename(
                    columns={
                        'placekey': 'competitor_placekey',
                        'open_yyyymm': 'competitor_open_yyyymm',
                        'total_ports': 'competitor_total_ports',
                        'ev_level2_evse_num': 'competitor_level2_ports',
                        'ev_dc_fast_num': 'competitor_dc_fast_ports',
                        'commercial_adjacent_evcs': 'competitor_commercial_adjacent',
                    }
                ),
                on='competitor_placekey',
                how='inner',
            )
        dedupe_cols = [
            col for col in ['placekey', 'competitor_placekey', 'evcs_index', 'evcs_id', 'competitor_open_yyyymm']
            if col in pair_df.columns
        ]
        pair_df = pair_df.drop_duplicates(subset=dedupe_cols)
        pair_path = os.path.join(OUTPUT_DIR, f"poi_competitor_matches{suffix}.parquet")
        pair_df.to_parquet(pair_path)
        logging.info(f"Saved {len(pair_df)} POI competitor matches to {pair_path}.")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--radius-m",
        type=int,
        default=1000,
        help="Same-sector competitor search radius in meters. Must be greater than 500m.",
    )
    parser.add_argument(
        "--output-suffix",
        default="",
        help="Optional suffix for robustness outputs, e.g. r1500 writes poi_competitor_matches_r1500.parquet.",
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    spatial_competition(radius_m=args.radius_m, output_suffix=args.output_suffix)
