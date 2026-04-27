"""
Final Compilation Script for Regression Panel.

Merges monthly_spend_panel + psm_covariate_matrix into df_pre_match.csv,
exactly following the Zheng et al. (2024) specification:

  - Temporal filter: 2019 (Period 1) + Feb 2021–Jun 2023 (Period 2)
    Jan 2020–Jan 2021 excluded (COVID lockdowns + data integration anomaly)
  - Treatment variable: port_treat = D_it × PC_it
      D_it   = 1 if an EVCS was active within 500m at time t
      PC_it  = total charging ports operational within 500m at time t
  - Charger-type port counts: PC_level2_it, PC_dc_it (for charger-type models)
  - Distance-bin dummies: X0_100m … X400_500m  × PC_idt (for distance models)
  - Bin-specific port counts for distance-varying treatment
  - sequential date index (uniform spacing for CS estimator)
  - Income-bracket columns: cus_<25k … cus_>150k  (from SafeGraph spend patterns)
  - Other outcomes: median_dwell, median_dist_home, avg_customer_income
"""

import os
import logging
import pandas as pd
import numpy as np
from pathlib import Path
import sys

PROJECT_ROOT_PATH = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_PATH))

from code.analysis_config import BROAD_WINDOW, MIN_COMMERCIAL_POIS_NEAR_EVCS

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

PROJECT_ROOT  = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")


# =============================================================================
# Helpers
# =============================================================================

def safe_log1p(series):
    return np.log1p(series.clip(lower=0).astype(float))


def build_period_to_seq(date_numeric_series):
    """
    Map YYYYMM integers to sequential 1-based integers.
    Returns (all_periods list, period_to_seq dict).
    """
    all_periods   = sorted(date_numeric_series.unique())
    period_to_seq = {p: i + 1 for i, p in enumerate(all_periods)}
    return all_periods, period_to_seq


def map_treat_period(yyyymm, period_to_seq, all_periods):
    """
    Map an EVCS open_date (YYYYMM) to the sequential index.
    Never-treated (0) stays 0. Dates outside the study window map to
    the nearest available period after them.
    """
    if yyyymm == 0:
        return 0
    if yyyymm in period_to_seq:
        return period_to_seq[yyyymm]
    candidates = [p for p in all_periods if p >= yyyymm]
    if candidates:
        return period_to_seq[candidates[0]]
    return 0   # after all periods → never-treated


def build_monthly_port_exposure(periods, period_to_seq):
    """
    Build POI-month active port exposure from charger-level POI matches.

    This preserves the original port-intensity treatment logic: only chargers
    whose own open month is <= the panel month contribute to PC_it.
    """
    match_path = os.path.join(PROCESSED_DIR, "poi_evcs_matches.parquet")
    if not os.path.exists(match_path):
        return None

    matches = pd.read_parquet(match_path)
    if matches.empty:
        return None

    required = {"placekey", "open_yyyymm", "total_ports", "ev_level2_evse_num", "ev_dc_fast_num", "bin"}
    missing = required - set(matches.columns)
    if missing:
        logging.warning("POI-EVCS match table is missing columns: %s", sorted(missing))
        return None

    matches = matches.copy()
    if "commercial_adjacent_evcs" in matches.columns:
        matches["commercial_adjacent_evcs"] = pd.to_numeric(
            matches["commercial_adjacent_evcs"], errors="coerce"
        ).fillna(0).astype(int)
        if "commercial_poi_count_500m" not in matches.columns:
            matches["commercial_poi_count_500m"] = matches["commercial_adjacent_evcs"]
        matches = matches[matches["commercial_adjacent_evcs"] == 1].copy()
    elif "commercial_poi_count_500m" in matches.columns:
        matches["commercial_poi_count_500m"] = pd.to_numeric(
            matches["commercial_poi_count_500m"], errors="coerce"
        ).fillna(0).astype(int)
        matches = matches[matches["commercial_poi_count_500m"] >= MIN_COMMERCIAL_POIS_NEAR_EVCS].copy()
    else:
        # Backward-compatible fallback for existing match files: this table is
        # produced only from target-business POI-to-EVCS pairs, so any matched
        # charger is commercial-adjacent by construction.
        matches["commercial_adjacent_evcs"] = 1
        matches["commercial_poi_count_500m"] = 1

    if matches.empty:
        return None

    for col in ["total_ports", "ev_level2_evse_num", "ev_dc_fast_num"]:
        matches[col] = pd.to_numeric(matches[col], errors="coerce").fillna(0.0)
    matches["open_yyyymm"] = pd.to_numeric(matches["open_yyyymm"], errors="coerce").fillna(0).astype(int)
    matches = matches[matches["open_yyyymm"] > 0].copy()
    if matches.empty:
        return None

    bin_vars = ['X0_100m', 'X100_200m', 'X200_300m', 'X300_400m', 'X400_500m']
    for b in bin_vars:
        matches[b] = np.where(matches["bin"] == b, matches["total_ports"], 0.0)

    static = (
        matches.groupby("placekey")
        .agg(
            open_yyyymm=("open_yyyymm", "min"),
            total_ports=("total_ports", "sum"),
            ev_level2_evse_num=("ev_level2_evse_num", "sum"),
            ev_dc_fast_num=("ev_dc_fast_num", "sum"),
            **{b: (b, "sum") for b in bin_vars},
            commercial_poi_count_500m=("commercial_poi_count_500m", "max"),
            commercial_adjacent_evcs=("commercial_adjacent_evcs", "max"),
        )
        .reset_index()
    )
    static["is_treated"] = 1
    static["first_treat_period"] = static["open_yyyymm"].apply(
        lambda x: map_treat_period(x, period_to_seq, periods)
    )

    active_frames = []
    exposure_cols = ["total_ports", "ev_level2_evse_num", "ev_dc_fast_num", *bin_vars]
    for period in periods:
        active = matches[matches["open_yyyymm"] <= period]
        if active.empty:
            continue
        month = active.groupby("placekey")[exposure_cols].sum().reset_index()
        month["date_numeric_orig"] = period
        active_frames.append(month)

    active_panel = pd.concat(active_frames, ignore_index=True) if active_frames else pd.DataFrame()
    if active_panel.empty:
        active_panel = pd.DataFrame(columns=["placekey", "date_numeric_orig", *exposure_cols])

    return static, active_panel


# =============================================================================
# Main
# =============================================================================

def main():
    covariate_path = os.path.join(PROCESSED_DIR, "psm_covariate_matrix.parquet")
    spend_path     = os.path.join(PROCESSED_DIR, "monthly_spend_panel.parquet")
    foot_traffic_path = os.path.join(PROCESSED_DIR, "monthly_foot_traffic_panel.parquet")

    if not os.path.exists(covariate_path) or not os.path.exists(spend_path):
        logging.error("Missing pipeline output. Run steps 1-3 first.")
        return

    logging.info("Loading component datasets...")
    covariate_df = pd.read_parquet(covariate_path)
    spend_df     = pd.read_parquet(spend_path)
    foot_traffic_df = pd.read_parquet(foot_traffic_path) if os.path.exists(foot_traffic_path) else None

    # ── Merge ────────────────────────────────────────────────────────────────
    merged = pd.merge(spend_df, covariate_df, on='placekey', how='inner')
    if foot_traffic_df is not None:
        merged = pd.merge(merged, foot_traffic_df, on=['placekey', 'year_month'], how='left')
    logging.info(f"After merge: {len(merged)} rows, {merged['placekey'].nunique()} POIs")

    # ── Date numerics ────────────────────────────────────────────────────────
    if 'year_month' in merged.columns:
        # year_month is a Period object (e.g. 2019-01)
        merged['year_month']    = merged['year_month'].astype(str)
        merged['date']          = merged['year_month']
        ymd                     = pd.PeriodIndex(merged['year_month'], freq='M')
        merged['date_numeric']  = ymd.year * 100 + ymd.month   # YYYYMM

    # ── Temporal Filter ──────────────────────────────────────────────────────
    # Keep: Jan–Dec 2019  (Period 1)
    #       Feb 2021–Jun 2023  (Period 2)
    # Exclude: all of 2020 + Jan 2021 (COVID lockdowns & data integration anomaly)
    pre_n = len(merged)
    keep_p1  = (
        (merged['date_numeric'] >= BROAD_WINDOW.period_1_start) &
        (merged['date_numeric'] <= BROAD_WINDOW.period_1_end)
    )
    keep_p2  = (
        (merged['date_numeric'] >= BROAD_WINDOW.period_2_start) &
        (merged['date_numeric'] <= BROAD_WINDOW.period_2_end)
    )
    merged   = merged[keep_p1 | keep_p2].copy()
    logging.info(
        f"Temporal filter: {pre_n} → {len(merged)} rows "
        f"(dropped {pre_n - len(merged)} rows outside study periods)"
    )

    # ── Sequential time index ────────────────────────────────────────────────
    all_periods, period_to_seq = build_period_to_seq(merged['date_numeric'])
    merged['date_numeric_orig'] = merged['date_numeric']
    merged['date_numeric']      = merged['date_numeric'].map(period_to_seq)
    # ── County FIPS ──────────────────────────────────────────────────────────
    if 'FIPS' in merged.columns:
        merged['county_fips'] = merged['FIPS'].astype(str).str[:5]
    elif 'county_fips' not in merged.columns:
        logging.warning("No FIPS detected. Fixed effects may fail.")

    # ── Load Spatial Treatment Assignments ───────────────────────────────────
    assigned_path = os.path.join(PROCESSED_DIR, "poi_treatment_assignments.parquet")
    if os.path.exists(assigned_path):
        assigned = pd.read_parquet(assigned_path)
        # Drop columns that might already be in merged to avoid duplication errors 
        # (naics_code might be in both but different schemas)
        overlap_cols = [c for c in assigned.columns if c in merged.columns and c != 'placekey']
        if overlap_cols:
            assigned = assigned.drop(columns=overlap_cols)
        
        # FIX: The assigned parquet has a 100% duplication rate due to double append in 01.
        # Defuse Cartesian merge memory explosion by strictly dropping duplicates.
        assigned = assigned.drop_duplicates(subset=['placekey'])
        
        merged = pd.merge(merged, assigned, on='placekey', how='left')
        merged['is_treated'] = merged['is_treated'].fillna(0).astype(int)
    else:
        logging.warning("Treatment assignments parquet not found. Model will have no variation.")
        merged['is_treated'] = 0

    dynamic_exposure = build_monthly_port_exposure(all_periods, period_to_seq)
    if dynamic_exposure is not None:
        static_exposure, active_exposure = dynamic_exposure

        # Drop static exposure columns from the legacy assignment merge before
        # replacing them with charger-level/month-correct values.
        exposure_static_cols = [
            'is_treated', 'open_date', 'open_yyyymm', 'first_treat_period',
            'total_ports', 'ev_level2_evse_num', 'ev_dc_fast_num',
            'X0_100m', 'X100_200m', 'X200_300m', 'X300_400m', 'X400_500m',
            'commercial_poi_count_500m', 'commercial_adjacent_evcs',
        ]
        merged = merged.drop(columns=[c for c in exposure_static_cols if c in merged.columns], errors='ignore')
        merged = merged.merge(static_exposure, on='placekey', how='left')
        merged['is_treated'] = merged['is_treated'].fillna(0).astype(int)
        merged['first_treat_period'] = merged['first_treat_period'].fillna(0).astype(int)
        merged['open_yyyymm'] = merged['open_yyyymm'].fillna(0).astype(int)
        merged['commercial_adjacent_evcs'] = merged['commercial_adjacent_evcs'].fillna(0).astype(int)
        merged['commercial_poi_count_500m'] = merged['commercial_poi_count_500m'].fillna(0).astype(int)

        # Month-specific active ports by charger opening date.
        active_cols = [
            'total_ports', 'ev_level2_evse_num', 'ev_dc_fast_num',
            'X0_100m', 'X100_200m', 'X200_300m', 'X300_400m', 'X400_500m',
        ]
        active_exposure = active_exposure.rename(
            columns={
                'total_ports': 'PC_it',
                'ev_level2_evse_num': 'PC_level2_it',
                'ev_dc_fast_num': 'PC_dc_it',
                'X0_100m': 'PC_X0_100m',
                'X100_200m': 'PC_X100_200m',
                'X200_300m': 'PC_X200_300m',
                'X300_400m': 'PC_X300_400m',
                'X400_500m': 'PC_X400_500m',
            }
        )
        merged = merged.merge(active_exposure, on=['placekey', 'date_numeric_orig'], how='left')
        for col in ['PC_it', 'PC_level2_it', 'PC_dc_it', 'PC_X0_100m', 'PC_X100_200m', 'PC_X200_300m', 'PC_X300_400m', 'PC_X400_500m']:
            merged[col] = merged[col].fillna(0.0)
        logging.info("Using charger-level monthly port exposure from poi_evcs_matches.parquet.")
    else:
        logging.warning("Using legacy POI-level exposure because poi_evcs_matches.parquet is missing.")

    # ── Dependent Variables (log(Y+1)) ───────────────────────────────────────
    merged['lcus']   = safe_log1p(merged.get('raw_visit_counts',  pd.Series(0, index=merged.index)))
    
    # Calculate total spending as average transaction size * visit counts
    raw_total_spend = merged.get('raw_total_spend', pd.Series(0, index=merged.index))
    merged['lspend'] = safe_log1p(raw_total_spend.astype(float))

    # ── Income-bracket customer counts ───────────────────────────────────────
    # SafeGraph Spending data carries these columns if present.
    # Names follow the convention used in the original Rmd: cus_X where X is the bracket.
    income_bracket_map = {
        'cus_.25k':      ['cus_.25k', 'customers_by_income_<25000', 'income_lt_25k'],
        'cus_25.45k':    ['cus_25.45k', 'customers_by_income_25000_45000', 'income_25_45k'],
        'cus_45.60k':    ['cus_45.60k', 'customers_by_income_45000_60000', 'income_45_60k'],
        'cus_60.75k':    ['cus_60.75k', 'customers_by_income_60000_75000', 'income_60_75k'],
        'cus_75.100k':   ['cus_75.100k', 'customers_by_income_75000_100000', 'income_75_100k'],
        'cus_100.150k':  ['cus_100.150k', 'customers_by_income_100000_150000', 'income_100_150k'],
        'cus_.150k':     ['cus_.150k', 'customers_by_income_>150000', 'income_gt_150k'],
    }
    for target_col, candidates in income_bracket_map.items():
        found = next((c for c in candidates if c in merged.columns), None)
        if found:
            merged[target_col] = merged[found].clip(lower=0)
        else:
            merged[target_col] = np.nan

    # ── Other outcome variables ───────────────────────────────────────────────
    if 'median_dwell'     not in merged.columns: merged['median_dwell']     = np.nan
    if 'median_dist_home' not in merged.columns: merged['median_dist_home'] = np.nan

    # Average customer income (based on income brackets, using bracket midpoints)
    bracket_midpoints = {
        'cus_.25k': 12_500, 'cus_25.45k': 35_000, 'cus_45.60k': 52_500,
        'cus_60.75k': 67_500, 'cus_75.100k': 87_500, 'cus_100.150k': 125_000,
        'cus_.150k': 175_000,
    }
    all_brackets = list(bracket_midpoints.keys())
    available_brackets = [b for b in all_brackets if b in merged.columns and not merged[b].isna().all()]
    if available_brackets:
        total_cus    = merged[available_brackets].clip(lower=0).sum(axis=1).replace(0, np.nan)
        weighted_inc = sum(
            merged[b].clip(lower=0) * bracket_midpoints[b]
            for b in available_brackets
        )
        merged['avg_customer_income'] = weighted_inc / total_cus
    else:
        merged['avg_customer_income'] = np.nan

    # ── first_treat_period (EVCS open_date → sequential index) ──────────────
    if 'first_treat_period' in merged.columns:
        merged['first_treat_period'] = pd.to_numeric(merged['first_treat_period'], errors='coerce').fillna(0).astype(int)
    elif 'open_date' in merged.columns:
        merged['open_date'] = pd.to_datetime(merged['open_date'], errors='coerce')
        merged['open_yyyymm'] = (
            merged['open_date'].dt.year * 100 + merged['open_date'].dt.month
        ).fillna(0).astype(int)
        # Controls → 0
        merged.loc[merged['is_treated'] == 0, 'open_yyyymm'] = 0
        merged['first_treat_period'] = merged['open_yyyymm'].apply(
            lambda x: map_treat_period(x, period_to_seq, all_periods)
        )
    else:
        merged['first_treat_period'] = 0

    logging.info(
        f"first_treat_period unique (sequential): "
        f"{sorted(merged['first_treat_period'].unique())[:25]}"
    )

    # ── Port counts (PC_it) — total active ports within 500m at time t ───────
    for col in ['total_ports', 'ev_level2_evse_num', 'ev_dc_fast_num', 'PC_it', 'PC_level2_it', 'PC_dc_it']:
        if col not in merged.columns:
            merged[col] = 0
        merged[col] = pd.to_numeric(merged[col], errors='coerce').fillna(0)

    bin_vars = ['X0_100m', 'X100_200m', 'X200_300m', 'X300_400m', 'X400_500m']
    for v in bin_vars:
        if v not in merged.columns:
            merged[v] = 0
        merged[v] = pd.to_numeric(merged[v], errors='coerce').fillna(0)

    active_mask = merged['PC_it'] > 0
    merged['D_it'] = active_mask.astype(int)

    # Treatment variable: port_treat = D_it * PC_it
    merged['port_treat'] = merged['PC_it']
    
    # Treatment by charger type
    merged['port_treat_level2'] = merged['PC_level2_it']
    merged['port_treat_dc']     = merged['PC_dc_it']

    # ── Distance-bin effects (Equation 2) ────────────────────────────────────
    # Impact varies by bin-specific port density
    for v in bin_vars:
        active_col = f"PC_{v}"
        if active_col in merged.columns:
            merged[f'port_treat_{v}'] = merged[active_col].fillna(0)
        else:
            merged[f'port_treat_{v}'] = active_mask.astype(int) * merged[v].fillna(0)

    # ── NAICS sector filter — keep only 44, 45, 71, 72 ──────────────────────
    if 'naics_code' in merged.columns:
        merged['naics_code'] = pd.to_numeric(merged['naics_code'], errors='coerce')
        merged['naics_sector'] = (merged['naics_code'] // 10000).astype('Int64')
        valid_sectors = {44, 45, 71, 72}
        pre_filter = len(merged)
        merged = merged[merged['naics_sector'].isin(valid_sectors)].copy()
        logging.info(
            f"NAICS filter (keep 44,45,71,72): {pre_filter} → {len(merged)} rows "
            f"(dropped {pre_filter - len(merged)})"
        )

    # ── Write df_pre_match.csv ───────────────────────────────────────────────
    output_path = os.path.join(PROCESSED_DIR, "df_pre_match.csv")
    merged.to_csv(output_path, index=False)
    logging.info(f"Saved df_pre_match.csv — shape: {merged.shape}")
    logging.info(f"  Treated POIs: {merged[merged['is_treated']==1]['placekey'].nunique()}")
    logging.info(f"  Control POIs: {merged[merged['is_treated']==0]['placekey'].nunique()}")
    logging.info(f"  Periods: {sorted(merged['date_numeric_orig'].unique())}")


if __name__ == "__main__":
    main()
