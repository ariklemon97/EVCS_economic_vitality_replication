"""
05_propensity_score_matching.py

Implements the two-step PSM from Zheng et al. (2024) for narrow and broad
study windows. Outputs are window-scoped so narrow and broad matched panels
cannot overwrite one another:
  1. Period 1 — All POIs           → df_psm_{window}_p1_all.csv
  2. Period 1 — Disadvantaged only → df_psm_{window}_p1_dis.csv
  3. Period 2 — All POIs           → df_psm_{window}_p2_all.csv
  4. Period 2 — Disadvantaged only → df_psm_{window}_p2_dis.csv

Each group is:
  Step 1: Exact match on NAICS 2-digit sector (44, 45, 71, 72)
  Step 2: 1:1 nearest-neighbour PSM on all covariates:
    Built Environment:   pop_density, building_density, road_miles_auto,
                         intersections_auto, walkability_index
    Socio-demographics:  Median_Household_Income, Pct_Employed, Pct_Male, Pct_Minority
    EV market:           EV_sales_per_1000
    POI-level baseline:  baseline_lcus, baseline_lspend

Disadvantaged definition: POI is in a CEC/Justice40 designated underprivileged
community. Flag column expected: 'is_disadvantaged' (1/0).
If absent, a fallback is derived from bottom quintile of Median_Household_Income.
"""

import argparse
import os
import logging
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from scipy.spatial import KDTree
from pathlib import Path
import sys

PROJECT_ROOT_PATH = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_PATH))

from code.analysis_config import BROAD_WINDOW, NARROW_WINDOW, StudyWindow

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

PROJECT_ROOT  = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")

# Full covariate list matching MEMORY.md / Data & Methods
MATCH_COVARIATES = [
    # Built environment (EPA SLD)
    'pop_density', 'building_density', 'road_miles_auto',
    'intersections_auto', 'walkability_index',
    # Socio-demographics (ACS)
    'Median_Household_Income', 'Pct_Employed', 'Pct_Male', 'Pct_Minority',
    # EV market (CEC)
    'EV_sales_per_1000',
    # POI-level baseline outcomes
    'baseline_lcus', 'baseline_lspend',
]

# Target NAICS sectors only (2-digit)
VALID_SECTORS = {44, 45, 71, 72}

WINDOWS: dict[str, StudyWindow] = {
    "narrow": NARROW_WINDOW,
    "broad": BROAD_WINDOW,
}


def panel_filename(window_name: str, period: str, sample: str) -> str:
    return f"df_psm_{window_name}_{period}_{sample}.csv"


# =============================================================================
# Helpers
# =============================================================================

def compute_baseline_features(df, baseline_period_max_seq):
    """
    Compute pre-treatment baseline lcus/lspend for each POI
    using periods with date_numeric <= baseline_period_max_seq.
    """
    bl = df[df['date_numeric'] <= baseline_period_max_seq].groupby('placekey').agg(
        baseline_lcus=('lcus',   'mean'),
        baseline_lspend=('lspend','mean'),
    ).reset_index()
    return bl


def build_poi_crosssection(df, baseline):
    """
    One row per placekey with treatment status, NAICS, and covariates.
    Uses 'max' (not 'first') for covariates because some panel rows have NaN
    for time-invariant covariates and 'first' may return a NaN row.
    'max' returns the non-NaN value whenever at least one row has a valid one.
    """
    agg_dict = {
        'is_treated':         ('is_treated',         'first'),
        'first_treat_period': ('first_treat_period',  'first'),
        'naics_code':         ('naics_code',          'first'),
    }
    # Covariates: use max() to surface any non-NaN value (they are time-invariant)
    for cov in MATCH_COVARIATES[:-2]:   # exclude baseline outcomes (added separately)
        if cov in df.columns:
            agg_dict[cov] = (cov, 'max')

    # Disadvantaged flag if present
    if 'is_disadvantaged' in df.columns:
        agg_dict['is_disadvantaged'] = ('is_disadvantaged', 'max')

    poi = df.groupby('placekey').agg(**agg_dict).reset_index()
    poi = poi.merge(baseline, on='placekey', how='left')

    poi['naics_sector'] = (
        pd.to_numeric(poi['naics_code'], errors='coerce') // 10000
    ).astype('Int64')

    return poi


def flag_disadvantaged(poi, acs_income_col='Median_Household_Income'):
    """
    If 'is_disadvantaged' is missing, approximate it as being in the
    bottom income quintile among all POIs in the sample.
    """
    if 'is_disadvantaged' not in poi.columns:
        logging.warning(
            "No 'is_disadvantaged' column found. "
            "Approximating with bottom-quintile Median_Household_Income."
        )
        if acs_income_col in poi.columns:
            threshold = poi[acs_income_col].quantile(0.2)
            poi['is_disadvantaged'] = (poi[acs_income_col] <= threshold).astype(int)
        else:
            poi['is_disadvantaged'] = 0
    return poi


def run_psm_for_group(poi_group, label):
    """
    Run exact-match-then-NN-PSM for a given POI cross-section subset.
    Returns (matched_treated_placekeys, matched_control_placekeys).
    """
    # Available covariates (paper uses all; fall back gracefully for missing ones)
    available_covs = [c for c in MATCH_COVARIATES if c in poi_group.columns]
    missing_covs   = [c for c in MATCH_COVARIATES if c not in poi_group.columns]
    if missing_covs:
        logging.warning(f"[{label}] Missing covariates (will be absent from matching): {missing_covs}")

    # Drop rows with NA in any available covariate
    poi_clean = poi_group.dropna(subset=available_covs).copy()
    logging.info(
        f"[{label}] {len(poi_group)} POIs → {len(poi_clean)} after dropping NAs "
        f"({len(poi_group)-len(poi_clean)} dropped)"
    )

    # Filter to valid NAICS sectors
    poi_clean = poi_clean[poi_clean['naics_sector'].isin(VALID_SECTORS)]
    logging.info(
        f"[{label}] {len(poi_clean)} POIs after NAICS filter "
        f"(kept sectors {sorted(poi_clean['naics_sector'].unique())})"
    )

    matched_treated = []
    matched_control = []

    for sector in sorted(poi_clean['naics_sector'].unique()):
        sector_data = poi_clean[poi_clean['naics_sector'] == sector]
        treated = sector_data[sector_data['is_treated'] == 1]
        control = sector_data[sector_data['is_treated'] == 0]

        if len(treated) == 0 or len(control) == 0:
            logging.warning(
                f"[{label}] Sector {sector}: {len(treated)} treated, "
                f"{len(control)} control — skipping"
            )
            continue

        logging.info(
            f"[{label}] Sector {sector}: {len(treated)} treated, {len(control)} control"
        )

        if len(treated) < 2 or len(control) < 2:
            logging.warning(f"[{label}] Sector {sector}: too few — skipping")
            continue

        # Standardise covariates within sector
        X        = sector_data[available_covs].values
        scaler   = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        y = sector_data['is_treated'].values

        # Logistic regression for propensity score
        try:
            lr     = LogisticRegression(max_iter=2000, solver='lbfgs', C=1.0)
            lr.fit(X_scaled, y)
            pscore = lr.predict_proba(X_scaled)[:, 1]
            auc_approx = pscore[y == 1].mean() - pscore[y == 0].mean()
            logging.info(
                f"[{label}] Sector {sector}: propensity score fit OK "
                f"(mean diff treated-control: {auc_approx:.4f})"
            )
        except Exception as e:
            logging.warning(
                f"[{label}] Sector {sector}: logistic regression failed ({e}) "
                f"— falling back to covariate distance"
            )
            pscore = np.zeros(len(sector_data))

        # 1:1 nearest-neighbour matching without replacement on propensity score
        treated_idx = np.where(y == 1)[0]
        control_idx = np.where(y == 0)[0]

        p_treated = pscore[treated_idx].reshape(-1, 1)
        p_control = pscore[control_idx].reshape(-1, 1)

        tree         = KDTree(p_control)
        used_controls = set()
        placekeys    = sector_data['placekey'].values

        # Sort treated by propensity score (closest to 0.5 first → hardest to match)
        treated_order = np.argsort(np.abs(p_treated.ravel() - 0.5))

        for ti in treated_order:
            k = min(20, len(control_idx))
            _, neighbors = tree.query(p_treated[ti].reshape(1, -1), k=k)
            for ni in neighbors[0]:
                if ni not in used_controls:
                    used_controls.add(ni)
                    matched_treated.append(placekeys[treated_idx[ti]])
                    matched_control.append(placekeys[control_idx[ni]])
                    break

    logging.info(f"[{label}] Total matched pairs: {len(matched_treated)}")
    return matched_treated, matched_control


def filter_and_save(df_full, matched_treated, matched_control, baseline,
                    output_path, label):
    """
    Filter the full panel to matched POIs, add match metadata, save as CSV.
    """
    matched_pois = set(matched_treated) | set(matched_control)
    df_m = df_full[df_full['placekey'].isin(matched_pois)].copy()
    df_m = df_m.merge(baseline, on='placekey', how='left', suffixes=('', '_bl'))

    # match_pair_id: same integer for each matched (treated, control) pair
    pair_map = {}
    for i, (t, c) in enumerate(zip(matched_treated, matched_control)):
        pair_map[t] = i
        pair_map[c] = i
    df_m['match_pair_id'] = df_m['placekey'].map(pair_map)

    df_m.to_csv(output_path, index=False)
    logging.info(
        f"[{label}] Saved {output_path} — shape: {df_m.shape}, "
        f"treated POIs: {df_m[df_m['is_treated']==1]['placekey'].nunique()}, "
        f"control POIs: {df_m[df_m['is_treated']==0]['placekey'].nunique()}"
    )

    # Balance diagnostics
    available_covs = [c for c in MATCH_COVARIATES if c in df_m.columns]
    logging.info(f"[{label}] === Balance Diagnostics (SMD) ===")
    for cov in available_covs:
        t_mean    = df_m.loc[df_m['is_treated'] == 1, cov].astype(float).mean()
        c_mean    = df_m.loc[df_m['is_treated'] == 0, cov].astype(float).mean()
        pool_std  = df_m[cov].astype(float).std()
        smd       = (t_mean - c_mean) / pool_std if pool_std > 0 else 0
        logging.info(
            f"[{label}]   {cov:35s}: T={t_mean:10.4f}, C={c_mean:10.4f}, SMD={smd:+.4f}"
        )


def build_window_panels(df: pd.DataFrame, window_name: str, window: StudyWindow) -> list[str]:
    logging.info("=== Building %s matched panels ===", window_name)

    # ── NAICS sector guard ────────────────────────────────────────────────────
    if 'naics_sector' not in df.columns and 'naics_code' in df.columns:
        df['naics_sector'] = (
            pd.to_numeric(df['naics_code'], errors='coerce') // 10000
        ).astype('Int64')
    df = df[df['naics_sector'].isin(VALID_SECTORS)].copy()
    logging.info(f"After NAICS sector filter: {len(df)} rows")

    if 'local_business_context' in df.columns:
        df['local_business_context'] = pd.to_numeric(
            df['local_business_context'], errors='coerce'
        ).fillna(0).astype(int)
        pre_context = len(df)
        df = df[df['local_business_context'] == 1].copy()
        logging.info(
            f"Local-business context filter: {pre_context} → {len(df)} rows "
            f"(dropped {pre_context - len(df)})"
        )
    else:
        logging.warning("local_business_context is missing; retaining all target POIs for backward compatibility.")

    # ── Disadvantaged flag ────────────────────────────────────────────────────
    # Will be derived per group if missing

    # ── Split into Period 1 and Period 2 for this study window ───────────────
    # Use original YYYYMM for period splits
    p1_mask = (df['date_numeric_orig'] >= window.period_1_start) & (df['date_numeric_orig'] <= window.period_1_end)
    p2_mask = (df['date_numeric_orig'] >= window.period_2_start) & (df['date_numeric_orig'] <= window.period_2_end)

    # CRITICAL: For matching, redefine treatment per period from the actual
    # first exposure month, not from arbitrary panel rows.
    if 'open_yyyymm' not in df.columns:
        raise SystemExit("open_yyyymm is missing; rerun 04_compile_regression_panel.py after rebuilding POI-EVCS matches.")
    df['open_yyyymm'] = pd.to_numeric(df['open_yyyymm'], errors='coerce').fillna(0).astype(int)
    df['is_treated_P1'] = (
        (df['open_yyyymm'] >= window.period_1_start) &
        (df['open_yyyymm'] <= window.period_1_end)
    ).astype(int)
    df['is_treated_P2'] = (
        (df['open_yyyymm'] >= window.period_2_start) &
        (df['open_yyyymm'] <= window.period_2_end)
    ).astype(int)

    df_p1 = df[p1_mask].copy()
    df_p2 = df[p2_mask].copy()
    
    # Map the relevant treatment flag to 'is_treated' for the build_poi_crosssection function
    df_p1['is_treated'] = df_p1['is_treated_P1']
    df_p2['is_treated'] = df_p2['is_treated_P2']
    
    logging.info(f"Period 1 rows: {len(df_p1)}, Period 2 rows: {len(df_p2)}")

    # ── Baseline features ─────────────────────────────────────────────────────
    # Period 1: baseline = all of 2019 (first 12 sequential periods)
    p1_max_seq = df_p1['date_numeric'].max()
    baseline_p1 = compute_baseline_features(df_p1, p1_max_seq)

    # Period 2: baseline = Feb 2021 only (first period of P2, seq index 13)
    p2_min_seq = df_p2['date_numeric'].min()
    baseline_p2 = compute_baseline_features(df_p2, p2_min_seq)

    # ── POI cross-sections ────────────────────────────────────────────────────
    poi_p1 = build_poi_crosssection(df_p1, baseline_p1)
    poi_p2 = build_poi_crosssection(df_p2, baseline_p2)

    # ── Disadvantaged flags ───────────────────────────────────────────────────
    poi_p1 = flag_disadvantaged(poi_p1)
    poi_p2 = flag_disadvantaged(poi_p2)

    # ─────────────────────────────────────────────────────────────────────────
    # GROUP 1: Period 1 — All POIs
    # ─────────────────────────────────────────────────────────────────────────
    outputs = []
    mt1, mc1 = run_psm_for_group(poi_p1, "P1_All")
    p1_all_path = os.path.join(PROCESSED_DIR, panel_filename(window_name, "p1", "all"))
    filter_and_save(df_p1, mt1, mc1, baseline_p1,
                    p1_all_path,
                    "P1_All")
    outputs.append(p1_all_path)

    # ─────────────────────────────────────────────────────────────────────────
    # GROUP 2: Period 1 — Disadvantaged POIs only
    # ─────────────────────────────────────────────────────────────────────────
    poi_p1_dis = poi_p1[poi_p1['is_disadvantaged'] == 1].copy()
    logging.info(f"[P1_Dis] POIs in disadvantaged communities: {len(poi_p1_dis)}")
    if len(poi_p1_dis) > 0:
        mt2, mc2 = run_psm_for_group(poi_p1_dis, "P1_Dis")
        p1_dis_path = os.path.join(PROCESSED_DIR, panel_filename(window_name, "p1", "dis"))
        filter_and_save(df_p1[df_p1['placekey'].isin(poi_p1_dis['placekey'])],
                        mt2, mc2, baseline_p1,
                        p1_dis_path,
                        "P1_Dis")
        outputs.append(p1_dis_path)
    else:
        logging.warning("[P1_Dis] No disadvantaged POIs found — skipping.")

    # ─────────────────────────────────────────────────────────────────────────
    # GROUP 3: Period 2 — All POIs
    # ─────────────────────────────────────────────────────────────────────────
    mt3, mc3 = run_psm_for_group(poi_p2, "P2_All")
    p2_all_path = os.path.join(PROCESSED_DIR, panel_filename(window_name, "p2", "all"))
    filter_and_save(df_p2, mt3, mc3, baseline_p2,
                    p2_all_path,
                    "P2_All")
    outputs.append(p2_all_path)

    # ─────────────────────────────────────────────────────────────────────────
    # GROUP 4: Period 2 — Disadvantaged POIs only
    # ─────────────────────────────────────────────────────────────────────────
    poi_p2_dis = poi_p2[poi_p2['is_disadvantaged'] == 1].copy()
    logging.info(f"[P2_Dis] POIs in disadvantaged communities: {len(poi_p2_dis)}")
    if len(poi_p2_dis) > 0:
        mt4, mc4 = run_psm_for_group(poi_p2_dis, "P2_Dis")
        p2_dis_path = os.path.join(PROCESSED_DIR, panel_filename(window_name, "p2", "dis"))
        filter_and_save(df_p2[df_p2['placekey'].isin(poi_p2_dis['placekey'])],
                        mt4, mc4, baseline_p2,
                        p2_dis_path,
                        "P2_Dis")
        outputs.append(p2_dis_path)
    else:
        logging.warning("[P2_Dis] No disadvantaged POIs found — skipping.")

    logging.info("=== %s PSM complete. Outputs: %s ===", window_name, ", ".join(os.path.basename(p) for p in outputs))
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--window",
        choices=["narrow", "broad", "all"],
        default="all",
        help="Study window to match. Default builds both window-scoped panel sets.",
    )
    return parser.parse_args()


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()
    input_path = os.path.join(PROCESSED_DIR, "df_pre_match.csv")
    if not os.path.exists(input_path):
        logging.error("df_pre_match.csv not found. Run 04_compile_regression_panel.py first.")
        return

    logging.info("Loading pre-match panel...")
    id_cols      = ['placekey', 'date', 'date_numeric', 'date_numeric_orig', 'is_treated',
                    'first_treat_period', 'open_yyyymm', 'naics_code', 'naics_sector', 'county_fips', 'total_ports',
                    'commercial_adjacent_evcs', 'commercial_poi_count_500m',
                    'local_business_context', 'local_business_count_500m']
    outcome_cols = ['lcus', 'lspend', 'port_treat', 'D_it', 'PC_it',
                    'port_treat_level2', 'port_treat_dc',
                    'median_dwell', 'median_dist_home', 'avg_customer_income',
                    'X0_100m', 'X100_200m', 'X200_300m', 'X300_400m', 'X400_500m',
                    'port_treat_X0_100m', 'port_treat_X100_200m', 'port_treat_X200_300m', 
                    'port_treat_X300_400m', 'port_treat_X400_500m']
    income_cols  = ['cus_.25k','cus_25.45k','cus_45.60k','cus_60.75k',
                    'cus_75.100k','cus_100.150k','cus_.150k']
    covariate_cols = [c for c in MATCH_COVARIATES[:-2]]  # exclude baseline outcomes
    dis_col        = ['is_disadvantaged'] if True else []

    use_cols = id_cols + outcome_cols + income_cols + covariate_cols + dis_col
    df = pd.read_csv(input_path, usecols=lambda c: c in use_cols, low_memory=False)
    logging.info(f"Loaded {len(df)} rows, {df['placekey'].nunique()} unique POIs")
    logging.info(f"Available columns: {df.columns.tolist()}")

    window_names = list(WINDOWS) if args.window == "all" else [args.window]
    outputs = []
    for window_name in window_names:
        outputs.extend(build_window_panels(df.copy(), window_name, WINDOWS[window_name]))
    logging.info("=== PSM complete. Wrote %s panel files. ===", len(outputs))


if __name__ == "__main__":
    main()
