"""
Stacked broad and spatial-competition regressions.

For each first-treatment cohort, this builds a cohort-specific stack containing:

* units first treated in that cohort, and
* controls that are not yet treated in each observed stack month.

Rows are restricted to a configurable event-time window. The regression absorbs
stack-specific POI and county-month fixed effects and clusters by the original
POI, which avoids using already-treated units as controls.

The final package uses this script for two paper-facing estimands:

* broad own-port exposure, and
* spatial competition exposure from nearby same-sector competitors.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import norm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from code.analysis_config import OUTPUT_DIR, PROCESSED_DIR
from code.estimation_utils import ensure_parent, run_absorbing_ls


TABLE_DIR = OUTPUT_DIR / "tables" / "broad"
MAIN_TABLE_DIR = OUTPUT_DIR / "tables" / "main"
DIAG_TABLE_DIR = OUTPUT_DIR / "tables" / "diagnostics"

REQUIRED_COLS = [
    "placekey",
    "date_numeric",
    "date_numeric_orig",
    "county_fips",
    "first_treat_period",
    "is_treated",
    "lcus",
    "lspend",
    "port_treat",
    "port_treat_level2",
    "port_treat_dc",
    "Treatment_Competitor",
    "competitor_first_treat_period",
    "competitor_ports_active",
    "competitor_level2_ports_active",
    "competitor_dc_fast_ports_active",
]

OPTIONAL_COLS = [
    "naics_code",
    "naics_code_str",
    "commercial_adjacent_evcs",
    "competitor_commercial_adjacent",
    "commercial_poi_count_500m",
    "local_business_context",
    "local_business_count_500m",
]


def add_inference_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["p.value"] = 2 * norm.sf(np.abs(out["estimate"] / out["std.error"]))
    out["pct_effect"] = np.expm1(out["estimate"]) * 100
    out["pct_ci_low95"] = np.expm1(out["ci_low95"]) * 100
    out["pct_ci_hi95"] = np.expm1(out["ci_hi95"]) * 100
    return out


def load_final_broad() -> pd.DataFrame:
    path = PROCESSED_DIR / "df_final_broad.csv"
    if not path.exists():
        raise SystemExit("Missing df_final_broad.csv. Run corrected exposure/finalize stages first.")

    header = pd.read_csv(path, nrows=0).columns
    base_required = {
        "placekey",
        "date_numeric",
        "date_numeric_orig",
        "county_fips",
        "first_treat_period",
        "is_treated",
        "lcus",
        "lspend",
        "port_treat",
        "port_treat_level2",
        "port_treat_dc",
    }
    missing = sorted(base_required - set(header))
    if missing:
        raise SystemExit(f"df_final_broad.csv is missing columns: {missing}")

    usecols = [col for col in REQUIRED_COLS + OPTIONAL_COLS if col in header]
    df = pd.read_csv(path, usecols=usecols, low_memory=False)
    df = df.sort_values(["placekey", "date_numeric"]).drop_duplicates(["placekey", "date_numeric"], keep="first")
    df["first_treat_period"] = pd.to_numeric(df["first_treat_period"], errors="coerce").fillna(0).astype(int)
    df["date_numeric"] = pd.to_numeric(df["date_numeric"], errors="coerce").astype(int)
    for col in [
        "Treatment_Competitor",
        "competitor_first_treat_period",
        "competitor_ports_active",
        "competitor_level2_ports_active",
        "competitor_dc_fast_ports_active",
    ]:
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    df["Treatment_Competitor"] = df["Treatment_Competitor"].astype(int)
    df["competitor_first_treat_period"] = df["competitor_first_treat_period"].astype(int)
    if "commercial_adjacent_evcs" not in df.columns:
        df["commercial_adjacent_evcs"] = df["is_treated"]
    df["commercial_adjacent_evcs"] = pd.to_numeric(df["commercial_adjacent_evcs"], errors="coerce").fillna(0).astype(int)
    if "competitor_commercial_adjacent" not in df.columns:
        df["competitor_commercial_adjacent"] = df["Treatment_Competitor"]
    df["competitor_commercial_adjacent"] = pd.to_numeric(
        df["competitor_commercial_adjacent"], errors="coerce"
    ).fillna(0).astype(int)
    if "local_business_context" not in df.columns:
        df["local_business_context"] = 1
    df["local_business_context"] = pd.to_numeric(df["local_business_context"], errors="coerce").fillna(0).astype(int)
    return df


def prepare_broad_own_panel(df: pd.DataFrame) -> pd.DataFrame:
    out = df[
        (df["local_business_context"] == 1)
        & ((df["first_treat_period"] == 0) | (df["commercial_adjacent_evcs"] == 1))
    ].copy()
    out["stack_treat_period"] = out["first_treat_period"]
    return out


def prepare_spatial_competition_panel(df: pd.DataFrame) -> pd.DataFrame:
    out = df[
        (df["is_treated"] == 0)
        & (df["local_business_context"] == 1)
        & ((df["Treatment_Competitor"] == 0) | (df["competitor_commercial_adjacent"] == 1))
    ].copy()
    out["stack_treat_period"] = out["competitor_first_treat_period"]
    out["competitor_port_treat"] = out["competitor_ports_active"].astype(float)
    out["competitor_port_treat_level2"] = out["competitor_level2_ports_active"].astype(float)
    out["competitor_port_treat_dc"] = out["competitor_dc_fast_ports_active"].astype(float)
    return out


def build_stacked_panel(
    df: pd.DataFrame,
    pre: int,
    post: int,
    min_treated: int,
    max_control_pois: int | None,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    max_time = int(df["date_numeric"].max())
    cohorts = sorted(g for g in df.loc[df["stack_treat_period"] > 0, "stack_treat_period"].unique() if g < max_time)
    pieces = []
    diagnostics = []
    rng = np.random.default_rng(seed)

    for cohort in cohorts:
        start = max(1, int(cohort) - pre)
        end = min(max_time, int(cohort) + post)
        treated_ids = df.loc[df["stack_treat_period"] == cohort, "placekey"].unique()
        if len(treated_ids) < min_treated:
            continue

        in_window = df["date_numeric"].between(start, end)
        treated = df["stack_treat_period"].eq(cohort)
        clean_control = df["stack_treat_period"].eq(0) | (df["stack_treat_period"] > df["date_numeric"])
        stack = df.loc[in_window & (treated | clean_control)].copy()
        if stack.empty:
            continue

        if max_control_pois is not None:
            control_ids = stack.loc[~stack["stack_treat_period"].eq(cohort), "placekey"].drop_duplicates().to_numpy()
            if len(control_ids) > max_control_pois:
                keep_controls = set(rng.choice(control_ids, size=max_control_pois, replace=False))
                keep_treated = stack["stack_treat_period"].eq(cohort)
                stack = stack.loc[keep_treated | stack["placekey"].isin(keep_controls)].copy()

        stack["stack_id"] = int(cohort)
        stack["event_time"] = stack["date_numeric"] - int(cohort)
        stack["stack_treated"] = stack["stack_treat_period"].eq(cohort).astype(int)
        stack["stack_unit"] = stack["stack_id"].astype(str) + "_" + stack["placekey"].astype(str)
        stack["stack_county_date"] = (
            stack["stack_id"].astype(str)
            + "_"
            + stack["county_fips"].astype(str)
            + "_"
            + stack["date_numeric"].astype(str)
        )
        pieces.append(stack)
        diagnostics.append(
            {
                "stack_id": int(cohort),
                "event_start": int(start - cohort),
                "event_end": int(end - cohort),
                "n_rows": int(len(stack)),
                "n_treated_pois": int(stack.loc[stack["stack_treated"].eq(1), "placekey"].nunique()),
                "n_control_pois": int(stack.loc[stack["stack_treated"].eq(0), "placekey"].nunique()),
                "max_control_pois": max_control_pois,
            }
        )

    if not pieces:
        raise SystemExit("No eligible stacks were built.")

    stacked = pd.concat(pieces, ignore_index=True)
    diag = pd.DataFrame(diagnostics)
    return stacked, diag


def run_stacked_fe(
    stacked: pd.DataFrame,
    outcome: str,
    regressors: list[str],
    dataset: str,
    model: str,
    pre: int,
    post: int,
) -> pd.DataFrame:
    result = run_absorbing_ls(
        stacked,
        outcome,
        regressors,
        ["stack_unit", "stack_county_date"],
        "placekey",
    )
    result["dataset"] = dataset
    result["model"] = model
    result["outcome"] = outcome
    result["pre_window"] = pre
    result["post_window"] = post
    result["n_stacks"] = stacked["stack_id"].nunique()
    result["n_original_pois"] = stacked["placekey"].nunique()
    result["n_stacked_rows"] = len(stacked)
    return add_inference_columns(result)


def run_broad_stacked(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    panel = prepare_broad_own_panel(df)
    stacked, diagnostics = build_stacked_panel(
        panel,
        args.pre,
        args.post,
        args.min_treated,
        args.max_control_pois,
        args.seed,
    )
    diagnostics.to_csv(DIAG_TABLE_DIR / "stacked_broad_own_port_diagnostics.csv", index=False)
    diagnostics.to_csv(TABLE_DIR / "stacked_own_port_intensity_diagnostics.csv", index=False)

    rows = []
    for outcome in ["lcus", "lspend"]:
        rows.append(
            run_stacked_fe(
                stacked,
                outcome,
                ["port_treat"],
                "Stacked_Broad_All",
                "stacked_own_port_intensity",
                args.pre,
                args.post,
            )
        )
        rows.append(
            run_stacked_fe(
                stacked,
                outcome,
                ["port_treat_level2", "port_treat_dc"],
                "Stacked_Broad_All",
                "stacked_own_port_intensity_by_charger",
                args.pre,
                args.post,
            )
        )
    return pd.concat(rows, ignore_index=True)


def run_spatial_stacked(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    panel = prepare_spatial_competition_panel(df)
    stacked, diagnostics = build_stacked_panel(
        panel,
        args.pre,
        args.post,
        args.min_treated,
        args.max_control_pois,
        args.seed,
    )
    diagnostics.to_csv(DIAG_TABLE_DIR / "stacked_spatial_competition_diagnostics.csv", index=False)

    rows = []
    for outcome in ["lcus", "lspend"]:
        rows.append(
            run_stacked_fe(
                stacked,
                outcome,
                ["competitor_port_treat"],
                "Stacked_Spatial_Competition",
                "stacked_competitor_port_intensity",
                args.pre,
                args.post,
            )
        )
        rows.append(
            run_stacked_fe(
                stacked,
                outcome,
                ["competitor_port_treat_level2", "competitor_port_treat_dc"],
                "Stacked_Spatial_Competition",
                "stacked_competitor_port_intensity_by_charger",
                args.pre,
                args.post,
            )
        )
    return pd.concat(rows, ignore_index=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target",
        choices=["all", "broad", "spatial"],
        default="all",
        help="Stacked model family to run for the final package.",
    )
    parser.add_argument("--pre", type=int, default=12, help="Months before cohort treatment to include in each stack.")
    parser.add_argument("--post", type=int, default=36, help="Months after cohort treatment to include in each stack.")
    parser.add_argument("--min-treated", type=int, default=25, help="Minimum treated POIs required for a cohort stack.")
    parser.add_argument(
        "--max-control-pois",
        type=int,
        default=None,
        help="Optional cap on control POIs sampled per stack. Omit to keep all clean controls.",
    )
    parser.add_argument("--seed", type=int, default=20260425, help="Random seed for capped control sampling.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    MAIN_TABLE_DIR.mkdir(parents=True, exist_ok=True)
    DIAG_TABLE_DIR.mkdir(parents=True, exist_ok=True)
    df = load_final_broad()
    outputs = []

    if args.target in {"all", "broad"}:
        broad = run_broad_stacked(df, args)
        broad.to_csv(TABLE_DIR / "stacked_own_port_intensity_results.csv", index=False)
        broad.to_csv(MAIN_TABLE_DIR / "broad_stacked_own_port_results.csv", index=False)
        outputs.append(broad)

    if args.target in {"all", "spatial"}:
        spatial = run_spatial_stacked(df, args)
        spatial.to_csv(TABLE_DIR / "stacked_spatial_competition_results.csv", index=False)
        spatial.to_csv(MAIN_TABLE_DIR / "spatial_competition_stacked_results.csv", index=False)
        outputs.append(spatial)

    out = pd.concat(outputs, ignore_index=True)
    ensure_parent(MAIN_TABLE_DIR / "stacked_results.csv")
    out.to_csv(MAIN_TABLE_DIR / "stacked_results.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
