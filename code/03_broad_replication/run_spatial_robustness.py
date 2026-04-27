"""
Spatial-competition robustness estimators.

This script keeps the primary spatial outputs untouched and writes two
diagnostic variants:

1. Broad-window binned event-time CS estimates.
2. Short-window CS estimates using 2019 and February 2021 through June 2023.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import norm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from code.analysis_config import OUTPUT_DIR, PROCESSED_DIR
from code.estimation_utils import CSDIDResult, run_csdid


TABLE_DIR = OUTPUT_DIR / "tables" / "broad"
CS_CONTROL_GROUP = "notyettreated"
OUTCOMES = ["lcus", "lspend"]


def load_spatial_panel(short_window: bool = False) -> pd.DataFrame:
    path = PROCESSED_DIR / "df_final_broad.csv"
    if not path.exists():
        raise SystemExit("Missing df_final_broad.csv. Run the broad processing pipeline first.")

    required = [
        "placekey",
        "date",
        "date_numeric",
        "is_treated",
        "Treatment_Competitor",
        "competitor_first_treat_period",
        "lcus",
        "lspend",
    ]
    header = pd.read_csv(path, nrows=0).columns
    optional = ["competitor_commercial_adjacent", "local_business_context"]
    usecols = required + [col for col in optional if col in header]
    df = pd.read_csv(path, usecols=usecols, low_memory=False)
    df = df.sort_values(["placekey", "date"]).drop_duplicates(["placekey", "date"], keep="first")
    if "competitor_commercial_adjacent" not in df.columns:
        df["competitor_commercial_adjacent"] = df["Treatment_Competitor"]
    df["competitor_commercial_adjacent"] = pd.to_numeric(
        df["competitor_commercial_adjacent"], errors="coerce"
    ).fillna(0).astype(int)
    if "local_business_context" not in df.columns:
        df["local_business_context"] = 1
    df["local_business_context"] = pd.to_numeric(df["local_business_context"], errors="coerce").fillna(0).astype(int)

    if short_window:
        df = df[(df["date"].between("2019-01", "2019-12")) | (df["date"].between("2021-02", "2023-06"))].copy()

    observed_periods = sorted(df["date_numeric"].unique())
    remap = {old: idx + 1 for idx, old in enumerate(observed_periods)}
    df["date_numeric_window"] = df["date_numeric"].map(remap).astype(int)
    df["spatial_group"] = np.where(df["Treatment_Competitor"] == 1, df["competitor_first_treat_period"], 0)
    df["spatial_group_window"] = df["spatial_group"].map(remap).fillna(0).astype(int)

    max_time = df["date_numeric_window"].max()
    keep = (
        (df["is_treated"] == 0)
        & (df["local_business_context"] == 1)
        & ((df["Treatment_Competitor"] == 0) | (df["competitor_commercial_adjacent"] == 1))
        & ((df["spatial_group_window"] == 0) | (df["spatial_group_window"] < max_time))
    )
    return df.loc[keep].copy()


def weighted_summary(df: pd.DataFrame) -> pd.Series:
    total_cells = len(df)
    valid = np.isfinite(df["att"]) & np.isfinite(df["se"]) & np.isfinite(df["group_size"])
    valid &= df["group_size"].astype(float) > 0
    valid_cells = int(valid.sum())
    if valid_cells == 0:
        return pd.Series(
            {
                "ATT": np.nan,
                "SE": np.nan,
                "CI_lower": np.nan,
                "CI_upper": np.nan,
                "p_value": np.nan,
                "n_total_cells": total_cells,
                "n_valid_cells": 0,
                "weight_sum": 0.0,
            }
        )

    work = df.loc[valid, ["att", "se", "group_size"]].astype(float)
    weights = work["group_size"]
    att = float(np.average(work["att"], weights=weights))
    se = float(np.sqrt(np.average(np.square(work["se"]), weights=weights)))
    return pd.Series(
        {
            "ATT": att,
            "SE": se,
            "CI_lower": att - 1.96 * se,
            "CI_upper": att + 1.96 * se,
            "p_value": 2 * norm.sf(abs(att / se)) if se else np.nan,
            "n_total_cells": total_cells,
            "n_valid_cells": valid_cells,
            "weight_sum": float(weights.sum()),
        }
    )


def event_bin(event_time: float) -> str | None:
    if event_time < 0:
        return "pre"
    if event_time <= 12:
        return "0_12"
    if event_time <= 36:
        return "13_36"
    return "37_plus"


def flatten_simple(result: CSDIDResult, dataset: str, outcome: str, window: str) -> pd.DataFrame:
    simple = result.simple.copy()
    simple["window"] = window
    simple["dataset"] = dataset
    simple["outcome"] = outcome
    simple["p_value"] = 2 * norm.sf(abs(simple["ATT"] / simple["SE"]))
    return simple


def flatten_bins(result: CSDIDResult, dataset: str, outcome: str, window: str) -> pd.DataFrame:
    att_gt = result.att_gt.copy()
    att_gt["bin"] = att_gt["event_time"].apply(event_bin)
    att_gt = att_gt[att_gt["bin"].notna()].copy()
    rows = (
        att_gt.groupby("bin", sort=False)
        .apply(weighted_summary, include_groups=False)
        .reset_index()
    )
    rows["window"] = window
    rows["dataset"] = dataset
    rows["outcome"] = outcome
    return rows


def run_window(window: str, short_window: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = load_spatial_panel(short_window=short_window)
    simple_rows = []
    binned_rows = []
    for outcome in OUTCOMES:
        result = run_csdid(
            df,
            outcome,
            [],
            time_col="date_numeric_window",
            group_col="spatial_group_window",
            control_group=CS_CONTROL_GROUP,
            est_method="reg",
        )
        simple_rows.append(flatten_simple(result, "Spatial_All", outcome, window))
        binned_rows.append(flatten_bins(result, "Spatial_All", outcome, window))
    return pd.concat(simple_rows, ignore_index=True), pd.concat(binned_rows, ignore_index=True)


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    broad_simple, broad_bins = run_window("2019_2021_2026", short_window=False)
    broad_bins.to_csv(TABLE_DIR / "spatial_competition_binned_broad.csv", index=False)

    short_simple, short_bins = run_window("2019_2021_2023", short_window=True)
    short_simple.to_csv(TABLE_DIR / "spatial_competition_short_window.csv", index=False)
    short_bins.to_csv(TABLE_DIR / "spatial_competition_binned_short_window.csv", index=False)

    comparison = pd.concat(
        [
            broad_simple.assign(specification="simple_broad"),
            broad_bins.assign(specification="binned_broad"),
            short_simple.assign(specification="simple_short_window"),
            short_bins.assign(specification="binned_short_window"),
        ],
        ignore_index=True,
        sort=False,
    )
    comparison.to_csv(TABLE_DIR / "spatial_competition_robustness_comparison.csv", index=False)

    print("Wrote spatial robustness outputs to", TABLE_DIR)


if __name__ == "__main__":
    main()
