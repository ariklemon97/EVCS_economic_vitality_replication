"""
Appendix TWFE intensity diagnostics for broad and spatial-extension models.

These models deliberately exclude binary-adoption estimands. They preserve the
original substantive treatment structure by estimating effects of active charger
port intensity:

* Broad replication: own active ports within 500m (`port_treat`).
* Spatial extension: competitor active ports within 1km among same NAICS-4
  competitor locations (`competitor_port_treat`).

The estimator is TWFE-style absorbed fixed effects with POI and county-month
fixed effects, clustered by POI, matching the original/narrow model family.

These outputs are retained for appendix comparability checks only. The final
package's preferred broad/spatial estimates are CS and stacked regressions.
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

BASE_COLS = [
    "placekey",
    "date",
    "date_numeric",
    "date_numeric_orig",
    "county_fips",
    "naics_code",
    "is_treated",
    "lcus",
    "lspend",
    "port_treat",
    "port_treat_level2",
    "port_treat_dc",
    "commercial_adjacent_evcs",
    "commercial_poi_count_500m",
    "local_business_context",
    "local_business_count_500m",
    "Treatment_Competitor",
    "competitor_commercial_adjacent",
    "competitor_first_treat_period",
    "competitor_total_ports",
    "competitor_ports_active",
    "competitor_level2_ports_active",
    "competitor_dc_fast_ports_active",
    "competitor_has_level2",
    "competitor_has_dc_fast",
]

INCOME_BUCKETS = [
    "cus_.25k",
    "cus_25.45k",
    "cus_45.60k",
    "cus_60.75k",
    "cus_75.100k",
    "cus_100.150k",
    "cus_.150k",
]


def map_poi_type(naics_code: object) -> str:
    value = str(naics_code)
    if value.startswith("721"):
        return "Hotels"
    if value.startswith("722"):
        return "Restaurants"
    if value.startswith("44711"):
        return "Gas_Convenience"
    if value.startswith("445"):
        return "Grocery"
    if value.startswith("448"):
        return "Clothing"
    if value.startswith("71"):
        return "Arts_Entertainment"
    if value.startswith(("44", "45")):
        return "Retail_Other"
    return "Other"


def add_inference_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["p.value"] = 2 * norm.sf(np.abs(out["estimate"] / out["std.error"]))
    out["pct_effect"] = np.expm1(out["estimate"]) * 100
    out["pct_ci_low95"] = np.expm1(out["ci_low95"]) * 100
    out["pct_ci_hi95"] = np.expm1(out["ci_hi95"]) * 100
    return out


def load_final_broad(extra_cols: list[str] | None = None) -> pd.DataFrame:
    path = PROCESSED_DIR / "df_final_broad.csv"
    if not path.exists():
        raise SystemExit("Missing df_final_broad.csv. Run processing/finalize stages first.")

    header = pd.read_csv(path, nrows=0).columns
    requested = BASE_COLS + (extra_cols or [])
    usecols = [col for col in requested if col in header]
    df = pd.read_csv(path, usecols=usecols, low_memory=False)
    df = df.sort_values(["placekey", "date"]).drop_duplicates(["placekey", "date"], keep="first")
    df["county_date"] = df["county_fips"].astype(str) + "_" + df["date"].astype(str)
    if "commercial_adjacent_evcs" not in df.columns:
        df["commercial_adjacent_evcs"] = df["is_treated"]
    df["commercial_adjacent_evcs"] = pd.to_numeric(df["commercial_adjacent_evcs"], errors="coerce").fillna(0).astype(int)
    if "competitor_commercial_adjacent" not in df.columns:
        df["competitor_commercial_adjacent"] = (
            df["Treatment_Competitor"] if "Treatment_Competitor" in df.columns else 0
        )
    df["competitor_commercial_adjacent"] = pd.to_numeric(
        df["competitor_commercial_adjacent"], errors="coerce"
    ).fillna(0).astype(int)
    if "local_business_context" not in df.columns:
        df["local_business_context"] = 1
    df["local_business_context"] = pd.to_numeric(df["local_business_context"], errors="coerce").fillna(0).astype(int)
    return df


def run_fe(df: pd.DataFrame, outcome: str, regressors: list[str], dataset: str, model: str) -> pd.DataFrame:
    result = run_absorbing_ls(
        df,
        outcome,
        regressors,
        ["placekey", "county_date"],
        "placekey",
    )
    result["dataset"] = dataset
    result["model"] = model
    result["outcome"] = outcome
    return add_inference_columns(result)


def run_broad_intensity() -> None:
    df = load_final_broad()
    df = df[
        (df["local_business_context"] == 1)
        & ((df["is_treated"] == 0) | (df["commercial_adjacent_evcs"] == 1))
    ].copy()
    rows = []
    for outcome in ["lcus", "lspend"]:
        rows.append(run_fe(df, outcome, ["port_treat"], "Broad_All", "own_port_intensity"))
        rows.append(
            run_fe(
                df,
                outcome,
                ["port_treat_level2", "port_treat_dc"],
                "Broad_All",
                "own_port_intensity_by_charger",
            )
        )
    out = pd.concat(rows, ignore_index=True)
    ensure_parent(TABLE_DIR / "broad_intensity_results.csv")
    out.to_csv(TABLE_DIR / "broad_intensity_results.csv", index=False)


def add_competitor_intensity(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ["Treatment_Competitor", "competitor_first_treat_period", "competitor_total_ports"]:
        if col not in out.columns:
            out[col] = 0
    if "competitor_ports_active" in out.columns:
        out["competitor_port_treat"] = pd.to_numeric(out["competitor_ports_active"], errors="coerce").fillna(0.0)
        out["competitor_port_treat_level2"] = pd.to_numeric(
            out.get("competitor_level2_ports_active", 0),
            errors="coerce",
        ).fillna(0.0)
        out["competitor_port_treat_dc"] = pd.to_numeric(
            out.get("competitor_dc_fast_ports_active", 0),
            errors="coerce",
        ).fillna(0.0)
    else:
        active = (
            (out["Treatment_Competitor"].fillna(0).astype(int) == 1)
            & (pd.to_numeric(out["competitor_first_treat_period"], errors="coerce").fillna(0) > 0)
            & (out["date_numeric"] >= pd.to_numeric(out["competitor_first_treat_period"], errors="coerce").fillna(0))
        )
        ports = pd.to_numeric(out["competitor_total_ports"], errors="coerce").fillna(0.0)
        out["competitor_port_treat"] = np.where(active, ports, 0.0)
        out["competitor_port_treat_level2"] = out["competitor_port_treat"] * out.get("competitor_has_level2", 0)
        out["competitor_port_treat_dc"] = out["competitor_port_treat"] * out.get("competitor_has_dc_fast", 0)
    out["poi_type"] = out["naics_code"].map(map_poi_type)
    return out


def run_spatial_intensity() -> None:
    df = load_final_broad(extra_cols=INCOME_BUCKETS)
    df = add_competitor_intensity(df)

    # Spatial competition asks what happens to direct non-EVCS POIs when nearby
    # same-sector competitors receive active charger-port exposure.
    controls = df[df["is_treated"] == 0].copy()
    controls = controls[
        (controls["local_business_context"] == 1)
        & ((controls["Treatment_Competitor"] == 0) | (controls["competitor_commercial_adjacent"] == 1))
    ].copy()

    rows = []
    for outcome in ["lcus", "lspend"]:
        rows.append(run_fe(controls, outcome, ["competitor_port_treat"], "Spatial_All", "competitor_port_intensity"))
        rows.append(
            run_fe(
                controls,
                outcome,
                ["competitor_port_treat_level2", "competitor_port_treat_dc"],
                "Spatial_All",
                "competitor_port_intensity_by_charger",
            )
        )

    pd.concat(rows, ignore_index=True).to_csv(TABLE_DIR / "spatial_competition_intensity_results.csv", index=False)

    poi_rows = []
    for poi_type in sorted(t for t in controls["poi_type"].dropna().unique() if t != "Other"):
        subset = controls[(controls["poi_type"] == poi_type) | (controls["Treatment_Competitor"] == 0)].copy()
        for outcome in ["lcus", "lspend"]:
            poi_rows.append(
                run_fe(
                    subset,
                    outcome,
                    ["competitor_port_treat"],
                    f"POI_{poi_type}",
                    "competitor_port_intensity_poi_type",
                )
            )
    pd.concat(poi_rows, ignore_index=True).to_csv(TABLE_DIR / "spatial_competition_intensity_poi_type.csv", index=False)

    income_rows = []
    for bucket in INCOME_BUCKETS:
        if bucket not in controls.columns:
            continue
        work = controls.copy()
        work[f"log_{bucket}"] = np.log1p(pd.to_numeric(work[bucket], errors="coerce").clip(lower=0))
        income_rows.append(
            run_fe(
                work,
                f"log_{bucket}",
                ["competitor_port_treat"],
                "Spatial_Income",
                "competitor_port_intensity_income",
            ).assign(income_bucket=bucket)
        )
    if income_rows:
        pd.concat(income_rows, ignore_index=True).to_csv(
            TABLE_DIR / "spatial_competition_intensity_income_groups.csv",
            index=False,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--section", choices=["all", "broad", "spatial"], default="all")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    if args.section in {"all", "broad"}:
        run_broad_intensity()
    if args.section in {"all", "spatial"}:
        run_spatial_intensity()


if __name__ == "__main__":
    main()
