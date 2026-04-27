"""
Python implementation of the broad replication and spatial-competition extension.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from code.analysis_config import OUTPUT_DIR, PROCESSED_DIR
from code.estimation_utils import CSDIDResult, ensure_parent, plot_event_study, run_csdid


TABLE_DIR = OUTPUT_DIR / "tables" / "broad"
FIGURE_DIR = OUTPUT_DIR / "figures" / "broad"

DR_COVARIATES: list[str] = []
CS_CONTROL_GROUP = "notyettreated"

INCOME_BUCKETS = [
    "cus_.25k",
    "cus_25.45k",
    "cus_45.60k",
    "cus_60.75k",
    "cus_75.100k",
    "cus_100.150k",
    "cus_.150k",
]


def load_dedup_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    return df.sort_values(["placekey", "date"]).drop_duplicates(["placekey", "date"], keep="first")


def load_broad_main_panel() -> pd.DataFrame:
    frames = []
    for filename in ["df_psm_broad_p1_all.csv", "df_psm_broad_p2_all.csv"]:
        path = PROCESSED_DIR / filename
        if path.exists():
            frames.append(load_dedup_csv(path))
    if not frames:
        raise SystemExit("Missing matched broad panels. Run 05_propensity_score_matching.py --window broad first.")
    pooled = pd.concat(frames, ignore_index=True).sort_values(["placekey", "date"]).drop_duplicates(["placekey", "date"], keep="first")
    if "commercial_adjacent_evcs" not in pooled.columns:
        pooled["commercial_adjacent_evcs"] = pooled["is_treated"]
    pooled["commercial_adjacent_evcs"] = pd.to_numeric(
        pooled["commercial_adjacent_evcs"], errors="coerce"
    ).fillna(0).astype(int)
    if "local_business_context" not in pooled.columns:
        pooled["local_business_context"] = 1
    pooled["local_business_context"] = pd.to_numeric(
        pooled["local_business_context"], errors="coerce"
    ).fillna(0).astype(int)
    return pooled[
        (pooled["local_business_context"] == 1)
        & ((pooled["is_treated"] == 0) | (pooled["commercial_adjacent_evcs"] == 1))
    ].copy()


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


def flatten_results(result: CSDIDResult, dataset: str, outcome: str) -> pd.DataFrame:
    pieces = []
    for frame in [result.simple, result.dynamic, result.group, result.calendar]:
        if frame.empty:
            continue
        work = frame.copy()
        work["dataset"] = dataset
        work["outcome"] = outcome
        pieces.append(work)
    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()


def run_main_broad() -> pd.DataFrame:
    pooled = load_broad_main_panel()
    rows = []
    for outcome in ["lcus", "lspend"]:
        result = run_csdid(
            pooled,
            outcome,
            DR_COVARIATES,
            control_group=CS_CONTROL_GROUP,
            est_method="reg",
        )
        rows.append(flatten_results(result, "All_POIs", outcome))
        plot_event_study(
            result.dynamic,
            title=f"CS Event Study: {outcome} | All POIs",
            output_path=FIGURE_DIR / f"cs_event_study_All_POIs_{outcome}.pdf",
        )
    final = pd.concat(rows, ignore_index=True)
    final.to_csv(TABLE_DIR / "cs_all_results.csv", index=False)
    final.query("agg_type == 'simple'")[["dataset", "outcome", "ATT", "SE", "CI_lower", "CI_upper"]].to_csv(
        TABLE_DIR / "cs_simple_att_summary.csv", index=False
    )
    return final


def load_extension_panel() -> pd.DataFrame:
    path = PROCESSED_DIR / "df_final_broad.csv"
    if not path.exists():
        raise SystemExit("Missing df_final_broad.csv.")
    df = load_dedup_csv(path)
    if "competitor_commercial_adjacent" not in df.columns:
        df["competitor_commercial_adjacent"] = (
            df["Treatment_Competitor"] if "Treatment_Competitor" in df.columns else 0
        )
    df["competitor_commercial_adjacent"] = pd.to_numeric(
        df["competitor_commercial_adjacent"], errors="coerce"
    ).fillna(0).astype(int)
    if "local_business_context" not in df.columns:
        df["local_business_context"] = 1
    df["local_business_context"] = pd.to_numeric(
        df["local_business_context"], errors="coerce"
    ).fillna(0).astype(int)
    df["poi_type"] = df["naics_code"].map(map_poi_type)
    df["spatial_group"] = np.where(df["Treatment_Competitor"] == 1, df["competitor_first_treat_period"], 0)
    max_time = df["date_numeric"].max()
    return df[
        (df["is_treated"] == 0)
        & (df["local_business_context"] == 1)
        & ((df["Treatment_Competitor"] == 0) | (df["competitor_commercial_adjacent"] == 1))
        & ((df["spatial_group"] == 0) | (df["spatial_group"] < max_time))
    ].copy()


def run_extension_spec(df: pd.DataFrame, dataset: str, outcomes: list[str]) -> pd.DataFrame:
    rows = []
    for outcome in outcomes:
        work = df.copy()
        if outcome in INCOME_BUCKETS:
            work[f"log_{outcome}"] = np.log1p(work[outcome].clip(lower=0))
            outcome_name = f"log_{outcome}"
        else:
            outcome_name = outcome
        result = run_csdid(
            work,
            outcome_name,
            DR_COVARIATES,
            group_col="spatial_group",
            control_group=CS_CONTROL_GROUP,
            est_method="reg",
        )
        rows.append(flatten_results(result, dataset, outcome))
        if outcome in {"lcus", "lspend"}:
            plot_event_study(
                result.dynamic,
                title=f"Spatial Competition Event Study: {outcome} | {dataset}",
                output_path=FIGURE_DIR / f"spatial_event_study_{dataset}_{outcome}.pdf",
            )
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def run_spatial_extension() -> None:
    df = load_extension_panel()

    main = run_extension_spec(df, "Spatial_All", ["lcus", "lspend"])
    main.to_csv(TABLE_DIR / "spatial_competition_results.csv", index=False)

    poi_rows = []
    for poi_type, chunk in df.groupby("poi_type"):
        if poi_type == "Other":
            continue
        subset = df[(df["poi_type"] == poi_type) | (df["Treatment_Competitor"] == 0)].copy()
        poi_rows.append(run_extension_spec(subset, f"POI_{poi_type}", ["lcus", "lspend"]))
    pd.concat(poi_rows, ignore_index=True).to_csv(TABLE_DIR / "spatial_competition_poi_type.csv", index=False)

    charger_rows = []
    for label, flag in [("Level2", "competitor_has_level2"), ("DCFast", "competitor_has_dc_fast")]:
        subset = df[(df[flag] == 1) | (df["Treatment_Competitor"] == 0)].copy()
        charger_rows.append(run_extension_spec(subset, f"Charger_{label}", ["lcus", "lspend"]))
    pd.concat(charger_rows, ignore_index=True).to_csv(TABLE_DIR / "spatial_competition_charger_type.csv", index=False)

    income = run_extension_spec(df, "Spatial_Income", INCOME_BUCKETS)
    income.to_csv(TABLE_DIR / "spatial_competition_income_groups.csv", index=False)

    simple = main.query("agg_type == 'simple'")
    beta_s = simple.loc[simple["outcome"] == "lspend", "ATT"].iloc[0] if not simple.loc[simple["outcome"] == "lspend"].empty else np.nan
    beta_c = simple.loc[simple["outcome"] == "lcus", "ATT"].iloc[0] if not simple.loc[simple["outcome"] == "lcus"].empty else np.nan
    exposed = df[df["Treatment_Competitor"] == 1].copy()
    monetary = pd.DataFrame(
        [
            {
                "dataset": "Spatial_All",
                "beta_spend": beta_s,
                "beta_customers": beta_c,
                "avg_competitor_ports": exposed["competitor_total_ports"].mean(),
                "avg_baseline_spending": np.expm1(exposed["lspend"]).mean(),
                "avg_baseline_customers": np.expm1(exposed["lcus"]).mean(),
                "annual_spend_effect": beta_s * np.expm1(exposed["lspend"]).mean() * 12 if pd.notna(beta_s) else np.nan,
                "annual_customer_effect": beta_c * np.expm1(exposed["lcus"]).mean() * 12 if pd.notna(beta_c) else np.nan,
            }
        ]
    )
    monetary.to_csv(TABLE_DIR / "spatial_competition_monetary_impacts.csv", index=False)


def run_spatial_section(section: str) -> None:
    df = load_extension_panel()
    if section == "spatial-main":
        run_extension_spec(df, "Spatial_All", ["lcus", "lspend"]).to_csv(TABLE_DIR / "spatial_competition_results.csv", index=False)
        return
    if section == "spatial-poi":
        poi_rows = []
        for poi_type, chunk in df.groupby("poi_type"):
            if poi_type == "Other":
                continue
            subset = df[(df["poi_type"] == poi_type) | (df["Treatment_Competitor"] == 0)].copy()
            poi_rows.append(run_extension_spec(subset, f"POI_{poi_type}", ["lcus", "lspend"]))
        pd.concat(poi_rows, ignore_index=True).to_csv(TABLE_DIR / "spatial_competition_poi_type.csv", index=False)
        return
    if section == "spatial-charger":
        charger_rows = []
        for label, flag in [("Level2", "competitor_has_level2"), ("DCFast", "competitor_has_dc_fast")]:
            subset = df[(df[flag] == 1) | (df["Treatment_Competitor"] == 0)].copy()
            charger_rows.append(run_extension_spec(subset, f"Charger_{label}", ["lcus", "lspend"]))
        pd.concat(charger_rows, ignore_index=True).to_csv(TABLE_DIR / "spatial_competition_charger_type.csv", index=False)
        return
    if section == "spatial-income":
        run_extension_spec(df, "Spatial_Income", INCOME_BUCKETS).to_csv(TABLE_DIR / "spatial_competition_income_groups.csv", index=False)
        return
    if section == "spatial-money":
        path = TABLE_DIR / "spatial_competition_results.csv"
        if not path.exists():
            raise SystemExit("Missing spatial_competition_results.csv. Run --section spatial-main first.")
        main = pd.read_csv(path)
        simple = main.query("agg_type == 'simple'")
        beta_s = simple.loc[simple["outcome"] == "lspend", "ATT"].iloc[0] if not simple.loc[simple["outcome"] == "lspend"].empty else np.nan
        beta_c = simple.loc[simple["outcome"] == "lcus", "ATT"].iloc[0] if not simple.loc[simple["outcome"] == "lcus"].empty else np.nan
        exposed = df[df["Treatment_Competitor"] == 1].copy()
        pd.DataFrame(
            [
                {
                    "dataset": "Spatial_All",
                    "beta_spend": beta_s,
                    "beta_customers": beta_c,
                    "avg_competitor_ports": exposed["competitor_total_ports"].mean(),
                    "avg_baseline_spending": np.expm1(exposed["lspend"]).mean(),
                    "avg_baseline_customers": np.expm1(exposed["lcus"]).mean(),
                    "annual_spend_effect": beta_s * np.expm1(exposed["lspend"]).mean() * 12 if pd.notna(beta_s) else np.nan,
                    "annual_customer_effect": beta_c * np.expm1(exposed["lcus"]).mean() * 12 if pd.notna(beta_c) else np.nan,
                }
            ]
        ).to_csv(TABLE_DIR / "spatial_competition_monetary_impacts.csv", index=False)
        return
    raise SystemExit(f"Unsupported section: {section}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--section",
        default="all",
        choices=["all", "main", "spatial-main", "spatial-poi", "spatial-charger", "spatial-income", "spatial-money"],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    if args.section == "all":
        run_main_broad()
        run_spatial_extension()
        return
    if args.section == "main":
        run_main_broad()
        return
    run_spatial_section(args.section)


if __name__ == "__main__":
    main()
