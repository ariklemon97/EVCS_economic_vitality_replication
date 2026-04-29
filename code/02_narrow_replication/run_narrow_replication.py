"""
Python implementation of the narrow replication using TWFE-style estimators.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from code.analysis_config import NARROW_WINDOW, OUTPUT_DIR, PROCESSED_DIR
from code.estimation_utils import ensure_parent, run_absorbing_ls


TABLE_DIR = OUTPUT_DIR / "tables" / "narrow"
FIGURE_DIR = OUTPUT_DIR / "figures" / "narrow"

plt.rcParams.update(
    {
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
    }
)

INCOME_BRACKETS = [
    ("cus_.25k", "<$25k"),
    ("cus_25.45k", "$25-45k"),
    ("cus_45.60k", "$45-60k"),
    ("cus_60.75k", "$60-75k"),
    ("cus_75.100k", "$75-100k"),
    ("cus_100.150k", "$100-150k"),
    ("cus_.150k", ">$150k"),
]

OTHER_OUTCOMES = {
    "median_dist_home": "Median Distance from Home",
    "median_dwell": "Median Dwell Time",
    "avg_customer_income": "Average Customer Income",
}

POI_MULTIPLIERS = {"Period1_2019": 15, "Period2_2021_2023": 8}


def panel_filename(period: str, sample: str) -> str:
    return f"df_psm_narrow_{period}_{sample}.csv"


def assert_panel_window(df: pd.DataFrame, period: str, sample: str, filename: str) -> None:
    date_numeric_orig = pd.to_numeric(df["date_numeric_orig"], errors="coerce")
    open_month = pd.to_numeric(df["open_yyyymm"], errors="coerce").fillna(0).astype(int)
    treated_open = open_month[df["is_treated"].eq(1)]

    if period == "p1":
        start, end = NARROW_WINDOW.period_1_start, NARROW_WINDOW.period_1_end
    else:
        start, end = NARROW_WINDOW.period_2_start, NARROW_WINDOW.period_2_end

    bad_dates = ~date_numeric_orig.between(start, end)
    if bad_dates.any():
        observed = f"{int(date_numeric_orig.min())}-{int(date_numeric_orig.max())}"
        raise ValueError(f"{filename} has panel months {observed}, expected {start}-{end}.")

    if not treated_open.empty and not treated_open.between(start, end).all():
        observed = f"{int(treated_open.min())}-{int(treated_open.max())}"
        raise ValueError(f"{filename} has treated open months {observed}, expected {start}-{end}.")

    if "match_pair_id" not in df.columns:
        raise ValueError(f"{filename} is missing match_pair_id.")

    if "local_business_context" in df.columns:
        context = pd.to_numeric(df["local_business_context"], errors="coerce").fillna(0).astype(int)
        if not context.eq(1).all():
            raise ValueError(f"{filename} contains POIs outside the local-business context.")

    treated = df.loc[df["is_treated"].eq(1), "placekey"].nunique()
    controls = df.loc[df["is_treated"].eq(0), "placekey"].nunique()
    if treated != controls:
        raise ValueError(f"{filename} has {treated} treated POIs and {controls} control POIs for {sample}.")


def load_panels() -> dict[tuple[str, str], pd.DataFrame]:
    mapping = {
        ("Period1_2019", "All"): ("p1", "all"),
        ("Period1_2019", "Disadvantaged"): ("p1", "dis"),
        ("Period2_2021_2023", "All"): ("p2", "all"),
        ("Period2_2021_2023", "Disadvantaged"): ("p2", "dis"),
    }
    out = {}
    for key, (period_code, sample_code) in mapping.items():
        filename = panel_filename(period_code, sample_code)
        path = PROCESSED_DIR / filename
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {filename}. Run code/01_data_processing/05_propensity_score_matching.py --window narrow first."
            )
        df = pd.read_csv(path, low_memory=False)
        assert_panel_window(df, period_code, key[1], filename)
        df = df.sort_values(["placekey", "date"]).drop_duplicates(["placekey", "date"], keep="first")
        df["county_date"] = df["county_fips"].astype(str) + "_" + df["date"].astype(str)
        out[key] = df
    return out


def append_metadata(result: pd.DataFrame, **kwargs) -> pd.DataFrame:
    for key, value in kwargs.items():
        result[key] = value
    return result


def run_main_models(panels: dict[tuple[str, str], pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for (period, sample), df in panels.items():
        for outcome in ["lcus", "lspend"]:
            result = run_absorbing_ls(df, outcome, ["port_treat"], ["placekey", "county_date"], "placekey")
            rows.append(append_metadata(result, outcome=outcome, period=period, sample=sample))
    final = pd.concat(rows, ignore_index=True)
    ensure_parent(TABLE_DIR / "01_main_model.csv")
    final.to_csv(TABLE_DIR / "01_main_model.csv", index=False)
    return final


def run_distance_models(panels: dict[tuple[str, str], pd.DataFrame]) -> pd.DataFrame:
    bin_vars = [
        ("port_treat_X0_100m", 50, "0-100"),
        ("port_treat_X100_200m", 150, "100-200"),
        ("port_treat_X200_300m", 250, "200-300"),
        ("port_treat_X300_400m", 350, "300-400"),
        ("port_treat_X400_500m", 450, "400-500"),
    ]
    rows = []
    for (period, sample), df in panels.items():
        regressors = [name for name, _, _ in bin_vars if name in df.columns]
        if not regressors:
            continue
        for outcome in ["lcus", "lspend"]:
            result = run_absorbing_ls(df, outcome, regressors, ["placekey", "county_date"], "placekey")
            result["distance_mid"] = result["term"].map({name: mid for name, mid, _ in bin_vars})
            result["distance_bin"] = result["term"].map({name: label for name, _, label in bin_vars})
            rows.append(append_metadata(result, outcome=outcome, period=period, sample=sample))
    final = pd.concat(rows, ignore_index=True)
    final.to_csv(TABLE_DIR / "02_distance_effects.csv", index=False)

    for (period, sample), chunk in final.groupby(["period", "sample"]):
        fig, ax = plt.subplots(figsize=(8, 5.2))
        for outcome, outcome_chunk in chunk.groupby("outcome"):
            outcome_chunk = outcome_chunk.sort_values("distance_mid")
            label = "Customers" if outcome == "lcus" else "Spending"
            ax.plot(outcome_chunk["distance_mid"], outcome_chunk["estimate"], marker="o", linewidth=1.8, label=label)
            ax.vlines(outcome_chunk["distance_mid"], outcome_chunk["ci_low95"], outcome_chunk["ci_hi95"], alpha=0.8)
        ax.axhline(0, color="grey", linestyle="--")
        ax.set_xlabel("Distance bin midpoint (m)")
        ax.set_ylabel("Estimate")
        ax.legend(frameon=False)
        path = FIGURE_DIR / f"02_distance_{period}_{sample}.pdf"
        ensure_parent(path)
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
    return final


def run_charger_models(panels: dict[tuple[str, str], pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for (period, sample), df in panels.items():
        regressors = [col for col in ["port_treat_level2", "port_treat_dc"] if col in df.columns]
        if len(regressors) != 2:
            continue
        for outcome in ["lcus", "lspend"]:
            result = run_absorbing_ls(df, outcome, regressors, ["placekey", "county_date"], "placekey")
            rows.append(append_metadata(result, outcome=outcome, period=period, sample=sample))
    final = pd.concat(rows, ignore_index=True)
    final.to_csv(TABLE_DIR / "03_charger_type_effects.csv", index=False)
    return final


def run_income_models(panels: dict[tuple[str, str], pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for (period, sample), df in panels.items():
        for outcome_col, label in INCOME_BRACKETS:
            if outcome_col not in df.columns or df[outcome_col].notna().sum() == 0:
                continue
            work = df.copy()
            work[f"log_{outcome_col}"] = np.log1p(work[outcome_col].clip(lower=0))
            result = run_absorbing_ls(work, f"log_{outcome_col}", ["port_treat"], ["placekey", "county_date"], "placekey")
            rows.append(append_metadata(result, period=period, sample=sample, income_group=label, outcome=outcome_col))
    final = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    final.to_csv(TABLE_DIR / "04_income_heterogeneity.csv", index=False)

    if not final.empty:
        for (period, sample), chunk in final.groupby(["period", "sample"]):
            order = {label: idx for idx, (_, label) in enumerate(INCOME_BRACKETS)}
            chunk = chunk.sort_values("income_group", key=lambda s: s.map(order))
            fig, ax = plt.subplots(figsize=(8.5, 5.2))
            ax.errorbar(chunk["income_group"], chunk["estimate"], yerr=1.96 * chunk["std.error"], fmt="o", capsize=4)
            ax.axhline(0, color="grey", linestyle="--")
            ax.set_ylabel("Estimate")
            ax.tick_params(axis="x", rotation=30)
            path = FIGURE_DIR / f"04_income_{period}_{sample}.pdf"
            ensure_parent(path)
            fig.tight_layout()
            fig.savefig(path)
            plt.close(fig)
    return final


def run_other_outcomes(panels: dict[tuple[str, str], pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for (period, sample), df in panels.items():
        for outcome, label in OTHER_OUTCOMES.items():
            if outcome not in df.columns or df[outcome].notna().sum() == 0:
                continue
            work = df[df[outcome].notna()].copy()
            work[f"log_{outcome}"] = np.log1p(work[outcome].clip(lower=0))
            result = run_absorbing_ls(work, f"log_{outcome}", ["port_treat"], ["placekey", "county_date"], "placekey")
            rows.append(append_metadata(result, period=period, sample=sample, outcome=outcome, outcome_label=label))
    final = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    final.to_csv(TABLE_DIR / "05_other_outcomes.csv", index=False)
    return final


def run_monetary_impacts(panels: dict[tuple[str, str], pd.DataFrame], main_results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (period, sample), df in panels.items():
        spending_beta = main_results.query("period == @period and sample == @sample and outcome == 'lspend'")["estimate"]
        customer_beta = main_results.query("period == @period and sample == @sample and outcome == 'lcus'")["estimate"]
        if spending_beta.empty or customer_beta.empty:
            continue
        treated = df[df["is_treated"] == 1].copy()
        if treated.empty:
            continue
        avg_ports = treated.groupby("placekey")["total_ports"].first().mean() if "total_ports" in treated.columns else np.nan
        avg_spending = treated.groupby("placekey")["lspend"].apply(lambda s: np.expm1(s).mean()).mean()
        avg_customers = treated.groupby("placekey")["lcus"].apply(lambda s: np.expm1(s).mean()).mean()
        p = POI_MULTIPLIERS.get(period, np.nan)
        beta_s = float(spending_beta.iloc[0])
        beta_c = float(customer_beta.iloc[0])
        m = beta_s * avg_ports * avg_spending * 12 if pd.notna(avg_ports) else np.nan
        f = beta_c * avg_ports * avg_customers * 12 if pd.notna(avg_ports) else np.nan
        rows.append(
            {
                "period": period,
                "sample": sample,
                "beta_spend": beta_s,
                "beta_customers": beta_c,
                "avg_ports_per_evcs": avg_ports,
                "avg_pretx_spending": avg_spending,
                "avg_pretx_customers": avg_customers,
                "avg_pois_in_500m": p,
                "M_per_poi_annual": m,
                "F_per_poi_annual": f,
                "M_all_annual": m * p if pd.notna(m) and pd.notna(p) else np.nan,
            }
        )
    final = pd.DataFrame(rows)
    final.to_csv(TABLE_DIR / "06_monetary_impacts.csv", index=False)
    return final


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    panels = load_panels()
    main_results = run_main_models(panels)
    run_distance_models(panels)
    run_charger_models(panels)
    run_income_models(panels)
    run_other_outcomes(panels)
    run_monetary_impacts(panels, main_results)


if __name__ == "__main__":
    main()
