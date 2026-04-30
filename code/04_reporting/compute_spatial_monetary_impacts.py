"""
Compute monetary interpretations for corrected stacked spatial estimates.

The source coefficient table is output/tables/main/spatial_competition_stacked_results.csv.
The exposure and baseline spending source is data/processed/df_final_broad.csv.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from code.analysis_config import OUTPUT_DIR, PROCESSED_DIR


RESULTS_PATH = OUTPUT_DIR / "tables" / "main" / "spatial_competition_stacked_results.csv"
PANEL_PATH = PROCESSED_DIR / "df_final_broad.csv"
OUT_PATH = OUTPUT_DIR / "tables" / "main" / "spatial_stacked_monetary_impacts.csv"

EFFECTS = [
    (
        "All competitor ports",
        "competitor_port_treat",
        "competitor_ports_active",
        "Not statistically significant",
    ),
    (
        "DC fast competitor ports",
        "competitor_port_treat_dc",
        "competitor_dc_fast_ports_active",
        "Preferred monetary interpretation",
    ),
    (
        "Level 2 competitor ports",
        "competitor_port_treat_level2",
        "competitor_level2_ports_active",
        "Not statistically significant",
    ),
]


def first_pre_month(panel: pd.DataFrame) -> int:
    treated_months = pd.to_numeric(
        panel.loc[panel["competitor_first_treat_period"] > 0, "competitor_first_treat_period"],
        errors="coerce",
    )
    if treated_months.empty:
        raise SystemExit("No spatially treated competitor POIs in df_final_broad.csv.")
    return int(treated_months.min())


def baseline_monthly_spending(panel: pd.DataFrame, treated_placekeys: pd.Index) -> float:
    pre_month = first_pre_month(panel)
    baseline = panel[
        panel["placekey"].isin(treated_placekeys)
        & (panel["date_numeric"] < pre_month)
    ].copy()
    if baseline.empty:
        baseline = panel[panel["placekey"].isin(treated_placekeys)].copy()
    return float(np.expm1(pd.to_numeric(baseline["lspend"], errors="coerce")).mean())


def average_active_ports(panel: pd.DataFrame, treated_placekeys: pd.Index, exposure_col: str) -> float:
    work = panel[panel["placekey"].isin(treated_placekeys)].copy()
    work[exposure_col] = pd.to_numeric(work[exposure_col], errors="coerce").fillna(0.0)
    active = work[work[exposure_col] > 0]
    if active.empty:
        return 0.0
    return float(active.groupby("placekey")[exposure_col].mean().mean())


def main() -> None:
    if not RESULTS_PATH.exists():
        raise SystemExit(f"Missing {RESULTS_PATH}")
    if not PANEL_PATH.exists():
        raise SystemExit(f"Missing {PANEL_PATH}")

    results = pd.read_csv(RESULTS_PATH)
    panel_cols = [
        "placekey",
        "date_numeric",
        "lspend",
        "Treatment_Competitor",
        "competitor_first_treat_period",
        "competitor_ports_active",
        "competitor_level2_ports_active",
        "competitor_dc_fast_ports_active",
    ]
    panel = pd.read_csv(PANEL_PATH, usecols=panel_cols, low_memory=False)
    panel["Treatment_Competitor"] = pd.to_numeric(
        panel["Treatment_Competitor"], errors="coerce"
    ).fillna(0).astype(int)
    treated_placekeys = pd.Index(panel.loc[panel["Treatment_Competitor"] == 1, "placekey"].unique())
    n_treated = len(treated_placekeys)
    avg_spend = baseline_monthly_spending(panel, treated_placekeys)

    rows = []
    for label, term, exposure_col, interpretation in EFFECTS:
        match = results[(results["term"] == term) & (results["outcome"] == "lspend")]
        beta = float(match["estimate"].iloc[0]) if not match.empty else np.nan
        avg_ports = average_active_ports(panel, treated_placekeys, exposure_col)
        monthly = beta * avg_ports * avg_spend if pd.notna(beta) else np.nan
        annual = monthly * 12 if pd.notna(monthly) else np.nan
        rows.append(
            {
                "effect": label,
                "beta_spend": beta,
                "avg_active_ports": avg_ports,
                "avg_baseline_monthly_spend": avg_spend,
                "n_treated_pois": n_treated,
                "monthly_spend_effect_per_poi": monthly,
                "annual_spend_effect_per_poi": annual,
                "total_annual_effect_all_treated_pois": annual * n_treated
                if pd.notna(annual)
                else np.nan,
                "interpretation": interpretation,
            }
        )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT_PATH, index=False)
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
