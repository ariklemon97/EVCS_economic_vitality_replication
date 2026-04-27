"""
Robustness checks for final stacked regressions.

This runner is intentionally separate from the main replication entrypoint.
It implements robustness checks for:

1. stacked event-window sensitivity,
2. control-cap sensitivity,
3. pretrend/placebo tests,
4. spatial competition radius sensitivity above the 500m direct-treatment
   radius, and
5. POI-type stacked spatial heterogeneity.

The script writes to output/tables/robustness/ and does not overwrite main
paper-facing stacked results.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy.stats import norm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_stacked_regression as stacked_models
from code.analysis_config import OUTPUT_DIR, PROCESSED_DIR
from code.estimation_utils import ensure_parent, run_absorbing_ls


ROBUST_DIR = OUTPUT_DIR / "tables" / "robustness"

DEFAULT_WINDOWS = [(6, 6), (6, 12), (12, 12), (12, 24)]
DEFAULT_CONTROL_CAPS = [5000, 10000, 20000, None]
DEFAULT_SPATIAL_RADII = [1000, 1500, 2000]


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


def robustness_args(pre: int, post: int, cap: int | None, min_treated: int, seed: int) -> SimpleNamespace:
    return SimpleNamespace(
        pre=pre,
        post=post,
        max_control_pois=cap,
        min_treated=min_treated,
        seed=seed,
    )


def run_stacked_spec(
    panel: pd.DataFrame,
    target: str,
    pre: int,
    post: int,
    cap: int | None,
    min_treated: int,
    seed: int,
    spec_label: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    stacked, diagnostics = stacked_models.build_stacked_panel(panel, pre, post, min_treated, cap, seed)
    diagnostics["target"] = target
    diagnostics["spec"] = spec_label
    diagnostics["pre_window"] = pre
    diagnostics["post_window"] = post
    diagnostics["max_control_pois"] = cap

    if target == "broad":
        dataset = "Robust_Stacked_Broad"
        specs = [
            ("stacked_own_port_intensity", ["port_treat"]),
            ("stacked_own_port_intensity_by_charger", ["port_treat_level2", "port_treat_dc"]),
        ]
    elif target == "spatial":
        dataset = "Robust_Stacked_Spatial"
        specs = [
            ("stacked_competitor_port_intensity", ["competitor_port_treat"]),
            (
                "stacked_competitor_port_intensity_by_charger",
                ["competitor_port_treat_level2", "competitor_port_treat_dc"],
            ),
        ]
    else:
        raise ValueError(f"Unsupported target: {target}")

    rows = []
    for outcome in ["lcus", "lspend"]:
        for model, regressors in specs:
            result = stacked_models.run_stacked_fe(stacked, outcome, regressors, dataset, model, pre, post)
            result["target"] = target
            result["spec"] = spec_label
            result["max_control_pois"] = cap
            rows.append(result)
    return pd.concat(rows, ignore_index=True), diagnostics


def run_window_sensitivity(df: pd.DataFrame, min_treated: int, cap: int | None, seed: int) -> None:
    results = []
    diagnostics = []
    for target, panel_builder in [
        ("broad", stacked_models.prepare_broad_own_panel),
        ("spatial", stacked_models.prepare_spatial_competition_panel),
    ]:
        panel = panel_builder(df)
        for pre, post in DEFAULT_WINDOWS:
            label = f"window_pre{pre}_post{post}"
            result, diag = run_stacked_spec(panel, target, pre, post, cap, min_treated, seed, label)
            results.append(result)
            diagnostics.append(diag)

    pd.concat(results, ignore_index=True).to_csv(ROBUST_DIR / "stacked_window_sensitivity.csv", index=False)
    pd.concat(diagnostics, ignore_index=True).to_csv(
        ROBUST_DIR / "stacked_window_sensitivity_diagnostics.csv",
        index=False,
    )


def run_control_cap_sensitivity(df: pd.DataFrame, pre: int, post: int, min_treated: int, seed: int) -> None:
    results = []
    diagnostics = []
    for target, panel_builder in [
        ("broad", stacked_models.prepare_broad_own_panel),
        ("spatial", stacked_models.prepare_spatial_competition_panel),
    ]:
        panel = panel_builder(df)
        for cap in DEFAULT_CONTROL_CAPS:
            cap_label = "none" if cap is None else str(cap)
            label = f"control_cap_{cap_label}"
            result, diag = run_stacked_spec(panel, target, pre, post, cap, min_treated, seed, label)
            results.append(result)
            diagnostics.append(diag)

    pd.concat(results, ignore_index=True).to_csv(ROBUST_DIR / "stacked_control_cap_sensitivity.csv", index=False)
    pd.concat(diagnostics, ignore_index=True).to_csv(
        ROBUST_DIR / "stacked_control_cap_sensitivity_diagnostics.csv",
        index=False,
    )


def add_pretrend_inference(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["p.value"] = 2 * norm.sf(np.abs(out["estimate"] / out["std.error"]))
    out["ci_low95"] = out["estimate"] - 1.96 * out["std.error"]
    out["ci_hi95"] = out["estimate"] + 1.96 * out["std.error"]
    return out


def run_pretrend_regressions(
    panel: pd.DataFrame,
    target: str,
    pre: int,
    post: int,
    cap: int | None,
    min_treated: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    stack, diagnostics = stacked_models.build_stacked_panel(panel, pre, post, min_treated, cap, seed)
    pre_stack = stack[stack["event_time"].between(-pre, -1)].copy()
    if pre_stack.empty:
        raise SystemExit(f"No pre-period stacked rows available for {target}.")

    lead_terms = []
    for event_time in range(-pre, -1):
        term = f"treated_lead_m{abs(event_time)}"
        pre_stack[term] = (
            pre_stack["stack_treated"].eq(1) & pre_stack["event_time"].eq(event_time)
        ).astype(int)
        if pre_stack[term].sum() > 0:
            lead_terms.append(term)

    placebo_term = "placebo_treated_last3_pre_months"
    pre_stack[placebo_term] = (
        pre_stack["stack_treated"].eq(1) & pre_stack["event_time"].between(-3, -1)
    ).astype(int)

    rows = []
    for outcome in ["lcus", "lspend"]:
        if lead_terms:
            leads = run_absorbing_ls(
                pre_stack,
                outcome,
                lead_terms,
                ["stack_unit", "stack_county_date"],
                "placekey",
            )
            leads["test"] = "event_time_pretrend"
            leads["outcome"] = outcome
            rows.append(add_pretrend_inference(leads))

        placebo = run_absorbing_ls(
            pre_stack,
            outcome,
            [placebo_term],
            ["stack_unit", "stack_county_date"],
            "placekey",
        )
        placebo["test"] = "placebo_last3_pre_months"
        placebo["outcome"] = outcome
        rows.append(add_pretrend_inference(placebo))

    out = pd.concat(rows, ignore_index=True)
    out["target"] = target
    out["pre_window"] = pre
    out["post_window"] = post
    out["max_control_pois"] = cap
    diagnostics["target"] = target
    diagnostics["test"] = "pretrend_placebo"
    return out, diagnostics


def run_pretrend_placebo(df: pd.DataFrame, pre: int, post: int, cap: int | None, min_treated: int, seed: int) -> None:
    results = []
    diagnostics = []
    for target, panel_builder in [
        ("broad", stacked_models.prepare_broad_own_panel),
        ("spatial", stacked_models.prepare_spatial_competition_panel),
    ]:
        panel = panel_builder(df)
        result, diag = run_pretrend_regressions(panel, target, pre, post, cap, min_treated, seed)
        results.append(result)
        diagnostics.append(diag)

    pd.concat(results, ignore_index=True).to_csv(ROBUST_DIR / "stacked_pretrend_placebo.csv", index=False)
    pd.concat(diagnostics, ignore_index=True).to_csv(
        ROBUST_DIR / "stacked_pretrend_placebo_diagnostics.csv",
        index=False,
    )


def period_map_from_panel(df: pd.DataFrame) -> tuple[dict[int, int], list[int]]:
    period_map = (
        df[["date_numeric_orig", "date_numeric"]]
        .drop_duplicates()
        .sort_values("date_numeric_orig")
        .set_index("date_numeric_orig")["date_numeric"]
        .astype(int)
        .to_dict()
    )
    return period_map, sorted(period_map)


def map_period(yyyymm: int, period_map: dict[int, int], ordered_periods: list[int]) -> int:
    if yyyymm == 0:
        return 0
    if yyyymm in period_map:
        return int(period_map[yyyymm])
    candidates = [period for period in ordered_periods if period >= yyyymm]
    return int(period_map[candidates[0]]) if candidates else 0


def pair_path_for_radius(radius_m: int) -> Path:
    if radius_m == 1000:
        return PROCESSED_DIR / "poi_competitor_matches.parquet"
    return PROCESSED_DIR / f"poi_competitor_matches_r{radius_m}.parquet"


def build_spatial_panel_from_pair_file(base_df: pd.DataFrame, radius_m: int) -> pd.DataFrame:
    if radius_m <= 500:
        raise SystemExit("Spatial radius robustness must use radii greater than 500m.")

    pair_path = pair_path_for_radius(radius_m)
    if not pair_path.exists():
        raise SystemExit(
            f"Missing {pair_path}. Build it first with: "
            f"python code/03_broad_replication/04_spatial_competition.py "
            f"--radius-m {radius_m} --output-suffix r{radius_m}"
        )

    pairs = pd.read_parquet(pair_path)
    if pairs.empty:
        raise SystemExit(f"{pair_path} contains no competitor matches.")

    required = {
        "placekey",
        "competitor_open_yyyymm",
        "competitor_total_ports",
        "competitor_level2_ports",
        "competitor_dc_fast_ports",
    }
    missing = required - set(pairs.columns)
    if missing:
        raise SystemExit(f"{pair_path} is missing columns: {sorted(missing)}")

    for col in ["competitor_open_yyyymm", "competitor_total_ports", "competitor_level2_ports", "competitor_dc_fast_ports"]:
        pairs[col] = pd.to_numeric(pairs[col], errors="coerce").fillna(0)
    if "competitor_commercial_adjacent" not in pairs.columns:
        pairs["competitor_commercial_adjacent"] = 1
    pairs["competitor_commercial_adjacent"] = pd.to_numeric(
        pairs["competitor_commercial_adjacent"], errors="coerce"
    ).fillna(0).astype(int)
    pairs = pairs[(pairs["competitor_commercial_adjacent"] == 1) & (pairs["competitor_open_yyyymm"] > 0)].copy()
    if pairs.empty:
        raise SystemExit(f"{pair_path} has no commercial-adjacent competitor matches after filtering.")

    period_map, ordered_periods = period_map_from_panel(base_df)
    static = (
        pairs.groupby("placekey")
        .agg(
            competitor_open_yyyymm=("competitor_open_yyyymm", "min"),
            competitor_total_ports=("competitor_total_ports", "sum"),
            competitor_level2_ports=("competitor_level2_ports", "sum"),
            competitor_dc_fast_ports=("competitor_dc_fast_ports", "sum"),
            competitor_commercial_adjacent=("competitor_commercial_adjacent", "max"),
        )
        .reset_index()
    )
    static["Treatment_Competitor"] = 1
    static["competitor_first_treat_period"] = static["competitor_open_yyyymm"].apply(
        lambda value: map_period(int(value), period_map, ordered_periods)
    )

    active_frames = []
    exposure_cols = ["competitor_total_ports", "competitor_level2_ports", "competitor_dc_fast_ports"]
    for period in ordered_periods:
        active = pairs[pairs["competitor_open_yyyymm"] <= period]
        if active.empty:
            continue
        month = active.groupby("placekey")[exposure_cols].sum().reset_index()
        month["date_numeric_orig"] = period
        active_frames.append(month)
    active_panel = pd.concat(active_frames, ignore_index=True) if active_frames else pd.DataFrame()
    active_panel = active_panel.rename(
        columns={
            "competitor_total_ports": "competitor_ports_active",
            "competitor_level2_ports": "competitor_level2_ports_active",
            "competitor_dc_fast_ports": "competitor_dc_fast_ports_active",
        }
    )

    drop_cols = [
        "Treatment_Competitor",
        "competitor_open_yyyymm",
        "competitor_total_ports",
        "competitor_first_treat_period",
        "competitor_commercial_adjacent",
        "competitor_ports_active",
        "competitor_level2_ports_active",
        "competitor_dc_fast_ports_active",
    ]
    out = base_df.drop(columns=[col for col in drop_cols if col in base_df.columns], errors="ignore").copy()
    out = out.merge(static, on="placekey", how="left")
    out = out.merge(active_panel, on=["placekey", "date_numeric_orig"], how="left")

    for col in [
        "Treatment_Competitor",
        "competitor_first_treat_period",
        "competitor_commercial_adjacent",
        "competitor_total_ports",
        "competitor_ports_active",
        "competitor_level2_ports_active",
        "competitor_dc_fast_ports_active",
    ]:
        if col not in out.columns:
            out[col] = 0
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0)
    out["Treatment_Competitor"] = out["Treatment_Competitor"].astype(int)
    out["competitor_first_treat_period"] = out["competitor_first_treat_period"].astype(int)
    out["competitor_commercial_adjacent"] = out["competitor_commercial_adjacent"].astype(int)
    return stacked_models.prepare_spatial_competition_panel(out)


def run_spatial_distance_sensitivity(
    df: pd.DataFrame,
    radii: list[int],
    pre: int,
    post: int,
    cap: int | None,
    min_treated: int,
    seed: int,
) -> None:
    invalid = [radius for radius in radii if radius <= 500]
    if invalid:
        raise SystemExit(f"Spatial distance radii must be greater than 500m. Invalid: {invalid}")

    results = []
    diagnostics = []
    for radius in radii:
        panel = build_spatial_panel_from_pair_file(df, radius)
        label = f"spatial_radius_{radius}m"
        result, diag = run_stacked_spec(panel, "spatial", pre, post, cap, min_treated, seed, label)
        result["competition_radius_m"] = radius
        diag["competition_radius_m"] = radius
        results.append(result)
        diagnostics.append(diag)

    pd.concat(results, ignore_index=True).to_csv(ROBUST_DIR / "stacked_spatial_distance_sensitivity.csv", index=False)
    pd.concat(diagnostics, ignore_index=True).to_csv(
        ROBUST_DIR / "stacked_spatial_distance_sensitivity_diagnostics.csv",
        index=False,
    )


def run_poi_type_spatial_heterogeneity(
    df: pd.DataFrame,
    pre: int,
    post: int,
    cap: int | None,
    min_treated: int,
    seed: int,
) -> None:
    if "naics_code" not in df.columns:
        raise SystemExit("POI-type stacked robustness requires naics_code in df_final_broad.csv.")
    panel = stacked_models.prepare_spatial_competition_panel(df)
    panel["poi_type"] = panel["naics_code"].map(map_poi_type)

    results = []
    diagnostics = []
    for poi_type in sorted(t for t in panel["poi_type"].dropna().unique() if t != "Other"):
        subset = panel[panel["poi_type"] == poi_type].copy()
        if subset.loc[subset["stack_treat_period"] > 0, "placekey"].nunique() < min_treated:
            continue
        label = f"poi_type_{poi_type}"
        result, diag = run_stacked_spec(subset, "spatial", pre, post, cap, min_treated, seed, label)
        result["poi_type"] = poi_type
        diag["poi_type"] = poi_type
        results.append(result)
        diagnostics.append(diag)

    if not results:
        raise SystemExit("No POI-type stacked spatial robustness models had enough treated POIs.")
    pd.concat(results, ignore_index=True).to_csv(ROBUST_DIR / "stacked_spatial_poi_type_heterogeneity.csv", index=False)
    pd.concat(diagnostics, ignore_index=True).to_csv(
        ROBUST_DIR / "stacked_spatial_poi_type_heterogeneity_diagnostics.csv",
        index=False,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check",
        choices=["all", "windows", "control-caps", "pretrend-placebo", "spatial-distance", "poi-type"],
        default="all",
    )
    parser.add_argument("--pre", type=int, default=6)
    parser.add_argument("--post", type=int, default=12)
    parser.add_argument("--min-treated", type=int, default=25)
    parser.add_argument("--max-control-pois", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260425)
    parser.add_argument(
        "--spatial-radii-m",
        type=int,
        nargs="+",
        default=DEFAULT_SPATIAL_RADII,
        help="Spatial competition radii for robustness. Values must be greater than 500m.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ROBUST_DIR.mkdir(parents=True, exist_ok=True)
    df = stacked_models.load_final_broad()

    if args.check in {"all", "windows"}:
        run_window_sensitivity(df, args.min_treated, args.max_control_pois, args.seed)
    if args.check in {"all", "control-caps"}:
        run_control_cap_sensitivity(df, args.pre, args.post, args.min_treated, args.seed)
    if args.check in {"all", "pretrend-placebo"}:
        run_pretrend_placebo(df, args.pre, args.post, args.max_control_pois, args.min_treated, args.seed)
    if args.check in {"all", "spatial-distance"}:
        run_spatial_distance_sensitivity(
            df,
            args.spatial_radii_m,
            args.pre,
            args.post,
            args.max_control_pois,
            args.min_treated,
            args.seed,
        )
    if args.check in {"all", "poi-type"}:
        run_poi_type_spatial_heterogeneity(
            df,
            args.pre,
            args.post,
            args.max_control_pois,
            args.min_treated,
            args.seed,
        )


if __name__ == "__main__":
    main()
