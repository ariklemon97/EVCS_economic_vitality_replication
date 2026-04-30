"""
Estimate and plot broad stacked own-port event-study coefficients.

This uses the same stack construction as the preferred broad stacked regression:
6-month pre-window, 12-month post-window, not-yet-treated controls, stack-
specific POI and county-month fixed effects, POI-clustered standard errors, and
a 10,000-control cap. The event-study coefficients are binary own-port exposure
effects relative to event month -1.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update(
    {
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
    }
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SCRIPT_DIR = PROJECT_ROOT / "code" / "03_broad_replication"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_stacked_regression as stacked_models
from code.analysis_config import OUTPUT_DIR
from code.estimation_utils import run_absorbing_ls


REFERENCE_EVENT_TIME = -1

TABLE_DIR = OUTPUT_DIR / "tables" / "broad"
FIGURE_DIR = OUTPUT_DIR / "figures" / "broad"

OUTCOME_LABELS = {
    "lcus": "Customers",
    "lspend": "Spending",
}


def term_name(event_time: int) -> str:
    if event_time < 0:
        return f"event_m{abs(event_time)}"
    return f"event_p{event_time}"


def add_event_terms(stacked: pd.DataFrame, pre: int, post: int) -> tuple[pd.DataFrame, list[str], dict[str, int]]:
    out = stacked.copy()
    terms = []
    term_to_event_time = {}
    for event_time in range(-pre, post + 1):
        if event_time == REFERENCE_EVENT_TIME:
            continue
        term = term_name(event_time)
        out[term] = (out["stack_treated"].eq(1) & out["event_time"].eq(event_time)).astype(np.int8)
        if out[term].sum() > 0:
            terms.append(term)
            term_to_event_time[term] = event_time
    return out, terms, term_to_event_time


def output_stem(pre: int, post: int) -> str:
    return f"stacked_broad_event_study_pre{pre}_post{post}"


def legacy_output(pre: int, post: int) -> bool:
    return pre == 6 and post == 12


def estimate_event_study(
    pre: int,
    post: int,
    min_treated: int,
    max_control_pois: int | None,
    seed: int,
) -> pd.DataFrame:
    df = stacked_models.load_final_broad()
    panel = stacked_models.prepare_broad_own_panel(df)
    stacked, diagnostics = stacked_models.build_stacked_panel(
        panel,
        pre,
        post,
        min_treated,
        max_control_pois,
        seed,
    )
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    stem = output_stem(pre, post)
    diagnostics.to_csv(TABLE_DIR / f"{stem}_diagnostics.csv", index=False)
    if legacy_output(pre, post):
        diagnostics.to_csv(TABLE_DIR / "stacked_broad_event_study_diagnostics.csv", index=False)

    stacked, terms, term_to_event_time = add_event_terms(stacked, pre, post)
    rows = []
    for outcome in ["lcus", "lspend"]:
        result = run_absorbing_ls(
            stacked,
            outcome,
            terms,
            ["stack_unit", "stack_county_date"],
            "placekey",
        )
        result["outcome"] = outcome
        result["event_time"] = result["term"].map(term_to_event_time)
        result["reference_event_time"] = REFERENCE_EVENT_TIME
        result["pre_window"] = pre
        result["post_window"] = post
        result["max_control_pois"] = max_control_pois
        result["n_stacks"] = stacked["stack_id"].nunique()
        result["n_original_pois"] = stacked["placekey"].nunique()
        result["n_stacked_rows"] = len(stacked)
        rows.append(result)

    out = pd.concat(rows, ignore_index=True)
    baseline_rows = []
    for outcome in ["lcus", "lspend"]:
        baseline_rows.append(
            {
                "term": "reference_m1",
                "estimate": 0.0,
                "std.error": np.nan,
                "t_stat": np.nan,
                "p.value": np.nan,
                "ci_low95": 0.0,
                "ci_hi95": 0.0,
                "nobs": np.nan,
                "outcome": outcome,
                "event_time": REFERENCE_EVENT_TIME,
                "reference_event_time": REFERENCE_EVENT_TIME,
                "pre_window": pre,
                "post_window": post,
                "max_control_pois": max_control_pois,
                "n_stacks": out["n_stacks"].max(),
                "n_original_pois": out["n_original_pois"].max(),
                "n_stacked_rows": out["n_stacked_rows"].max(),
            }
        )
    out = pd.concat([out, pd.DataFrame(baseline_rows)], ignore_index=True)
    out["pct_effect"] = np.expm1(out["estimate"]) * 100
    out["pct_ci_low95"] = np.expm1(out["ci_low95"]) * 100
    out["pct_ci_hi95"] = np.expm1(out["ci_hi95"]) * 100
    out = out.sort_values(["outcome", "event_time"])
    out.to_csv(TABLE_DIR / f"{stem}.csv", index=False)
    if legacy_output(pre, post):
        out.to_csv(TABLE_DIR / "stacked_broad_event_study.csv", index=False)
    return out


def style_event_axis(ax: plt.Axes) -> None:
    ax.axhline(0, color="grey", linestyle="--", linewidth=1)
    ax.axvline(-0.5, color="red", linestyle="--", linewidth=1)


def plot_outcome(dynamic: pd.DataFrame, outcome: str, pre: int, post: int) -> None:
    sub = dynamic[dynamic["outcome"] == outcome].sort_values("event_time")
    fig, ax = plt.subplots(figsize=(8, 5.2))
    ax.plot(sub["event_time"], sub["pct_effect"], marker="o", linewidth=1.8)
    ax.vlines(
        sub["event_time"],
        sub["pct_ci_low95"],
        sub["pct_ci_hi95"],
        alpha=0.8,
        linewidth=1.4,
    )
    style_event_axis(ax)
    ax.set_xlabel("Months relative to first EVCS exposure")
    ax.set_ylabel("Effect on outcome (%)")
    fig.tight_layout()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"broad_stacked_event_study_pre{pre}_post{post}_{outcome}"
    fig.savefig(FIGURE_DIR / f"{stem}.pdf", bbox_inches="tight")
    if legacy_output(pre, post):
        fig.savefig(FIGURE_DIR / f"broad_stacked_event_study_{outcome}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_combined(dynamic: pd.DataFrame, pre: int, post: int) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2), sharey=True)
    for ax, outcome in zip(axes, ["lcus", "lspend"], strict=True):
        sub = dynamic[dynamic["outcome"] == outcome].sort_values("event_time")
        ax.plot(sub["event_time"], sub["pct_effect"], marker="o", linewidth=1.8)
        ax.vlines(
            sub["event_time"],
            sub["pct_ci_low95"],
            sub["pct_ci_hi95"],
            alpha=0.8,
            linewidth=1.4,
        )
        style_event_axis(ax)
        ax.set_title(OUTCOME_LABELS[outcome])
        ax.set_xlabel("Months relative to first EVCS exposure")
    axes[0].set_ylabel("Effect on outcome (%)")
    fig.tight_layout()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_DIR / f"broad_stacked_event_study_pre{pre}_post{post}.pdf", bbox_inches="tight")
    if legacy_output(pre, post):
        fig.savefig(FIGURE_DIR / "broad_stacked_event_study.pdf", bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Read stacked_broad_event_study.csv and regenerate only the PDFs.",
    )
    parser.add_argument("--pre", type=int, default=6, help="Months before cohort treatment to include.")
    parser.add_argument("--post", type=int, default=12, help="Months after cohort treatment to include.")
    parser.add_argument("--min-treated", type=int, default=25, help="Minimum treated POIs required per stack.")
    parser.add_argument(
        "--max-control-pois",
        type=int,
        default=10000,
        help="Optional cap on control POIs sampled per stack.",
    )
    parser.add_argument("--seed", type=int, default=20260425, help="Random seed for capped control sampling.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv_path = TABLE_DIR / f"{output_stem(args.pre, args.post)}.csv"
    if legacy_output(args.pre, args.post) and not csv_path.exists():
        csv_path = TABLE_DIR / "stacked_broad_event_study.csv"
    if args.plot_only:
        if not csv_path.exists():
            raise SystemExit(f"Missing {csv_path}; rerun without --plot-only first.")
        dynamic = pd.read_csv(csv_path)
    else:
        dynamic = estimate_event_study(
            args.pre,
            args.post,
            args.min_treated,
            args.max_control_pois,
            args.seed,
        )
    plot_combined(dynamic, args.pre, args.post)
    for outcome in ["lcus", "lspend"]:
        plot_outcome(dynamic, outcome, args.pre, args.post)


if __name__ == "__main__":
    main()
