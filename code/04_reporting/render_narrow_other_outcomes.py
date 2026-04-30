"""
Render auxiliary narrow-replication outcome effects for the manuscript.

The figure uses output/tables/narrow/05_other_outcomes.csv as the source of
truth and writes a compact coefficient plot to output/figures/narrow/.
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "output" / "tables" / "narrow" / "05_other_outcomes.csv"
OUTPUT = ROOT / "output" / "figures" / "narrow" / "05_other_outcomes.pdf"

PERIOD_LABELS = {
    "Period1_2019": "2019",
    "Period2_2021_2023": "2021-2023",
}

OUTCOME_ORDER = {
    "avg_customer_income": 0,
    "median_dist_home": 1,
    "median_dwell": 2,
}


def load_results() -> pd.DataFrame:
    if not INPUT.exists():
        raise SystemExit(f"Missing auxiliary outcome table: {INPUT}")
    df = pd.read_csv(INPUT)
    if df.empty:
        raise SystemExit(f"Auxiliary outcome table is empty: {INPUT}")

    df = df.copy()
    df["period_label"] = df["period"].map(PERIOD_LABELS).fillna(df["period"])
    df["plot_label"] = (
        df["outcome_label"]
        + " | "
        + df["period_label"]
        + " | "
        + df["sample"]
        + " (N="
        + df["nobs"].map(lambda value: f"{int(value):,}")
        + ")"
    )
    df["outcome_order"] = df["outcome"].map(OUTCOME_ORDER).fillna(99)
    return df.sort_values(["outcome_order", "period", "sample"], ascending=[True, True, True])


def render(df: pd.DataFrame) -> None:
    fig_height = max(4.2, 0.42 * len(df) + 1.2)
    fig, ax = plt.subplots(figsize=(9.5, fig_height))

    y = list(range(len(df)))
    x = df["estimate"]
    xerr = [
        x - df["ci_low95"],
        df["ci_hi95"] - x,
    ]
    colors = df["sample"].map({"All": "#2563eb", "Disadvantaged": "#b45309"}).fillna("#374151")

    for idx, (_, row) in enumerate(df.iterrows()):
        ax.errorbar(
            row["estimate"],
            idx,
            xerr=[[row["estimate"] - row["ci_low95"]], [row["ci_hi95"] - row["estimate"]]],
            fmt="o",
            color=colors.iloc[idx],
            ecolor=colors.iloc[idx],
            elinewidth=1.5,
            capsize=3,
            markersize=5,
        )

    ax.axvline(0, color="#4b5563", linewidth=1.0)
    ax.grid(axis="x", color="#e5e7eb", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_yticks(y)
    ax.set_yticklabels(df["plot_label"])
    ax.invert_yaxis()
    ax.set_xlabel("Estimated log-point effect of one additional active port")
    ax.set_ylabel("")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)

    fig.text(
        0.5,
        0.01,
        "Points show estimates from narrow TWFE-style models; bars show 95% confidence intervals.",
        ha="center",
        fontsize=9,
        color="#374151",
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(OUTPUT, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    render(load_results())


if __name__ == "__main__":
    main()
