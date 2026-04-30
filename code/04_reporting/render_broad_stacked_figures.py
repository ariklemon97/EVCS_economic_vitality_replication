"""
Render manuscript-ready figures for broad stacked own-port estimates.

The figures use output/tables/main/broad_stacked_own_port_results.csv as the
source of truth and write PDF outputs to output/figures/broad/.
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams.update(
    {
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
    }
)


ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "output" / "tables" / "main" / "broad_stacked_own_port_results.csv"
OUT_DIR = ROOT / "output" / "figures" / "broad"

TERM_LABELS = {
    "port_treat": "All ports",
    "port_treat_level2": "Level 2 ports",
    "port_treat_dc": "DC fast ports",
}

OUTCOME_LABELS = {
    "lcus": "Customers",
    "lspend": "Spending",
}

TERM_ORDER = ["port_treat", "port_treat_level2", "port_treat_dc"]


def load_results() -> pd.DataFrame:
    if not INPUT.exists():
        raise SystemExit(f"Missing broad stacked result table: {INPUT}")
    df = pd.read_csv(INPUT)
    df = df[df["term"].isin(TERM_ORDER)].copy()
    df["term_label"] = pd.Categorical(
        df["term"].map(TERM_LABELS),
        categories=[TERM_LABELS[t] for t in TERM_ORDER],
        ordered=True,
    )
    df["outcome_label"] = df["outcome"].map(OUTCOME_LABELS)
    return df.sort_values(["outcome", "term_label"])


def style_axis(ax: plt.Axes) -> None:
    ax.axvline(0, color="#4b5563", linewidth=1.0)
    ax.grid(axis="x", color="#e5e7eb", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)


def plot_combined(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2), sharex=True)
    colors = {"Customers": "#2563eb", "Spending": "#b45309"}

    for ax, outcome in zip(axes, ["Customers", "Spending"], strict=True):
        sub = df[df["outcome_label"] == outcome].sort_values("term_label", ascending=False)
        y = range(len(sub))
        x = sub["pct_effect"]
        xerr = [
            x - sub["pct_ci_low95"],
            sub["pct_ci_hi95"] - x,
        ]
        ax.errorbar(
            x,
            y,
            xerr=xerr,
            fmt="o",
            color=colors[outcome],
            ecolor=colors[outcome],
            elinewidth=1.8,
            capsize=4,
            markersize=6,
        )
        ax.set_yticks(list(y))
        ax.set_yticklabels(sub["term_label"])
        ax.set_xlabel("Effect on outcome (%)")
        style_axis(ax)

    fig.text(
        0.5,
        0.01,
        "Points show percent effects; bars show 95% confidence intervals.",
        ha="center",
        fontsize=9,
        color="#374151",
    )
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / "broad_stacked_own_port_coefficients.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_outcome(df: pd.DataFrame, outcome: str) -> None:
    sub = df[df["outcome"] == outcome].sort_values("term_label", ascending=False)
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    y = range(len(sub))
    x = sub["pct_effect"]
    xerr = [
        x - sub["pct_ci_low95"],
        sub["pct_ci_hi95"] - x,
    ]
    ax.errorbar(
        x,
        y,
        xerr=xerr,
        fmt="o",
        color="#1f2937",
        ecolor="#1f2937",
        elinewidth=1.8,
        capsize=4,
        markersize=6,
    )
    ax.set_yticks(list(y))
    ax.set_yticklabels(sub["term_label"])
    ax.set_xlabel("Effect on outcome (%)")
    style_axis(ax)
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"broad_stacked_own_port_{outcome}.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    df = load_results()
    plot_combined(df)
    for outcome in ["lcus", "lspend"]:
        plot_outcome(df, outcome)


if __name__ == "__main__":
    main()
