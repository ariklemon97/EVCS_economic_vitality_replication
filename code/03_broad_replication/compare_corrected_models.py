"""
Collect corrected intensity and CS model outputs into side-by-side tables.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm


TABLE_DIR = Path("output/tables/broad")


def pct(x: pd.Series) -> pd.Series:
    return (np.exp(x) - 1) * 100


def load_twfe(path: Path, family: str) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    out = pd.DataFrame(
        {
            "family": family,
            "model": df["model"],
            "dataset": df["dataset"],
            "outcome": df["outcome"],
            "term": df["term"],
            "estimate": df["estimate"],
            "se": df["std.error"],
            "ci_low": df["ci_low95"],
            "ci_high": df["ci_hi95"],
            "p_value": df["p.value"],
            "nobs": df["nobs"],
        }
    )
    return out


def load_cs_main(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    out = pd.DataFrame(
        {
            "family": "CS timing",
            "model": "own_binary_timing",
            "dataset": df["dataset"],
            "outcome": df["outcome"],
            "term": "ATT",
            "estimate": df["ATT"],
            "se": df["SE"],
            "ci_low": df["CI_lower"],
            "ci_high": df["CI_upper"],
        }
    )
    out["p_value"] = 2 * norm.sf(np.abs(out["estimate"] / out["se"]))
    out["nobs"] = np.nan
    return out


def load_cs_spatial(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df = df[df["agg_type"] == "simple"].copy()
    out = pd.DataFrame(
        {
            "family": "CS timing",
            "model": "competitor_binary_timing",
            "dataset": df["dataset"],
            "outcome": df["outcome"],
            "term": "ATT",
            "estimate": df["ATT"],
            "se": df["SE"],
            "ci_low": df["CI_lower"],
            "ci_high": df["CI_upper"],
        }
    )
    out["p_value"] = 2 * norm.sf(np.abs(out["estimate"] / out["se"]))
    out["nobs"] = np.nan
    return out


def main() -> None:
    pieces = [
        load_twfe(TABLE_DIR / "broad_intensity_results.csv", "TWFE intensity"),
        load_twfe(TABLE_DIR / "spatial_competition_intensity_results.csv", "TWFE intensity"),
        load_cs_main(TABLE_DIR / "corrected_cs_main_summary.csv"),
        load_cs_spatial(TABLE_DIR / "corrected_cs_spatial_results.csv"),
    ]
    combined = pd.concat([p for p in pieces if not p.empty], ignore_index=True)
    combined["pct_effect"] = pct(combined["estimate"])
    combined["pct_ci_low"] = pct(combined["ci_low"])
    combined["pct_ci_high"] = pct(combined["ci_high"])
    combined.to_csv(TABLE_DIR / "corrected_model_side_by_side.csv", index=False)

    summary = combined[
        [
            "family",
            "model",
            "dataset",
            "outcome",
            "term",
            "estimate",
            "se",
            "p_value",
            "pct_effect",
            "pct_ci_low",
            "pct_ci_high",
            "nobs",
        ]
    ].copy()
    summary.to_csv(TABLE_DIR / "corrected_model_side_by_side_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
