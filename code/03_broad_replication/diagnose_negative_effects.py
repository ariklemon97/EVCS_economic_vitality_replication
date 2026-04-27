"""
Summarize diagnostics for negative broad/spatial EVCS effects.

This script does not refit the main models. It reads existing event-study and
subgroup output tables and writes compact diagnostics that answer:

1. Do treated/spatial-exposed POIs show negative pre-treatment event-study
   coefficients before treatment?
2. Are negative estimates concentrated in specific POI types or charger types?
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

from code.analysis_config import OUTPUT_DIR


TABLE_DIR = OUTPUT_DIR / "tables" / "broad"
DIAG_DIR = OUTPUT_DIR / "tables" / "diagnostics"


def read_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def add_p_values(df: pd.DataFrame, estimate_col: str, se_col: str) -> pd.DataFrame:
    out = df.copy()
    out["t_stat"] = out[estimate_col] / out[se_col]
    out["p_value"] = 2 * norm.sf(np.abs(out["t_stat"]))
    return out


def summarize_pretrends() -> pd.DataFrame:
    sources = [
        ("spatial_all", TABLE_DIR / "spatial_competition_results.csv"),
        ("spatial_by_charger", TABLE_DIR / "spatial_competition_charger_type.csv"),
        ("spatial_robustness", TABLE_DIR / "spatial_competition_robustness_comparison.csv"),
    ]
    rows: list[dict[str, object]] = []

    for source_name, path in sources:
        df = read_if_exists(path)
        if df.empty or "event_time" not in df.columns:
            continue

        dynamic = df[df.get("agg_type").eq("dynamic")].copy()
        dynamic["event_time"] = pd.to_numeric(dynamic["event_time"], errors="coerce")
        dynamic["ATT"] = pd.to_numeric(dynamic["ATT"], errors="coerce")
        dynamic["SE"] = pd.to_numeric(dynamic["SE"], errors="coerce")
        dynamic = dynamic.dropna(subset=["event_time", "ATT", "SE"])
        dynamic = add_p_values(dynamic, "ATT", "SE")

        # Focus on the year before treatment; very long negative lags can be
        # sparse and less useful for a parallel-trends read.
        pre = dynamic[(dynamic["event_time"] >= -12) & (dynamic["event_time"] <= -1)].copy()
        if pre.empty:
            continue

        group_cols = [col for col in ["dataset", "outcome", "specification", "bin"] if col in pre.columns]
        for keys, group in pre.groupby(group_cols, dropna=False):
            if not isinstance(keys, tuple):
                keys = (keys,)
            row = dict(zip(group_cols, keys))
            sig = group["p_value"] < 0.05
            row.update(
                {
                    "source": source_name,
                    "n_pre_months": int(len(group)),
                    "mean_pre_ATT": group["ATT"].mean(),
                    "median_pre_ATT": group["ATT"].median(),
                    "share_negative_pre_ATT": (group["ATT"] < 0).mean(),
                    "n_sig_pre_5pct": int(sig.sum()),
                    "n_sig_negative_pre_5pct": int((sig & (group["ATT"] < 0)).sum()),
                    "n_sig_positive_pre_5pct": int((sig & (group["ATT"] > 0)).sum()),
                    "max_abs_pre_t": group["t_stat"].abs().max(),
                    "min_pre_p_value": group["p_value"].min(),
                }
            )
            rows.append(row)

    return pd.DataFrame(rows)


def collect_effect_concentration() -> pd.DataFrame:
    sources = [
        ("own_ports_all", TABLE_DIR / "broad_intensity_results.csv"),
        ("competitor_ports_all", TABLE_DIR / "spatial_competition_intensity_results.csv"),
        ("competitor_ports_by_poi_type", TABLE_DIR / "spatial_competition_intensity_poi_type.csv"),
        ("competitor_ports_by_income", TABLE_DIR / "spatial_competition_intensity_income_groups.csv"),
    ]
    rows: list[pd.DataFrame] = []

    for source_name, path in sources:
        df = read_if_exists(path)
        if df.empty:
            continue

        keep_cols = [
            "dataset",
            "model",
            "outcome",
            "term",
            "income_bucket",
            "estimate",
            "std.error",
            "p.value",
            "pct_effect",
            "pct_ci_low95",
            "pct_ci_hi95",
            "nobs",
        ]
        keep = [col for col in keep_cols if col in df.columns]
        work = df[keep].copy()
        work["source"] = source_name
        if "pct_effect" not in work.columns and "estimate" in work.columns:
            work["pct_effect"] = np.expm1(pd.to_numeric(work["estimate"], errors="coerce")) * 100
        if "p.value" not in work.columns and {"estimate", "std.error"}.issubset(work.columns):
            work = add_p_values(work, "estimate", "std.error").rename(columns={"p_value": "p.value"})
        rows.append(work)

    if not rows:
        return pd.DataFrame()

    out = pd.concat(rows, ignore_index=True)
    out["pct_effect"] = pd.to_numeric(out["pct_effect"], errors="coerce")
    out["p.value"] = pd.to_numeric(out["p.value"], errors="coerce")
    out["significant_5pct"] = out["p.value"] < 0.05
    out["negative_and_significant"] = (out["pct_effect"] < 0) & out["significant_5pct"]
    return out.sort_values(["outcome", "pct_effect"], na_position="last")


def write_markdown(pretrends: pd.DataFrame, concentration: pd.DataFrame) -> None:
    lines = [
        "# Negative-Effect Diagnostics",
        "",
        "## How to Read This",
        "",
        "- `pretrend_diagnostics.csv` checks event-study coefficients from months -12 to -1.",
        "- `effect_concentration.csv` shows which model/subgroup estimates are most negative.",
        "- A worrying pretrend pattern is many significant negative pre-treatment coefficients, not just one noisy lag.",
        "",
    ]

    if not pretrends.empty:
        flagged = pretrends[
            (pretrends["n_sig_negative_pre_5pct"] >= 2)
            | (pretrends["share_negative_pre_ATT"] >= 0.75)
        ].copy()
        lines.extend(
            [
                "## Pretrend Flags",
                "",
                f"- Rows checked: {len(pretrends)}",
                f"- Rows with at least two significant negative pre-period coefficients or at least 75% negative pre-period coefficients: {len(flagged)}",
                "",
            ]
        )
    else:
        lines.extend(["## Pretrend Flags", "", "- No event-study pretrend rows were available.", ""])

    if not concentration.empty:
        neg_sig = concentration[concentration["negative_and_significant"]].copy()
        top = neg_sig.nsmallest(10, "pct_effect")
        lines.extend(["## Most Negative Significant Effects", ""])
        if top.empty:
            lines.append("- No negative significant subgroup/intensity effects found.")
        else:
            for _, row in top.iterrows():
                label = row.get("dataset", "unknown")
                term = row.get("term", "unknown")
                outcome = row.get("outcome", "unknown")
                pct = row.get("pct_effect", np.nan)
                pval = row.get("p.value", np.nan)
                lines.append(f"- `{label}` / `{outcome}` / `{term}`: {pct:.4f}% per port, p={pval:.4g}")
        lines.append("")

    (DIAG_DIR / "negative_effect_diagnostics.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    DIAG_DIR.mkdir(parents=True, exist_ok=True)
    pretrends = summarize_pretrends()
    concentration = collect_effect_concentration()

    pretrends.to_csv(DIAG_DIR / "pretrend_diagnostics.csv", index=False)
    concentration.to_csv(DIAG_DIR / "effect_concentration.csv", index=False)
    write_markdown(pretrends, concentration)

    print(f"Wrote {DIAG_DIR / 'pretrend_diagnostics.csv'}")
    print(f"Wrote {DIAG_DIR / 'effect_concentration.csv'}")
    print(f"Wrote {DIAG_DIR / 'negative_effect_diagnostics.md'}")


if __name__ == "__main__":
    main()
