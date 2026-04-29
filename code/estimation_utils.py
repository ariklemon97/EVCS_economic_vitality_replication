"""
Common estimation helpers for the final Python-only replication package.
"""

from __future__ import annotations

import re
import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from csdid.att_gt import ATTgt
from linearmodels.iv.absorbing import AbsorbingLS
from scipy.stats import norm

plt.rcParams.update(
    {
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
    }
)


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def sanitize_name(name: str) -> str:
    clean = re.sub(r"[^0-9a-zA-Z_]+", "_", name)
    if clean and clean[0].isdigit():
        clean = f"v_{clean}"
    return clean.lower()


def run_absorbing_ls(
    df: pd.DataFrame,
    outcome: str,
    regressors: list[str],
    absorb_cols: list[str],
    cluster_col: str,
) -> pd.DataFrame:
    base_cols = [outcome, *regressors, *absorb_cols, cluster_col]
    ordered_cols = list(dict.fromkeys(base_cols))
    work = df[ordered_cols].dropna().copy()
    for col in absorb_cols:
        work[col] = work[col].astype("category")
    cluster_codes = pd.Categorical(work[cluster_col]).codes
    model = AbsorbingLS(work[outcome], work[regressors], absorb=work[absorb_cols])
    result = model.fit(cov_type="clustered", clusters=cluster_codes)
    rows = []
    for term in regressors:
        estimate = float(result.params[term])
        std_error = float(result.std_errors[term])
        t_stat = estimate / std_error if std_error else np.nan
        p_value = 2 * norm.sf(abs(t_stat)) if std_error else np.nan
        rows.append(
            {
                "term": term,
                "estimate": estimate,
                "std.error": std_error,
                "t_stat": t_stat,
                "p.value": p_value,
                "ci_low95": estimate - 1.96 * std_error,
                "ci_hi95": estimate + 1.96 * std_error,
                "nobs": int(result.nobs),
            }
        )
    return pd.DataFrame(rows)


@dataclass
class CSDIDResult:
    att_gt: pd.DataFrame
    simple: pd.DataFrame
    dynamic: pd.DataFrame
    group: pd.DataFrame
    calendar: pd.DataFrame


def _weighted_summary(
    df: pd.DataFrame,
    value_col: str,
    se_col: str,
    weight_col: str,
) -> tuple[float, float, int, int, float]:
    if df.empty:
        return np.nan, np.nan, 0, 0, 0.0

    work = df[[value_col, se_col, weight_col]].copy()
    total_cells = len(work)
    valid = np.isfinite(work[value_col]) & np.isfinite(work[se_col]) & np.isfinite(work[weight_col])
    valid &= work[weight_col].astype(float) > 0
    valid_cells = int(valid.sum())

    if valid_cells == 0:
        return np.nan, np.nan, total_cells, 0, 0.0

    work = work.loc[valid].astype(float)
    weights = work[weight_col]
    att = np.average(work[value_col], weights=weights)
    se = np.sqrt(np.average(np.square(work[se_col]), weights=weights))
    return float(att), float(se), total_cells, valid_cells, float(weights.sum())


def run_csdid(
    df: pd.DataFrame,
    outcome: str,
    covariates: list[str],
    *,
    id_col: str = "placekey",
    time_col: str = "date_numeric",
    group_col: str = "first_treat_period",
    control_group: str = "nevertreated",
    est_method: str = "reg",
    biters: int = 200,
) -> CSDIDResult:
    use_cols = [id_col, time_col, group_col, outcome, *covariates]
    work = df[[col for col in use_cols if col in df.columns]].dropna().copy()
    rename_map = {col: sanitize_name(col) for col in work.columns}
    work = work.rename(columns=rename_map)
    id_col = rename_map[id_col]
    time_col = rename_map[time_col]
    group_col = rename_map[group_col]
    outcome = rename_map[outcome]
    covariates = [rename_map[col] for col in covariates if col in rename_map]
    work["unit_id"] = pd.factorize(work[id_col])[0] + 1
    max_time = work[time_col].max()

    # CS estimators require post-treatment observations for treated cohorts.
    # Drop cohorts first treated in the final observed period, which otherwise
    # create singular ATT(g,t) cells after extending the sample window.
    valid_ids = work.loc[(work[group_col] == 0) | (work[group_col] < max_time), "unit_id"].unique()
    work = work[work["unit_id"].isin(valid_ids)].copy()

    formula = "~ " + " + ".join(covariates) if covariates else None
    attgt = ATTgt(
        yname=outcome,
        tname=time_col,
        idname="unit_id",
        gname=group_col,
        data=work,
        control_group=control_group,
        xformla=formula,
        panel=True,
        allow_unbalanced_panel=True,
        biters=biters,
    ).fit(est_method=est_method, bstrap=False)

    att_gt = pd.DataFrame(
        {
            "group": attgt.results["group"],
            "time": attgt.results["year"],
            "att": attgt.results["att"],
            "se": attgt.results["se"],
        }
    )
    att_gt["event_time"] = att_gt["time"] - att_gt["group"]
    group_sizes = work.loc[work[group_col] > 0].groupby(group_col)["unit_id"].nunique().rename("group_size")
    att_gt = att_gt.merge(group_sizes, left_on="group", right_index=True, how="left")
    att_gt["group_size"] = att_gt["group_size"].fillna(0)
    att_gt["post_treatment"] = (att_gt["time"] >= att_gt["group"]).astype(int)

    post = att_gt[att_gt["post_treatment"] == 1].copy()
    simple_att, simple_se, total_cells, valid_cells, weight_sum = _weighted_summary(post, "att", "se", "group_size")
    simple = pd.DataFrame(
        [
            {
                "agg_type": "simple",
                "ATT": simple_att,
                "SE": simple_se,
                "CI_lower": simple_att - 1.96 * simple_se,
                "CI_upper": simple_att + 1.96 * simple_se,
                "n_total_cells": total_cells,
                "n_valid_cells": valid_cells,
                "weight_sum": weight_sum,
            }
        ]
    )

    dynamic_rows = []
    for event_time, chunk in att_gt.groupby("event_time", sort=True):
        att, se, total_cells, valid_cells, weight_sum = _weighted_summary(chunk, "att", "se", "group_size")
        dynamic_rows.append(
            {
                "agg_type": "dynamic",
                "event_time": event_time,
                "ATT": att,
                "SE": se,
                "CI_lower": att - 1.96 * se,
                "CI_upper": att + 1.96 * se,
                "n_total_cells": total_cells,
                "n_valid_cells": valid_cells,
                "weight_sum": weight_sum,
            }
        )
    dynamic = pd.DataFrame(dynamic_rows)

    group_rows = []
    for group, chunk in post.groupby("group", sort=True):
        att, se, total_cells, valid_cells, weight_sum = _weighted_summary(chunk, "att", "se", "group_size")
        group_rows.append(
            {
                "agg_type": "group",
                "event_time": group,
                "ATT": att,
                "SE": se,
                "CI_lower": att - 1.96 * se,
                "CI_upper": att + 1.96 * se,
                "n_total_cells": total_cells,
                "n_valid_cells": valid_cells,
                "weight_sum": weight_sum,
            }
        )
    group = pd.DataFrame(group_rows)

    calendar_rows = []
    for time_value, chunk in post.groupby("time", sort=True):
        att, se, total_cells, valid_cells, weight_sum = _weighted_summary(chunk, "att", "se", "group_size")
        calendar_rows.append(
            {
                "agg_type": "calendar",
                "event_time": time_value,
                "ATT": att,
                "SE": se,
                "CI_lower": att - 1.96 * se,
                "CI_upper": att + 1.96 * se,
                "n_total_cells": total_cells,
                "n_valid_cells": valid_cells,
                "weight_sum": weight_sum,
            }
        )
    calendar = pd.DataFrame(calendar_rows)

    return CSDIDResult(att_gt=att_gt, simple=simple, dynamic=dynamic, group=group, calendar=calendar)


def plot_event_study(dynamic: pd.DataFrame, title: str, output_path: Path) -> None:
    if dynamic.empty:
        return
    ensure_parent(output_path)
    fig, ax = plt.subplots(figsize=(8, 5.2))
    ax.plot(dynamic["event_time"], dynamic["ATT"], marker="o", linewidth=1.8)
    ax.vlines(dynamic["event_time"], dynamic["CI_lower"], dynamic["CI_upper"], alpha=0.8, linewidth=1.4)
    ax.axhline(0, color="grey", linestyle="--", linewidth=1)
    ax.axvline(-0.5, color="red", linestyle="--", linewidth=1)
    ax.set_xlabel("Months relative to treatment")
    ax.set_ylabel("ATT")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
