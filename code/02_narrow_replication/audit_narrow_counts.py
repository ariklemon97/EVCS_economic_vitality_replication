"""
Audit narrow matched-panel construction and regression observation counts.
"""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from code.analysis_config import NARROW_WINDOW, OUTPUT_DIR, PROCESSED_DIR


TABLE_DIR = OUTPUT_DIR / "tables" / "narrow"

PANELS = {
    ("Period1_2019", "All"): "df_psm_narrow_p1_all.csv",
    ("Period1_2019", "Disadvantaged"): "df_psm_narrow_p1_dis.csv",
    ("Period2_2021_2023", "All"): "df_psm_narrow_p2_all.csv",
    ("Period2_2021_2023", "Disadvantaged"): "df_psm_narrow_p2_dis.csv",
}

PUBLISHED_ALL_NOBS = {
    "Period1_2019": 133_649,
    "Period2_2021_2023": 1_235_819,
}


def expected_months(start: int, end: int) -> list[int]:
    periods = pd.period_range(str(start), str(end), freq="M")
    return [int(p.strftime("%Y%m")) for p in periods]


def summarize_panel(period: str, sample: str, filename: str) -> dict[str, object]:
    path = PROCESSED_DIR / filename
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path, low_memory=False)
    df = df.sort_values(["placekey", "date"]).drop_duplicates(["placekey", "date"], keep="first")
    df["date_numeric_orig"] = pd.to_numeric(df["date_numeric_orig"], errors="raise").astype(int)
    df["open_yyyymm"] = pd.to_numeric(df["open_yyyymm"], errors="coerce").fillna(0).astype(int)
    df["is_treated"] = pd.to_numeric(df["is_treated"], errors="raise").astype(int)

    months = sorted(df["date_numeric_orig"].unique())
    if period == "Period1_2019":
        expected = expected_months(NARROW_WINDOW.period_1_start, NARROW_WINDOW.period_1_end)
        treatment_start = NARROW_WINDOW.period_1_start
    else:
        expected = expected_months(NARROW_WINDOW.period_2_start, NARROW_WINDOW.period_2_end)
        treatment_start = NARROW_WINDOW.p2_treatment_start

    treated_open = df.loc[df["is_treated"].eq(1), "open_yyyymm"]
    treated_open = treated_open[treated_open.gt(0)]
    bad_treatment_open = int((treated_open.lt(treatment_start) | treated_open.gt(expected[-1])).sum())

    has_dac = "is_disadvantaged" in df.columns
    if has_dac:
        dac = pd.to_numeric(df["is_disadvantaged"], errors="coerce")
        dac_missing = int(dac.isna().sum())
        dac_rows = int(dac.fillna(0).eq(1).sum())
        non_dac_rows_in_dis_sample = int(dac.fillna(0).ne(1).sum()) if sample == "Disadvantaged" else 0
    else:
        dac_missing = len(df)
        dac_rows = 0
        non_dac_rows_in_dis_sample = len(df) if sample == "Disadvantaged" else 0

    complete_cols = ["lcus", "lspend", "port_treat", "placekey", "county_fips", "date"]
    nobs_main_inputs = int(df[complete_cols].dropna().shape[0])
    unique_pois = int(df["placekey"].nunique())
    n_months = len(months)
    balanced_expected_rows = unique_pois * n_months
    published = PUBLISHED_ALL_NOBS.get(period) if sample == "All" else None

    row = {
        "period": period,
        "sample": sample,
        "filename": filename,
        "rows": int(len(df)),
        "main_input_complete_rows": nobs_main_inputs,
        "unique_pois": unique_pois,
        "treated_pois": int(df.loc[df["is_treated"].eq(1), "placekey"].nunique()),
        "control_pois": int(df.loc[df["is_treated"].eq(0), "placekey"].nunique()),
        "matched_pairs": int(df["match_pair_id"].nunique()) if "match_pair_id" in df.columns else 0,
        "min_month": int(min(months)),
        "max_month": int(max(months)),
        "n_months": n_months,
        "expected_months": " ".join(map(str, expected)),
        "observed_months": " ".join(map(str, months)),
        "has_all_expected_months": months == expected,
        "balanced_expected_rows": int(balanced_expected_rows),
        "unbalanced_row_gap": int(balanced_expected_rows - len(df)),
        "has_is_disadvantaged": has_dac,
        "disadvantaged_rows": dac_rows,
        "missing_disadvantaged_rows": dac_missing,
        "non_disadvantaged_rows_in_dis_sample": non_dac_rows_in_dis_sample,
        "treatment_start_month": treatment_start,
        "bad_treated_open_rows": bad_treatment_open,
        "published_all_nobs_reference": published,
        "matches_published_all_nobs": (len(df) == published) if published is not None else None,
    }
    return row


def validate(audit: pd.DataFrame) -> None:
    failures = []
    if not audit["has_all_expected_months"].all():
        failures.append("one or more panels do not contain exactly the expected study months")
    if not (audit["treated_pois"] == audit["control_pois"]).all():
        failures.append("treated/control matched POI counts are unequal")
    if not audit["has_is_disadvantaged"].all():
        failures.append("one or more panels are missing is_disadvantaged")
    if audit["missing_disadvantaged_rows"].sum() > 0:
        failures.append("is_disadvantaged has missing values")
    if audit.loc[audit["sample"].eq("Disadvantaged"), "non_disadvantaged_rows_in_dis_sample"].sum() > 0:
        failures.append("disadvantaged panels contain non-disadvantaged rows")
    if audit["bad_treated_open_rows"].sum() > 0:
        failures.append("treated rows include opening months outside the allowed treatment window")

    if failures:
        raise SystemExit("Narrow count audit failed: " + "; ".join(failures))


def main() -> None:
    rows = [summarize_panel(period, sample, filename) for (period, sample), filename in PANELS.items()]
    audit = pd.DataFrame(rows)

    result_path = TABLE_DIR / "01_main_model.csv"
    if result_path.exists():
        results = pd.read_csv(result_path)
        nobs = (
            results.groupby(["period", "sample"])["nobs"]
            .first()
            .rename("regression_nobs")
            .reset_index()
        )
        audit = audit.merge(nobs, on=["period", "sample"], how="left")
        audit["regression_nobs_matches_panel_rows"] = audit["regression_nobs"].fillna(-1).astype(int).eq(audit["rows"])

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    out_path = TABLE_DIR / "00_sample_audit.csv"
    audit.to_csv(out_path, index=False)

    print("\nNarrow sample audit")
    print(audit.to_string(index=False))
    print(f"\nSaved {out_path}")

    validate(audit)
    published_checks = audit.loc[audit["sample"].eq("All"), ["period", "rows", "published_all_nobs_reference", "matches_published_all_nobs"]]
    print("\nPublished all-sample observation-count comparison")
    print(published_checks.to_string(index=False))


if __name__ == "__main__":
    main()
