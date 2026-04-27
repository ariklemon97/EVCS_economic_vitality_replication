"""
Builds a monthly spend panel from Dewey SafeGraph spend-pattern parquets.

This script extracts:
  - customer counts
  - total spend
  - spend per transaction
  - income-bucket customer counts
  - income-weighted average customer income

Foot-traffic metrics are merged later from the weekly Dewey foot-traffic feed.
"""

from __future__ import annotations

import ast
import glob
import json
import logging
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from code.analysis_config import PROCESSED_DIR, RAW_DIR


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

SPEND_DIR = RAW_DIR / "dewey" / "spend_patterns" / "spend"

INCOME_BUCKET_MAP = {
    "<25k": "cus_.25k",
    "25-45k": "cus_25.45k",
    "45-60k": "cus_45.60k",
    "60-75k": "cus_60.75k",
    "75-100k": "cus_75.100k",
    "100-150k": "cus_100.150k",
    ">150k": "cus_.150k",
}

INCOME_BUCKET_MIDPOINTS = {
    "cus_.25k": 12_500,
    "cus_25.45k": 35_000,
    "cus_45.60k": 52_500,
    "cus_60.75k": 67_500,
    "cus_75.100k": 87_500,
    "cus_100.150k": 125_000,
    "cus_.150k": 175_000,
}


def parse_jsonish(value: object) -> dict[str, float]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return {}
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return ast.literal_eval(value)
    return {}


def extract_income_counts(series: pd.Series) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    for value in series:
        parsed = parse_jsonish(value)
        row = {target: float(parsed.get(source, 0.0) or 0.0) for source, target in INCOME_BUCKET_MAP.items()}
        rows.append(row)
    return pd.DataFrame(rows, index=series.index)


def compute_avg_customer_income(df: pd.DataFrame) -> pd.Series:
    weighted = sum(df[col].fillna(0) * midpoint for col, midpoint in INCOME_BUCKET_MIDPOINTS.items())
    counts = df[list(INCOME_BUCKET_MIDPOINTS)].fillna(0).sum(axis=1)
    return weighted.div(counts.where(counts > 0))


def main() -> None:
    poi_path = PROCESSED_DIR / "poi_treatment_assignments.parquet"
    if not poi_path.exists():
        logging.error("Missing POI assignments from Step 1.")
        return

    valid_keys = set(pd.read_parquet(poi_path, columns=["placekey"])["placekey"].unique())
    parquet_files = sorted(glob.glob(str(SPEND_DIR / "**" / "*.parquet"), recursive=True))
    if not parquet_files:
        logging.warning("No spend-pattern parquet files found.")
        return

    logging.info("Processing %d spend parquet files...", len(parquet_files))

    monthly_chunks: list[pd.DataFrame] = []
    base_cols = [
        "PLACEKEY",
        "SPEND_DATE_RANGE_START",
        "RAW_NUM_CUSTOMERS",
        "RAW_TOTAL_SPEND",
        "MEDIAN_SPEND_PER_TRANSACTION",
        "BUCKETED_CUSTOMER_INCOMES",
    ]

    for idx, file_path in enumerate(parquet_files, start=1):
        try:
            df = pd.read_parquet(file_path, columns=base_cols)
        except Exception as exc:
            logging.error("Failed to read %s: %s", file_path, exc)
            continue

        df.columns = [col.lower() for col in df.columns]
        df = df[df["placekey"].isin(valid_keys)]
        if df.empty:
            continue

        df["spend_date_range_start"] = pd.to_datetime(df["spend_date_range_start"], utc=True).dt.tz_localize(None)
        df["year_month"] = df["spend_date_range_start"].dt.to_period("M").astype(str)

        df["raw_visit_counts"] = pd.to_numeric(df["raw_num_customers"], errors="coerce").fillna(0.0)
        df["raw_total_spend"] = pd.to_numeric(df["raw_total_spend"], errors="coerce").fillna(0.0)
        df["spend_by_transaction"] = pd.to_numeric(df["median_spend_per_transaction"], errors="coerce")

        income_df = extract_income_counts(df["bucketed_customer_incomes"])
        df = pd.concat([df, income_df], axis=1)
        df["avg_customer_income"] = compute_avg_customer_income(df)

        keep_cols = [
            "placekey",
            "year_month",
            "raw_visit_counts",
            "raw_total_spend",
            "spend_by_transaction",
            *INCOME_BUCKET_MIDPOINTS.keys(),
            "avg_customer_income",
        ]

        grouped = (
            df[keep_cols]
            .groupby(["placekey", "year_month"], as_index=False)
            .agg(
                {
                    "raw_visit_counts": "sum",
                    "raw_total_spend": "sum",
                    "spend_by_transaction": "mean",
                    **{col: "sum" for col in INCOME_BUCKET_MIDPOINTS},
                    "avg_customer_income": "mean",
                }
            )
        )
        monthly_chunks.append(grouped)

        if idx % 25 == 0:
            logging.info("Processed %d / %d spend files", idx, len(parquet_files))

    if not monthly_chunks:
        logging.warning("No spend data matched the assigned POIs.")
        return

    final_temporal = pd.concat(monthly_chunks, ignore_index=True)
    final_temporal = (
        final_temporal.groupby(["placekey", "year_month"], as_index=False)
        .agg(
            {
                "raw_visit_counts": "sum",
                "raw_total_spend": "sum",
                "spend_by_transaction": "mean",
                **{col: "sum" for col in INCOME_BUCKET_MIDPOINTS},
                "avg_customer_income": "mean",
            }
        )
        .sort_values(["placekey", "year_month"])
    )

    out_path = PROCESSED_DIR / "monthly_spend_panel.parquet"
    final_temporal.to_parquet(out_path, index=False)
    logging.info("Created monthly spend panel %s with shape %s", out_path, final_temporal.shape)


if __name__ == "__main__":
    main()
