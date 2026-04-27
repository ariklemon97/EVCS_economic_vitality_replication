"""
Builds monthly Dewey foot-traffic metrics for the assigned POIs.

The weekly Dewey extract uses vendor-specific column names, so this script
searches a small set of plausible aliases for dwell time and distance from home.
"""

from __future__ import annotations

import glob
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

FOOT_TRAFFIC_DIR = RAW_DIR / "dewey" / "foot_traffic"
GLOBAL_PLACES_DIR = RAW_DIR / "dewey" / "global_places"

DATE_COLS = ["DATE_RANGE_START", "START_DATE", "WEEK_START", "VISIT_DATE_RANGE_START"]
DWELL_COLS = ["MEDIAN_DWELL", "MEDIAN_DWELL_TIME", "MEDIAN_MINUTES_DWELL"]
DIST_COLS = ["MEDIAN_DISTANCE_FROM_HOME", "DISTANCE_FROM_HOME", "MEDIAN_HOME_DISTANCE"]


def first_present(columns: list[str], candidates: list[str]) -> str | None:
    upper_map = {col.upper(): col for col in columns}
    for candidate in candidates:
        if candidate in upper_map:
            return upper_map[candidate]
    return None


def build_placekey_crosswalk(valid_keys: set[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    global_files = sorted(glob.glob(str(GLOBAL_PLACES_DIR / "**" / "*.parquet"), recursive=True))
    if not global_files:
        return pd.DataFrame(columns=["id_store", "placekey"]), pd.DataFrame(columns=["lat6", "lon6", "location_name_norm", "placekey"])

    mapping_frames: list[pd.DataFrame] = []
    cols = ["PLACEKEY", "STORE_ID", "LATITUDE", "LONGITUDE", "LOCATION_NAME"]
    for file_path in global_files:
        try:
            df = pd.read_parquet(file_path, columns=cols)
        except Exception as exc:
            logging.error("Failed to read %s: %s", file_path, exc)
            continue
        df = df[df["PLACEKEY"].isin(valid_keys)].copy()
        if df.empty:
            continue
        df["lat6"] = pd.to_numeric(df["LATITUDE"], errors="coerce").round(6)
        df["lon6"] = pd.to_numeric(df["LONGITUDE"], errors="coerce").round(6)
        df["location_name_norm"] = df["LOCATION_NAME"].fillna("").astype(str).str.strip().str.lower()
        mapping_frames.append(df.rename(columns={"PLACEKEY": "placekey", "STORE_ID": "id_store"}))

    if not mapping_frames:
        return pd.DataFrame(columns=["id_store", "placekey"]), pd.DataFrame(columns=["lat6", "lon6", "location_name_norm", "placekey"])

    crosswalk = pd.concat(mapping_frames, ignore_index=True)
    store_map = crosswalk[["id_store", "placekey"]].dropna(subset=["id_store"]).drop_duplicates("id_store")
    fallback_map = (
        crosswalk[["lat6", "lon6", "location_name_norm", "placekey"]]
        .dropna(subset=["lat6", "lon6"])
        .drop_duplicates(["lat6", "lon6", "location_name_norm"])
    )
    return store_map, fallback_map


def main() -> None:
    poi_path = PROCESSED_DIR / "poi_treatment_assignments.parquet"
    if not poi_path.exists():
        logging.error("Missing POI assignments from Step 1.")
        return

    valid_keys = set(pd.read_parquet(poi_path, columns=["placekey"])["placekey"].unique())
    store_map, fallback_map = build_placekey_crosswalk(valid_keys)
    if store_map.empty and fallback_map.empty:
        logging.warning("No placekey crosswalk available for Dewey foot-traffic matching.")
        return

    parquet_files = sorted(glob.glob(str(FOOT_TRAFFIC_DIR / "**" / "*.parquet"), recursive=True))
    if not parquet_files:
        logging.warning("No foot-traffic parquet files found under %s", FOOT_TRAFFIC_DIR)
        return

    monthly_chunks: list[pd.DataFrame] = []
    for idx, file_path in enumerate(parquet_files, start=1):
        try:
            df = pd.read_parquet(file_path)
        except Exception as exc:
            logging.error("Failed to read %s: %s", file_path, exc)
            continue

        placekey_col = first_present(list(df.columns), ["PLACEKEY"])
        store_col = first_present(list(df.columns), ["ID_STORE", "STORE_ID", "PERSISTENT_ID_STORE"])
        date_col = first_present(list(df.columns), DATE_COLS)
        dwell_col = first_present(list(df.columns), DWELL_COLS)
        dist_col = first_present(list(df.columns), DIST_COLS)
        if not placekey_col or not date_col:
            if not store_col or not date_col:
                continue

        keep_cols = [date_col]
        if placekey_col:
            keep_cols.append(placekey_col)
        if store_col and store_col not in keep_cols:
            keep_cols.append(store_col)
        for extra in ["LATITUDE", "LONGITUDE", "LOCATION_NAME"]:
            col = first_present(list(df.columns), [extra])
            if col and col not in keep_cols:
                keep_cols.append(col)
        if dwell_col:
            keep_cols.append(dwell_col)
        if dist_col:
            keep_cols.append(dist_col)

        rename_map = {date_col: "date_range_start"}
        if placekey_col:
            rename_map[placekey_col] = "placekey"
        if store_col:
            rename_map[store_col] = "id_store"
        if "LATITUDE" in keep_cols:
            rename_map["LATITUDE"] = "latitude"
        if "LONGITUDE" in keep_cols:
            rename_map["LONGITUDE"] = "longitude"
        if "LOCATION_NAME" in keep_cols:
            rename_map["LOCATION_NAME"] = "location_name"

        df = df[keep_cols].rename(columns=rename_map)
        if "placekey" not in df.columns:
            df["placekey"] = pd.NA
        if "id_store" in df.columns and not store_map.empty:
            df = df.merge(store_map, on="id_store", how="left", suffixes=("", "_store"))
            df["placekey"] = df["placekey"].fillna(df.get("placekey_store"))
            df = df.drop(columns=[col for col in ["placekey_store"] if col in df.columns])
        if df["placekey"].isna().any() and {"latitude", "longitude", "location_name"}.issubset(df.columns) and not fallback_map.empty:
            missing = df["placekey"].isna()
            fallback = df.loc[missing, ["latitude", "longitude", "location_name"]].copy()
            fallback["lat6"] = pd.to_numeric(fallback["latitude"], errors="coerce").round(6)
            fallback["lon6"] = pd.to_numeric(fallback["longitude"], errors="coerce").round(6)
            fallback["location_name_norm"] = fallback["location_name"].fillna("").astype(str).str.strip().str.lower()
            fallback = fallback.merge(fallback_map, on=["lat6", "lon6", "location_name_norm"], how="left")
            df.loc[missing, "placekey"] = fallback["placekey_y"].values if "placekey_y" in fallback.columns else fallback["placekey"].values

        df = df[df["placekey"].isin(valid_keys)]
        if df.empty:
            continue

        df["date_range_start"] = pd.to_datetime(df["date_range_start"], utc=True).dt.tz_localize(None)
        df["year_month"] = df["date_range_start"].dt.to_period("M").astype(str)
        if dwell_col:
            df["median_dwell"] = pd.to_numeric(df[dwell_col], errors="coerce")
        else:
            df["median_dwell"] = np.nan
        if dist_col:
            df["median_dist_home"] = pd.to_numeric(df[dist_col], errors="coerce")
        else:
            df["median_dist_home"] = np.nan

        monthly_chunks.append(
            df[["placekey", "year_month", "median_dwell", "median_dist_home"]]
            .groupby(["placekey", "year_month"], as_index=False)
            .mean()
        )

        if idx % 25 == 0:
            logging.info("Processed %d / %d foot-traffic files", idx, len(parquet_files))

    if not monthly_chunks:
        logging.warning("No foot-traffic data matched the assigned POIs.")
        return

    monthly = (
        pd.concat(monthly_chunks, ignore_index=True)
        .groupby(["placekey", "year_month"], as_index=False)
        .mean()
        .sort_values(["placekey", "year_month"])
    )
    out_path = PROCESSED_DIR / "monthly_foot_traffic_panel.parquet"
    monthly.to_parquet(out_path, index=False)
    logging.info("Created monthly foot-traffic panel %s with shape %s", out_path, monthly.shape)


if __name__ == "__main__":
    main()
