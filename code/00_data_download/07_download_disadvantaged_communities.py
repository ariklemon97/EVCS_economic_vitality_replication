#!/usr/bin/env python3
"""
Download the CEC low-income/disadvantaged community geography.

The ArcGIS service enforces a 2,000-feature page limit, so this script pages
through the REST endpoint and writes a complete GeoJSON file used by
03_incorporate_covariates.py.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = PROJECT_ROOT / "data" / "raw" / "california_energy_commission"
OUT_PATH = OUT_DIR / "ca_low_income_or_disadvantaged_communities.geojson"

SERVICE_URL = (
    "https://services5.arcgis.com/tAovI6khEYSBCxdW/arcgis/rest/services/"
    "Low_Income_or_Disadvantaged_Communities_Designated_by_California/"
    "FeatureServer/0/query"
)


def existing_file_is_valid(path: Path) -> bool:
    if not path.exists() or path.stat().st_size < 1_000:
        return False
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError:
        return False
    return data.get("type") == "FeatureCollection" and len(data.get("features", [])) > 0


def get_json(params: dict[str, object]) -> dict:
    response = requests.get(SERVICE_URL, params=params, timeout=120)
    response.raise_for_status()
    data = response.json()
    if "error" in data:
        raise RuntimeError(data["error"])
    return data


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if existing_file_is_valid(OUT_PATH):
        print(f"Existing disadvantaged-community GeoJSON is valid: {OUT_PATH}")
        return

    count_data = get_json({"where": "1=1", "returnCountOnly": "true", "f": "json"})
    total = int(count_data["count"])
    print(f"Downloading {total} CEC low-income/disadvantaged community features...")

    features = []
    page_size = 2000
    crs = {"type": "name", "properties": {"name": "EPSG:4326"}}
    for offset in range(0, total, page_size):
        page = get_json(
            {
                "where": "1=1",
                "outFields": "*",
                "outSR": 4326,
                "f": "geojson",
                "resultOffset": offset,
                "resultRecordCount": page_size,
            }
        )
        page_features = page.get("features", [])
        features.extend(page_features)
        print(f"  fetched {len(features)}/{total}")
        if len(page_features) == 0:
            break

    if len(features) != total:
        raise RuntimeError(f"Expected {total} features, downloaded {len(features)}.")

    collection = {
        "type": "FeatureCollection",
        "crs": crs,
        "features": features,
    }
    OUT_PATH.write_text(json.dumps(collection))
    print(f"Saved {OUT_PATH} ({len(features)} features)")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
