"""
Download weekly Dewey foot-traffic parquet partitions for the study window.

The API key should be supplied via `DEWEY_FOOT_TRAFFIC_KEY`.
"""

from __future__ import annotations

import argparse
import logging
from datetime import date
from pathlib import Path
import sys

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from code.analysis_config import BROAD_WINDOW, FOOT_TRAFFIC_API_URL, RAW_DIR


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--start-date", default=f"{BROAD_WINDOW.period_1_start // 100:04d}-{BROAD_WINDOW.period_1_start % 100:02d}-01")
    parser.add_argument("--end-date", default=f"{BROAD_WINDOW.period_2_end // 100:04d}-{BROAD_WINDOW.period_2_end % 100:02d}-31")
    parser.add_argument("--output-dir", default=str(RAW_DIR / "dewey" / "foot_traffic"))
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--timeout", type=int, default=300)
    return parser.parse_args()


def iter_download_links(api_key: str) -> list[dict]:
    session = requests.Session()
    page = 1
    links: list[dict] = []

    while True:
        response = session.get(
            FOOT_TRAFFIC_API_URL,
            headers={"X-API-KEY": api_key},
            params={"page": page},
            timeout=60,
        )
        response.raise_for_status()
        payload = response.json()
        links.extend(payload.get("download_links", []))
        if page >= payload.get("total_pages", 1):
            break
        page += 1

    return links


def in_window(partition_key: str, start_date: date, end_date: date) -> bool:
    part_date = date.fromisoformat(partition_key)
    return start_date <= part_date <= end_date


def download_file(url: str, destination: Path, timeout: int) -> None:
    response = requests.get(url, stream=True, timeout=timeout)
    response.raise_for_status()
    with destination.open("wb") as handle:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                handle.write(chunk)


def main() -> None:
    args = parse_args()
    api_key = args.api_key or __import__("os").environ.get("DEWEY_FOOT_TRAFFIC_KEY")
    if not api_key:
        raise SystemExit("Missing API key. Set DEWEY_FOOT_TRAFFIC_KEY or pass --api-key.")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    start_date = date.fromisoformat(args.start_date)
    end_date = date.fromisoformat(args.end_date)

    logging.info("Fetching Dewey foot-traffic metadata...")
    links = iter_download_links(api_key)
    selected = [item for item in links if in_window(item["partition_key"], start_date, end_date)]
    selected.sort(key=lambda item: item["partition_key"])
    if args.max_files is not None:
        selected = selected[: args.max_files]

    logging.info("Selected %d foot-traffic files between %s and %s", len(selected), start_date, end_date)
    for idx, item in enumerate(selected, start=1):
        partition_dir = out_dir / item["partition_key"]
        partition_dir.mkdir(parents=True, exist_ok=True)
        destination = partition_dir / item["file_name"]
        if destination.exists() and destination.stat().st_size == item.get("file_size_bytes", -1):
            logging.info("Skipping existing file %s", destination.name)
            continue
        logging.info("[%d/%d] Downloading %s", idx, len(selected), destination.name)
        download_file(item["link"], destination, args.timeout)


if __name__ == "__main__":
    main()
