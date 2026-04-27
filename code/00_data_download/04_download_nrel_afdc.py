#!/usr/bin/env python3
"""
04_download_nrel_afdc.py
Downloads Alternative Fuel Station data from the NREL/NLR AFDC API.

Source: US Department of Energy's Alternative Fuels Data Center
Paper reference: "We collected data about EVCS from the United States
  Department of Energy's Alternative Fuels Data Center (AFDC)."
  (Zheng et al. 2024, Methods section)

The paper uses EVCS opened in 2019 and Feb 2021 – Jun 2023 in California.
This script downloads ALL electric stations in California for completeness,
which can then be filtered during data processing.

API Docs: https://developer.nlr.gov/docs/transportation/alt-fuel-stations-v1/

Usage:
    python3 code/00_data_download/04_download_nrel_afdc.py

Note: The repo already includes a snapshot CSV in data/alt_fuel_stations (Jul 30 2023).csv
      This script fetches the latest data from the API for broader replication.
"""

import json
import os
import sys
import urllib.request
import urllib.parse
from datetime import datetime
from pathlib import Path

# Resolve project root
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

# Load .env manually (no external deps needed)
def load_env(env_path):
    """Simple .env loader — no dependency on python-dotenv."""
    env_vars = {}
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, val = line.partition("=")
                    env_vars[key.strip()] = val.strip()
    return env_vars

env = load_env(PROJECT_ROOT / ".env")
API_KEY = env.get("NREL_API_KEY", os.environ.get("NREL_API_KEY", "DEMO_KEY"))

OUTPUT_DIR = PROJECT_ROOT / "data" / "raw" / "nrel_afdc"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def download_all_ev_stations_california():
    """
    Download all electric vehicle charging stations in California.
    Uses the /api/alt-fuel-stations/v1 endpoint with JSON format.
    """
    # Try the new NLR domain first, fall back to legacy NREL domain
    base_urls = [
        "https://developer.nlr.gov/api/alt-fuel-stations/v1.json",
        "https://developer.nrel.gov/api/alt-fuel-stations/v1.json",
    ]

    params = {
        "api_key": API_KEY,
        "fuel_type": "ELEC",         # Electric only
        "state": "CA",               # California
        "status": "E,T",             # E=Open, T=Temporarily unavailable
        "access": "public",          # Public stations only (per paper)
        "limit": "all",              # Get all results
    }

    query_string = urllib.parse.urlencode(params)

    for base_url in base_urls:
        url = f"{base_url}?{query_string}"
        print(f"Requesting: {base_url}")
        print(f"  Params: fuel_type=ELEC, state=CA, status=E,T, access=public")

        try:
            req = urllib.request.Request(url)
            req.add_header("User-Agent", "EVCS-Replication/1.0")

            with urllib.request.urlopen(req, timeout=120) as response:
                data = json.loads(response.read().decode("utf-8"))

            total = data.get("total_results", 0)
            # API may return stations under either key depending on version
            stations = data.get("fuel_stations", data.get("alt_fuel_stations", []))
            print(f"  ✓ Retrieved {total} stations ({len(stations)} records)")

            # Save full JSON response
            timestamp = datetime.now().strftime("%Y%m%d")
            json_path = OUTPUT_DIR / f"afdc_california_ev_stations_{timestamp}.json"
            with open(json_path, "w") as f:
                json.dump(data, f, indent=2)
            print(f"  ✓ Saved JSON → {json_path}")

            # Also save as CSV for convenience
            csv_path = OUTPUT_DIR / f"afdc_california_ev_stations_{timestamp}.csv"
            if stations:
                import csv
                fieldnames = stations[0].keys()
                with open(csv_path, "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(stations)
                print(f"  ✓ Saved CSV  → {csv_path}")

            return True

        except urllib.error.HTTPError as e:
            print(f"  ✗ HTTP Error {e.code}: {e.reason}")
            if e.code == 403:
                print("    → API key may be invalid. Sign up at https://developer.nlr.gov/signup/")
            continue
        except urllib.error.URLError as e:
            print(f"  ✗ URL Error: {e.reason}")
            continue

    print("\nERROR: Could not download data from any API endpoint.")
    print("Tip: You already have a snapshot at data/alt_fuel_stations (Jul 30 2023).csv")
    return False


def download_ev_networks():
    """Download the list of EV charging networks for reference."""
    base_urls = [
        "https://developer.nlr.gov/api/alt-fuel-stations/v1/electric-networks.json",
        "https://developer.nrel.gov/api/alt-fuel-stations/v1/electric-networks.json",
    ]

    params = {"api_key": API_KEY}
    query_string = urllib.parse.urlencode(params)

    for base_url in base_urls:
        url = f"{base_url}?{query_string}"
        try:
            req = urllib.request.Request(url)
            req.add_header("User-Agent", "EVCS-Replication/1.0")
            with urllib.request.urlopen(req, timeout=30) as response:
                data = json.loads(response.read().decode("utf-8"))

            networks_path = OUTPUT_DIR / "ev_charging_networks.json"
            with open(networks_path, "w") as f:
                json.dump(data, f, indent=2)
            print(f"  ✓ Saved EV networks → {networks_path}")
            return True
        except Exception as e:
            print(f"  ✗ Networks download failed: {e}")
            continue
    return False


if __name__ == "__main__":
    print("=" * 60)
    print("NREL/NLR AFDC — EV Charging Station Data Download")
    print(f"API Key: {'DEMO_KEY (limited)' if API_KEY == 'DEMO_KEY' else '***' + API_KEY[-4:]}")
    print(f"Output:  {OUTPUT_DIR}")
    print("=" * 60)

    success = download_all_ev_stations_california()

    print("\n--- Additional data ---")
    download_ev_networks()

    if success:
        print("\n✓ NREL/AFDC download complete!")
    else:
        print("\n✗ Download failed. Check API key and network connection.")
        sys.exit(1)
