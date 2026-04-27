#!/usr/bin/env python3
"""
05_download_supplementary_data.py
Downloads publicly available supplementary datasets used in the paper.

Datasets:
  1. EPA Smart Location Database — walkability index, road density, etc.
  2. California Energy Commission — disadvantaged/low-income communities,
     EV sales data
  3. ACS Census data — socio-demographics at census tract level
  4. OpenStreetMap building footprints for California

Paper references (Zheng et al. 2024, Methods/Data section):
  - "Auto-oriented road miles... and walkability index... from EPA's Smart
     Location Database"
  - "Locations of underprivileged communities... from California Energy
     Commission"
  - "Data on EV sales at the county level was also obtained from the
     California Energy Commission"
  - "Socio-demographic data at the census tract level... from ACS"
  - "Building footprint data... from OpenStreetMap"

Usage:
    python3 code/00_data_download/05_download_supplementary_data.py
"""

import os
import sys
import urllib.request
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent


def download_file(url, dest_path, description=""):
    """Download a file from URL to dest_path with progress indication."""
    dest_path = Path(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    if dest_path.exists():
        size_mb = dest_path.stat().st_size / (1024 * 1024)
        print(f"  ⏭  Already exists ({size_mb:.1f} MB): {dest_path.name}")
        return True

    print(f"  ↓  Downloading: {description or url}")
    print(f"     → {dest_path}")

    try:
        req = urllib.request.Request(url)
        req.add_header("User-Agent", "EVCS-Replication/1.0")

        with urllib.request.urlopen(req, timeout=300) as response:
            total_size = int(response.headers.get("Content-Length", 0))
            block_size = 8192
            downloaded = 0

            with open(dest_path, "wb") as f:
                while True:
                    buffer = response.read(block_size)
                    if not buffer:
                        break
                    downloaded += len(buffer)
                    f.write(buffer)

                    if total_size > 0:
                        pct = downloaded / total_size * 100
                        mb = downloaded / (1024 * 1024)
                        print(f"\r     {mb:.1f} MB ({pct:.0f}%)", end="", flush=True)

        print(f"\n  ✓  Done: {dest_path.name}")
        return True

    except Exception as e:
        print(f"\n  ✗  Failed: {e}")
        if dest_path.exists():
            dest_path.unlink()  # Remove partial download
        return False


def download_epa_smart_location():
    """
    EPA Smart Location Database — contains walkability index,
    auto-oriented road/intersection density at block group level.
    """
    print("\n--- [1/4] EPA Smart Location Database ---")
    dest_dir = PROJECT_ROOT / "data" / "raw" / "epa_smart_location"

    # The SLD is distributed as a geodatabase or CSV
    # National CSV download:
    url = "https://edg.epa.gov/EPADataCommons/public/OA/SLD/SmartLocationDatabaseV3.zip"
    return download_file(
        url,
        dest_dir / "SmartLocationDatabaseV3.zip",
        "EPA Smart Location Database V3 (national)"
    )


def download_california_energy_commission():
    """
    California Energy Commission data:
      - Disadvantaged/low-income communities (CalEnviroScreen + Justice40)
      - EV sales by county
    """
    print("\n--- [2/4] California Energy Commission ---")
    dest_dir = PROJECT_ROOT / "data" / "raw" / "california_energy_commission"

    results = []

    # Disadvantaged communities GeoJSON
    # From: https://cecgis-caenergy.opendata.arcgis.com/datasets/
    dac_url = (
        "https://opendata.arcgis.com/api/v3/datasets/"
        "ec0adaef7db349dfa584ee33ea4c3f1f_0/downloads/"
        "data?format=geojson&spatialRefId=4326"
    )
    results.append(download_file(
        dac_url,
        dest_dir / "ca_justice40_disadvantaged_communities.geojson",
        "CA + Justice40 disadvantaged/low-income communities"
    ))

    # EV sales data — typically available as Excel from CEC website
    # Direct download URL may change; provide instructions
    ev_sales_url = (
        "https://www.energy.ca.gov/data-reports/energy-almanac/"
        "zero-emission-vehicle-and-infrastructure-statistics/new-zev-sales"
    )
    readme_path = dest_dir / "README_ev_sales.txt"
    readme_path.parent.mkdir(parents=True, exist_ok=True)
    with open(readme_path, "w") as f:
        f.write(f"EV Sales Data by County\n")
        f.write(f"=======================\n")
        f.write(f"Download manually from:\n{ev_sales_url}\n\n")
        f.write(f"Look for 'New ZEV Sales in California' Excel file.\n")
        f.write(f"Save as: new_zev_sales.xlsx in this directory.\n")
    print(f"  📝  Created download instructions: {readme_path.name}")

    return all(results)


def download_acs_census():
    """
    ACS Census data — socio-demographic variables at census tract level.
    Population, median household income, employment, gender, race/ethnicity.

    The paper used 5-year ACS estimates. We download via Census API or
    provide instructions for bulk download.
    """
    print("\n--- [3/4] ACS Census Data ---")
    dest_dir = PROJECT_ROOT / "data" / "raw" / "census_acs"
    dest_dir.mkdir(parents=True, exist_ok=True)

    readme_path = dest_dir / "README_acs.txt"
    with open(readme_path, "w") as f:
        f.write("American Community Survey (ACS) Data\n")
        f.write("====================================\n\n")
        f.write("Required variables (at census tract level for California):\n")
        f.write("  - Total population\n")
        f.write("  - Median household income\n")
        f.write("  - Employed population / employment rate\n")
        f.write("  - Gender distribution (% male, % female)\n")
        f.write("  - Race/ethnicity composition\n\n")
        f.write("Download options:\n")
        f.write("  1. Census API: https://api.census.gov/data.html\n")
        f.write("     (free API key at https://api.census.gov/data/key_signup.html)\n")
        f.write("  2. data.census.gov bulk download\n")
        f.write("  3. NHGIS (https://www.nhgis.org/) for pre-processed tables\n\n")
        f.write("Tables needed:\n")
        f.write("  - B01003: Total Population\n")
        f.write("  - B19013: Median Household Income\n")
        f.write("  - B23025: Employment Status\n")
        f.write("  - B01001: Sex by Age\n")
        f.write("  - B03002: Hispanic or Latino Origin by Race\n\n")
        f.write("Year: 2019 5-Year ACS for the 2019 analysis period\n")
        f.write("      2021 5-Year ACS for the 2021-2023 analysis period\n")
    print(f"  📝  Created download instructions: {readme_path.name}")
    return True


def download_osm_california():
    """
    OpenStreetMap building footprints for California.
    From Geofabrik.
    """
    print("\n--- [4/4] OpenStreetMap California ---")
    dest_dir = PROJECT_ROOT / "data" / "raw" / "openstreetmap"

    url = "https://download.geofabrik.de/north-america/us/california-latest-free.shp.zip"
    return download_file(
        url,
        dest_dir / "california-latest-free.shp.zip",
        "OpenStreetMap California shapefiles (Geofabrik)"
    )


if __name__ == "__main__":
    print("=" * 60)
    print("Supplementary Data Downloads")
    print("=" * 60)

    results = {
        "EPA Smart Location": download_epa_smart_location(),
        "CA Energy Commission": download_california_energy_commission(),
        "ACS Census": download_acs_census(),
        "OpenStreetMap CA": download_osm_california(),
    }

    print("\n" + "=" * 60)
    print("Download Summary")
    print("=" * 60)
    for name, status in results.items():
        icon = "✓" if status else "✗"
        print(f"  {icon}  {name}")

    if not all(results.values()):
        print("\n⚠  Some downloads failed. See messages above.")
        sys.exit(1)
    else:
        print("\n✓  All supplementary data downloads complete!")
