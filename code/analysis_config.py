"""
Shared study-window and category configuration for the replication pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUT_DIR = PROJECT_ROOT / "output"


TARGET_NAICS_PREFIXES = ("72", "44", "45", "71")

# The paper-facing estimand is local business spillovers from chargers that are
# spatially adjacent to consumer-facing POIs, not the average effect of every
# public EV installation in California.
COMMERCIAL_ADJACENT_RADIUS_M = 500
MIN_COMMERCIAL_POIS_NEAR_EVCS = 1
LOCAL_BUSINESS_CONTEXT_RADIUS_M = 500
MIN_NEARBY_LOCAL_BUSINESSES = 1


NARROW_PERIOD_1_START = 201901
NARROW_PERIOD_1_END = 201912
NARROW_PERIOD_2_START = 202102
NARROW_PERIOD_2_END = 202306


BROAD_PERIOD_1_START = 201901
BROAD_PERIOD_1_END = 201912
BROAD_PERIOD_2_START = 202102
BROAD_PERIOD_2_END = 202601


FOOT_TRAFFIC_API_URL = (
    "https://api.deweydata.io/api/v1/external/data/"
    "prj_noxsy6zo__cdst_ycpyx7nuje33xjci"
)


@dataclass(frozen=True)
class StudyWindow:
    period_1_start: int
    period_1_end: int
    period_2_start: int
    period_2_end: int


NARROW_WINDOW = StudyWindow(
    period_1_start=NARROW_PERIOD_1_START,
    period_1_end=NARROW_PERIOD_1_END,
    period_2_start=NARROW_PERIOD_2_START,
    period_2_end=NARROW_PERIOD_2_END,
)


BROAD_WINDOW = StudyWindow(
    period_1_start=BROAD_PERIOD_1_START,
    period_1_end=BROAD_PERIOD_1_END,
    period_2_start=BROAD_PERIOD_2_START,
    period_2_end=BROAD_PERIOD_2_END,
)
