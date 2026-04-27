"""
Master Python entrypoint for the final replication package.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PIPELINE = [
    ["code/01_data_processing/01_spatial_join_pois_evcs.py"],
    ["code/01_data_processing/02_clean_and_aggregate_spend.py"],
    ["code/01_data_processing/02b_clean_and_aggregate_foot_traffic.py"],
    ["code/01_data_processing/03_incorporate_covariates.py"],
    ["code/01_data_processing/04_compile_regression_panel.py"],
    ["code/01_data_processing/05_propensity_score_matching.py", "--window", "all"],
    ["code/03_broad_replication/04_spatial_competition.py"],
    ["code/03_broad_replication/06_finalize_panel.py"],
    ["code/02_narrow_replication/run_narrow_replication.py"],
    ["code/03_broad_replication/run_broad_replication.py", "--section", "main"],
    [
        "code/03_broad_replication/run_stacked_regression.py",
        "--target",
        "all",
        "--pre",
        "6",
        "--post",
        "12",
        "--min-treated",
        "25",
        "--max-control-pois",
        "10000",
        "--seed",
        "20260425",
    ],
]


def main() -> None:
    for command in PIPELINE:
        script = PROJECT_ROOT / command[0]
        args = command[1:]
        print(f"=== Running {' '.join(command)} ===", flush=True)
        subprocess.run([sys.executable, str(script), *args], check=True, cwd=PROJECT_ROOT)


if __name__ == "__main__":
    main()
