"""
Finalize the broad-replication panel.

This stage keeps only the spatial-competition extension. PDI outputs are
deliberately excluded from the final paper-facing pipeline.
"""

from __future__ import annotations

import logging
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from code.analysis_config import PROCESSED_DIR


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def build_monthly_competitor_exposure(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    pair_path = PROCESSED_DIR / "poi_competitor_matches.parquet"
    if not pair_path.exists():
        return None

    pairs = pd.read_parquet(pair_path)
    if pairs.empty:
        return None

    required = {
        "placekey",
        "competitor_open_yyyymm",
        "competitor_total_ports",
        "competitor_level2_ports",
        "competitor_dc_fast_ports",
    }
    missing = required - set(pairs.columns)
    if missing:
        logging.warning("Competitor match table is missing columns: %s", sorted(missing))
        return None

    for col in ["competitor_open_yyyymm", "competitor_total_ports", "competitor_level2_ports", "competitor_dc_fast_ports"]:
        pairs[col] = pd.to_numeric(pairs[col], errors="coerce").fillna(0)
    if "competitor_commercial_adjacent" in pairs.columns:
        pairs["competitor_commercial_adjacent"] = pd.to_numeric(
            pairs["competitor_commercial_adjacent"], errors="coerce"
        ).fillna(0).astype(int)
        pairs = pairs[pairs["competitor_commercial_adjacent"] == 1].copy()
    else:
        # Backward-compatible fallback: competitor matches are built from treated
        # target-business POIs, so the competitor charger exposure is
        # commercial-adjacent by construction in existing files.
        pairs["competitor_commercial_adjacent"] = 1
    pairs = pairs[pairs["competitor_open_yyyymm"] > 0].copy()
    if pairs.empty:
        return None

    period_map = (
        df[["date_numeric_orig", "date_numeric"]]
        .drop_duplicates()
        .sort_values("date_numeric_orig")
        .set_index("date_numeric_orig")["date_numeric"]
        .to_dict()
    )
    ordered_periods = sorted(period_map)

    def map_period(yyyymm: int) -> int:
        if yyyymm == 0:
            return 0
        if yyyymm in period_map:
            return int(period_map[yyyymm])
        candidates = [period for period in ordered_periods if period >= yyyymm]
        return int(period_map[candidates[0]]) if candidates else 0

    static = (
        pairs.groupby("placekey")
        .agg(
            competitor_open_yyyymm=("competitor_open_yyyymm", "min"),
            competitor_total_ports=("competitor_total_ports", "sum"),
            competitor_level2_ports=("competitor_level2_ports", "sum"),
            competitor_dc_fast_ports=("competitor_dc_fast_ports", "sum"),
            competitor_commercial_adjacent=("competitor_commercial_adjacent", "max"),
        )
        .reset_index()
    )
    static["Treatment_Competitor"] = 1
    static["competitor_has_level2"] = (static["competitor_level2_ports"] > 0).astype(int)
    static["competitor_has_dc_fast"] = (static["competitor_dc_fast_ports"] > 0).astype(int)
    static["competitor_first_treat_period"] = static["competitor_open_yyyymm"].apply(map_period)

    active_frames = []
    exposure_cols = ["competitor_total_ports", "competitor_level2_ports", "competitor_dc_fast_ports"]
    for period in ordered_periods:
        active = pairs[pairs["competitor_open_yyyymm"] <= period]
        if active.empty:
            continue
        month = active.groupby("placekey")[exposure_cols].sum().reset_index()
        month["date_numeric_orig"] = period
        active_frames.append(month)

    active_panel = pd.concat(active_frames, ignore_index=True) if active_frames else pd.DataFrame()
    active_panel = active_panel.rename(
        columns={
            "competitor_total_ports": "competitor_ports_active",
            "competitor_level2_ports": "competitor_level2_ports_active",
            "competitor_dc_fast_ports": "competitor_dc_fast_ports_active",
        }
    )
    return static, active_panel


def main() -> None:
    panel_path = PROCESSED_DIR / "df_pre_match.csv"
    comp_path = PROCESSED_DIR / "poi_spatial_competition.parquet"
    if not panel_path.exists():
        raise SystemExit("Missing df_pre_match.csv. Run the processing pipeline first.")

    df = pd.read_csv(panel_path)
    df = df.sort_values(["placekey", "date"]).drop_duplicates(["placekey", "date"], keep="first")

    dynamic_competition = build_monthly_competitor_exposure(df)
    if dynamic_competition is not None:
        static_comp, active_comp = dynamic_competition
        drop_cols = [
            "Treatment_Competitor",
            "competitor_open_yyyymm",
            "competitor_total_ports",
            "competitor_has_level2",
            "competitor_has_dc_fast",
            "competitor_first_treat_period",
            "competitor_commercial_adjacent",
        ]
        df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
        df = df.merge(static_comp, on="placekey", how="left")
        df = df.merge(active_comp, on=["placekey", "date_numeric_orig"], how="left")
        logging.info("Using month-specific competitor port exposure from poi_competitor_matches.parquet.")
    elif comp_path.exists():
        df_comp = pd.read_parquet(comp_path)
        keep_cols = [
            col
            for col in [
                "placekey",
                "Treatment_Competitor",
                "competitor_open_yyyymm",
                "competitor_total_ports",
                "competitor_has_level2",
                "competitor_has_dc_fast",
                "competitor_commercial_adjacent",
            ]
            if col in df_comp.columns
        ]
        if keep_cols:
            df_comp = df_comp[keep_cols].drop_duplicates("placekey")
            df = df.merge(df_comp, on="placekey", how="left")

    for col in ["Treatment_Competitor", "competitor_has_level2", "competitor_has_dc_fast"]:
        if col not in df.columns:
            df[col] = 0
        df[col] = df[col].fillna(0).astype(int)
    if "competitor_commercial_adjacent" not in df.columns:
        df["competitor_commercial_adjacent"] = df["Treatment_Competitor"]
    df["competitor_commercial_adjacent"] = pd.to_numeric(
        df["competitor_commercial_adjacent"], errors="coerce"
    ).fillna(0).astype(int)
    for col in ["competitor_total_ports", "competitor_ports_active", "competitor_level2_ports_active", "competitor_dc_fast_ports_active"]:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    if "competitor_total_ports" not in df.columns:
        df["competitor_total_ports"] = 0.0
    df["competitor_total_ports"] = pd.to_numeric(df["competitor_total_ports"], errors="coerce").fillna(0.0)

    if "competitor_open_yyyymm" not in df.columns:
        df["competitor_open_yyyymm"] = 0
    df["competitor_open_yyyymm"] = pd.to_numeric(df["competitor_open_yyyymm"], errors="coerce").fillna(0).astype(int)

    period_map = (
        df[["date_numeric_orig", "date_numeric"]]
        .drop_duplicates()
        .sort_values("date_numeric_orig")
        .set_index("date_numeric_orig")["date_numeric"]
        .to_dict()
    )
    ordered_periods = sorted(period_map)

    def map_competitor_period(yyyymm: int) -> int:
        if yyyymm == 0:
            return 0
        if yyyymm in period_map:
            return int(period_map[yyyymm])
        candidates = [period for period in ordered_periods if period >= yyyymm]
        return int(period_map[candidates[0]]) if candidates else 0

    df["competitor_first_treat_period"] = df["competitor_open_yyyymm"].apply(map_competitor_period)

    output_path = PROCESSED_DIR / "df_final_broad.csv"
    df.to_csv(output_path, index=False)
    logging.info("Saved final broad panel %s with shape %s", output_path, df.shape)


if __name__ == "__main__":
    main()
