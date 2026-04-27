## Model Architecture

This package separates paper-facing analyses from appendix diagnostics.

### Main Results

| Component | Preferred model | Main scripts | Main outputs |
|---|---|---|---|
| Original / narrow replication | TWFE | `code/02_narrow_replication/run_narrow_replication.py` | `output/tables/narrow/` |
| Broad replication | Callaway-Sant'Anna + stacked regression | `code/03_broad_replication/run_broad_replication.py`; `code/03_broad_replication/run_stacked_regression.py --target broad` | `output/tables/broad/cs_*`; `output/tables/main/broad_stacked_own_port_results.csv` |
| Spatial competition extension | Stacked regression | `code/03_broad_replication/run_stacked_regression.py --target spatial` | `output/tables/main/spatial_competition_stacked_results.csv` |

The top-level entrypoint runs all main analyses:

```bash
python code/run_replication_package.py
```

### Appendix / Diagnostics

| Component | Role | Scripts | Outputs |
|---|---|---|---|
| TWFE intensity models | Exploratory comparability checks, not preferred broad/spatial estimates | `code/03_broad_replication/run_intensity_models.py` | `output/tables/broad/*intensity*.csv` |
| Spatial CS robustness | Event-study and robustness diagnostics for spatial competition timing | `code/03_broad_replication/run_spatial_robustness.py` | `output/tables/broad/spatial_competition_robustness_comparison.csv` |
| Stacked robustness | Window, control-cap, pretrend/placebo, spatial-radius, and POI-type checks for stacked models | `code/03_broad_replication/run_stacked_robustness.py` | `output/tables/robustness/` |
| Negative-effect diagnostics | Pretrend and concentration summaries | `code/03_broad_replication/diagnose_negative_effects.py` | `output/tables/diagnostics/` |
| Model comparison tables | Side-by-side audit of exploratory and preferred estimates | `code/03_broad_replication/compare_corrected_models.py` | `output/tables/broad/corrected_model_side_by_side*.csv` |

Appendix diagnostics are intentionally not called by the top-level package
entrypoint. Run them explicitly when needed.

### Interpretation Rules

- Main text should not mix TWFE intensity estimates into the broad or spatial
  competition interpretation unless they are explicitly labeled as exploratory.
- Broad replication results should be described as CS timing estimates and
  stacked near-treatment estimates.
- Spatial competition results should be described using the stacked competition
  estimates, where the treated unit is a non-EVCS POI whose nearby same-sector
  competitor gains charger exposure.
- Diagnostic tables can explain sign patterns, pretrends, and subgroup
  concentration, but they should not replace the preferred model hierarchy.
- Spatial competition radius robustness should use radii greater than 500m
  because 500m is the direct EVCS-proximity treatment radius.
