## Reproduction Guide

### Scope

The final package reproduces:

- Narrow replication with TWFE-style estimators
- Broad replication with Callaway-Sant'Anna staggered-adoption estimators and stacked near-treatment regressions
- Spatial competition extension with stacked near-treatment regressions

The study design focuses on EVCS installations that are spatially adjacent to
consumer-facing POIs. It does not estimate effects for all public EV
installations regardless of commercial context. Treated and control POIs are
both restricted to target-category businesses in a local-business context,
defined as having at least one other target-category business within 500m.

The PDI extension is intentionally excluded from the final GitHub package.

### Environment

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

### End-to-end run

From the repository root:

```bash
python code/run_replication_package.py
```

### Stage-level runners

If you need to run stages individually:

- `python code/01_data_processing/01_spatial_join_pois_evcs.py`
- `python code/01_data_processing/02_clean_and_aggregate_spend.py`
- `python code/01_data_processing/02b_clean_and_aggregate_foot_traffic.py`
- `python code/01_data_processing/03_incorporate_covariates.py`
- `python code/01_data_processing/04_compile_regression_panel.py`
- `python code/01_data_processing/05_propensity_score_matching.py --window all`
- `python code/03_broad_replication/04_spatial_competition.py`
- `python code/03_broad_replication/06_finalize_panel.py`
- `python code/02_narrow_replication/run_narrow_replication.py`
- `python code/03_broad_replication/run_broad_replication.py --section main`
- `python code/03_broad_replication/run_stacked_regression.py --target all --pre 6 --post 12 --min-treated 25 --max-control-pois 10000 --seed 20260425`

Appendix-only diagnostics can be run explicitly:

- `python code/03_broad_replication/run_intensity_models.py --section all`
- `python code/03_broad_replication/run_broad_replication.py --section all`
- `python code/03_broad_replication/run_spatial_robustness.py`
- `python code/03_broad_replication/diagnose_negative_effects.py`
- `python code/03_broad_replication/run_stacked_robustness.py --check all --spatial-radii-m 1000 1500 2000`

Before running spatial-radius robustness for radii above 1km, build the
corresponding competitor-match files:

```bash
python code/03_broad_replication/04_spatial_competition.py --radius-m 1500 --output-suffix r1500
python code/03_broad_replication/04_spatial_competition.py --radius-m 2000 --output-suffix r2000
```

Do not use 500m for spatial competition radius robustness. The 500m radius is
the direct EVCS-proximity treatment radius, so a 500m competition radius would
blur the treatment and control definitions.

### Key outputs

- Narrow tables: `output/tables/narrow/`
- Broad tables: `output/tables/broad/`
- Main stacked tables: `output/tables/main/`
- Diagnostic tables: `output/tables/diagnostics/`
- Robustness tables: `output/tables/robustness/`
- Narrow figures: `output/figures/narrow/`
- Broad figures: `output/figures/broad/`

### Model choices in the final package

- Narrow replication preserves the original TWFE-style design and keeps the original 2021-2023 narrow window for business-adjacent EVCS exposure.
- Broad replication extends the second study window through January 2026 and uses Python Callaway-Sant'Anna estimators plus stacked near-treatment regressions for business-adjacent EVCS exposure.
- Broad CS estimators use a not-yet-treated comparison group and drop cohorts first treated in the final observed period so ATT(g,t) cells have post-treatment support.
- Matched PSM panels are window-scoped: `df_psm_narrow_*` feeds the narrow replication and `df_psm_broad_*` feeds the broad CS models.
- Spatial competition models use stacked near-treatment regressions where the focal unit is a non-EVCS POI and treatment is charger exposure at a nearby same-sector competitor.
- TWFE intensity models are retained as appendix diagnostics only. They are not the preferred broad-replication or spatial-competition estimates.
- Control POIs are filtered to the same local-business context as treated POIs before matching and broad/spatial estimation.

### Replication notes

- The codebase is intentionally Python-only in the final package.
- Generated paper-facing tables and figures are kept.
- Proprietary raw and processed data are excluded from version control.
