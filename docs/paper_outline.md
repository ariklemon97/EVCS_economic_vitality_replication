## Paper Outline

Use `output/tables/` as the numerical source of truth and `paper/tables/` for
APA-style manuscript tables. The structure below follows a concise applied
econometrics format: question, data, estimator, results, robustness, conclusion.

### 1. Introduction

- Research question: do EV charging stations affect nearby business activity?
- Contribution: replicate direct EVCS spillover estimates and extend the design
  to same-sector spatial competition.
- Preview findings:
  - narrow TWFE estimates are positive for all POIs;
  - broad stacked own-port estimates are negative;
  - spatial competition estimates show robust negative DC fast competitor
    exposure, especially for restaurants;
  - pretrend/placebo diagnostics require cautious causal language.

### 2. Data and Variables

- Data: Dewey spend/traffic measures, NREL AFDC EVCS records, Census/ACS
  covariates, and built-environment controls.
- Sample: consumer-facing target NAICS sectors in a local-business context.
- Windows:
  - narrow: 2019 and January 2021-June 2023 outcomes, with Period 2 EVCS treatment starting in February 2021;
  - broad: 2019 and January 2021-January 2026 outcomes, with Period 2 EVCS treatment starting in February 2021.
- Direct exposure: active EVCS ports within 500m, split into all ports, Level 2,
  and DC fast ports.
- Spatial competition exposure: active charger ports at nearby same-sector
  competitors; main radius is 1000m.

### 3. Econometric Specifications

- Narrow replication: TWFE with POI and county-month fixed effects; clustered by
  POI.
- Broad timing diagnostic: Callaway-Sant'Anna estimator with not-yet-treated
  controls.
- Preferred broad and spatial models: stacked near-treatment regressions with
  cohort stacks, clean controls, 6-month pre-window, 12-month post-window,
  stack-specific POI and county-month fixed effects, and a 10,000-control cap.
- Interpretation: emphasize robust associations; note that significant
  pretrend/placebo terms limit strong causal claims.

### 4. Narrow Replication Results

- Present main TWFE estimates: positive and significant for all POIs; not
  significant for disadvantaged-only samples.
- Summarize distance, charger-type, income, and other-outcome patterns.
- Report monetary interpretation.

Include:

- Table 1: `paper/tables/table_01_narrow_twfe_main.md`.
- Table 2: `paper/tables/table_02_narrow_monetary.md`.
- Figures:
  - `output/figures/narrow/02_distance_Period1_2019_All.pdf`
  - `output/figures/narrow/02_distance_Period2_2021_2023_All.pdf`
  - `output/figures/narrow/04_income_Period1_2019_All.pdf`
  - `output/figures/narrow/04_income_Period2_2021_2023_All.pdf`

### 5. Broad Replication Results

- Report CS timing estimates as imprecise broad-window diagnostics.
- Present stacked own-port estimates as the preferred broad specification:
  negative all-port effects and stronger negative DC fast effects.
- Explain why these results differ from the narrow TWFE estimates: longer
  window, staggered timing, and different treatment estimands.

Include:

- Table 3: `paper/tables/table_03_broad_cs_stacked.md`.
- Figures:
  - `output/figures/broad/cs_event_study_All_POIs_lcus.pdf`
  - `output/figures/broad/cs_event_study_All_POIs_lspend.pdf`

### 6. Spatial Competition Extension

- Present the design: focal POIs are non-EVCS businesses; treatment is charger
  exposure at nearby same-sector competitors.
- Main result: all competitor ports are near zero at 1000m, while DC fast
  competitor ports are negative and significant for customers and spending.
- Monetary result: DC fast competitor exposure implies about `$236` lower
  annual spending per affected competitor POI and about `$9.46M` annually across
  treated competitor POIs in the stacked spatial sample.
- Heterogeneity: restaurant DC fast effects are the clearest subgroup result;
  grocery all-port effects are weaker/borderline.

Include:

- Table 4: `paper/tables/table_04_spatial_stacked_main.md`.
- Table 5: `paper/tables/table_05_spatial_monetary.md`.
- Table 7: `paper/tables/table_07_spatial_poi_type.md`.

### 7. Robustness and Diagnostics

- Summarize stacked robustness checks:
  - event-window sensitivity;
  - control-cap sensitivity;
  - spatial-radius sensitivity;
  - POI-type heterogeneity;
  - pretrend/placebo diagnostics.
- Robust patterns:
  - broad own-port negative effects;
  - spatial DC fast competitor negative effects;
  - restaurant DC fast subgroup effects.
- Caution: pretrend/placebo checks are significant, so results should be framed
  as robust associations rather than definitive causal effects.

Include:

- Table 6: `paper/tables/table_06_spatial_radius_robustness.md`.
- Table 8: `paper/tables/table_08_stacked_pretrend_placebo.md`.
- Source robustness tables in `output/tables/robustness/`.

### 8. Conclusion

- Narrow TWFE replication supports positive local associations in the original
  windows.
- Broader staggered/stacked estimates change the interpretation, producing
  negative own-port effects.
- Spatial competition evidence points to modest but economically meaningful
  losses from DC fast competitor exposure.
- Close with the replication contribution and the need for cautious causal
  language given pretrend diagnostics.

### Appendix

- Data access and reproducibility: `docs/data_access.md`,
  `docs/reproduction.md`.
- Model hierarchy: `docs/model_architecture.md`.
- Deferred improvements: `docs/future_work.md`.
- Detail tables:
  - narrow distance, charger-type, income, and other outcomes;
  - stacked window and control-cap robustness;
  - stacked spatial radius and POI-type robustness;
  - exploratory broad/spatial intensity diagnostics.
