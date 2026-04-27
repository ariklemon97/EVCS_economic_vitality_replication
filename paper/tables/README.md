## APA-Style Table Drafts

These tables are rendered from the current CSV outputs by:

```bash
python code/04_reporting/render_apa_tables.py
```

Each table is available as Markdown (`.md`) for quick drafting and LaTeX
(`.tex`) for manuscript integration.

| Table | Topic | Source CSV |
|---|---|---|
| Table 1 | Narrow TWFE main effects | `output/tables/narrow/01_main_model.csv` |
| Table 2 | Narrow monetary interpretation | `output/tables/narrow/06_monetary_impacts.csv` |
| Table 3 | Broad CS and stacked own-port estimates | `output/tables/broad/corrected_cs_main_summary.csv`; `output/tables/main/broad_stacked_own_port_results.csv` |
| Table 4 | Spatial competition stacked estimates | `output/tables/main/spatial_competition_stacked_results.csv` |
| Table 5 | Spatial competition monetary interpretation | `output/tables/main/spatial_stacked_monetary_impacts.csv` |
| Table 6 | Spatial radius robustness | `output/tables/robustness/stacked_spatial_distance_sensitivity.csv` |
| Table 7 | Spatial POI-type heterogeneity | `output/tables/robustness/stacked_spatial_poi_type_heterogeneity.csv` |
| Table 8 | Stacked pretrend/placebo diagnostics | `output/tables/robustness/stacked_pretrend_placebo.csv` |
