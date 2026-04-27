## Future Work

These items are intentionally deferred until after the GitHub upload. They are
not blockers for shipping the current replication package.

### Stacked Pretrend Controls

The stacked robustness suite finds robust negative DC fast effects, but the
pretrend/placebo checks also flag significant pre-period terms for both broad
own-port and spatial competition models. A later revision should test whether
the estimates are stable after adding stronger controls for pre-treatment
variation.

Recommended follow-up checks:

1. Add stack-by-sector-by-month fixed effects, such as
   `stack_id x naics_sector x date_numeric`, so comparisons are made within
   sector-month cells.
2. Add pre-treatment outcome controls, including baseline spend/customers and
   pre-period growth rates.
3. Rebuild matched controls using pre-treatment trajectories, not only level
   covariates.
4. Estimate stacked event-study coefficients and report joint tests of
   pre-treatment event-time indicators.
5. Run cohort leave-one-out and drop-problem-cohort sensitivity checks.
6. Compare shorter symmetric event windows, especially `pre6/post6`, with the
   current preferred `pre6/post12` specification.

### Reporting Extensions

- Regenerate summary figures from current tables before any final manuscript
  submission. The PNG summary report in `output/figures/summary/` predates the
  final stacked results.
- If needed for the manuscript, implement stacked income heterogeneity and
  stacked monetary-impact tables for broad own-port effects. Current preferred
  monetary interpretation for the extension is computed from stacked spatial
  coefficients.
