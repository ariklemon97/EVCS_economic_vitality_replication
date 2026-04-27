## EV Charging Stations and Local Economic Vitality: Replication Package

This repository contains the final Python replication package for the study of electric vehicle charging stations and nearby business outcomes. The package is organized for a GitHub submission that excludes proprietary data while retaining the full code, generated tables, figures, and documentation needed to reproduce the analyses once the underlying data are obtained.

The package follows the software and code expectations in `desai-2013-editorial-marketing-science-replication-and-disclosure-policy (1).pdf`: the estimation code is included, the programming language and package requirements are documented, the workflow is reproducible from a single Python entrypoint, and proprietary datasets are replaced by clear acquisition notes rather than being redistributed.

### Final analytical scope

The final package keeps only the analyses intended for the paper:

- Narrow replication: original TWFE-style specification for business-adjacent EVCS exposure.
- Broad replication: Callaway-Sant'Anna staggered-adoption estimators and stacked near-treatment regressions.
- Spatial competition extension: stacked near-treatment regressions using competitor exposure from business-adjacent EVCS locations.

TWFE intensity models, negative-effect diagnostics, and subgroup exploratory
tables are retained as appendix/diagnostic material only. They are not the
preferred broad-replication or spatial-competition estimates.

The final package excludes the PDI extension from paper-facing outputs and from
the GitHub-ready code surface.

### Study windows

- Narrow replication: January 2019 to December 2019 and February 2021 to June 2023.
- Broad replication: January 2019 to December 2019 and February 2021 to January 2026.

### Estimand

The paper-facing estimand is local business spillovers from EV charging stations
that are near consumer-facing POIs. The models do not estimate the average
effect of every public EV installation in California. Charger exposure is
constructed only from EVCS-POI matches around target business categories
(`72`, `44`, `45`, `71`), and downstream broad/spatial models preserve explicit
commercial-adjacent flags. Control POIs are also restricted to the same local
business context: target-category POIs with at least one other target-category
business nearby.

### Repository layout

- `code/`: Python replication scripts.
- `docs/`: data access, reproduction notes, model architecture, and paper outline.
- `paper/tables/`: APA-style manuscript table drafts rendered from current CSV outputs.
- `output/tables/`: final paper-facing tables.
- `output/figures/`: final paper-facing figures.
- `data/raw/`: local staging area for proprietary raw data. Not tracked in Git.
- `data/processed/`: local intermediate files. Not tracked in Git.

### Main entrypoint

Run the full retained Python pipeline from the repository root:

```bash
python code/run_replication_package.py
```

This executes:

1. POI and EVCS spatial assignment
2. Dewey spend aggregation
3. Dewey foot-traffic aggregation if raw files are present locally
4. Covariate construction
5. Regression-panel assembly
6. Propensity-score matching
7. Spatial competition construction
8. Final broad-panel assembly
9. Narrow replication
10. Broad replication and spatial competition extension

### Data access

No proprietary datasets are included in this repository.

See:

- [docs/data_access.md](docs/data_access.md)
- [docs/reproduction.md](docs/reproduction.md)
- [docs/model_architecture.md](docs/model_architecture.md)
- [docs/paper_outline.md](docs/paper_outline.md)
- [docs/github_upload_checklist.md](docs/github_upload_checklist.md)

### Environment

Install the Python dependencies with:

```bash
python -m pip install -r requirements.txt
```

### Notes

- The Dewey foot-traffic workflow is implemented in Python, but raw weekly foot-traffic files must be downloaded locally before the other-outcomes analysis can recover dwell-time and distance-from-home measures.
- Propensity-score matched panels are written with explicit window names (`df_psm_narrow_*`, `df_psm_broad_*`) so narrow and broad analyses cannot accidentally reuse the wrong matched sample.
- Main stacked outputs are written to `output/tables/main/`; legacy broad outputs remain under `output/tables/broad/` for compatibility.
- Appendix diagnostics are written to `output/tables/diagnostics/` or explicitly labeled broad diagnostic files.
- Stacked robustness checks are implemented in `code/03_broad_replication/run_stacked_robustness.py` and write to `output/tables/robustness/`; they are not run by the main entrypoint.
- APA-style table drafts can be regenerated with `python code/04_reporting/render_apa_tables.py`.
- Summary PNGs in `output/figures/summary/` predate the final stacked results and should be regenerated before manuscript submission if they are needed.
- Raw and processed proprietary data should remain outside version control.
