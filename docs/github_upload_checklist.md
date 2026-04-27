## Essential GitHub Package

The `.gitignore` uses an allowlist. The final public package should include
only the files below.

### Root

- `README.md`
- `requirements.txt`
- `.env.example`
- `.gitignore`

### Code

- Python scripts under `code/`
- No Slurm scripts, bytecode, notebooks, or archived R scripts

### Documentation

- `docs/data_access.md`
- `docs/dewey_file_manifest.csv`
- `docs/dewey_schemas.md`
- `docs/future_work.md`
- `docs/github_upload_checklist.md`
- `docs/model_architecture.md`
- `docs/paper_outline.md`
- `docs/reproduction.md`

### Manuscript Tables

- All files in `paper/tables/`

### Current Output Tables

- `output/tables/narrow/*.csv`
- `output/tables/main/*.csv`
- `output/tables/broad/corrected_cs_main_summary.csv`
- `output/tables/broad/cs_all_results.csv`
- `output/tables/robustness/*.csv`

### Current Figures

- `output/figures/narrow/*.pdf`
- `output/figures/broad/cs_event_study_All_POIs_lcus.pdf`
- `output/figures/broad/cs_event_study_All_POIs_lspend.pdf`

### Excluded

- Raw, intermediate, and processed data
- Logs and Slurm outputs
- `.env`, `.codex`, and `.archive`
- Stale summary PNGs
- Superseded broad heterogeneity PDFs and CSVs
- Internal audit notes and old R/Rmd files
- Nonessential PDFs
