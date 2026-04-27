## Data Access

This repository does not redistribute proprietary or restricted datasets. To reproduce the analyses, the user must obtain access to the data sources below and place them in the local `data/raw/` structure expected by the Python pipeline.

### Required data sources

- U.S. Department of Energy Alternative Fuels Data Center (AFDC)
  Purpose: EV charging station locations, opening timing, and charger-port counts.

- Dewey / SafeGraph Global Places
  Purpose: POI locations, POI categories, identifiers, and opening or closing metadata.

- Dewey / SafeGraph Spend Patterns
  Purpose: monthly customer counts, spending, and customer income-bucket distributions.

- Dewey / Advan Foot Traffic
  Purpose: median dwell time and median distance-from-home measures used in other-outcomes analysis.

- American Community Survey (ACS)
  Purpose: tract-level socio-demographic covariates.

- California Energy Commission
  Purpose: county EV adoption and disadvantaged-community context.

- EPA Smart Location Database
  Purpose: built-environment and walkability covariates.

- OpenStreetMap
  Purpose: supplementary built-environment context where needed.

### Local raw-data structure

The current Python pipeline expects files under:

- `data/raw/dewey/global_places/`
- `data/raw/dewey/spend_patterns/`
- `data/raw/dewey/foot_traffic/`
- `data/raw/california_energy_commission/`
- `data/raw/census_acs/`
- `data/raw/epa_smart_location/`
- `data/raw/openstreetmap/`
- `data/raw/nrel_afdc/`

### Foot-traffic note

The foot-traffic feed is not bundled here. The downloader is:

- [code/00_data_download/01_download_dewey_foot_traffic.py](../code/00_data_download/01_download_dewey_foot_traffic.py)

This workflow aggregates weekly Dewey foot-traffic files to monthly POI-level measures after joining through Dewey global-places identifiers.

### Data policy

- Do not commit proprietary raw data.
- Do not commit processed panels derived from proprietary vendor data.
- Keep only code, generated paper-facing outputs, and documentation in the public package.

### What you can include without shipping the Dewey data

If you want the public package to be more informative without adding large proprietary files, the safest options are lightweight metadata artifacts rather than the data themselves:

- File manifests: one CSV per Dewey source listing file names, partition dates, row counts, and file sizes.
- Schema snapshots: column names, data types, and a short field description for each Dewey feed.
- Access instructions: exact API endpoint patterns, expected local folder structure, and the downloader commands.
- Small synthetic examples: a toy CSV or parquet with fake values but the same columns as the real feeds.
- Redacted summary stats: counts of files, date coverage, and non-confidential aggregate diagnostics.

For a Marketing Science replication package, those metadata files are usually the right compromise: they document the proprietary inputs clearly, keep the repository small, and avoid redistributing licensed data.

This repository now includes lightweight Dewey metadata artifacts:

- [docs/dewey_file_manifest.csv](dewey_file_manifest.csv)
- [docs/dewey_schemas.md](dewey_schemas.md)
