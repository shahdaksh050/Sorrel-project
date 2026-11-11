# Supported Local File Formats

Version: 1.0.1
Last Updated: 2026-10-16
Canonical: /docs/file-formats.md

Sorrel reads local tabular files through the Streamlit workspace or CLI. Common
supported formats include CSV/TSV, Excel (`.xlsx`, `.xls`), JSON/JSON Lines,
Parquet, Feather, HDF5, NetCDF, Stata, SAS, SPSS, and Arrow/ORC where the local
environment has the corresponding reader dependency.

The local reader limits input to 1,000,000 rows by default (`DSA_MAX_ROWS`).
The UI has separately configurable upload validation limits. Format support and
reader availability can vary with the installed Python dependencies.

There is no remote upload endpoint, presigned upload slot, HTTP error contract,
or service-side dataset retention policy in this release.
