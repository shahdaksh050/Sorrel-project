# Supported File Formats and Upload Limits

Version: 1.0.0  
Last Updated: 2026-10-12  
Canonical: http://localhost:8501/docs/file-formats.md  

Sorrel accepts standard tabular datasets for automated profiling and analysis. All uploaded files undergo pre-parse quarantine validation to ensure structural integrity and prevent malicious file exploitation.

---

## 1. Supported Formats

| Format | Extension | Media Type | Notes |
| :--- | :--- | :--- | :--- |
| **CSV** | `.csv` | `text/csv` | Delimiters: comma `,`, semicolon `;`, tab `\t`. UTF-8 preferred. |
| **Parquet** | `.parquet` | `application/vnd.apache.parquet` | Snappy or gzip compressed columnar data. Highly recommended for tables > 100k rows. |
| **JSON** | `.json` | `application/json` | Array of records `[{...}, {...}]` or JSON-lines format. |
| **TSV** | `.tsv`, `.tab` | `text/tab-separated-values` | Tab-delimited plain text tables. |

---

## 2. Size and Dimension Limits

- **Maximum Upload File Size**: 250 MB per file.
- **Maximum Row Count**: 1,000,000 rows (sub-sampled with stratified sampling for exploratory phases if exceeded).
- **Maximum Column Count**: 500 columns.
- **Minimum Data Requirements**: At least 5 observations (rows) and 2 columns.

---

## 3. Quarantine and Validation Boundary

Before files are handed to tabular parsers (`pandas` / `pyarrow`), the `validate_upload()` engine enforces:

1. **Magic Byte Verification**: File header signatures must match the declared file extension (e.g. Parquet files must begin with `PAR1`).
2. **Decompression Bomb Protection**: Compressed uploads (gzip, zip) enforce a strict decompression ratio ceiling (maximum 10:1) and maximum uncompressed size of 500 MB.
3. **Encoding Coercion**: Files are parsed with strict UTF-8 decoding, falling back to Latin-1/CP1252 if BOM or multi-byte errors are detected.
4. **Archive Integrity**: Multi-file archives containing sub-folders, symlinks, executables, or ambiguous multiple datasets are rejected with HTTP `422 Unprocessable Entity`.
