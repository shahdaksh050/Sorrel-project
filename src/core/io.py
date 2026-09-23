"""
Unified dataset reader — bytes to DataFrame, one code path.

Before this module, five near-identical `_read_df` functions were
duplicated across src/tools/*.py and src/core/controller.py, one of them
(controller's) not even knowing about .tsv. That divergence is exactly how
a semicolon- or tab-delimited export silently parses as a single column:
a confident, complete analysis of data that doesn't actually exist.

Detection chain: extension -> format, encoding, delimiter, header
validation. Every guess is recorded on the returned ReadReport instead of
being applied silently — the report is what item 6 (report restructure)
surfaces as "what was detected vs assumed at read time".

Pure I/O + parsing. No LLM calls, no profiling, no coercion (src.core.coercion).
"""
from __future__ import annotations

import contextlib
import csv
import gzip
import io
import json
import os
import re
import tempfile
import threading
import zipfile
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from src.core.sentinels import null_sentinels

#: Extensions read_any knows how to ingest. The single source of truth —
#: src.core.security.ALLOWED_EXTENSIONS and the Streamlit uploader's type
#: list both derive from this instead of maintaining their own copies.
#: ".gz"/".zip" are only meaningful paired with .csv/.tsv (e.g. "sales.csv.gz")
#: — see _detect_format — but have to be listed bare here too, since the
#: security/uploader layers gate on Path.suffix, which never sees the
#: compound form.
SUPPORTED_EXTENSIONS: frozenset[str] = frozenset({
    ".csv", ".tsv", ".xlsx", ".xls",
    ".json", ".jsonl", ".parquet",
    ".dta", ".sas7bdat", ".xpt", ".sav", ".zsav", ".feather",
    ".h5", ".hdf5", ".nc",
    ".gz", ".zip",
})

#: Hard cap on rows x columns-of-index-product an xarray/NetCDF dataset may
#: expand to when flattened with to_dataframe() — a gridded climate file can
#: be a few MB on disk and billions of cells once unrolled.
NETCDF_MAX_CELLS = 5_000_000

#: Delimiters considered when sniffing a .csv file's separator.
_CSV_DELIMITER_CANDIDATES = ",;\t|"

#: Bytes read from the front of a text file to sniff its delimiter/headers.
_SNIFF_SAMPLE_BYTES = 64 * 1024

#: pandas' own default `na_values` (pandas._libs.parsers.STR_NA_VALUES),
#: minus the literal string "None". pandas >= 2 treats "None" as NA by
#: default, but real datasets legitimately use it as a category label (e.g.
#: a Social_Media_Influence column of High/Medium/Low/None) — left as the
#: default, one real "None" in five drops out and the column reads as 25%
#: missing. "" is kept: an empty field is still NA. Derived from the pandas
#: constant when importable (keeps this in sync with pandas' own list);
#: hard-coded as a fallback in case that private module moves.
#: Numeric placeholders (-999, 9999, -200...) are deliberately NOT listed: a
#: blanket rule erases real values (an employee ID 9999). src.core.sentinels
#: detects them from the data instead and discloses every one it nulls.
try:
    from pandas._libs.parsers import STR_NA_VALUES as _PANDAS_STR_NA_VALUES
    NA_VALUES_KEEP_NONE_STRING: list[str] = sorted(
        (_PANDAS_STR_NA_VALUES - {"None"}) | {"?", " ? "}
    )
except ImportError:
    NA_VALUES_KEEP_NONE_STRING = [
        "", "#N/A", "#N/A N/A", "#NA", "-1.#IND", "-1.#QNAN", "-NaN", "-nan",
        "1.#IND", "1.#QNAN", "<NA>", "N/A", "NA", "NULL", "NaN", "n/a",
        "nan", "null", "?", " ? ",
    ]

#: Depth at which nested JSON fields stop being flattened into dotted column
#: names (address.city -> ...) and are left as dict/list cell values instead.
#: Uncapped flattening on adversarial or deeply-nested input is an easy way
#: to explode a handful of records into thousands of columns.
JSON_FLATTEN_MAX_DEPTH = 3

#: Hard ceiling on rows loaded into memory for analysis. One constant, one
#: application point (_cap_rows) — see get_max_rows(). The plan is to make
#: this scale (chunked/streaming readers) later; until then, raising the
#: limit is a config change (DSA_MAX_ROWS), not a hunt through call sites.
MAX_ROWS_DEFAULT = 1_000_000

_FORMAT_BY_SUFFIX = {
    ".csv": "csv", ".tsv": "tsv", ".xlsx": "xlsx", ".xls": "xls",
    ".json": "json", ".jsonl": "jsonl", ".parquet": "parquet",
    ".dta": "stata", ".sas7bdat": "sas", ".xpt": "sas", ".sav": "spss",
    ".zsav": "spss", ".feather": "feather", ".h5": "hdf5", ".hdf5": "hdf5",
    ".nc": "netcdf",
}

#: Compression sniffed off a compound extension (e.g. "sales.csv.gz"). Only
#: ever paired with .csv/.tsv today — see _detect_format.
_COMPRESSION_BY_SUFFIX = {".gz": "gzip", ".zip": "zip"}

#: Parsed frames, keyed on (resolved path, mtime_ns, size). Bounded because
#: the values are whole DataFrames — a 1M-row file is ~50MB resident.
_READ_CACHE_MAX_ENTRIES = 4
_READ_CACHE: OrderedDict[tuple[str, int, int], tuple[pd.DataFrame, ReadReport]] = (
    OrderedDict()
)
_READ_CACHE_LOCK = threading.Lock()  # guards _READ_CACHE; never held during a file read


def get_max_rows() -> int:
    """
    Active row cap for read_any: the DSA_MAX_ROWS env var if set (and
    parseable), else MAX_ROWS_DEFAULT. A function rather than a bare
    constant so ops can raise the ceiling on a bigger box with zero code
    changes.
    """
    try:
        return max(1, int(os.getenv("DSA_MAX_ROWS", str(MAX_ROWS_DEFAULT))))
    except ValueError:
        return MAX_ROWS_DEFAULT


@dataclass
class ReadReport:
    """What read_any actually did — surfaced, never assumed."""

    path: str
    format: str                    # csv | tsv | xlsx | xls | json | parquet | stata | sas | spss | feather | hdf5 | netcdf
    encoding: str
    encoding_confident: bool       # False when guessed via the cp1252 fallback
    delimiter: str | None
    delimiter_sniffed: bool        # False when it came from the extension
    duplicate_headers: list[str] = field(default_factory=list)
    flattened: bool = False            # True when nested JSON was flattened via json_normalize
    flattened_columns: int | None = None  # resulting column count, when flattened
    sampled: bool = False              # True when the frame exceeded get_max_rows()
    sampled_from: int | None = None    # original row count, when sampled
    sampled_to: int | None = None      # rows kept after sampling (== get_max_rows() at read time)
    notes: list[str] = field(default_factory=list)
    sentinels: list[dict[str, Any]] = field(default_factory=list)  # numeric placeholders nulled (src.core.sentinels)
    header_row_offset: int = 0
    subtotals_excluded: int = 0
    reshaped_from_wide: bool = False
    wide_time_vars: list[str] = field(default_factory=list)
    extra_tables: dict[str, Any] = field(default_factory=dict)


class DatasetReadError(Exception):
    """Raised when a dataset cannot be read at all: empty, corrupt, or an
    unsupported format. Never raised for data that merely looks unusual —
    that's the profiler's and coercion's job to flag, not the reader's."""


#: Within this much chaos of the top candidate, prefer cp1252 over an
#: equally-clean but far rarer codepage — charset_normalizer's tiebreak is a
#: generic multi-language coherence score that has no special preference for
#: cp1252 even though it's overwhelmingly the real-world encoding behind
#: "Excel European export" files (the exact case this detector exists for).
_CP1252_TIEBREAK_MARGIN = 0.05


def _detect_encoding(raw: bytes) -> tuple[str, bool, list[str]]:
    notes: list[str] = []
    if raw.startswith(b"\xef\xbb\xbf"):
        return "utf-8-sig", True, notes
    try:
        raw.decode("utf-8")
        return "utf-8", True, notes
    except UnicodeDecodeError:
        pass
    try:
        from charset_normalizer import from_bytes
        matches = from_bytes(raw)
        best = matches.best()
    except Exception:
        matches, best = None, None
    if best is not None and best.encoding:
        encoding = best.encoding
        if encoding != "cp1252" and matches is not None:
            for candidate in matches:
                if candidate.encoding == "cp1252" and candidate.chaos <= best.chaos + _CP1252_TIEBREAK_MARGIN:
                    encoding = "cp1252"
                    break
        notes.append(f"Encoding auto-detected as {encoding} (file is not valid UTF-8).")
        return encoding, True, notes
    notes.append(
        "Encoding could not be confidently detected; assumed cp1252. "
        "Re-export as UTF-8 if any characters look wrong."
    )
    return "cp1252", False, notes


def _sniff_delimiter(text_sample: str) -> tuple[str, bool]:
    try:
        dialect = csv.Sniffer().sniff(text_sample, delimiters=_CSV_DELIMITER_CANDIDATES)
        return dialect.delimiter, True
    except csv.Error:
        # Genuinely single-column data (or too small a sample) — csv.Sniffer
        # raises rather than returning a best guess. Comma is a safe default:
        # a single-column file has no delimiter to get wrong either way.
        return ",", False


def _find_duplicate_headers(text_sample: str, delimiter: str) -> list[str]:
    first_line = text_sample.splitlines()[0] if text_sample else ""
    try:
        names = next(csv.reader([first_line], delimiter=delimiter))
    except (csv.Error, StopIteration):
        return []
    seen: set[str] = set()
    dupes: list[str] = []
    for name in names:
        if name in seen and name not in dupes:
            dupes.append(name)
        seen.add(name)
    return dupes


def _decompress(raw: bytes, compression: str, filename: str) -> bytes:
    """Unwrap gzip/zip so the rest of _read_delimited never knows the
    input was compressed — same encoding/delimiter detection either way."""
    if compression == "gzip":
        try:
            return gzip.decompress(raw)
        except OSError as exc:
            raise DatasetReadError(f"'{filename}' could not be gunzipped: {exc}") from exc
    if compression == "zip":
        try:
            with zipfile.ZipFile(io.BytesIO(raw)) as zf:
                members = [n for n in zf.namelist() if not n.endswith("/")]
                if not members:
                    raise DatasetReadError(f"'{filename}' zip archive contains no files.")
                return zf.read(members[0])
        except zipfile.BadZipFile as exc:
            raise DatasetReadError(f"'{filename}' is not a valid zip archive: {exc}") from exc
    return raw


# ---------------------------------------------------------------------------
# Layout and Subtotal Detection (Phase 1, FutureScope 5.1)
# ---------------------------------------------------------------------------

_SUBTOTAL_EXACT_KEYWORDS = frozenset({
    "total", "subtotal", "sub-total", "grand total", "all", "overall",
    "totalen", "somme", "gesamt", "totale",
})

_YEAR_RE = re.compile(r"^(?:19|20)\d{2}$")
_YEAR_PREFIX_RE = re.compile(r"^[YyFf](?:19|20)\d{2}$")
_YEAR_MONTH_RE = re.compile(r"^(?:19|20)\d{2}[-_/](?:0[1-9]|1[0-2])$")
_MONTH_NAME_RE = re.compile(
    r"^(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
    r"(?:uary|ruary|ch|il|e|y|ust|ember|ober)?$",
    re.IGNORECASE,
)
_SEQ_STEP_RE = re.compile(r"^[TtQq]\d+$")


def _is_numeric_str(s: str) -> bool:
    try:
        float(s.replace(",", ""))
        return True
    except ValueError:
        return False


def _detect_header_offset(text: str, delimiter: str) -> tuple[int, int, list[str]]:
    """
    Detect if the real header row is preceded by title/metadata rows or blank lines,
    and if the file has trailing footnote lines.

    Returns:
        (header_offset, trailing_trim_count, notes)
    """
    lines = text.splitlines()
    if len(lines) <= 2:
        return 0, 0, []

    scan_limit = min(25, len(lines))
    parsed_lines: list[list[str]] = []
    for line in lines[:scan_limit]:
        if not line.strip():
            parsed_lines.append([])
            continue
        try:
            fields = next(csv.reader([line], delimiter=delimiter))
            parsed_lines.append([f.strip() for f in fields])
        except csv.Error:
            parsed_lines.append([])

    field_counts = [len(f) for f in parsed_lines if len(f) > 1]
    if not field_counts:
        return 0, 0, []

    from collections import Counter
    modal_count, _ = Counter(field_counts).most_common(1)[0]
    if modal_count < 2:
        return 0, 0, []

    header_offset = 0
    found_offset = False
    for i, fields in enumerate(parsed_lines):
        if len(fields) == modal_count:
            non_numeric = sum(1 for f in fields if f and not _is_numeric_str(f))
            if non_numeric >= max(1, modal_count // 2):
                header_offset = i
                found_offset = True
                break

    notes: list[str] = []
    if found_offset and header_offset > 0:
        notes.append(f"Header detected at row {header_offset + 1}; skipped {header_offset} title/metadata row(s).")

    trailing_trim = 0
    for j in range(len(lines) - 1, max(header_offset, len(lines) - 15), -1):
        line = lines[j].strip()
        if not line:
            trailing_trim += 1
            continue
        if any(line.startswith(prefix) for prefix in ("*", "Note", "Notes:", "Source:", "Footnote:", "Disclaimer:")):
            trailing_trim += 1
            continue
        try:
            fields = next(csv.reader([line], delimiter=delimiter))
            if len(fields) < modal_count:
                trailing_trim += 1
                continue
        except csv.Error:
            trailing_trim += 1
            continue
        break

    if trailing_trim > 0:
        notes.append(f"Trimmed {trailing_trim} trailing footnote/metadata line(s).")

    return header_offset, trailing_trim, notes


def detect_and_exclude_subtotals(df: pd.DataFrame) -> tuple[pd.DataFrame, list[int], list[str]]:
    """
    Identify rows that are subtotals or grand totals (e.g. 'North Total', 'Grand Total',
    or rows matching the sum of preceding rows) and exclude them from analysis to prevent double counting.
    """
    if df.empty or len(df) <= 2:
        return df, [], []

    # Vectorized, not a per-row `df.iterrows()` loop: iterrows() constructs a
    # new pandas Series per row, which turns this into the dominant cost of
    # every dataset load at a few hundred thousand rows (measured: 114s of a
    # 118s read on a 500k-row file, before this fix — every tool re-reading
    # the same file paid it again on a read-cache miss).
    text_cols = [c for c in df.columns if df[c].dtype == object or pd.api.types.is_string_dtype(df[c])]
    if text_cols:
        normalized = df[text_cols].astype(str).apply(lambda s: s.str.strip().str.lower())
        is_subtotal_row = (
            normalized.isin(_SUBTOTAL_EXACT_KEYWORDS)
            | normalized.apply(lambda s: s.str.endswith((" total", " subtotal")))
        ).any(axis=1)
        subtotal_indices: list[int] = [int(i) for i in df.index[is_subtotal_row]]
    else:
        subtotal_indices = []

    # Check if bottom row is a mathematical sum of the remaining rows
    if not (subtotal_indices and subtotal_indices[-1] == len(df) - 1):
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if len(numeric_cols) >= 2 and len(df) >= 4:
            last_row = df.iloc[-1]
            non_last = df.iloc[:-1]
            sums = non_last[numeric_cols].sum()
            matches = 0
            for c in numeric_cols:
                val = last_row[c]
                if pd.notna(val) and abs(val - sums[c]) < 1e-4:
                    matches += 1
            if matches == len(numeric_cols):
                if (len(df) - 1) not in subtotal_indices:
                    subtotal_indices.append(len(df) - 1)

    if not subtotal_indices:
        return df, [], []

    subtotal_indices = sorted(set(subtotal_indices))
    clean_df = df.drop(index=subtotal_indices).reset_index(drop=True)
    notes = [f"Excluded {len(subtotal_indices)} subtotal/total row(s) to prevent double-counting."]
    return clean_df, subtotal_indices, notes


def detect_wide_time_headers(df: pd.DataFrame) -> tuple[list[str], list[str]] | None:
    """
    Detect if the dataframe has columns representing time points (years, months, steps).
    Returns (id_vars, time_vars) if at least 3 time columns are found, else None.
    """
    if df.shape[1] < 4:
        return None

    time_cols: list[str] = []
    for col in df.columns:
        c_str = str(col).strip()
        if (
            _YEAR_RE.match(c_str)
            or _YEAR_PREFIX_RE.match(c_str)
            or _YEAR_MONTH_RE.match(c_str)
            or _MONTH_NAME_RE.match(c_str)
            or _SEQ_STEP_RE.match(c_str)
        ):
            time_cols.append(str(col))

    if len(time_cols) >= 3:
        id_cols = [str(c) for c in df.columns if str(c) not in time_cols]
        valid_numeric = 0
        for c in time_cols:
            if pd.api.types.is_numeric_dtype(df[c]):
                valid_numeric += 1
            else:
                try:
                    pd.to_numeric(df[c].dropna().head(10))
                    valid_numeric += 1
                except (ValueError, TypeError):
                    pass
        if valid_numeric >= len(time_cols) - 1:
            return id_cols, time_cols

    return None


def reshape_wide_to_long(
    df: pd.DataFrame,
    id_vars: list[str],
    time_vars: list[str],
    time_col: str = "time",
    value_col: str = "value",
) -> pd.DataFrame:
    """Reshape a wide time-in-header table into a long panel format."""
    long_df = pd.melt(
        df,
        id_vars=id_vars,
        value_vars=time_vars,
        var_name=time_col,
        value_name=value_col,
    )
    return long_df


def _read_delimited(
    path: Path, format_: str, compression: str | None = None
) -> tuple[pd.DataFrame, ReadReport]:
    raw = path.read_bytes()
    if not raw:
        raise DatasetReadError(f"'{path.name}' is empty — no data to read.")

    notes: list[str] = []
    if compression is not None:
        raw = _decompress(raw, compression, path.name)
        if not raw:
            raise DatasetReadError(f"'{path.name}' decompressed to no data.")
        notes.append(f"Decompressed from {compression} before parsing.")

    encoding, encoding_confident, encoding_notes = _detect_encoding(raw)
    notes.extend(encoding_notes)
    try:
        text = raw.decode(encoding)
    except UnicodeDecodeError as exc:
        raise DatasetReadError(
            f"'{path.name}' could not be decoded as {encoding}: {exc}"
        ) from exc
    sample = text[:_SNIFF_SAMPLE_BYTES]

    if format_ == "tsv":
        delimiter, delimiter_sniffed = "\t", False
    else:
        delimiter, delimiter_sniffed = _sniff_delimiter(sample)

    header_offset, trailing_trim, layout_notes = _detect_header_offset(text, delimiter)
    notes.extend(layout_notes)
    if header_offset > 0 or trailing_trim > 0:
        lines = text.splitlines()
        end_idx = len(lines) - trailing_trim if trailing_trim > 0 else len(lines)
        text = "\n".join(lines[header_offset:end_idx])
        sample = text[:_SNIFF_SAMPLE_BYTES]

    duplicate_headers = _find_duplicate_headers(sample, delimiter)
    if duplicate_headers:
        notes.append(
            f"Duplicate column name(s) {duplicate_headers} were disambiguated "
            "(e.g. 'a' -> 'a.1')."
        )

    try:
        df = pd.read_csv(
            io.StringIO(text),
            sep=delimiter,
            keep_default_na=False,
            na_values=NA_VALUES_KEEP_NONE_STRING,
        )
    except pd.errors.EmptyDataError as exc:
        raise DatasetReadError(f"'{path.name}' has no columns to parse.") from exc

    report = ReadReport(
        path=str(path),
        format=format_,
        encoding=encoding,
        encoding_confident=encoding_confident,
        delimiter=delimiter,
        delimiter_sniffed=delimiter_sniffed,
        duplicate_headers=duplicate_headers,
        notes=notes,
        header_row_offset=header_offset,
    )
    return df, report


def _read_excel(path: Path, format_: str) -> tuple[pd.DataFrame, ReadReport]:
    engine = "openpyxl" if format_ == "xlsx" else "xlrd"
    notes: list[str] = []
    extra_tables: dict[str, Any] = {}
    try:
        xl = pd.ExcelFile(path, engine=engine)
        sheet_names = xl.sheet_names
        if not sheet_names:
            raise DatasetReadError(f"'{path.name}' contains no sheets.")
        if len(sheet_names) == 1:
            df = xl.parse(sheet_names[0])
        else:
            sheets: dict[str, pd.DataFrame] = {}
            for s in sheet_names:
                try:
                    s_df = xl.parse(s)
                    if not s_df.empty:
                        sheets[s] = s_df
                except Exception:
                    continue
            if not sheets:
                raise DatasetReadError(f"'{path.name}' sheets are all empty.")

            cols_list = [set(map(str, s_df.columns)) for s_df in sheets.values()]
            first_cols = cols_list[0]
            matching_schemas = all(
                cols == first_cols or (len(cols & first_cols) / max(1, len(cols | first_cols)) >= 0.7)
                for cols in cols_list
            )
            if matching_schemas:
                concat_list: list[pd.DataFrame] = []
                for s_name, s_df in sheets.items():
                    s_copy = s_df.copy()
                    s_copy["_sheet_name"] = s_name
                    concat_list.append(s_copy)
                df = pd.concat(concat_list, ignore_index=True)
                notes.append(
                    f"Auto-concatenated {len(sheets)} sheets with matching schemas "
                    f"({', '.join(sheets.keys())}) adding '_sheet_name' column."
                )
            else:
                primary_name = max(sheets.keys(), key=lambda k: len(sheets[k]))
                df = sheets[primary_name]
                extra_tables = {k: v for k, v in sheets.items() if k != primary_name}
                notes.append(
                    f"Detected multi-tab Excel with {len(sheet_names)} distinct sheets "
                    f"({', '.join(sheet_names)}). Primary sheet '{primary_name}' loaded "
                    f"({len(df)} rows); {len(extra_tables)} additional tables preserved."
                )
    except DatasetReadError:
        raise
    except Exception as exc:
        raise DatasetReadError(f"'{path.name}' could not be read as {format_}: {exc}") from exc

    report = ReadReport(
        path=str(path),
        format=format_,
        encoding="n/a",
        encoding_confident=True,
        delimiter=None,
        delimiter_sniffed=False,
        notes=notes,
        extra_tables=extra_tables,
    )
    return df, report


def _is_nested(value: object) -> bool:
    return isinstance(value, (dict, list))


def _read_json(path: Path, format_: str) -> tuple[pd.DataFrame, ReadReport]:
    notes: list[str] = []
    flattened = False
    flattened_columns: int | None = None
    df: pd.DataFrame
    try:
        if format_ == "jsonl":
            df = pd.read_json(path, lines=True)
        else:
            with open(path, encoding="utf-8", errors="replace") as f:
                raw = json.load(f)
            if isinstance(raw, list):
                df = pd.json_normalize(raw, max_level=JSON_FLATTEN_MAX_DEPTH, sep=".")
                flattened = True
                flattened_columns = len(df.columns)
            elif isinstance(raw, dict):
                record_key = next((k for k in ("data", "items", "records", "results", "rows", "values") if isinstance(raw.get(k), list)), None)
                if record_key is not None and isinstance(raw[record_key], list):
                    meta_keys = [k for k in raw.keys() if k != record_key and not isinstance(raw[k], (list, dict))]
                    df = pd.json_normalize(
                        raw[record_key],
                        meta=meta_keys if meta_keys else None,
                        max_level=JSON_FLATTEN_MAX_DEPTH,
                        sep=".",
                    )
                    flattened = True
                    flattened_columns = len(df.columns)
                    notes.append(f"Unpacked nested records list from key '{record_key}' with {len(df)} records.")
                else:
                    df = pd.json_normalize(raw, max_level=JSON_FLATTEN_MAX_DEPTH, sep=".")
                    flattened = True
                    flattened_columns = len(df.columns)
            else:
                df = pd.read_json(path)
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        raise DatasetReadError(f"'{path.name}' could not be parsed as {format_}: {exc}") from exc
    if df.empty:
        raise DatasetReadError(f"'{path.name}' contains no records.")

    nested_cols = [c for c in df.columns if df[c].map(_is_nested).any()]
    if nested_cols and not flattened:
        cols_before = len(df.columns)
        df = pd.json_normalize(
            df.to_dict(orient="records"), max_level=JSON_FLATTEN_MAX_DEPTH, sep="."
        )
        flattened = True
        flattened_columns = len(df.columns)
        notes.append(
            f"Nested field(s) {nested_cols} were flattened into dotted column names "
            f"(e.g. 'address.city'), capped at depth {JSON_FLATTEN_MAX_DEPTH}; "
            f"{cols_before} -> {flattened_columns} columns."
        )

    report = ReadReport(
        path=str(path),
        format=format_,
        encoding="utf-8",
        encoding_confident=True,
        delimiter=None,
        delimiter_sniffed=False,
        flattened=flattened,
        flattened_columns=flattened_columns,
        notes=notes,
    )
    return df, report


def _read_parquet(path: Path) -> tuple[pd.DataFrame, ReadReport]:
    try:
        df = pd.read_parquet(path, engine="pyarrow")
    except Exception as exc:
        raise DatasetReadError(f"'{path.name}' could not be read as parquet: {exc}") from exc
    report = ReadReport(
        path=str(path),
        format="parquet",
        encoding="n/a",
        encoding_confident=True,
        delimiter=None,
        delimiter_sniffed=False,
    )
    return df, report


def _binary_report(path: Path, format_: str, notes: list[str] | None = None) -> ReadReport:
    return ReadReport(
        path=str(path),
        format=format_,
        encoding="n/a",
        encoding_confident=True,
        delimiter=None,
        delimiter_sniffed=False,
        notes=notes if notes is not None else [],
    )


def _read_hdf(path: Path) -> tuple[pd.DataFrame, ReadReport]:
    notes: list[str] = []
    try:
        with pd.HDFStore(str(path), mode="r") as store:
            keys = list(store.keys())
            if not keys:
                raise DatasetReadError(f"'{path.name}' contains no pandas-readable tables.")
            sizes = {k: int(getattr(store.get_storer(k), "nrows", 0) or 0) for k in keys}
            key = max(keys, key=lambda k: sizes[k])
            df = store.get(key)
    except ImportError as exc:
        raise DatasetReadError(
            f"'{path.name}': install pytables to read HDF5 files ({exc})"
        ) from exc
    except DatasetReadError:
        raise
    except Exception as exc:
        raise DatasetReadError(f"'{path.name}' could not be read as hdf5: {exc}") from exc
    if not isinstance(df, pd.DataFrame):
        df = df.to_frame()
    if len(keys) > 1:
        notes.append(f"HDF5 file has {len(keys)} tables; the largest ('{key}') was read.")
    return df, _binary_report(path, "hdf5", notes)


def _read_netcdf(path: Path) -> tuple[pd.DataFrame, ReadReport]:
    import math

    try:
        import xarray as xr
    except ImportError as exc:
        raise DatasetReadError(
            f"'{path.name}': install xarray and netCDF4 to read NetCDF files"
        ) from exc
    try:
        with xr.open_dataset(path) as ds:
            cells = math.prod(int(n) for n in ds.sizes.values())
            if cells > NETCDF_MAX_CELLS:
                raise DatasetReadError(
                    f"'{path.name}' expands to {cells:,} cells, over the "
                    f"{NETCDF_MAX_CELLS:,}-cell NetCDF cap. Subset it first "
                    "(fewer time steps / a smaller region)."
                )
            df = ds.to_dataframe().reset_index()
    except ImportError as exc:
        raise DatasetReadError(
            f"'{path.name}': install xarray and netCDF4 to read NetCDF files ({exc})"
        ) from exc
    except DatasetReadError:
        raise
    except Exception as exc:
        raise DatasetReadError(f"'{path.name}' could not be read as netcdf: {exc}") from exc
    return df, _binary_report(path, "netcdf")


def _read_statistical(path: Path, format_: str) -> tuple[pd.DataFrame, ReadReport]:
    """Stata / SAS / SPSS / Feather. Value labels stay as categoricals."""
    try:
        if format_ == "stata":
            df = pd.read_stata(path, convert_categoricals=True)
        elif format_ == "sas":
            is_xpt = path.suffix.lower() == ".xpt"
            df = pd.read_sas(
                path, format="xport" if is_xpt else "sas7bdat",
                encoding="latin-1" if is_xpt else "infer",
            )
        elif format_ == "spss":
            df = pd.read_spss(path, convert_categoricals=True)
        else:
            df = pd.read_feather(path)
    except ImportError as exc:
        if format_ == "spss":
            raise DatasetReadError(
                f"'{path.name}': install pyreadstat to read SPSS files"
            ) from exc
        raise DatasetReadError(f"'{path.name}' could not be read as {format_}: {exc}") from exc
    except Exception as exc:
        raise DatasetReadError(f"'{path.name}' could not be read as {format_}: {exc}") from exc
    return df, _binary_report(path, format_)


def _detect_format(path: Path) -> tuple[str | None, str | None]:
    """
    Resolve a path to (format, compression).

    format is one of the _FORMAT_BY_SUFFIX values, or None when neither the
    plain nor the compound extension (e.g. "sales.csv.gz") is recognised.
    compression is "gzip"/"zip" for a compressed .csv/.tsv, else None —
    that is the only combination supported today.
    """
    suffix = path.suffix.lower()
    format_ = _FORMAT_BY_SUFFIX.get(suffix)
    if format_ is not None:
        return format_, None
    if suffix in _COMPRESSION_BY_SUFFIX and len(path.suffixes) >= 2:
        inner_format = _FORMAT_BY_SUFFIX.get(path.suffixes[-2].lower())
        if inner_format in ("csv", "tsv"):
            return inner_format, _COMPRESSION_BY_SUFFIX[suffix]
    return None, None


def _cap_rows(df: pd.DataFrame, report: ReadReport) -> pd.DataFrame:
    """
    The one place get_max_rows() is consulted. Every _read_uncached path
    (delimited, excel, json, parquet) funnels through here before its
    result is cached or handed back — so raising the cap later is a config
    change (DSA_MAX_ROWS), not a hunt through call sites.

    Sampling, never truncation: a plain head(n) systematically keeps
    whatever a time-ordered or pre-sorted export puts first, which is
    exactly the silent bias this module exists to avoid. Every reader here
    already materialises the full frame before this runs, so a uniform
    `.sample()` without replacement *is* the distribution Algorithm-R
    reservoir sampling converges to — no separate streaming pass needed
    until the readers themselves become streaming.
    """
    cap = get_max_rows()
    total = len(df)
    if total <= cap:
        return df
    sampled = df.sample(n=cap, random_state=0).sort_index()
    report.sampled = True
    report.sampled_from = total
    report.sampled_to = cap
    report.notes.append(
        f"Dataset has {total:,} rows, over the {cap:,}-row analysis cap. "
        f"A random sample of {cap:,} rows (not the first {cap:,}) was used "
        "instead — set DSA_MAX_ROWS to raise the cap."
    )
    return sampled


def _drop_empty_columns(df: pd.DataFrame, report: ReadReport) -> pd.DataFrame:
    """Drop columns that are entirely null (e.g. the empty columns a trailing
    ';;' leaves behind); they carry no information and mislead target/type
    detection. Nothing is dropped from an empty frame or when every column is
    empty."""
    if df.empty:
        return df
    empty = [c for c in df.columns if bool(df[c].isna().all())]
    if not empty or len(empty) == df.shape[1]:
        return df
    report.notes.append(f"Dropped {len(empty)} empty column(s): {', '.join(str(c) for c in empty)}.")
    return df.drop(columns=empty)


def _read_uncached(file_path: str) -> tuple[pd.DataFrame, ReadReport]:
    path = Path(file_path)
    format_, compression = _detect_format(path)
    if format_ is None:
        attempted = (
            "".join(path.suffixes[-2:])
            if len(path.suffixes) >= 2 and path.suffix.lower() in _COMPRESSION_BY_SUFFIX
            else (path.suffix or "(none)")
        )
        allowed = ", ".join(sorted(SUPPORTED_EXTENSIONS - set(_COMPRESSION_BY_SUFFIX)))
        raise DatasetReadError(
            f"Unsupported file extension '{attempted}'. Supported: {allowed} "
            "(.csv/.tsv may also be gzip- or zip-compressed, e.g. 'sales.csv.gz')."
        )
    if format_ in ("csv", "tsv"):
        df, report = _read_delimited(path, format_, compression=compression)
    elif format_ in ("json", "jsonl"):
        df, report = _read_json(path, format_)
    elif format_ == "parquet":
        df, report = _read_parquet(path)
    elif format_ == "hdf5":
        df, report = _read_hdf(path)
    elif format_ == "netcdf":
        df, report = _read_netcdf(path)
    elif format_ in ("stata", "sas", "spss", "feather"):
        df, report = _read_statistical(path, format_)
    else:
        df, report = _read_excel(path, format_)
    df = _drop_empty_columns(df, report)
    df, subtotal_indices, subtotal_notes = detect_and_exclude_subtotals(df)
    report.subtotals_excluded = len(subtotal_indices)
    report.notes.extend(subtotal_notes)

    wide_spec = detect_wide_time_headers(df)
    if wide_spec is not None:
        id_vars, time_vars = wide_spec
        report.wide_time_vars = [str(c) for c in time_vars]
        if id_vars:
            df = reshape_wide_to_long(df, id_vars, time_vars)
            report.reshaped_from_wide = True
            report.notes.append(
                f"Reshaped wide table with {len(time_vars)} time columns ({time_vars[0]}..{time_vars[-1]}) "
                "to long panel format."
            )

    df = _cap_rows(df, report)
    df, report.sentinels = null_sentinels(df)
    report.notes.extend(rec["note"] for rec in report.sentinels)
    return df, report


def _cache_key(file_path: str) -> tuple[str, int, int] | None:
    """
    Identity of a file's *content*: resolved path + mtime + size.

    Same key shape AgentController._step_cache already uses, so an in-place
    edit invalidates rather than serving a stale frame. None when the file
    cannot be stat'd — the caller then reads uncached and raises the real
    error.
    """
    try:
        stat = Path(file_path).stat()
    except OSError:
        return None
    return (str(Path(file_path).resolve()), stat.st_mtime_ns, stat.st_size)


def clear_read_cache() -> None:
    """Drop every cached frame. For tests and long-lived processes."""
    with _READ_CACHE_LOCK:
        _READ_CACHE.clear()


def invalidate_read_cache(file_path: str) -> None:
    """
    Drop any cached frame for `file_path`. **Call this after writing a
    dataset to a path that may already have been read.**

    The (path, mtime, size) key cannot be trusted on its own for a rewrite:
    Windows `st_mtime_ns` has ~10-15ms granularity despite the name, so a
    same-size rewrite inside one tick keeps the old key and would serve the
    previous frame. That is a silently wrong analysis rather than a visible
    error — the exact failure class this module exists to remove — so the
    writers invalidate explicitly instead of relying on the timestamp.
    """
    try:
        resolved = str(Path(file_path).resolve())
    except OSError:
        return
    with _READ_CACHE_LOCK:
        for key in [k for k in _READ_CACHE if k[0] == resolved]:
            del _READ_CACHE[key]


def read_any(file_path: str) -> tuple[pd.DataFrame, ReadReport]:
    """
    Read a dataset from disk, detecting format/encoding/delimiter.

    Results are cached on (resolved path, mtime, size): a nine-tool run over
    one dataset used to pay the parse nine times, since every tool reads
    independently. Callers get a defensive copy of the frame — tools mutate
    what they read, and a shared frame would let one tool's cleaning leak
    into another's input.

    Raises:
        DatasetReadError: file is unsupported, empty, or unreadable as
            claimed (corrupt Excel container, undecodable text, ...).
    """
    key = _cache_key(file_path)
    if key is not None:
        with _READ_CACHE_LOCK:
            hit = _READ_CACHE.get(key)
            if hit is not None:
                _READ_CACHE.move_to_end(key)
        if hit is not None:
            df, report = hit
            return df.copy(), report

    df, report = _read_uncached(file_path)

    if key is not None:
        entry = (df.copy(), report)
        with _READ_CACHE_LOCK:
            _READ_CACHE[key] = entry
            # Frames are large; keep only the few most recent. A single run
            # touches one dataset plus its cleaned copy, so this is ample.
            while len(_READ_CACHE) > _READ_CACHE_MAX_ENTRIES:
                _READ_CACHE.popitem(last=False)
    return df, report


def read_any_bytes(raw: bytes, filename: str) -> tuple[pd.DataFrame, ReadReport]:
    """
    read_any for an in-memory upload.

    Streamlit hands the uploader raw bytes rather than a path, and app.py
    previously previewed those with a bare `pd.read_csv` — a sixth reader
    that knew nothing about delimiters or encodings. The result was a
    preview showing a single mangled column for the exact
    semicolon/cp1252 exports read_any exists to handle, while the analysis
    behind it was correct. Spilling to a temp file is what lets the one
    detection chain serve both.
    """
    name = Path(filename)
    suffix = name.suffix.lower()
    if suffix in _COMPRESSION_BY_SUFFIX and len(name.suffixes) >= 2:
        # Keep the compound extension (e.g. ".csv.gz") on the temp file too —
        # _detect_format needs both parts, and Path.suffix alone only ever
        # sees the last one.
        suffix = "".join(name.suffixes[-2:]).lower()
    handle = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    try:
        handle.write(raw)
        handle.close()
        # _read_uncached, not read_any: the temp path is deleted
        # immediately, so caching it would hold a whole DataFrame under
        # a key nothing can hit again and evict the real dataset.
        df, report = _read_uncached(handle.name)
    finally:
        with contextlib.suppress(OSError):
            os.unlink(handle.name)
    # Report the user's filename, not the throwaway temp path.
    report.path = filename
    return df, report
