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
import os
import tempfile
import zipfile
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

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
    ".gz", ".zip",
})

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
try:
    from pandas._libs.parsers import STR_NA_VALUES as _PANDAS_STR_NA_VALUES
    NA_VALUES_KEEP_NONE_STRING: list[str] = sorted(_PANDAS_STR_NA_VALUES - {"None"})
except ImportError:
    NA_VALUES_KEEP_NONE_STRING = [
        "", "#N/A", "#N/A N/A", "#NA", "-1.#IND", "-1.#QNAN", "-NaN", "-nan",
        "1.#IND", "1.#QNAN", "<NA>", "N/A", "NA", "NULL", "NaN", "n/a",
        "nan", "null",
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
    format: str                    # csv | tsv | xlsx | xls
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
        # The extension is unambiguous — sniffing a small sample of quoted
        # or numeric text is what mis-detects, so trust the extension.
        delimiter, delimiter_sniffed = "\t", False
    else:
        delimiter, delimiter_sniffed = _sniff_delimiter(sample)

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
    )
    return df, report


def _read_excel(path: Path, format_: str) -> tuple[pd.DataFrame, ReadReport]:
    engine = "openpyxl" if format_ == "xlsx" else "xlrd"
    try:
        df = pd.read_excel(path, engine=engine)
    except Exception as exc:
        raise DatasetReadError(f"'{path.name}' could not be read as {format_}: {exc}") from exc
    report = ReadReport(
        path=str(path),
        format=format_,
        encoding="n/a",
        encoding_confident=True,
        delimiter=None,
        delimiter_sniffed=False,
    )
    return df, report


def _is_nested(value: object) -> bool:
    return isinstance(value, (dict, list))


def _read_json(path: Path, format_: str) -> tuple[pd.DataFrame, ReadReport]:
    notes: list[str] = []
    try:
        df = pd.read_json(path, lines=(format_ == "jsonl"))
    except (ValueError, OSError) as exc:
        raise DatasetReadError(f"'{path.name}' could not be parsed as {format_}: {exc}") from exc
    if df.empty:
        raise DatasetReadError(f"'{path.name}' contains no records.")

    nested_cols = [c for c in df.columns if df[c].map(_is_nested).any()]
    flattened = False
    flattened_columns: int | None = None
    if nested_cols:
        cols_before = len(df.columns)
        # Depth-capped: a record nested deeper than JSON_FLATTEN_MAX_DEPTH
        # keeps its remaining structure as a dict/list cell value rather
        # than exploding into unbounded dotted columns.
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
    else:
        df, report = _read_excel(path, format_)
    df = _cap_rows(df, report)
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
        hit = _READ_CACHE.get(key)
        if hit is not None:
            _READ_CACHE.move_to_end(key)
            df, report = hit
            return df.copy(), report

    df, report = _read_uncached(file_path)

    if key is not None:
        _READ_CACHE[key] = (df.copy(), report)
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
