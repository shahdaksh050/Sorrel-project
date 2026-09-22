"""
Security utilities for the Agentic Data Analysis System.

Centralises every check applied to user-supplied input before it touches
the filesystem or the analysis pipeline:

  - Filename sanitisation (path-traversal, reserved names, unsafe chars)
  - Upload validation (extension allowlist, size limits, content sniffing)
  - Safe output-path resolution (no writes outside the output root)

Design rules:
  - Pure functions, no I/O side effects except where documented.
  - Fail closed: anything suspicious raises UploadValidationError.
  - Limits are environment-configurable but have safe defaults.
"""
from __future__ import annotations

import io
import os
import re
import zipfile
from pathlib import Path

import pandas as pd

from src.core.io import SUPPORTED_EXTENSIONS

#: File extensions the pipeline knows how to ingest. Single source of truth
#: is src.core.io.SUPPORTED_EXTENSIONS — read_any's own dispatch table —
#: so this allowlist can never drift out of sync with what the reader
#: actually handles (previously missing .tsv here while every tool's
#: reader already accepted it).
ALLOWED_EXTENSIONS: frozenset[str] = SUPPORTED_EXTENSIONS

#: Default upload ceiling in megabytes (override with MAX_UPLOAD_MB env var).
DEFAULT_MAX_UPLOAD_MB = 200

#: Default uncompressed container size ceiling in megabytes (override with MAX_UNCOMPRESSED_MB).
DEFAULT_MAX_UNCOMPRESSED_MB = 500

#: Default maximum compression ratio for archives before flagging as a potential bomb.
DEFAULT_MAX_COMPRESSION_RATIO = 100.0

#: Maximum entries in a zip archive (e.g. .xlsx) before rejection.
DEFAULT_MAX_ZIP_ENTRIES = 5000

#: Windows reserved device names — writing to these can hang or misbehave.
_RESERVED_NAMES: frozenset[str] = frozenset(
    {"con", "prn", "aux", "nul"}
    | {f"com{i}" for i in range(1, 10)}
    | {f"lpt{i}" for i in range(1, 10)}
)

#: Characters allowed in a sanitised filename stem.
_SAFE_STEM_RE = re.compile(r"[^A-Za-z0-9._ -]+")

#: Magic bytes that identify formats we must reject inside a ".csv".
_BINARY_SIGNATURES: tuple[bytes, ...] = (
    b"\x7fELF",        # ELF executable
    b"MZ",             # Windows PE executable
    b"\x89PNG",        # PNG
    b"\xff\xd8\xff",   # JPEG
    b"%PDF",           # PDF
)

#: Magic bytes for legitimate Excel containers.
_XLSX_SIGNATURE = b"PK\x03\x04"   # xlsx = zip container
_XLS_SIGNATURE = b"\xd0\xcf\x11\xe0"  # legacy OLE2 compound file


class UploadValidationError(Exception):
    """Raised when an uploaded file fails any security or format check."""


def max_upload_bytes() -> int:
    """Return the configured upload size ceiling in bytes."""
    try:
        mb = int(os.getenv("MAX_UPLOAD_MB", str(DEFAULT_MAX_UPLOAD_MB)))
    except ValueError:
        mb = DEFAULT_MAX_UPLOAD_MB
    return max(1, mb) * 1024 * 1024


def max_uncompressed_bytes() -> int:
    """Return the configured uncompressed container size ceiling in bytes."""
    try:
        mb = int(os.getenv("MAX_UNCOMPRESSED_MB", str(DEFAULT_MAX_UNCOMPRESSED_MB)))
    except ValueError:
        mb = DEFAULT_MAX_UNCOMPRESSED_MB
    return max(1, mb) * 1024 * 1024


def max_compression_ratio() -> float:
    """Return the configured maximum compression ratio before flagging as a bomb."""
    try:
        return float(os.getenv("MAX_COMPRESSION_RATIO", str(DEFAULT_MAX_COMPRESSION_RATIO)))
    except ValueError:
        return DEFAULT_MAX_COMPRESSION_RATIO


def sanitize_filename(name: str) -> str:
    """
    Reduce a user-supplied filename to a safe basename.

    - Strips any directory components (defeats ../ traversal).
    - Removes characters outside [A-Za-z0-9._ -].
    - Renames Windows reserved device names.
    - Guarantees a non-empty stem.

    Returns:
        A safe filename preserving the original extension (lower-cased).
    """
    # Take only the final path component, regardless of separator style
    base = Path(name.replace("\\", "/")).name
    stem = Path(base).stem
    suffix = Path(base).suffix.lower()
    # Extension-only names like ".csv": pathlib treats them as a hidden file
    # with no suffix — reinterpret as an empty stem plus the extension.
    if not suffix and stem.startswith("."):
        suffix = stem.lower()
        stem = ""

    stem = _SAFE_STEM_RE.sub("_", stem).strip(" .")
    if not stem or stem.lower() in _RESERVED_NAMES:
        stem = f"upload_{stem}" if stem else "upload"
    return f"{stem}{suffix}"


def validate_upload(filename: str, raw_bytes: bytes) -> str:
    """
    Validate an uploaded dataset before it is written to disk.

    Checks, in order:
      1. Extension is in the allowlist.
      2. File is non-empty and under the configured size ceiling.
      3. Content matches the claimed format (magic-byte sniffing).

    Args:
        filename:  The name supplied by the client (untrusted).
        raw_bytes: The full file payload.

    Returns:
        The sanitised filename safe to use on the local filesystem.

    Raises:
        UploadValidationError: On any failed check.
    """
    safe_name = sanitize_filename(filename)
    suffix = Path(safe_name).suffix

    if suffix not in ALLOWED_EXTENSIONS:
        allowed = ", ".join(sorted(ALLOWED_EXTENSIONS))
        raise UploadValidationError(
            f"Unsupported file type '{suffix or '(none)'}'. Allowed: {allowed}."
        )

    if not raw_bytes:
        raise UploadValidationError("Uploaded file is empty.")

    limit = max_upload_bytes()
    if len(raw_bytes) > limit:
        raise UploadValidationError(
            f"File is {len(raw_bytes) / 1_048_576:.1f} MB — exceeds the "
            f"{limit // 1_048_576} MB limit. Sample or split the dataset first."
        )

    head = raw_bytes[:8]
    if suffix == ".csv":
        for sig in (*_BINARY_SIGNATURES, _XLSX_SIGNATURE, _XLS_SIGNATURE):
            if head.startswith(sig):
                raise UploadValidationError(
                    "File claims to be CSV but contains binary content."
                )
        if b"\x00" in raw_bytes[:4096]:
            raise UploadValidationError(
                "File claims to be CSV but contains null bytes (binary data)."
            )
    elif suffix == ".xlsx":
        if not head.startswith(_XLSX_SIGNATURE):
            raise UploadValidationError("File claims to be .xlsx but is not a valid Excel container.")
        if zipfile.is_zipfile(io.BytesIO(raw_bytes)):
            _check_decompression_safety(raw_bytes, suffix)
    elif suffix == ".xls" and not head.startswith(_XLS_SIGNATURE):
        raise UploadValidationError("File claims to be .xls but is not a valid Excel container.")
    elif suffix == ".parquet":
        _check_decompression_safety(raw_bytes, suffix)

    return safe_name


def _check_decompression_safety(raw_bytes: bytes, suffix: str) -> None:
    """
    Inspect compressed containers (e.g. .xlsx zip archives, .parquet) to prevent
    decompression bombs (zip bombs) or excessive memory allocation before reading.
    """
    max_uncompressed = max_uncompressed_bytes()
    max_ratio = max_compression_ratio()

    if suffix == ".xlsx":
        try:
            with zipfile.ZipFile(io.BytesIO(raw_bytes)) as zf:
                infolist = zf.infolist()
                if len(infolist) > DEFAULT_MAX_ZIP_ENTRIES:
                    raise UploadValidationError(
                        f"Excel archive contains {len(infolist)} entries, exceeding the "
                        f"safety limit of {DEFAULT_MAX_ZIP_ENTRIES}."
                    )
                total_uncompressed = sum(info.file_size for info in infolist)
                if total_uncompressed > max_uncompressed:
                    raise UploadValidationError(
                        f"Uncompressed Excel content is {total_uncompressed / 1_048_576:.1f} MB, "
                        f"exceeding the {max_uncompressed // 1_048_576} MB limit."
                    )
                ratio = total_uncompressed / max(1, len(raw_bytes))
                if ratio > max_ratio:
                    raise UploadValidationError(
                        f"Suspicious compression ratio ({ratio:.1f}:1 > {max_ratio:.0f}:1). "
                        "File rejected as a potential decompression bomb."
                    )
        except zipfile.BadZipFile as err:
            raise UploadValidationError(f"Malformed or corrupted Excel archive: {err}") from err

    elif suffix == ".parquet":
        try:
            import pyarrow.parquet as pq

            pf = pq.ParquetFile(io.BytesIO(raw_bytes))  # type: ignore[no-untyped-call]
            metadata = pf.metadata
            total_uncompressed = sum(
                metadata.row_group(i).total_byte_size
                for i in range(metadata.num_row_groups)
            )
            if total_uncompressed > max_uncompressed:
                raise UploadValidationError(
                    f"Uncompressed Parquet content is {total_uncompressed / 1_048_576:.1f} MB, "
                    f"exceeding the {max_uncompressed // 1_048_576} MB limit."
                )
            ratio = total_uncompressed / max(1, len(raw_bytes))
            if ratio > max_ratio:
                raise UploadValidationError(
                    f"Suspicious Parquet compression ratio ({ratio:.1f}:1 > {max_ratio:.0f}:1). "
                    "File rejected as a potential decompression bomb."
                )
        except UploadValidationError:
            raise
        except Exception as err:
            raise UploadValidationError(f"Invalid or corrupted Parquet file: {err}") from err


#: Characters that make a spreadsheet treat a CSV cell as a formula.
_FORMULA_PREFIXES = ("=", "+", "-", "@", "\t", "\r")

#: Control characters and zero-width/bidi unicode stripped from prompt text.
_PROMPT_UNSAFE_RE = re.compile(r"[\x00-\x1f\x7f​-‏  ‪-‮]")


def sanitize_for_prompt(text: object, max_len: int = 80) -> str:
    """
    Make a dataset-derived string safe for injection into an LLM prompt.

    Dataset content (column names, category values) is untrusted: a column
    literally named "ignore previous instructions…" must ride into the
    planner as inert data, not as a directive. This strips control and
    zero-width/bidi characters, collapses all whitespace (no newlines means
    no fake prompt sections), and clips the length.
    """
    s = str(text)
    # Collapse whitespace FIRST so newlines become spaces (word boundaries
    # preserved), then strip the remaining non-whitespace control characters.
    s = " ".join(s.split())
    s = _PROMPT_UNSAFE_RE.sub("", s)
    if len(s) > max_len:
        s = s[: max_len - 1] + "…"
    return s


#: Personal-data patterns, unanchored: src.core.profiler fullmatches them
#: against column values, redact_pii_text substitutes them inside free text.
#: PHONE_RE demands a leading "+" or separators between digit groups so that
#: plain counts ("1234567890 rows") and dates never read as phone numbers.
EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
IPV4_RE = re.compile(r"(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)")
SSN_RE = re.compile(r"\d{3}-\d{2}-\d{4}")
CARD_RE = re.compile(r"\d(?:[ -]?\d){12,18}")
PHONE_RE = re.compile(
    r"\+\d{8,15}|(?:\+\d{1,3}[\s.-]?\(?\d{1,4}\)?|\(\d{2,4}\)|\d{2,4})[\s.-]\d{3,4}[\s.-]\d{3,4}"
)

_EMAIL_TEXT_RE = re.compile(rf"(?<![\w.+-]){EMAIL_RE.pattern}")
_IPV4_TEXT_RE = re.compile(rf"(?<![\d.]){IPV4_RE.pattern}(?!\.?\d)")
_SSN_TEXT_RE = re.compile(rf"(?<![\d-]){SSN_RE.pattern}(?![\d-])")
_CARD_TEXT_RE = re.compile(rf"(?<![\d.]){CARD_RE.pattern}(?!\.?\d)")
_PHONE_TEXT_RE = re.compile(rf"(?<![\w.+-])(?:{PHONE_RE.pattern})(?!\.?\d)")


def luhn_valid(digits: str) -> bool:
    """Luhn checksum over the digits of `digits` (separators ignored) —
    separates real card numbers from arbitrary 13-19 digit numbers."""
    nums = [int(ch) for ch in digits if ch.isdigit()]
    if not 13 <= len(nums) <= 19:
        return False
    total = 0
    for i, n in enumerate(reversed(nums)):
        if i % 2:
            n *= 2
            if n > 9:
                n -= 9
        total += n
    return total % 10 == 0


def pii_redaction_enabled() -> bool:
    """Whether PII values are withheld from LLM prompts. On unless the
    REDACT_PII environment variable is "false"; read per call so an
    operator can change it without a restart."""
    return os.getenv("REDACT_PII", "true").strip().lower() != "false"


def redact_pii_text(text: str) -> str:
    """
    Mask emails, IPv4 addresses, card numbers (Luhn-valid), SSN-style IDs
    and phone numbers inside free text bound for an LLM (finding headlines,
    error messages). Ordinary numbers — p-values, counts, years, dates —
    are left alone.
    """
    text = _EMAIL_TEXT_RE.sub("[redacted: email]", text)
    text = _IPV4_TEXT_RE.sub("[redacted: ip_address]", text)
    text = _CARD_TEXT_RE.sub(
        lambda m: "[redacted: card_number]" if luhn_valid(m.group()) else m.group(), text
    )
    text = _SSN_TEXT_RE.sub("[redacted: gov_id]", text)
    return _PHONE_TEXT_RE.sub("[redacted: phone]", text)


def escape_csv_formulas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Neutralise spreadsheet formula injection in a DataFrame bound for CSV export.

    Cells beginning with =, +, -, @ (or tab/CR) execute as formulas when the
    CSV is opened in Excel/Sheets. Prefixing a single quote renders them inert.
    Only string-typed cells are touched; use ONLY on user-facing exports,
    never on files read back by the pipeline.
    """
    out = df.copy()
    for col in out.columns:
        if not pd.api.types.is_numeric_dtype(out[col]) and not pd.api.types.is_bool_dtype(out[col]):
            out[col] = out[col].map(
                lambda v: f"'{v}" if isinstance(v, str) and v.startswith(_FORMULA_PREFIXES) else v
            )
    return out


def resolve_output_path(output_root: str | Path, *parts: str) -> Path:
    """
    Join path parts under output_root, refusing any escape from the root.

    Raises:
        UploadValidationError: If the resolved path falls outside output_root.
    """
    root = Path(output_root).resolve()
    candidate = root.joinpath(*parts).resolve()
    if root != candidate and root not in candidate.parents:
        raise UploadValidationError(
            f"Refusing to write outside the output directory: {candidate}"
        )
    return candidate
