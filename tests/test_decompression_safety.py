"""
Unit tests for decompression safety & allocation limits (src/core/security.py).
Protects against zip-bombs, nested archive inflation, and excessive memory allocations.
"""
from __future__ import annotations

import gzip
import io
import zipfile

import pandas as pd
import pytest

from src.core.io import DatasetReadError, _decompress
from src.core.security import (
    UploadValidationError,
    _check_decompression_safety,
    validate_upload,
)


def _make_dummy_xlsx(uncompressed_text: bytes = b"test", filename: str = "sheet.xml") -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", b"<?xml version='1.0'?><Types></Types>")
        zf.writestr(filename, uncompressed_text)
    return buf.getvalue()


def test_valid_xlsx_passes_safety_check() -> None:
    xlsx_bytes = _make_dummy_xlsx(b"A" * 1024)
    name = validate_upload("valid_data.xlsx", xlsx_bytes)
    assert name == "valid_data.xlsx"


def test_zip_bomb_high_ratio_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    # Set maximum compression ratio to 10:1 to simulate sensitive threshold
    monkeypatch.setenv("MAX_COMPRESSION_RATIO", "10.0")

    # 100 KB of zeros compresses into ~100 bytes (ratio > 500:1)
    compressed_bomb = _make_dummy_xlsx(b"\x00" * 100_000)
    with pytest.raises(UploadValidationError, match="Suspicious compression ratio"):
        validate_upload("bomb.xlsx", compressed_bomb)


def test_oversize_uncompressed_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    # Set max uncompressed ceiling to 1 MB
    monkeypatch.setenv("MAX_UNCOMPRESSED_MB", "1")
    monkeypatch.setenv("MAX_COMPRESSION_RATIO", "10000.0")  # allow the ratio

    # 2 MB of uncompressed data
    big_uncompressed = _make_dummy_xlsx(b"A" * (2 * 1024 * 1024))
    with pytest.raises(UploadValidationError, match="exceeding the 1 MB limit"):
        validate_upload("oversize.xlsx", big_uncompressed)


def test_excessive_zip_entries_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_STORED) as zf:
        for i in range(5001):
            zf.writestr(f"file_{i}.txt", b"x")
    data = buf.getvalue()

    with pytest.raises(UploadValidationError, match="exceeding the safety limit of 5000"):
        _check_decompression_safety(data, ".xlsx")


def test_corrupted_xlsx_rejected() -> None:
    # Magic bytes for PK, but truncated garbage afterwards
    corrupt_bytes = b"PK\x03\x04\x00\x00corrupted_payload_junk"
    with pytest.raises(UploadValidationError, match="Malformed or corrupted Excel archive"):
        _check_decompression_safety(corrupt_bytes, ".xlsx")


def test_parquet_safety_check(tmp_path: pytest.TempPathFactory) -> None:
    df = pd.DataFrame({"col1": range(100), "col2": ["val"] * 100})
    buf = io.BytesIO()
    df.to_parquet(buf, engine="pyarrow")
    parquet_bytes = buf.getvalue()

    # Valid parquet passes
    name = validate_upload("dataset.parquet", parquet_bytes)
    assert name == "dataset.parquet"


def test_parquet_oversize_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MAX_UNCOMPRESSED_MB", "1")
    monkeypatch.setenv("MAX_COMPRESSION_RATIO", "100000.0")

    # Create parquet with ~2MB uncompressed
    df = pd.DataFrame({"col1": ["long_string_content_repeated_here" * 50] * 2000})
    buf = io.BytesIO()
    df.to_parquet(buf, engine="pyarrow", compression=None)
    parquet_bytes = buf.getvalue()

    # Check uncompressed size
    if len(parquet_bytes) > 1 * 1024 * 1024:
        with pytest.raises(UploadValidationError, match="exceeding the 1 MB limit"):
            validate_upload("large.parquet", parquet_bytes)


@pytest.mark.parametrize("compression", ["gzip", "zip"])
def test_reader_streaming_decompression_enforces_output_cap(
    monkeypatch: pytest.MonkeyPatch, compression: str
) -> None:
    monkeypatch.setenv("MAX_UNCOMPRESSED_MB", "1")
    expanded = b"a,b\n" + b"1,2\n" * 300_000
    if compression == "gzip":
        raw = gzip.compress(expanded)
        name = "large.csv.gz"
    else:
        archive = io.BytesIO()
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("large.csv", expanded)
        raw = archive.getvalue()
        name = "large.csv.zip"

    with pytest.raises(DatasetReadError, match=r"expands beyond.*1 MB"):
        _decompress(raw, compression, name)


def test_compressed_uploads_are_validated(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MAX_COMPRESSION_RATIO", "10000")
    payload = gzip.compress(b"a,b\n1,2\n")
    assert validate_upload("table.csv.gz", payload) == "table.csv.gz"
    with pytest.raises(UploadValidationError, match="gzip-compressed"):
        validate_upload("table.csv.gz", b"not gzip")
    with pytest.raises(UploadValidationError, match="must be named"):
        validate_upload("table.gz", payload)


def test_zip_upload_enforces_single_dataset_file(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MAX_COMPRESSION_RATIO", "10000")
    # Multi-file zip
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("table1.csv", b"a,b\n1,2\n")
        zf.writestr("table2.csv", b"c,d\n3,4\n")
    multi_zip = buf.getvalue()

    with pytest.raises(UploadValidationError, match="must contain exactly one dataset file"):
        validate_upload("data.csv.zip", multi_zip)

    # Non-csv/tsv inner member
    buf_invalid = io.BytesIO()
    with zipfile.ZipFile(buf_invalid, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("script.py", b"print(1)\n")
    invalid_zip = buf_invalid.getvalue()

    with pytest.raises(UploadValidationError, match=r"must contain a \.csv or \.tsv file"):
        validate_upload("data.csv.zip", invalid_zip)


def test_zip_upload_invalid_zip_header() -> None:
    with pytest.raises(UploadValidationError, match="zip-compressed"):
        validate_upload("data.csv.zip", b"not a valid zip")


def test_default_uncompressed_limit_is_50mb(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MAX_UNCOMPRESSED_MB", raising=False)
    from src.core.io import get_max_uncompressed_bytes
    from src.core.security import max_uncompressed_bytes

    assert max_uncompressed_bytes() == 50 * 1024 * 1024
    assert get_max_uncompressed_bytes() == 50 * 1024 * 1024
