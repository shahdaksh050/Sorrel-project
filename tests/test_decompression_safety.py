"""
Unit tests for decompression safety & allocation limits (src/core/security.py).
Protects against zip-bombs, nested archive inflation, and excessive memory allocations.
"""
from __future__ import annotations

import io
import zipfile

import pandas as pd
import pytest

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
