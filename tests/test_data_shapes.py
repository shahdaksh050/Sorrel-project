"""
Edge-case dataset matrix — the regression harness for Round 5 items 2, 3, 5.

Contract under test: for every fixture in tests/fixtures, the pipeline must
either succeed on correctly-parsed data, or fail with an actionable error.
It must never report success on corrupt or misparsed input.

Item 2 (src.core.io.read_any) has landed, so this file now reads through
it directly. Item 5 (DatasetProfile.is_sufficient) has not, so
TestSufficiencyGate stays `xfail(strict=True)` — once it lands those tests
XPASS, strict=True turns that into a failure, and that's the signal to
drop the markers.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.core.io import DatasetReadError, read_any
from src.core.profiler import profile_dataframe
from tests.fixtures import (
    all_categorical,
    all_text,
    cp1252_csv,
    duplicate_headers,
    geo,
    headers_only,
    high_cardinality,
    high_missing,
    json_records,
    million_rows,
    mixed_type_column,
    panel,
    semicolon_csv,
    single_column,
    single_row,
    time_series,
    tsv_file,
    wide_frame,
)


def _read(path: Path) -> pd.DataFrame:
    df, _report = read_any(str(path))
    return df


class TestDelimiterAndEncoding:
    """U0.1 / U0.2 — fixed by item 2."""

    def test_tsv_file_parses_as_three_columns(self, tmp_path: Path) -> None:
        df, report = read_any(str(tsv_file(tmp_path)))
        assert df.shape[1] == 3
        assert report.delimiter == "\t"
        assert report.delimiter_sniffed is False

    def test_semicolon_csv_parses_as_three_columns(self, tmp_path: Path) -> None:
        df, report = read_any(str(semicolon_csv(tmp_path)))
        assert df.shape[1] == 3
        assert report.delimiter == ";"
        assert report.delimiter_sniffed is True

    def test_cp1252_csv_decodes_correctly(self, tmp_path: Path) -> None:
        df, report = read_any(str(cp1252_csv(tmp_path)))
        assert "François et José" in df.loc[3, "note"]
        assert "Zürich" in df.loc[3, "note"]
        assert report.encoding != "utf-8"

    def test_single_column_csv_falls_back_to_comma(self, tmp_path: Path) -> None:
        """csv.Sniffer raises on genuinely single-column data — must fall
        back to ',' rather than propagate."""
        df, report = read_any(str(single_column(tmp_path)))
        assert df.shape == (50, 1)
        assert report.delimiter == ","
        assert report.delimiter_sniffed is False

    def test_bom_csv_yields_clean_column_names(self, tmp_path: Path) -> None:
        p = tmp_path / "bom.csv"
        p.write_bytes(b"\xef\xbb\xbfa,b\n1,2\n")
        df, report = read_any(str(p))
        assert list(df.columns) == ["a", "b"]
        assert report.encoding == "utf-8-sig"
        assert report.encoding_confident is True

    def test_cp1252_fallback_marks_encoding_unconfident(self, tmp_path: Path) -> None:
        """A byte sequence charset_normalizer can't confidently place at all
        still must not crash the pipeline — cp1252 accepts virtually any
        byte, but the report must say it was a guess."""
        p = tmp_path / "ambiguous.csv"
        p.write_bytes(b"a,b\n\xff\xfe,2\n")
        _df, report = read_any(str(p))
        if not report.encoding_confident:
            assert report.encoding == "cp1252"


class TestJsonRecordsSupported:
    """JSON support (item U1.2) landed in Round 7 — records-oriented JSON
    parses cleanly into a DataFrame rather than being rejected. Was
    `TestUnsupportedFormatFailsLoud`, asserting the pre-U1.2 behavior; kept
    as a locked-in-format regression check instead of a rejection check."""

    def test_json_records_parses_to_dataframe(self, tmp_path: Path) -> None:
        df, report = read_any(str(json_records(tmp_path)))
        assert list(df.columns) == ["a", "b"]
        assert df.to_dict(orient="records") == [{"a": 1, "b": 2}, {"a": 3, "b": 4}]
        assert report.format == "json"


class TestAlreadyCorrectFailLoud:
    """Already-correct behaviors — locked in against regression."""

    def test_empty_file_raises(self, tmp_path: Path) -> None:
        p = tmp_path / "empty.csv"
        p.write_bytes(b"")
        with pytest.raises(DatasetReadError):
            read_any(str(p))

    def test_single_column_reads_correctly(self, tmp_path: Path) -> None:
        df = _read(single_column(tmp_path))
        assert df.shape == (50, 1)

    def test_all_categorical_has_no_numeric_columns(self, tmp_path: Path) -> None:
        df = _read(all_categorical(tmp_path))
        profile = profile_dataframe(df)
        assert all(c.kind != "numeric" for c in profile.columns)

    def test_all_text_classified_as_text(self, tmp_path: Path) -> None:
        df = _read(all_text(tmp_path))
        profile = profile_dataframe(df)
        kinds = {c.name: c.kind for c in profile.columns}
        assert kinds["review"] == "text"
        assert kinds["comment"] == "text"

    def test_high_missing_flagged(self, tmp_path: Path) -> None:
        df = _read(high_missing(tmp_path))
        profile = profile_dataframe(df)
        col_a = next(c for c in profile.columns if c.name == "a")
        assert "high_missing" in col_a.flags

    def test_mixed_type_column_not_silently_numeric(self, tmp_path: Path) -> None:
        """A column that's 75% numeric with one stray literal must never be
        silently treated as clean numeric data (below item 3's 95% coercion
        threshold either way — this holds before and after item 3)."""
        df = _read(mixed_type_column(tmp_path))
        profile = profile_dataframe(df)
        col = next(c for c in profile.columns if c.name == "value")
        assert col.kind != "numeric"

    def test_duplicate_headers_mangled_and_reported(self, tmp_path: Path) -> None:
        """pandas already disambiguates a,a,b -> a,a.1,b rather than
        silently dropping the second 'a'; item 2's ReadReport additionally
        records the original duplicate so the report can surface it."""
        df, report = read_any(str(duplicate_headers(tmp_path)))
        assert list(df.columns) == ["a", "a.1", "b"]
        assert "a" in report.duplicate_headers


class TestSufficiencyGate:
    """U1.3 — item 5's DatasetProfile.is_sufficient hard floor. Both
    fixtures have row_count < 2."""

    def test_headers_only_flagged_insufficient(self, tmp_path: Path) -> None:
        df = _read(headers_only(tmp_path))
        profile = profile_dataframe(df)
        assert profile.is_sufficient is False

    def test_single_row_flagged_insufficient(self, tmp_path: Path) -> None:
        df = _read(single_row(tmp_path))
        profile = profile_dataframe(df)
        assert profile.is_sufficient is False


class TestDatasetNature:
    """Nature-detection fixtures already work — locked in against regression."""

    def test_wide_frame_is_high_dimensional(self, tmp_path: Path) -> None:
        df = _read(wide_frame(tmp_path))
        profile = profile_dataframe(df)
        assert profile.is_high_dimensional is True

    def test_time_series_detected(self, tmp_path: Path) -> None:
        df = _read(time_series(tmp_path))
        profile = profile_dataframe(df)
        assert profile.is_time_series is True

    def test_panel_group_cols_detected(self, tmp_path: Path) -> None:
        df = _read(panel(tmp_path))
        profile = profile_dataframe(df)
        assert "group" in profile.panel_group_cols

    def test_geo_columns_detected(self, tmp_path: Path) -> None:
        df = _read(geo(tmp_path))
        profile = profile_dataframe(df)
        assert profile.has_geo()

    def test_high_cardinality_flagged(self, tmp_path: Path) -> None:
        df = _read(high_cardinality(tmp_path))
        profile = profile_dataframe(df)
        col = next(c for c in profile.columns if c.name == "sku")
        assert "high_cardinality" in col.flags


@pytest.mark.slow
class TestScale:
    def test_million_rows_reads_and_profiles(self, tmp_path: Path) -> None:
        df = _read(million_rows(tmp_path))
        assert df.shape == (1_000_000, 4)
        profile = profile_dataframe(df)
        assert profile.row_count == 1_000_000


def test_all_null_columns_dropped_with_note(tmp_path) -> None:
    path = tmp_path / "trail.csv"
    path.write_text("a;b;;\n1,5;x;;\n2,5;y;;\n", encoding="utf-8")
    df, report = read_any(str(path))
    assert list(df.columns) == ["a", "b"]
    assert any("Dropped 2 empty column(s)" in n for n in report.notes)
