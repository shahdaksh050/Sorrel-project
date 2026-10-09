"""Deterministic unit-aware feature extraction for composite tabular fields.

This tool keeps source columns intact and writes a derived CSV containing only
additive, typed features.  It is intentionally pattern based: it does not use
an LLM and its output records coverage and parser provenance rather than raw
dataset values.
"""
from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from src.core.analysis_io import read_analysis_df
from src.core.io import DatasetReadError, invalidate_read_cache
from src.core.security import UploadValidationError, resolve_output_path
from src.core.semantic_readiness import assess_semantic_readiness
from src.tools.base import BaseTool, ToolExecutionError

if TYPE_CHECKING:
    from src.core.memory import DatasetMetadata
    from src.core.profiler import DatasetProfile

_CURRENCY_RE = re.compile(
    r"(?P<marker>₹|\b(?:inr|rs\.?)\b|[$€£¥])\s*(?P<number>\d{1,3}(?:,\d{2,3})+|\d+(?:\.\d+)?)",
    re.IGNORECASE,
)
_RAM_SIZE_RE = re.compile(r"(?P<number>\d+(?:\.\d+)?)\s*(?P<unit>tb|gb|mb)\s*(?:ram|memory)\b", re.IGNORECASE)
_STORAGE_SIZE_RE = re.compile(
    r"(?P<number>\d+(?:\.\d+)?)\s*(?P<unit>tb|gb|mb)\s*"
    r"(?:storage|rom|internal(?:\s+(?:storage|memory))?|inbuilt(?:\s+(?:storage|memory))?|"
    r"built[\s-]?in(?:\s+(?:storage|memory))?)\b",
    re.IGNORECASE,
)
_MAH_RE = re.compile(r"(?P<number>\d+(?:\.\d+)?)\s*mah\b", re.IGNORECASE)
_POWER_RE = re.compile(r"(?P<number>\d+(?:\.\d+)?)\s*(?P<unit>kw|w|watt)\b", re.IGNORECASE)
_LENGTH_RE = re.compile(r"(?P<number>\d+(?:\.\d+)?)\s*(?P<unit>in(?:ch(?:es)?)?|cm|mm|ft|m)\b", re.IGNORECASE)
_REFRESH_RE = re.compile(r"(?P<number>\d+(?:\.\d+)?)\s*hz\b", re.IGNORECASE)
_MEGAPIXEL_RE = re.compile(r"(?P<number>\d+(?:\.\d+)?)\s*(?:mp|megapixel)\b", re.IGNORECASE)
_FREQUENCY_RE = re.compile(r"(?P<number>\d+(?:\.\d+)?)\s*(?P<unit>ghz|mhz|khz)\b", re.IGNORECASE)

_PLATFORM_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("Android", re.compile(r"\bandroid\b", re.IGNORECASE)),
    ("iOS", re.compile(r"\b(?:ios|ipados)\b", re.IGNORECASE)),
    ("Windows", re.compile(r"\bwindows(?:\s+phone)?\b", re.IGNORECASE)),
    ("macOS", re.compile(r"\bmacos\b", re.IGNORECASE)),
    ("Linux", re.compile(r"\blinux\b", re.IGNORECASE)),
    ("HarmonyOS", re.compile(r"\bharmonyos\b", re.IGNORECASE)),
    ("ChromeOS", re.compile(r"\bchrome\s*os\b", re.IGNORECASE)),
)
_BOOLEAN_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("has_5g", re.compile(r"\b5g\b", re.IGNORECASE)),
    ("has_nfc", re.compile(r"\bnfc\b", re.IGNORECASE)),
    ("has_bluetooth", re.compile(r"\bbluetooth\b", re.IGNORECASE)),
    ("has_wifi", re.compile(r"\b(?:wi-?fi|wifi)\b", re.IGNORECASE)),
    ("has_fm_radio", re.compile(r"\bfm\s+radio\b", re.IGNORECASE)),
    ("has_fast_charging", re.compile(r"\bfast\s+charg(?:e|ing)\b", re.IGNORECASE)),
)

# A handful of recognised cells are enough to diagnose a shifted column while
# avoiding removal on a one-off note or a nearly-empty field.
_QUARANTINE_MIN_CLASSIFIED_VALUES = 3
_QUARANTINE_MAX_CONFORMANCE_PCT = 75.0


def _to_float(number: str) -> float:
    """Parse a unit-bearing decimal without locale inference."""
    return float(number.replace(",", ""))


def _first_number(text: str, pattern: re.Pattern[str]) -> float | None:
    """Return the first named ``number`` capture matched by a parser."""
    match = pattern.search(text)
    return _to_float(match.group("number")) if match else None


def _size_in_gb(number: str, unit: str) -> float:
    """Normalise data sizes to decimal gigabytes for comparability."""
    value = _to_float(number)
    factor = {"mb": 0.001, "gb": 1.0, "tb": 1000.0}[unit.lower()]
    return value * factor


def _first_size_gb(text: str, pattern: re.Pattern[str]) -> float | None:
    """Extract the first data-size value and normalise it to GB."""
    match = pattern.search(text)
    if match is None:
        return None
    return _size_in_gb(match.group("number"), match.group("unit"))


def _length_inches(text: str) -> float | None:
    """Extract the first explicit length and normalise it to inches."""
    match = _LENGTH_RE.search(text)
    if match is None:
        return None
    value = _to_float(match.group("number"))
    factor = {"in": 1.0, "inch": 1.0, "inches": 1.0, "cm": 1 / 2.54, "mm": 1 / 25.4, "ft": 12.0, "m": 100 / 2.54}
    return value * factor[match.group("unit").lower()]


def _frequency_ghz(text: str) -> float | None:
    """Extract a clock frequency and normalise it to GHz."""
    match = _FREQUENCY_RE.search(text)
    if match is None:
        return None
    value = _to_float(match.group("number"))
    factor = {"ghz": 1.0, "mhz": 0.001, "khz": 0.000001}[match.group("unit").lower()]
    return value * factor


def _platform(text: str) -> str | None:
    """Map platform mentions to a conservative canonical label."""
    for name, pattern in _PLATFORM_PATTERNS:
        if pattern.search(text):
            return name
    return None


def _feature_presence(text: str, pattern: re.Pattern[str]) -> bool | None:
    """Return a nullable capability flag, retaining unmentioned values as NA."""
    match = pattern.search(text)
    if match is None:
        return None
    before = text[max(0, match.start() - 24):match.start()]
    return not bool(re.search(r"\b(?:no|without|not)\b[^,;]{0,18}$", before, re.IGNORECASE))


def _presence_extractor(pattern: re.Pattern[str]) -> Callable[[str], object]:
    """Bind a capability pattern to a fully typed row extractor."""
    def extract(text: str) -> object:
        return _feature_presence(text, pattern)

    return extract


def _currency_amount(text: str) -> tuple[float | None, str | None]:
    """Extract a marked currency amount; plain numeric columns stay untouched."""
    match = _CURRENCY_RE.search(text)
    if match:
        marker = match.group("marker").lower()
        code = "INR" if marker in {"₹", "inr", "rs", "rs."} else {"$": "USD", "€": "EUR", "£": "GBP", "¥": "JPY"}[marker]
        return _to_float(match.group("number")), code
    return None, None


def _add_feature(
    output: pd.DataFrame,
    source_column: str,
    source_slug: str,
    feature_name: str,
    values: Sequence[object],
    semantic_type: str,
    parser: str,
    unit: str | None = None,
    nullable_boolean: bool = False,
    source_columns: list[str] | None = None,
    preferred_column: str | None = None,
) -> dict[str, Any] | None:
    """Append one derived feature when it has observed values and describe it."""
    non_null = sum(value is not None and not pd.isna(value) for value in values)
    if non_null == 0:
        return None
    column = (
        preferred_column
        if preferred_column and preferred_column not in output.columns
        else f"derived__{source_slug}__{feature_name}"
    )
    if nullable_boolean:
        output[column] = pd.array(values, dtype="boolean")
    else:
        output[column] = values
    result: dict[str, Any] = {
        "source_column": source_column,
        "derived_column": column,
        "semantic_type": semantic_type,
        "parser": parser,
        "unit": unit,
        "non_null_count": non_null,
        "coverage_pct": round(100.0 * non_null / max(len(output), 1), 2),
    }
    if source_columns is not None:
        result["source_columns"] = source_columns
        result["selection_policy"] = "prefer_name_matched_source_then_column_order"
    return result


def _ordered_source_indexes(source_names: Sequence[str], name_hints: Sequence[str]) -> list[int]:
    """Prefer semantically named columns while retaining deterministic fallback order."""
    return sorted(
        range(len(source_names)),
        key=lambda index: 0 if any(hint in source_names[index].lower() for hint in name_hints) else 1,
    )


def _consolidate_rows(
    source_names: Sequence[str],
    source_texts: Sequence[Sequence[str]],
    extractor: Callable[[str], object],
    name_hints: Sequence[str],
) -> tuple[list[object], list[str]]:
    """Extract one deterministic row-level feature across all source columns."""
    if not source_texts:
        return [], []
    order = _ordered_source_indexes(source_names, name_hints)
    values: list[object] = []
    used_indexes: set[int] = set()
    for row_index in range(len(source_texts[0])):
        selected: object = None
        for source_index in order:
            text = source_texts[source_index][row_index]
            if not text:
                continue
            candidate = extractor(text)
            if candidate is not None:
                selected = candidate
                used_indexes.add(source_index)
                break
        values.append(selected)
    return values, [source_names[index] for index in sorted(used_indexes)]


class StructuredFeatureExtractionTool(BaseTool):
    """Add deterministic typed features parsed from common units and concepts."""

    name = "extract_structured_features"
    description = (
        "Create an additive derived CSV by parsing unit-bearing or composite columns into typed "
        "currency, size, capacity, power, dimension, frequency, platform, and capability fields. "
        "Reports parser coverage and semantic readiness without exposing raw values."
    )
    output_subdir = "derived"

    def applies_to(self, profile: DatasetProfile | None, metadata: DatasetMetadata | None) -> float:
        """Gate on composite/textual candidates to avoid copying clean numeric data."""
        if profile is None:
            return 0.0
        if profile.text_cols:
            return 1.0
        name_hints = re.compile(
            r"(?:price|cost|amount|ram|memory|storage|battery|power|charg|display|screen|camera|"
            r"resolution|processor|cpu|frequency|platform|operating|network|connect|feature)",
            re.IGNORECASE,
        )
        for column in profile.columns:
            if column.kind in {"categorical", "text"} and name_hints.search(column.name):
                return 0.9
        return 0.0

    def execute(  # type: ignore[override]
        self,
        file_path: str,
        output_dir: str | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        """Read, enrich, and persist an additive CSV without changing source columns."""
        try:
            df = read_analysis_df(file_path)
        except DatasetReadError as exc:
            raise ToolExecutionError(str(exc)) from exc
        if df.empty:
            raise ToolExecutionError("Dataset has no rows; no structured features can be extracted.")

        readiness = assess_semantic_readiness(df)
        output = df.copy()
        provenance: list[dict[str, Any]] = []
        source_names = [str(source) for source in df.columns]
        source_texts = [
            ["" if pd.isna(value) else str(value) for value in df[source].tolist()]
            for source in df.columns
        ]

        # Emit one canonical feature per concept, not one copy per source
        # column. Source-local copies are highly correlated aliases of the
        # same fact and sparse matches in shifted cells create misleading
        # columns such as "processor_data_size". `_consolidate_rows` still
        # scans every source so a value shifted into the wrong raw column is
        # recovered, while provenance records every source that contributed.

        row_extractors: tuple[tuple[str, Callable[[str], object], str, str, str | None, bool, tuple[str, ...]], ...] = (
            ("currency_amount", lambda text: _currency_amount(text)[0], "numeric", "currency", None, False, ("price", "cost", "amount", "revenue", "salary", "income", "fare", "fee")),
            ("currency_code", lambda text: _currency_amount(text)[1], "categorical", "currency_marker", None, False, ("price", "cost", "amount", "revenue", "salary", "income", "fare", "fee")),
            ("memory_gb", lambda text: _first_size_gb(text, _RAM_SIZE_RE), "numeric", "memory_size", "GB", False, ("ram", "memory")),
            ("storage_gb", lambda text: _first_size_gb(text, _STORAGE_SIZE_RE), "numeric", "storage_size", "GB", False, ("storage", "rom", "disk")),
            ("battery_mah", lambda text: _first_number(text, _MAH_RE), "numeric", "battery_capacity", "mAh", False, ("battery", "capacity")),
            ("power_w", lambda text: _first_number(text, _POWER_RE), "numeric", "power", "W", False, ("power", "charg")),
            ("length_inches", _length_inches, "numeric", "length", "in", False, ("display", "screen", "dimension", "size")),
            ("refresh_hz", lambda text: _first_number(text, _REFRESH_RE), "numeric", "refresh_rate", "Hz", False, ("refresh", "display", "screen")),
            ("resolution_mp", lambda text: _first_number(text, _MEGAPIXEL_RE), "numeric", "resolution", "MP", False, ("camera", "resolution", "image", "sensor")),
            ("frequency_ghz", _frequency_ghz, "numeric", "frequency", "GHz", False, ("processor", "cpu", "clock", "frequency", "speed")),
            ("platform", _platform, "categorical", "platform", None, False, ("os", "operating", "platform", "system")),
        )
        for feature_name, extractor, semantic_type, parser, unit, nullable_boolean, name_hints in row_extractors:
            parsed, contributing_sources = _consolidate_rows(source_names, source_texts, extractor, name_hints)
            item = _add_feature(
                output,
                "multiple_sources",
                "row",
                feature_name,
                parsed,
                semantic_type,
                parser,
                unit,
                nullable_boolean,
                contributing_sources,
                feature_name,
            )
            if item is not None:
                provenance.append(item)
        for feature_name, pattern in _BOOLEAN_PATTERNS:
            parsed, contributing_sources = _consolidate_rows(
                source_names,
                source_texts,
                _presence_extractor(pattern),
                ("network", "connect", "wireless", "feature", "capability", "sim"),
            )
            item = _add_feature(
                output,
                "multiple_sources",
                "row",
                feature_name,
                parsed,
                "boolean",
                "capability_flag",
                None,
                True,
                contributing_sources,
                feature_name,
            )
            if item is not None:
                provenance.append(item)

        quarantined_columns: list[dict[str, Any]] = []
        for assessment in readiness["per_column"]:
            conformance = assessment["conformance_pct"]
            classified_count = assessment["classified_value_count"]
            column = assessment["column"]
            if (
                conformance is not None
                and float(conformance) < _QUARANTINE_MAX_CONFORMANCE_PCT
                and int(classified_count) >= _QUARANTINE_MIN_CLASSIFIED_VALUES
                and column in output.columns
            ):
                output = output.drop(columns=[column])
                quarantined_columns.append(
                    {
                        "source_column": column,
                        "action": "excluded_from_analysis_ready_dataset",
                        "reason": "low_name_value_semantic_conformance",
                        "conformance_pct": conformance,
                        "classified_value_count": classified_count,
                    }
                )

        destination_dir = Path(output_dir) if output_dir else Path(file_path).parent / "derived"
        destination_dir.mkdir(parents=True, exist_ok=True)
        try:
            destination = resolve_output_path(destination_dir, f"{Path(file_path).stem}_structured.csv")
        except UploadValidationError as exc:
            raise ToolExecutionError(str(exc)) from exc
        output.to_csv(destination, index=False)
        invalidate_read_cache(str(destination))

        return {
            "summary": (
                f"Created {len(provenance)} typed derived feature(s) from {len(df.columns)} source column(s); "
                f"semantic readiness is {readiness['readiness_score']:.1f}/100. Saved to '{destination}'."
            ),
            "derived_dataset": {
                "name": "analysis_ready_structured",
                "path": str(destination),
                "rows": len(output),
                "source_columns": len(df.columns),
                "added_columns": len(provenance),
                "total_columns": len(output.columns),
                "columns": [str(column) for column in output.columns],
            },
            "provenance": provenance,
            "quarantined_columns": quarantined_columns,
            "semantic_readiness": readiness,
        }

    def get_schema(self) -> dict[str, Any]:
        """Return the planner-visible tool schema."""
        return {
            "file_path": {"type": "string", "description": "Path to the cleaned dataset.", "required": True},
            "output_dir": {
                "type": "string",
                "description": "Directory for the additive derived CSV; defaults to this run's derived output folder.",
                "required": False,
            },
        }
