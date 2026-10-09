"""Deterministic semantic-readiness checks for heterogeneous tabular data.

The profiler describes how a value is stored.  This module separately checks
whether a column's *name* agrees with the units and concepts encoded in its
values.  It deliberately returns counts and coverage only: values from an
uploaded dataset must not become prompt material through this diagnostic.
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Any

import pandas as pd

_MAX_ROWS_TO_SCAN = 10_000

_CURRENCY_RE = re.compile(r"(?:₹|\b(?:inr|rs\.?)\b|[$€£¥])\s*\d", re.IGNORECASE)
_MEMORY_RE = re.compile(r"\b\d+(?:\.\d+)?\s*(?:mb|gb|tb)\b", re.IGNORECASE)
_CAPACITY_RE = re.compile(r"\b\d+(?:\.\d+)?\s*(?:mah|ah|wh|kwh)\b", re.IGNORECASE)
_POWER_RE = re.compile(r"\b\d+(?:\.\d+)?\s*(?:w|watt|kw)\b", re.IGNORECASE)
_DISPLAY_RE = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:in(?:ch(?:es)?)?|cm|mm|ft|m)\b|\b\d{3,5}\s*[x×]\s*\d{3,5}\b",
    re.IGNORECASE,
)
_RESOLUTION_RE = re.compile(r"\b\d+(?:\.\d+)?\s*(?:mp|megapixel)\b|\b\d{3,5}\s*[x×]\s*\d{3,5}\b", re.IGNORECASE)
_FREQUENCY_RE = re.compile(r"\b\d+(?:\.\d+)?\s*(?:ghz|mhz|khz|hz)\b", re.IGNORECASE)
_PLATFORM_RE = re.compile(
    r"\b(?:android|ios|ipados|windows(?:\s+phone)?|macos|linux|harmonyos|chrome\s*os)\b",
    re.IGNORECASE,
)
_CONNECTIVITY_RE = re.compile(r"\b(?:5g|4g|lte|wifi|wi-fi|bluetooth|ethernet|nfc|gps)\b", re.IGNORECASE)
_CAPABILITY_RE = re.compile(
    r"\b(?:supports?|support|enabled?|available|fast\s+charg(?:e|ing)|fm\s+radio|dual\s+sim|memory\s+card)\b",
    re.IGNORECASE,
)

_FAMILIES: dict[str, tuple[tuple[str, ...], re.Pattern[str]]] = {
    "currency": (("price", "cost", "amount", "revenue", "salary", "income", "fare", "fee", "currency"), _CURRENCY_RE),
    "memory_storage": (("ram", "memory", "storage", "rom", "disk"), _MEMORY_RE),
    "energy_capacity": (("battery", "capacity", "energy"), _CAPACITY_RE),
    "power": (("power", "watt", "charging", "charger"), _POWER_RE),
    "display_dimension": (("display", "screen", "dimension", "height", "width", "length", "size"), _DISPLAY_RE),
    "resolution": (("camera", "resolution", "image", "pixel", "sensor"), _RESOLUTION_RE),
    "frequency": (("processor", "cpu", "clock", "frequency", "speed"), _FREQUENCY_RE),
    "platform": (("os", "platform", "operating", "system"), _PLATFORM_RE),
    "connectivity": (("network", "connectivity", "wireless", "sim", "communication"), _CONNECTIVITY_RE),
    "capability": (("feature", "capability", "support", "card", "radio", "ports"), _CAPABILITY_RE),
}

_NUMERIC_FAMILIES = frozenset({
    "currency",
    "memory_storage",
    "energy_capacity",
    "power",
    "display_dimension",
    "resolution",
    "frequency",
})


def _normalise_name(column: object) -> str:
    """Return a lower-case token string for a column name."""
    return re.sub(r"[^a-z0-9]+", " ", str(column).lower()).strip()


def _expected_families(column: object) -> list[str]:
    """Infer semantic families suggested by a column name, without values."""
    name = _normalise_name(column)
    words = set(name.split())
    expected: list[str] = []
    for family, (hints, _pattern) in _FAMILIES.items():
        if any(hint in words or hint in name for hint in hints):
            expected.append(family)
    return expected


def _observed_values(series: pd.Series) -> pd.Series:
    """Return the deterministic bounded set of non-null values to inspect."""
    values = series.dropna().astype(str)
    if len(values) > _MAX_ROWS_TO_SCAN:
        return values.sample(n=_MAX_ROWS_TO_SCAN, random_state=42)
    return values


def _semantic_counts(values: pd.Series) -> Counter[str]:
    """Count recognised semantic unit/concept signatures in inspected cells."""
    counts: Counter[str] = Counter()
    for value in values:
        for family, (_hints, pattern) in _FAMILIES.items():
            if pattern.search(value):
                counts[family] += 1
    return counts


def assess_semantic_readiness(df: pd.DataFrame) -> dict[str, Any]:
    """Assess name/value semantic agreement without exposing raw cell values.

    A column is assessed when either its name signals a recognised semantic
    family or its non-null values contain a unit/platform/capability pattern.
    ``readiness_score`` is a diagnostic prioritisation signal, not a claim
    that data are statistically valid.
    """
    columns: list[dict[str, Any]] = []
    total_weight = 0
    weighted_score = 0.0
    issues_total = 0

    for column in df.columns:
        values = _observed_values(df[column])
        non_null_count = len(values)
        detected_counts = _semantic_counts(values)
        expected = _expected_families(column)
        if not expected and not detected_counts:
            continue

        coverage = {
            family: round(100.0 * count / max(non_null_count, 1), 2)
            for family, count in sorted(detected_counts.items())
        }
        classified_value_count = sum(
            1
            for value in values
            if any(pattern.search(value) for _hints, pattern in _FAMILIES.values())
        )
        expected_match_count = sum(
            1
            for value in values
            if any(_FAMILIES[family][1].search(value) for family in expected)
        )
        numeric_semantic = (
            bool(expected)
            and pd.api.types.is_numeric_dtype(df[column])
            and any(family in _NUMERIC_FAMILIES for family in expected)
        )
        conformance_pct = (
            100.0
            if numeric_semantic
            else round(100.0 * expected_match_count / max(non_null_count, 1), 2)
            if expected
            else None
        )
        primary_detected = max(detected_counts, key=lambda family: detected_counts[family]) if detected_counts else None
        issues: list[str] = []
        if expected and conformance_pct is not None and conformance_pct < 50.0:
            issues.append("name_value_misalignment")
        unexpected = [family for family in detected_counts if family not in expected]
        if expected and unexpected and max(detected_counts[family] for family in unexpected) / max(non_null_count, 1) >= 0.25:
            issues.append("mixed_or_shifted_semantics")
        if not expected and len(detected_counts) > 1:
            issues.append("multiple_semantic_signatures")
        if non_null_count == 0:
            issues.append("no_observed_values")

        if expected:
            score = float(conformance_pct or 0.0)
        elif primary_detected:
            score = coverage[primary_detected]
        else:
            score = 0.0
        total_weight += max(non_null_count, 1)
        weighted_score += score * max(non_null_count, 1)
        issues_total += len(issues)
        columns.append(
            {
                "column": str(column),
                "non_null_count": non_null_count,
                "classified_value_count": classified_value_count,
                "expected_semantics": expected,
                "detected_semantic_coverage_pct": coverage,
                "primary_detected_semantic": primary_detected,
                "conformance_pct": conformance_pct,
                "issues": issues,
            }
        )

    readiness_score = round(weighted_score / total_weight, 2) if total_weight else 100.0
    ready_columns = sum(1 for result in columns if not result["issues"])
    return {
        "readiness_score": readiness_score,
        "assessment_scope": "name/value semantic conformance; no raw values included",
        "columns_assessed": len(columns),
        "columns_ready": ready_columns,
        "columns_with_issues": len(columns) - ready_columns,
        "issue_count": issues_total,
        "per_column": columns,
    }
