"""Validated, bounded contract for LLM-authored domain interpretation.

This module deliberately contains no LLM calls, dataset I/O, or raw values.
It turns a *candidate* structured reply into a conservative, serializable
description that downstream planning may inspect.  A candidate cannot name a
column that was not supplied by the dataset metadata, and code-like text is
discarded rather than becoming instructions for another component.
"""
from __future__ import annotations

import math
import re
from collections.abc import Collection, Mapping
from dataclasses import asdict, dataclass, field
from typing import Any

MAX_TEXT_LENGTH = 240
MAX_LIST_ITEMS = 12
MAX_COLUMNS_PER_ROLE = 20
MAX_UNITS = 30

_CODE_LIKE = re.compile(
    r"(?:```|\b(?:import|exec|eval|lambda|subprocess|powershell|bash|cmd)\b|"
    r"__\w+__|\b(?:os|sys|pathlib)\s*\.|[{};`]|https?://)",
    re.IGNORECASE,
)
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_SPACE = re.compile(r"\s+")


@dataclass(frozen=True)
class DomainAnalysisSpec:
    """Safe, grounded semantic proposal from an LLM response."""

    row_entity: str | None = None
    domain: str | None = None
    archetype: str | None = None
    key_measures: tuple[str, ...] = ()
    dimensions: tuple[str, ...] = ()
    units: dict[str, str] = field(default_factory=dict)
    domain_questions: tuple[str, ...] = ()
    suggested_methods: tuple[str, ...] = ()
    assumptions: tuple[str, ...] = ()
    prohibited_analyses: tuple[str, ...] = ()
    missing_information: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-serializable contract data with no raw dataset rows."""
        return asdict(self)


@dataclass(frozen=True)
class DomainSpecValidation:
    """Validation output, including deterministic quality signals."""

    spec: DomainAnalysisSpec
    warnings: tuple[str, ...]
    readiness: str
    readiness_score: float
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable result appropriate for planner context."""
        return {
            "spec": self.spec.to_dict(),
            "warnings": list(self.warnings),
            "readiness": self.readiness,
            "readiness_score": self.readiness_score,
            "confidence": self.confidence,
        }


def _safe_text(value: object, field_name: str, warnings: list[str]) -> str | None:
    """Accept one bounded, non-executable plain-text value."""
    if not isinstance(value, str):
        if value is not None:
            warnings.append(f"Discarded non-text value for '{field_name}'.")
        return None
    text = _SPACE.sub(" ", _CONTROL_CHARS.sub("", value)).strip()
    if not text:
        return None
    if _CODE_LIKE.search(text):
        warnings.append(f"Discarded unsafe or code-like text for '{field_name}'.")
        return None
    if len(text) > MAX_TEXT_LENGTH:
        warnings.append(f"Truncated overlong text for '{field_name}'.")
        text = text[:MAX_TEXT_LENGTH].rstrip()
    return text


def _safe_text_list(value: object, field_name: str, warnings: list[str]) -> tuple[str, ...]:
    """Normalize a bounded list of safe text, keeping its original order."""
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)):
        warnings.append(f"Discarded non-list value for '{field_name}'.")
        return ()
    texts: list[str] = []
    for index, item in enumerate(value):
        if index >= MAX_LIST_ITEMS:
            warnings.append(f"Truncated '{field_name}' to {MAX_LIST_ITEMS} items.")
            break
        cleaned = _safe_text(item, field_name, warnings)
        if cleaned is not None and cleaned not in texts:
            texts.append(cleaned)
    return tuple(texts)


def _known_columns(
    value: object, field_name: str, columns: frozenset[str], warnings: list[str]
) -> tuple[str, ...]:
    """Keep only exact supplied column names, without fuzzy matching."""
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)):
        warnings.append(f"Discarded non-list value for '{field_name}'.")
        return ()
    result: list[str] = []
    for index, item in enumerate(value):
        if index >= MAX_COLUMNS_PER_ROLE:
            warnings.append(f"Truncated '{field_name}' to {MAX_COLUMNS_PER_ROLE} columns.")
            break
        if not isinstance(item, str) or item not in columns:
            warnings.append(f"Omitted unknown column from '{field_name}'.")
        elif item not in result:
            result.append(item)
    return tuple(result)


def _units(
    value: object, columns: frozenset[str], warnings: list[str]
) -> dict[str, str]:
    """Keep safe units keyed only by exact supplied column names."""
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        warnings.append("Discarded non-mapping value for 'units'.")
        return {}
    result: dict[str, str] = {}
    for index, (column, unit) in enumerate(value.items()):
        if index >= MAX_UNITS:
            warnings.append(f"Truncated 'units' to {MAX_UNITS} entries.")
            break
        if not isinstance(column, str) or column not in columns:
            warnings.append("Omitted unit for an unknown column.")
            continue
        cleaned = _safe_text(unit, "units", warnings)
        if cleaned is not None:
            result[column] = cleaned
    return result


def _reported_confidence(value: object, warnings: list[str]) -> float | None:
    """Return a finite probability supplied by the model, if one was supplied."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        warnings.append("Discarded non-numeric 'confidence'.")
        return None
    confidence = float(value)
    if not math.isfinite(confidence):
        warnings.append("Discarded non-finite 'confidence'.")
        return None
    if not 0.0 <= confidence <= 1.0:
        warnings.append("Clamped 'confidence' to the range 0..1.")
        confidence = min(1.0, max(0.0, confidence))
    return confidence


def validate_domain_analysis_spec(
    reply: Mapping[str, object],
    dataset_columns: Collection[str],
    metadata: Mapping[str, object] | None = None,
) -> DomainSpecValidation:
    """Validate an LLM domain-analysis reply against metadata-only context.

    ``dataset_columns`` must be the exact column names from the loaded
    dataset. ``metadata`` is intentionally accepted for call-site symmetry
    with profile-driven prompts, but is not inspected for values or emitted;
    this protects the no-raw-rows boundary.  Readiness measures how grounded
    the proposal is, while confidence combines that score with any model
    self-rating so an unsupported high-confidence reply cannot look reliable.
    """
    del metadata
    warnings: list[str] = []
    if not isinstance(reply, Mapping):
        warnings.append("Discarded non-mapping domain-analysis reply.")
        reply = {}
    columns = frozenset(column for column in dataset_columns if isinstance(column, str))
    if not columns:
        warnings.append("No valid dataset columns were supplied for grounding.")

    spec = DomainAnalysisSpec(
        row_entity=_safe_text(reply.get("row_entity"), "row_entity", warnings),
        domain=_safe_text(reply.get("domain"), "domain", warnings),
        archetype=_safe_text(reply.get("archetype"), "archetype", warnings),
        key_measures=_known_columns(reply.get("key_measures"), "key_measures", columns, warnings),
        dimensions=_known_columns(
            reply.get("dimensions", reply.get("key_dimensions")),
            "dimensions",
            columns,
            warnings,
        ),
        units=_units(reply.get("units"), columns, warnings),
        domain_questions=_safe_text_list(reply.get("domain_questions"), "domain_questions", warnings),
        suggested_methods=_safe_text_list(reply.get("suggested_methods"), "suggested_methods", warnings),
        assumptions=_safe_text_list(reply.get("assumptions"), "assumptions", warnings),
        prohibited_analyses=_safe_text_list(reply.get("prohibited_analyses"), "prohibited_analyses", warnings),
        missing_information=_safe_text_list(reply.get("missing_information"), "missing_information", warnings),
    )
    score_parts = (
        (0.15, spec.domain is not None),
        (0.15, spec.row_entity is not None),
        (0.20, bool(spec.key_measures or spec.dimensions)),
        (0.15, bool(spec.domain_questions)),
        (0.15, bool(spec.suggested_methods)),
        (0.10, bool(spec.assumptions)),
        (0.10, bool(spec.prohibited_analyses)),
    )
    readiness_score = round(sum(weight for weight, present in score_parts if present), 2)
    readiness = "ready" if readiness_score >= 0.7 else "partial" if readiness_score >= 0.35 else "insufficient"
    reported = _reported_confidence(reply.get("confidence"), warnings)
    confidence = round(readiness_score * (reported if reported is not None else 0.5), 2)
    return DomainSpecValidation(
        spec=spec,
        warnings=tuple(warnings),
        readiness=readiness,
        readiness_score=readiness_score,
        confidence=confidence,
    )
