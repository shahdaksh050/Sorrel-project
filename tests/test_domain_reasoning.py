"""Tests for the metadata-only LLM domain-reasoning contract."""
from __future__ import annotations

from src.core.domain_reasoning import validate_domain_analysis_spec


def test_valid_reply_is_grounded_and_serializable() -> None:
    result = validate_domain_analysis_spec(
        {
            "row_entity": "one retail transaction",
            "domain": "retail commerce",
            "archetype": "transaction log",
            "key_measures": ["revenue"],
            "dimensions": ["region", "order_date"],
            "units": {"revenue": "USD", "region": "not a unit"},
            "domain_questions": ["Which regions have the highest revenue?"],
            "suggested_methods": ["grouped descriptive statistics"],
            "assumptions": ["Revenue is recorded consistently."],
            "prohibited_analyses": ["Do not make causal claims from this observational extract."],
            "confidence": 0.8,
        },
        ["revenue", "region", "order_date"],
        {"row_count": 100},
    )

    assert result.spec.key_measures == ("revenue",)
    assert result.spec.dimensions == ("region", "order_date")
    assert result.spec.units == {"revenue": "USD", "region": "not a unit"}
    assert result.readiness == "ready"
    assert result.confidence == 0.8
    assert result.to_dict()["spec"]["domain"] == "retail commerce"


def test_unknown_columns_and_unsafe_text_are_omitted() -> None:
    result = validate_domain_analysis_spec(
        {
            "row_entity": "exec('bad')",
            "key_measures": ["revenue", "missing"],
            "dimensions": "region",
            "units": {"missing": "USD", "revenue": "```python"},
            "domain_questions": ["https://example.invalid", "Safe question?"],
        },
        ["revenue", "region"],
    )

    assert result.spec.row_entity is None
    assert result.spec.key_measures == ("revenue",)
    assert result.spec.dimensions == ()
    assert result.spec.units == {}
    assert result.spec.domain_questions == ("Safe question?",)
    assert result.readiness == "partial"
    assert any("unknown column" in warning for warning in result.warnings)
    assert any("unsafe" in warning for warning in result.warnings)


def test_bounds_confidence_and_non_finite_values() -> None:
    long_items = [f"Question {number}" for number in range(20)]
    result = validate_domain_analysis_spec(
        {"domain_questions": long_items, "confidence": float("inf")},
        ["value"],
    )

    assert len(result.spec.domain_questions) == 12
    assert result.confidence == 0.07
    assert any("non-finite" in warning for warning in result.warnings)
    assert any("Truncated 'domain_questions'" in warning for warning in result.warnings)


def test_exact_column_matching_prevents_normalized_aliases() -> None:
    result = validate_domain_analysis_spec(
        {"key_measures": ["Revenue", "revenue"]},
        ["Revenue"],
    )

    assert result.spec.key_measures == ("Revenue",)
    assert any("unknown column" in warning for warning in result.warnings)


def test_prompt_key_dimensions_alias_is_accepted() -> None:
    result = validate_domain_analysis_spec(
        {"key_dimensions": ["region"]},
        ["region"],
    )

    assert result.spec.dimensions == ("region",)
