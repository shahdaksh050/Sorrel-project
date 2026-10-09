"""Tests for deterministic causal-design validation."""
from __future__ import annotations

import pytest

from src.core.causal_design import (
    CausalDesignProposal,
    objective_requests_causality,
    validate_adversarial_critique,
    validate_causal_design,
)

COLUMNS = ("treatment", "outcome", "age", "date", "customer_id", "score", "z")


def test_randomized_route_requires_assignment_assumption() -> None:
    result = validate_causal_design(
        {"treatment": "treatment", "outcome": "outcome", "study_design": "randomized_experiment", "assumptions": ["random_assignment"]},
        COLUMNS,
        "randomized_experiment",
    )
    assert result.status == "identified"
    assert result.estimator == "intention_to_treat_or_ancova"


def test_untrusted_column_name_is_refused() -> None:
    result = validate_causal_design(
        {"treatment": "not_a_column", "outcome": "outcome", "study_design": "randomized_experiment", "assumptions": ["random_assignment"]}, COLUMNS,
    )
    assert result.status == "not_identified"
    assert "valid treatment column" in result.missing_requirements


def test_observational_route_is_only_partially_identified() -> None:
    result = validate_causal_design(
        {"treatment": "treatment", "outcome": "outcome", "pre_treatment_confounders": ["age"], "study_design": "observational cross section", "assumptions": ["conditional exchangeability", "positivity", "consistency"]}, COLUMNS,
    )
    assert result.status == "partially_identified"
    assert result.estimator == "doubly_robust_adjustment"
    assert result.warnings


def test_mediator_or_collider_cannot_be_adjusted_for() -> None:
    result = validate_causal_design(
        {"treatment": "treatment", "outcome": "outcome", "pre_treatment_confounders": ["age"], "mediators": ["age"], "study_design": "observational_cross_section", "assumptions": ["conditional_exchangeability", "positivity", "consistency"]}, COLUMNS,
    )
    assert result.status == "not_identified"
    assert "remove mediator/collider" in result.missing_requirements[0]


def test_design_conflict_is_refused() -> None:
    result = validate_causal_design(
        {"treatment": "treatment", "outcome": "outcome", "study_design": "randomized_experiment", "assumptions": ["random_assignment"]}, COLUMNS, "instrumental_variables",
    )
    assert result.status == "not_identified"
    assert "conflict" in result.warnings[0]


def test_difference_in_differences_routes_with_required_metadata() -> None:
    result = validate_causal_design(
        {"treatment": "treatment", "outcome": "outcome", "time_column": "date", "entity_column": "customer_id", "study_design": "panel_difference_in_differences", "assumptions": ["parallel_trends", "no_anticipation"]}, COLUMNS,
    )
    assert result.status == "partially_identified"
    assert result.estimator == "difference_in_differences_event_study"


def test_regression_discontinuity_needs_cutoff() -> None:
    result = validate_causal_design(
        {"treatment": "treatment", "outcome": "outcome", "running_variable": "score", "study_design": "regression_discontinuity", "assumptions": ["continuity", "no_sorting_at_cutoff"]}, COLUMNS,
    )
    assert result.status == "not_identified"
    assert "cutoff" in result.missing_requirements[0]


def test_iv_requires_instrument_and_explicit_assumptions() -> None:
    result = validate_causal_design(
        {"treatment": "treatment", "outcome": "outcome", "instruments": ["z"], "study_design": "instrumental_variables", "assumptions": ["instrument_relevance", "exclusion_restriction", "instrument_independence"]}, COLUMNS,
    )
    assert result.status == "partially_identified"
    assert result.estimator == "two_stage_least_squares"


@pytest.mark.parametrize(
    ("design", "proposal", "estimator"),
    [
        (
            "interrupted_time_series",
            {"treatment": "treatment", "outcome": "outcome", "time_column": "date", "assumptions": ["no_concurrent_interventions", "stable_measurement"]},
            "segmented_regression",
        ),
        (
            "survival",
            {"treatment": "treatment", "outcome": "outcome", "time_column": "date", "assumptions": ["independent_censoring", "positivity"]},
            "weighted_survival_model",
        ),
        (
            "repeated_measures",
            {"treatment": "treatment", "outcome": "outcome", "time_column": "date", "entity_column": "customer_id", "assumptions": ["conditional_exchangeability", "positivity", "no_time_varying_unmeasured_confounding"]},
            "marginal_structural_model",
        ),
    ],
)
def test_supported_routes_are_conservative(
    design: str, proposal: dict[str, object], estimator: str
) -> None:
    result = validate_causal_design({**proposal, "study_design": design}, COLUMNS)
    assert result.status == "partially_identified"
    assert result.estimator == estimator


def test_not_requested_and_serializable_result() -> None:
    result = validate_causal_design({"causal_requested": False}, COLUMNS)
    assert result.status == "not_requested"
    assert result.to_dict()["status"] == "not_requested"


def test_prompt_aliases_are_validated_and_roles_are_serialized() -> None:
    result = validate_causal_design(
        {
            "requested": True,
            "design": "observational",
            "treatment": "treatment",
            "outcome": "outcome",
            "confounders": ["age"],
            "assumptions": ["conditional_exchangeability", "positivity", "consistency"],
        },
        COLUMNS,
    )

    payload = result.to_dict()
    assert result.status == "partially_identified"
    assert payload["treatment"] == "treatment"
    assert payload["outcome"] == "outcome"
    assert payload["pre_treatment_confounders"] == ["age"]


def test_proposal_bounds_and_sanitizes_untrusted_text() -> None:
    proposal = CausalDesignProposal.from_mapping({"treatment": "treatment\n" + "x" * 500, "assumptions": ["random_assignment", 1]})
    assert proposal.treatment is not None
    assert "\n" not in proposal.treatment
    assert len(proposal.treatment) == 240
    assert proposal.assumptions == ("random_assignment",)


def test_randomization_claimed_only_by_llm_is_not_confirmed() -> None:
    result = validate_causal_design(
        {"treatment": "treatment", "outcome": "outcome", "study_design": "randomized_experiment", "assumptions": ["random_assignment"]},
        COLUMNS,
    )
    assert result.status == "partially_identified"


@pytest.mark.parametrize(
    ("objective", "expected"),
    [
        ("Estimate the causal effect of treatment on recovery", True),
        ("What is the impact of price on demand?", True),
        ("Does the intervention reduce readmissions?", True),
        ("Find variables associated with recovery", False),
        ("What predicts churn?", False),
        ("", False),
    ],
)
def test_causal_intent_comes_from_user_objective(objective: str, expected: bool) -> None:
    assert objective_requests_causality(objective) is expected


def test_adversarial_critique_validation_sanitizes_and_extracts_threats() -> None:
    critique = validate_adversarial_critique({
        "critique_summary": "Unmeasured patient socioeconomic status could confound treatment assignment.",
        "plausible_unmeasured_confounders": ["household_income", "baseline_diet"],
        "reverse_causality_risk": "high risk because sicker patients seek treatment",
        "mediator_collider_concerns": ["post_recovery_score"],
        "untestable_assumption_caveats": ["Assumes no spillover between clinics."],
        "recommendation": "conditional_interpretation",
    })

    assert "socioeconomic status" in critique.critique_summary
    assert critique.plausible_unmeasured_confounders == ("household_income", "baseline_diet")
    assert "high" in critique.reverse_causality_risk
    assert critique.mediator_collider_concerns == ("post_recovery_score",)
    assert critique.untestable_assumption_caveats == ("Assumes no spillover between clinics.",)
    assert critique.recommendation == "conditional_interpretation"
    assert any("unmeasured confounders" in w for w in critique.warnings)
    assert any("reverse causality" in w for w in critique.warnings)


def test_adversarial_critique_reject_recommendation() -> None:
    critique = validate_adversarial_critique({
        "critique_summary": "Critical collider stratification makes effect uninterpretable.",
        "recommendation": "reject_causal_claim",
    })

    assert critique.recommendation == "reject_causal_claim"


def test_adversarial_critique_fallback_on_none_or_malformed() -> None:
    critique = validate_adversarial_critique(None, default_summary="Observational fallback.")
    assert critique.critique_summary == "Observational fallback."
    assert critique.recommendation == "conditional_interpretation"
    assert len(critique.warnings) >= 1

