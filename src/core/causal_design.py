"""Deterministic validation and routing for proposed causal analyses.

This module deliberately does not inspect rows or call an LLM.  An LLM may
*propose* a design, but a proposal only becomes actionable after its named
columns, study design and required identifying assumptions pass these gates.
"""
from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from math import isfinite
from typing import Any, Literal

CausalStatus = Literal[
    "identified", "partially_identified", "not_identified", "not_requested"
]

_MAX_TEXT_LENGTH = 240
_MAX_ITEMS = 32
_ALLOWED_DESIGNS = frozenset(
    {
        "randomized_experiment",
        "observational_cross_section",
        "panel_difference_in_differences",
        "interrupted_time_series",
        "regression_discontinuity",
        "instrumental_variables",
        "survival",
        "repeated_measures",
    }
)
_SPACE = re.compile(r"\s+")
_CONTROL = re.compile(r"[\x00-\x1f\x7f]")
_CAUSAL_INTENT = re.compile(
    r"\b(?:caus(?:e|es|al|ality)|causal effect|treatment effect|intervention effect|"
    r"counterfactual|what would happen if|difference[- ]in[- ]differences|diff[- ]in[- ]diff|"
    r"regression discontinuity|instrumental variable|randomi[sz]ed|a/b test|ab test)\b|"
    r"\b(?:impact|effect)\s+of\b.+\bon\b|\bdoes\b.+\b(?:affect|change|increase|reduce)\b",
    re.IGNORECASE,
)


def objective_requests_causality(objective: str) -> bool:
    """Whether the user's own words explicitly request a causal answer."""
    return bool(_CAUSAL_INTENT.search(objective or ""))


def _bounded_text(value: object) -> str | None:
    """Return a bounded, single-line scalar string; reject non-strings."""
    if not isinstance(value, str):
        return None
    cleaned = _SPACE.sub(" ", _CONTROL.sub(" ", value)).strip()
    return cleaned[:_MAX_TEXT_LENGTH] or None


def _bounded_strings(value: object) -> tuple[str, ...]:
    """Normalize a proposed string list without accepting arbitrary objects."""
    if not isinstance(value, (list, tuple, set, frozenset)):
        return ()
    values: list[str] = []
    for item in value:
        cleaned = _bounded_text(item)
        if cleaned and cleaned not in values:
            values.append(cleaned)
        if len(values) >= _MAX_ITEMS:
            break
    return tuple(values)


@dataclass(frozen=True)
class CausalDesignProposal:
    """An LLM-authored causal design proposal, before deterministic validation."""

    treatment: str | None = None
    outcome: str | None = None
    study_design: str | None = None
    pre_treatment_confounders: tuple[str, ...] = ()
    mediators: tuple[str, ...] = ()
    colliders: tuple[str, ...] = ()
    instruments: tuple[str, ...] = ()
    time_column: str | None = None
    entity_column: str | None = None
    running_variable: str | None = None
    cutoff: float | None = None
    assumptions: tuple[str, ...] = ()
    estimand: str | None = None
    causal_requested: bool = True

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> CausalDesignProposal:
        """Safely construct a proposal from untrusted structured LLM output."""
        cutoff_value = value.get("cutoff")
        cutoff = None
        if isinstance(cutoff_value, (int, float)) and not isinstance(cutoff_value, bool):
            candidate = float(cutoff_value)
            cutoff = candidate if isfinite(candidate) else None
        requested = value.get("causal_requested", value.get("requested", True))
        return cls(
            treatment=_bounded_text(value.get("treatment")),
            outcome=_bounded_text(value.get("outcome")),
            study_design=_bounded_text(value.get("study_design", value.get("design"))),
            pre_treatment_confounders=_bounded_strings(
                value.get("pre_treatment_confounders", value.get("confounders"))
            ),
            mediators=_bounded_strings(value.get("mediators")),
            colliders=_bounded_strings(value.get("colliders")),
            instruments=_bounded_strings(value.get("instruments")),
            time_column=_bounded_text(value.get("time_column")),
            entity_column=_bounded_text(value.get("entity_column")),
            running_variable=_bounded_text(value.get("running_variable")),
            cutoff=cutoff,
            assumptions=_bounded_strings(value.get("assumptions")),
            estimand=_bounded_text(value.get("estimand")),
            causal_requested=requested is not False,
        )


@dataclass(frozen=True)
class CausalDesignResult:
    """Serializable, conservative decision emitted by the identification gate."""

    status: CausalStatus
    study_design: str | None
    estimand: str | None
    estimator: str | None
    diagnostics_required: tuple[str, ...] = ()
    assumptions: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    missing_requirements: tuple[str, ...] = ()
    treatment: str | None = None
    outcome: str | None = None
    pre_treatment_confounders: tuple[str, ...] = ()
    mediators: tuple[str, ...] = ()
    colliders: tuple[str, ...] = ()
    instruments: tuple[str, ...] = ()
    time_column: str | None = None
    entity_column: str | None = None
    running_variable: str | None = None
    cutoff: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-serializable primitives only."""
        return {
            "status": self.status,
            "study_design": self.study_design,
            "design": self.study_design,
            "estimand": self.estimand,
            "estimator": self.estimator,
            "diagnostics_required": list(self.diagnostics_required),
            "assumptions": list(self.assumptions),
            "warnings": list(self.warnings),
            "missing_requirements": list(self.missing_requirements),
            "treatment": self.treatment,
            "outcome": self.outcome,
            "pre_treatment_confounders": list(self.pre_treatment_confounders),
            "mediators": list(self.mediators),
            "colliders": list(self.colliders),
            "instruments": list(self.instruments),
            "time_column": self.time_column,
            "entity_column": self.entity_column,
            "running_variable": self.running_variable,
            "cutoff": self.cutoff,
        }


def _normalise_design(value: str | None) -> str | None:
    if value is None:
        return None
    design = value.lower().replace("-", "_").replace(" ", "_")
    if design == "observational":
        design = "observational_cross_section"
    return design if design in _ALLOWED_DESIGNS else None


def _requirements_for(design: str) -> tuple[tuple[str, ...], tuple[str, ...], str, str]:
    """Columns, assumptions, estimator and diagnostics for a known design."""
    routes: dict[str, tuple[tuple[str, ...], tuple[str, ...], str, str]] = {
        "randomized_experiment": (
            ("treatment", "outcome"), ("random_assignment",),
            "intention_to_treat_or_ancova", "randomization balance and attrition",
        ),
        "observational_cross_section": (
            ("treatment", "outcome", "pre_treatment_confounders"),
            ("conditional_exchangeability", "positivity", "consistency"),
            "doubly_robust_adjustment", "covariate balance, overlap and sensitivity analysis",
        ),
        "panel_difference_in_differences": (
            ("treatment", "outcome", "time_column", "entity_column"),
            ("parallel_trends", "no_anticipation"),
            "difference_in_differences_event_study", "pre-trends, treatment timing and clustered standard errors",
        ),
        "interrupted_time_series": (
            ("treatment", "outcome", "time_column"),
            ("no_concurrent_interventions", "stable_measurement"),
            "segmented_regression", "autocorrelation, seasonality and placebo interruptions",
        ),
        "regression_discontinuity": (
            ("treatment", "outcome", "running_variable", "cutoff"),
            ("continuity", "no_sorting_at_cutoff"),
            "local_polynomial_regression_discontinuity", "density, covariate continuity and bandwidth sensitivity",
        ),
        "instrumental_variables": (
            ("treatment", "outcome", "instruments"),
            ("instrument_relevance", "exclusion_restriction", "instrument_independence"),
            "two_stage_least_squares", "first-stage strength, weak-instrument robust inference and exclusion review",
        ),
        "survival": (
            ("treatment", "outcome", "time_column"),
            ("independent_censoring", "positivity"),
            "weighted_survival_model", "censoring, proportional hazards and overlap",
        ),
        "repeated_measures": (
            ("treatment", "outcome", "entity_column", "time_column"),
            ("conditional_exchangeability", "positivity", "no_time_varying_unmeasured_confounding"),
            "marginal_structural_model", "within-entity support, treatment timing and time-varying confounding",
        ),
    }
    return routes[design]


def validate_causal_design(
    proposal: CausalDesignProposal | Mapping[str, object] | None,
    dataset_columns: Iterable[str],
    known_study_design: str | None = None,
) -> CausalDesignResult:
    """Validate a proposal against exact metadata and return a safe analysis route.

    ``known_study_design`` is authoritative where supplied.  Neither that
    value nor an LLM assertion establishes untestable assumptions; those are
    returned explicitly and must be audited by an execution tool.
    """
    if proposal is None:
        return CausalDesignResult("not_requested", None, None, None)
    parsed = proposal if isinstance(proposal, CausalDesignProposal) else CausalDesignProposal.from_mapping(proposal)
    if not parsed.causal_requested:
        return CausalDesignResult("not_requested", None, None, None)

    proposed_design = _normalise_design(parsed.study_design)
    known_design = _normalise_design(_bounded_text(known_study_design))
    warnings: list[str] = []
    if parsed.study_design and proposed_design is None:
        warnings.append("The proposed study design is not supported.")
    if known_study_design and known_design is None:
        warnings.append("The supplied known study design is not supported.")
    if proposed_design and known_design and proposed_design != known_design:
        return CausalDesignResult(
            "not_identified", known_design, None, None,
            warnings=("Proposed and known study designs conflict; causal routing was refused.",),
            missing_requirements=("a single verified study design",),
        )
    design = known_design or proposed_design
    if design is None:
        return CausalDesignResult(
            "not_identified", None, None, None, warnings=tuple(warnings),
            missing_requirements=("a supported, verified study design",),
        )

    metadata_columns = {column for column in dataset_columns if isinstance(column, str)}
    named_columns: dict[str, tuple[str, ...]] = {
        "treatment": (parsed.treatment,) if parsed.treatment else (),
        "outcome": (parsed.outcome,) if parsed.outcome else (),
        "pre_treatment_confounders": parsed.pre_treatment_confounders,
        "instruments": parsed.instruments,
        "time_column": (parsed.time_column,) if parsed.time_column else (),
        "entity_column": (parsed.entity_column,) if parsed.entity_column else (),
        "running_variable": (parsed.running_variable,) if parsed.running_variable else (),
    }
    required_columns, required_assumptions, estimator, diagnostics = _requirements_for(design)
    missing: list[str] = []
    for role in required_columns:
        if role == "cutoff":
            if parsed.cutoff is None:
                missing.append("a numeric regression-discontinuity cutoff")
            continue
        candidates = named_columns[role]
        if not candidates:
            missing.append(f"{role.replace('_', ' ')} column")
        elif any(column not in metadata_columns for column in candidates):
            missing.append(f"valid {role.replace('_', ' ')} column")

    forbidden = set(parsed.mediators) | set(parsed.colliders)
    adjusted = set(parsed.pre_treatment_confounders)
    invalid_adjustments = sorted(adjusted & forbidden)
    if invalid_adjustments:
        missing.append("remove mediator/collider variables from the adjustment set")
        warnings.append("Adjustment set contains proposed mediator or collider columns: " + ", ".join(invalid_adjustments) + ".")

    supplied_assumptions = {assumption.lower().replace(" ", "_") for assumption in parsed.assumptions}
    absent_assumptions = [item for item in required_assumptions if item not in supplied_assumptions]
    if absent_assumptions:
        missing.extend(f"assumption: {item.replace('_', ' ')}" for item in absent_assumptions)
    if missing:
        return CausalDesignResult(
            "not_identified", design, None, None, (), tuple(parsed.assumptions),
            tuple(warnings), tuple(dict.fromkeys(missing)),
        )

    estimand = parsed.estimand or "average treatment effect"
    status: CausalStatus = (
        "identified"
        if design == "randomized_experiment" and known_design == "randomized_experiment"
        else "partially_identified"
    )
    if status == "partially_identified":
        warnings.append("Identification is conditional on untestable design assumptions; do not state causation until required diagnostics pass.")
    return CausalDesignResult(
        status=status,
        study_design=design,
        estimand=estimand,
        estimator=estimator,
        diagnostics_required=(diagnostics,),
        assumptions=tuple(parsed.assumptions),
        warnings=tuple(warnings),
        treatment=parsed.treatment,
        outcome=parsed.outcome,
        pre_treatment_confounders=parsed.pre_treatment_confounders,
        mediators=parsed.mediators,
        colliders=parsed.colliders,
        instruments=parsed.instruments,
        time_column=parsed.time_column,
        entity_column=parsed.entity_column,
        running_variable=parsed.running_variable,
        cutoff=parsed.cutoff,
    )


@dataclass(frozen=True)
class AdversarialCritique:
    """Structured, verified adversarial critique of a causal analysis proposal."""

    critique_summary: str
    plausible_unmeasured_confounders: tuple[str, ...] = ()
    reverse_causality_risk: str = "unknown"
    mediator_collider_concerns: tuple[str, ...] = ()
    untestable_assumption_caveats: tuple[str, ...] = ()
    recommendation: str = "conditional_interpretation"
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-serializable primitives only."""
        return {
            "critique_summary": self.critique_summary,
            "plausible_unmeasured_confounders": list(self.plausible_unmeasured_confounders),
            "reverse_causality_risk": self.reverse_causality_risk,
            "mediator_collider_concerns": list(self.mediator_collider_concerns),
            "untestable_assumption_caveats": list(self.untestable_assumption_caveats),
            "recommendation": self.recommendation,
            "warnings": list(self.warnings),
        }


def validate_adversarial_critique(
    reply: Mapping[str, object] | None,
    default_summary: str = "Standard observational cautions apply.",
) -> AdversarialCritique:
    """Deterministically sanitize and validate an adversarial critique reply."""
    if not isinstance(reply, Mapping):
        return AdversarialCritique(
            critique_summary=default_summary,
            warnings=("No structured critique was provided; default observational caveats applied.",),
        )

    summary = _bounded_text(reply.get("critique_summary")) or default_summary
    confounders = _bounded_strings(
        reply.get("plausible_unmeasured_confounders", reply.get("confounders"))
    )
    reverse_risk = _bounded_text(reply.get("reverse_causality_risk")) or "unassessed"
    concerns = _bounded_strings(reply.get("mediator_collider_concerns"))
    caveats = _bounded_strings(
        reply.get("untestable_assumption_caveats", reply.get("assumptions"))
    )

    rec = str(reply.get("recommendation", "conditional_interpretation")).lower()
    recommendation = (
        "reject_causal_claim" if "reject" in rec else "conditional_interpretation"
    )

    warnings: list[str] = []
    if confounders:
        warnings.append(
            f"Adversarial review flagged plausible unmeasured confounders: {', '.join(confounders)}."
        )
    if "high" in reverse_risk.lower():
        warnings.append("Adversarial review identified high risk of reverse causality.")
    if concerns:
        warnings.append(
            f"Adversarial review flagged potential collider/mediator conditioning: {', '.join(concerns)}."
        )

    return AdversarialCritique(
        critique_summary=summary,
        plausible_unmeasured_confounders=confounders,
        reverse_causality_risk=reverse_risk,
        mediator_collider_concerns=concerns,
        untestable_assumption_caveats=caveats,
        recommendation=recommendation,
        warnings=tuple(warnings),
    )
