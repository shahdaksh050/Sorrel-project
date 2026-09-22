"""
Causal-claim guard and study design classifier.

FutureScope Phase 4 (FutureScope.md §5.3):
- Identifies study design (randomized experiment, quasi-experiment, observational).
- Guards against unjustified causal claims in observational studies by downgrading
  causal language ("causes", "drives", "leads to") to association ("is associated with",
  "correlates with") and appending appropriate confounding caveats.
"""
from __future__ import annotations

import re
from typing import Any

import pandas as pd

_EXPERIMENTAL_COL_PATTERNS = (
    re.compile(r"^(?:treatment|variant|arm|experiment_group|test_control|group_assigned)$", re.I),
    re.compile(r"^(?:is_treatment|in_treatment|control_variant)$", re.I),
)

_QUASI_EXPERIMENTAL_PATTERNS = (
    re.compile(r"^(?:pre_post|before_after|post_intervention|post_treatment)$", re.I),
)

_CAUSAL_PHRASES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bcauses\b", re.I), "is associated with"),
    (re.compile(r"\bcaused by\b", re.I), "associated with"),
    (re.compile(r"\bdrives\b", re.I), "correlates with"),
    (re.compile(r"\bdriver of\b", re.I), "correlate of"),
    (re.compile(r"\bdrivers of\b", re.I), "correlates of"),
    (re.compile(r"\bleads to\b", re.I), "is linked to"),
    (re.compile(r"\bimpacts\b", re.I), "is associated with changes in"),
    (re.compile(r"\bresults in\b", re.I), "tends to co-occur with"),
]

CAVEAT_OBSERVATIONAL = (
    "Caveat: This analysis is based on observational data. Uncontrolled confounding "
    "cannot be ruled out; observed effects represent statistical associations rather than "
    "proven causal mechanisms."
)


def classify_study_design(df: pd.DataFrame | None = None, objective: str = "") -> str:
    """
    Classify the data-generating design:
    - 'randomized_experiment'
    - 'quasi_experiment'
    - 'observational'
    """
    if objective:
        obj_lower = objective.lower()
        if any(w in obj_lower for w in ("a/b test", "ab test", "randomized", "experiment arm", "clinical trial")):
            return "randomized_experiment"
        if any(w in obj_lower for w in ("diff-in-diff", "difference in differences", "interrupted time series")):
            return "quasi_experiment"

    if df is not None:
        for col in df.columns:
            for pat in _EXPERIMENTAL_COL_PATTERNS:
                if pat.search(str(col)):
                    return "randomized_experiment"
            for pat in _QUASI_EXPERIMENTAL_PATTERNS:
                if pat.search(str(col)):
                    return "quasi_experiment"

    return "observational"


def guard_causal_claims(
    text: str,
    study_design: str = "observational",
) -> tuple[str, list[str]]:
    """
    Audit and sanitize text for unjustified causal claims if the study design
    is observational.

    Returns:
        (sanitized_text, warnings)
    """
    if study_design == "randomized_experiment":
        return text, []

    warnings: list[str] = []
    sanitized = text

    for pattern, replacement in _CAUSAL_PHRASES:
        matches = pattern.findall(sanitized)
        if matches:
            warnings.append(
                f"Downgraded causal phrasing '{matches[0]}' to '{replacement}' due to observational design."
            )
            sanitized = pattern.sub(replacement, sanitized)

    return sanitized, warnings


def audit_findings_causal_language(
    findings: list[Any],
    study_design: str = "observational",
) -> list[str]:
    """
    In-place check of Finding objects: downgrades causal phrasing in headlines
    and details, and attaches the observational caveat if claims were modified.
    """
    all_warnings: list[str] = []
    if study_design == "randomized_experiment":
        return all_warnings

    for f in findings:
        headline = getattr(f, "headline", "")
        detail = getattr(f, "detail", "")

        new_h, warn_h = guard_causal_claims(headline, study_design)
        new_d, warn_d = guard_causal_claims(detail, study_design)

        if warn_h or warn_d:
            f.headline = new_h
            f.detail = new_d
            caveats = getattr(f, "caveats", None)
            if caveats is not None and CAVEAT_OBSERVATIONAL not in caveats:
                caveats.append(CAVEAT_OBSERVATIONAL)
            all_warnings.extend(warn_h + warn_d)

    return all_warnings
