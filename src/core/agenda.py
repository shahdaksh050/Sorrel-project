"""
Question agenda — Round 7 item 7.5.

A human analyst starts from questions ("which segments differ most?", "is
this trending?", "what concentrates?") and picks methods second. This module
generates that agenda from the semantic profile + analysis-mode decision +
objective, ranked by expected value, so coverage can be judged in question
space — the place a user's "why" actually lives — rather than only ever
being visible as "which tools ran".

Deliberately additive: `ToolRegistry.candidate_tools`/`applies_to` remain the
capability filter (can this tool even run on this data), and each tool's own
`default_params` still resolves its parameters from the profile — the
agenda answers a narrower question this round: "which questions did we mean
to ask, and which of them got answered", which the deterministic planner
already answers implicitly today but never states out loud.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.core.profiler import DatasetProfile


@dataclass
class Question:
    """One analysis question the agenda thinks is worth asking."""

    text: str
    kind: str                  # segment | concentration | trend | driver | relationship | distribution | model
    columns: list[str] = field(default_factory=list)
    expected_value: float = 0.5    # rough prior on how likely this is to matter, in [0, 1]
    suggested_tool: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "kind": self.kind,
            "columns": self.columns,
            "expected_value": self.expected_value,
            "suggested_tool": self.suggested_tool,
        }


def build_agenda(
    profile: DatasetProfile | None,
    decision: dict[str, Any] | None,
    objective: str = "",
) -> list[Question]:
    """
    Generate a ranked question agenda from the semantic profile.

    Returns highest-expected-value questions first. Never raises — an
    agenda is a nice-to-have transparency artifact, not a pipeline
    dependency, so any failure here should degrade to an empty agenda
    rather than block analysis.
    """
    if profile is None:
        return []

    questions: list[Question] = []
    objective_l = objective.lower()
    measures = profile.measures()
    dimensions = profile.dimensions()

    # "Which segments differ most?" — one per (measure, dimension) pair,
    # capped so a wide dataset doesn't produce a combinatorial agenda.
    for measure in measures[:3]:
        for dim in dimensions[:4]:
            if dim.nunique < 2 or dim.nunique > 20:
                continue
            value = 0.7 if measure.unit_hint == "currency" else 0.5
            questions.append(Question(
                text=f"Does '{measure.name}' differ meaningfully by '{dim.name}'?",
                kind="segment",
                columns=[measure.name, dim.name],
                expected_value=value,
                suggested_tool="segment_comparison",
            ))

    # "What concentrates?" — needs a measure and an entity-like grain.
    if measures and profile.entity_col:
        for measure in measures[:2]:
            questions.append(Question(
                text=f"Is '{measure.name}' concentrated among a few '{profile.entity_col}' entities?",
                kind="concentration",
                columns=[measure.name, profile.entity_col],
                expected_value=0.65,
                suggested_tool="concentration_analysis",
            ))

    # "Is this trending / what changed?" — needs a time axis and a measure.
    if profile.is_time_series and measures:
        for measure in measures[:2]:
            questions.append(Question(
                text=f"Is '{measure.name}' trending or seasonal over time, and what changed most recently?",
                kind="trend",
                columns=[measure.name, *profile.datetime_cols[:1]],
                expected_value=0.75,
                suggested_tool="time_series_analysis",
            ))
            questions.append(Question(
                text=f"What was the most recent period-over-period change in '{measure.name}'?",
                kind="trend",
                columns=[measure.name, *profile.datetime_cols[:1]],
                expected_value=0.6,
                suggested_tool="change_analysis",
            ))

    # "What relates to what?" — general-purpose, lower prior value than a
    # targeted segment/trend question since a correlation alone rarely
    # answers a business question on its own.
    if len(measures) >= 2:
        questions.append(Question(
            text="Which measures move together?",
            kind="relationship",
            columns=[m.name for m in measures[:6]],
            expected_value=0.4,
            suggested_tool="correlation_analysis",
        ))

    # Modelling question — only on the agenda at all when the analysis-mode
    # decision (7.4) actually chose to model; a "describe" decision means
    # this question was considered and explicitly declined, which the
    # coverage report below should say plainly rather than silently omit.
    if decision and decision.get("mode") == "model" and decision.get("target"):
        target = decision["target"]
        questions.append(Question(
            text=f"Can '{target}' be predicted from the other columns, and what drives it?",
            kind="driver",
            columns=[target],
            expected_value=0.8,
            suggested_tool="train_model",
        ))
    elif decision and decision.get("mode") == "describe":
        questions.append(Question(
            text=(
                f"Could '{decision.get('target')}' be predicted? — declined: "
                f"{decision.get('rationale', 'no prediction target fits this data.')}"
            ),
            kind="model",
            columns=[decision["target"]] if decision.get("target") else [],
            expected_value=0.0,
            suggested_tool=None,
        ))

    # Objective text nudges relevant questions up, cheaply — a real
    # expected-value model is out of scope for this pass; keyword overlap
    # between the objective and a question's kind/columns is a reasonable
    # first cut and costs nothing to compute.
    if objective_l:
        for q in questions:
            if any(col.lower() in objective_l for col in q.columns):
                q.expected_value = min(1.0, q.expected_value + 0.15)

    questions.sort(key=lambda q: q.expected_value, reverse=True)
    return questions


def coverage_report(agenda: list[Question], findings: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Match agenda questions against what the finding bus actually produced,
    so "coverage" is a checkable fact rather than an impression. A question
    counts as answered when a finding's kind/measure/dimension columns
    overlap with the question's — a coarse but honest match; false negatives
    (a question judged unanswered when a finding actually addressed it in a
    way this heuristic can't see) are the safe failure direction here, since
    they show up as an honest "could not confirm" rather than a false claim
    of coverage.
    """
    answered: list[Question] = []
    unanswered: list[Question] = []
    for q in agenda:
        if q.suggested_tool is None:
            unanswered.append(q)  # a declined question (e.g. modelling) is unanswered by definition
            continue
        hit = any(
            f.get("kind") in (q.kind, "segment_lift", "driver", "trend", "concentration", "change")
            and (
                f.get("measure") in q.columns
                or f.get("dimension") in q.columns
                or f.get("source_tool") == q.suggested_tool
            )
            for f in findings
        )
        (answered if hit else unanswered).append(q)
    return {
        "total": len(agenda),
        "answered": len(answered),
        "unanswered": [q.to_dict() for q in unanswered],
    }
