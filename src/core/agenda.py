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

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.core.profiler import ColumnProfile, DatasetProfile


@dataclass
class Question:
    """One analysis question the agenda thinks is worth asking."""

    text: str
    kind: str                  # segment | test | concentration | trend | driver | relationship | distribution | model
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

    from src.core.profiler import ARCHETYPE_EXPERIMENT, SEMANTIC_ORDINAL, find_experiment_arm

    questions: list[Question] = []
    objective_l = objective.lower()
    measures = profile.measures()
    dimensions = profile.dimensions()
    # Ordinal scales (satisfaction 1-5) are compared by their mean across
    # segments just like a measure, but never summed or trended as totals.
    segment_measures = [*measures[:3], *profile.columns_of_role(SEMANTIC_ORDINAL)[:2]]
    # An experiment's arm is compared by a significance test (see
    # _archetype_questions), not by a descriptive segment question.
    arm = (
        find_experiment_arm(profile.columns, profile.row_count)
        if profile.archetype == ARCHETYPE_EXPERIMENT else None
    )

    # "Which segments differ most?" — one per (measure, dimension) pair,
    # capped so a wide dataset doesn't produce a combinatorial agenda.
    for measure in segment_measures:
        for dim in dimensions[:4]:
            if dim.nunique < 2 or dim.nunique > 20 or (arm is not None and dim.name == arm.name):
                continue
            value = 0.7 if measure.unit_hint == "currency" else 0.5
            questions.append(Question(
                text=(
                    f"Does '{measure.name}' (total and per-row average) differ meaningfully by '{dim.name}'?"
                    if measure.aggregation == "sum"
                    else f"Does average '{measure.name}' differ meaningfully by '{dim.name}'?"
                ),
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
                text=f"Is {_stat(measure)} '{measure.name}' trending or seasonal over time, and what changed most recently?",
                kind="trend",
                columns=[measure.name, *profile.datetime_cols[:1]],
                expected_value=0.75,
                suggested_tool="time_series_analysis",
            ))
            questions.append(Question(
                text=f"What was the most recent period-over-period change in {_stat(measure)} '{measure.name}'?",
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

    # Archetype-specific questions (experiment, survey, panel/event log,
    # sensor series), skipping any the generic passes above already asked.
    seen = {(q.kind, q.suggested_tool, tuple(sorted(q.columns))) for q in questions}
    for q in _archetype_questions(profile, arm):
        key = (q.kind, q.suggested_tool, tuple(sorted(q.columns)))
        if key not in seen:
            seen.add(key)
            questions.append(q)

    for q in _specialist_questions(profile):
        key = (q.kind, q.suggested_tool, tuple(sorted(q.columns)))
        if key not in seen:
            seen.add(key)
            questions.append(q)

    # Fallback so a dataset with no dimension, entity or time axis (a sensor
    # dump, a lab table) still gets an agenda: ask about each top measure's
    # distribution and outliers (a generic "what correlates" doesn't count).
    if not any(q.kind != "relationship" for q in questions):
        for measure in [*measures[:3], *profile.columns_of_role(SEMANTIC_ORDINAL)[:1]]:
            questions.append(Question(
                text=f"How is '{measure.name}' distributed, and which values are outliers?",
                kind="distribution",
                columns=[measure.name],
                expected_value=0.45,
                suggested_tool="detect_outliers",
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
        # `target` is None whenever mode == "describe" (that's what "not
        # modelling" means, and load_dataset relies on it) — but a
        # candidate column may still have been found and declined (see
        # _decide_analysis_mode's "candidate" key). Prefer that for the
        # question text so a real declined column is named instead of the
        # question reading "Could 'None' be predicted?".
        candidate = decision.get("candidate") or decision.get("target")
        rationale = decision.get("rationale", "no prediction target fits this data.")
        text = (
            f"Could '{candidate}' be predicted? — declined: {rationale}"
            if candidate
            else f"Could a prediction target be found in this data? — declined: {rationale}"
        )
        questions.append(Question(
            text=text,
            kind="model",
            columns=[candidate] if candidate else [],
            expected_value=0.0,
            suggested_tool=None,
        ))

    # Objective text nudges relevant questions up, cheaply — a real
    # expected-value model is out of scope for this pass; keyword overlap
    # between the objective and a question's kind/columns is a reasonable
    # first cut and costs nothing to compute.
    if objective_l:
        for q in questions:
            if any(re.search(rf"(?<!\w){re.escape(col.lower())}(?!\w)", objective_l) for col in q.columns):
                q.expected_value = min(1.0, q.expected_value + 0.15)

    questions.sort(key=lambda q: q.expected_value, reverse=True)
    return questions


def _archetype_questions(profile: DatasetProfile, arm: ColumnProfile | None) -> list[Question]:
    """Questions a given data archetype (profile.archetype) calls for, on
    top of the generic segment/trend/relationship passes."""
    from src.core.profiler import (
        ARCHETYPE_EVENT_LOG,
        ARCHETYPE_PANEL,
        ARCHETYPE_SENSOR,
        ARCHETYPE_SURVEY,
        SEMANTIC_FLAG,
        SEMANTIC_ORDINAL,
        ordinal_scale_columns,
    )

    archetype = profile.archetype
    measures = profile.measures()
    time_cols = profile.datetime_cols[:1]
    questions: list[Question] = []

    if arm is not None:
        outcomes = [
            c for c in (*profile.columns_of_role(SEMANTIC_FLAG), *measures, *profile.columns_of_role(SEMANTIC_ORDINAL))
            if c.name != arm.name
        ][:2]
        for outcome in outcomes:
            questions.append(Question(
                text=(
                    f"Does '{outcome.name}' differ between '{arm.name}' levels, "
                    "and is the difference statistically significant?"
                ),
                kind="test",
                columns=[outcome.name, arm.name],
                expected_value=0.9,
                suggested_tool="select_statistical_test",
            ))
        outcome_names = {c.name for c in outcomes}
        for covariate in [m for m in measures if m.name not in outcome_names][:2]:
            questions.append(Question(
                text=(
                    f"Are the '{arm.name}' groups balanced on '{covariate.name}'? "
                    "An imbalance would confound the treatment effect."
                ),
                kind="segment",
                columns=[covariate.name, arm.name],
                expected_value=0.6,
                suggested_tool="segment_comparison",
            ))

    elif archetype == ARCHETYPE_SURVEY:
        items = ordinal_scale_columns(profile.columns)
        if items:
            questions.append(Question(
                text=(
                    "How are responses to the rating items distributed — which score highest "
                    "and lowest, and are there floor/ceiling effects?"
                ),
                kind="distribution",
                columns=[c.name for c in items[:6]],
                expected_value=0.6,
                suggested_tool="detect_outliers",
            ))
        item_names = {c.name for c in items}
        segments = [
            d for d in profile.dimensions()
            if d.name not in item_names and 2 <= d.nunique <= 20
        ][:2]
        for item in items[:3]:
            for dim in segments:
                questions.append(Question(
                    text=f"Do responses to '{item.name}' differ by '{dim.name}'?",
                    kind="segment",
                    columns=[item.name, dim.name],
                    expected_value=0.6,
                    suggested_tool="segment_comparison",
                ))
        if len(items) >= 2:
            questions.append(Question(
                text="Which rating items move together (a shared underlying attitude)?",
                kind="relationship",
                columns=[c.name for c in items[:8]],
                expected_value=0.55,
                suggested_tool="correlation_analysis",
            ))

    elif archetype in (ARCHETYPE_PANEL, ARCHETYPE_EVENT_LOG):
        entity = profile.entity_col or (profile.panel_group_cols[0] if profile.panel_group_cols else None)
        if entity:
            for measure in measures[:2]:
                questions.append(Question(
                    text=f"Do '{entity}' entities follow different trends in {_stat(measure)} '{measure.name}' over time?",
                    kind="trend",
                    columns=[measure.name, entity, *time_cols],
                    expected_value=0.65,
                    suggested_tool="time_series_analysis",
                ))
                questions.append(Question(
                    text=f"Is '{measure.name}' concentrated among a few '{entity}' entities?",
                    kind="concentration",
                    columns=[measure.name, entity],
                    expected_value=0.65,
                    suggested_tool="concentration_analysis",
                ))
            if archetype == ARCHETYPE_EVENT_LOG:
                questions.append(Question(
                    text=f"Do a few '{entity}' entities generate most of the events?",
                    kind="concentration",
                    columns=[entity],
                    expected_value=0.6,
                    suggested_tool="concentration_analysis",
                ))

    elif archetype == ARCHETYPE_SENSOR:
        for measure in measures[:3]:
            questions.append(Question(
                text=f"Is {_stat(measure)} '{measure.name}' trending or seasonal over time, and what changed most recently?",
                kind="trend",
                columns=[measure.name, *time_cols],
                expected_value=0.75,
                suggested_tool="time_series_analysis",
            ))
            questions.append(Question(
                text=f"Are there anomalous readings or sudden level shifts in '{measure.name}'?",
                kind="distribution",
                columns=[measure.name],
                expected_value=0.6,
                suggested_tool="detect_outliers",
            ))

    return questions


#: kind -> (module, class, value, question template, default_params keys).
_SPECIALIST_TOOLS: dict[str, tuple[str, str, float, str, tuple[str, ...]]] = {
    "survival": (
        "survival", "SurvivalAnalysisTool", 0.75,
        "How long until '{1}' happens, and do some groups leave sooner?", ("duration_column", "event_column"),
    ),
    "curve_fit": (
        "curve_fit", "CurveFitAnalysisTool", 0.7,
        "What curve does '{1}' follow as '{0}' changes?", ("x_column", "y_column"),
    ),
    "mixed_model": (
        "mixed_model", "MixedModelAnalysisTool", 0.7,
        "How much of '{0}' differs between '{1}' groups rather than within them?", ("target_column", "group_column"),
    ),
    "forecast": (
        "forecast", "ForecastAnalysisTool", 0.7,
        "Where is '{1}' heading over the next periods?", ("date_column", "value_column"),
    ),
    "association": (
        "basket", "BasketAnalysisTool", 0.7,
        "Which items are usually bought together?", ("order_column", "item_column"),
    ),
    "elasticity": (
        "elasticity", "PriceElasticityTool", 0.7,
        "How sensitive is '{1}' to '{0}'?", ("price_column", "quantity_column"),
    ),
    "equity": (
        "equity", "EquityAnalysisTool", 0.75,
        "Do '{1}' results differ by '{0}' after allowing for other factors?", ("group_column", "outcome_column"),
    ),
}


def _specialist_questions(profile: DatasetProfile) -> list[Question]:
    """One question per specialist tool (survival, curve fit, mixed model,
    forecast, basket, elasticity, equity) whose `applies_to` fits the data,
    with columns resolved by the tool's own `default_params`."""
    import importlib

    questions: list[Question] = []
    for kind, (module, cls, value, template, keys) in _SPECIALIST_TOOLS.items():
        try:
            tool = getattr(importlib.import_module(f"src.tools.{module}"), cls)()
            if tool.applies_to(profile, None) <= 0:
                continue
            params = tool.default_params(profile, None)
            cols = [str(params[k]) for k in keys]
            questions.append(Question(
                text=template.format(*cols),
                kind=kind,
                columns=cols,
                expected_value=value,
                suggested_tool=tool.name,
            ))
        except Exception:
            continue
    return questions


def _stat(column: Any) -> str:
    """Aggregation-aware wording: "total" for an additive measure, "average"
    for everything else (levels, rates, ordinal scales)."""
    return "total" if getattr(column, "aggregation", None) == "sum" else "average"


_LLM_KINDS = ("custom_analysis", "generated_tool")
_KIND_ALIASES: dict[str, tuple[str, ...]] = {
    "relationship": ("correlation",),
    "segment": ("cohort", "workforce", "financial"),
    "trend": ("financial",),
    "concentration": ("cohort",),
    "driver": ("model_performance",),
}


def _mentioned(finding: dict[str, Any], columns: list[str]) -> int:
    """How many of `columns` a finding is about: named in its measure/dimension,
    its evidence, or (whole-word) its headline and detail."""
    ev = finding.get("evidence") or {}
    text = " ".join(
        str(x) for x in (
            finding.get("headline"), finding.get("detail"), finding.get("measure"), finding.get("dimension"),
            *(ev.keys() if isinstance(ev, dict) else ()),
            *(v for v in (ev.values() if isinstance(ev, dict) else ()) if isinstance(v, str)),
        ) if x
    ).lower()
    return sum(
        1 for c in set(columns)
        if re.search(rf"(?<!\w){re.escape(c.lower())}(?!\w)", text)
    )


def _about(finding: dict[str, Any], columns: list[str]) -> bool:
    """A finding is about a question's columns. With two or more columns
    (e.g. "M by D2") it must name BOTH its measure and dimension among them,
    else a finding on "M by D1" would mark "M by D2" answered."""
    m, d = finding.get("measure"), finding.get("dimension")
    if len(columns) >= 2 and m and d:
        return m in columns and d in columns
    return m in columns or d in columns


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
    from src.core.findings import CAVEAT_FINDING_KINDS

    answered: list[Question] = []
    unanswered: list[Question] = []
    for q in agenda:
        if q.suggested_tool is None:
            unanswered.append(q)  # a declined question (e.g. modelling) is unanswered by definition
            continue
        kind_hit = (q.kind, "segment_lift", "driver", "trend", "concentration", "change", *_KIND_ALIASES.get(q.kind, ()))
        # LLM-authored findings rarely fill measure/dimension, so they are
        # matched on the columns their text names instead.
        llm_hit = any(
            f.get("kind") in _LLM_KINDS and _mentioned(f, q.columns) >= min(2, len(q.columns))
            for f in findings
        ) if q.columns else False
        if q.kind in _SPECIALIST_TOOLS:
            # A specialist tool having run and reported (not just a caveat)
            # is the answer, whatever columns its own resolver picked.
            hit = any(
                f.get("source_tool") == q.suggested_tool and f.get("kind") not in CAVEAT_FINDING_KINDS
                for f in findings
            )
        elif q.kind == "distribution":
            # detect_outliers reports method_fit findings that carry no
            # measure column, so its having reported at all is the answer.
            hit = llm_hit or any(f.get("source_tool") == q.suggested_tool for f in findings)
        elif q.columns:
            # A question that names specific columns is only answered by a
            # finding that's actually ABOUT those columns — matching on
            # source_tool alone let any finding from the right tool count
            # for every question that tool could ever answer (e.g. one
            # segment_comparison finding on an unrelated measure/dimension
            # pair would mark every OTHER segment question "answered" too).
            hit = llm_hit or any(
                f.get("kind") in kind_hit and _about(f, q.columns) for f in findings
            )
        else:
            # No columns to check against (e.g. the general "which measures
            # move together?" question) — the bare source_tool match is the
            # only signal available, same as before.
            hit = any(
                f.get("kind") in kind_hit and f.get("source_tool") == q.suggested_tool
                for f in findings
            )
        (answered if hit else unanswered).append(q)
    return {
        "total": len(agenda),
        "answered": len(answered),
        "unanswered": [q.to_dict() for q in unanswered],
    }
