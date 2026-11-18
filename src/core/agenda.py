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
    objective_fit: float = 0.0
    selection_rationale: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "kind": self.kind,
            "columns": self.columns,
            "expected_value": self.expected_value,
            "suggested_tool": self.suggested_tool,
            "objective_fit": self.objective_fit,
            "selection_rationale": self.selection_rationale,
        }


def objective_fit(objective: str, *, kind: str, columns: list[str]) -> float:
    """Conservative, deterministic relevance score for a proposed question.

    Column mentions are strong evidence of fit.  Intent-word matches provide a
    smaller lift, while an unspecified objective deliberately remains neutral
    rather than inventing a business decision.
    """
    text = objective.lower().strip()
    if not text:
        return 0.5
    column_hit = any(re.search(rf"(?<!\w){re.escape(column.lower())}(?!\w)", text) for column in columns)
    terms: dict[str, tuple[str, ...]] = {
        "trend": ("trend", "forecast", "time", "season", "change"),
        "segment": ("segment", "compare", "group", "region", "cohort"),
        "relationship": ("relationship", "correlat", "associate", "driver"),
        "distribution": ("distribution", "outlier", "anomal", "overview"),
        "driver": ("predict", "driver", "model", "churn", "risk"),
        "model": ("predict", "model", "classif", "regress"),
        "quality": ("quality", "audit", "clean", "missing"),
        "forecast": ("forecast", "future", "next", "project"),
        "survival": ("survival", "retain", "churn", "time until", "duration"),
        "curve_fit": ("curve", "nonlinear", "fit"),
        "mixed_model": ("repeated", "within", "between", "panel"),
        "association": ("basket", "bought together", "association"),
        "elasticity": ("price", "elasticity", "demand"),
        "equity": ("fair", "equity", "bias", "disparity"),
    }
    intent_hit = any(term in text for term in terms.get(kind, ()))
    return 1.0 if column_hit else (0.75 if intent_hit else 0.25)


def build_agenda(
    profile: DatasetProfile | None,
    decision: dict[str, Any] | None,
    objective: str = "",
    excluded_columns: set[str] | None = None,
    specialist_tools: list[Any] | None = None,
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
    excluded = excluded_columns or set()
    measures = [column for column in profile.measures() if column.name not in excluded]
    dimensions = [
        column
        for column in profile.dimensions()
        if column.name not in excluded and 2 <= column.nunique <= 20
    ]
    def _is_id_like_name(col_name: str) -> bool:
        norm = re.sub(r"[^a-z0-9]", "", col_name.lower())
        return (
            norm.endswith(("id", "key", "code", "index", "guid", "uuid"))
            or norm in ("id", "key", "code", "index", "guid", "uuid", "pk", "fk")
        )

    # Column order is an ingestion accident. Prefer complete, compact
    # dimensions, and penalize surrogate numeric IDs so semantic text
    # categories (e.g. brand, category, league_name) sort ahead of IDs.
    dimensions.sort(
        key=lambda column: (
            1 if _is_id_like_name(column.name) else 0,
            column.missing_pct,
            abs(column.nunique - 5),
            column.name.lower(),
        )
    )
    # Ordinal scales (satisfaction 1-5) are compared by their mean across
    # segments just like a measure, but never summed or trended as totals.
    # Prioritize derived performance deltas (actual vs expected) as primary comparative measures.
    delta_measures = [
        column for column in measures
        if any(column.name.lower().endswith(s) for s in ("_delta", "_diff", "_gap", "_ratio", "_outperformance"))
    ]
    std_measures = [column for column in measures if column not in delta_measures]
    segment_measures = [*delta_measures[:2], *std_measures[:3], *profile.columns_of_role(SEMANTIC_ORDINAL)[:2]]
    # An experiment's arm is compared by a significance test (see
    # _archetype_questions), not by a descriptive segment question.
    arm = (
        find_experiment_arm(profile.columns, profile.row_count)
        if profile.archetype == ARCHETYPE_EXPERIMENT else None
    )

    # "Which segments differ most?" — one per (measure, dimension) pair,
    # capped so a wide dataset doesn't produce a combinatorial agenda.
    for measure in segment_measures:
        for dim in dimensions[:6]:
            if arm is not None and dim.name == arm.name:
                continue
            # Higher expected value for prominent categories and key measures:
            # Comparing a core measure across 2-10 clean categories (e.g. brand,
            # platform, category) is the primary story of tabular business data.
            base_val = 0.75 if measure.unit_hint == "currency" else 0.60
            if 2 <= dim.nunique <= 10:
                base_val += 0.10
            dim_lower = dim.name.lower()
            if any(term in dim_lower for term in ("brand", "platform", "category", "type", "segment", "tier", "region", "group")):
                base_val += 0.05
            if measure in delta_measures:
                base_val += 0.15
            value = round(min(0.95, base_val), 2)
            if measure in delta_measures:
                q_text = f"Does performance delta '{measure.name}' (observed vs expected) differ meaningfully by '{dim.name}'?"
            elif measure.aggregation == "sum":
                q_text = f"Does '{measure.name}' (total and per-row average) differ meaningfully by '{dim.name}'?"
            else:
                q_text = f"Does average '{measure.name}' differ meaningfully by '{dim.name}'?"
            questions.append(Question(
                text=q_text,
                kind="segment",
                columns=[measure.name, dim.name],
                expected_value=value,
                suggested_tool="segment_comparison",
            ))

    # Distributions and data quality are first-class questions even when
    # the data also supports group comparisons. Previously they appeared
    # only as a last-resort fallback, so a catalogue with one dubious
    # categorical column could suppress the basic price/rating overview.
    for measure in measures[:3]:
        questions.append(Question(
            text=f"How is '{measure.name}' distributed, where is its centre/spread, and which values are unusual?",
            kind="distribution",
            columns=[measure.name],
            expected_value=0.6,
            suggested_tool="detect_outliers",
        ))

    incomplete = [
        column.name for column in profile.columns
        if column.name not in excluded and column.missing_count > 0
    ]
    questions.append(Question(
        text=(
            "Is the dataset complete and semantically usable, and is missingness systematic?"
            if incomplete
            else "Is the dataset structurally and semantically usable for analysis?"
        ),
        kind="quality",
        columns=incomplete[:6],
        expected_value=0.9,
        suggested_tool="integrity_audit",
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
    for q in _archetype_questions(profile, arm, excluded=excluded):
        key = (q.kind, q.suggested_tool, tuple(sorted(q.columns)))
        if key not in seen:
            seen.add(key)
            questions.append(q)

    for q in _specialist_questions(profile, specialist_tools or [], objective=objective):
        key = (q.kind, q.suggested_tool, tuple(sorted(q.columns)))
        if key not in seen:
            seen.add(key)
            questions.append(q)

    # Fallback so a dataset with no dimension, entity or time axis (a sensor
    # dump, a lab table) still gets an agenda: ask about each top measure's
    # distribution and outliers (a generic "what correlates" doesn't count).
    if not any(q.kind not in {"relationship", "quality"} for q in questions):
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
    for q in questions:
        q.objective_fit = objective_fit(objective, kind=q.kind, columns=q.columns)
        q.expected_value = round(min(1.0, q.expected_value + (0.15 if q.objective_fit >= 0.75 else 0.0)), 2)
        q.selection_rationale = (
            f"objective fit {q.objective_fit:.2f}; expected decision value {q.expected_value:.2f}"
        )

    # Python's sort is stable, so equal-value questions retain the semantic
    # column preference established above (for example, names before IDs).
    questions.sort(key=lambda q: -q.expected_value)
    # A wide table can generate dozens of nearly identical comparisons. Keep
    # the agenda decision-focused and deterministic: one question is a unit of
    # coverage, while each tool gets a small, explicit exploration budget.
    per_tool_budget = 3
    selected: list[Question] = []
    per_tool_count: dict[str | None, int] = {}
    for question in questions:
        count = per_tool_count.get(question.suggested_tool, 0)
        if question.suggested_tool is not None and count >= per_tool_budget:
            continue
        selected.append(question)
        per_tool_count[question.suggested_tool] = count + 1
    return selected


def _archetype_questions(
    profile: DatasetProfile,
    arm: ColumnProfile | None,
    excluded: set[str] | None = None,
) -> list[Question]:
    """Questions a given data archetype (profile.archetype) calls for, on
    top of the generic segment/trend/relationship passes."""
    from src.core.profiler import (
        ARCHETYPE_BASKET,
        ARCHETYPE_CONTINUOUS_SERIES,
        ARCHETYPE_CROSS_SECTIONAL,
        ARCHETYPE_EVENT_LOG,
        ARCHETYPE_GEOSPATIAL,
        ARCHETYPE_HYBRID_TEXT,
        ARCHETYPE_NETWORK,
        ARCHETYPE_PANEL,
        ARCHETYPE_SENSOR,
        ARCHETYPE_SURVEY,
        ARCHETYPE_SURVIVAL,
        ARCHETYPE_TRANSACTIONAL,
        SEMANTIC_FLAG,
        SEMANTIC_ORDINAL,
        ordinal_scale_columns,
    )

    excluded_set = excluded or set()
    archetype = profile.archetype
    measures = [m for m in profile.measures() if m.name not in excluded_set]
    time_cols = [c for c in profile.datetime_cols[:1] if c not in excluded_set]
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

    elif archetype in (ARCHETYPE_PANEL,):
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
                    text=f"How much of '{measure.name}' variance is between '{entity}' groups versus within them over time?",
                    kind="mixed_model",
                    columns=[measure.name, entity],
                    expected_value=0.70,
                    suggested_tool="mixed_model_analysis",
                ))

    elif archetype in (ARCHETYPE_TRANSACTIONAL, ARCHETYPE_EVENT_LOG):
        entity = profile.entity_col or (profile.panel_group_cols[0] if profile.panel_group_cols else None)
        if entity:
            for measure in measures[:2]:
                questions.append(Question(
                    text=f"Is '{measure.name}' concentrated among a few '{entity}' entities (Pareto 80/20 distribution)?",
                    kind="concentration",
                    columns=[measure.name, entity],
                    expected_value=0.70,
                    suggested_tool="concentration_analysis",
                ))
            questions.append(Question(
                text=f"Do a few '{entity}' entities generate most of the transactions?",
                kind="concentration",
                columns=[entity],
                expected_value=0.65,
                suggested_tool="concentration_analysis",
            ))

    elif archetype in (ARCHETYPE_CONTINUOUS_SERIES, ARCHETYPE_SENSOR):
        for measure in measures[:3]:
            questions.append(Question(
                text=f"Is {_stat(measure)} '{measure.name}' trending or exhibiting multi-period seasonality, and are returns stationary?",
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

    elif archetype == ARCHETYPE_SURVIVAL:
        dur_col = next((c.name for c in profile.columns if any(k in c.name.lower() for k in ("tenure", "duration", "time", "months")) and c.kind == "numeric"), None)
        event_col = next((c.name for c in profile.columns if any(k in c.name.lower() for k in ("churn", "died", "event", "status", "cancel", "failed"))), None)
        dims = [d.name for d in profile.dimensions() if 2 <= d.nunique <= 10]
        if dur_col and event_col:
            questions.append(Question(
                text=f"What is the median survival duration before '{event_col}', and what factors significantly alter the hazard rate?",
                kind="survival",
                columns=[dur_col, event_col],
                expected_value=0.85,
                suggested_tool="survival_analysis",
            ))
            if dims:
                questions.append(Question(
                    text=f"Does survival probability significantly differ across '{dims[0]}' cohorts?",
                    kind="survival",
                    columns=[dur_col, event_col, dims[0]],
                    expected_value=0.80,
                    suggested_tool="survival_analysis",
                ))

    elif archetype == ARCHETYPE_BASKET:
        order_col = next((c.name for c in profile.columns if any(k in c.name.lower() for k in ("order", "basket", "invoice", "receipt", "transaction"))), None)
        item_col = next((c.name for c in profile.columns if any(k in c.name.lower() for k in ("item", "product", "sku", "description"))), None)
        if order_col and item_col:
            questions.append(Question(
                text=f"Which '{item_col}' products are most frequently co-purchased in '{order_col}' baskets with association lift > 1.5?",
                kind="association",
                columns=[order_col, item_col],
                expected_value=0.85,
                suggested_tool="basket_analysis",
            ))

    elif archetype == ARCHETYPE_NETWORK:
        src_col = next((c.name for c in profile.columns if any(k in c.name.lower() for k in ("source", "from", "src", "origin", "sender"))), None)
        dst_col = next((c.name for c in profile.columns if any(k in c.name.lower() for k in ("target", "to", "dst", "destination", "receiver"))), None)
        if src_col and dst_col:
            questions.append(Question(
                text=f"Which nodes in '{src_col}' -> '{dst_col}' exhibit high centrality and form structural bottlenecks across the network?",
                kind="relationship",
                columns=[src_col, dst_col],
                expected_value=0.80,
                suggested_tool="graph_analysis",
            ))

    elif archetype == ARCHETYPE_GEOSPATIAL:
        lat_col = next((c.name for c in profile.columns if re.search(r"lat", c.name, re.I)), None)
        lon_col = next((c.name for c in profile.columns if re.search(r"lon|lng", c.name, re.I)), None)
        if lat_col and lon_col:
            questions.append(Question(
                text=f"Are observations clustered geographically across '{lat_col}' and '{lon_col}', and do metric outcomes correlate with spatial coordinates?",
                kind="distribution",
                columns=[lat_col, lon_col, *[m.name for m in measures[:2]]],
                expected_value=0.75,
                suggested_tool="cluster_data",
            ))

    elif archetype == ARCHETYPE_HYBRID_TEXT:
        text_col = next((c.name for c in profile.columns if c.kind == "text" or c.stats.get("avg_word_count", 0) >= 6), None)
        if text_col:
            questions.append(Question(
                text=f"What dominant topics and sentiment polarities in '{text_col}' correlate most strongly with key metric outcomes?",
                kind="relationship",
                columns=[text_col, *[m.name for m in measures[:2]]],
                expected_value=0.75,
                suggested_tool="correlation_analysis",
            ))

    elif archetype == ARCHETYPE_CROSS_SECTIONAL:
        if measures:
            dims = [d.name for d in profile.dimensions() if 2 <= d.nunique <= 10]
            if dims:
                questions.append(Question(
                    text=f"Do primary outcomes in '{measures[0].name}' differ significantly across '{dims[0]}' segments?",
                    kind="segment",
                    columns=[measures[0].name, dims[0]],
                    expected_value=0.70,
                    suggested_tool="segment_comparison",
                ))

    return [q for q in questions if not any(c in excluded_set for c in q.columns)]


#: tool name -> (kind, value, question template, default_params keys).
_SPECIALIST_TOOLS: dict[str, tuple[str, float, str, tuple[str, ...]]] = {
    "survival_analysis": (
        "survival", 0.75,
        "How long until '{1}' happens, and do some groups leave sooner?", ("duration_column", "event_column"),
    ),
    "curve_fit_analysis": (
        "curve_fit", 0.7,
        "What curve does '{1}' follow as '{0}' changes?", ("x_column", "y_column"),
    ),
    "mixed_model_analysis": (
        "mixed_model", 0.7,
        "How much of '{0}' differs between '{1}' groups rather than within them?", ("target_column", "group_column"),
    ),
    "forecast_analysis": (
        "forecast", 0.7,
        "Where is '{1}' heading over the next periods?", ("date_column", "value_column"),
    ),
    "basket_analysis": (
        "basket", 0.7,
        "Which items are usually bought together?", ("order_column", "item_column"),
    ),
    "price_elasticity_analysis": (
        "elasticity", 0.7,
        "How sensitive is '{1}' to '{0}'?", ("price_column", "quantity_column"),
    ),
    "equity_analysis": (
        "equity", 0.75,
        "Do '{1}' results differ by '{0}' after allowing for other factors?", ("group_column", "outcome_column"),
    ),
}


def _specialist_questions(
    profile: DatasetProfile,
    tools: list[Any],
    objective: str = "",
) -> list[Question]:
    """One question per specialist tool (survival, curve fit, mixed model,
    forecast, basket, elasticity, equity) whose `applies_to` fits the data,
    with columns resolved by the tool's own `default_params`."""
    questions: list[Question] = []
    for tool in tools:
        spec = _SPECIALIST_TOOLS.get(str(getattr(tool, "name", "")))
        if spec is None:
            continue
        kind, value, template, keys = spec
        try:
            applicability = tool.applies_to(profile, None)
            if applicability <= 0:
                continue
            params = tool.default_params(profile, None)
            cols = [str(params[k]) for k in keys]
            # Specialist selection has two hard prerequisites: an executable
            # parameterization and a meaningful question.  With no supplied
            # objective we retain a neutral exploratory fit; an explicitly
            # unrelated objective does not spend a specialist-tool budget.
            fit = objective_fit(objective, kind=kind, columns=cols)
            if objective and fit < 0.5:
                continue
            if kind == "survival" and params.get("group_column"):
                cols.append(str(params["group_column"]))  # optional: the group comparison
            questions.append(Question(
                text=template.format(*cols),
                kind=kind,
                columns=cols,
                expected_value=value,
                suggested_tool=tool.name,
                objective_fit=fit,
                selection_rationale=(
                    f"specialist applicability {applicability:.2f}; objective fit {fit:.2f}"
                ),
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
    """A finding is about a question's columns. Matches if question columns are covered
    by measure/dimension, target/segment, or evidence columns."""
    if not columns:
        return False
    m = str(finding.get("measure") or "").lower().strip()
    d = str(finding.get("dimension") or "").lower().strip()
    ev = finding.get("evidence") or {}
    found_cols: set[str] = {c for c in (m, d) if c}
    if isinstance(ev, dict):
        for k in ("col_a", "col_b", "target_column", "segment_column", "feature_column", "group_column", "value_column", "time_column", "column"):
            val = str(ev.get(k) or "").lower().strip()
            if val:
                found_cols.add(val)
        for list_k in ("columns", "features", "features_analyzed"):
            arr = ev.get(list_k)
            if isinstance(arr, (list, tuple)):
                found_cols.update(str(x).lower().strip() for x in arr if x)

    q_cols = {str(c).lower().strip() for c in columns if c}
    if not q_cols:
        return False
    if len(q_cols) >= 2:
        return q_cols.issubset(found_cols)
    return any(c in found_cols for c in q_cols)


def coverage_report(
    agenda: list[Question],
    findings: list[dict[str, Any]],
    tool_results: list[Any] | None = None,
) -> dict[str, Any]:
    """
    Match agenda questions against what the finding bus actually produced and
    what analytical steps were successfully executed. A question counts as answered
    when:
      1. A finding's kind/measure/dimension/evidence columns address the question, OR
      2. An executed tool step successfully ran and evaluated the question's target
         dimensions/measures.
    """
    from src.core.findings import CAVEAT_FINDING_KINDS

    # Build structured index of successful tool executions
    executed_tools: list[dict[str, Any]] = []
    if tool_results:
        for r in tool_results:
            if hasattr(r, "tool_name"):
                name = getattr(r, "tool_name", "")
                status = getattr(r, "status", "")
                params = getattr(r, "params", {}) or {}
                output = getattr(r, "output", {}) or {}
            elif isinstance(r, dict):
                name = str(r.get("tool_name", ""))
                status = str(r.get("status", ""))
                params = r.get("params", {}) or {}
                output = r.get("output", {}) or {}
            else:
                continue
            if status == "success":
                executed_tools.append({
                    "tool": name,
                    "params": params if isinstance(params, dict) else {},
                    "output": output if isinstance(output, dict) else {},
                })

    answered: list[Question] = []
    unanswered: list[Question] = []

    for q in agenda:
        if q.suggested_tool is None:
            unanswered.append(q)  # a declined question (e.g. modelling) is unanswered by definition
            continue

        kind_hit = (q.kind, "segment_lift", "driver", "trend", "concentration", "change", *_KIND_ALIASES.get(q.kind, ()))

        llm_hit = any(
            f.get("kind") in _LLM_KINDS and _mentioned(f, q.columns) >= min(2, len(q.columns))
            for f in findings
        ) if q.columns else False

        hit = False
        if q.kind in {spec[0] for spec in _SPECIALIST_TOOLS.values()}:
            hit = any(
                f.get("source_tool") == q.suggested_tool and f.get("kind") not in CAVEAT_FINDING_KINDS
                for f in findings
            ) or any(ex["tool"] == q.suggested_tool for ex in executed_tools)
        elif q.kind == "distribution":
            hit = (
                llm_hit
                or any(f.get("source_tool") == q.suggested_tool for f in findings)
                or any(ex["tool"] in (q.suggested_tool, "detect_outliers") for ex in executed_tools)
            )
        elif q.kind == "quality":
            hit = (
                any(f.get("source_tool") in (q.suggested_tool, "clean_data", "ingest_dataset", "integrity_audit") for f in findings)
                or any(ex["tool"] in (q.suggested_tool, "clean_data", "ingest_dataset", "integrity_audit") for ex in executed_tools)
            )
        elif q.columns:
            hit = llm_hit or any(
                f.get("kind") in kind_hit and _about(f, q.columns) for f in findings
            )
            if not hit and executed_tools:
                for ex in executed_tools:
                    if ex["tool"] != q.suggested_tool:
                        if not (q.suggested_tool in ("time_series_analysis", "change_analysis") and ex["tool"] in ("time_series_analysis", "change_analysis")):
                            continue
                    tool_cols: set[str] = set()
                    for v in ex["params"].values():
                        if isinstance(v, str):
                            tool_cols.add(v.lower())
                        elif isinstance(v, (list, tuple)):
                            tool_cols.update(str(item).lower() for item in v)
                    for k in ("target_column", "segment_column", "value_column", "feature_column", "group_column", "time_column", "price_column", "quantity_column"):
                        v = ex["output"].get(k)
                        if isinstance(v, str):
                            tool_cols.add(v.lower())
                    analyzed = ex["output"].get("features_analyzed") or ex["output"].get("columns_analyzed") or []
                    if isinstance(analyzed, (list, tuple)):
                        tool_cols.update(str(c).lower() for c in analyzed)
                    outliers_cols = ex["output"].get("outliers_by_column")
                    if isinstance(outliers_cols, dict):
                        tool_cols.update(str(c).lower() for c in outliers_cols.keys())

                    q_lower = {c.lower().strip() for c in q.columns}
                    if q.suggested_tool in ("time_series_analysis", "change_analysis"):
                        measure_col = q.columns[0].lower().strip()
                        if measure_col in tool_cols:
                            hit = True
                            break
                    if len(q_lower) >= 2:
                        if q_lower.issubset(tool_cols):
                            hit = True
                            break
                    elif len(q_lower) == 1:
                        if next(iter(q_lower)) in tool_cols:
                            hit = True
                            break
        else:
            hit = any(
                f.get("kind") in kind_hit and f.get("source_tool") == q.suggested_tool
                for f in findings
            ) or any(ex["tool"] == q.suggested_tool for ex in executed_tools)

        (answered if hit else unanswered).append(q)

    return {
        "total": len(agenda),
        "answered": len(answered),
        "unanswered": [q.to_dict() for q in unanswered],
    }
