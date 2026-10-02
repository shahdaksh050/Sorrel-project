"""
RunView: one read-only, framework-free object between backend results and the UI.

The console used to read loose dicts (`final_result`, `tool_results`, memory
contexts). Several results the backend produces never reached it at all
(`analysis_decision`, `degradations`, `unverified_claims`, `deliverable_audit`,
`llm_usage`, `hypothesis_tree`). This module gathers them into frozen
dataclasses so every screen reads the same typed object.

Rules this module keeps:

* It adds no analysis. Every value is copied or counted from an existing result.
* Absent means absent. A missing or malformed input yields `None` or an empty
  tuple, never a placeholder, a tick, a zero cost or an invented usage record.
* It imports nothing from `memory`, `tools`, `engine`, `controller` or
  Streamlit. The caller snapshots the memory contexts it needs into a plain
  dict (`RUN_VIEW_CONTEXT_KEYS`) while the controller still exists, because the
  controller is gone once the page reruns.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from src.core.audited_entry import audited_checks

#: Memory context keys the caller must snapshot (while the controller exists)
#: and pass to `build_run_view` as a plain dict.
RUN_VIEW_CONTEXT_KEYS: tuple[str, ...] = (
    "analysis_decision",
    "degradations",
    "unverified_claims",
    "deliverable_audit",
    "llm_usage",
    "api_telemetry",
    "hypothesis_tree",
)

#: Caps applied while building. True totals are kept next to each capped tuple.
MAX_HEADLINE_FINDINGS = 5
MAX_FALLBACKS = 8
MAX_UNTRACED_EXAMPLES = 5
MAX_HYPOTHESES = 8
MAX_LIST_ITEMS = 8

#: Finding kinds that describe the method or its limits, not a result.
_NON_HEADLINE_KINDS = ("method_fit", "coverage_gap")
_HEADLINE_LAYERS = ("exec", "analyst")
#: Hypothesis statuses that mean a test actually happened, in display order.
_TESTED_STATUSES = ("refuted", "supported", "inconclusive")


@dataclass(frozen=True)
class Verdict:
    """How many audited headline findings held up. `needs_more = audited - held_up`."""

    held_up: int
    audited: int

    @property
    def needs_more(self) -> int:
        return self.audited - self.held_up


@dataclass(frozen=True)
class Decision:
    """The analysis-mode decision and the options it turned down."""

    mode: str
    rationale: str
    rejected: tuple[str, ...]


@dataclass(frozen=True)
class Deliverables:
    """What the objective asked for: delivered, missing, repaired."""

    delivered: tuple[str, ...]
    missing: tuple[str, ...]
    repaired: tuple[str, ...]


@dataclass(frozen=True)
class Hypothesis:
    """A hypothesis that was actually tested (supported, refuted or inconclusive)."""

    statement: str
    status: str
    confidence: float


@dataclass(frozen=True)
class Usage:
    """LLM usage for the run. Absent (`None` on `HowWeGotHere`) when no AI was used."""

    calls: int
    tokens: int
    cost_usd: float
    is_estimate: bool


@dataclass(frozen=True)
class ProvisionalFinding:
    """A finding seen while a run is still going. Deliberately carries no
    evidence or check marks: checks, the run-level correction and the audits are
    attached after the loop, so anything shown here may still change."""

    finding_id: str
    kind: str
    headline: str


@dataclass(frozen=True)
class HowWeGotHere:
    """The decisions and work-arounds behind the answer (the Details audit trail)."""

    decision: Decision | None = None
    fallbacks: tuple[str, ...] = ()
    fallbacks_total: int = 0
    untraced_numbers: tuple[str, ...] = ()
    untraced_total: int = 0
    deliverables: Deliverables | None = None
    hypotheses: tuple[Hypothesis, ...] = ()
    #: (status, count) for every tested status present, over all hypotheses.
    hypothesis_counts: tuple[tuple[str, int], ...] = ()
    usage: Usage | None = None
    #: True when the run used no LLM (so "No AI was used" is a fact, not a guess).
    no_ai: bool = False


@dataclass(frozen=True)
class RunView:
    """Everything the result screens read. Read-only; derived, never computed."""

    objective: str
    is_sample: bool
    verdict: Verdict | None
    findings: tuple[dict[str, Any], ...]
    headline_findings: tuple[dict[str, Any], ...]
    coverage: dict[str, Any]
    recommendations: tuple[str, ...]
    insights: tuple[str, ...]
    reasoning: str
    how: HowWeGotHere = field(default_factory=HowWeGotHere)
    #: Always empty on a view built from a finished run; live runs show their own.
    provisional_findings: tuple[ProvisionalFinding, ...] = ()


# ── coercion helpers ─────────────────────────────────────────────────────────


def _as_str(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _str_tuple(value: Any, cap: int | None = None) -> tuple[str, ...]:
    """Non-empty strings from a list/tuple. Anything else yields an empty tuple."""
    if not isinstance(value, (list, tuple)):
        return ()
    items = tuple(s for s in (_as_str(v) for v in value) if s)
    return items[:cap] if cap is not None else items


def _as_int(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, (int, float)):
        return int(value)
    return 0


def _as_float(value: Any) -> float:
    if isinstance(value, bool):
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    return 0.0


def _dict_tuple(value: Any) -> tuple[dict[str, Any], ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(v for v in value if isinstance(v, dict))


# ── selection and verdict ────────────────────────────────────────────────────


def is_headline_finding(layer: Any, kind: Any) -> bool:
    """Whether a finding with this layer and kind is a headline candidate."""
    return layer in _HEADLINE_LAYERS and kind not in _NON_HEADLINE_KINDS


def select_headline_findings(
    findings: tuple[dict[str, Any], ...] | list[dict[str, Any]],
) -> tuple[dict[str, Any], ...]:
    """The findings that get a card: result-bearing layers, no method notes, first five.

    The single copy of the selection the Answers tab used to build inline.
    """
    picked = [f for f in findings if is_headline_finding(f.get("layer"), f.get("kind"))]
    return tuple(picked[:MAX_HEADLINE_FINDINGS])


def compute_verdict(headline: tuple[dict[str, Any], ...]) -> Verdict | None:
    """`N of M held up`, counting only findings whose audits actually ran.

    `None` when no finding was audited: a "0 of 0" would claim a check that
    did not happen. A finding held up when none of its checks failed.
    """
    audited = [marks for marks in (audited_checks(f.get("evidence")) for f in headline) if marks]
    if not audited:
        return None
    held_up = sum(1 for marks in audited if not any(m.state == "fail" for m in marks))
    return Verdict(held_up=held_up, audited=len(audited))


# ── how we got here ──────────────────────────────────────────────────────────


def _build_decision(raw: Any) -> Decision | None:
    if not isinstance(raw, Mapping):
        return None
    mode = _as_str(raw.get("mode"))
    if not mode:
        return None
    return Decision(
        mode=mode,
        rationale=_as_str(raw.get("rationale")),
        rejected=_str_tuple(raw.get("alternatives_rejected"), MAX_LIST_ITEMS),
    )


def _build_deliverables(raw: Any) -> Deliverables | None:
    if not isinstance(raw, Mapping):
        return None
    out = Deliverables(
        delivered=_str_tuple(raw.get("delivered"), MAX_LIST_ITEMS),
        missing=_str_tuple(raw.get("missing"), MAX_LIST_ITEMS),
        repaired=_str_tuple(raw.get("repaired"), MAX_LIST_ITEMS),
    )
    return out if (out.delivered or out.missing or out.repaired) else None


def _build_hypotheses(raw: Any) -> tuple[tuple[Hypothesis, ...], tuple[tuple[str, int], ...]]:
    """Tested hypotheses (capped, refuted first) and true per-status counts."""
    if not isinstance(raw, Mapping):
        return (), ()
    nodes = raw.get("nodes")
    if not isinstance(nodes, Mapping):
        return (), ()
    tested: list[Hypothesis] = []
    for node in nodes.values():
        if not isinstance(node, Mapping):
            continue
        status = _as_str(node.get("status"))
        statement = _as_str(node.get("statement"))
        if status in _TESTED_STATUSES and statement:
            tested.append(Hypothesis(statement, status, _as_float(node.get("confidence"))))
    counts = tuple(
        (s, n) for s in _TESTED_STATUSES if (n := sum(1 for h in tested if h.status == s))
    )
    ordered = sorted(tested, key=lambda h: _TESTED_STATUSES.index(h.status))
    return tuple(ordered[:MAX_HYPOTHESES]), counts


def _build_usage(raw: Any) -> Usage | None:
    if not isinstance(raw, Mapping):
        return None
    calls = _as_int(raw.get("call_count"))
    if calls <= 0:
        return None
    return Usage(
        calls=calls,
        tokens=_as_int(raw.get("total_tokens")),
        cost_usd=_as_float(raw.get("estimated_cost_usd")),
        # Cost comes from hand-maintained rates; absent flag is treated as an estimate.
        is_estimate=bool(raw.get("is_estimate", True)),
    )


def _build_how(context: Mapping[str, Any], *, no_ai: bool) -> HowWeGotHere:
    fallbacks_all = _str_tuple(context.get("degradations"))
    untraced_all = _str_tuple(context.get("unverified_claims"))
    hypotheses, counts = _build_hypotheses(context.get("hypothesis_tree"))
    usage = _build_usage(context.get("llm_usage"))
    return HowWeGotHere(
        decision=_build_decision(context.get("analysis_decision")),
        fallbacks=fallbacks_all[:MAX_FALLBACKS],
        fallbacks_total=len(fallbacks_all),
        untraced_numbers=untraced_all[:MAX_UNTRACED_EXAMPLES],
        untraced_total=len(untraced_all),
        deliverables=_build_deliverables(context.get("deliverable_audit")),
        hypotheses=hypotheses,
        hypothesis_counts=counts,
        usage=usage,
        no_ai=no_ai,
    )


# ── builder ──────────────────────────────────────────────────────────────────


def build_run_view(
    final_result: Mapping[str, Any] | None,
    context: Mapping[str, Any] | None,
    *,
    objective: str,
    is_sample: bool,
) -> RunView:
    """Assemble the read-only view from a finished run.

    `final_result` is the controller's result dict; `context` is a plain
    snapshot of `RUN_VIEW_CONTEXT_KEYS`. Either may be `None` or partial.
    """
    result: Mapping[str, Any] = final_result if isinstance(final_result, Mapping) else {}
    ctx: Mapping[str, Any] = context if isinstance(context, Mapping) else {}

    findings = _dict_tuple(result.get("findings"))
    headline = select_headline_findings(findings)
    coverage = result.get("coverage")
    return RunView(
        objective=objective.strip(),
        is_sample=is_sample,
        verdict=compute_verdict(headline),
        findings=findings,
        headline_findings=headline,
        coverage=dict(coverage) if isinstance(coverage, Mapping) else {},
        recommendations=_str_tuple(result.get("recommendations")),
        insights=_str_tuple(result.get("insights")),
        reasoning=_as_str(result.get("reasoning")),
        # `deterministic_mode` is set by the controller only when the LLM was off;
        # missing usage alone proves nothing, so it never implies "no AI".
        how=_build_how(ctx, no_ai=result.get("deterministic_mode") is True),
    )
