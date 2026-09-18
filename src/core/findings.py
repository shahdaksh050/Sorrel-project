"""
Finding bus — the single structured representation of "something the
analysis discovered", shared by every output surface.

Round 7 item 7.1. Tools keep returning their existing `output` dicts (no
tool rewrite required); a tool that wants to contribute to the narrative,
the reports, and the dashboard implements `BaseTool.findings()` to derive
a list of `Finding` objects from that same output. `MemorySystem`
accumulates them; `ranked_findings()` is the one ordering every projection
(the Markdown report, the HTML report, the CLI synthesis, the dashboard
panel selector) reads from — so a tool orphaned from one surface is
orphaned from none of them, and none of them independently.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

#: A finding whose |effect| falls below this is suppressed outright unless
#: it is the only finding available — the T5 "triviality suppression" rule.
TRIVIAL_EFFECT_FLOOR = 0.02

#: Ranking weights for `Finding.importance` — effect and confidence matter
#: most; surprise and objective-fit break ties among similar-effect findings.
_W_EFFECT = 0.4
_W_CONFIDENCE = 0.25
_W_SURPRISE = 0.15
_W_OBJECTIVE_FIT = 0.20


@dataclass
class Finding:
    """One structured, audience-facing analytical claim."""

    finding_id: str
    kind: str                          # segment_lift | driver | trend | concentration | correlation | outlier_scan | cluster | test | change | cohort | financial | workforce | geospatial | text | dimensionality
    headline: str                      # audience-facing, numbers embedded
    detail: str = ""                   # analyst-facing elaboration
    evidence: dict[str, Any] = field(default_factory=dict)   # exact numbers, traceable to a ToolResult
    source_tool: str = ""

    measure: str | None = None
    dimension: str | None = None
    level: str | None = None

    effect: float | None = None        # magnitude, in whatever unit effect_kind implies
    effect_kind: str | None = None     # lift | cohens_d | r | eta_sq | pct | share

    p_value: float | None = None
    p_adjusted: float | None = None

    confidence: float = 0.5            # evidence strength x sufficiency, in [0, 1]
    surprise: float = 0.3              # distance from base rate / prior expectation, in [0, 1]
    objective_fit: float = 0.0         # how well this answers the stated objective, in [0, 1]
    importance: float = 0.0            # ranking key, computed by compute_importance()

    caveats: list[str] = field(default_factory=list)
    chart_hint: dict[str, Any] | None = None   # {"kind": "bar"/"line"/..., "data": {...}} — what would show this
    layer: str = "analyst"             # exec | analyst | appendix — which report tier this belongs on by default

    def compute_importance(self) -> float:
        """Deterministic ranking key: effect x confidence, plus surprise and
        objective-fit as tie-breakers. Effect is normalised into [0, 1] via a
        soft cap so a huge outlier effect doesn't blow the other terms out."""
        norm_effect = min(1.0, abs(self.effect) / 2.0) if self.effect is not None else 0.3
        score = (
            _W_EFFECT * norm_effect
            + _W_CONFIDENCE * self.confidence
            + _W_SURPRISE * self.surprise
            + _W_OBJECTIVE_FIT * self.objective_fit
        )
        self.importance = round(score, 6)
        return self.importance

    def to_dict(self) -> dict[str, Any]:
        return {
            "finding_id": self.finding_id,
            "kind": self.kind,
            "headline": self.headline,
            "detail": self.detail,
            "evidence": self.evidence,
            "source_tool": self.source_tool,
            "measure": self.measure,
            "dimension": self.dimension,
            "level": self.level,
            "effect": self.effect,
            "effect_kind": self.effect_kind,
            "p_value": self.p_value,
            "p_adjusted": self.p_adjusted,
            "confidence": self.confidence,
            "surprise": self.surprise,
            "objective_fit": self.objective_fit,
            "importance": self.importance,
            "caveats": self.caveats,
            "chart_hint": self.chart_hint,
            "layer": self.layer,
        }


def is_trivial(finding: Finding) -> bool:
    """Shared triviality predicate (T5): a finding whose effect is
    negligible, or that is explicitly tagged as a method-fit warning rather
    than a discovery, should not compete for headline space."""
    if finding.kind in ("method_fit", "coverage_gap"):
        return False  # these are meant to be shown, just not ranked as insight
    if finding.effect is None:
        return False
    return abs(finding.effect) < TRIVIAL_EFFECT_FLOOR


def rank_findings(findings: list[Finding], suppress_trivial: bool = True) -> list[Finding]:
    """Compute importance for every finding and return them ranked
    descending. Trivial findings are dropped (not just sorted low) unless
    doing so would empty the list, so a genuinely flat dataset still reports
    something rather than nothing."""
    for f in findings:
        f.compute_importance()
    ranked = sorted(findings, key=lambda f: f.importance, reverse=True)
    if not suppress_trivial:
        return ranked
    kept = [f for f in ranked if not is_trivial(f)]
    return kept if kept else ranked
