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

import re
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

#: Finding kinds that are caveats about method/coverage, not discoveries —
#: shown, but never ranked against real insights.
CAVEAT_FINDING_KINDS = ("method_fit", "coverage_gap")

#: effect_kind -> the effect magnitude that counts as "large" by convention
#: (Cohen's d 0.8, r 0.5, eta-squared 0.14, Cramer's V 0.3 ...), so effects
#: on different scales normalise onto one [0, 1] axis. `lift` is stored as
#: ratio - 1 (segment_comparison, cohort, time_series), so |lift| 0.5 = 1.5x.
#: Spellings cover both the finding bus names and statistical_analysis's
#: effect_size_metric strings, which it passes through as effect_kind.
_EFFECT_BENCHMARKS: dict[str, float] = {
    "cohens_d": 0.8, "hedges_g": 0.8,
    "r": 0.5, "rank_biserial": 0.5,
    "eta_sq": 0.14, "eta_squared": 0.14, "epsilon_squared": 0.14,
    "cramers_v": 0.3,
    "lift": 0.5, "pct": 0.5, "share": 0.5,
    "r2": 0.5, "silhouette": 0.5,
}

#: Evidence keys that carry a sample size; below _SMALL_N the confidence is
#: discounted however small the p-value.
_SAMPLE_SIZE_KEYS = ("n", "n_level", "sample_size", "n_segment", "n_obs")
_SMALL_N = 30

_STOPWORDS = frozenset(
    "a an the of in on for to and or by with what which who how why is are was were "
    "do does did be been it its this that these those from at as vs versus than per "
    "across between over under into about my our me i we you show find tell analyse "
    "analyze analysis data dataset any most more less much many should can could would "
    "there their them".split()
)
_TOKEN_RE = re.compile(r"[a-z0-9]+")
_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


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

    def normalized_effect(self) -> float:
        """|effect| on a common [0, 1] scale, 1.0 = a conventionally large
        effect for its effect_kind (see _EFFECT_BENCHMARKS). A `share` whose
        evidence carries an expected/uniform share is scored on its excess
        over that baseline (concentration's top-10% share vs an even 10%). No effect -> 0.15, an unknown kind -> the old /2 cap."""
        if self.effect is None:
            return 0.15
        effect = abs(self.effect)
        benchmark = _EFFECT_BENCHMARKS.get(self.effect_kind or "")
        if benchmark is None:
            return min(1.0, effect / 2.0)
        if self.effect_kind == "share":
            evidence = self.evidence if isinstance(self.evidence, dict) else {}
            expected = evidence.get("expected_share", evidence.get("uniform_share"))
            if isinstance(expected, (int, float)):
                effect = abs(self.effect - expected)
        return min(1.0, effect / benchmark)

    def effective_confidence(self) -> float:
        """Confidence for ranking. A tool that left the 0.5 default but
        reported a p-value gets one derived from it; a small sample in the
        evidence discounts either way."""
        confidence = self.confidence
        p = self.p_adjusted if self.p_adjusted is not None else self.p_value
        if confidence == 0.5 and p is not None:
            confidence = 0.9 if p < 0.001 else 0.8 if p < 0.01 else 0.65 if p < 0.05 else 0.3
        if isinstance(self.evidence, dict):
            sizes = [
                v for k in _SAMPLE_SIZE_KEYS
                if isinstance(v := self.evidence.get(k), (int, float)) and not isinstance(v, bool)
            ]
            if sizes and min(sizes) < _SMALL_N:
                confidence *= max(0.3, min(sizes) / _SMALL_N)
        return confidence

    def compute_importance(self) -> float:
        """Deterministic ranking key: effect x confidence, plus surprise and
        objective-fit as tie-breakers. Effect is normalised per effect_kind
        (normalized_effect) so a Cohen's d and a lift compete on one scale."""
        score = (
            _W_EFFECT * self.normalized_effect()
            + _W_CONFIDENCE * self.effective_confidence()
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
    if finding.kind in CAVEAT_FINDING_KINDS:
        return False  # these are meant to be shown, just not ranked as insight
    if finding.effect is None:
        return False
    return abs(finding.effect) < TRIVIAL_EFFECT_FLOOR


def _diversify_by_source(findings: list[Finding], max_per_source: int) -> list[Finding]:
    """Greedy source-diversity pass (importance order is assumed already
    applied to `findings`): the first `max_per_source` findings seen for
    each (source_tool, kind) bucket stay in place, everything past that cap
    is pushed to the end (still in importance order among itself). Nothing
    is dropped — a single tool/kind combination just can't monopolise the
    front of the list the way 7 month-of-year findings from one time-series
    call did in practice."""
    counts: dict[tuple[str, str], int] = {}
    head: list[Finding] = []
    overflow: list[Finding] = []
    for f in findings:
        key = (f.source_tool, f.kind)
        if counts.get(key, 0) < max_per_source:
            head.append(f)
            counts[key] = counts.get(key, 0) + 1
        else:
            overflow.append(f)
    return head + overflow


def _tokens(text: str) -> set[str]:
    """Lowercase content tokens with stopwords dropped and a crude plural
    stem ("regions" -> "region"), splitting snake_case and camelCase."""
    words = _TOKEN_RE.findall(_CAMEL.sub(" ", str(text)).lower())
    return {
        w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith("ss") else w
        for w in words if w not in _STOPWORDS
    }


def score_objective_fit(finding: Finding, objective: str) -> float:
    """How directly `finding` answers `objective`, in [0, 1]. Token overlap
    between the objective and the finding's measure/dimension/level/headline,
    with a strong boost when the finding's measure or dimension column
    (split on _) is named in the objective — "why does churn differ by
    region" should lift the churn-by-region finding above an unrelated one."""
    objective_tokens = _tokens(objective)
    if not objective_tokens:
        return 0.0
    score = 0.0
    for column in (finding.measure, finding.dimension):
        if column:
            col_tokens = _tokens(column)
            if col_tokens and col_tokens <= objective_tokens:
                score += 0.4
            elif col_tokens & objective_tokens:
                score += 0.2
    text = " ".join(str(x) for x in (finding.level, finding.headline) if x)
    overlap = len(objective_tokens & _tokens(text)) / len(objective_tokens)
    return round(min(1.0, score + 0.4 * overlap), 4)


def _dedupe(findings: list[Finding]) -> tuple[list[Finding], list[Finding]]:
    """Split importance-ordered findings into (kept, duplicates). Two tools
    reporting the same (measure, dimension, level), or the same kind on the
    same measure with the same leading headline numbers, are one discovery;
    the higher-importance copy stays, the other goes to overflow."""
    seen: set[tuple[Any, ...]] = set()
    kept: list[Finding] = []
    dupes: list[Finding] = []
    for f in findings:
        keys: list[tuple[Any, ...]] = []
        if f.measure and (f.dimension or f.level):
            keys.append(("cell", f.measure, f.dimension, f.level))
        numbers = tuple(round(float(n), 1) for n in _NUMBER_RE.findall(f.headline.replace(",", ""))[:3])
        if f.measure and numbers:
            keys.append(("numbers", f.kind, f.measure, numbers))
        if any(k in seen for k in keys):
            dupes.append(f)
            continue
        seen.update(keys)
        kept.append(f)
    return kept, dupes


def rank_findings(
    findings: list[Finding],
    suppress_trivial: bool = True,
    max_per_source: int | None = 2,
) -> list[Finding]:
    """Compute importance for every finding and return them ranked
    descending. Trivial findings are dropped (not just sorted low) unless
    doing so would empty the list, so a genuinely flat dataset still reports
    something rather than nothing.

    `max_per_source` (default 2) then applies a source-diversity pass: at
    most that many findings from the same (source_tool, kind) combination
    are allowed into the front of the list before findings from other
    tools/kinds get a turn. This is a reordering, not a filter — every
    finding that survived trivial-suppression is still returned, just with
    the rest of a dominant source's findings pushed after the more varied
    front. Pass `max_per_source=None` to skip this and keep a plain
    importance sort (e.g. for callers that want the raw ranking).

    Near-duplicates across tools (see _dedupe) move behind the distinct
    findings, and caveat kinds (CAVEAT_FINDING_KINDS) always come after
    every discovery — they are shown, not ranked as insight.
    """
    for f in findings:
        f.compute_importance()
    ranked = sorted(findings, key=lambda f: f.importance, reverse=True)
    if suppress_trivial:
        kept = [f for f in ranked if not is_trivial(f)]
        ranked = kept if kept else ranked
    result, dupes = _dedupe(ranked)
    if max_per_source is not None:
        result = _diversify_by_source(result, max_per_source)
    result += dupes
    return (
        [f for f in result if f.kind not in CAVEAT_FINDING_KINDS]
        + [f for f in result if f.kind in CAVEAT_FINDING_KINDS]
    )
