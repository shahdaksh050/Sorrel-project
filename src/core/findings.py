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
from typing import Any, Literal, cast

from src.core.privacy import redact_level_in_text
from src.core.stats_utils import get_base_column_name, is_derived_pair

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
LLM_FINDING_KINDS = ("custom_analysis", "generated_tool")
HYGIENE_FINDING_KINDS = frozenset({
    "data_quality", "integrity", "flatline", "constraint", "metadata", "dimension_check",
})
HYGIENE_SOURCE_TOOLS = frozenset({"integrity_audit", "clean_data", "ingest_dataset"})
HYGIENE_HEADLINE_PREFIXES = (
    "sensor flatline", "constraint violation", "dataset contains",
    "systematic missingness", "missing values in", "duplicate rows",
)

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
_SAMPLE_SIZE_KEYS = ("n", "n_level", "sample_size", "n_segment", "n_obs", "total", "n_total", "population_size")
_SMALL_N = 30

#: Mirrors `src.tools.ml_pipeline.OVERFIT_THRESHOLD`. Not imported directly —
#: `tools/*` may depend on `core` (AGENTS.md), never the reverse. Only the
#: `..._overfit_gap` synthetic Finding ever carries `train_test_gap` in its
#: evidence, and only once that gap already exceeds this threshold, so this
#: check can only ever fail; there is no Finding for the passing case to
#: attach a tick to.
_OVERFIT_GAP_THRESHOLD = 0.10

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

EvidenceTier = Literal["observation", "hypothesis", "pilot", "causal_action"]


@dataclass
class Finding:
    """One structured, audience-facing analytical claim."""

    finding_id: str
    kind: str                          # segment_lift | driver | trend | concentration | correlation | outlier_scan | cluster | test | change | cohort | financial | workforce | geospatial | text | dimensionality | invariance | uniformity
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
    raw_importance_rank: int | None = None
    display_rank: int | None = None
    ranking_policy: str = "importance_then_source_diversity_v1"

    caveats: list[str] = field(default_factory=list)
    chart_hint: dict[str, Any] | None = None   # {"kind": "bar"/"line"/..., "data": {...}} — what would show this
    layer: str = "analyst"             # exec | analyst | appendix — which report tier this belongs on by default
    evidence_tier: EvidenceTier = "observation"
    primary_eligible: bool = True
    eligibility_reasons: list[str] = field(default_factory=list)

    disposition: Literal["monitor", "investigate", "pilot", "act"] | None = None
    outcome: str | None = None
    unit: str | None = None
    desired_direction: str | None = None
    evidence_design: str | None = None
    controllability: str | None = None
    source: str | None = None
    statistical_state: Literal["unperformed", "inconclusive", "detected_difference", "demonstrated_equivalence"] | None = None
    practical_margin: float | None = None

    def __post_init__(self) -> None:
        """Small-cell guard: a finding naming a level whose evidence `n` is
        below the minimum cell size must not carry that name in its text."""
        if not self.level or not isinstance(self.evidence, dict):
            return
        sizes = [
            v for k in ("n", "n_level")
            if isinstance(v := self.evidence.get(k), (int, float)) and not isinstance(v, bool)
        ]
        if not sizes:
            return
        n = min(sizes)
        self.headline = redact_level_in_text(self.headline, self.level, n)
        self.detail = redact_level_in_text(self.detail, self.level, n)

    def normalized_effect(self) -> float:
        """|effect| on a common [0, 1] scale, 1.0 = a conventionally large
        effect for its effect_kind (see _EFFECT_BENCHMARKS). A `share` whose
        evidence carries an expected/uniform share is scored on its excess
        over that baseline (concentration's top-10% share vs an even 10%). No effect -> 0.15, an unknown kind -> the old /2 cap."""
        if self.kind in ("uniformity", "invariance"):
            return 0.0  # Historical null-test claims have no equivalence evidence.
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
        if self.kind == "cluster":
            evidence = self.evidence if isinstance(self.evidence, dict) else {}
            sil = evidence.get("silhouette_score")
            quality = evidence.get("separation_quality")
            if (isinstance(sil, (int, float)) and sil < 0.35) or quality in ("weak", "none"):
                sil_val = max(0.05, float(sil)) if isinstance(sil, (int, float)) else 0.25
                effect = effect * min(1.0, sil_val / 0.5)
        return min(1.0, effect / benchmark)

    def effective_confidence(self) -> float:
        """Confidence for ranking. A tool that left the 0.5 default but
        reported a p-value gets one derived from it; a small sample in the
        evidence discounts either way."""
        confidence = self.confidence
        p = self.p_adjusted if self.p_adjusted is not None else self.p_value
        if p is not None:
            p_conf = 0.9 if p < 0.001 else 0.8 if p < 0.01 else 0.65 if p < 0.05 else 0.3
            if self.kind in ("uniformity", "invariance"):
                return 0.0
            elif self.kind in ("segment_lift", "driver", "test") and p < 0.05:
                confidence = max(confidence, p_conf)
            elif confidence == 0.5:
                confidence = p_conf
        if self.kind == "cluster":
            evidence = self.evidence if isinstance(self.evidence, dict) else {}
            sil = evidence.get("silhouette_score")
            quality = evidence.get("separation_quality")
            if (isinstance(sil, (int, float)) and sil < 0.35) or quality in ("weak", "none"):
                confidence = min(confidence, 0.25)
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

    def refresh_eligibility(self) -> bool:
        """Recompute whether this finding may appear in a primary insight.

        Explicit audit failures are hard gates. Unknown checks do not hide an
        otherwise useful descriptive observation, but its evidence tier still
        prevents it from becoming an intervention recommendation.
        """
        reasons: list[str] = []
        if self.kind in ("invariance", "uniformity"):
            if self.statistical_state != "demonstrated_equivalence" and not (
                isinstance(self.evidence, dict) and self.evidence.get("statistical_state") == "demonstrated_equivalence"
            ):
                reasons.append("equivalence was not tested")
        checks = self.evidence.get("checks", {}) if isinstance(self.evidence, dict) else {}
        if isinstance(checks, dict):
            labels = {
                "claim_verified": "claim verification failed",
                "robust": "robustness check failed",
                "leakage_free": "target leakage detected",
                "paradox_free": "subgroup reversal detected",
                "model_converged": "selected model did not converge",
            }
            for key, label in labels.items():
                if checks.get(key) is False:
                    reasons.append(label)
            if self.evidence_tier == "causal_action" and checks.get("causal_language_safe") is False:
                reasons.append("causal language was not supported")
        if self.kind in CAVEAT_FINDING_KINDS:
            reasons.append("method or coverage caveat")
        if self.statistical_state == "inconclusive":
            reasons.append("comparison did not establish difference or equivalence")
        self.eligibility_reasons = list(dict.fromkeys(reasons))
        self.primary_eligible = not self.eligibility_reasons
        return self.primary_eligible

    def recommendation_disposition(self) -> Literal["monitor", "investigate", "pilot", "act"]:
        """Return the strongest recommendation language supported by evidence.
        Resolves each finding's disposition once and stores outcome, unit, desired_direction,
        evidence_design, controllability, and source on self.
        """
        if self.disposition is not None:
            return self.disposition

        self.refresh_eligibility()

        # Determine source
        if not self.source:
            self.source = self.source_tool or str(self.evidence.get("source_tool") or self.kind)

        # Determine outcome and unit
        if not self.outcome:
            self.outcome = self.measure or self.evidence.get("target_column") or self.evidence.get("measure_column")
        if not self.unit:
            u = self.evidence.get("unit") or self.evidence.get("measure_unit")
            self.unit = str(u) if u else None

        # Determine controllability
        feature = self.dimension or self.measure
        if not self.controllability:
            self.controllability = classify_feature_controllability(feature, self.kind, self.headline)

        # Determine evidence design
        if not self.evidence_design:
            if self.evidence_tier == "causal_action":
                self.evidence_design = "causal_inference"
            elif self.evidence_tier == "pilot":
                self.evidence_design = "pilot_experiment"
            elif self.evidence_tier == "hypothesis":
                self.evidence_design = "exploratory_hypothesis"
            elif self.kind in ("segment_lift", "driver", "correlation", "test"):
                self.evidence_design = "observational_association"
            elif self.kind in ("cluster", "outlier_scan", "cohort", "text", "dimensionality"):
                self.evidence_design = "descriptive"
            else:
                self.evidence_design = "observational"

        # Determine desired direction if not set
        if not self.desired_direction and self.evidence.get("desired_direction") in ("increase", "decrease"):
            self.desired_direction = self.evidence["desired_direction"]

        # 1. Eligibility gate
        if not self.primary_eligible:
            self.disposition = "investigate"
            return self.disposition

        # 2. Sensor telemetry / passive meters:
        # Must identify meters' relationship as sensor telemetry / additive components of active power,
        # and MUST NEVER recommend raising consumption or changing meter readings.
        if self.controllability == "sensor_telemetry":
            self.disposition = "monitor"
            return self.disposition

        # 3. Descriptive findings with actionable business advice (e.g. duplicate billing cluster)
        if self.controllability == "descriptive":
            hl_lower = (self.headline + " " + self.detail).lower()
            if any(term in hl_lower for term in ("duplicate", "billing", "anomaly", "error", "mismatch", "defect", "cluster", "leakage", "hygiene")):
                self.disposition = "investigate"
                return self.disposition
            self.disposition = "monitor"
            return self.disposition

        # 4. Uncontrollable / environmental features
        if self.controllability == "uncontrollable":
            self.disposition = "monitor"
            return self.disposition

        # 5. Causal rollout recommendation ('act')
        # Requires evidence supporting intervention (causal_action or validated experiment)
        # and explicit feasibility / manipulable lever.
        if self.evidence_tier == "causal_action" and self.controllability == "manipulable":
            if self.evidence.get("feasibility_supported") is True:
                self.disposition = "act"
                return self.disposition
            else:
                self.disposition = "pilot"
                return self.disposition

        # 6. Controlled pilot ('pilot')
        # Controlled pilot requires:
        # - manipulable lever (controllability == 'manipulable')
        # - measurable outcome (self.outcome is not None)
        # - candidate for prospective pilot: evidence_tier == "pilot", or robust contrast with adequate sample (n >= 100)
        sample_sizes = [
            v for k in (*_SAMPLE_SIZE_KEYS, "n_rest", "segment_n", "population_size", "n_records")
            if isinstance(v := self.evidence.get(k), (int, float)) and not isinstance(v, bool)
        ]
        has_adequate_sample = min(sample_sizes) >= 100 if sample_sizes else True
        if (
            self.controllability == "manipulable"
            and self.outcome is not None
            and (
                self.evidence_tier == "pilot"
                or (
                    self.kind in ("segment_lift", "decision_scenario")
                    and self.effective_confidence() >= 0.70
                    and has_adequate_sample
                    and not any("paradox" in c.lower() for c in self.caveats)
                )
            )
        ):
            self.disposition = "pilot"
            return self.disposition

        # 7. Measured association (observational)
        # Measured association may justify 'investigate' (or 'monitor'), NEVER 'act'.
        if self.controllability == "manipulable":
            if isinstance(self.effect, (int, float)) and self.effect < -0.15:
                self.disposition = "investigate"
            elif self.effective_confidence() >= 0.50:
                self.disposition = "investigate"
            else:
                self.disposition = "monitor"
            return self.disposition

        self.disposition = "monitor"
        return self.disposition

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
            "raw_importance_rank": self.raw_importance_rank,
            "display_rank": self.display_rank,
            "ranking_policy": self.ranking_policy,
            "caveats": self.caveats,
            "chart_hint": self.chart_hint,
            "layer": self.layer,
            "evidence_tier": self.evidence_tier,
            "primary_eligible": self.primary_eligible,
            "eligibility_reasons": self.eligibility_reasons,
            "disposition": self.disposition,
            "outcome": self.outcome,
            "unit": self.unit,
            "desired_direction": self.desired_direction,
            "evidence_design": self.evidence_design,
            "controllability": self.controllability,
            "source": self.source or self.source_tool,
            "statistical_state": self.statistical_state,
            "practical_margin": self.practical_margin,
        }


def is_data_hygiene_finding(item: Finding | dict[str, Any]) -> bool:
    """Return whether a finding describes dataset hygiene rather than a business lever.

    Integrity-audit findings currently use the broad ``outlier_scan`` kind, so
    source and headline checks are intentionally part of this shared predicate.
    Keeping the rule here prevents the action matrix and report projections from
    drifting apart.
    """
    if isinstance(item, Finding):
        kind = item.kind
        headline = item.headline
        source_tool = item.source_tool
    else:
        kind = str(item.get("kind", ""))
        headline = str(item.get("headline", ""))
        source_tool = str(item.get("source_tool", ""))
    if kind in HYGIENE_FINDING_KINDS or source_tool in HYGIENE_SOURCE_TOOLS:
        return True
    # Headline prefix check only applies to audit/hygiene tools or generic scans,
    # preventing analytical discoveries (e.g. "Missing values in income predict churn")
    # from being incorrectly classified as data hygiene.
    if kind in ("outlier_scan", "data_quality", "integrity", "") or not source_tool:
        return headline.lower().strip().startswith(HYGIENE_HEADLINE_PREFIXES)
    return False


def attach_finding_checks(
    findings: list[Finding],
    *,
    leakage_alerts: list[dict[str, Any]] | None = None,
    fragility_by_id: dict[str, bool] | None = None,
    causal_flagged_ids: set[str] | None = None,
    simpsons_by_id: dict[str, bool] | None = None,
) -> None:
    """Write the audited-entry check marks (FrontendPlan.md section 5) into
    each finding's `evidence["checks"]`, in place.

    Every check is derived only from data the finding, or a caller-supplied
    per-run audit result, already carries — nothing here re-runs a test or
    invents a verdict. A check that did not run for a given finding is left
    out of its `checks` dict entirely; the UI must render "nothing", never a
    fake tick, for a key that is absent (see FrontendPlan.md risk #2).

    Callers pass the per-finding results of audits that already ran
    elsewhere in the pipeline (`detect_target_leakage`, `audit_finding_sensitivity`,
    `audit_simpsons_paradox`, `audit_findings_causal_language` — see `AgentController`),
    because those audits key off dataframes and study design this module has no access to.
    """
    leaked_columns = {
        str(alert["column"]) for alert in (leakage_alerts or []) if alert.get("column")
    }
    fragility_by_id = fragility_by_id or {}
    causal_flagged_ids = causal_flagged_ids or set()
    simpsons_by_id = simpsons_by_id or {}

    for f in findings:
        evidence = f.evidence if isinstance(f.evidence, dict) else {}
        raw_checks = evidence.get("checks")
        checks: dict[str, Any] = dict(raw_checks) if isinstance(raw_checks, dict) else {}
        checks["claim_verified"] = True

        p = f.p_adjusted if f.p_adjusted is not None else f.p_value
        if p is not None:
            if f.kind in ("uniformity", "invariance"):
                checks["not_luck"] = False
            else:
                checks["not_luck"] = p < 0.05
        if f.kind in ("uniformity", "invariance"):
            checks["invariance_verified"] = False

        sizes = [
            v for k in _SAMPLE_SIZE_KEYS
            if isinstance(v := evidence.get(k), (int, float)) and not isinstance(v, bool)
        ]
        if sizes:
            checks["records"] = min(sizes) >= _SMALL_N

        if f.finding_id in fragility_by_id:
            checks["robust"] = not fragility_by_id[f.finding_id]

        if f.finding_id in simpsons_by_id:
            checks["paradox_free"] = not simpsons_by_id[f.finding_id]

        if leakage_alerts is not None:
            checks["leakage_free"] = not bool({f.dimension, f.measure} & leaked_columns)
        if leaked_columns and ({f.dimension, f.measure} & leaked_columns):
            checks["leakage"] = True

        if causal_flagged_ids is not None:
            checks["causal_language_safe"] = f.finding_id not in causal_flagged_ids
        if f.finding_id in causal_flagged_ids:
            checks["cause"] = "neutral"

        gap = evidence.get("train_test_gap")
        if isinstance(gap, (int, float)) and not isinstance(gap, bool):
            checks["model_gap"] = gap <= _OVERFIT_GAP_THRESHOLD
        converged = evidence.get("converged")
        if isinstance(converged, bool):
            checks["model_converged"] = converged

        if checks:
            evidence["checks"] = checks
            f.evidence = evidence
        f.refresh_eligibility()


def _is_self_derived_finding(finding: Finding | dict[str, Any]) -> bool:
    """Detect if a finding represents a relationship between a base column and its
    own derived transform or two derived transforms of the same base variable."""
    if isinstance(finding, Finding):
        m, d = finding.measure, finding.dimension
        ev = finding.evidence or {}
    elif isinstance(finding, dict):
        m, d = finding.get("measure"), finding.get("dimension")
        ev = finding.get("evidence") or {}
    else:
        return False
    col_a = ev.get("col_a") if isinstance(ev, dict) else None
    col_b = ev.get("col_b") if isinstance(ev, dict) else None
    if is_derived_pair(m, d):
        return True
    if is_derived_pair(col_a, col_b):
        return True
    if is_derived_pair(m, col_b) or is_derived_pair(col_a, d):
        return True
    return False


def is_trivial(finding: Finding) -> bool:
    """Shared triviality predicate (T5): a finding whose effect is
    negligible, or that is explicitly tagged as a method-fit warning rather
    than a discovery, should not compete for headline space."""
    if _is_self_derived_finding(finding):
        return True
    if finding.statistical_state in ("demonstrated_equivalence", "inconclusive"):
        return False
    if finding.kind in ("uniformity", "invariance"):
        return False
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
        # Each LLM-authored finding is a deliberate, distinct discovery, not
        # one of many rows from a single tool call, so they share no cap.
        if f.kind in LLM_FINDING_KINDS or counts.get(key, 0) < max_per_source:
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


_NO_SKILL_LIFT = 0.05  # same floor train_model uses to call a model "barely beats no-skill"


def _drop_unskilled_drivers(findings: list[Finding]) -> list[Finding]:
    """Permutation-importance drivers (evaluate_model) of a target whose best
    model does not beat a no-skill baseline explain nothing: on pure-noise data
    they still name a "top" feature/level. Drop them; the model_performance
    finding, which carries the "not a usable predictor" caveat, stays."""
    unskilled = {
        f.measure for f in findings
        if f.kind == "model_performance" and f.measure
        and isinstance(lift := f.evidence.get("lift_over_baseline"), (int, float))
        and lift < _NO_SKILL_LIFT
    }
    if not unskilled:
        return findings
    return [
        f for f in findings
        if not (f.kind == "driver" and f.source_tool == "evaluate_model" and f.measure in unskilled)
    ]


def _is_weak_cluster(f: Finding) -> bool:
    if f.kind != "cluster":
        return False
    evidence = f.evidence if isinstance(f.evidence, dict) else {}
    sil = evidence.get("silhouette_score")
    quality = evidence.get("separation_quality")
    return (isinstance(sil, (int, float)) and sil < 0.35) or quality in ("weak", "none")


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
    findings = _drop_unskilled_drivers(findings)
    findings = [f for f in findings if not _is_self_derived_finding(f)]
    for f in findings:
        f.compute_importance()
    ranked = sorted(
        findings,
        key=lambda f: (
            not _is_weak_cluster(f),
            round(f.importance, 4),
            round(abs(f.effect), 4) if f.effect is not None else 0.0,
            round(f.effective_confidence(), 4),
            f.kind or "",
            f.measure or "",
            f.dimension or "",
            f.headline or "",
        ),
        reverse=True,
    )
    for raw_rank, finding in enumerate(ranked, 1):
        finding.raw_importance_rank = raw_rank
    if suppress_trivial:
        kept = [f for f in ranked if not is_trivial(f)]
        ranked = kept if kept else ranked
    result, dupes = _dedupe(ranked)
    if max_per_source is not None:
        result = _diversify_by_source(result, max_per_source)
    result += dupes
    ordered = (
        [f for f in result if f.kind not in CAVEAT_FINDING_KINDS]
        + [f for f in result if f.kind in CAVEAT_FINDING_KINDS]
    )
    for display_rank, finding in enumerate(ordered, 1):
        finding.display_rank = display_rank
        finding.ranking_policy = (
            "importance_only_v1" if max_per_source is None
            else "importance_then_source_diversity_v1"
        )
    return ordered


_OUTCOME_FEATURE_STEMS = (
    "revenue", "totalamount", "profit", "margin",
    "gmv", "income", "earnings", "target", "label", "loss", "churn",
    "churned", "conversionrate", "clickthrough",
)

_SENSOR_TELEMETRY_STEMS = (
    "submetering", "submeter", "metering", "meter", "globalactivepower", "globalreactivepower",
    "activepower", "reactivepower", "voltage", "globalintensity", "telemetry",
    "transducer", "sensor", "thermocouple", "flowrate", "ammeter", "voltmeter",
    "cogt", "nmhcgt", "c6h6gt", "noxgt", "no2gt", "pt08",
)


def is_sensor_telemetry_feature(col: str | None) -> bool:
    """Identify if a column represents passive physical or electrical sensor telemetry."""
    if not col or not isinstance(col, str):
        return False
    norm = re.sub(r"[^a-z0-9]", "", col.lower())
    return any(s in norm for s in _SENSOR_TELEMETRY_STEMS)


def classify_feature_controllability(col: str | None, kind: str | None = None, headline: str = "") -> str:
    """Classify feature controllability into 4 MECE categories:
    - 'sensor_telemetry': physical measurements, electrical meters, power loads, telemetry sensors
    - 'descriptive': clusters, unsupervised segments, data hygiene, billing discrepancies
    - 'uncontrollable': external market quotes, surrogate IDs, timestamps, weather, environment
    - 'manipulable': controllable operational or business levers (pricing, discount, policies)
    """
    if is_sensor_telemetry_feature(col):
        return "sensor_telemetry"
    hl_lower = headline.lower() if headline else ""
    if kind in ("cluster", "outlier_scan", "cohort", "text", "dimensionality"):
        return "descriptive"
    if col and any(col.lower().startswith(p) for p in ("cluster", "segment")):
        return "descriptive"
    if any(k in hl_lower for k in ("duplicate", "billing", "anomaly", "hygiene", "mismatch", "defect")):
        return "descriptive"
    if not _is_actionable_feature(col):
        return "uncontrollable"
    return "manipulable"


def _is_actionable_feature(col: str | None) -> bool:
    """Determine if a column is a controllable business/system lever rather than
    an external market quote, surrogate key, system timestamp, weather condition,
    outcome/revenue KPI, sensor telemetry, or self-derived transform."""
    if not col or not isinstance(col, str):
        return False
    # Derived transforms (e.g. _log_return, _lag1, _delta1) are not direct levers
    if get_base_column_name(col).lower() != col.strip().lower():
        return False
    norm = re.sub(r"[^a-z0-9]", "", col.lower())
    # Sensor telemetry / physical electrical meters / power loads
    if is_sensor_telemetry_feature(col):
        return False
    # Outcome/KPI targets are metrics to optimize, not direct operational levers
    clean_stems = tuple(s.replace("_", "") for s in _OUTCOME_FEATURE_STEMS)
    if any(norm == s or norm.startswith(s) or norm.endswith(s) for s in clean_stems):
        return False
    if norm in ("amount", "totalamount", "spend", "totalspend", "customerspend"):
        return False
    if norm.startswith(("sales",)) and not norm.startswith(("salesrep", "saleschannel", "salesperson")):
        return False
    # External market odds, betting quotes, tickers, fx rates
    if any(norm.startswith(p) for p in ("b365", "bw", "iw", "lb", "ps", "wh", "vc", "bet", "odds", "quote", "ticker", "fx")):
        return False
    if "odds" in norm or "prob" in norm or "exchange" in norm:
        return False
    # Surrogate IDs and keys
    if norm.endswith(("id", "key", "code", "guid", "uuid", "pk", "fk", "index")) or norm in ("id", "key", "code", "guid", "uuid", "pk", "fk"):
        return False
    # System timestamps / temporal parts
    if any(t in norm for t in ("timestamp", "date", "time", "year", "month", "day", "hour", "minute", "second", "createdat", "updatedat")):
        return False
    # Environmental / weather conditions
    if any(w in norm for w in ("weather", "temperature", "humidity", "rainfall", "wind", "precipitation", "celsius", "fahrenheit")):
        return False
    # Synthetic cluster/segment labels from unsupervised ML
    if norm.startswith("cluster") or norm.startswith("segment"):
        return False
    return True


def classify_action_matrix(
    findings: list[Finding] | list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Classify findings into an actionable 2x2 Impact vs Feasibility matrix.

    Returns:
        {
            "quick_wins": [...],          # High Impact, High Feasibility (immediate execution)
            "strategic_bets": [...],       # High Impact, Lower Feasibility / Structural
            "incremental": [...],          # Moderate/Low Impact, High Feasibility (routine optimizations)
            "watch_caveats": [...],        # Needs Guardrails / Paradoxes / Fragility / High Uncertainty
        }
    """
    matrix: dict[str, list[dict[str, Any]]] = {
        "quick_wins": [],
        "strategic_bets": [],
        "incremental": [],
        "pilot": [],
        "watch_caveats": [],
    }
    matrix["a_b_test"] = matrix["pilot"]

    for item in findings:
        evidence: dict[str, Any] = {}
        if isinstance(item, Finding):
            item.refresh_eligibility()
            finding_id = item.finding_id
            kind = item.kind
            headline = item.headline
            detail = item.detail
            dimension = item.dimension
            level = item.level
            effect = item.effect
            conf = item.effective_confidence()
            norm_eff = item.normalized_effect()
            imp = item.importance or item.compute_importance()
            caveats = list(item.caveats)
            evidence_tier: EvidenceTier = item.evidence_tier
            primary_eligible = item.primary_eligible
            eligibility_reasons = list(item.eligibility_reasons)
            if isinstance(item.evidence, dict):
                evidence = item.evidence
        elif isinstance(item, dict):
            finding_id = str(item.get("finding_id", ""))
            kind = str(item.get("kind", ""))
            headline = str(item.get("headline", ""))
            detail = str(item.get("detail", ""))
            measure = item.get("measure")
            dimension = item.get("dimension")
            level = item.get("level")
            effect = item.get("effect")
            effect_kind = item.get("effect_kind")
            source_tool = str(item.get("source_tool", ""))
            conf = float(item.get("confidence", 0.5))
            norm_eff = abs(float(effect)) if isinstance(effect, (int, float)) else 0.0
            imp = float(item.get("importance", 0.5))
            caveats = list(item.get("caveats") or [])
            raw_tier = str(item.get("evidence_tier", "observation"))
            evidence_tier = cast(
                EvidenceTier,
                raw_tier
                if raw_tier in {"observation", "hypothesis", "pilot", "causal_action"}
                else "observation",
            )
            primary_eligible = bool(item.get("primary_eligible", True))
            eligibility_reasons = [str(v) for v in item.get("eligibility_reasons", [])]
            raw_ev = item.get("evidence")
            if isinstance(raw_ev, dict):
                evidence = raw_ev
        else:
            continue

        checks = evidence.get("checks", {}) if isinstance(evidence, dict) else {}
        has_paradox = checks.get("paradox_free") is False
        is_fragile = checks.get("robust") is False
        is_caveat_kind = kind in CAVEAT_FINDING_KINDS

        # Data hygiene belongs in the dedicated integrity section, never in
        # a strategic or operational action quadrant.
        if is_data_hygiene_finding(item):
            continue

        # Self-derived mathematical tautologies are dropped completely
        if _is_self_derived_finding(item):
            continue

        layer = getattr(item, "layer", None) or (item.get("layer") if isinstance(item, dict) else "")
        is_appendix = str(layer).lower() == "appendix"

        target_feature = dimension or getattr(item, "measure", None) or (item.get("measure") if isinstance(item, dict) else None)
        is_controllable = _is_actionable_feature(target_feature)

        if isinstance(item, Finding):
            disposition = item.recommendation_disposition()
            controllability = item.controllability or classify_feature_controllability(target_feature, kind, headline)
        else:
            raw_disp = item.get("disposition")
            controllability = item.get("controllability") or classify_feature_controllability(target_feature, kind, headline)
            if raw_disp in ("monitor", "investigate", "pilot", "act"):
                disposition = raw_disp
            else:
                temp_f = Finding(
                    finding_id=finding_id, kind=kind, headline=headline, detail=detail,
                    evidence=evidence, source_tool=source_tool, measure=measure,
                    dimension=dimension, level=level, effect=effect, effect_kind=effect_kind,
                    confidence=conf, evidence_tier=evidence_tier, caveats=caveats,
                    primary_eligible=primary_eligible, eligibility_reasons=eligibility_reasons,
                )
                disposition = temp_f.recommendation_disposition()
                controllability = temp_f.controllability or classify_feature_controllability(target_feature, kind, headline)

        level_label = evidence.get("level_label") if isinstance(evidence, dict) else None
        lever = headline
        if kind == "driver" and dimension and level:
            lever = f"Target {dimension}='{level}'"
        elif kind == "decision_scenario" and dimension:
            lever = f"Simulate {dimension} intervention"
        elif kind == "interaction_synergy" and dimension and level:
            lever = f"Target synergy: {dimension}='{level}'"
        elif kind in ("correlation", "association"):
            meas = getattr(item, "measure", None) or (item.get("measure") if isinstance(item, dict) else None)
            lever = f"Context: {meas} & {dimension} (Correlation)" if (meas and dimension) else f"Context: {meas or dimension}"
        elif dimension and level:
            if level_label and str(level_label) != str(level):
                lever = f"Target '{level_label}' ({dimension}='{level}')"
            else:
                lever = f"Target {dimension}='{level}'"
        elif dimension:
            lever = f"Target {dimension} segment" if kind == "segment_lift" else f"Target {dimension}"
        elif target_feature:
            lever = f"Target {target_feature}" if is_controllable else f"Outcome metric: {target_feature}"

        entry = {
            "finding_id": finding_id,
            "headline": headline,
            "lever": lever,
            "kind": kind,
            "effect": effect,
            "confidence": round(conf, 2),
            "importance": round(imp, 3),
            "caveats": caveats,
            "evidence_tier": evidence_tier,
            "disposition": disposition,
            "controllability": controllability,
        }

        # Check sensor telemetry:
        if controllability == "sensor_telemetry" or is_sensor_telemetry_feature(target_feature):
            entry["rationale"] = (
                "Sensor telemetry / additive component of active power load — "
                "passive instrumentation readout; do not treat as an intervention lever or manipulate meter readings."
            )
            entry["quadrant"] = "watch_caveats"
            entry["impact_level"] = "Context"
            entry["feasibility_level"] = "Monitoring"
            matrix["watch_caveats"].append(entry)
            continue

        # Preserve actionable descriptive advice (e.g. duplicate billing cluster)
        if any(phrase in (headline + " " + detail).lower() for phrase in ("duplicate", "billing", "anomaly", "mismatch", "defect", "cluster", "leakage", "hygiene")):
            entry["lever"] = f"Investigate data integrity: {lever}" if not lever.startswith("Investigate") else lever
            entry["rationale"] = "Descriptive operational finding: investigate duplicate records and data hygiene before changing downstream policy."
            entry["quadrant"] = "watch_caveats"
            entry["impact_level"] = "High"
            entry["feasibility_level"] = "Audit / Investigation"
            matrix["watch_caveats"].append(entry)
            continue

        # Observational confidence is not intervention validity.
        # Check sample size adequacy: entity-aggregated data (e.g. states, cities, stores)
        # has a lower natural n cutoff (n >= 20) than raw row-level records (n >= 100).
        sample_sizes = [
            v for k in (*_SAMPLE_SIZE_KEYS, "segment_n", "population_size", "n_records")
            if isinstance(v := evidence.get(k), (int, float)) and not isinstance(v, bool)
        ]
        is_entity_aggregated = any("repeated rows aggregated" in str(c).lower() or "per '" in str(c).lower() for c in caveats) or bool(evidence.get("entity_column"))
        min_sample_threshold = 20 if is_entity_aggregated else 100
        has_adequate_sample = (
            min(sample_sizes) >= min_sample_threshold
            if sample_sizes
            else (checks.get("records") is not False)
        )
        sample_size_val = min(sample_sizes) if sample_sizes else None
        entry["sample_size"] = sample_size_val

        # Detect data defect caveats (imputed data, flatlines, extreme skew)
        has_data_defect = any(
            any(phrase in c.lower() for phrase in ("imputed", "missing data", "zero variance", "flatline", "extreme skew"))
            for c in caveats
        )

        # Observational disclaimers that suggest prospective pilot validation
        has_observational_disclaimer = bool(caveats) and any(
            any(phrase in c.lower() for phrase in (
                "proof of cause", "adjusted association", "observational", "observation", "unadjusted",
                "selection effect", "confounding", "additive driver",
            ))
            for c in caveats
        )
        is_observational_segment_contrast = (
            primary_eligible
            and is_controllable
            and not is_appendix
            and not is_caveat_kind
            and not has_paradox
            and not is_fragile
            and has_adequate_sample
            and conf >= 0.50
            and (kind in ("segment_lift", "decision_scenario") or evidence_tier == "pilot")
            and (has_observational_disclaimer or evidence_tier == "pilot")
        )
        if is_observational_segment_contrast and evidence_tier != "causal_action":
            if disposition != "pilot":
                entry.update(lever=headline, rationale="Investigate the observed difference before changing policy.",
                             quadrant="watch_caveats", impact_level="Context", feasibility_level="Investigation")
                matrix["watch_caveats"].append(entry)
                continue

            entry["lever"] = f"Pilot A/B test: {lever}" if not lever.startswith("Pilot") else lever
            entry["rationale"] = (
                f"High-contrast observational segment ({int(conf * 100)}% confidence); "
                f"validate via prospective A/B test or pilot before scaling."
            )
            entry["quadrant"] = "pilot"
            entry["impact_level"] = "High"
            entry["feasibility_level"] = "Pilot Validation"
            matrix["pilot"].append(entry)
            continue

        is_correlation = kind in ("correlation", "association")
        raw_source = getattr(item, "source_tool", None) or (item.get("source_tool") if isinstance(item, dict) else "")
        if raw_source in ("correlation_analysis",):
            is_correlation = True

        # Non-actionable or fatal risk items route to watch_caveats:
        # Reversals, fragility, data defects, non-controllable factors, correlations, or sub-threshold confidence.
        has_fatal_caveat = any(
            any(p in c.lower() for p in (
                "direction reverses", "possible confounding", "does not hold within",
                "sensitive to outlier trimming", "fragile", "extreme skew",
                "zero variance", "flatline", "imputed", "missing data",
                "overfit", "memorised", "leakage", "parallel trends violated", "positivity violated",
            ))
            for c in caveats
        )

        if (
            not primary_eligible
            or is_caveat_kind
            or has_paradox
            or is_fragile
            or has_data_defect
            or has_fatal_caveat
            or conf < 0.45
            or is_correlation
            or (not is_controllable and not is_appendix)
            or (sample_sizes and not has_adequate_sample)
        ):
            reason = []
            reason.extend(eligibility_reasons)
            if is_correlation:
                reason.append("Observational correlation — monitor as context or covariate, not an intervention lever")
            if not is_controllable:
                reason.append(f"External non-controllable factor ('{target_feature}') — monitor as risk/context, not direct lever")
            if has_paradox:
                reason.append("Subgroup reversal detected (Simpson's paradox)")
            if is_fragile:
                reason.append("Sensitive to outlier trimming")
            if has_data_defect:
                reason.append("Data hygiene/imputation caveat — review data quality before action")
            if sample_sizes and not has_adequate_sample:
                reason.append(f"Small sample size (n < {min_sample_threshold}) for observational contrast")
            if caveats:
                clean_cavs = [
                    c for c in caveats
                    if c not in reason
                    and not any(p in c.lower() for p in ("not proof of cause", "observation evidence", "do not treat as a causal lever"))
                ]
                reason.extend(clean_cavs[:2])
            if not reason:
                reason.append("Low statistical confidence or caveat kind")
            entry["rationale"] = "; ".join(reason)
            entry["quadrant"] = "watch_caveats"
            entry["impact_level"] = "Context"
            entry["feasibility_level"] = "Monitoring"
            matrix["watch_caveats"].append(entry)
            continue

        if is_appendix or not is_controllable or is_correlation:
            continue

        # 2. Impact evaluation
        # High impact requires substantial normalized effect, raw lift >= 15%, high importance >= 0.40,
        # or targeted structural synergies/scenarios with measurable magnitude.
        is_high_impact = (
            norm_eff >= 0.15
            or (isinstance(effect, (int, float)) and abs(effect) >= 0.15)
            or imp >= 0.40
            or (kind in ("decision_scenario", "interaction_synergy") and (effect is None or abs(effect or 0) >= 0.08))
            or (kind == "driver" and (norm_eff >= 0.10 or abs(effect or 0) >= 0.10 or imp >= 0.35))
        )

        # 3. Feasibility evaluation
        # High feasibility means high confidence and direct operational controllability without
        # cross-functional structural alignment.
        is_high_feasibility = (
            conf >= 0.55
            and kind in ("segment_lift", "driver", "concentration", "decision_scenario")
            and kind != "interaction_synergy"  # synergies are structural bets
        )

        if is_high_impact and is_high_feasibility:
            if disposition == "act":
                entry["rationale"] = f"High lift with robust statistical confidence ({int(conf * 100)}%). Direct operational execution candidate."
                entry["quadrant"] = "quick_wins"
                entry["impact_level"] = "High"
                entry["feasibility_level"] = "Immediate Execution"
                matrix["quick_wins"].append(entry)
            elif disposition == "pilot":
                entry["lever"] = f"Pilot A/B test: {lever}" if not lever.startswith("Pilot") else lever
                entry["rationale"] = f"Candidate for controlled pilot ({int(conf * 100)}% confidence); validate via prospective test before scaling."
                entry["quadrant"] = "pilot"
                entry["impact_level"] = "High"
                entry["feasibility_level"] = "Pilot Validation"
                matrix["pilot"].append(entry)
            else:
                entry["rationale"] = f"Observational association ({int(conf * 100)}% confidence); investigate underlying drivers before policy changes."
                entry["quadrant"] = "watch_caveats"
                entry["impact_level"] = "High"
                entry["feasibility_level"] = "Investigation"
                matrix["watch_caveats"].append(entry)
        elif is_high_impact:
            if disposition == "act":
                entry["rationale"] = "High potential upside; requires structural or cross-dimensional alignment."
                entry["quadrant"] = "strategic_bets"
                entry["impact_level"] = "High"
                entry["feasibility_level"] = "Structural Initiative"
                matrix["strategic_bets"].append(entry)
            elif disposition == "pilot":
                entry["lever"] = f"Pilot A/B test: {lever}" if not lever.startswith("Pilot") else lever
                entry["rationale"] = "High potential upside; validate via controlled pilot before structural alignment."
                entry["quadrant"] = "pilot"
                entry["impact_level"] = "High"
                entry["feasibility_level"] = "Pilot Validation"
                matrix["pilot"].append(entry)
            else:
                entry["rationale"] = "Observational signal with high potential upside; investigate before committing resources."
                entry["quadrant"] = "watch_caveats"
                entry["impact_level"] = "High"
                entry["feasibility_level"] = "Investigation"
                matrix["watch_caveats"].append(entry)
        elif is_high_feasibility:
            if disposition == "act":
                entry["rationale"] = "Low-risk operational tuning with clear evidence."
                entry["quadrant"] = "incremental"
                entry["impact_level"] = "Moderate"
                entry["feasibility_level"] = "Operational Optimization"
                matrix["incremental"].append(entry)
            elif disposition == "pilot":
                entry["lever"] = f"Pilot A/B test: {lever}" if not lever.startswith("Pilot") else lever
                entry["rationale"] = "Low-risk operational candidate; validate with small pilot."
                entry["quadrant"] = "pilot"
                entry["impact_level"] = "Moderate"
                entry["feasibility_level"] = "Pilot Validation"
                matrix["pilot"].append(entry)
            else:
                entry["rationale"] = "Observational contrast with moderate impact; monitor or investigate as routine context."
                entry["quadrant"] = "incremental"
                entry["impact_level"] = "Moderate"
                entry["feasibility_level"] = "Operational Optimization"
                matrix["incremental"].append(entry)
        else:
            entry["rationale"] = "Moderate effect with standard feasibility."
            entry["quadrant"] = "incremental"
            entry["impact_level"] = "Moderate"
            entry["feasibility_level"] = "Operational Optimization"
            matrix["incremental"].append(entry)

    return matrix
