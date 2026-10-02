"""Tests for src/core/run_view.py: the read-only results object."""
from __future__ import annotations

import dataclasses
from typing import Any

import pytest

from src.core.audited_entry import audited_checks
from src.core.run_view import (
    MAX_FALLBACKS,
    MAX_HEADLINE_FINDINGS,
    MAX_HYPOTHESES,
    MAX_UNTRACED_EXAMPLES,
    RUN_VIEW_CONTEXT_KEYS,
    RunView,
    build_run_view,
    compute_verdict,
    select_headline_findings,
)


def _finding(
    fid: str,
    *,
    layer: str = "exec",
    kind: str = "effect",
    checks: dict[str, Any] | None = None,
) -> dict[str, Any]:
    evidence: dict[str, Any] = {"n": 120}
    if checks is not None:
        evidence["checks"] = checks
    return {"id": fid, "headline": f"headline {fid}", "layer": layer, "kind": kind, "evidence": evidence}


def _full_context() -> dict[str, Any]:
    return {
        "analysis_decision": {
            "mode": "describe",
            "target": None,
            "candidate": "amount",
            "rationale": "No prediction was requested.",
            "alternatives_rejected": ["regression on 'amount'"],
        },
        "degradations": ["Model training skipped: too few rows.", "Chart fell back to a table."],
        "unverified_claims": ["42 percent", "3.1x"],
        "deliverable_audit": {"delivered": ["a chart"], "missing": ["a forecast"], "repaired": ["a table"]},
        "llm_usage": {"call_count": 7, "total_tokens": 12345, "estimated_cost_usd": 0.0421, "is_estimate": True},
        "api_telemetry": {"anything": 1},
        "hypothesis_tree": {
            "primary_objective": "why churn",
            "nodes": {
                "h0": {"statement": "Primary Goal: why churn", "status": "untested", "confidence": 0.5},
                "h1": {"statement": "Tenure drives churn", "status": "supported", "confidence": 0.8},
                "h2": {"statement": "Region drives churn", "status": "refuted", "confidence": 0.2},
                "h3": {"statement": "Price drives churn", "status": "inconclusive", "confidence": 0.5},
            },
        },
    }


def _full_result() -> dict[str, Any]:
    return {
        "reasoning": "Churn is driven by tenure.",
        "insights": ["an insight"],
        "recommendations": ["Call new customers early."],
        "coverage": {"answered": 2, "unanswered": [{"text": "Why in March?"}]},
        "findings": [
            _finding("a", checks={"not_luck": True, "records": True}),
            _finding("b", checks={"not_luck": False}),
            _finding("c", layer="exec", kind="method_fit"),
        ],
    }


# ── full snapshot ────────────────────────────────────────────────────────────


def test_full_snapshot_populates_every_field() -> None:
    view = build_run_view(_full_result(), _full_context(), objective="  why churn?  ", is_sample=False)

    assert view.objective == "why churn?"
    assert view.is_sample is False
    assert view.reasoning == "Churn is driven by tenure."
    assert view.recommendations == ("Call new customers early.",)
    assert view.insights == ("an insight",)
    assert view.coverage["unanswered"] == [{"text": "Why in March?"}]
    assert len(view.findings) == 3
    assert [f["id"] for f in view.headline_findings] == ["a", "b"]

    how = view.how
    assert how.decision is not None
    assert how.decision.mode == "describe"
    assert how.decision.rationale == "No prediction was requested."
    assert how.decision.rejected == ("regression on 'amount'",)
    assert how.fallbacks == ("Model training skipped: too few rows.", "Chart fell back to a table.")
    assert how.fallbacks_total == 2
    assert how.untraced_numbers == ("42 percent", "3.1x")
    assert how.untraced_total == 2
    assert how.deliverables is not None
    assert how.deliverables.missing == ("a forecast",)
    assert how.deliverables.repaired == ("a table",)
    assert how.usage is not None
    assert (how.usage.calls, how.usage.tokens, how.usage.is_estimate) == (7, 12345, True)
    assert how.usage.cost_usd == pytest.approx(0.0421)
    assert how.no_ai is False


def test_run_view_is_frozen() -> None:
    view = build_run_view(_full_result(), _full_context(), objective="q", is_sample=False)
    with pytest.raises(dataclasses.FrozenInstanceError):
        view.objective = "other"  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        view.how.usage = None  # type: ignore[misc]


def test_context_keys_cover_every_how_input() -> None:
    assert set(RUN_VIEW_CONTEXT_KEYS) == {
        "analysis_decision", "degradations", "unverified_claims",
        "deliverable_audit", "llm_usage", "api_telemetry", "hypothesis_tree",
    }


# ── no-AI run ────────────────────────────────────────────────────────────────


def test_no_ai_run_has_no_usage_but_keeps_the_decision() -> None:
    result = _full_result() | {"deterministic_mode": True}
    context = {
        "analysis_decision": _full_context()["analysis_decision"],
        "llm_usage": {"call_count": 0, "total_tokens": 0, "estimated_cost_usd": 0.0, "is_estimate": True},
    }
    view = build_run_view(result, context, objective="", is_sample=True)

    assert view.is_sample is True
    assert view.how.usage is None
    assert view.how.no_ai is True
    assert view.how.decision is not None
    assert view.how.fallbacks == ()
    assert view.how.untraced_numbers == ()
    assert view.how.hypotheses == ()


def test_missing_usage_alone_does_not_claim_no_ai() -> None:
    view = build_run_view(_full_result(), {}, objective="q", is_sample=False)
    assert view.how.usage is None
    assert view.how.no_ai is False


# ── planted cases ────────────────────────────────────────────────────────────


def test_planted_refuted_hypothesis_and_missing_deliverable_surface() -> None:
    view = build_run_view(_full_result(), _full_context(), objective="q", is_sample=False)

    statuses = {h.statement: h.status for h in view.how.hypotheses}
    assert statuses["Region drives churn"] == "refuted"
    assert statuses["Tenure drives churn"] == "supported"
    # refuted first, and the untested primary goal is not a tested idea
    assert view.how.hypotheses[0].status == "refuted"
    assert "Primary Goal: why churn" not in statuses
    assert dict(view.how.hypothesis_counts) == {"refuted": 1, "supported": 1, "inconclusive": 1}
    assert view.how.deliverables is not None
    assert "a forecast" in view.how.deliverables.missing


# ── absent / malformed inputs ────────────────────────────────────────────────


@pytest.mark.parametrize("final_result", [None, {}, {"findings": None}, {"findings": "nope"}])
@pytest.mark.parametrize("context", [None, {}, dict.fromkeys(RUN_VIEW_CONTEXT_KEYS)])
def test_absent_inputs_never_raise_and_never_fabricate(
    final_result: dict[str, Any] | None, context: dict[str, Any] | None
) -> None:
    view = build_run_view(final_result, context, objective="", is_sample=False)

    assert isinstance(view, RunView)
    assert view.verdict is None
    assert view.findings == ()
    assert view.headline_findings == ()
    assert view.recommendations == ()
    assert view.coverage == {}
    assert view.reasoning == ""
    how = view.how
    assert how.decision is None
    assert how.deliverables is None
    assert how.usage is None
    assert how.no_ai is False
    assert how.fallbacks == how.untraced_numbers == how.hypotheses == ()


def test_wrong_types_inside_context_are_ignored() -> None:
    context: dict[str, Any] = {
        "analysis_decision": "describe",
        "degradations": "not a list",
        "unverified_claims": [1, None, "  ", "real one"],
        "deliverable_audit": {"delivered": "x", "missing": None, "repaired": 3},
        "llm_usage": {"call_count": "7", "total_tokens": True},
        "hypothesis_tree": {"nodes": {"h": "bad", "g": {"statement": "", "status": "refuted"}}},
    }
    view = build_run_view({"findings": [1, "x", {"id": "ok"}]}, context, objective="q", is_sample=False)

    assert view.how.decision is None
    assert view.how.fallbacks == ()
    assert view.how.untraced_numbers == ("real one",)
    assert view.how.deliverables is None
    assert view.how.usage is None
    assert view.how.hypotheses == ()
    assert len(view.findings) == 1


def test_decision_without_a_mode_is_absent() -> None:
    view = build_run_view({}, {"analysis_decision": {"rationale": "orphan"}}, objective="", is_sample=False)
    assert view.how.decision is None


def test_cost_flag_defaults_to_estimate_when_absent() -> None:
    usage = {"call_count": 2, "total_tokens": 10, "estimated_cost_usd": 0.01}
    view = build_run_view({}, {"llm_usage": usage}, objective="", is_sample=False)
    assert view.how.usage is not None
    assert view.how.usage.is_estimate is True


# ── caps keep the true count ─────────────────────────────────────────────────


def test_caps_apply_but_totals_stay_true() -> None:
    context: dict[str, Any] = {
        "degradations": [f"fallback {i}" for i in range(MAX_FALLBACKS + 4)],
        "unverified_claims": [f"claim {i}" for i in range(MAX_UNTRACED_EXAMPLES + 6)],
        "hypothesis_tree": {
            "nodes": {
                f"h{i}": {"statement": f"idea {i}", "status": "supported", "confidence": 0.7}
                for i in range(MAX_HYPOTHESES + 3)
            }
        },
    }
    view = build_run_view({}, context, objective="", is_sample=False)

    assert len(view.how.fallbacks) == MAX_FALLBACKS
    assert view.how.fallbacks_total == MAX_FALLBACKS + 4
    assert len(view.how.untraced_numbers) == MAX_UNTRACED_EXAMPLES
    assert view.how.untraced_total == MAX_UNTRACED_EXAMPLES + 6
    assert len(view.how.hypotheses) == MAX_HYPOTHESES
    assert dict(view.how.hypothesis_counts) == {"supported": MAX_HYPOTHESES + 3}


def test_headline_findings_capped_at_five() -> None:
    findings = tuple(_finding(str(i)) for i in range(MAX_HEADLINE_FINDINGS + 3))
    assert len(select_headline_findings(findings)) == MAX_HEADLINE_FINDINGS


# ── selection and verdict parity with the Answers tab ────────────────────────


def _legacy_answers_tab_verdict(report: dict[str, Any]) -> tuple[int, int] | None:
    """The computation `render_answers_tab` did inline before RunView, verbatim."""
    all_findings: list[dict[str, Any]] = report.get("findings") or []
    card_findings = [
        f
        for f in all_findings
        if f.get("layer") in ("exec", "analyst")
        and f.get("kind") not in ("method_fit", "coverage_gap")
    ][:5]
    audited = [(f, audited_checks(f.get("evidence"))) for f in card_findings]
    audited = [(f, marks) for f, marks in audited if marks]
    m_count = len(audited)
    if m_count == 0:
        return None
    held_up = sum(1 for _, marks in audited if not any(mk.state == "fail" for mk in marks))
    return held_up, m_count


def test_selection_excludes_method_notes_and_other_layers() -> None:
    findings = (
        _finding("keep1", layer="exec"),
        _finding("keep2", layer="analyst"),
        _finding("drop-kind", kind="coverage_gap"),
        _finding("drop-kind2", kind="method_fit"),
        _finding("drop-layer", layer="detail"),
    )
    assert [f["id"] for f in select_headline_findings(findings)] == ["keep1", "keep2"]


def test_verdict_matches_the_legacy_inline_computation() -> None:
    cases: list[list[dict[str, Any]]] = [
        [],
        [_finding("a")],  # no checks at all
        [_finding("a", checks={"not_luck": True})],
        [_finding("a", checks={"not_luck": True}), _finding("b", checks={"robust": False})],
        [
            _finding("a", checks={"leakage": True}),
            _finding("b", checks={"cause": True}),
            _finding("c", checks={"records": True, "model_gap": True}),
            _finding("d", kind="method_fit", checks={"not_luck": False}),
            _finding("e"),
            _finding("f", checks={"not_luck": True}),
            _finding("g", checks={"not_luck": False}),
        ],
    ]
    for findings in cases:
        report = {"findings": findings}
        expected = _legacy_answers_tab_verdict(report)
        got = build_run_view(report, {}, objective="", is_sample=False).verdict
        if expected is None:
            assert got is None
        else:
            assert got is not None
            assert (got.held_up, got.audited) == expected
            assert got.needs_more == got.audited - got.held_up


def test_verdict_is_none_when_no_audit_ran() -> None:
    assert compute_verdict((_finding("a"), _finding("b"))) is None
    assert compute_verdict(()) is None
