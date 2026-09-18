"""
Planted-effects recovery harness — IMPROVEMENTS.md 7.11.

Every fixture in tests/fixtures/planted_effects.py bakes in a documented,
known effect (a segment lift, a seasonal swing, a drawdown, a pay gap, a
dominant categorical driver, a within-group trend, a sentiment skew — or,
for one fixture, deliberately *no* plantable target at all). This file
runs the real, deterministic (`use_llm=False`) pipeline end to end on each
one and checks whether `result["findings"]` actually recovered the planted
effect — not just that the pipeline didn't crash.

This is NOT a smoke test. A fixture that used to pass could start failing
the moment a tool's threshold, sign convention, or ranking weight changes
upstream — that's the point: it means a real, documented effect stopped
being recoverable, which the shape-fixture matrix in test_data_shapes.py
has no way to catch.

Style/conventions mirrored from tests/test_controller.py (AgentController
import, tmp_path-based CSV fixtures) and tests/test_data_shapes.py
(fixtures imported from tests.fixtures, one generator per dataset shape).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from src.core.controller import AgentController
from tests.fixtures.planted_effects import (
    generate_churn,
    generate_hr_roster,
    generate_panel,
    generate_price_series,
    generate_pure_description,
    generate_text_column,
    generate_transactional,
)

# ---------------------------------------------------------------------------
# Assertion helpers
# ---------------------------------------------------------------------------

#: Criteria keys that carry a comparator prefix (`min_x`, `max_x`, `gt_x`,
#: `lt_x`, `gte_x`, `lte_x`) rather than requiring exact equality. The
#: field they test for is looked up first on the Finding dict's top-level
#: keys (e.g. `min_effect` -> `finding["effect"]`), then inside
#: `finding["evidence"]` (e.g. `min_pct` -> `finding["evidence"]["pct"]`) —
#: this is what lets a caller write
#: `assert_no_finding(findings, kind="outlier_scan", min_pct=30)` without
#: this helper needing to know `outlier_scan`'s evidence shape in advance.
_COMPARATORS = {
    "min": lambda a, b: a >= b,
    "max": lambda a, b: a <= b,
    "gte": lambda a, b: a >= b,
    "lte": lambda a, b: a <= b,
    "gt": lambda a, b: a > b,
    "lt": lambda a, b: a < b,
}


def _criterion_matches(finding: dict[str, Any], key: str, expected: Any) -> bool:
    """Single-criterion matcher shared by assert_finding/assert_no_finding.

    - `key == "evidence"` and `expected` a dict: partial match against
      `finding["evidence"]` (every key in `expected` must be present and
      equal in the finding's evidence — extra evidence keys are ignored).
    - `key` prefixed `min_`/`max_`/`gt_`/`lt_`/`gte_`/`lte_`: comparator
      match against the named field (top-level first, evidence second).
    - otherwise: exact match against the named top-level field, EXCEPT
      when `expected` is a list/tuple/set, in which case it's a
      containment check (`finding[key] in expected`) — this is what lets
      a caller write `level=["November", "December"]` when either month
      recovering the plant is an acceptable match.
    """
    if key == "evidence" and isinstance(expected, dict):
        ev = finding.get("evidence") or {}
        return all(ev.get(k) == v for k, v in expected.items())

    prefix, sep, field = key.partition("_")
    if sep and prefix in _COMPARATORS and field:
        actual = finding.get(field)
        if actual is None:
            actual = (finding.get("evidence") or {}).get(field)
        if actual is None:
            return False
        try:
            return bool(_COMPARATORS[prefix](actual, expected))
        except TypeError:
            return False

    if key in finding:
        actual = finding.get(key)
    else:
        actual = (finding.get("evidence") or {}).get(key)
        if key not in (finding.get("evidence") or {}):
            return False

    if isinstance(expected, (list, tuple, set)):
        return actual in expected
    return actual == expected


def _find_matches(
    findings: list[dict[str, Any]], kind: str, criteria: dict[str, Any]
) -> list[tuple[int, dict[str, Any]]]:
    out = []
    for idx, f in enumerate(findings):
        if f.get("kind") != kind:
            continue
        if all(_criterion_matches(f, k, v) for k, v in criteria.items()):
            out.append((idx, f))
    return out


def assert_finding(
    findings: list[dict[str, Any]],
    kind: str,
    *,
    min_rank: int | None = None,
    effect: float | None = None,
    effect_tol: float = 0.3,
    **criteria: Any,
) -> dict[str, Any]:
    """Assert `findings` (a list of `Finding.to_dict()`-shaped dicts,
    assumed pre-sorted by importance descending, e.g. `result["findings"]`)
    contains one matching `kind` and every criterion in `criteria`
    (dimension=..., level=..., measure=..., etc. — see `_criterion_matches`
    for the matching rules, including nested `evidence={...}` checks).

    `effect`, if given, is checked against the matched finding's `effect`
    field within `effect_tol` *relative* error (planted magnitudes are
    never exactly recovered — the plant is real data with noise on top).

    `min_rank`, if given, asserts the match's position in `findings` is at
    index <= `min_rank` (i.e. it ranks at or above that position — a way
    to assert "this isn't just present, it's actually important").

    Returns the matched finding dict (the first/highest-ranked match) so
    callers can inspect further fields the generic criteria matcher
    doesn't cover. Raises AssertionError with a diff-friendly message
    (what was expected vs. what kind=`kind` findings actually looked like)
    on failure.
    """
    matches = _find_matches(findings, kind, criteria)

    if effect is not None:
        surviving = []
        for idx, f in matches:
            fe = f.get("effect")
            if fe is None:
                continue
            if abs(effect) < 1e-9:
                ok = abs(fe - effect) < 1e-6
            else:
                ok = abs(fe - effect) / abs(effect) <= effect_tol
            if ok:
                surviving.append((idx, f))
        if not surviving and matches:
            observed = [m[1].get("effect") for m in matches]
            raise AssertionError(
                f"assert_finding(kind={kind!r}, {criteria!r}): found "
                f"{len(matches)} matching finding(s) but none had effect "
                f"within {effect_tol:.0%} of {effect!r} — observed "
                f"effect(s): {observed!r}"
            )
        matches = surviving

    if not matches:
        candidates = [f for f in findings if f.get("kind") == kind]
        if candidates:
            summary = [
                {k: c.get(k) for k in ("dimension", "level", "measure", "effect")}
                for c in candidates[:10]
            ]
            raise AssertionError(
                f"assert_finding(kind={kind!r}, criteria={criteria!r}, "
                f"effect={effect!r}): no match. {len(candidates)} finding(s) "
                f"of kind {kind!r} were present but none matched:\n{summary!r}"
            )
        all_kinds = sorted({f.get("kind") for f in findings})
        raise AssertionError(
            f"assert_finding(kind={kind!r}, criteria={criteria!r}, "
            f"effect={effect!r}): no finding of kind {kind!r} at all. "
            f"Kinds present in this run: {all_kinds!r} "
            f"({len(findings)} finding(s) total)."
        )

    idx, f = matches[0]
    if min_rank is not None and idx > min_rank:
        raise AssertionError(
            f"assert_finding(kind={kind!r}, criteria={criteria!r}): matched "
            f"finding {f.get('finding_id')!r} but at rank {idx}, expected "
            f"within the top {min_rank + 1} (min_rank={min_rank})."
        )
    return f


def assert_no_finding(
    findings: list[dict[str, Any]], kind: str, **criteria: Any
) -> None:
    """Inverse of assert_finding: fails if any finding of `kind` matches
    every criterion in `criteria`. Supports the same comparator-prefixed
    keys, e.g. `assert_no_finding(findings, kind="outlier_scan",
    min_pct=30)` asserts there is no outlier_scan finding whose evidence
    (or top-level field) `pct` is >= 30.
    """
    matches = _find_matches(findings, kind, criteria)
    if matches:
        offenders = [
            {k: f.get(k) for k in ("dimension", "level", "measure", "effect")}
            for _, f in matches
        ]
        raise AssertionError(
            f"assert_no_finding(kind={kind!r}, criteria={criteria!r}): "
            f"expected no match, found {len(matches)}: {offenders!r}"
        )


# ---------------------------------------------------------------------------
# Scoring summary
# ---------------------------------------------------------------------------

#: Finding kinds that are structurally "housekeeping" rather than a
#: discovery — mirrors src/core/findings.py `is_trivial`'s method_fit/
#: coverage_gap carve-out, plus a magnitude floor for anything with an
#: `effect` field near zero.
_NOISE_KINDS = frozenset({"method_fit", "coverage_gap"})
_NOISE_EFFECT_FLOOR = 0.02


def score_recovery(
    fixture_name: str, plants: dict[str, Any], findings: list[dict[str, Any]]
) -> dict[str, Any]:
    """Score how much of `plants["checks"]` was actually recovered in
    `findings`, plus a noise count (method_fit/coverage_gap findings, or
    any finding whose effect is trivially close to zero) — the "print a
    table" deliverable metric IMPROVEMENTS.md 7.11 asks for, independent
    of pass/fail on any individual assertion.
    """
    checks = plants.get("checks", [])
    recovered = 0
    for chk in checks:
        try:
            assert_finding(
                findings,
                chk["kind"],
                effect=chk.get("effect"),
                effect_tol=chk.get("effect_tol", 0.3),
                min_rank=chk.get("min_rank"),
                **chk.get("criteria", {}),
            )
            recovered += 1
        except AssertionError:
            pass

    noise = sum(
        1
        for f in findings
        if f.get("kind") in _NOISE_KINDS
        or (f.get("effect") is not None and abs(f["effect"]) < _NOISE_EFFECT_FLOOR)
    )

    return {
        "fixture": fixture_name,
        "plants_total": len(checks),
        "plants_recovered": recovered,
        "recovery_rate": (recovered / len(checks)) if checks else None,
        "noise_findings": noise,
        "total_findings": len(findings),
    }


_RESULTS: list[dict[str, Any]] = []


@pytest.fixture(scope="session", autouse=True)
def _print_recovery_summary_at_session_end():
    """Collects every test's score_recovery() result and prints a plain
    table once the session ends — the actual deliverable metric, not just
    pass/fail per test."""
    yield
    if not _RESULTS:
        return
    print("\n" + "=" * 92)
    print("Planted-effect recovery summary (IMPROVEMENTS.md 7.11)")
    print("=" * 92)
    header = (
        f"{'fixture':42} {'recovered/total':>16} {'rate':>7} "
        f"{'noise':>7} {'findings':>10}"
    )
    print(header)
    print("-" * len(header))
    for r in _RESULTS:
        rate = f"{r['recovery_rate'] * 100:.0f}%" if r["recovery_rate"] is not None else "n/a"
        frac = f"{r['plants_recovered']}/{r['plants_total']}"
        print(
            f"{r['fixture']:42} {frac:>16} {rate:>7} "
            f"{r['noise_findings']:>7} {r['total_findings']:>10}"
        )
    print("=" * 92)


def _record(score: dict[str, Any]) -> None:
    print(f"\n[score_recovery] {score}")
    _RESULTS.append(score)


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------

def _run_pipeline(csv_path: Path) -> dict[str, Any]:
    """Deterministic, network-free pipeline run: no LLM, no RLM planner —
    matches AgentController's documented `use_llm=False` mode (fully
    deterministic: no network, no narrative synthesis, plan comes from the
    profile) rather than test_controller.py's scripted-LLM pattern, since
    these tests care about which tools fire and what they find, not about
    LLM-driven planning behaviour."""
    agent = AgentController(use_llm=False, enable_rlm=False)
    agent.load_dataset(str(csv_path), interactive=False)
    return agent.analyze()


# ---------------------------------------------------------------------------
# Tests — one per fixture generator
# ---------------------------------------------------------------------------

def test_transactional_recovers_west_premium_q4_lift_and_concentration(
    tmp_path: Path,
) -> None:
    csv_path, plants = generate_transactional(tmp_path / "transactional.csv")
    result = _run_pipeline(csv_path)
    findings = result["findings"]

    assert_finding(
        findings,
        kind="segment_lift",
        dimension="region",
        level="West",
        measure="amount",
        effect=plants["region_premium"]["lift"],
        effect_tol=0.35,
    )
    assert_finding(
        findings,
        kind="change",
        dimension="month",
        level=plants["seasonal_lift"]["levels"],  # ["November", "December"]
        measure="amount",
        effect=plants["seasonal_lift"]["lift"],
        effect_tol=0.4,
    )
    assert_finding(
        findings, kind="concentration", dimension="customer", measure="revenue_share"
    )
    assert_finding(
        findings, kind="cohort", dimension="customer", measure="repeat_purchase_rate"
    )

    _record(score_recovery("transactional", plants, findings))


def test_churn_recovers_dominant_categorical_driver(tmp_path: Path) -> None:
    csv_path, plants = generate_churn(tmp_path / "churn.csv")
    result = _run_pipeline(csv_path)
    findings = result["findings"]

    plant = plants["dominant_categorical_driver"]
    assert_finding(
        findings,
        kind="segment_lift",
        dimension="contract",
        level=plant["level"],
        measure="churn",
        effect=plant["expected_lift"],
        effect_tol=0.5,
    )
    # The weak numeric driver should never out-muscle the dominant one —
    # no driver finding on monthly_charges with a large effect.
    assert_no_finding(findings, kind="driver", dimension="monthly_charges", min_effect=0.3)

    _record(score_recovery("churn", plants, findings))


def test_price_series_recovers_drawdown(tmp_path: Path) -> None:
    csv_path, plants = generate_price_series(tmp_path / "price_series.csv")
    result = _run_pipeline(csv_path)
    findings = result["findings"]

    plant = plants["drawdown"]
    found = assert_finding(findings, kind="financial", measure="max_drawdown")
    observed_pct = found.get("evidence", {}).get("max_drawdown_pct")
    assert observed_pct is not None, (
        f"'financial'/max_drawdown finding is missing evidence.max_drawdown_pct: {found!r}"
    )
    # Sign convention isn't pinned down by this fixture — compare magnitude.
    planted = plant["drawdown_pct"]  # 30.0
    assert 0.5 * planted <= abs(observed_pct) <= 1.5 * planted, (
        f"expected a drawdown near {planted}% (planted peak {plant['peak_date']} -> "
        f"trough {plant['trough_date']}), observed max_drawdown_pct={observed_pct}"
    )

    _record(score_recovery("price_series", plants, findings))


def test_hr_roster_recovers_pay_gap(tmp_path: Path) -> None:
    csv_path, plants = generate_hr_roster(tmp_path / "hr_roster.csv")
    result = _run_pipeline(csv_path)
    findings = result["findings"]

    plant = plants["pay_gap"]
    found = assert_finding(
        findings,
        kind="segment_lift",
        dimension="department",
        measure="median_pay",
        effect=plant["gap_pct"] / 100.0,
        effect_tol=0.5,
    )
    assert plant["higher_level"] in (found.get("level") or "") and plant["lower_level"] in (
        found.get("level") or ""
    ), f"expected level to mention both {plant['higher_level']!r} and {plant['lower_level']!r}, got {found.get('level')!r}"

    _record(score_recovery("hr_roster", plants, findings))


def test_panel_recovers_within_group_trend(tmp_path: Path) -> None:
    csv_path, plants = generate_panel(tmp_path / "panel.csv")
    result = _run_pipeline(csv_path)
    findings = result["findings"]

    trend = assert_finding(findings, kind="trend", measure="sales")
    assert (trend.get("effect") or 0) > 0, (
        f"planted a {plants['within_group_trend']['total_growth_pct']}% upward trend, "
        f"got a non-positive trend effect: {trend.get('effect')!r}"
    )

    entity = plants["entity_lift"]
    assert_finding(
        findings,
        kind="segment_lift",
        dimension="store_id",
        level=entity["level"],
        measure="sales",
        effect=entity["lift"],
        effect_tol=0.4,
    )

    _record(score_recovery("panel", plants, findings))


def test_pure_description_declines_to_model(tmp_path: Path) -> None:
    csv_path, plants = generate_pure_description(tmp_path / "pure_description.csv")

    agent = AgentController(use_llm=False, enable_rlm=False)
    agent.load_dataset(str(csv_path), interactive=False)
    decision = agent.memory.get_context("analysis_decision")

    assert decision is not None, "expected an analysis_decision to be recorded in memory"
    assert decision.get("mode") == plants["expected_mode"], (
        f"expected mode={plants['expected_mode']!r} (T2: decline to model — no "
        f"plantable target exists in this fixture by construction), got "
        f"decision={decision!r}"
    )
    assert decision.get("target") == plants["expected_target"], (
        f"expected no target column to be selected, got {decision.get('target')!r}"
    )

    result = agent.analyze()
    findings = result["findings"]
    _record(score_recovery("pure_description", plants, findings))


def test_text_column_recovers_sentiment_skew(tmp_path: Path) -> None:
    csv_path, plants = generate_text_column(tmp_path / "text_column.csv")
    result = _run_pipeline(csv_path)
    findings = result["findings"]

    assert_finding(
        findings,
        kind="text",
        dimension=plants["text_column"],
        level=plants["signature_keyword"],
        effect=plants["negative_fraction"],
        effect_tol=0.4,
    )

    _record(score_recovery("text_column", plants, findings))
