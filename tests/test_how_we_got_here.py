"""Tests for ui/components/how_we_got_here.py: the pure Details audit-trail builder."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from ui.components.how_we_got_here import build_how_html

from src.core.run_view import (
    Decision,
    Deliverables,
    HowWeGotHere,
    Hypothesis,
    Usage,
    build_run_view,
)

HOSTILE = '<script>alert("x")</script><img src=x onerror=alert(1)>'
ROOT = Path(__file__).resolve().parents[1]


def _full() -> HowWeGotHere:
    return HowWeGotHere(
        decision=Decision("describe", "No prediction asked for.", ("regression on 'amount'",)),
        fallbacks=("Training skipped.",),
        fallbacks_total=3,
        untraced_numbers=("42 percent",),
        untraced_total=1,
        deliverables=Deliverables(("a chart",), ("a forecast",), ("a table",)),
        hypotheses=(
            Hypothesis("Region drives churn", "refuted", 0.2),
            Hypothesis("Tenure drives churn", "supported", 0.8),
        ),
        hypothesis_counts=(("refuted", 1), ("supported", 1)),
        usage=Usage(calls=7, tokens=12345, cost_usd=0.0421, is_estimate=True),
    )


def test_full_view_renders_every_section_in_order() -> None:
    out = build_how_html(_full())
    titles = re.findall(r'<h4 class="how-h">([^<]+)</h4>', out)
    assert titles == [
        "What we decided",
        "What changed along the way",
        "Numbers we could not trace",
        "What you asked for",
        "Ideas we tested",
        "What it cost",
    ]
    assert "Options we turned down" in out
    assert "a forecast" in out
    assert "1 refuted, 1 supported." in out
    assert "and 2 more." in out  # fallbacks_total 3, one shown
    assert "12,345 tokens" in out
    assert "estimated cost about" in out


def test_hostile_text_is_escaped_everywhere() -> None:
    how = HowWeGotHere(
        decision=Decision(HOSTILE, HOSTILE, (HOSTILE,)),
        fallbacks=(HOSTILE,),
        fallbacks_total=1,
        untraced_numbers=(HOSTILE,),
        untraced_total=1,
        deliverables=Deliverables((HOSTILE,), (HOSTILE,), (HOSTILE,)),
        hypotheses=(Hypothesis(HOSTILE, HOSTILE, 0.5),),
        hypothesis_counts=(("refuted", 1),),
    )
    out = build_how_html(how)
    assert "<script" not in out
    assert "<img" not in out
    assert "onerror=alert(1)>" not in out
    assert "&lt;script&gt;" in out


def test_empty_view_returns_empty_string() -> None:
    assert build_how_html(HowWeGotHere()) == ""


def test_empty_sections_are_omitted_not_placeholdered() -> None:
    how = HowWeGotHere(fallbacks=("Only this.",), fallbacks_total=1)
    out = build_how_html(how)
    assert "What changed along the way" in out
    for absent in ("What we decided", "Numbers we could not trace", "What you asked for",
                   "Ideas we tested", "What it cost"):
        assert absent not in out
    assert "None" not in out and "N/A" not in out


def test_no_ai_run_says_so_and_shows_no_cost() -> None:
    out = build_how_html(HowWeGotHere(decision=Decision("model", "Auto-detected target.", ()), no_ai=True))
    assert "No AI was used in this run." in out
    assert "tokens" not in out
    assert "cost" not in out.lower().replace("what it cost", "")


def test_missing_usage_without_no_ai_flag_omits_the_cost_block() -> None:
    out = build_how_html(HowWeGotHere(decision=Decision("model", "r", ())))
    assert "What it cost" not in out


def test_exact_cost_is_not_labelled_an_estimate() -> None:
    how = HowWeGotHere(usage=Usage(calls=1, tokens=10, cost_usd=0.5, is_estimate=False))
    out = build_how_html(how)
    assert "1 AI call," in out
    assert "estimate" not in out


def test_dollar_sign_is_an_entity_so_markdown_cannot_read_math() -> None:
    out = build_how_html(_full())
    assert "$" not in out
    assert "&#36;0.0421" in out


def test_output_has_no_inline_style() -> None:
    assert "style=" not in build_how_html(_full())
    assert "style=" not in build_how_html(HowWeGotHere(no_ai=True))


def test_markup_has_no_blank_lines_that_would_break_markdown_html() -> None:
    assert "\n" not in build_how_html(_full())


def test_every_class_the_builder_uses_is_defined_in_styles() -> None:
    styles = (ROOT / "ui" / "styles.py").read_text(encoding="utf-8")
    classes = set(re.findall(r'class="([^"]+)"', build_how_html(_full())))
    names = {c for cls in classes for c in cls.split()}
    # The status modifiers appear only as compound selectors (".how-tag.refuted").
    for name in sorted(names):
        assert f".{name}" in styles, f"missing CSS for .{name}"


def test_how_css_uses_tokens_not_hex_colours() -> None:
    styles = (ROOT / "ui" / "styles.py").read_text(encoding="utf-8")
    start = styles.index('/* ── "How we got here"')
    block = styles[start : styles.index("</style>", start)]
    assert not re.search(r"#[0-9a-fA-F]{3,8}\b", block)


def test_builds_from_a_real_runview() -> None:
    ctx: dict[str, Any] = {"analysis_decision": {"mode": "describe", "rationale": "r"}}
    view = build_run_view({"deterministic_mode": True}, ctx, objective="", is_sample=True)
    out = build_how_html(view.how)
    assert "What we decided" in out
    assert "No AI was used in this run." in out


def test_hypothesis_statements_are_rewritten_in_plain_language_and_still_escaped() -> None:
    how = HowWeGotHere(
        hypotheses=(
            Hypothesis("tenure differs (Mann-Whitney U, rank_biserial=0.625, p_adj=0.0000)", "supported", 0.9),
            Hypothesis(HOSTILE, "refuted", 0.1),
        ),
        hypothesis_counts=(("refuted", 1), ("supported", 1)),
    )
    out = build_how_html(how)
    assert "p_adj" not in out and "rank_biserial" not in out
    assert "correcting for the other tests" in out
    assert "<script" not in out and "<img" not in out and "&lt;script&gt;" in out


def test_a_capped_hypothesis_list_says_how_many_were_tested() -> None:
    shown = tuple(Hypothesis(f"idea {i}", "supported", 0.7) for i in range(8))
    how = HowWeGotHere(hypotheses=shown, hypothesis_counts=(("supported", 33),))
    out = build_how_html(how)
    assert "Showing 8 of 33." in out
    assert "Showing" not in build_how_html(
        HowWeGotHere(hypotheses=shown[:3], hypothesis_counts=(("supported", 3),))
    )
