"""Plain-language rendering: jargon in, readable sentences out, nothing mutated."""
from __future__ import annotations

import copy

import pytest

from src.core.findings import Finding
from src.core.plain_language import describe_uncertainty, effect_words, fallback_caption, plainify
from src.tools.report_generator import _format_evidence, _format_top_findings


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Churn differs by region (p=0.003).", "Churn differs by region (a result unlikely to be down to chance)."),
        ("Sales differ (p < 0.001).", "Sales differ (a result very unlikely to be down to chance)."),
        ("No link (p = 0.4).", "No link (a result that could easily be down to chance)."),
        ("Gap is real (p<0.01**).", "Gap is real (a result unlikely to be down to chance)."),
        ("**p<0.01** here", "**a result unlikely to be down to chance** here"),
        ("Effect (Cohen's d = 0.85).", "Effect (a large effect)."),
        ("Groups differ, d=0.3.", "Groups differ, a small effect."),
        ("Price and demand move together (r=-0.62).", "Price and demand move together (a strong negative relationship)."),
        ("The model has R2=0.41.", "The model has a fit that explains about 41% of the variation."),
        ("Region matters (eta2=0.09).", "Region matters (a moderate effect that accounts for about 9% of the variation)."),
        ("The gap is 2.3 (95% CI [1.2, 3.4]).", "The gap is 2.3 (likely between 1.2 and 3.4)."),
        ("Smokers have OR=2.34 for the outcome.", "Smokers have 2.3 times the odds for the outcome."),
        ("Odds ratio of 0.6 here.", "40% lower odds here."),
        ("Sample (n=1,204).", "Sample (1,204 records)."),
        ("Revenue was $12,345,678 over 1,500,000 rows.", "Revenue was $12.3 million over 1.5 million rows."),
        ("It is statistically significant.", "It is unlikely to be down to chance."),
        ("A statistically significant difference.", "A clear difference."),
        ("Not statistically significant.", "Not clearly different."),
        ("Result (t=3.2, p<0.01).", "Result (a result unlikely to be down to chance)."),
    ],
)
def test_plainify_rewrites_notation(raw: str, expected: str) -> None:
    assert plainify(raw) == expected


def test_plainify_leaves_plain_numbers_and_prose_alone() -> None:
    text = "Sales rose 12.5% in March (from 1,204 to 1,354). Tier d: 5 items, step=3."
    assert plainify(text) == text
    assert plainify("") == ""


def test_effect_words() -> None:
    assert effect_words("cohens_d", 0.9) == "a large effect"
    assert effect_words("r2", 0.41) == "explains about 41% of the variation"
    assert effect_words("lift", 0.5) == "about 50% higher"
    assert effect_words("lift", -0.2) == "about 20% lower"
    assert effect_words(None, 0.3) == "an effect of 0.3"
    assert effect_words("cohens_d", None) == ""


def test_describe_uncertainty_reads_p_n_ci_and_caveats() -> None:
    finding = {"p_value": 0.0004, "evidence": {"n": 12, "ci_lower": 1.234, "ci_upper": 3.456}, "caveats": ["Sample is small."]}
    text = describe_uncertainty(finding)
    assert text is not None
    assert text.startswith("This is very unlikely to be down to chance.")
    assert "likely between 1.23 and 3.46" in text
    assert describe_uncertainty({}) is None
    weak = describe_uncertainty({"p_value": 0.3, "p_adjusted": 0.5})
    assert weak is not None and "hint" in weak


def test_fallback_caption_is_one_short_plain_sentence() -> None:
    caption = fallback_caption({"headline": "Churn is higher in the West (18.2% vs 11.1%), p=0.003. More text."})
    assert caption.startswith("Churn is higher in the West (18.2% vs 11.1%)")
    assert "p=" not in caption and caption.count(". ") == 0
    long = fallback_caption({"headline": "word " * 80})
    assert len(long) <= 160 and long.endswith("…")
    assert fallback_caption({"headline": "Churn differs by region", "effect": 0.9, "effect_kind": "cohens_d"}) == (
        "Churn differs by region; a large effect."
    )
    assert fallback_caption({}) == ""


def test_helpers_do_not_mutate_findings_and_render_keeps_bus_intact() -> None:
    finding = Finding(
        finding_id="f1", kind="test", headline="Churn differs by region (p=0.003)", detail="d=0.8 across n=500",
        evidence={"p": 0.003, "n": 500}, effect=0.8, effect_kind="cohens_d", p_value=0.003, layer="analyst",
    )
    as_dict = finding.to_dict()
    before = copy.deepcopy(as_dict)
    describe_uncertainty(as_dict)
    fallback_caption(as_dict)
    top = "\n".join(_format_top_findings([as_dict]))
    evidence = "\n".join(_format_evidence([as_dict]))
    assert as_dict == before
    assert finding.headline == "Churn differs by region (p=0.003)"
    assert "p=0.003)" not in top and "unlikely to be down to chance" in top
    assert "a large effect" in evidence and "p=0.003" in evidence
