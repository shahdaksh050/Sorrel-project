"""Unit tests for src/core/html_report.py — the shareable HTML report."""
from __future__ import annotations

import re
from typing import Any

from src.core.design_tokens import palette
from src.core.html_report import build_html_report, retheme_report_html

INSIGHTS: dict[str, Any] = {
    "reasoning": "Support calls drive churn.",
    "insights": ["High call volume predicts churn."],
    "recommendations": ["Add proactive support outreach."],
    "best_model": "random_forest",
    "key_metrics": {"cv_mean": 0.86},
}

CHARTS: list[dict[str, Any]] = [{
    "chart_id": "hist_x",
    "title": "Distribution — x",
    "description": "Histogram of x.",
    "spec": {"data": {"values": [{"x": 1}]}, "mark": "bar", "encoding": {}},
}]

TOOL_RESULTS: list[dict[str, Any]] = [
    {"tool_name": "train_model", "status": "success",
     "output": {"summary": "trained", "treatments_applied": ["Applied log1p to 'amount'."]}},
    {"tool_name": "evaluate_model", "status": "success",
     "output": {"summary": "evaluated",
                "driver_narrative": ["#1 driver: 'support_calls' — higher values push toward '1'."]}},
]


class TestBuildHtmlReport:
    def test_contains_all_sections(self) -> None:
        doc = build_html_report(
            "churn", INSIGHTS, TOOL_RESULTS, CHARTS,
            objective="what drives churn?",
            profile={"quality_score": 91, "row_count": 300, "column_count": 8},
        )
        assert doc.startswith("<!DOCTYPE html>")
        assert "what drives churn?" in doc
        assert "Support calls drive churn." in doc
        assert "High call volume predicts churn." in doc
        assert "Add proactive support outreach." in doc
        assert "support_calls" in doc          # drivers section
        assert "log1p" in doc                  # treatments section
        assert "quality 91/100" in doc
        assert "vegaEmbed" in doc and "Distribution — x" in doc

    def test_html_escaping_of_malicious_content(self) -> None:
        evil = {"reasoning": "<script>alert('xss')</script>", "insights": [], "recommendations": []}
        doc = build_html_report("ds", evil, [], [])
        assert "<script>alert" not in doc
        assert "&lt;script&gt;" in doc

    def test_script_close_tag_in_chart_data_neutralised(self) -> None:
        charts = [{
            "chart_id": "c", "title": "t", "description": "d",
            "spec": {"data": {"values": [{"v": "</script><script>alert(1)</script>"}]},
                     "mark": "bar"},
        }]
        doc = build_html_report("ds", {}, [], charts)
        # The raw close tag must never appear inside the embedded JSON
        assert "</script><script>alert(1)" not in doc

    def test_minimal_inputs_produce_valid_shell(self) -> None:
        doc = build_html_report("empty", {}, [], [])
        assert "<h1>What we found in empty</h1>" in doc
        assert "The charts" not in doc  # no charts, no section

    def test_new_params_default_to_none_without_error(self) -> None:
        """Back-compat: the original call signature must keep working."""
        doc = build_html_report("ds", {}, [], [])
        assert doc.startswith("<!DOCTYPE html>")
        assert "Why these analyses" not in doc
        assert "Limitations" not in doc

    def test_methodology_and_limitations_sections_render(self) -> None:
        doc = build_html_report(
            "ds", {}, [], [],
            profile={"quality_score": 40, "row_count": 5, "column_count": 2,
                     "is_sufficient": False, "sufficiency_reason": "Only 5 rows.",
                     "warnings": ["Only 5 rows — results will have high variance."]},
            read_report={"format": "csv", "encoding": "cp1252", "encoding_confident": False,
                         "notes": ["Encoding could not be confidently detected; assumed cp1252."]},
            coercions=[{"column": "amount", "rule": "currency", "n_converted": 9, "n_failed": 1}],
            plan_rationales=[{"step_number": 1, "tool_name": "clean_data", "rationale": "High missingness."}],
            statistical_test_pvalues=[
                {"feature_column": "a", "test_name": "Independent T-Test", "p_value": 0.001},
                {"feature_column": "b", "test_name": "Independent T-Test", "p_value": 0.6},
            ],
            unverified_claims=["'up 40%' [unverified: 40%]"],
            profile_status="failed: could not parse",
        )
        assert "Why these analyses" in doc
        assert "clean_data" in doc
        assert "Limitations &amp; caveats" in doc
        assert "Only 5 rows" in doc
        assert "degraded mode" in doc
        assert "could not parse" in doc
        assert "amount" in doc
        assert "40%" in doc
        assert "Benjamini-Hochberg" in doc


class TestSorrelReport:
    def test_header_title_and_footer_say_sorrel(self) -> None:
        doc = build_html_report("ds", {}, [], [])
        assert "<title>Sorrel analysis report — ds</title>" in doc
        assert '<span class="mark">Sorrel</span>' in doc
        assert "by Sorrel, your data assistant" in doc
        assert "Sorrel, the working name of DSA Agent" in doc
        assert "Ledger" not in doc and "Agentic Data Analysis" not in doc

    def test_report_css_uses_sorrel_tokens_and_families(self) -> None:
        doc = build_html_report("ds", {}, [], [])
        assert "--pen: #1f4634;" in doc and "--pen: #326d48;" in doc  # Day and Night from design_tokens
        css = doc[doc.rindex("<style>") :]
        assert "'Geist'" in css and "'Newsreader'" in css and "'Geist Mono'" in css
        for old in ("Baloo", "Mukta", "Bricolage", "Public Sans"):
            assert old not in css
        assert "box-shadow" not in css and "gradient" not in css  # a printed sheet: no lift, no glow

    def test_text_uses_the_aa_safe_tokens_not_raw_pen_or_risk(self) -> None:
        css = build_html_report("ds", {}, [], [])
        css = css[css.rindex("<style>") :]
        assert ".warn { border-left-color: var(--risk); color: var(--danger-text); }" in css
        assert ".check.risk { color: var(--danger-text); }" in css
        assert "color: var(--accent-text)" in css
        # a bare `color:` (not border-left-color etc.) never takes the raw fill tokens
        assert not re.search(r"(?<![-\w])color:\s*var\(--(risk|pen)\)", css)


class TestRethemeReportHtml:
    """The embedded preview follows the app's Day/Night toggle; the saved file keeps following the browser."""

    def _doc(self) -> str:
        return build_html_report("churn", INSIGHTS, TOOL_RESULTS, CHARTS, objective="what drives churn?")

    def test_the_saved_report_follows_the_browser_setting(self) -> None:
        doc = self._doc()
        assert "prefers-color-scheme: dark" in doc and "color-scheme: light dark" in doc
        assert 'id="forced-theme"' not in doc

    def test_night_and_day_force_their_own_tokens_and_chart_theme(self) -> None:
        doc = self._doc()
        for theme, mode, literal in (("night", "night", "true"), ("day", "day", "false")):
            out = retheme_report_html(doc, theme)
            assert out.count('id="forced-theme"') == 1
            at = out.index('id="forced-theme"')
            forced = out[at : out.index("</style>", at)]
            body = out[out.rindex("</head><body>") :]  # not the first "</head>": Vega's source contains one
            assert f"--stock: {palette(mode)['stock']};" in forced  # type: ignore[index]
            assert f"color-scheme: {'dark' if mode == 'night' else 'light'};" in forced
            assert f"const _dark = {literal};" in body and "matchMedia('(prefers-color-scheme: dark)')" not in body

    def test_the_forced_tokens_come_after_the_report_rules_so_they_win(self) -> None:
        out = retheme_report_html(self._doc(), "night")
        media = re.search(r"@media\s*\(prefers-color-scheme:\s*dark\)", out)  # the CSS is minified
        assert media is not None
        forced_at = out.rindex('id="forced-theme"')
        assert media.start() < forced_at
        # It sits in the report's own head, right before the body, never inside an embedded script
        # (the Vega bundles contain the text "</head>").
        assert out[forced_at:].split("</style>", 1)[1].startswith("</head><body>")

    def test_the_original_is_not_changed_and_the_dark_spelling_is_accepted(self) -> None:
        doc = self._doc()
        before = doc
        assert retheme_report_html(doc, "dark") == retheme_report_html(doc, "night")
        assert doc == before

    def test_a_document_without_a_head_is_returned_with_only_the_chart_choice_fixed(self) -> None:
        out = retheme_report_html("<p>x</p>", "night")
        assert out == "<p>x</p>"
