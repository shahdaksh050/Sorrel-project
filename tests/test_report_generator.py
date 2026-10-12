"""Unit tests for the Report Generator Tool — Stage 5."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.tools.report_generator import GenerateReportTool

LLM_INSIGHTS = {
    "status": "complete",
    "reasoning": "All analyses completed successfully.",
    "insights": ["Feature A predicts the target.", "No overfitting detected."],
    "recommendations": ["Deploy the best model."],
    "best_model": "random_forest",
    "key_metrics": {"cv_mean": "0.87", "train_test_gap": "0.03"},
}

TOOL_RESULTS = [
    {
        "tool_name": "clean_data",
        "status": "success",
        "output": {"summary": "Cleaned 5 missing values."},
        "error": None,
    }
]


@pytest.fixture
def report_dir(tmp_path: pytest.TempPathFactory) -> str:
    return str(tmp_path / "reports")


class TestGenerateReportTool:
    def test_markdown_and_json_files_created(self, report_dir: str) -> None:
        result = GenerateReportTool().run(
            dataset_name="unit_test",
            tool_results_json=json.dumps(TOOL_RESULTS),
            llm_insights=LLM_INSIGHTS,
            output_dir=report_dir,
        )
        assert result.status == "success"
        assert Path(result.output["markdown_path"]).exists()
        assert Path(result.output["json_path"]).exists()

    def test_markdown_contains_expected_sections(self, report_dir: str) -> None:
        result = GenerateReportTool().run(
            dataset_name="unit_test",
            tool_results_json=json.dumps(TOOL_RESULTS),
            llm_insights=LLM_INSIGHTS,
            output_dir=report_dir,
        )
        content = Path(result.output["markdown_path"]).read_text(encoding="utf-8")
        assert "## Key Insights" in content
        assert "## Recommendations" in content
        assert "## Model Performance" in content
        assert "## Tool Execution Log" in content
        assert "clean_data" in content

    def test_tolerates_missing_llm_insights(self, report_dir: str) -> None:
        result = GenerateReportTool().run(
            dataset_name="no_insights",
            tool_results_json="[]",
            llm_insights=None,
            output_dir=report_dir,
        )
        assert result.status == "success"

    def test_tolerates_double_serialised_insights(self, report_dir: str) -> None:
        """The LLM sometimes passes llm_insights as a JSON string."""
        result = GenerateReportTool().run(
            dataset_name="stringy",
            tool_results_json="[]",
            llm_insights=json.dumps(LLM_INSIGHTS),
            output_dir=report_dir,
        )
        assert result.status == "success"
        assert result.output["n_insights"] == 2

    def test_tolerates_malformed_tool_results_json(self, report_dir: str) -> None:
        result = GenerateReportTool().run(
            dataset_name="broken",
            tool_results_json="{not json at all",
            llm_insights=LLM_INSIGHTS,
            output_dir=report_dir,
        )
        assert result.status == "success"

    def test_summary_key_present(self, report_dir: str) -> None:
        result = GenerateReportTool().run(
            dataset_name="x", tool_results_json="[]", llm_insights={}, output_dir=report_dir
        )
        assert "summary" in result.output

    def test_report_surfaces_domain_and_causal_limits(self, report_dir: str) -> None:
        insights = {
            **LLM_INSIGHTS,
            "domain_analysis": {
                "domain": "epidemiology",
                "row_entity": "one participant",
                "readiness": "partial",
                "suggested_methods": ["survival analysis"],
                "missing_information": ["outcome codebook"],
            },
            "causal_design": {
                "status": "not_identified",
                "design": "observational_cross_section",
                "treatment": "exposure",
                "outcome": "recovery",
                "missing_requirements": ["pre-treatment confounders"],
            },
        }
        result = GenerateReportTool().run(
            dataset_name="causal_limits",
            tool_results_json="[]",
            llm_insights=insights,
            output_dir=report_dir,
        )

        content = Path(result.output["markdown_path"]).read_text(encoding="utf-8")
        assert "## Domain Analysis Plan" in content
        assert "## Causal Identification" in content
        assert "not_identified" in content
        assert "pre-treatment confounders" in content

    def test_additional_analyses_section_for_unbespoke_tools(self, report_dir: str) -> None:
        """cluster_data (and text/geo/time-series/dimensionality) must get a
        real subsection, not survive only as a truncated Tool Execution Log
        row — the file-based report has the same gap the Streamlit UI had."""
        results = [
            *TOOL_RESULTS,
            {
                "tool_name": "cluster_data",
                "status": "success",
                "output": {
                    "summary": "Found 3 clusters (silhouette=0.61, strong separation).",
                    "n_clusters": 3,
                    "silhouette_score": 0.61,
                    "separation_quality": "strong",
                },
                "error": None,
            },
        ]
        result = GenerateReportTool().run(
            dataset_name="with_cluster",
            tool_results_json=json.dumps(results),
            llm_insights=LLM_INSIGHTS,
            output_dir=report_dir,
        )
        content = Path(result.output["markdown_path"]).read_text(encoding="utf-8")
        assert "## Additional Analyses" in content
        assert "### Cluster Data" in content
        assert "Clusters found: 3" in content
        assert "Silhouette score: 0.61" in content

    def test_no_additional_analyses_section_when_nothing_else_ran(self, report_dir: str) -> None:
        result = GenerateReportTool().run(
            dataset_name="unit_test",
            tool_results_json=json.dumps(TOOL_RESULTS),
            llm_insights=LLM_INSIGHTS,
            output_dir=report_dir,
        )
        content = Path(result.output["markdown_path"]).read_text(encoding="utf-8")
        assert "## Additional Analyses" not in content

    def test_text_analysis_with_dict_top_tokens_generates_report(self, report_dir: str) -> None:
        results = [
            {
                "tool_name": "text_analysis",
                "status": "success",
                "output": {
                    "summary": "Profiled notes.",
                    "vocab_size": 5,
                    "avg_word_count": 4.2,
                    "top_tokens": {"camera": 12, "battery": 9},
                },
                "error": None,
            },
        ]
        result = GenerateReportTool().run(
            dataset_name="text_tokens",
            tool_results_json=json.dumps(results),
            llm_insights=LLM_INSIGHTS,
            output_dir=report_dir,
        )
        content = Path(result.output["markdown_path"]).read_text(encoding="utf-8")
        assert result.status == "success"
        assert "Most frequent words: camera, battery" in content

    def test_boilerplate_summary_is_replaced_and_hygiene_is_separate(
        self, report_dir: str
    ) -> None:
        findings = [
            {
                "finding_id": "integrity_1",
                "kind": "outlier_scan",
                "headline": "Constraint violation: order_id >= quantity failed once",
                "source_tool": "integrity_audit",
                "confidence": 0.8,
            },
            {
                "finding_id": "lift_1",
                "kind": "segment_lift",
                "headline": "Enterprise revenue is 45% above the remaining segments",
                "source_tool": "segment_comparison",
                "effect": 0.45,
                "confidence": 0.85,
                "dimension": "tier",
                "level": "Enterprise",
            },
        ]
        result = GenerateReportTool().run(
            dataset_name="grounded",
            tool_results_json="[]",
            llm_insights={
                "reasoning": (
                    "User objective: reduce churn. Deterministic run: the LLM was switched off, "
                    "so the plan came from the dataset profile."
                ),
                "insights": [],
            },
            output_dir=report_dir,
            findings=findings,
        )

        content = Path(result.output["markdown_path"]).read_text(encoding="utf-8")
        summary = content.split("## Executive Summary", 1)[1].split("##", 1)[0]
        matrix = content.split("## Action Prioritization Matrix", 1)[1]
        assert "Enterprise revenue is 45%" in summary
        assert "Deterministic run" not in summary
        assert "## Data Integrity & Hygiene" in content
        assert "Constraint violation" not in matrix

    def test_old_four_params_still_succeed_without_new_sections(self, report_dir: str) -> None:
        """Back-compat: scripts/validate.py and other direct callers only
        ever pass the original four params — must keep working, and the
        new sections must simply not appear rather than error."""
        result = GenerateReportTool().run(
            dataset_name="unit_test",
            tool_results_json=json.dumps(TOOL_RESULTS),
            llm_insights=LLM_INSIGHTS,
            output_dir=report_dir,
        )
        assert result.status == "success"
        content = Path(result.output["markdown_path"]).read_text(encoding="utf-8")
        assert "## Data Overview" not in content
        assert "## Methodology" not in content
        assert "## Limitations & Caveats" not in content

    def test_data_overview_section_with_profile_and_read_report(self, report_dir: str) -> None:
        data_profile = {
            "row_count": 500, "column_count": 4, "quality_score": 72,
            "duplicate_rows": 3, "is_sufficient": True,
            "columns": [{"kind": "numeric"}, {"kind": "numeric"}, {"kind": "categorical"}, {"kind": "datetime"}],
            "warnings": ["3 duplicate rows (0.6%)."],
        }
        read_report = {
            "format": "csv", "encoding": "cp1252", "encoding_confident": False,
            "delimiter": ";", "delimiter_sniffed": True,
            "notes": ["Encoding could not be confidently detected; assumed cp1252."],
        }
        coercions = [{"column": "amount", "rule": "currency", "n_converted": 495, "n_failed": 5}]
        result = GenerateReportTool().run(
            dataset_name="rich",
            tool_results_json=json.dumps(TOOL_RESULTS),
            llm_insights=LLM_INSIGHTS,
            output_dir=report_dir,
            data_profile=data_profile,
            read_report=read_report,
            coercions=coercions,
        )
        content = Path(result.output["markdown_path"]).read_text(encoding="utf-8")
        assert "## Data Overview" in content
        assert "500 rows" in content
        assert "encoding `cp1252` (guessed)" in content
        assert "amount" in content and "currency" in content

    def test_methodology_section_names_each_tool_with_rationale(self, report_dir: str) -> None:
        plan_rationales = [
            {"step_number": 1, "tool_name": "clean_data", "rationale": "High missingness flagged by profile."},
            {"step_number": 2, "tool_name": "select_statistical_test", "rationale": "Compare groups on target."},
        ]
        result = GenerateReportTool().run(
            dataset_name="methodology",
            tool_results_json=json.dumps(TOOL_RESULTS),
            llm_insights=LLM_INSIGHTS,
            output_dir=report_dir,
            plan_rationales=plan_rationales,
        )
        content = Path(result.output["markdown_path"]).read_text(encoding="utf-8")
        assert "## Methodology" in content
        assert "clean_data" in content
        assert "High missingness flagged by profile." in content
        assert "select_statistical_test" in content

    def test_limitations_section_contains_warnings_and_bh_correction(self, report_dir: str) -> None:
        data_profile = {"row_count": 10, "column_count": 2, "quality_score": 40,
                         "duplicate_rows": 0, "is_sufficient": False,
                         "sufficiency_reason": "Only 1 row(s) — not enough data.",
                         "columns": [], "warnings": ["Only 10 rows — results will have high variance."]}
        statistical_test_pvalues = [
            {"step_number": 1, "feature_column": "a", "test_name": "Independent T-Test", "p_value": 0.001},
            {"step_number": 2, "feature_column": "b", "test_name": "Independent T-Test", "p_value": 0.04},
            {"step_number": 3, "feature_column": "c", "test_name": "Independent T-Test", "p_value": 0.5},
        ]
        result = GenerateReportTool().run(
            dataset_name="limitations",
            tool_results_json=json.dumps(TOOL_RESULTS),
            llm_insights=LLM_INSIGHTS,
            output_dir=report_dir,
            data_profile=data_profile,
            statistical_test_pvalues=statistical_test_pvalues,
            unverified_claims=["'accuracy improved by 40%' [unverified: 40%]"],
        )
        content = Path(result.output["markdown_path"]).read_text(encoding="utf-8")
        assert "## Limitations & Caveats" in content
        assert "Only 10 rows" in content
        assert "Benjamini-Hochberg" in content
        assert "Unverified claims" in content
        assert "40%" in content

    def test_degraded_profiling_surfaced_in_limitations(self, report_dir: str) -> None:
        result = GenerateReportTool().run(
            dataset_name="degraded",
            tool_results_json=json.dumps(TOOL_RESULTS),
            llm_insights=LLM_INSIGHTS,
            output_dir=report_dir,
            profile_status="failed: could not parse file",
        )
        content = Path(result.output["markdown_path"]).read_text(encoding="utf-8")
        assert "degraded mode" in content
        assert "could not parse file" in content
