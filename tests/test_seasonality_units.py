"""
Unit tests for non-explosive seasonality units, tool precondition guards,
recommendations deduplication, and agenda question answering accounting.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.core.agenda import Question, coverage_report
from src.core.controller_report import ReportMixin
from src.core.findings import Finding
from src.core.profiler import profile_dataframe
from src.tools.multi_seasonal_analysis import MultiSeasonalAnalysisTool
from src.tools.threshold_analysis import ThresholdAnalysisTool
from src.tools.time_series import TimeSeriesAnalysisTool


def test_time_series_seasonality_avoids_explosive_percentages_for_near_zero_mean(tmp_path: Path) -> None:
    # A series with near-zero mean (e.g. daily log returns or mean-centered anomalies)
    dates = pd.date_range("2023-01-01", periods=180, freq="D")
    # Base mean is ~0.001, with a slight seasonal shift in certain months
    values = [0.001 + (0.05 if d.month == 4 else -0.02 if d.month == 2 else 0.0) for d in dates]
    df = pd.DataFrame({"date": dates, "log_return": values})

    csv_path = str(tmp_path / "ts_near_zero.csv")
    df.to_csv(csv_path, index=False)

    tool = TimeSeriesAnalysisTool()
    res = tool.run(file_path=csv_path, time_column="date", measure="log_return")
    assert res.status == "success"

    summary = res.output.get("summary", "")
    assert "Calendar seasonality:" in summary
    assert "units vs" in summary
    # Explosive percentage like -5000% or +3000% must NOT be present
    assert "%" not in summary.split("Calendar seasonality:")[1].split("vs. the yearly average")[0]


def test_multi_seasonal_analysis_rejects_demographic_attributes(tmp_path: Path) -> None:
    # Tabular dataset with demographic attributes and no valid temporal axis
    df = pd.DataFrame({
        "Customer_ID": range(1, 101),
        "Age": [25, 30, 45, 50, 60] * 20,
        "Spending_Score": [50, 60, 40, 70, 80] * 20,
    })
    csv_path = str(tmp_path / "demographics.csv")
    df.to_csv(csv_path, index=False)

    tool = MultiSeasonalAnalysisTool()
    # Explicitly attempting to run FFT on demographic 'Age' must return an error
    res = tool.run(file_path=csv_path, time_column="", value_column="Age")
    assert res.status == "error"
    assert "temporal" in (res.error_message or "").lower() or "datetime" in (res.error_message or "").lower()

    # Precondition gate: profile check returns 0.0 applicability
    p = profile_dataframe(df)
    assert tool.applies_to(p, None) == 0.0


def test_threshold_analysis_reports_no_relationship_when_r2_negligible(tmp_path: Path) -> None:
    # Independent, unlinked random variables with negligible R² (< 0.02)
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        "Age": rng.uniform(18, 70, size=200),
        "Spend": rng.uniform(10, 500, size=200),
    })
    csv_path = str(tmp_path / "random_pairs.csv")
    df.to_csv(csv_path, index=False)

    tool = ThresholdAnalysisTool()
    res = tool.run(file_path=csv_path, feature_column="Age", target_column="Spend")
    assert res.status == "success"

    out = res.output
    # When no breakpoint is found and R² is negligible, it must report no meaningful relationship
    assert out.get("has_breakpoint") is False
    assert "no meaningful relationship detected" in out.get("summary", "").lower()


def test_threshold_analysis_reports_continuous_linear_when_strong_linear(tmp_path: Path) -> None:
    # Strongly linked linear variables (R² > 0.95) with no breakpoint
    rng = np.random.default_rng(42)
    x = np.linspace(10, 100, 150)
    y = 2.5 * x + rng.normal(0, 1.0, 150)
    df = pd.DataFrame({"x": x, "y": y})
    csv_path = str(tmp_path / "linear_pairs.csv")
    df.to_csv(csv_path, index=False)

    tool = ThresholdAnalysisTool()
    res = tool.run(file_path=csv_path, feature_column="x", target_column="y")
    assert res.status == "success"

    out = res.output
    assert out.get("has_breakpoint") is False
    assert out.get("linear_r2", 0) > 0.95
    assert "relationship is continuous linear" in out.get("summary", "").lower()
    assert "no meaningful relationship detected" not in out.get("summary", "").lower()


def test_multi_seasonal_analysis_rejects_single_numeric_column_without_temporal_index(tmp_path: Path) -> None:
    # Non-temporal dataset with a single numeric column (e.g. Age) and text categories
    df = pd.DataFrame({
        "Age": [25, 30, 45, 50, 60] * 20,
        "Category": ["A", "B", "C", "D", "E"] * 20,
    })
    csv_path = str(tmp_path / "survey_no_time.csv")
    df.to_csv(csv_path, index=False)

    tool = MultiSeasonalAnalysisTool()
    res = tool.run(file_path=csv_path, time_column="", value_column="Age")
    assert res.status == "error"
    assert "temporal" in (res.error_message or "").lower() or "datetime" in (res.error_message or "").lower()


def test_agenda_coverage_credits_executed_tools() -> None:
    agenda = [
        Question(
            text="Which segments generate the highest average revenue?",
            kind="segment",
            columns=["revenue", "tier"],
            suggested_tool="segment_comparison",
        ),
        Question(
            text="Are there temporal trend patterns in sales?",
            kind="trend",
            columns=["sales", "date"],
            suggested_tool="time_series_analysis",
        ),
    ]

    tool_results = [
        {
            "tool_name": "segment_comparison",
            "status": "success",
            "params": {"target_column": "revenue", "segment_column": "tier"},
            "output": {"summary": "Tier A spends significantly more than Tier B."},
        },
        {
            "tool_name": "time_series_analysis",
            "status": "success",
            "params": {"time_column": "date", "measure": "sales"},
            "output": {"summary": "Sales exhibit steady upward trend."},
        },
    ]

    report = coverage_report(agenda, findings=[], tool_results=tool_results)
    assert report["answered"] == 2
    assert report["total"] == 2
    assert len(report["unanswered"]) == 0


def test_agenda_coverage_does_not_falsely_credit_unexecuted_dimensions() -> None:
    agenda = [
        Question(
            text="How does Total Amount differ by Product Category?",
            kind="segment",
            columns=["Total Amount", "Product Category"],
            suggested_tool="segment_comparison",
        ),
        Question(
            text="How does Total Amount differ by Gender?",
            kind="segment",
            columns=["Total Amount", "Gender"],
            suggested_tool="segment_comparison",
        ),
    ]

    # Only Product Category was executed
    tool_results = [
        {
            "tool_name": "segment_comparison",
            "status": "success",
            "params": {"target_column": "Total Amount", "segment_column": "Product Category"},
            "output": {"summary": "Electronics exceeds Clothing."},
        },
    ]

    report = coverage_report(agenda, findings=[], tool_results=tool_results)
    # Exactly 1 question answered (Product Category), Gender must remain unanswered!
    assert report["answered"] == 1
    assert report["total"] == 2
    assert len(report["unanswered"]) == 1
    assert report["unanswered"][0]["columns"] == ["Total Amount", "Gender"]


def test_report_mixin_deduplicates_recommendations() -> None:
    class MockMemory:
        def __init__(self, findings: list[Finding]) -> None:
            self._findings = findings

        def ranked_findings(self) -> list[Finding]:
            return self._findings

    class DummyController(ReportMixin):
        def __init__(self, memory: MockMemory) -> None:
            self.memory = memory  # type: ignore[assignment]

        def _primary_findings(self) -> list[Finding]:
            return self.memory.ranked_findings()

    findings = [
        # Two findings on the same outcome column
        Finding(
            finding_id="f1",
            source_tool="statistical_analysis",
            headline="Total Amount differs by Region",
            dimension=None,
            measure="Total Amount",
            kind="segment_lift",
            effect=0.5,
            p_value=0.01,
        ),
        Finding(
            finding_id="f2",
            source_tool="detect_outliers",
            headline="Total Amount contains 12 outliers",
            dimension=None,
            measure="Total Amount",
            kind="distribution_anomaly",
            effect=0.05,
        ),
    ]

    controller = DummyController(MockMemory(findings))
    final_result: dict[str, object] = {}
    controller._rebuild_primary_synthesis(final_result)

    recs = final_result.get("recommendations", [])
    assert isinstance(recs, list)
    # Ensure no duplicate recommendations for Total Amount
    total_amount_recs = [r for r in recs if "Total Amount" in str(r)]
    assert len(total_amount_recs) == 1, f"Expected 1 deduplicated outcome recommendation, got: {total_amount_recs}"
    assert "Track 'Total Amount' as a key outcome metric" in str(total_amount_recs[0])
