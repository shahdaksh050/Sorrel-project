"""Unit tests for Data Processing Tools — Stages 1 & 3."""
from __future__ import annotations

import pandas as pd
import pytest

from src.tools.data_processing import (
    CleanDataTool,
    CorrelationAnalysisTool,
    DetectOutliersTool,
    IngestDatasetTool,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_csv(tmp_path: pytest.TempPathFactory) -> str:
    df = pd.DataFrame(
        {
            "age": [25, 30, 35, None, 28, 45, 200, 22],  # 200 is an outlier
            "income": [50_000.0, 60_000.0, None, 70_000.0, 55_000.0, 80_000.0, 90_000.0, 40_000.0],
            "score": [0.8, 0.6, 0.9, 0.7, 0.5, 0.85, 0.95, 0.4],
            "churn": [0, 0, 1, 1, 0, 0, 1, 1],
        }
    )
    p = tmp_path / "test_data.csv"
    df.to_csv(p, index=False)
    return str(p)


@pytest.fixture
def clean_csv(tmp_path: pytest.TempPathFactory) -> str:
    """A CSV with no missing values for correlation tests."""
    df = pd.DataFrame(
        {
            "a": [1.0, 2.0, 3.0, 4.0, 5.0],
            "b": [2.0, 4.0, 6.0, 8.0, 10.0],
            "c": [5.0, 4.0, 3.0, 2.0, 1.0],
            "target": [0, 1, 0, 1, 0],
        }
    )
    p = tmp_path / "clean.csv"
    df.to_csv(p, index=False)
    return str(p)


# ---------------------------------------------------------------------------
# Stage 1: IngestDatasetTool
# ---------------------------------------------------------------------------

class TestIngestDatasetTool:
    def test_ingests_csv_successfully(self, sample_csv: str) -> None:
        result = IngestDatasetTool().run(file_path=sample_csv)
        assert result.status == "success"
        assert result.output["metadata"]["row_count"] == 8
        assert result.output["metadata"]["column_count"] == 4

    def test_returns_correct_numerical_cols(self, sample_csv: str) -> None:
        result = IngestDatasetTool().run(file_path=sample_csv)
        assert "age" in result.output["metadata"]["numerical_cols"]

    def test_detects_missing_values(self, sample_csv: str) -> None:
        result = IngestDatasetTool().run(file_path=sample_csv)
        missing = result.output["metadata"]["missing_values"]
        assert missing.get("age", 0) == 1
        assert missing.get("income", 0) == 1

    def test_infers_classification_task_type(self, sample_csv: str) -> None:
        result = IngestDatasetTool().run(file_path=sample_csv, target_column="churn")
        assert result.output["metadata"]["task_type"] == "classification"

    def test_file_not_found_returns_error(self) -> None:
        result = IngestDatasetTool().run(file_path="/nonexistent/path.csv")
        assert result.status == "error"
        assert result.error_message is not None

    def test_summary_key_present(self, sample_csv: str) -> None:
        result = IngestDatasetTool().run(file_path=sample_csv)
        assert "summary" in result.output


# ---------------------------------------------------------------------------
# Stage 3: CleanDataTool
# ---------------------------------------------------------------------------

class TestCleanDataTool:
    def test_median_imputation_removes_missing(self, sample_csv: str) -> None:
        result = CleanDataTool().run(file_path=sample_csv, strategy="median")
        assert result.status == "success"
        assert result.output["missing_after"] == 0

    def test_drop_rows_reduces_row_count(self, sample_csv: str) -> None:
        result = CleanDataTool().run(file_path=sample_csv, strategy="drop_rows")
        assert result.status == "success"
        assert result.output["rows_after"] < result.output["rows_before"]

    def test_cleaned_file_path_returned(self, sample_csv: str) -> None:
        result = CleanDataTool().run(file_path=sample_csv, strategy="median")
        assert "cleaned_file_path" in result.output
        import os
        assert os.path.exists(result.output["cleaned_file_path"])

    def test_invalid_strategy_returns_error(self, sample_csv: str) -> None:
        result = CleanDataTool().run(file_path=sample_csv, strategy="invalid")
        assert result.status == "error"

    def test_mean_strategy(self, sample_csv: str) -> None:
        result = CleanDataTool().run(file_path=sample_csv, strategy="mean")
        assert result.status == "success"
        assert result.output["missing_after"] == 0


# ---------------------------------------------------------------------------
# Stage 3: DetectOutliersTool
# ---------------------------------------------------------------------------

class TestDetectOutliersTool:
    def test_iqr_detects_outlier_in_age(self, sample_csv: str) -> None:
        result = DetectOutliersTool().run(file_path=sample_csv, method="iqr")
        assert result.status == "success"
        assert result.output["total_outliers"] >= 1

    def test_zscore_method_runs(self, sample_csv: str) -> None:
        result = DetectOutliersTool().run(file_path=sample_csv, method="zscore", threshold=2.0)
        assert result.status == "success"

    def test_flagged_file_path_returned(self, sample_csv: str) -> None:
        result = DetectOutliersTool().run(file_path=sample_csv, method="iqr")
        assert "flagged_file_path" in result.output

    def test_unknown_method_returns_error(self, sample_csv: str) -> None:
        result = DetectOutliersTool().run(file_path=sample_csv, method="invalid_method")
        assert result.status == "error"

    def test_outlier_percentage_between_0_and_100(self, sample_csv: str) -> None:
        result = DetectOutliersTool().run(file_path=sample_csv, method="iqr")
        pct = result.output["outlier_percentage"]
        assert 0.0 <= pct <= 100.0

    def test_row_count_in_output(self, sample_csv: str) -> None:
        result = DetectOutliersTool().run(file_path=sample_csv, method="iqr")
        assert "row_count" in result.output
        assert result.output["row_count"] == 8

    def test_question_mark_parsed_as_nan_by_ingest(self, tmp_path: pytest.TempPathFactory) -> None:
        p = tmp_path / "sentinel.csv"
        p.write_text("a,b\n1,2.5\n?,3.0\n3,?\n", encoding="utf-8")
        result = IngestDatasetTool().run(file_path=str(p))
        assert result.status == "success"
        meta = result.output["metadata"]
        assert meta["row_count"] == 3
        # Both columns should be numerical because ? parsed as NaN
        assert "a" in meta["numerical_cols"]
        assert "b" in meta["numerical_cols"]
        assert meta["missing_values"]["a"] == 1
        assert meta["missing_values"]["b"] == 1



# ---------------------------------------------------------------------------
# Stage 3: CorrelationAnalysisTool
# ---------------------------------------------------------------------------

class TestCorrelationAnalysisTool:
    def test_returns_top_correlations(self, clean_csv: str) -> None:
        result = CorrelationAnalysisTool().run(file_path=clean_csv)
        assert result.status == "success"
        assert len(result.output["top_correlations"]) > 0

    def test_target_correlations_excludes_self(self, clean_csv: str) -> None:
        result = CorrelationAnalysisTool().run(file_path=clean_csv, target_column="target")
        assert "target" not in result.output["target_correlations"]

    def test_spearman_method(self, clean_csv: str) -> None:
        result = CorrelationAnalysisTool().run(file_path=clean_csv, method="spearman")
        assert result.status == "success"
        assert result.output["method"] == "spearman"

    def test_perfect_correlation_detected(self, clean_csv: str) -> None:
        """Columns a and b are perfectly correlated (b = 2a)."""
        result = CorrelationAnalysisTool().run(file_path=clean_csv)
        top = result.output["top_correlations"][0]
        assert abs(top["correlation"]) > 0.99
