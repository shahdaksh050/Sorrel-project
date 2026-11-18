"""
End-to-end integration and pipeline integrity tests.

Verifies:
1. Cleaned and reshaped feature activation in downstream step validation.
2. Unsupervised autonomous clustering unblocking when no target is present.
3. Causal analysis file_path injection and execution without parameter crash.
4. Identifier heuristic recalibration preserving valid business metrics (e.g. revenue=[10, 20, 30]).
5. Complete diagnostics persistence in raw JSON report.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.core.controller import AgentController
from src.core.memory import AnalysisStep
from src.core.step_validation import validate_step


class _ScriptedLLM:
    """Stands in for LLMClient — returns scripted responses per call."""

    def __init__(self, responses: list[dict[str, Any] | Exception]) -> None:
        self._responses = responses
        self.calls = 0
        self.provider = "openrouter"
        self.model = "test-model"

    def call(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        idx = min(self.calls, len(self._responses) - 1)
        self.calls += 1
        item = self._responses[idx]
        if isinstance(item, Exception):
            raise item
        return item

    def get_context_window(self) -> int:
        return 32_000


@pytest.fixture(autouse=True)
def _disable_chart_design(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CHART_DESIGN", "false")


def test_continuous_series_e2e_reshaped_features_accessible(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify clean_data creates archetype features and downstream tools access them cleanly."""
    out_dir = tmp_path / "output"
    monkeypatch.setenv("OUTPUT_DIR", str(out_dir))

    # Continuous time-series dataset with 60 hourly points
    n = 60
    dates = pd.date_range("2026-01-01", periods=n, freq="h")
    prices = 100.0 + np.cumsum(np.random.default_rng(42).normal(0.1, 1.0, n))
    df = pd.DataFrame({
        "timestamp": dates,
        "asset_price": prices,
        "volume": np.random.default_rng(42).integers(100, 1000, n),
    })
    csv_path = tmp_path / "continuous_series.csv"
    df.to_csv(csv_path, index=False)

    controller = AgentController(max_iterations=2, enable_rlm=False)
    controller.load_dataset(str(csv_path), interactive=False)

    # Initial profile confirms archetype
    assert controller.last_profile is not None

    plan = {
        "status": "in_progress",
        "reasoning": "Clean data to generate log returns, then compute statistics.",
        "steps": [
            {
                "step_number": 1,
                "tool_name": "clean_data",
                "parameters": {"file_path": str(csv_path), "strategy": "none"},
                "rationale": "Clean dataset and create continuous log-returns.",
            },
            {
                "step_number": 2,
                "tool_name": "threshold_analysis",
                "parameters": {"target_column": "asset_price_log_return", "feature_column": "volume"},
                "rationale": "Analyze threshold relationship on generated log returns.",
            },
        ],
    }
    final_response: dict[str, Any] = {
        "status": "complete",
        "reasoning": "Completed successfully.",
        "insights": ["Log returns are approximately normal."],
        "recommendations": ["Monitor volatility."],
        "best_model": None,
        "key_metrics": {},
    }
    controller.llm_client = _ScriptedLLM([plan, final_response])  # type: ignore[assignment]

    result = controller.analyze()
    assert result["status"] in {"succeeded", "succeeded_with_warnings", "partial"}

    # Verify downstream step executed without column validation error
    step_results = controller.memory.tool_results
    assert len(step_results) >= 2
    threshold_result = next((r for r in step_results if r.tool_name == "threshold_analysis"), None)
    assert threshold_result is not None
    assert threshold_result.status == "success"


def test_unsupervised_autonomous_clustering_e2e(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify autonomous mode unblocks cluster_data when no supervised target is present."""
    out_dir = tmp_path / "output"
    monkeypatch.setenv("OUTPUT_DIR", str(out_dir))

    rng = np.random.default_rng(42)
    n = 90
    df = pd.DataFrame({
        "dim_1": rng.normal(0, 1, n),
        "dim_2": rng.normal(5, 2, n),
        "dim_3": rng.uniform(10, 20, n),
    })
    csv_path = tmp_path / "unsupervised_data.csv"
    df.to_csv(csv_path, index=False)

    # Empty objective -> autonomous mode
    controller = AgentController(max_iterations=2, enable_rlm=False, objective="")
    controller.load_dataset(str(csv_path), interactive=False)

    # Check that clustering is allowed for unsupervised data
    assert controller._objective_allows_tool("cluster_data") is True

    plan = {
        "status": "in_progress",
        "reasoning": "Cluster unsupervised feature space.",
        "steps": [
            {
                "step_number": 1,
                "tool_name": "cluster_data",
                "parameters": {"n_clusters": 3},
                "rationale": "Discover natural groupings.",
            },
        ],
    }
    final_response: dict[str, Any] = {
        "status": "complete",
        "reasoning": "Three distinct clusters found.",
        "insights": ["Three feature clusters identified."],
        "recommendations": ["Tailor strategies to each segment."],
        "best_model": "kmeans",
        "key_metrics": {},
    }
    controller.llm_client = _ScriptedLLM([plan, final_response])  # type: ignore[assignment]

    result = controller.analyze()
    assert result["status"] in {"succeeded", "succeeded_with_warnings", "partial"}

    cluster_result = next((r for r in controller.memory.tool_results if r.tool_name == "cluster_data"), None)
    assert cluster_result is not None
    assert cluster_result.status == "success"


def test_causal_analysis_e2e_file_path_injected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify _ensure_causal_analysis_step injects file_path and passes step validation."""
    out_dir = tmp_path / "output"
    monkeypatch.setenv("OUTPUT_DIR", str(out_dir))

    rng = np.random.default_rng(42)
    n = 100
    treatment = rng.integers(0, 2, n)
    covar = rng.normal(10, 2, n)
    outcome = 2.5 * treatment + 0.8 * covar + rng.normal(0, 1, n)

    df = pd.DataFrame({
        "treated": treatment,
        "covariate_x": covar,
        "outcome_y": outcome,
    })
    csv_path = tmp_path / "causal_data.csv"
    df.to_csv(csv_path, index=False)

    controller = AgentController(max_iterations=2, enable_rlm=False, objective="Evaluate causal impact of treated on outcome_y")
    controller.load_dataset(str(csv_path), target_hint="outcome_y", interactive=False)

    # Set up validated causal design
    controller.memory.set_context("causal_design", {
        "status": "identified",
        "study_design": "observational_cross_section",
        "treatment": "treated",
        "outcome": "outcome_y",
        "pre_treatment_confounders": ["covariate_x"],
    })

    # Call _ensure_causal_analysis_step
    initial_steps: list[AnalysisStep] = []
    augmented_steps = controller._ensure_causal_analysis_step(initial_steps)

    assert len(augmented_steps) == 1
    causal_step = augmented_steps[0]
    assert causal_step.tool_name == "causal_analysis"
    assert "file_path" in causal_step.parameters
    assert causal_step.parameters["file_path"] != ""

    # Validate step against schema to ensure no parameter errors
    tool = controller.tool_registry.get("causal_analysis")
    _params, _dropped, errors, _note = validate_step(controller.memory, tool, causal_step.parameters, causal_step.parameters)
    assert not errors, f"Step validation failed: {errors}"


def test_identifier_recalibration_e2e(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify small business tables with constant progression (revenue=[10, 20, 30]) are NOT flagged as IDs."""
    out_dir = tmp_path / "output"
    monkeypatch.setenv("OUTPUT_DIR", str(out_dir))

    df = pd.DataFrame({
        "revenue": [10, 20, 30],
        "sales": [100, 200, 300],
        "category": ["A", "B", "C"],
    })
    csv_path = tmp_path / "small_business.csv"
    df.to_csv(csv_path, index=False)

    controller = AgentController(max_iterations=1, enable_rlm=False)
    meta = controller.load_dataset(str(csv_path), interactive=False)

    profile = controller.last_profile
    assert profile is not None

    rev_col = next((c for c in profile.columns if c.name == "revenue"), None)
    assert rev_col is not None
    assert rev_col.kind == "numeric"
    assert "identifier" not in rev_col.flags

    sales_col = next((c for c in profile.columns if c.name == "sales"), None)
    assert sales_col is not None
    assert sales_col.kind == "numeric"
    assert "identifier" not in sales_col.flags

    assert "revenue" in meta.numerical_cols
    assert "sales" in meta.numerical_cols
    assert controller._numeric_column_count() == 2


def test_report_raw_json_persists_full_diagnostics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify raw.json serializes coverage, domain_analysis, causal_design, governance, and active_domain_pack."""
    out_dir = tmp_path / "output"
    monkeypatch.setenv("OUTPUT_DIR", str(out_dir))

    df = pd.DataFrame({
        "feature_1": [1.0, 2.0, 3.0, 4.0, 5.0],
        "target": [0, 1, 0, 1, 0],
    })
    csv_path = tmp_path / "diag_data.csv"
    df.to_csv(csv_path, index=False)

    controller = AgentController(max_iterations=1, enable_rlm=False, objective="Test diagnostics persistence")
    controller.load_dataset(str(csv_path), target_hint="target", interactive=False)

    # Set contexts
    controller.memory.set_context("domain_analysis", {"industry": "fintech"})
    controller.memory.set_context("causal_design", {"status": "none"})
    controller.memory.set_context("governance", {"code_executions": 0})
    controller.memory.set_context("active_domain_pack", "finance")

    controller.analyze()

    raw_json_path = out_dir / "reports" / "diag_data_raw.json"
    assert raw_json_path.exists()

    with raw_json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    assert "coverage" in data
    assert isinstance(data["coverage"], dict)
    assert "total" in data["coverage"]
    assert "domain_analysis" in data
    assert data["domain_analysis"] == {"industry": "fintech"}
    assert "causal_design" in data
    assert "governance" in data
    assert "active_domain_pack" in data
    assert data["active_domain_pack"] == "finance"


def test_integer_time_axis_continuous_series_archetype(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify integer temporal axis (year with 120 rows) without datetime column infers continuous_series."""
    out_dir = tmp_path / "output"
    monkeypatch.setenv("OUTPUT_DIR", str(out_dir))

    n = 120
    df = pd.DataFrame({
        "year": list(range(1900, 1900 + n)),
        "gdp": np.random.default_rng(42).normal(1000, 50, n),
        "population": np.random.default_rng(42).normal(500, 20, n),
    })
    csv_path = tmp_path / "yearly_macro.csv"
    df.to_csv(csv_path, index=False)

    controller = AgentController(max_iterations=1, enable_rlm=False)
    controller.load_dataset(str(csv_path), interactive=False)

    profile = controller.last_profile
    assert profile is not None
    assert profile.archetype == "continuous_series"
    assert any("year" in ev for ev in profile.archetype_evidence)
    assert profile.datetime_cols == ["year"]
    assert profile.is_time_series is True


def test_columns_for_targets_cleaned_vs_raw(tmp_path: Path) -> None:
    """Verify step_validation.columns_for respects file targets and does not leak cleaned columns to raw files."""
    from src.core.memory import DatasetMetadata, MemorySystem
    from src.core.step_validation import columns_for

    mem = MemorySystem()
    raw_path = tmp_path / "raw.csv"
    cleaned_path = tmp_path / "cleaned.csv"
    other_path = tmp_path / "other.csv"

    mem.dataset_metadata = DatasetMetadata(
        file_path=str(raw_path),
        row_count=100,
        column_count=2,
        columns={"a": "int64", "b": "int64"},
        missing_values={"a": 0, "b": 0},
        numerical_cols=["a", "b"],
        categorical_cols=[],
    )
    mem.set_context("cleaned_file_path", str(cleaned_path))
    mem.set_context("derived_datasets", {
        "cleaned_dataset": {
            "name": "cleaned_dataset",
            "path": str(cleaned_path),
            "rows": 100,
            "columns": ["a", "b", "a_log_return"],
        }
    })

    # Targeting cleaned file (explicit or omitted) returns cleaned columns
    assert columns_for(mem, str(cleaned_path)) == ["a", "b", "a_log_return"]
    assert columns_for(mem, None) == ["a", "b", "a_log_return"]

    # Targeting raw file or other file does NOT return cleaned_dataset columns
    assert columns_for(mem, str(raw_path)) == ["a", "b"]
    assert columns_for(mem, str(other_path)) == ["a", "b"]


def test_time_series_tool_integer_year_execution(tmp_path: Path) -> None:
    """Verify TimeSeriesAnalysisTool executes successfully on integer calendar years."""
    from src.tools.time_series import TimeSeriesAnalysisTool

    n = 40
    df = pd.DataFrame({
        "year": list(range(1980, 1980 + n)),
        "metric": np.random.default_rng(42).normal(100, 10, n),
    })
    csv_path = tmp_path / "yearly_data.csv"
    df.to_csv(csv_path, index=False)

    tool = TimeSeriesAnalysisTool()
    out = tool.execute(file_path=str(csv_path), date_column="year", value_column="metric")
    assert "summary" in out
    assert out.get("trend_direction") in ("increasing", "decreasing", "stable", "no clear trend", "no trend")


