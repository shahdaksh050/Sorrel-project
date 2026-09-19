"""
Unit tests for sandbox toolkit enhancements:
- dsa.cramers_v
- dsa.crosstab_shares
- dsa.baseline_accuracy
- dsa.chart additions (waterfall, lorenz, dot_ci, dual_axis)
- validate_chart_spec and spec_to_vegalite for new chart types
- AST pre-screening for ML code when use_ml=False
"""
from __future__ import annotations

import pandas as pd
import pytest

from src.core.chart_spec import spec_to_vegalite, validate_chart_spec
from src.core.controller import AgentController
from src.core.llm_client import LLMClient
from src.core.sandbox_toolkit import (
    Toolkit,
    dot_ci,
    dual_axis,
    lorenz,
    waterfall,
)


@pytest.fixture
def sample_categorical_df() -> pd.DataFrame:
    return pd.DataFrame({
        "gender": ["M", "M", "F", "F", "M", "F", "M", "F"],
        "purchased": ["Yes", "Yes", "No", "No", "Yes", "No", "No", "Yes"],
        "category": ["A", "B", "A", "B", "A", "B", "A", "B"],
        "revenue": [100.0, 150.0, 50.0, 80.0, 120.0, 60.0, 90.0, 110.0],
        "cost": [70.0, 90.0, 40.0, 50.0, 80.0, 45.0, 60.0, 75.0],
    })


class TestToolkitStatisticalMethods:
    def test_cramers_v_valid(self, sample_categorical_df: pd.DataFrame) -> None:
        result = Toolkit.cramers_v(sample_categorical_df, "gender", "purchased")
        assert "cramers_v" in result
        assert "chi2" in result
        assert "p_value" in result
        assert 0.0 <= result["cramers_v"] <= 1.0
        assert result["n"] == 8

    def test_cramers_v_missing_columns_raises(self, sample_categorical_df: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="Column 'non_existent' not found"):
            Toolkit.cramers_v(sample_categorical_df, "gender", "non_existent")

    def test_cramers_v_single_level_is_zero(self) -> None:
        frame = pd.DataFrame({"a": ["x"] * 4, "b": ["p", "q", "p", "q"]})
        result = Toolkit.cramers_v(frame, "a", "b")
        assert result["cramers_v"] == 0.0
        assert result["p_value"] == 1.0

    def test_cramers_v_rejects_id_like_columns(self) -> None:
        ids = [f"id{i}" for i in range(1000)]
        frame = pd.DataFrame({"a": ids, "b": ids})
        with pytest.raises(ValueError, match="categorical"):
            Toolkit.cramers_v(frame, "a", "b")

    def test_crosstab_shares_keeps_row_labels(self, sample_categorical_df: pd.DataFrame) -> None:
        ct = Toolkit.crosstab_shares(sample_categorical_df, "gender", "purchased", normalize="index")
        assert isinstance(ct, pd.DataFrame)
        assert list(ct.columns) == ["gender", "No", "Yes"]
        by_gender = ct.set_index("gender")
        assert abs(float(by_gender.loc["M"].sum()) - 100.0) < 0.1
        assert abs(float(by_gender.loc["F"].sum()) - 100.0) < 0.1

    def test_crosstab_shares_rejects_bad_normalize(self, sample_categorical_df: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="normalize"):
            Toolkit.crosstab_shares(sample_categorical_df, "gender", "purchased", normalize="rows")

    def test_baseline_accuracy_majority(self, sample_categorical_df: pd.DataFrame) -> None:
        result = Toolkit.baseline_accuracy(sample_categorical_df, "purchased")
        assert result["target"] == "purchased"
        assert result["majority_baseline_accuracy"] == 0.5  # 4 Yes, 4 No
        assert result["n"] == 8
        assert result["n_classes"] == 2

    def test_baseline_accuracy_rejects_continuous_target(self) -> None:
        frame = pd.DataFrame({"y": [i * 0.37 for i in range(200)]})
        with pytest.raises(ValueError, match="categorical"):
            Toolkit.baseline_accuracy(frame, "y")


class TestChartSpecAdditions:
    def test_waterfall_chart_spec(self) -> None:
        data = [
            {"step": "Starting", "change": 100},
            {"step": "Revenue", "change": 50},
            {"step": "Costs", "change": -30},
        ]
        spec = waterfall(data, x="step", y="change", title="Net Income Waterfall")
        clean, err = validate_chart_spec(spec)
        assert err is None
        assert clean is not None
        assert clean["type"] == "waterfall"
        vl = spec_to_vegalite(clean)
        assert "transform" in vl
        assert vl["mark"]["type"] == "bar"

    def test_lorenz_chart_spec(self) -> None:
        data = [
            {"cum_pop": 0.0, "cum_val": 0.0},
            {"cum_pop": 0.5, "cum_val": 0.2},
            {"cum_pop": 1.0, "cum_val": 1.0},
        ]
        spec = lorenz(data, x="cum_pop", y="cum_val", title="Lorenz Curve")
        clean, err = validate_chart_spec(spec)
        assert err is None
        assert clean is not None
        assert clean["type"] == "lorenz"
        vl = spec_to_vegalite(clean)
        assert "layer" in vl
        assert len(vl["layer"]) == 2  # equality diagonal + curve

    def test_dot_ci_chart_spec(self) -> None:
        data = [
            {"group": "A", "estimate": 10.0, "ci_lower": 8.0, "ci_upper": 12.0},
            {"group": "B", "estimate": 15.0, "ci_lower": 13.0, "ci_upper": 17.0},
        ]
        spec = dot_ci(
            data, x="group", y="estimate", y_lower="ci_lower", y_upper="ci_upper", title="Effect by Group"
        )
        clean, err = validate_chart_spec(spec)
        assert err is None
        assert clean is not None
        assert clean["type"] == "dot_ci"
        vl = spec_to_vegalite(clean)
        assert "layer" in vl

    def test_dual_axis_chart_spec(self) -> None:
        data = [
            {"date": "2024-01-01", "revenue": 100.0, "margin": 0.2},
            {"date": "2024-02-01", "revenue": 120.0, "margin": 0.25},
        ]
        spec = dual_axis(data, x="date", y="revenue", y2="margin", title="Revenue & Margin")
        clean, err = validate_chart_spec(spec)
        assert err is None
        assert clean is not None
        assert clean["type"] == "dual_axis"
        vl = spec_to_vegalite(clean)
        assert "layer" in vl
        assert vl.get("resolve", {}).get("scale", {}).get("y") == "independent"

    def test_chart_spec_with_thresholds(self) -> None:
        data = [
            {"category": "X", "value": 10},
            {"category": "Y", "value": 20},
        ]
        spec = {
            "type": "bar",
            "data": data,
            "x": "category",
            "y": "value",
            "annotations": [{"y": 15}, {"y": 18, "label": "Target"}],
        }
        clean, err = validate_chart_spec(spec)
        assert err is None
        assert clean is not None
        assert "annotations" in clean
        assert len(clean["annotations"]) == 2
        vl = spec_to_vegalite(clean)
        assert "layer" in vl


class TestMLPreScreening:
    def test_detects_sklearn_import(self) -> None:
        code = "import sklearn\nRESULT = 1\n"
        assert AgentController._detects_ml_code(code) is True

    def test_detects_sklearn_from_import(self) -> None:
        code = "from sklearn.ensemble import RandomForestClassifier\nRESULT = 1\n"
        assert AgentController._detects_ml_code(code) is True

    def test_detects_estimator_names(self) -> None:
        code = "clf = RandomForestClassifier()\nRESULT = 1\n"
        assert AgentController._detects_ml_code(code) is True

    def test_passes_non_ml_code(self) -> None:
        code = "g = df.groupby('region')['amount'].mean()\nRESULT = g\n"
        assert AgentController._detects_ml_code(code) is False


class TestGroqLLMProvider:
    def test_groq_default_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LLM_PROVIDER", "groq")
        monkeypatch.setenv("GROQ_API_KEY", "dummy_key")
        monkeypatch.delenv("LLM_MODEL", raising=False)
        client = LLMClient()
        assert client.provider == "groq"
        assert client.model == "llama-3.3-70b-versatile"
