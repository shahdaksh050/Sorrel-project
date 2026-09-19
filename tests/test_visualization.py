"""Unit tests for Visualization Tools — Stage 3 chart generation."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.tools.ml_pipeline import TrainModelTool
from src.tools.visualization import GenerateVisualizationsTool


@pytest.fixture
def numeric_csv(tmp_path: pytest.TempPathFactory) -> str:
    rng = np.random.default_rng(42)
    n = 150
    df = pd.DataFrame(
        {
            "f1": rng.normal(0, 1, n),
            "f2": rng.normal(5, 2, n),
            "f3": rng.uniform(0, 10, n),
            "label": (rng.normal(0, 1, n) > 0).astype(int),
        }
    )
    p = tmp_path / "viz.csv"
    df.to_csv(p, index=False)
    return str(p)


class TestGenerateVisualizationsTool:
    def test_correlation_heatmap_saves_png(self, numeric_csv: str, tmp_path: pytest.TempPathFactory) -> None:
        out = str(tmp_path / "charts")
        result = GenerateVisualizationsTool().run(
            file_path=numeric_csv, chart_type="correlation_heatmap", output_dir=out
        )
        assert result.status == "success"
        assert "charts" in result.output and len(result.output["charts"]) >= 1

    def test_distributions_save_one_png_per_numeric_col(
        self, numeric_csv: str, tmp_path: pytest.TempPathFactory
    ) -> None:
        out = str(tmp_path / "charts")
        result = GenerateVisualizationsTool().run(
            file_path=numeric_csv, chart_type="distributions", output_dir=out
        )
        assert result.status == "success"
        assert "charts" in result.output and len(result.output["charts"]) == 3  # f1, f2, f3; the 2-value label is rejected by the critic

    def test_unknown_chart_type_errors(self, numeric_csv: str) -> None:
        result = GenerateVisualizationsTool().run(
            file_path=numeric_csv, chart_type="pie_of_doom"
        )
        assert result.status == "error"

    def test_feature_importance_requires_model_path(self, numeric_csv: str) -> None:
        result = GenerateVisualizationsTool().run(
            file_path=numeric_csv, chart_type="feature_importance", target_column="label"
        )
        assert result.status == "error"

    def test_feature_importance_with_trained_model(
        self, numeric_csv: str, tmp_path: pytest.TempPathFactory
    ) -> None:
        models_dir = str(tmp_path / "models")
        train = TrainModelTool().run(
            file_path=numeric_csv,
            target_column="label",
            task_type="classification",
            models=["random_forest"],
            n_cv_folds=3,
            output_dir=models_dir,
        )
        assert train.status == "success"
        model_path = train.output["models_trained"]["random_forest"]["model_path"]

        out = str(tmp_path / "charts")
        result = GenerateVisualizationsTool().run(
            file_path=numeric_csv,
            chart_type="feature_importance",
            target_column="label",
            model_path=model_path,
            output_dir=out,
        )
        assert result.status == "success"
        assert "charts" in result.output and len(result.output["charts"]) >= 1

    def test_roc_curve_rejects_non_binary_target(
        self, tmp_path: pytest.TempPathFactory
    ) -> None:
        rng = np.random.default_rng(0)
        df = pd.DataFrame({"x": rng.normal(0, 1, 60), "y": rng.integers(0, 5, 60)})
        p = tmp_path / "multi.csv"
        df.to_csv(p, index=False)
        result = GenerateVisualizationsTool().run(
            file_path=str(p),
            chart_type="roc_curve",
            target_column="y",
            model_path="irrelevant.pkl",
        )
        assert result.status == "error"
