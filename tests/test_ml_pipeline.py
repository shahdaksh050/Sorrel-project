"""Unit tests for ML Pipeline Tools — Stage 3 model training."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.tools.ml_pipeline import EvaluateModelTool, TrainModelTool


@pytest.fixture
def classification_csv(tmp_path: pytest.TempPathFactory) -> str:
    import numpy as np
    rng = np.random.default_rng(42)
    n = 200
    df = pd.DataFrame(
        {
            "f1": rng.normal(0, 1, n),
            "f2": rng.normal(0, 1, n),
            "f3": rng.normal(0, 1, n),
            "label": (rng.normal(0, 1, n) > 0).astype(int),
        }
    )
    p = tmp_path / "clf.csv"
    df.to_csv(p, index=False)
    return str(p)


@pytest.fixture
def regression_csv(tmp_path: pytest.TempPathFactory) -> str:
    import numpy as np
    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame(
        {
            "x1": rng.normal(0, 1, n),
            "x2": rng.normal(0, 1, n),
            "price": rng.normal(100, 15, n),
        }
    )
    p = tmp_path / "reg.csv"
    df.to_csv(p, index=False)
    return str(p)


class TestTrainModelTool:
    def test_classification_trains_successfully(self, classification_csv: str) -> None:
        result = TrainModelTool().run(
            file_path=classification_csv,
            target_column="label",
            task_type="classification",
            models=["random_forest", "logistic_regression"],
        )
        assert result.status == "success"
        assert "best_model" in result.output

    def test_cv_scores_present(self, classification_csv: str) -> None:
        result = TrainModelTool().run(
            file_path=classification_csv,
            target_column="label",
            models=["logistic_regression"],
        )
        assert result.status == "success"
        model_results = result.output["models_trained"]
        for m in model_results.values():
            assert "cv_mean" in m
            assert "cv_std" in m

    def test_train_test_gap_present(self, classification_csv: str) -> None:
        result = TrainModelTool().run(
            file_path=classification_csv,
            target_column="label",
            models=["logistic_regression"],
        )
        model_results = result.output["models_trained"]
        for m in model_results.values():
            assert "train_test_gap" in m

    def test_regression_trains_successfully(self, regression_csv: str) -> None:
        result = TrainModelTool().run(
            file_path=regression_csv,
            target_column="price",
            task_type="regression",
            models=["linear_regression", "ridge"],
        )
        assert result.status == "success"

    def test_low_max_depth_reduces_overfit_risk(self, classification_csv: str) -> None:
        """With max_depth=2, RF should generalise well on random data."""
        result = TrainModelTool().run(
            file_path=classification_csv,
            target_column="label",
            models=["random_forest"],
            max_depth=2,
            # Isolate the depth cap: tuning (tested separately) adds CV-selection
            # noise on this zero-signal dataset.
            tune_hyperparameters=False,
        )
        assert result.status == "success"
        gap = result.output["models_trained"]["random_forest"]["train_test_gap"]
        # Capped depth should keep gap < 0.30 even on random data
        assert gap < 0.30

    def test_missing_target_returns_error(self, classification_csv: str) -> None:
        result = TrainModelTool().run(
            file_path=classification_csv,
            target_column="nonexistent_col",
        )
        assert result.status == "error"


class TestEvaluateModelTool:
    def test_evaluate_after_train(self, classification_csv: str, tmp_path: pytest.TempPathFactory) -> None:
        output_dir = str(tmp_path / "models")
        train_result = TrainModelTool().run(
            file_path=classification_csv,
            target_column="label",
            models=["logistic_regression"],
            output_dir=output_dir,
        )
        assert train_result.status == "success"
        model_path = train_result.output["models_trained"]["logistic_regression"]["model_path"]

        eval_result = EvaluateModelTool().run(
            model_path=model_path,
            file_path=classification_csv,
            target_column="label",
            task_type="classification",
        )
        assert eval_result.status == "success"
        assert "accuracy" in eval_result.output

    def test_negative_train_test_gap_is_not_flagged_as_overfitting(self) -> None:
        # A negative gap means test scored higher than train — the opposite
        # of overfitting — and must not raise a "possible overfitting"
        # finding just because abs() was applied to the sign.
        found = EvaluateModelTool().findings(
            {"train_test_gap": -0.15, "task_type": "classification", "summary": "s"},
            profile=None,
            metadata=None,
        )
        assert not any(f.kind == "method_fit" for f in found)

    def test_positive_train_test_gap_over_threshold_is_flagged(self) -> None:
        found = EvaluateModelTool().findings(
            {"train_test_gap": 0.15, "task_type": "classification", "summary": "s"},
            profile=None,
            metadata=None,
        )
        assert any(f.kind == "method_fit" for f in found)

    def test_model_path_outside_output_root_is_rejected(self, tmp_path: pytest.TempPathFactory) -> None:
        # execute() unpickles model_path unsandboxed — a planner-supplied
        # path that exists but sits outside the run's output_root must be
        # replaced with the trusted best_model_path, never used as-is.
        from src.core.memory import MemorySystem

        output_root = tmp_path / "run"
        trusted = output_root / "models" / "trusted.pkl"
        trusted.parent.mkdir(parents=True)
        trusted.write_bytes(b"not a real pickle")
        outside = tmp_path / "outside.pkl"
        outside.write_bytes(b"not a real pickle")

        memory = MemorySystem()
        memory.set_context("best_model_path", str(trusted))
        params = EvaluateModelTool().prepare_params(
            {"model_path": str(outside), "file_path": "x.csv", "target_column": "label"},
            memory,
            str(output_root),
        )
        assert params["model_path"] == str(trusted)

    def test_model_path_outside_output_root_with_no_fallback_is_blanked(self, tmp_path: pytest.TempPathFactory) -> None:
        from src.core.memory import MemorySystem

        output_root = tmp_path / "run"
        outside = tmp_path / "outside.pkl"
        outside.write_bytes(b"not a real pickle")

        memory = MemorySystem()
        params = EvaluateModelTool().prepare_params(
            {"model_path": str(outside), "file_path": "x.csv", "target_column": "label"},
            memory,
            str(output_root),
        )
        assert params["model_path"] == ""


@pytest.fixture
def string_target_csv(tmp_path: pytest.TempPathFactory) -> str:
    """String labels — regression test for XGBoost/pandas-3 target encoding."""
    import numpy as np
    rng = np.random.default_rng(7)
    n = 150
    f1 = rng.normal(0, 1, n)
    df = pd.DataFrame(
        {
            "f1": f1,
            "f2": rng.normal(0, 1, n),
            "segment": rng.choice(["gold", "silver"], n),  # categorical feature
            "churn": np.where(f1 > 0, "yes", "no"),        # string target
        }
    )
    p = tmp_path / "str_target.csv"
    df.to_csv(p, index=False)
    return str(p)


class TestStringTargetSupport:
    def test_random_forest_trains_on_string_target(self, string_target_csv: str) -> None:
        result = TrainModelTool().run(
            file_path=string_target_csv,
            target_column="churn",
            task_type="classification",
            models=["random_forest"],
            n_cv_folds=3,
        )
        assert result.status == "success"
        assert result.output["class_labels"] == ["no", "yes"]

    def test_task_type_auto_detects_classification(self, string_target_csv: str) -> None:
        result = TrainModelTool().run(
            file_path=string_target_csv,
            target_column="churn",
            task_type="auto",
            models=["logistic_regression"],
            n_cv_folds=3,
        )
        assert result.status == "success"
        assert result.output["task_type"] == "classification"

    def test_evaluate_maps_class_keys_back_to_labels(
        self, string_target_csv: str, tmp_path: pytest.TempPathFactory
    ) -> None:
        models_dir = str(tmp_path / "models")
        train = TrainModelTool().run(
            file_path=string_target_csv,
            target_column="churn",
            task_type="classification",
            models=["logistic_regression"],
            n_cv_folds=3,
            output_dir=models_dir,
        )
        assert train.status == "success"
        model_path = train.output["models_trained"]["logistic_regression"]["model_path"]

        result = EvaluateModelTool().run(
            model_path=model_path,
            file_path=string_target_csv,
            target_column="churn",
            task_type="classification",
        )
        assert result.status == "success"
        assert "train_test_gap" in result.output
        report_keys = set(result.output["classification_report"].keys())
        assert {"no", "yes"} <= report_keys


def test_train_model_clustering_branch(tmp_path: Path) -> None:
    """task_type='clustering' fits every model on the full frame: no split, no baseline."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "a": rng.normal(size=60), "b": rng.normal(size=60),
        "c": rng.integers(0, 3, 60), "y": rng.integers(0, 2, 60),
    })
    path = tmp_path / "c.csv"
    df.to_csv(path, index=False)
    out = TrainModelTool().execute(
        file_path=str(path), target_column="y", task_type="clustering",
        output_dir=str(tmp_path / "models"),
    )
    assert out["task_type"] == "clustering"
    assert out["best_model"] == "kmeans"
    assert sorted(out["models_trained"]) == ["dbscan", "kmeans"]
    assert out["baseline_cv_mean"] is None and out["split_strategy"] == "n/a"
    assert out["train_samples"] == out["test_samples"] == 60
    assert (tmp_path / "models" / "kmeans.pkl.sig").exists()
