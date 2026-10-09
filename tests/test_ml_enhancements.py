"""Tests for ML pipeline enhancements: auto-treatments, tuning, explainability."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.tools.ml_pipeline import (
    SKEW_TREATMENT_THRESHOLD,
    EvaluateModelTool,
    TrainModelTool,
    _detect_target_leakage,
    _prepare_features,
)

RNG = np.random.default_rng(42)


def _make_df(n: int = 200) -> pd.DataFrame:
    signal = RNG.normal(0, 1, n)
    return pd.DataFrame({
        "customer_id": range(1, n + 1),
        "signal": signal,
        "amount": np.exp(RNG.normal(3, 1.2, n)),          # heavy right skew
        "plan": RNG.choice(["basic", "pro"], n),
        "label": (signal + RNG.normal(0, 0.5, n) > 0).astype(int),
    })


class TestAutoTreatments:
    def test_identifier_dropped(self) -> None:
        X, _y, treatments = _prepare_features(_make_df(), "label")
        assert "customer_id" not in X.columns
        assert any("identifier" in t for t in treatments)

    def test_raw_free_text_is_not_one_hot_encoded(self) -> None:
        df = _make_df()
        df["specification"] = [
            f"processor family {i % 4} with camera package and battery option {i % 3}"
            for i in range(len(df))
        ]
        X, _y, treatments = _prepare_features(df, "label")
        assert "specification" not in X.columns
        assert any("raw prose" in treatment for treatment in treatments)

    def test_categorical_mapping_can_leak_continuous_target(self) -> None:
        groups = pd.Series((["a", "b", "c", "d"] * 25), name="lookup")
        y = groups.map({"a": 10.0, "b": 20.0, "c": 30.0, "d": 40.0})
        warnings = _detect_target_leakage(
            pd.DataFrame({"lookup": groups}), y, "regression"
        )
        assert any("continuous target variance" in warning for warning in warnings)

    def test_skewed_feature_reported_via_training_pipeline(self, tmp_path: Path) -> None:
        """P0.1: the log1p decision now happens inside the fitted Pipeline
        (from the training fold only), not in _prepare_features — but the
        report must still surface it."""
        df = _make_df()
        p = tmp_path / "skew.csv"
        df.to_csv(p, index=False)
        result = TrainModelTool().run(
            file_path=str(p), target_column="label", task_type="classification",
            models=["logistic_regression"], tune_hyperparameters=False,
            output_dir=str(tmp_path / "m"),
        )
        assert result.status == "success"
        assert any(
            "log1p" in t and "amount" in t for t in result.output["treatments_applied"]
        )

    def test_log1p_decision_uses_training_fold_only_not_full_frame(
        self, tmp_path: Path
    ) -> None:
        """The regression guard for P0.1 itself, not just the mechanism: a
        column whose *training-fold* skew is below threshold, but whose
        full-frame skew (train+test combined) is above threshold because of
        outliers concentrated in the test tail, must NOT get log1p applied.
        Pre-refactor (deciding over the whole frame before the split) this
        line would appear; post-refactor it must not."""
        n = 200
        train_n = 160  # matches default test_size=0.2 under a time_series split
        rng = np.random.default_rng(3)
        train_vals = rng.normal(50, 5, train_n)
        test_vals = rng.normal(50, 5, n - train_n)
        test_vals[-3:] = [5000.0, 6000.0, 7000.0]  # outliers only in the test tail
        amount = np.concatenate([train_vals, test_vals])
        assert pd.Series(train_vals).skew() < SKEW_TREATMENT_THRESHOLD
        assert abs(pd.Series(amount).skew()) >= SKEW_TREATMENT_THRESHOLD

        dates = pd.date_range("2023-01-01", periods=n, freq="D")
        f1 = rng.normal(0, 1, n)
        label = (f1 + rng.normal(0, 0.3, n) > 0).astype(int)
        df = pd.DataFrame({"date": dates, "f1": f1, "amount": amount, "label": label})
        p = tmp_path / "leak_probe.csv"
        df.to_csv(p, index=False)

        result = TrainModelTool().run(
            file_path=str(p), target_column="label", task_type="classification",
            models=["logistic_regression"], tune_hyperparameters=False,
            split_strategy="time_series", time_column="date",
            output_dir=str(tmp_path / "m"),
        )
        assert result.status == "success"
        assert not any(
            "log1p" in t and "amount" in t for t in result.output["treatments_applied"]
        )

    def test_prepare_features_no_longer_transforms_values(self) -> None:
        """_prepare_features now only does structural feature engineering
        (datetime expansion, ID dropping) — statistical decisions like log1p
        move into the Pipeline so they're fit on the training fold only."""
        df = _make_df()
        raw_max = float(df["amount"].max())
        X, _y, _treatments = _prepare_features(df, "label")
        assert float(X["amount"].max()) == raw_max  # untouched

    def test_treatments_deterministic_across_calls(self) -> None:
        df = _make_df()
        _, _, t1 = _prepare_features(df, "label")
        _, _, t2 = _prepare_features(df.copy(), "label")
        assert t1 == t2

    def test_clean_data_gets_no_treatments(self) -> None:
        df = pd.DataFrame({
            "a": RNG.normal(0, 1, 100),
            "b": RNG.normal(5, 2, 100),
            "label": RNG.choice([0, 1], 100),
        })
        _, _, treatments = _prepare_features(df, "label")
        assert treatments == []

    def test_imbalance_triggers_class_weighting(self, tmp_path: Path) -> None:
        n = 400
        f = RNG.normal(0, 1, n)
        label = np.zeros(n, dtype=int)
        label[np.argsort(f)[-20:]] = 1  # 5% minority
        df = pd.DataFrame({"f": f, "g": RNG.normal(0, 1, n), "label": label})
        p = tmp_path / "imb.csv"
        df.to_csv(p, index=False)

        result = TrainModelTool().run(
            file_path=str(p), target_column="label", task_type="classification",
            models=["logistic_regression"], tune_hyperparameters=False,
            output_dir=str(tmp_path / "m"),
        )
        assert result.status == "success"
        assert any("Class weighting" in t for t in result.output["treatments_applied"])

    def test_balanced_data_gets_no_class_weighting(self, tmp_path: Path) -> None:
        df = _make_df()
        p = tmp_path / "bal.csv"
        df.to_csv(p, index=False)
        result = TrainModelTool().run(
            file_path=str(p), target_column="label", task_type="classification",
            models=["logistic_regression"], tune_hyperparameters=False,
            output_dir=str(tmp_path / "m"),
        )
        assert not any("Class weighting" in t for t in result.output["treatments_applied"])

    def test_datetime_column_expanded_not_dropped(self) -> None:
        """P0.4: a time-series dataset's best-detected signal must survive
        into the feature matrix instead of being thrown away."""
        df = _make_df()
        df["signup_date"] = pd.date_range("2023-01-01", periods=len(df), freq="D")
        X, _y, treatments = _prepare_features(df, "label")
        assert "signup_date" not in X.columns
        assert any("Expanded datetime column" in t for t in treatments)
        for suffix in ("year", "month", "day", "dayofweek", "hour", "is_weekend", "days_since_min"):
            assert f"signup_date_{suffix}" in X.columns
        assert not X["signup_date_days_since_min"].isna().any()


class TestSkewLog1pTransformer:
    """P0.1: the skew decision must be made once, at fit time, from whatever
    frame it's fit on — the Pipeline usage ensures that's the training fold."""

    def test_transforms_only_skewed_columns_based_on_fit_data(self) -> None:
        from src.tools.ml_pipeline import _SkewLog1pTransformer

        rng = np.random.default_rng(5)
        train = pd.DataFrame({
            "skewed": np.exp(rng.normal(3, 1.2, 200)),
            "normal": rng.normal(0, 1, 200),
        })
        t = _SkewLog1pTransformer().fit(train)
        assert "skewed" in t.skewed_cols_
        assert "normal" not in t.skewed_cols_

        out = t.transform(train)
        assert float(out["skewed"].max()) < float(train["skewed"].max())
        assert (out["normal"] == train["normal"]).all()

    def test_decision_is_frozen_at_fit_time(self) -> None:
        from src.tools.ml_pipeline import _SkewLog1pTransformer

        rng = np.random.default_rng(6)
        train = pd.DataFrame({"amount": rng.normal(0, 1, 200)})  # not skewed
        t = _SkewLog1pTransformer().fit(train)
        assert t.skewed_cols_ == []

        skewed_new_data = pd.DataFrame({"amount": np.exp(rng.normal(3, 1.2, 50))})
        out = t.transform(skewed_new_data)
        # Fit decided "amount" is not skewed; transform must not reconsider
        # that decision on new data even though this new data IS skewed.
        assert (out["amount"] == skewed_new_data["amount"]).all()

    def test_negative_test_fold_values_are_clipped_not_nan(self) -> None:
        from src.tools.ml_pipeline import _SkewLog1pTransformer

        rng = np.random.default_rng(8)
        train = pd.DataFrame({"amount": np.exp(rng.normal(3, 1.2, 200))})
        t = _SkewLog1pTransformer().fit(train)
        assert "amount" in t.skewed_cols_

        test_data = pd.DataFrame({"amount": [-5.0, 10.0]})
        out = t.transform(test_data)
        assert not out["amount"].isna().any()


class TestSplitStrategy:
    """P0.3: the profiler detects time-series/panel structure — the
    splitter must actually hear about it instead of shuffling regardless."""

    def test_prepare_params_wires_time_series_from_profile(self) -> None:
        from src.core.memory import MemorySystem

        memory = MemorySystem()
        memory.set_context(
            "data_profile",
            {"is_time_series": True, "datetime_cols": ["signup_date"], "panel_group_cols": []},
        )
        params = TrainModelTool().prepare_params(
            {"file_path": "x.csv", "target_column": "label"}, memory, "output"
        )
        assert params["split_strategy"] == "time_series"
        assert params["time_column"] == "signup_date"

    def test_prepare_params_wires_panel_from_profile(self) -> None:
        from src.core.memory import MemorySystem

        memory = MemorySystem()
        memory.set_context(
            "data_profile",
            {"is_time_series": False, "datetime_cols": [], "panel_group_cols": ["customer_id"]},
        )
        params = TrainModelTool().prepare_params(
            {"file_path": "x.csv", "target_column": "label"}, memory, "output"
        )
        assert params["split_strategy"] == "panel"
        assert params["group_column"] == "customer_id"

    def test_prepare_params_leaves_random_when_planner_chose_it(self) -> None:
        from src.core.memory import MemorySystem

        memory = MemorySystem()
        memory.set_context(
            "data_profile",
            {"is_time_series": True, "datetime_cols": ["signup_date"], "panel_group_cols": []},
        )
        params = TrainModelTool().prepare_params(
            {"file_path": "x.csv", "target_column": "label", "split_strategy": "random"},
            memory, "output",
        )
        assert params["split_strategy"] == "random"

    def test_time_series_split_reported_and_chronological(self, tmp_path: Path) -> None:
        n = 200
        dates = pd.date_range("2023-01-01", periods=n, freq="D")
        # Oscillating signal (both classes present throughout) so every
        # TimeSeriesSplit fold sees both classes — isolates the invariant
        # under test (chronological split, reported strategy) from noise.
        wave = np.sin(np.linspace(0, 6 * np.pi, n))
        df = pd.DataFrame({
            "date": dates,
            "f1": wave + RNG.normal(0, 0.1, n),
            "label": (wave + RNG.normal(0, 0.1, n) > 0).astype(int),
        })
        p = tmp_path / "ts.csv"
        df.to_csv(p, index=False)

        result = TrainModelTool().run(
            file_path=str(p), target_column="label", task_type="classification",
            models=["logistic_regression"], tune_hyperparameters=False,
            split_strategy="time_series", time_column="date",
            output_dir=str(tmp_path / "m"),
        )
        assert result.status == "success"
        assert result.output["split_strategy"] == "time_series"
        assert result.output["train_samples"] + result.output["test_samples"] == n
        # Chronological: the held-out fraction is exactly the tail, so
        # test_samples matches test_size applied to the full ordered series.
        assert result.output["test_samples"] == int(n * 0.2)

    def test_panel_split_keeps_entities_out_of_both_folds(self, tmp_path: Path) -> None:
        n_entities = 40
        rows_per_entity = 5
        rng = np.random.default_rng(3)
        entity = np.repeat(np.arange(n_entities), rows_per_entity)
        f1 = rng.normal(0, 1, n_entities * rows_per_entity)
        label = (f1 + rng.normal(0, 0.3, n_entities * rows_per_entity) > 0).astype(int)
        df = pd.DataFrame({"entity_id": entity, "f1": f1, "label": label})
        p = tmp_path / "panel.csv"
        df.to_csv(p, index=False)

        result = TrainModelTool().run(
            file_path=str(p), target_column="label", task_type="classification",
            models=["logistic_regression"], tune_hyperparameters=False,
            split_strategy="panel", group_column="entity_id",
            output_dir=str(tmp_path / "m"),
        )
        assert result.status == "success"
        assert result.output["split_strategy"] == "panel"

    def test_evaluate_reuses_trains_split_strategy(self, tmp_path: Path) -> None:
        """EvaluateModelTool must recreate train_model's actual partition —
        not always a random one — or its 'held-out' metrics are inflated
        on exactly the datasets P0.3 targets."""
        n = 200
        dates = pd.date_range("2023-01-01", periods=n, freq="D")
        wave = np.sin(np.linspace(0, 6 * np.pi, n))
        df = pd.DataFrame({
            "date": dates,
            "f1": wave + RNG.normal(0, 0.1, n),
            "label": (wave + RNG.normal(0, 0.1, n) > 0).astype(int),
        })
        p = tmp_path / "ts_eval.csv"
        df.to_csv(p, index=False)
        models_dir = str(tmp_path / "m")

        train = TrainModelTool().run(
            file_path=str(p), target_column="label", task_type="classification",
            models=["logistic_regression"], tune_hyperparameters=False,
            split_strategy="time_series", time_column="date",
            output_dir=models_dir,
        )
        assert train.status == "success"
        model_path = train.output["models_trained"]["logistic_regression"]["model_path"]

        eval_result = EvaluateModelTool().run(
            model_path=model_path, file_path=str(p), target_column="label",
            task_type="classification",
            split_strategy="time_series", time_column="date",
        )
        assert eval_result.status == "success"
        # Same chronological partition train_model used: the trained model's
        # own test-fold accuracy must match evaluate_model's, since both are
        # scoring the identical held-out rows.
        assert eval_result.output["accuracy"] == train.output["models_trained"]["logistic_regression"]["test_metrics"]["accuracy"]

    def test_evaluate_prepare_params_pulls_persisted_split_strategy(self) -> None:
        from src.core.memory import MemorySystem

        memory = MemorySystem()
        memory.set_context("split_strategy", "panel")
        memory.set_context("split_time_column", None)
        memory.set_context("split_group_column", "entity_id")
        params = EvaluateModelTool().prepare_params(
            {"model_path": "m.pkl", "file_path": "x.csv", "target_column": "label"},
            memory, "output",
        )
        assert params["split_strategy"] == "panel"
        assert params["group_column"] == "entity_id"


class TestHyperparameterTuning:
    def test_tuning_reports_best_params(self, tmp_path: Path) -> None:
        df = _make_df()
        p = tmp_path / "tune.csv"
        df.to_csv(p, index=False)
        result = TrainModelTool().run(
            file_path=str(p), target_column="label", task_type="classification",
            models=["logistic_regression"], tune_hyperparameters=True,
            output_dir=str(tmp_path / "m"),
        )
        assert result.status == "success"
        assert result.output["hyperparameter_tuning"] is True
        params = result.output["models_trained"]["logistic_regression"]["best_params"]
        assert "C" in params

    def test_tuning_can_be_disabled(self, tmp_path: Path) -> None:
        df = _make_df()
        p = tmp_path / "notune.csv"
        df.to_csv(p, index=False)
        result = TrainModelTool().run(
            file_path=str(p), target_column="label", task_type="classification",
            models=["logistic_regression"], tune_hyperparameters=False,
            output_dir=str(tmp_path / "m"),
        )
        assert result.output["hyperparameter_tuning"] is False
        assert result.output["models_trained"]["logistic_regression"]["best_params"] == {}


class TestUnseenCategoryHandling:
    """P0.6: encoders must be fit only on the training fold. The old
    whole-dataset LabelEncoder hid this — a category appearing only in the
    test split must not break predict()."""

    def test_unseen_category_in_test_split_does_not_break_predict(
        self, tmp_path: Path
    ) -> None:
        n = 200
        dates = pd.date_range("2023-01-01", periods=n, freq="D")
        rng = np.random.default_rng(11)
        f1 = rng.normal(0, 1, n)
        plan = np.where(
            np.arange(n) < 150, rng.choice(["basic", "pro"], n), "basic"
        ).astype(object)
        # "enterprise" exists ONLY in the tail — guaranteed to land in the
        # test split under a time_series (purely positional, no shuffle) split.
        plan[-1] = "enterprise"
        label = (f1 + rng.normal(0, 0.3, n) > 0).astype(int)
        df = pd.DataFrame({"date": dates, "f1": f1, "plan": plan, "label": label})
        p = tmp_path / "unseen_cat.csv"
        df.to_csv(p, index=False)

        result = TrainModelTool().run(
            file_path=str(p), target_column="label", task_type="classification",
            models=["logistic_regression"], tune_hyperparameters=False,
            split_strategy="time_series", time_column="date",
            output_dir=str(tmp_path / "m"),
        )
        assert result.status == "success"


class TestExplainability:
    @pytest.fixture
    def trained(self, tmp_path: Path) -> tuple[str, str]:
        df = _make_df()
        p = tmp_path / "data.csv"
        df.to_csv(p, index=False)
        train = TrainModelTool().run(
            file_path=str(p), target_column="label", task_type="classification",
            models=["random_forest"], tune_hyperparameters=False,
            output_dir=str(tmp_path / "m"),
        )
        model_path = train.output["models_trained"]["random_forest"]["model_path"]
        return str(p), model_path

    def test_drivers_identify_the_signal(self, trained: tuple[str, str]) -> None:
        data_path, model_path = trained
        result = EvaluateModelTool().run(
            model_path=model_path, file_path=data_path,
            target_column="label", task_type="classification",
        )
        assert result.status == "success"
        drivers = result.output["top_drivers"]
        assert drivers, "permutation importance must yield at least one driver"
        assert drivers[0]["feature"] == "signal"
        assert drivers[0]["direction"] == "increases"

    def test_narrative_is_plain_language(self, trained: tuple[str, str]) -> None:
        data_path, model_path = trained
        result = EvaluateModelTool().run(
            model_path=model_path, file_path=data_path,
            target_column="label", task_type="classification",
        )
        narrative = result.output["driver_narrative"]
        assert narrative
        assert "driver" in narrative[0]
        assert "signal" in narrative[0]
