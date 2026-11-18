"""Unit tests for src/core/profiler.py — automated dataset profiling."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.core.profiler import DatasetProfile, profile_dataframe

RNG = np.random.default_rng(42)


@pytest.fixture
def clean_df() -> pd.DataFrame:
    n = 200
    return pd.DataFrame({
        "customer_id": range(1, n + 1),
        "age": RNG.integers(18, 80, n),
        "income": RNG.normal(50_000, 12_000, n),
        "plan": RNG.choice(["basic", "pro", "enterprise"], n),
        "signup_date": pd.date_range("2023-01-01", periods=n, freq="D"),
        "active": RNG.choice([True, False], n),
        "churn": RNG.choice([0, 1], n, p=[0.7, 0.3]),
    })


def _col(profile: DatasetProfile, name: str):  # type: ignore[no-untyped-def]
    return next(c for c in profile.columns if c.name == name)


class TestColumnKinds:
    def test_numeric_detection(self, clean_df: pd.DataFrame) -> None:
        profile = profile_dataframe(clean_df)
        assert _col(profile, "income").kind == "numeric"
        assert "mean" in _col(profile, "income").stats

    def test_categorical_detection(self, clean_df: pd.DataFrame) -> None:
        profile = profile_dataframe(clean_df)
        col = _col(profile, "plan")
        assert col.kind == "categorical"
        assert col.top_values  # value counts captured

    def test_boolean_detection(self, clean_df: pd.DataFrame) -> None:
        assert _col(profile_dataframe(clean_df), "active").kind == "boolean"

    def test_identifier_detection(self, clean_df: pd.DataFrame) -> None:
        assert _col(profile_dataframe(clean_df), "customer_id").kind == "identifier"

    def test_datetime_detection(self, clean_df: pd.DataFrame) -> None:
        assert _col(profile_dataframe(clean_df), "signup_date").kind == "datetime"

    def test_datetime_string_detection(self) -> None:
        df = pd.DataFrame({
            "when": ["2024-01-01", "2024-02-01", "2024-03-01"] * 10,
            "v": range(30),
        })
        assert _col(profile_dataframe(df), "when").kind == "datetime"

    def test_constant_detection(self) -> None:
        df = pd.DataFrame({"fixed": [7] * 50, "v": range(50)})
        profile = profile_dataframe(df)
        assert _col(profile, "fixed").kind == "constant"
        assert any("constant" in w for w in profile.warnings)


class TestFlagsAndWarnings:
    def test_severe_skew_flagged(self) -> None:
        skewed = np.concatenate([RNG.normal(10, 1, 195), np.full(5, 10_000.0)])
        df = pd.DataFrame({"amount": skewed, "v": range(200)})
        assert "severe_skew" in _col(profile_dataframe(df), "amount").flags

    def test_high_missing_flagged(self) -> None:
        vals = [1.0] * 60 + [None] * 40
        df = pd.DataFrame({"sparse": vals, "v": range(100)})
        profile = profile_dataframe(df)
        assert "high_missing" in _col(profile, "sparse").flags
        assert any("missing" in w for w in profile.warnings)

    def test_duplicates_counted(self) -> None:
        df = pd.DataFrame({"a": [1, 1, 2], "b": ["x", "x", "y"]})
        profile = profile_dataframe(df)
        assert profile.duplicate_rows == 1
        assert any("duplicate" in w for w in profile.warnings)

    def test_class_imbalance_warning(self) -> None:
        df = pd.DataFrame({
            "f": RNG.normal(0, 1, 300),
            "fraud": [1] * 6 + [0] * 294,
        })
        profile = profile_dataframe(df, target_column="fraud")
        assert any("imbalanced" in w for w in profile.warnings)

    def test_small_dataset_warning(self) -> None:
        df = pd.DataFrame({"a": range(20), "b": RNG.normal(0, 1, 20)})
        assert any("rows" in w for w in profile_dataframe(df).warnings)


class TestQualityScore:
    def test_clean_data_scores_high(self, clean_df: pd.DataFrame) -> None:
        assert profile_dataframe(clean_df).quality_score >= 85

    def test_dirty_data_scores_lower_than_clean(self, clean_df: pd.DataFrame) -> None:
        dirty = clean_df.copy()
        dirty.loc[dirty.index[:80], "income"] = None
        dirty["useless"] = 1
        dirty = pd.concat([dirty, dirty.head(30)], ignore_index=True)
        assert profile_dataframe(dirty).quality_score < profile_dataframe(clean_df).quality_score

    def test_score_bounded(self) -> None:
        df = pd.DataFrame({"a": [None, None, 1]})
        score = profile_dataframe(df).quality_score
        assert 0 <= score <= 100


class TestPromptString:
    def test_prompt_string_mentions_score_and_ids(self, clean_df: pd.DataFrame) -> None:
        text = profile_dataframe(clean_df).to_prompt_string()
        assert "quality score" in text
        assert "customer_id" in text  # identifier exclusion advice

    def test_to_dict_round_trips_json(self, clean_df: pd.DataFrame) -> None:
        import json
        payload = json.dumps(profile_dataframe(clean_df).to_dict(), default=str)
        assert json.loads(payload)["row_count"] == 200

    def test_columns_of_kind_filter(self, clean_df: pd.DataFrame) -> None:
        profile = profile_dataframe(clean_df)
        numeric_names = [c.name for c in profile.columns_of_kind("numeric")]
        assert "income" in numeric_names
        assert "customer_id" not in numeric_names


class TestTransactionValueSemanticDisambiguation:
    def test_unit_rate_and_transaction_value_predicates(self) -> None:
        from src.core.profiler import is_transaction_value_name, is_unit_rate_name

        assert is_transaction_value_name("total_amount")
        assert is_transaction_value_name("revenue")
        assert is_transaction_value_name("sales")
        assert is_transaction_value_name("total_price")
        assert not is_transaction_value_name("unit_price")
        assert not is_transaction_value_name("price_per_unit")

        assert is_unit_rate_name("unit_price")
        assert is_unit_rate_name("price")
        assert is_unit_rate_name("rate")
        assert is_unit_rate_name("price_per_unit")
        assert not is_unit_rate_name("total_amount")
        assert not is_unit_rate_name("revenue")

        # Compound / phrase token disambiguation and non-pricing rate exclusions
        assert is_transaction_value_name("extended_price") is True
        assert is_unit_rate_name("extended_price") is False
        assert is_transaction_value_name("order_value") is True
        assert is_unit_rate_name("heart_rate") is False

    def test_column_profile_flags_and_aggregation(self) -> None:
        from src.core.profiler import profile_dataframe

        df = pd.DataFrame({
            "unit_price": [10.0, 15.0, 20.0],
            "total_amount": [100.0, 150.0, 200.0],
        })
        profile = profile_dataframe(df)
        up_col = profile.columns[0]
        ta_col = profile.columns[1]

        assert "unit_rate" in up_col.flags
        assert up_col.is_unit_rate()
        assert up_col.aggregation == "mean"

        assert "transaction_value" in ta_col.flags
        assert ta_col.is_transaction_value()
        assert ta_col.aggregation == "sum"

    def test_pick_measures_prioritizes_transaction_value(self) -> None:
        from src.core.profiler import pick_measures, profile_dataframe

        df = pd.DataFrame({
            "unit_price": [10.0, 15.0, 20.0],
            "general_metric": [5.0, 6.0, 7.0],
            "total_amount": [100.0, 150.0, 200.0],
        })
        profile = profile_dataframe(df)
        ordered = pick_measures(profile)
        names = [c.name for c in ordered]
        assert names[0] == "total_amount"
        assert names[-1] == "unit_price"
