"""
Unit tests for Target Leakage Guard (Phase 12).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.core.model_io import load_model
from src.core.reshaping import aggregate_point_in_time_event_features
from src.core.sensitivity import detect_target_leakage
from src.tools.ml_common import _prepare_features
from src.tools.ml_pipeline import TrainModelTool


def test_target_leakage_detection() -> None:
    df = pd.DataFrame({
        "customer_id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "churn": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        "churn_is_churned": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],          # semantic leakage
        "post_cancellation_survey": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],  # temporal prefix leakage
        "almost_perfect_leak": [0.01, 0.99, 0.02, 0.98, 0.03, 0.97, 0.01, 0.99, 0.02, 0.98], # r > 0.99
        "normal_feature": [15, 42, 33, 21, 55, 60, 12, 80, 25, 40], # benign
    })

    alerts = detect_target_leakage(df, target_col="churn")

    leaked_cols = {a["column"]: a["leakage_type"] for a in alerts}
    assert "churn_is_churned" in leaked_cols
    assert "post_cancellation_survey" in leaked_cols
    assert "almost_perfect_leak" in leaked_cols
    assert "normal_feature" not in leaked_cols


def test_point_in_time_event_leakage_guard() -> None:
    # Event stream of customer transactions over time
    df = pd.DataFrame({
        "client_id": ["U1", "U1", "U1", "U2", "U2", "U3"],
        "event_time": ["2024-01-01", "2024-03-01", "2024-06-01", "2024-02-01", "2024-04-01", "2024-05-01"],
        "amount": [10.0, 20.0, 30.0, 15.0, 25.0, 50.0],
    })
    out, notes = aggregate_point_in_time_event_features(df, cust_col="client_id", date_col="event_time")

    # For customer U1: 3 total transactions
    u1_events = out[out["client_id"] == "U1"].sort_values("event_time")
    # Row 1 (2024-01-01) must have prior_event_count == 0 (no future events known!)
    assert u1_events["cust_prior_event_count"].tolist() == [0, 1, 2]
    # Retrospective summary preserves lifetime frequency (3 for U1)
    assert u1_events["cust_lifetime_frequency"].tolist() == [3, 3, 3]
    # Lookahead guard note is present
    assert any("preventing lookahead leakage" in n for n in notes)


def test_prior_event_count_uses_only_strictly_earlier_known_events() -> None:
    events = pd.DataFrame({
        "customer": ["A", "A", "A", "A", None, "B"],
        "date": ["2025-01-02", "2025-01-01T12:00:00Z", "2025-01-01T07:00:00-05:00",
                 "invalid", "2025-01-03", "2025-01-02"],
    }, index=[4, 3, 3, 9, 8, 2])
    out, _ = aggregate_point_in_time_event_features(events, "customer", "date")
    expected = pd.Series([2, 0, 0, pd.NA, pd.NA, 0], index=events.index,
                         dtype="Int64", name="cust_prior_event_count")
    pd.testing.assert_series_equal(out["cust_prior_event_count"], expected)


def test_future_transactions_do_not_change_predictors_or_enter_saved_model(tmp_path: Path) -> None:
    records = [
        (f"C{i}", pd.Timestamp("2025-01-01") + pd.Timedelta(days=j), (i + j) % 4, (i + j) % 2)
        for i in range(30) for j in range(2 + i % 5)
    ]
    events = pd.DataFrame(records, columns=["customer", "date", "visits", "label"])
    future = events.iloc[:32].assign(date=pd.Timestamp("2025-03-01"))
    old, _ = aggregate_point_in_time_event_features(events, "customer", "date")
    new, _ = aggregate_point_in_time_event_features(pd.concat([events, future], ignore_index=True), "customer", "date")
    descriptive = ["cust_lifetime_frequency", "cust_event_frequency"]
    assert old.iloc[0]["cust_lifetime_frequency"] != new.iloc[0]["cust_lifetime_frequency"]
    for table in (old, new):
        for col in descriptive:
            table[f"{col}_quartile"] = pd.qcut(table[col], q=4, labels=False, duplicates="drop")
    old_x, _, _ = _prepare_features(old, "label")
    new_x, _, _ = _prepare_features(new, "label")
    pd.testing.assert_frame_equal(old_x, new_x.iloc[:len(old_x)])

    path = tmp_path / "transactions.csv"
    new.to_csv(path, index=False)
    run = TrainModelTool().run(file_path=str(path), target_column="label", models=["logistic_regression"],
                               output_dir=str(tmp_path / "models"))
    assert run.status == "success", run.error
    saved = load_model(run.output["models_trained"]["logistic_regression"]["model_path"])
    names = set(saved.feature_names_in_)
    assert "cust_prior_event_count" in names
    assert set(descriptive + [f"{c}_quartile" for c in descriptive]).isdisjoint(names)
