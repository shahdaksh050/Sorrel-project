"""Unit tests for automated pre-modeling reshaping pipeline (All_ArchetypePlan §4.2)."""
from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd

from src.core.profiler import DatasetArchetype
from src.core.reshaping import (
    aggregate_customer_rfm,
    aggregate_point_in_time_event_features,
    assess_survey_reliability_and_polarity,
    auto_reshape_for_archetype,
    compute_geospatial_features,
    compute_log_returns_and_volatility,
    compute_panel_deltas_and_lags,
    detect_and_strip_summary_rows,
    detect_wide_temporal_columns,
    extract_text_sentiment_and_features,
    pivot_basket_incidence_matrix,
    unpivot_temporal_frame,
    verify_and_prepare_survival_data,
)
from src.tools.data_processing import CleanDataTool


def test_detect_and_unpivot_wide_temporal_columns() -> None:
    df_wide = pd.DataFrame({
        "country": ["USA", "Germany", "Japan"],
        "2018": [100, 200, 300],
        "2019": [110, 210, 310],
        "2020": [120, 220, 320],
        "2021": [130, 230, 330],
    })
    temporal = detect_wide_temporal_columns(list(df_wide.columns))
    assert len(temporal) == 4

    unpivoted = unpivot_temporal_frame(df_wide, temporal, var_name="year", value_name="gdp")
    assert "year" in unpivoted.columns
    assert "gdp" in unpivoted.columns
    assert len(unpivoted) == 12
    assert set(unpivoted["country"]) == {"USA", "Germany", "Japan"}


def test_detect_and_strip_summary_rows() -> None:
    categories = [f"Category_{i}" for i in range(19)] + ["Total"]
    revenues = [100 * (i + 1) for i in range(19)] + [19000]
    df = pd.DataFrame({"category": categories, "revenue": revenues})
    cleaned, notes = detect_and_strip_summary_rows(df)
    assert len(cleaned) == 19
    assert "Total" not in cleaned["category"].values
    assert len(notes) > 0


def test_compute_log_returns_and_volatility() -> None:
    dates = pd.date_range("2024-01-01", periods=40, freq="D")
    prices = [100.0 * (1.01 ** i) for i in range(40)]
    df = pd.DataFrame({"date": dates, "close_price": prices})

    out, notes = compute_log_returns_and_volatility(df, price_cols=["close_price"])
    assert "close_price_log_return" in out.columns
    assert "close_price_rolling_vol_30" in out.columns
    assert len(notes) == 1
    # Check that log returns are approx ln(1.01) = 0.00995
    valid_rets = out["close_price_log_return"].dropna()
    assert len(valid_rets) == 39
    assert np.isclose(valid_rets.iloc[0], np.log(1.01), atol=1e-4)


def test_aggregate_customer_rfm() -> None:
    records = [
        {"customer_id": "C1", "order_date": "2024-01-01", "spend": 50.0},
        {"customer_id": "C1", "order_date": "2024-01-15", "spend": 100.0},
        {"customer_id": "C2", "order_date": "2024-01-10", "spend": 200.0},
        {"customer_id": "C3", "order_date": "2024-01-20", "spend": 30.0},
    ]
    df = pd.DataFrame(records)
    rfm, notes = aggregate_customer_rfm(df, "customer_id", "order_date", "spend")

    assert len(rfm) == 3
    assert set(rfm.columns) == {"customer_id", "recency_days", "frequency", "monetary_total", "monetary_avg"}
    c1 = rfm[rfm["customer_id"] == "C1"].iloc[0]
    assert c1["frequency"] == 2
    assert c1["monetary_total"] == 150.0
    assert c1["monetary_avg"] == 75.0
    assert len(notes) == 1


def test_pivot_basket_incidence_matrix() -> None:
    records = [
        {"order_id": "O1", "product": "Apple"},
        {"order_id": "O1", "product": "Banana"},
        {"order_id": "O2", "product": "Apple"},
        {"order_id": "O2", "product": "Carrot"},
        {"order_id": "O3", "product": "Banana"},
    ]
    df = pd.DataFrame(records)
    incidence, notes = pivot_basket_incidence_matrix(df, "order_id", "product")

    assert len(incidence) == 3
    assert "Apple" in incidence.columns
    assert "Banana" in incidence.columns
    assert "Carrot" in incidence.columns
    assert incidence.loc[incidence["order_id"] == "O1", "Apple"].iloc[0] == 1
    assert incidence.loc[incidence["order_id"] == "O3", "Apple"].iloc[0] == 0
    assert len(notes) == 1


def test_extract_text_sentiment_and_features() -> None:
    df = pd.DataFrame({
        "id": [1, 2],
        "feedback": [
            "Great product, excellent quality and fast delivery. Absolutely love it!",
            "Terrible customer service, broken device and horrible slow response.",
        ],
    })
    out, notes = extract_text_sentiment_and_features(df, "feedback")

    assert "feedback_sentiment_score" in out.columns
    assert "feedback_word_count" in out.columns
    assert "feedback_char_len" in out.columns
    assert out.loc[0, "feedback_sentiment_score"] > 0
    assert out.loc[1, "feedback_sentiment_score"] < 0
    assert len(notes) == 1


def test_auto_reshape_for_archetype() -> None:
    # Test continuous series with prices
    dates = pd.date_range("2024-01-01", periods=30, freq="D")
    df_prices = pd.DataFrame({"timestamp": dates, "asset_price": [50.0 + i for i in range(30)]})
    out, notes = auto_reshape_for_archetype(df_prices, DatasetArchetype.CONTINUOUS_SERIES)
    assert "asset_price_log_return" in out.columns
    assert any("stationary log returns" in n for n in notes)


def test_compute_log_returns_with_nan_and_zeros() -> None:
    # Column with a missing value and valid positive prices
    prices = [100.0, np.nan, 102.0, 105.0, 103.0, 107.0, 110.0]
    df = pd.DataFrame({"close": prices})
    out, notes = compute_log_returns_and_volatility(df)
    assert "close_log_return" in out.columns
    assert len(notes) > 0


def test_extract_text_sentiment_filters_stopwords() -> None:
    df = pd.DataFrame({
        "feedback": [
            "The customer service was fast and the response was great for the product",
            "The customer service was slow and the response was terrible for the product",
        ],
    })
    out, _ = extract_text_sentiment_and_features(df, "feedback")
    cols = list(out.columns)
    # Stop words must never be keyword features
    assert "feedback_has_the" not in cols
    assert "feedback_has_was" not in cols
    assert "feedback_has_and" not in cols
    assert "feedback_has_for" not in cols
    # Meaningful domain terms are extracted
    assert any("feedback_has_service" in c or "feedback_has_customer" in c for c in cols)


def test_verify_and_prepare_survival_data() -> None:
    df = pd.DataFrame({
        "tenure": [-5.0, 0.0, 12.0, 24.0, 36.0],
        "churn": ["yes", "no", "yes", "no", "no"],
    })
    out, notes = verify_and_prepare_survival_data(df)
    # Row 0 with -5.0 is excluded, row 1 with 0.0 is preserved as a valid time-zero event
    assert (out["tenure"] >= 0).all()
    assert 0.0 in out["tenure"].values
    assert -5.0 not in out["tenure"].values
    assert len(out) == 4
    assert out["churn"].tolist() == [0, 1, 0, 0]
    assert any("excluded due to negative duration" in n for n in notes)
    assert any("censoring rate" in n for n in notes)


def test_survival_cleaning_unknown_labels_missing_and_mapping() -> None:
    df = pd.DataFrame({
        "time": [-10.0, 0.0, 5.0, 15.0, 25.0, 30.0],
        "status": ["yes", "dead", "unknown", None, "alive", "lost"],
    })
    out, notes = verify_and_prepare_survival_data(df, duration_col="time", event_col="status")
    # -10 is excluded (negative)
    # "unknown", None, "lost" are unmapped/unknown status -> excluded
    # Only rows with valid non-negative time and recognized event ("dead"->1, "alive"->0) remain:
    # row 1: time=0.0, status="dead" -> 1
    # row 4: time=25.0, status="alive" -> 0
    assert len(out) == 2
    assert out["time"].tolist() == [0.0, 25.0]
    assert out["status"].tolist() == [1, 0]
    assert any("4 excluded" in n for n in notes)

    # Test numeric 1/2 codes without and with event_mapping
    df_med = pd.DataFrame({
        "months": [10.0, 20.0, 30.0, 40.0],
        "status_code": [1, 2, 1, 2],
    })
    # Without mapping -> codes {1, 2} require explicit mapping; rows excluded
    out_unmapped, notes_unmapped = verify_and_prepare_survival_data(df_med, duration_col="months", event_col="status_code")
    assert len(out_unmapped) == 0
    assert any("requires explicit event_mapping" in n for n in notes_unmapped)

    # With mapping -> {1: 0, 2: 1} maps cleanly
    out_mapped, notes_mapped = verify_and_prepare_survival_data(
        df_med, duration_col="months", event_col="status_code", event_mapping={1: 0, 2: 1}
    )
    assert len(out_mapped) == 4
    assert out_mapped["status_code"].tolist() == [0, 1, 0, 1]
    assert any("4 records valid in risk set" in n for n in notes_mapped)


def test_assess_survey_reliability_and_polarity() -> None:
    # 1-5 instrument with responses restricted to 2-4 and one reverse-coded item
    df = pd.DataFrame({
        "q1_satisfaction": [4, 4, 3, 4, 3, 4, 4, 3, 4, 4],
        "q2_support": [4, 3, 3, 4, 4, 3, 4, 3, 4, 4],
        "q3_difficult_navigation": [2, 3, 3, 2, 2, 3, 2, 3, 2, 2],  # negative question (aligned: 6-x)
        "q4_recommend": [4, 4, 3, 3, 4, 4, 4, 3, 4, 4],
    })
    out, notes = assess_survey_reliability_and_polarity(
        df,
        declared_endpoints=(1.0, 5.0),
        scoring_key={"q3_difficult_navigation": -1},
    )
    assert "q3_difficult_navigation_rev" in out.columns
    # 6 - x with endpoints (1.0, 5.0)
    assert out["q3_difficult_navigation_rev"].tolist() == [4, 3, 3, 4, 4, 3, 4, 3, 4, 4]
    assert any("Reverse-coded negative question" in n for n in notes)
    # Cronbach's alpha must be positive and report complete responses count
    alpha_note = next(n for n in notes if "Cronbach's alpha" in n)
    assert "alpha = " in alpha_note
    assert "n=10 complete responses" in alpha_note
    # Parse alpha
    match = re.search(r"alpha = ([0-9]+(?:\.[0-9]+)?)", alpha_note)
    assert match is not None
    assert float(match.group(1)) > 0.70


def test_aggregate_point_in_time_event_features_no_lookahead() -> None:
    # A customer transacting 3 times over 3 dates
    df = pd.DataFrame({
        "customer_id": ["C1", "C1", "C1", "C2", "C2"],
        "trans_date": ["2024-01-01", "2024-02-01", "2024-03-01", "2024-01-15", "2024-02-15"],
        "spend": [100.0, 150.0, 200.0, 50.0, 80.0],
    })
    out, notes = aggregate_point_in_time_event_features(df, "customer_id", "trans_date")
    assert "cust_prior_event_count" in out.columns
    assert "cust_lifetime_frequency" in out.columns
    # First transaction of C1 has 0 prior events
    c1_rows = out[out["customer_id"] == "C1"].sort_values("trans_date")
    assert c1_rows["cust_prior_event_count"].tolist() == [0, 1, 2]
    # Retrospective lifetime frequency reflects total events (3 for C1, 2 for C2)
    assert c1_rows["cust_lifetime_frequency"].tolist() == [3, 3, 3]
    assert any("preventing lookahead leakage" in n for n in notes)


def test_compute_geospatial_features_dateline_and_antipodal() -> None:
    # Date line cluster: points around 180 antimeridian (+179 and -179)
    df_dateline = pd.DataFrame({
        "latitude": [0.0, 0.0, 0.0, 0.0],
        "longitude": [179.0, 179.0, -179.0, -179.0],
    })
    out, notes = compute_geospatial_features(df_dateline)
    assert len(notes) > 0
    assert "dist_from_centroid_km" in out.columns
    # 1 degree from equator is ~111.19 km, NOT 19,900 km!
    dists = out["dist_from_centroid_km"].tolist()
    for d in dists:
        assert 110.0 < d < 112.5

    # Invalid coordinates (e.g. lat > 90)
    df_invalid = pd.DataFrame({
        "latitude": [105.0, 0.0],
        "longitude": [0.0, 0.0],
    })
    out_inv, notes_inv = compute_geospatial_features(df_invalid)
    assert np.isnan(out_inv["dist_from_centroid_km"].iloc[0])
    assert any("invalid coordinate rows" in n for n in notes_inv)

    # Antipodal points: resultant cancels out, center is undefined
    df_antipodal = pd.DataFrame({
        "latitude": [0.0, 0.0],
        "longitude": [0.0, 180.0],
    })
    out_anti, notes_anti = compute_geospatial_features(df_antipodal)
    assert out_anti["dist_from_centroid_km"].isna().all()
    assert any("Geographic centroid is undefined" in n for n in notes_anti)


def test_compute_panel_deltas_and_lags() -> None:
    records = []
    for store in ["S1", "S2"]:
        for t in [1, 2, 3, 4]:
            records.append({"store_id": store, "period": t, "revenue": 100.0 * t if store == "S1" else 200.0 * t})
    df = pd.DataFrame(records)
    out, notes = compute_panel_deltas_and_lags(df, entity_col="store_id", time_col="period")
    assert "revenue_lag1" in out.columns
    assert "revenue_delta1" in out.columns
    assert len(notes) > 0


def test_compute_geospatial_features() -> None:
    df = pd.DataFrame({
        "latitude": [40.71, 40.72, 40.73, 40.74],
        "longitude": [-74.00, -74.01, -73.99, -74.02],
    })
    out, notes = compute_geospatial_features(df)
    assert "dist_from_centroid_km" in out.columns
    assert (out["dist_from_centroid_km"] >= 0).all()
    assert len(notes) > 0


def test_clean_data_tool_with_archetype(tmp_path: Any) -> None:
    dates = pd.date_range("2024-01-01", periods=25, freq="D")
    df = pd.DataFrame({"date": dates, "price": [10.0 + i for i in range(25)]})
    csv_path = tmp_path / "prices.csv"
    df.to_csv(csv_path, index=False)

    tool = CleanDataTool()
    result = tool.run(
        file_path=str(csv_path),
        output_dir=str(tmp_path / "out"),
        archetype="continuous_series",
    )
    assert result.status == "success"
    cleaned_file = result.output["cleaned_file_path"]
    cleaned_df = pd.read_csv(cleaned_file)
    assert "price_log_return" in cleaned_df.columns

