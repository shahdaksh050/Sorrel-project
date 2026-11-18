"""Universal Data Archetypes Integration Test Matrix (All_ArchetypePlan §5 & §6).

Tests deterministic classification, automated reshaping, agenda question generation,
and domain pack evaluation across all 10 canonical data archetypes:
1. Transactional Event Stream
2. Continuous Time-Series & Signals
3. Panel / Longitudinal Repeated
4. Cross-Sectional Entity Snapshot
5. Survival / Censored Duration
6. Survey / Psychometric Ordinal
7. Market Basket / Combinatorial Sets
8. Network / Graph / Relational
9. Geospatial / Spatial Field
10. Hybrid Text-Tabular
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.core.agenda import build_agenda
from src.core.domain_packs import detect_domain_pack, evaluate_domain_pack
from src.core.profiler import (
    ARCHETYPE_BASKET,
    ARCHETYPE_CONTINUOUS_SERIES,
    ARCHETYPE_CROSS_SECTIONAL,
    ARCHETYPE_GEOSPATIAL,
    ARCHETYPE_HYBRID_TEXT,
    ARCHETYPE_NETWORK,
    ARCHETYPE_PANEL,
    ARCHETYPE_SURVEY,
    ARCHETYPE_SURVIVAL,
    ARCHETYPE_TRANSACTIONAL,
    DatasetArchetype,
    profile_dataframe,
)
from src.core.reshaping import (
    auto_reshape_for_archetype,
    extract_text_sentiment_and_features,
)

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def test_reference_air_quality_dataset_classification() -> None:
    csv_path = DATA_DIR / "AirQualityUCI.csv"
    if not csv_path.exists():
        pytest.skip("AirQualityUCI.csv not found")
    df = pd.read_csv(csv_path, sep=";", decimal=",", nrows=200)
    # Filter empty or trailing delimiter columns
    df = df.dropna(how="all", axis=1)
    # Combine Date & Time into timestamp
    if "Date" in df.columns and "Time" in df.columns:
        df["timestamp"] = pd.to_datetime(df["Date"] + " " + df["Time"], format="%d/%m/%Y %H.%M.%S", errors="coerce")
        df = df.dropna(subset=["timestamp"])
    profile = profile_dataframe(df)
    assert profile.archetype == ARCHETYPE_CONTINUOUS_SERIES
    questions = build_agenda(profile, {"mode": "describe"})
    assert any(q.kind == "trend" for q in questions)


def test_reference_churn_dataset_classification() -> None:
    csv_path = DATA_DIR / "sample_customer_churn.csv"
    if not csv_path.exists():
        pytest.skip("sample_customer_churn.csv not found")
    df = pd.read_csv(csv_path, nrows=300)
    profile = profile_dataframe(df)
    assert profile.archetype == ARCHETYPE_SURVIVAL
    questions = build_agenda(profile, {"mode": "describe"})
    assert any(q.kind == "survival" for q in questions)

    # Verify SaaS domain pack activates
    pack = detect_domain_pack(df, objective="Predict customer churn and analyze MRR")
    assert pack is not None
    assert pack.name == "saas_subscription"
    findings = evaluate_domain_pack(df, pack=pack)
    assert any("churn rate" in f.headline.lower() for f in findings)


def test_archetype_1_transactional_pipeline() -> None:
    # Granularity: row-per-event with irregular transaction timestamps
    timestamps = [
        pd.Timestamp("2024-01-01 08:14:00") + pd.Timedelta(minutes=int(i * 17 + (i % 5) * 3))
        for i in range(60)
    ]
    records = []
    for i in range(60):
        records.append({
            "order_id": f"ORD_{i:04d}",
            "customer_id": f"CUST_{i % 8:02d}",  # repeats across customers
            "timestamp": timestamps[i],
            "amount": round(float(np.random.uniform(20.0, 250.0)), 2),
            "quantity": np.random.randint(1, 5),
        })
    df = pd.DataFrame(records)
    profile = profile_dataframe(df)
    assert profile.archetype == ARCHETYPE_TRANSACTIONAL

    # Reshaping: RFM aggregation
    rfm, notes = auto_reshape_for_archetype(df, DatasetArchetype.TRANSACTIONAL, aggregate_granularity=True)
    assert len(rfm) == 8
    assert "monetary_total" in rfm.columns
    assert len(notes) > 0

    # Agenda question generation
    questions = build_agenda(profile, {"mode": "describe"})
    assert any(q.kind == "concentration" for q in questions)


def test_archetype_2_continuous_series_pipeline() -> None:
    dates = pd.date_range("2024-01-01", periods=50, freq="D")
    df = pd.DataFrame({
        "timestamp": dates,
        "asset_price": [100.0 * (1.008 ** i) for i in range(50)],
        "volume": np.random.randint(1000, 5000, 50),
        "temperature": np.random.normal(22.0, 3.0, 50),
    })
    profile = profile_dataframe(df)
    assert profile.archetype == ARCHETYPE_CONTINUOUS_SERIES

    # Reshaping: Log returns and volatility
    reshaped, notes = auto_reshape_for_archetype(df, DatasetArchetype.CONTINUOUS_SERIES)
    assert "asset_price_log_return" in reshaped.columns
    assert "asset_price_rolling_vol_30" in reshaped.columns
    assert len(notes) > 0

    # Agenda: Trend & seasonality
    questions = build_agenda(profile, {"mode": "describe"})
    assert any(q.kind == "trend" for q in questions)


def test_archetype_3_panel_longitudinal_pipeline() -> None:
    dates = pd.date_range("2024-01-01", periods=8, freq="W")
    records = []
    for store in ["Store_1", "Store_2", "Store_3", "Store_4"]:
        for d in dates:
            records.append({
                "store_id": store,
                "week_date": d,
                "weekly_sales": np.random.uniform(20000, 80000),
                "foot_traffic": np.random.randint(1000, 5000),
            })
    df = pd.DataFrame(records)
    profile = profile_dataframe(df)
    assert profile.archetype == ARCHETYPE_PANEL

    questions = build_agenda(profile, {"mode": "describe"})
    assert any(q.kind in ("trend", "mixed_model") for q in questions)


def test_archetype_4_cross_sectional_pipeline() -> None:
    df = pd.DataFrame({
        "employee_id": [f"EMP_{i:04d}" for i in range(80)],
        "department": np.random.choice(["Engineering", "Sales", "Support", "Marketing"], size=80),
        "salary": np.random.uniform(60000, 160000, 80),
        "performance_rating": np.random.choice([1, 2, 3, 4, 5], size=80),
        "years_at_company": np.random.randint(1, 15, 80),
    })
    profile = profile_dataframe(df)
    assert profile.archetype == ARCHETYPE_CROSS_SECTIONAL

    questions = build_agenda(profile, {"mode": "describe"})
    assert any(q.kind == "segment" for q in questions)


def test_archetype_5_survival_pipeline() -> None:
    df = pd.DataFrame({
        "account_id": [f"ACC_{i:04d}" for i in range(60)],
        "tenure_months": np.random.randint(1, 60, 60),
        "churn": np.random.choice([0, 1], size=60, p=[0.75, 0.25]),
        "contract_type": np.random.choice(["Monthly", "OneYear", "TwoYear"], size=60),
        "monthly_spend": np.random.uniform(30.0, 150.0, 60),
    })
    profile = profile_dataframe(df)
    assert profile.archetype == ARCHETYPE_SURVIVAL

    questions = build_agenda(profile, {"mode": "describe"})
    assert any(q.kind == "survival" for q in questions)


def test_archetype_6_survey_ordinal_pipeline() -> None:
    df = pd.DataFrame({
        "respondent_id": [f"RESP_{i:03d}" for i in range(60)],
        "q1_product_satisfaction": np.random.randint(1, 6, 60),
        "q2_support_responsiveness": np.random.randint(1, 6, 60),
        "q3_pricing_fairness": np.random.randint(1, 6, 60),
        "q4_net_promoter_intent": np.random.randint(1, 6, 60),
    })
    profile = profile_dataframe(df)
    assert profile.archetype == ARCHETYPE_SURVEY

    questions = build_agenda(profile, {"mode": "describe"})
    assert any(q.kind in ("distribution", "relationship") for q in questions)


def test_archetype_7_market_basket_pipeline() -> None:
    records = []
    items = ["Milk", "Bread", "Eggs", "Cereal", "Yogurt", "Coffee", "Tea"]
    for basket_idx in range(25):
        basket_items = np.random.choice(items, size=np.random.randint(2, 5), replace=False)
        for itm in basket_items:
            records.append({
                "basket_id": f"BSK_{basket_idx:03d}",
                "item_name": itm,
                "item_qty": 1,
            })
    df = pd.DataFrame(records)
    profile = profile_dataframe(df)
    assert profile.archetype == ARCHETYPE_BASKET

    incidence, notes = auto_reshape_for_archetype(df, DatasetArchetype.BASKET, aggregate_granularity=True)
    assert len(incidence) == 25
    assert len(notes) > 0

    questions = build_agenda(profile, {"mode": "describe"})
    assert any(q.kind == "association" for q in questions)


def test_archetype_8_network_graph_pipeline() -> None:
    df = pd.DataFrame({
        "source_id": [f"Router_{i % 6}" for i in range(50)],
        "target_id": [f"Router_{(i * 2 + 1) % 6}" for i in range(50)],
        "bandwidth_gbps": np.random.uniform(1.0, 10.0, 50),
        "packet_loss_pct": np.random.uniform(0.0, 0.02, 50),
    })
    profile = profile_dataframe(df)
    assert profile.archetype == ARCHETYPE_NETWORK

    questions = build_agenda(profile, {"mode": "describe"})
    assert any(q.suggested_tool == "graph_analysis" for q in questions)


def test_archetype_9_geospatial_pipeline() -> None:
    df = pd.DataFrame({
        "ride_id": [f"RIDE_{i:04d}" for i in range(50)],
        "pickup_latitude": np.random.uniform(40.71, 40.78, 50),
        "pickup_longitude": np.random.uniform(-74.01, -73.95, 50),
        "fare_amount": np.random.uniform(10.0, 60.0, 50),
    })
    profile = profile_dataframe(df)
    assert profile.archetype == ARCHETYPE_GEOSPATIAL

    questions = build_agenda(profile, {"mode": "describe"})
    assert any(q.suggested_tool == "cluster_data" for q in questions)


def test_archetype_10_hybrid_text_pipeline() -> None:
    comments = [
        "The software architecture is well thought out, fast and highly extensible for data science.",
        "Constant crashes during large file exports, completely unreliable and lacks customer support.",
        "Decent user interface but sluggish when loading large dashboards with complex visualizations.",
        "Exceptional performance, smooth animations, and clean type-safe python implementation throughout.",
        "Terrible documentation, ambiguous error messages and very confusing navigation across views.",
    ]
    df = pd.DataFrame({
        "ticket_id": [f"TCK_{i:03d}" for i in range(50)],
        "feedback_text": [comments[i % len(comments)] for i in range(50)],
        "csat_score": np.random.choice([1, 2, 3, 4, 5], size=50),
        "resolution_hours": np.random.uniform(0.5, 24.0, 50),
    })
    profile = profile_dataframe(df)
    assert profile.archetype == ARCHETYPE_HYBRID_TEXT

    reshaped, notes = extract_text_sentiment_and_features(df, "feedback_text")
    assert "feedback_text_sentiment_score" in reshaped.columns
    assert "feedback_text_word_count" in reshaped.columns
    assert len(notes) > 0

    questions = build_agenda(profile, {"mode": "describe"})
    assert any(q.suggested_tool == "correlation_analysis" for q in questions)


def test_safe_quantitative_routing_archetype_guards(tmp_path: Path) -> None:
    from src.core.controller import AgentController

    # 1. Survival dataset with churn target: regression_analysis must be rejected
    df_surv = pd.DataFrame({
        "account_id": [f"ACC_{i}" for i in range(50)],
        "tenure_months": [10.0 + i for i in range(50)],
        "churn": [0] * 40 + [1] * 10,
        "spend": [50.0 + i * 2 for i in range(50)],
    })
    csv_surv = tmp_path / "surv.csv"
    df_surv.to_csv(csv_surv, index=False)

    agent = AgentController(min_iterations=1, max_iterations=1, enable_rlm=False, use_llm=False)
    agent.load_dataset(str(csv_surv), target_hint="churn", interactive=False)

    err = agent._tool_hard_prerequisite_error("regression_analysis")
    assert err is not None
    assert "survivorship bias" in err
    assert "survival_analysis" in err

    # 2. Continuous time series with raw price (non-stationary): regression_analysis must be rejected
    dates = pd.date_range("2024-01-01", periods=40, freq="D")
    df_series = pd.DataFrame({
        "timestamp": dates,
        "stock_price": [100.0 * (1.01 ** i) for i in range(40)],
        "index_level": [200.0 * (1.005 ** i) for i in range(40)],
    })
    csv_series = tmp_path / "series.csv"
    df_series.to_csv(csv_series, index=False)

    agent_series = AgentController(min_iterations=1, max_iterations=1, enable_rlm=False, use_llm=False)
    agent_series.load_dataset(str(csv_series), interactive=False)

    err_series = agent_series._tool_hard_prerequisite_error("regression_analysis")
    assert err_series is not None
    assert "spurious correlation" in err_series

