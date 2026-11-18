"""Unit tests for deterministic structural archetype classification (All_ArchetypePlan §4.1)."""
from __future__ import annotations

import numpy as np
import pandas as pd

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
    DatasetArchetype,
    profile_dataframe,
)


def test_archetype_enum_members() -> None:
    expected = {
        "transactional", "continuous_series", "panel", "cross_sectional",
        "survival", "survey", "basket", "network", "geospatial", "hybrid_text",
    }
    actual = {a.value for a in DatasetArchetype}
    assert actual == expected


def test_geospatial_archetype_detection() -> None:
    df = pd.DataFrame({
        "location_id": [f"LOC_{i:03d}" for i in range(50)],
        "latitude": np.random.uniform(37.7, 37.8, 50),
        "longitude": np.random.uniform(-122.5, -122.4, 50),
        "delay_minutes": np.random.uniform(5.0, 45.0, 50),
    })
    profile = profile_dataframe(df)
    assert profile.archetype == ARCHETYPE_GEOSPATIAL
    assert any("spatial coordinate" in e.lower() for e in profile.archetype_evidence)


def test_network_archetype_detection() -> None:
    df = pd.DataFrame({
        "source_id": [f"N_{i % 10}" for i in range(60)],
        "target_id": [f"N_{(i * 3 + 1) % 10}" for i in range(60)],
        "latency_ms": np.random.uniform(10.0, 100.0, 60),
        "packet_loss": np.random.uniform(0.0, 0.05, 60),
    })
    profile = profile_dataframe(df)
    assert profile.archetype == ARCHETYPE_NETWORK
    assert any("graph edge" in e.lower() for e in profile.archetype_evidence)


def test_survival_archetype_detection() -> None:
    df = pd.DataFrame({
        "customer_id": [f"C_{i:03d}" for i in range(80)],
        "tenure_months": np.random.randint(1, 72, 80),
        "is_churn": np.random.choice([0, 1], size=80, p=[0.7, 0.3]),
        "monthly_charges": np.random.uniform(20.0, 120.0, 80),
    })
    profile = profile_dataframe(df)
    assert profile.archetype == ARCHETYPE_SURVIVAL
    assert any("duration/tenure" in e.lower() for e in profile.archetype_evidence)


def test_basket_archetype_detection() -> None:
    # 30 orders across 100 rows -> avg > 3 items per basket
    df = pd.DataFrame({
        "order_id": [f"ORD_{i // 3:03d}" for i in range(90)],
        "item_name": np.random.choice(["Milk", "Bread", "Eggs", "Coffee", "Butter", "Cheese"], size=90),
        "quantity": np.random.randint(1, 5, 90),
        "price": np.random.uniform(1.5, 9.99, 90),
    })
    profile = profile_dataframe(df)
    assert profile.archetype == ARCHETYPE_BASKET
    assert any("basket" in e.lower() for e in profile.archetype_evidence)


def test_continuous_series_archetype_detection() -> None:
    dates = pd.date_range("2024-01-01", periods=60, freq="D")
    df = pd.DataFrame({
        "timestamp": dates,
        "co_sensor": np.random.normal(1.2, 0.3, 60),
        "no2_sensor": np.random.normal(45.0, 10.0, 60),
        "temperature": np.random.normal(20.0, 5.0, 60),
    })
    profile = profile_dataframe(df)
    assert profile.archetype == ARCHETYPE_CONTINUOUS_SERIES
    assert any("axis" in e.lower() for e in profile.archetype_evidence)


def test_panel_archetype_detection() -> None:
    dates = pd.date_range("2024-01-01", periods=10, freq="W")
    records = []
    for store in ["Store_A", "Store_B", "Store_C", "Store_D"]:
        for d in dates:
            records.append({
                "store_id": store,
                "date": d,
                "weekly_sales": np.random.uniform(10000, 50000),
                "foot_traffic": np.random.randint(500, 2000),
            })
    df = pd.DataFrame(records)
    profile = profile_dataframe(df)
    assert profile.archetype == ARCHETYPE_PANEL


def test_survey_archetype_detection() -> None:
    df = pd.DataFrame({
        "respondent_id": [f"RESP_{i:03d}" for i in range(50)],
        "q1_satisfaction": np.random.randint(1, 6, 50),
        "q2_recommend": np.random.randint(1, 6, 50),
        "q3_value": np.random.randint(1, 6, 50),
        "q4_support": np.random.randint(1, 6, 50),
    })
    profile = profile_dataframe(df)
    assert profile.archetype == ARCHETYPE_SURVEY
    assert any("rating" in e.lower() or "scale" in e.lower() for e in profile.archetype_evidence)


def test_hybrid_text_archetype_detection() -> None:
    reviews = [
        "The battery life is exceptional and lasted two full days on a single charge under heavy workload.",
        "Terrible build quality, the screen began flickering after two weeks and customer service refused replacement.",
        "Good value for money overall but the software UI feels clunky and unintuitive compared to competitors.",
        "Outstanding performance for 4k video editing without any thermal throttling or fan noise during export.",
        "Average experience, does what it advertises but nothing particularly impressive or groundbreaking here.",
    ]
    df = pd.DataFrame({
        "review_id": [f"REV_{i:03d}" for i in range(50)],
        "review_text": [reviews[i % len(reviews)] for i in range(50)],
        "star_rating": np.random.choice([1.0, 2.0, 3.0, 4.0, 5.0], size=50),
        "upvotes": np.random.randint(0, 100, 50),
    })
    profile = profile_dataframe(df)
    assert profile.archetype == ARCHETYPE_HYBRID_TEXT
    assert any("free-text" in e.lower() or "text" in e.lower() for e in profile.archetype_evidence)


def test_cross_sectional_archetype_detection() -> None:
    df = pd.DataFrame({
        "employee_id": [f"EMP_{i:04d}" for i in range(60)],
        "department": np.random.choice(["Engineering", "Sales", "Marketing", "HR"], size=60),
        "salary": np.random.uniform(50000, 150000, 60),
        "performance_score": np.random.uniform(1.0, 5.0, 60),
        "years_experience": np.random.randint(1, 25, 60),
    })
    profile = profile_dataframe(df)
    assert profile.archetype == ARCHETYPE_CROSS_SECTIONAL
