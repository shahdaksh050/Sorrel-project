"""
Hidden assumption fixtures — programmatic dataset generators.

Covers the 15 hidden assumptions listed in FutureScope.md Section 8:
- Header not on row 1, blank spacer rows, footnotes.
- Subtotal and total rows mixed with unit records.
- Wide "time in the header" tables (years / t1..tn).
- Dependent rows (serial autocorrelation, clustered entities).
- Non-interval scales (compositional shares, circular hours/angles, ordinal).
- Censored values (<LOD, >max) and heaped values (digit preference on 0/5).
- Benford's law anomalies.

All factories accept a tmp_path: Path and write a file on demand.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

_RNG = np.random.default_rng(42)


def header_offset_and_notes_csv(tmp_path: Path) -> Path:
    """Assumption 1: Header is NOT on row 1; file contains title, blank lines, footnotes."""
    content = (
        "QUARTERLY REGIONAL SALES PERFORMANCE REPORT\n"
        "Confidential - Internal Use Only - Generated 2026-01-15\n"
        "\n"
        "Region,Product,Quarter,Units_Sold,Revenue\n"
        "North,Widgets,Q1,120,24000.0\n"
        "North,Gadgets,Q1,85,17000.0\n"
        "South,Widgets,Q1,140,28000.0\n"
        "South,Gadgets,Q1,90,18000.0\n"
        "East,Widgets,Q1,110,22000.0\n"
        "East,Gadgets,Q1,95,19000.0\n"
        "West,Widgets,Q1,160,32000.0\n"
        "West,Gadgets,Q1,105,21000.0\n"
        "\n"
        "* Notes: Revenue in USD before tax adjustments.\n"
        "* Data verified by Finance Operations.\n"
    )
    p = tmp_path / "header_offset.csv"
    p.write_text(content, encoding="utf-8")
    return p


def subtotals_mixed_csv(tmp_path: Path) -> Path:
    """Assumption 2: Rows are NOT all unit records; subtotals and grand totals are mixed in."""
    content = (
        "Region,Department,Headcount,Budget\n"
        "North,Engineering,25,2500000.0\n"
        "North,Sales,15,1200000.0\n"
        "North,Support,10,600000.0\n"
        "North Total,All,50,4300000.0\n"
        "South,Engineering,30,3000000.0\n"
        "South,Sales,20,1600000.0\n"
        "South,Support,15,900000.0\n"
        "South Total,All,65,5500000.0\n"
        "Grand Total,All,115,9800000.0\n"
    )
    p = tmp_path / "subtotals_mixed.csv"
    p.write_text(content, encoding="utf-8")
    return p


def wide_time_headers_csv(tmp_path: Path) -> Path:
    """Assumption 3: Columns are NOT all variables; time is in the column headers."""
    content = (
        "Country,Indicator,2019,2020,2021,2022,2023\n"
        "USA,GDP_Growth,2.3, -3.4,5.9,2.1,2.5\n"
        "USA,Inflation,1.8,1.2,4.7,8.0,4.1\n"
        "DEU,GDP_Growth,1.1, -4.6,3.2,1.8, -0.3\n"
        "DEU,Inflation,1.4,0.5,3.1,6.9,5.9\n"
        "JPN,GDP_Growth, -0.4, -4.3,2.2,1.0,1.9\n"
        "JPN,Inflation,0.5,0.0, -0.2,2.5,3.2\n"
    )
    p = tmp_path / "wide_time_headers.csv"
    p.write_text(content, encoding="utf-8")
    return p


def autocorrelated_series_csv(tmp_path: Path, n: int = 500) -> Path:
    """Assumption 4: Rows are NOT independent; high serial dependence AR(1) phi=0.95."""
    rng = np.random.default_rng(42)
    phi = 0.95
    errors = rng.normal(0, 1, size=n)
    series = np.zeros(n)
    for t in range(1, n):
        series[t] = phi * series[t - 1] + errors[t]

    dates = pd.date_range("2025-01-01", periods=n, freq="h")
    df = pd.DataFrame({
        "timestamp": dates,
        "sensor_reading": series,
        "target_correlated": series * 0.8 + rng.normal(0, 0.5, size=n),
    })
    p = tmp_path / "autocorrelated.csv"
    df.to_csv(p, index=False)
    return p


def clustered_entities_csv(tmp_path: Path, n_clusters: int = 20, cluster_size: int = 30) -> Path:
    """Assumption 4 (cluster): Nested observations (students within schools)."""
    rng = np.random.default_rng(42)
    rows: list[dict[str, object]] = []
    for cluster_id in range(n_clusters):
        school_effect = rng.normal(0, 15)
        for student_idx in range(cluster_size):
            study_hours = rng.uniform(5, 25)
            score = 50.0 + school_effect + 1.5 * study_hours + rng.normal(0, 5)
            rows.append({
                "school_id": f"SCH_{cluster_id:03d}",
                "student_id": f"STU_{cluster_id:03d}_{student_idx:03d}",
                "study_hours": round(study_hours, 1),
                "test_score": round(score, 1),
            })
    df = pd.DataFrame(rows)
    p = tmp_path / "clustered_schools.csv"
    df.to_csv(p, index=False)
    return p


def compositional_shares_csv(tmp_path: Path, n: int = 100) -> Path:
    """Assumption 6: Variables are NOT on interval scale; shares sum to 100%."""
    rng = np.random.default_rng(42)
    # Generate Dirichlet distributed shares
    alphas = [2.0, 3.0, 4.0, 1.0]
    raw_shares = rng.dirichlet(alphas, size=n) * 100.0
    df = pd.DataFrame(raw_shares, columns=["share_marketing", "share_rd", "share_ops", "share_sales"])
    df["firm_id"] = [f"FIRM_{i:04d}" for i in range(n)]
    p = tmp_path / "compositional_shares.csv"
    df.to_csv(p, index=False)
    return p


def circular_variables_csv(tmp_path: Path, n: int = 200) -> Path:
    """Assumption 6: Angular/periodic variables (hours around midnight: 23, 0, 1)."""
    rng = np.random.default_rng(42)
    # Peak at 00:00 (midnight): wrapped normal around 0 hours (variance 1.5 hours)
    raw_hours = rng.normal(0, 1.5, size=n)
    hours = np.round(raw_hours % 24).astype(int)
    # Peak at North (360/0 degrees): wrapped normal around 0
    raw_degrees = rng.normal(0, 20.0, size=n)
    degrees = np.round(raw_degrees % 360).astype(int)
    df = pd.DataFrame({
        "event_id": [f"EVT_{i:04d}" for i in range(n)],
        "hour_of_day": hours,
        "wind_direction_deg": degrees,
        "noise_db": rng.uniform(40, 80, size=n).round(1),
    })
    p = tmp_path / "circular_events.csv"
    df.to_csv(p, index=False)
    return p


def heaped_and_censored_csv(tmp_path: Path, n: int = 300) -> Path:
    """Assumption 11: Values are NOT exact; heaped ages and censored concentrations."""
    rng = np.random.default_rng(42)
    # 40% of people round age to nearest 5
    raw_ages = rng.integers(18, 80, size=n)
    heaped_ages = [
        int(round(age / 5.0) * 5) if rng.random() < 0.4 else int(age)
        for age in raw_ages
    ]
    # Concentrations with <0.05 LOD and >500 top-coded values
    raw_conc = rng.exponential(scale=20.0, size=n)
    conc_str: list[str] = []
    for val in raw_conc:
        if val < 0.05:
            conc_str.append("<0.05")
        elif val > 80.0:
            conc_str.append(">80.0")
        else:
            conc_str.append(f"{val:.2f}")

    df = pd.DataFrame({
        "patient_id": [f"PAT_{i:04d}" for i in range(n)],
        "reported_age": heaped_ages,
        "assay_concentration": conc_str,
    })
    p = tmp_path / "heaped_and_censored.csv"
    df.to_csv(p, index=False)
    return p


def benford_anomalous_csv(tmp_path: Path, n: int = 1000) -> Path:
    """Assumption 11: Fabricated amounts violating Benford's first-digit law."""
    rng = np.random.default_rng(42)
    # First digits uniformly distributed between 1 and 9 (violates Benford ~30% for 1)
    first_digits = rng.integers(1, 10, size=n)
    remainders = rng.uniform(0, 1, size=n)
    magnitudes = rng.choice([10, 100, 1000, 10000], size=n)
    amounts = (first_digits + remainders) * magnitudes

    df = pd.DataFrame({
        "invoice_id": [f"INV_{i:05d}" for i in range(n)],
        "amount": amounts.round(2),
    })
    p = tmp_path / "benford_anomaly.csv"
    df.to_csv(p, index=False)
    return p
