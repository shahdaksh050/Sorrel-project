"""
Recall rubric on a realistic messy table: an hourly air-quality-like export
(`;`-separated, decimal commas, -200 placeholders, a 90%-missing column, a
diurnal cycle with a weekend dip, a seasonal humidity swing over 14 months,
and redundant sensors). The deterministic pipeline must recover what is there
and must not headline artefacts.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.core.controller import AgentController

PEAK_HOUR = 8
CYCLE_MEASURES = ("CO_GT", "NO2_GT", "C6H6_GT")


def air_quality_table(seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2004-03-01", "2005-04-30 23:00", freq="h")  # 14 months
    n = len(ts)
    hour = np.asarray(ts.hour)
    weekend = np.asarray(ts.dayofweek >= 5)
    cycle = np.cos(2 * np.pi * (hour - PEAK_HOUR) / 24)
    dip = np.where(weekend, 0.7, 1.0)
    df = pd.DataFrame({"timestamp": ts})
    for name, base in zip(CYCLE_MEASURES, (2.0, 100.0, 10.0), strict=True):
        df[name] = base * (1 + 0.5 * cycle) * dip + rng.normal(0, 0.08 * base, n)
    # Seasonal humidity: winter high, summer low (annual cycle, no trend).
    doy = np.asarray(ts.dayofyear)
    df["RH"] = 55 + 15 * np.cos(2 * np.pi * (doy - 15) / 365.25) + rng.normal(0, 4, n)
    # Three redundant sensor channels driven by one latent factor.
    latent = rng.normal(0, 1, n)
    for name in ("S1", "S2", "S3"):
        df[name] = 1000 + 100 * latent + rng.normal(0, 20, n)
    df["T"] = 15 + rng.normal(0, 3, n)  # deliberately non-seasonal: humidity is the seasonal measure
    df["NMHC_GT"] = np.where(rng.random(n) < 0.9, np.nan, rng.gamma(4, 50, n))
    df = df.round(3)
    for col in ("CO_GT", "NO2_GT", "RH", "S1", "NMHC_GT"):
        df.loc[rng.random(n) < 0.05, col] = -200.0
    return df


def write_air_quality(path: Path, df: pd.DataFrame) -> Path:
    out = df.copy()
    for col in ("CO_GT", "C6H6_GT", "T"):  # decimal-comma strings
        out[col] = out[col].map(lambda v: f"{v:g}".replace(".", ","))
    out.to_csv(path, sep=";", index=False)
    return path


@pytest.fixture(scope="module")
def air(tmp_path_factory):
    d = tmp_path_factory.mktemp("air")
    path = write_air_quality(d / "air.csv", air_quality_table())
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("OUTPUT_DIR", str(d / "out"))
        # The time-series measure is picked objective-aware; a user asking about
        # humidity is the realistic way to make the seasonal series the subject.
        mp.setenv("USER_OBJECTIVE", "Is humidity RH changing over time?")
        agent = AgentController(use_llm=False, enable_rlm=False)
        agent.load_dataset(str(path), interactive=False)
        result = agent.analyze()
    return result, agent


RAW_PLACEHOLDER = re.compile(r"(?<![\d.])-200(?:\.0+)?(?![\d])")


def _by(findings, **kw):
    return [f for f in findings if all(f.get(k) == v for k, v in kw.items())]


def test_placeholders_nulled_with_notes(air):
    _, agent = air
    notes = " ".join(agent.memory.get_context("degradations"))  # the disclosed-repairs log
    for col in ("CO_GT", "NO2_GT", "RH", "S1"):
        assert re.search(rf"{col}.*-200|-200.*{col}", notes), (col, notes)
    coerced = {c["column"] for c in agent.memory.get_context("coercions")}
    assert {"CO_GT", "C6H6_GT", "T"} <= coerced  # decimal-comma strings repaired


def test_no_finding_cites_raw_placeholder(air):
    result, _ = air
    text = json.dumps(result["findings"], default=str) + json.dumps(result.get("insights"), default=str)
    hits = RAW_PLACEHOLDER.findall(text)
    assert not hits, hits


def test_diurnal_cycle_recovered(air):
    findings = air[0]["findings"]
    for m in CYCLE_MEASURES:
        (f,) = _by(findings, kind="trend", dimension="hour_of_day", measure=m)
        assert abs(int(f["level"][:2]) - PEAK_HOUR) <= 1, f["level"]
        ratio = f["evidence"]["peak_trough_ratio"]
        assert 0.8 * 3.0 <= ratio <= 1.2 * 3.0, ratio  # planted (1+.5)/(1-.5)
        # weekend dip: planted level x0.7 -> gap -0.3
        gap = f["evidence"]["weekend_gap"]
        assert gap is not None and -0.3 * 1.35 <= gap <= -0.3 * 0.65, gap
    assert not _by(findings, dimension="hour_of_day", measure="RH")


def test_redundant_sensors_are_one_finding(air):
    findings = [f for f in air[0]["findings"] if f["kind"] == "correlation"]
    groups = [f for f in findings if {"S1", "S2", "S3"} <= set(f["evidence"].get("columns") or [])]
    assert len(groups) == 1 and set(groups[0]["evidence"]["columns"]) == {"S1", "S2", "S3"}
    pairs = [f for f in findings if {f["evidence"].get("col_a"), f["evidence"].get("col_b")} <= {"S1", "S2", "S3"}
             and "col_a" in f["evidence"]]
    assert all(f["layer"] == "appendix" for f in pairs)


def test_missingness_finding_names_sparse_column(air):
    gaps = [f for f in air[0]["findings"] if f["kind"] == "coverage_gap"]
    assert any("NMHC_GT" in f["headline"] and "9" in f["headline"] for f in gaps), gaps


def test_seasonal_humidity_not_headlined_as_a_rise(air):
    findings = [f for f in air[0]["findings"] if f.get("measure") == "RH"]
    rises = [
        f for f in findings
        if f["kind"] in ("trend", "change") and f.get("dimension") != "month"
        and f.get("layer") != "appendix"  # a demoted, "may be the season" caveat is not a headline
        and re.search(r"increas|rise|rose|growing|grew|up|higher", f["headline"], re.I)
    ]
    assert not rises, [f["headline"] for f in rises]
