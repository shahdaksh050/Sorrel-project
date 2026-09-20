"""summary.json written by the controller, and the run-to-run diff."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.core.controller import AgentController
from src.core.run_compare import compare_runs, list_runs


def test_summary_json_shape(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CHART_DESIGN", "false")
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path / "out"))
    monkeypatch.setenv("USER_OBJECTIVE", "what drives the label")
    rng = np.random.default_rng(1)
    fa = rng.normal(0, 1, 120)
    df = pd.DataFrame({
        "feature_a": fa, "feature_b": rng.normal(5, 2, 120),
        "label": (fa + rng.normal(0, 0.5, 120) > 0).astype(int),
    })
    csv = tmp_path / "sample.csv"
    df.to_csv(csv, index=False)
    agent = AgentController(max_iterations=2, enable_rlm=False, use_llm=False)
    agent.load_dataset(str(csv), target_hint="label", interactive=False)
    agent.analyze()

    data = json.loads((tmp_path / "out" / "reports" / "summary.json").read_text(encoding="utf-8"))
    assert data["dataset"] == {"name": "sample", "rows": 120, "cols": 3}
    assert data["objective"] == "what drives the label"
    assert data["llm"]["enabled"] is False
    assert isinstance(data["tool_seconds"], dict) and data["tool_seconds"]
    assert all(isinstance(v, float) for v in data["tool_seconds"].values())
    assert {"finding_id", "kind", "layer", "importance", "headline"} <= set(data["findings"][0])
    assert all({"chart_id", "title"} <= set(c) for c in data["charts"])
    assert data["charts"]


def _run(findings: list[dict], charts: list[dict], secs: dict[str, float]) -> dict:
    return {"findings": findings, "charts": charts, "tool_seconds": secs}


def test_compare_runs_diff() -> None:
    old = _run(
        [
            {"finding_id": "a", "headline": "Sales up 12% in North", "importance": 0.5},
            {"finding_id": "b", "headline": "Churn is 8% higher for new users", "importance": 0.6},
            {"finding_id": "gone", "headline": "Old finding", "importance": 0.4},
            {"finding_id": "s", "headline": "Stable", "importance": 0.5},
        ],
        [{"chart_id": "c1", "title": "Sales"}, {"chart_id": "c2", "title": "Old chart"}],
        {"clean_data": 1.0, "train_model": 10.0},
    )
    new = _run(
        [
            {"finding_id": "a", "headline": "Sales up 12% in North", "importance": 0.8},   # +0.3
            {"finding_id": "zzz", "headline": "Churn is 11% higher for new users", "importance": 0.3},  # headline match, -0.3
            {"finding_id": "n", "headline": "Brand new", "importance": 0.7},
            {"finding_id": "s", "headline": "Stable", "importance": 0.55},   # within threshold
        ],
        [{"chart_id": "c1", "title": "Sales"}, {"chart_id": "c3", "title": "New chart"}],
        {"clean_data": 1.5, "train_model": 4.0, "extra": 2.0},
    )
    d = compare_runs(old, new)
    assert d["findings_added"] == ["Brand new"]
    assert d["findings_removed"] == ["Old finding"]
    changed = {c["headline"]: round(c["delta"], 2) for c in d["findings_changed"]}
    assert changed == {"Sales up 12% in North": 0.3, "Churn is 11% higher for new users": -0.3}
    assert d["charts_added"] == ["New chart"]
    assert d["charts_removed"] == ["Old chart"]
    deltas = {t["tool"]: round(t["delta"], 2) for t in d["tool_time_deltas"]}
    assert deltas == {"clean_data": 0.5, "extra": 2.0, "train_model": -6.0}
    assert d["tool_time_deltas"][0]["tool"] == "train_model"   # biggest change first


def test_list_runs_newest_first(tmp_path: Path) -> None:
    import os

    for i, name in enumerate(["first", "second"]):
        p = tmp_path / f"run{i}" / "reports"
        p.mkdir(parents=True)
        f = p / "summary.json"
        f.write_text(json.dumps({"dataset": {"name": name}, "created": f"2026-01-0{i + 1}T10:00:00"}))
        os.utime(f, (1000 + i, 1000 + i))
    (tmp_path / "bad" / "reports").mkdir(parents=True)
    (tmp_path / "bad" / "reports" / "summary.json").write_text("{not json")
    runs = list_runs(tmp_path)
    assert [r["dataset"]["name"] for r in runs] == ["second", "first"]
    assert runs[0]["label"].startswith("second")
