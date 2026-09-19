"""Tests for multi-column relation discovery and its prompt / toolkit surfaces."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.core.memory import DatasetMetadata, MemorySystem
from src.core.profiler import profile_dataframe
from src.core.relations import find_relations
from src.core.sandbox_toolkit import Toolkit
from src.tools.ml_pipeline import _detect_formula_leakage

N = 300


def _frame(**extra: np.ndarray) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    data = {
        "width": rng.uniform(1, 20, N),
        "height": rng.uniform(1, 20, N),
        "noise": rng.normal(50, 9, N),
    }
    data["area"] = data["width"] * data["height"]
    data.update(extra)
    return pd.DataFrame(data)


def _members(rel: dict) -> set[str]:
    return {rel["target"], *rel["terms"]}


def test_exact_product_reported_once() -> None:
    rels = find_relations(_frame())
    assert len(rels) == 1  # a = b*c, b = a/c and c = a/b are one relation
    rel = rels[0]
    assert rel["kind"] == "product" and rel["exact"] is True and rel["confidence"] == "high"
    assert rel["target"] == "area" and sorted(rel["terms"]) == ["height", "width"]
    assert rel["expr"] == "area = height * width"


def test_exact_sum_of_parts() -> None:
    rng = np.random.default_rng(3)
    df = pd.DataFrame({c: rng.uniform(10, 100, N) for c in ("north", "south", "east")})
    df["total"] = df["north"] + df["south"] + df["east"]
    (rel,) = find_relations(df)
    assert rel["kind"] == "part_of_total" and rel["target"] == "total" and rel["exact"] is True


def test_wide_total_uses_all_parts() -> None:
    rng = np.random.default_rng(4)
    df = pd.DataFrame({f"p{i}": rng.uniform(10, 100, N) for i in range(4)})
    df["grand"] = df.sum(axis=1)
    rel = next(r for r in find_relations(df) if r["target"] == "grand")
    assert sorted(rel["terms"]) == ["p0", "p1", "p2", "p3"]


def test_difference_reads_as_subtraction() -> None:
    rng = np.random.default_rng(5)
    df = pd.DataFrame({"revenue": rng.uniform(100, 500, N), "cost": rng.uniform(10, 90, N)})
    df["profit"] = df["revenue"] - df["cost"]
    (rel,) = find_relations(df)
    assert rel["kind"] == "difference" and rel["target"] == "profit"
    assert rel["expr"] == "profit = revenue - cost"


def test_near_relation_reports_error() -> None:
    rng = np.random.default_rng(11)
    df = _frame()
    df["area"] = df["width"] * df["height"] * (1 + rng.normal(0, 0.002, N))
    (rel,) = find_relations(df)
    assert rel["exact"] is False and 0 < rel["max_rel_err"] < 0.05 and rel["r2"] > 0.999


def test_cumulative_identity() -> None:
    rng = np.random.default_rng(13)
    df = pd.DataFrame({"step": rng.uniform(1, 9, N)})
    df["running"] = df["step"].cumsum()
    rel = next(r for r in find_relations(df) if r["kind"] == "cumulative")
    assert rel["target"] == "running" and rel["terms"] == ["step"] and rel["exact"] is True


def test_ids_constants_flags_and_nulls_are_ignored() -> None:
    df = _frame(row_id=np.arange(N), const=np.full(N, 3.0), flag=np.tile([0.0, 1.0], N // 2))
    assert all(not ({"row_id", "const", "flag"} & _members(r)) for r in find_relations(df))
    assert find_relations(df, exclude={"width"}) == []  # the only formula needed the excluded column


def test_independent_and_tiny_frames_give_nothing() -> None:
    rng = np.random.default_rng(1)
    assert find_relations(pd.DataFrame(rng.normal(size=(N, 6)), columns=list("abcdef"))) == []
    assert find_relations(pd.DataFrame()) == []
    assert find_relations(_frame().head(10)) == []


def test_profile_carries_relations() -> None:
    profile = profile_dataframe(_frame())
    assert [r["target"] for r in profile.relations] == ["area"]
    assert profile.to_dict()["relations"] == profile.relations


def _memory(relations: list[dict]) -> MemorySystem:
    memory = MemorySystem()
    memory.dataset_metadata = DatasetMetadata(
        file_path="d.csv", row_count=N, column_count=4, columns={}, missing_values={},
        numerical_cols=[], categorical_cols=[],
    )
    memory.set_context("data_profile", {"relations": relations, "columns": [{"name": "ssn", "pii": "gov_id"}]})
    return memory


def test_metadata_prompt_relations_block() -> None:
    rels = find_relations(_frame())
    for compact in (True, False):
        prompt = _memory(rels).get_metadata_prompt(compact=compact)
        assert "Relations" in prompt and "area = height * width (exact)" in prompt
    assert "Relations" not in _memory([]).get_metadata_prompt()
    hidden = [{"expr": "ssn = a * b", "target": "ssn", "terms": ["a", "b"], "exact": True}]
    assert "Relations" not in _memory(hidden).get_metadata_prompt()


def test_formula_leakage_warning() -> None:
    df = _frame()
    assert _detect_formula_leakage(df, "area", ["width", "height", "noise"])
    assert _detect_formula_leakage(df, "noise", ["width", "height"]) == []


def test_derive(tmp_path: Path) -> None:
    df = pd.DataFrame({"revenue": [10.0, 20.0, 30.0], "cost": [4.0, 0.0, 10.0], "unit price": [1.0, 2.0, 3.0]})
    out = Toolkit.derive(df, "margin", "(revenue - cost) / cost")
    assert "margin" not in df.columns and out["margin"].iloc[0] == pytest.approx(1.5)
    assert np.isnan(out["margin"].iloc[1])  # division by zero -> NaN, not inf
    assert Toolkit.derive(df, "x", "`unit price` ** 2 + 1")["x"].tolist() == [2.0, 5.0, 10.0]
    for bad in ("revenue.sum()", "__import__('os')", "revenue ** 99", "revenue ** cost", "revenue > 1", "nope + 1"):
        with pytest.raises(ValueError):
            Toolkit.derive(df, "y", bad)
    with pytest.raises(ValueError):
        Toolkit.derive(df, "cost", "revenue")  # existing name


def test_share_and_contribution_helpers() -> None:
    df = pd.DataFrame({
        "period": [1, 1, 2, 2], "region": ["n", "s", "n", "s"], "sales": [10.0, 10.0, 15.0, 5.0],
    })
    shares = Toolkit.share_of_total(df, "sales", by="region")
    assert dict(zip(shares["region"], shares["share_pct"], strict=True)) == {"n": 62.5, "s": 37.5}
    totals = Toolkit.contribution_to_change(df, "sales", "period")
    assert totals["change"].iloc[1] == 0.0
    parts = Toolkit.contribution_to_change(df, "sales", "period", by="region").set_index("region")
    assert parts.loc["n", "change"] == 5.0 and parts.loc["s", "change"] == -5.0
