"""Tests for BasketAnalysisTool (src/tools/basket.py)."""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.core.profiler import profile_dataframe
from src.tools.basket import BasketAnalysisTool


def _lines(n_orders: int = 5000, seed: int = 1) -> pd.DataFrame:
    """A in 4% of orders; B alongside A 40% of the time, else 3.4% -> B overall 5%."""
    rng = np.random.default_rng(seed)
    rows: list[tuple[int, str]] = []
    for o in range(n_orders):
        basket: set[str] = set()
        if rng.random() < 0.04:
            basket.add("item_A")
            if rng.random() < 0.40:
                basket.add("item_B")
        elif rng.random() < 0.0354:
            basket.add("item_B")
        for _ in range(rng.integers(1, 4)):
            basket.add(f"item_{rng.integers(1, 29):02d}")  # 28 filler items, never A/B
        rows.extend((o, it) for it in basket)
    return pd.DataFrame(rows, columns=["order_id", "item"])


def _run(tmp_path: Path, df: pd.DataFrame, **params: Any) -> Any:
    path = tmp_path / "b.csv"
    df.to_csv(path, index=False)
    res = BasketAnalysisTool().run(file_path=str(path), order_column="order_id", item_column="item", **params)
    assert res.status == "success", res.error_message
    return res


def test_planted_pair_is_top_rule(tmp_path: Path) -> None:
    df = _lines()
    orders = df.groupby("order_id")["item"].apply(set)
    has_a = orders.apply(lambda s: "item_A" in s)
    true_conf = float(orders[has_a].apply(lambda s: "item_B" in s).mean())
    base_b = float(orders.apply(lambda s: "item_B" in s).mean())
    assert abs(base_b - 0.05) < 0.02
    true_lift = true_conf / base_b

    out = _run(tmp_path, df).output
    rule = out["rules"][0]
    assert {rule["antecedent"], rule["consequent"]} == {"item_A", "item_B"}
    assert abs(rule["lift"] - true_lift) / true_lift < 0.25
    assert rule["antecedent"] == "item_A" and abs(rule["confidence"] - true_conf) < 0.05
    assert out["n_orders"] == 5000 and out["chart"]["type"] == "bar"
    assert all(r["orders"] >= out["min_support_orders"] for r in out["rules"])

    f = BasketAnalysisTool().findings(out, None, None)[0]
    assert f.kind == "association" and "item_A" in f.headline and "item_B" in f.headline
    assert "more often than the average customer" in f.headline


def test_support_threshold_respected(tmp_path: Path) -> None:
    out = _run(tmp_path, _lines(), min_support=0.05).output
    assert out["min_support_orders"] == 250
    assert all(r["orders"] >= 250 for r in out["rules"])
    high = _run(tmp_path, _lines(), min_support=0.5).output
    assert high["rules"] == [] and "No pair" in high["summary"]


def test_applies_to_negative_and_positive() -> None:
    tool = BasketAnalysisTool()
    unique = pd.DataFrame({"order_id": range(1000), "item": np.random.default_rng(0).choice(list("abcdefgh"), 1000)})
    assert tool.applies_to(profile_dataframe(unique), None) == 0.0
    unrelated = pd.DataFrame({"x": np.arange(500.0), "y": np.arange(500.0) ** 2, "region": ["n", "s"] * 250})
    assert tool.applies_to(profile_dataframe(unrelated), None) == 0.0
    assert tool.applies_to(profile_dataframe(_lines(1000)), None) > 0


def test_unique_order_ids_error(tmp_path: Path) -> None:
    df = pd.DataFrame({"order_id": range(300), "item": ["a", "b", "c"] * 100})
    path = tmp_path / "u.csv"
    df.to_csv(path, index=False)
    res = BasketAnalysisTool().run(file_path=str(path), order_column="order_id", item_column="item")
    assert res.status != "success"


def test_perf_and_thread_safety(tmp_path: Path) -> None:
    rng = np.random.default_rng(3)
    big = pd.DataFrame({"order_id": rng.integers(0, 15000, 50000), "item": rng.integers(0, 40, 50000).astype(str)})
    path = tmp_path / "big.csv"
    big.to_csv(path, index=False)
    tool = BasketAnalysisTool()
    kw: dict[str, Any] = {"file_path": str(path), "order_column": "order_id", "item_column": "item"}
    t0 = time.perf_counter()
    first = tool.run(**kw)
    assert time.perf_counter() - t0 < 6 and first.status == "success"
    with ThreadPoolExecutor(4) as ex:
        outs = list(ex.map(lambda _: tool.run(**kw).output, range(4)))
    assert all(o == first.output for o in outs)
