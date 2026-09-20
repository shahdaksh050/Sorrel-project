"""Tests for PriceElasticityTool (src/tools/elasticity.py)."""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.core.profiler import profile_dataframe
from src.tools.elasticity import PriceElasticityTool


def _frame(n_products: int = 12, n: int = 100, seed: int = 5, constant: bool = False) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    parts = []
    for i in range(1, n_products + 1):
        e = -1.5 if i <= 6 else -0.4
        price = np.exp(rng.normal(np.log(10), 0.25, n))
        qty = np.exp(6 + e * np.log(price) + rng.normal(0, 0.15, n))
        parts.append(pd.DataFrame({"product": f"P{i:02d}", "price": price, "quantity": qty}))
    df = pd.concat(parts, ignore_index=True)
    if constant:
        df.loc[df["product"] == "P01", "price"] = 10.0
    return df


def _run(tmp_path: Path, df: pd.DataFrame, **kw: Any) -> Any:
    path = tmp_path / "e.csv"
    df.to_csv(path, index=False)
    res = PriceElasticityTool().run(
        file_path=str(path), price_column="price", quantity_column="quantity", product_column="product", **kw
    )
    assert res.status == "success", res.error_message
    return res


def test_recovers_elasticities_and_pooled(tmp_path: Path) -> None:
    out = _run(tmp_path, _frame()).output
    assert len(out["products"]) == 12
    for p in out["products"]:
        truth = -1.5 if int(p["product"][1:]) <= 6 else -0.4
        assert abs(p["elasticity"] - truth) < 0.25
        assert p["ci_low"] <= truth <= p["ci_high"], p
    pooled = out["pooled"]
    assert pooled is not None and -1.1 < pooled["elasticity"] < -0.8
    assert "10% price rise" in out["summary"] and "revenue" in out["summary"]
    assert out["chart"]["type"] == "dot_ci"
    f = PriceElasticityTool().findings(out, None, None)
    assert f and f[0].kind == "elasticity"


def test_constant_price_product_skipped(tmp_path: Path) -> None:
    out = _run(tmp_path, _frame(constant=True)).output
    assert "P01" not in [p["product"] for p in out["products"]]
    assert out["n_products_skipped"] == 1 and "1 skipped" in out["summary"]


def test_positive_elasticity_gets_confounding_caveat(tmp_path: Path) -> None:
    rng = np.random.default_rng(2)
    price = np.exp(rng.normal(np.log(10), 0.25, 200))
    df = pd.DataFrame({"price": price, "quantity": np.exp(2 + 1.0 * np.log(price) + rng.normal(0, 0.1, 200))})
    path = tmp_path / "pos.csv"
    df.to_csv(path, index=False)
    tool = PriceElasticityTool()
    res = tool.run(file_path=str(path), price_column="price", quantity_column="quantity")
    assert res.status == "success", res.error_message
    assert res.output["products"][0]["elasticity"] > 0
    f = tool.findings(res.output, None, None)[0]
    assert any("confounding" in c for c in f.caveats) and "promotions" in f.headline


def test_applies_to() -> None:
    tool = PriceElasticityTool()
    unrelated = pd.DataFrame({"age": np.arange(300), "score": np.arange(300) * 1.5, "city": ["a", "b", "c"] * 100})
    assert tool.applies_to(profile_dataframe(unrelated), None) == 0.0
    assert tool.applies_to(profile_dataframe(_frame(3)), None) > 0


def test_perf_and_thread_safety(tmp_path: Path) -> None:
    df = _frame(n_products=10, n=5000)
    path = tmp_path / "big.csv"
    df.to_csv(path, index=False)
    tool = PriceElasticityTool()
    kw: dict[str, Any] = {
        "file_path": str(path), "price_column": "price", "quantity_column": "quantity", "product_column": "product",
    }
    t0 = time.perf_counter()
    first = tool.run(**kw)
    assert time.perf_counter() - t0 < 6 and first.status == "success"
    with ThreadPoolExecutor(4) as ex:
        outs = list(ex.map(lambda _: tool.run(**kw).output, range(4)))
    assert all(o == first.output for o in outs)
