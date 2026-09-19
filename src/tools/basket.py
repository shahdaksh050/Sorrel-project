"""
Basket Analysis Tool — Execution Layer.

Market-basket association rules ("customers who buy A also buy B") without a
heavy library. Order and item ids are factorised into integer codes, an
order x item boolean CSR matrix is built, and pair co-occurrence counts come
from one sparse product (X.T @ X) over the 60 most frequent items. From those
counts: support, confidence and lift for both directions of every pair.

A rule is kept when it holds in at least ``min_support`` orders and its lift is
above 1 (the pair is bought together more often than chance). The top rules by
lift are reported in plain words, plus the single most frequently co-bought
pair. Everything is deterministic and fitted per call; nothing is stored.
"""
from __future__ import annotations

import math
import re
from itertools import pairwise
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from src.core.chart_spec import validate_chart_spec
from src.core.findings import Finding
from src.tools.base import BaseTool, ToolExecutionError
from src.tools.data_processing import _read_df

if TYPE_CHECKING:
    from src.core.memory import DatasetMetadata
    from src.core.profiler import DatasetProfile

_MAX_ORDERS = 200_000
_TOP_ITEMS = 60
_MAX_RULES = 10
_MAX_FINDINGS = 5
_MIN_ORDERS = 20
_MIN_ROWS_TO_APPLY = 200
_MIN_ROWS_PER_ORDER = 1.5
_ORDER_TOKENS = ("order", "transaction", "invoice", "basket", "receipt", "cart", "ticket", "session")
_ITEM_TOKENS = ("item", "product", "sku", "article", "description", "category", "good")
_NOT_AN_ID_TOKENS = frozenset({"date", "time", "datetime", "timestamp", "day", "month", "year", "amount", "total", "value"})
_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_SPLIT = re.compile(r"[^a-z0-9]+")


def _tokens(name: str) -> set[str]:
    words = [t for t in _SPLIT.split(_CAMEL.sub("_", str(name)).lower()) if t]
    return set(words) | {"".join(p) for p in pairwise(words)}


def _priority(name: str, vocabulary: tuple[str, ...]) -> int | None:
    tokens = _tokens(name)
    for i, word in enumerate(vocabulary):
        if word in tokens:
            return i
    return None


def _detect_columns(profile: DatasetProfile | None) -> tuple[str, str] | None:
    """(order id column, item column) when the profile clearly looks like
    line-item transaction data, else None. Whole-token name matching only."""
    if profile is None or profile.row_count < _MIN_ROWS_TO_APPLY:
        return None
    orders: list[tuple[int, str]] = []
    items: list[tuple[int, str]] = []
    for c in profile.columns:
        if c.kind == "datetime" or c.nunique < 2:
            continue
        if _tokens(c.name) & _NOT_AN_ID_TOKENS:
            continue
        rank = _priority(c.name, _ORDER_TOKENS)
        if rank is not None and c.semantic_role != "measure" and profile.row_count / c.nunique >= _MIN_ROWS_PER_ORDER:
            orders.append((rank, c.name))
        rank = _priority(c.name, _ITEM_TOKENS)
        if rank is not None and c.kind != "numeric" and c.nunique >= 3:
            items.append((rank, c.name))
    for _, order in sorted(orders):
        for _, item in sorted(items):
            if item != order:
                return order, item
    return None


def _short(text: str, n: int = 40) -> str:
    return text if len(text) <= n else text[: n - 1].rstrip() + "…"


def _pct(x: float) -> str:
    return f"{x * 100:.0f}%"


class BasketAnalysisTool(BaseTool):
    """Which items are bought together (association rules)."""

    name = "basket_analysis"
    description = (
        "Market-basket analysis: which items are bought together. Groups line-item "
        "rows into orders and reports the strongest 'customers who buy A also buy B' "
        "rules (how often, and how many times more often than the average customer), "
        "plus the most frequently co-bought pair. Needs an order/transaction id column "
        "and an item/product column."
    )

    def applies_to(self, profile: DatasetProfile | None, metadata: DatasetMetadata | None) -> float:
        return 0.6 if _detect_columns(profile) else 0.0

    def default_params(
        self, profile: DatasetProfile | None, metadata: DatasetMetadata | None
    ) -> dict[str, Any]:
        found = _detect_columns(profile)
        return {"order_column": found[0], "item_column": found[1]} if found else {}

    def execute(  # type: ignore[override]
        self,
        file_path: str,
        order_column: str | None = None,
        item_column: str | None = None,
        min_support: int | float | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        if not order_column or not item_column:
            raise ToolExecutionError(
                "Pass order_column (the order/transaction id) and item_column (the product or item)."
            )
        df = _read_df(file_path)
        for col in (order_column, item_column):
            if col not in df.columns:
                raise ToolExecutionError(f"Column '{col}' not found in dataset.")
        if order_column == item_column:
            raise ToolExecutionError("order_column and item_column must be different columns.")

        from scipy import sparse

        order_codes, order_names = pd.factorize(df[order_column])
        item_codes, item_names = pd.factorize(df[item_column])
        keep = (order_codes >= 0) & (item_codes >= 0)
        order_codes, item_codes = order_codes[keep], item_codes[keep]
        if len(order_codes) == 0:
            raise ToolExecutionError("No rows with both an order id and an item.")

        n_orders_total = len(order_names)
        sampled = False
        if n_orders_total > _MAX_ORDERS:
            chosen = np.sort(np.random.default_rng(0).choice(n_orders_total, _MAX_ORDERS, replace=False))
            lookup = np.full(n_orders_total, -1, dtype=np.int64)
            lookup[chosen] = np.arange(_MAX_ORDERS)
            order_codes = lookup[order_codes]
            in_sample = order_codes >= 0
            order_codes, item_codes = order_codes[in_sample], item_codes[in_sample]
            sampled = True
        n_orders = _MAX_ORDERS if sampled else n_orders_total
        if n_orders < _MIN_ORDERS:
            raise ToolExecutionError(
                f"Only {n_orders} orders found in '{order_column}'; need at least {_MIN_ORDERS}."
            )

        matrix = sparse.coo_matrix(
            (np.ones(len(order_codes), dtype=np.int32), (order_codes, item_codes)),
            shape=(n_orders, len(item_names)),
        ).tocsr()
        matrix.data = np.ones_like(matrix.data)  # repeated (order, item) lines count once
        item_orders = np.asarray(matrix.sum(axis=0)).ravel()
        basket_sizes = np.asarray(matrix.sum(axis=1)).ravel()
        mean_items = float(basket_sizes.mean())
        if mean_items < 1.05:
            raise ToolExecutionError(
                f"Almost every '{order_column}' value has a single '{item_column}'; there are no "
                "baskets to analyse. Check that order_column is the order id, not a line id."
            )

        top = np.argsort(-item_orders, kind="stable")[:_TOP_ITEMS]
        sub = matrix[:, top]
        counts = item_orders[top].astype(float)
        cooc = (sub.T @ sub).toarray().astype(float)

        floor = max(5, math.ceil(0.005 * n_orders))
        if min_support is None:
            support_floor = floor
        else:
            try:
                value = float(min_support)
            except (TypeError, ValueError):
                raise ToolExecutionError("min_support must be a number of orders (or a fraction below 1).") from None
            support_floor = max(1, math.ceil(value * n_orders)) if 0 < value < 1 else max(1, int(value))

        a, b = np.triu_indices(len(top), 1)
        co = cooc[a, b]
        with np.errstate(divide="ignore", invalid="ignore"):
            lift = co * n_orders / (counts[a] * counts[b])
            conf_ab = co / counts[a]
            conf_ba = co / counts[b]
        forward = conf_ab >= conf_ba  # report each pair in its stronger direction
        ant = np.where(forward, a, b)
        con = np.where(forward, b, a)
        confidence = np.where(forward, conf_ab, conf_ba)

        ok = (co >= support_floor) & (lift > 1)
        idx = np.flatnonzero(ok)
        idx = idx[np.lexsort((-co[idx], -lift[idx]))][:_MAX_RULES]
        names = [str(item_names[i]) for i in top]
        rules = [
            {
                "antecedent": names[int(ant[i])],
                "consequent": names[int(con[i])],
                "orders": int(co[i]),
                "support": round(float(co[i] / n_orders), 4),
                "confidence": round(float(confidence[i]), 4),
                "lift": round(float(lift[i]), 3),
                "consequent_base_rate": round(float(counts[con[i]] / n_orders), 4),
            }
            for i in idx
        ]
        top_pair = None
        if len(co) and float(co.max()) > 0:
            j = int(np.argmax(co))
            top_pair = {
                "item_a": names[int(a[j])], "item_b": names[int(b[j])],
                "orders": int(co[j]), "support": round(float(co[j] / n_orders), 4),
                "lift": round(float(lift[j]), 3),
            }

        result: dict[str, Any] = {
            "order_column": order_column, "item_column": item_column,
            "n_orders": int(n_orders), "n_items": len(item_names),
            "n_items_considered": len(top), "mean_items_per_order": round(mean_items, 2),
            "min_support_orders": int(support_floor), "sampled": sampled,
            "n_orders_before_sampling": int(n_orders_total),
            "rules": rules, "top_pair": top_pair,
        }
        chart = self._chart(rules, order_column, item_column)
        if chart is not None:
            result["chart"] = chart
        result["summary"] = self._summary(result)
        return result

    @staticmethod
    def _chart(rules: list[dict[str, Any]], order_column: str, item_column: str) -> dict[str, Any] | None:
        if len(rules) < 2:
            return None
        rows = [
            {"rule": f"{_short(r['antecedent'], 26)} → {_short(r['consequent'], 26)}", "lift": r["lift"]}
            for r in rules
        ]
        clean, _error = validate_chart_spec({
            "type": "bar", "data": rows, "x": "rule", "y": "lift",
            "title": f"Items bought together, by lift ({item_column})",
            "y_title": "Times more likely than the average order",
            "y_format": "number", "sort": "desc",
            "annotations": [{"y": 1, "label": "1x = no link"}],
            "caption": f"Each bar: how many times more often the second item appears in orders ({order_column}) that contain the first.",
        })
        return clean

    @staticmethod
    def _summary(r: dict[str, Any]) -> str:
        n = f"{r['n_orders']:,} orders"
        if r["rules"]:
            t = r["rules"][0]
            return (
                f"Customers who buy {t['antecedent']} also buy {t['consequent']} {_pct(t['confidence'])} of the time, "
                f"{t['lift']:.1f}x more often than the average customer ({len(r['rules'])} rule(s) across {n})."
            )
        if r["top_pair"]:
            p = r["top_pair"]
            return (
                f"No pair of items is bought together more than chance at the minimum of "
                f"{r['min_support_orders']} orders across {n}; most often together: {p['item_a']} and {p['item_b']} "
                f"({p['orders']:,} orders)."
            )
        return f"No two items were ever bought in the same order across {n}."

    def findings(
        self,
        output: dict[str, Any],
        profile: DatasetProfile | None,
        metadata: DatasetMetadata | None,
    ) -> list[Finding]:
        order_col, item_col = str(output.get("order_column")), str(output.get("item_column"))
        n_orders = int(output.get("n_orders", 0))
        results: list[Finding] = []
        for r in output.get("rules", [])[:_MAX_FINDINGS]:
            a, b = r["antecedent"], r["consequent"]
            caveats = [
                "This is association, not causation: buying A does not make customers buy B.",
                "Popular items inflate confidence on their own; lift corrects for that by comparing with how common the item is overall.",
            ]
            if r["orders"] < 30:
                caveats.append(f"Only {r['orders']} orders contain both items, so the rate is rough.")
            if output.get("sampled"):
                caveats.append(f"Computed on a random sample of {n_orders:,} of {output.get('n_orders_before_sampling'):,} orders.")
            if output.get("n_items", 0) > output.get("n_items_considered", 0):
                caveats.append(f"Only the {output.get('n_items_considered')} most frequently bought items were compared.")
            results.append(Finding(
                finding_id=f"basket_{a}_{b}".replace(" ", "_")[:120],
                kind="association",
                headline=(
                    f"Customers who buy {_short(a, 60)} also buy {_short(b, 60)} {_pct(r['confidence'])} of the time, "
                    f"{r['lift']:.1f}x more often than the average customer."
                ),
                detail=(
                    f"{r['orders']:,} of {n_orders:,} orders contain both; {_pct(r['consequent_base_rate'])} of all orders "
                    f"contain {b}."
                ),
                evidence={
                    "antecedent": a, "consequent": b, "support": r["support"], "support_orders": r["orders"],
                    "confidence": r["confidence"], "lift": r["lift"], "n_orders": n_orders,
                    "consequent_base_rate": r["consequent_base_rate"], "n": r["orders"],
                },
                source_tool=self.name, measure=order_col, dimension=item_col, level=a,
                effect=round(r["lift"] - 1.0, 4), effect_kind="lift",
                confidence=min(0.9, max(0.3, r["orders"] / 100.0)),
                surprise=min(1.0, (r["lift"] - 1.0) / 4.0),
                caveats=caveats, chart_hint=output.get("chart"),
            ))
        return results

    def get_schema(self) -> dict[str, Any]:
        return {
            "file_path": {"type": "string", "description": "Path to the dataset.", "required": True},
            "order_column": {
                "type": "string",
                "description": "Order / transaction / invoice id column (one value repeated over the order's line items).",
                "required": True,
            },
            "item_column": {
                "type": "string",
                "description": "Product / item / SKU column identifying what was bought.",
                "required": True,
            },
            "min_support": {
                "type": "number",
                "description": (
                    "Minimum number of orders a pair must appear in (a value below 1 is a share of orders). "
                    "Default: the larger of 5 orders and 0.5% of orders."
                ),
                "required": False,
            },
        }
