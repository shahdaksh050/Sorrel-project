"""
Price Elasticity Tool — Execution Layer.

How much units sold respond to price, estimated as a constant elasticity
(log-log regression): ln(quantity) = a + e * ln(price). For each of the top
products by revenue (at least 8 observations and a real amount of price
variation) e is fitted by OLS with heteroskedasticity-robust (HC1) standard
errors, so it comes with a 95% interval and an R2. When a date column is
supplied, month-of-year dummies absorb seasonality so a December price cut is
not credited with December demand. With three or more products a pooled
estimate is fitted with product fixed effects (within-demeaning), which uses
each product's own price movements only.

An elasticity of -1.4 means a 1% price rise goes with roughly 1.4% fewer units
sold; a 10% rise scales units by 1.1**e and revenue by 1.1**(1+e). Positive or
near-zero estimates are reported as likely promotion / stock-out confounding,
not as a real preference for higher prices. Everything is fitted per call.
"""
from __future__ import annotations

import math
from itertools import pairwise
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from src.core.chart_spec import validate_chart_spec
from src.core.findings import Finding
from src.core.vocab import name_tokens, role_tokens
from src.tools.base import BaseTool, ToolExecutionError
from src.tools.data_processing import _read_df

if TYPE_CHECKING:
    from src.core.memory import DatasetMetadata
    from src.core.profiler import DatasetProfile

_MAX_ROWS = 50_000
_MAX_PRODUCTS = 15
_MIN_OBS = 8
_MIN_PRICE_CV = 0.05
_MIN_ROWS_TO_APPLY = 50
_NEAR_ZERO = 0.1
_POOLED_MIN_PRODUCTS = 3
_MAX_FINDINGS = 5
_PRICE_TOKENS = role_tokens("price")
_QTY_TOKENS = role_tokens("quantity")
_ALL = "All products"


def _tokens(name: str) -> set[str]:
    words = name_tokens(name)
    return set(words) | {"_".join(p) for p in pairwise(words)} | {"".join(p) for p in pairwise(words)}


def _matches(name: str, vocabulary: tuple[str, ...]) -> int | None:
    tokens = _tokens(name)
    for i, word in enumerate(vocabulary):
        if word in tokens:
            return i
    return None


def _detect_columns(profile: DatasetProfile | None) -> tuple[str, str] | None:
    if profile is None or profile.row_count < _MIN_ROWS_TO_APPLY:
        return None
    prices: list[tuple[int, str]] = []
    quantities: list[tuple[int, str]] = []
    overrides = profile.role_overrides
    for c in profile.columns:
        if c.kind != "numeric" or c.nunique < 3:
            continue
        if overrides.get(c.name) == "price":
            prices.append((-1, c.name))
        elif overrides.get(c.name) == "quantity":
            quantities.append((-1, c.name))
        elif (rank := _matches(c.name, _PRICE_TOKENS)) is not None:
            prices.append((rank, c.name))
        elif (rank := _matches(c.name, _QTY_TOKENS)) is not None:
            quantities.append((rank, c.name))
    if prices and quantities:
        return sorted(prices)[0][1], sorted(quantities)[0][1]
    return None


def _fit(y: np.ndarray, x: np.ndarray, absorbed: int = 0) -> dict[str, float] | None:
    """OLS of y on x (x[:, 0] is ln price) with HC1 robust SEs. `absorbed` is
    the number of parameters swept out by within-demeaning. None if not
    identified."""
    from scipy import stats

    n, k = x.shape
    dof = n - k - absorbed
    if dof < 1 or np.linalg.matrix_rank(x) < k:
        return None
    xtx_inv = np.linalg.inv(x.T @ x)
    beta = xtx_inv @ (x.T @ y)
    resid = y - x @ beta
    meat = (x * (resid**2)[:, None]).T @ x
    cov = (n / dof) * xtx_inv @ meat @ xtx_inv
    se = math.sqrt(max(float(cov[0, 0]), 0.0))
    e = float(beta[0])
    tcrit = float(stats.t.ppf(0.975, dof))
    ssr = float(resid @ resid)
    tss = float(((y - y.mean()) ** 2).sum()) if absorbed == 0 else float(y @ y)
    return {
        "elasticity": e, "se": se, "ci_low": e - tcrit * se, "ci_high": e + tcrit * se,
        "p_value": float(2 * stats.t.sf(abs(e / se), dof)) if se > 0 else 0.0,
        "r2": 1 - ssr / tss if tss > 0 else 0.0, "dof": float(dof),
    }


def _month_dummies(months: pd.Series) -> np.ndarray:
    """Month-of-year indicators, first observed month dropped."""
    d = pd.get_dummies(months, dtype=float)
    return np.asarray(d.iloc[:, 1:].to_numpy(dtype=float))


def _units_change(e: float, rise: float = 0.10) -> float:
    return float((1 + rise) ** e - 1)


def _revenue_change(e: float, rise: float = 0.10) -> float:
    return float((1 + rise) ** (1 + e) - 1)


def _flags(row: dict[str, Any]) -> list[str]:
    flags: list[str] = []
    if row["ci_low"] <= 0 <= row["ci_high"]:
        flags.append("ci_spans_zero")
    if row["elasticity"] > 0:
        flags.append("positive")
    elif row["elasticity"] > -_NEAR_ZERO:
        flags.append("near_zero")
    return flags


class PriceElasticityTool(BaseTool):
    """How strongly units sold respond to price."""

    name = "price_elasticity_analysis"
    description = (
        "Estimate price elasticity of demand: how much units sold change when price changes "
        "(log-log regression with robust confidence intervals), per product and pooled, with "
        "seasonality removed when a date column is given. Says in plain words what a 10% "
        "price rise does to units sold and revenue. Needs a price column and a quantity column."
    )

    def applies_to(self, profile: DatasetProfile | None, metadata: DatasetMetadata | None) -> float:
        return 0.6 if _detect_columns(profile) else 0.0

    def default_params(
        self, profile: DatasetProfile | None, metadata: DatasetMetadata | None
    ) -> dict[str, Any]:
        found = _detect_columns(profile)
        if not found or profile is None:
            return {}
        params: dict[str, Any] = {"price_column": found[0], "quantity_column": found[1]}
        if profile.datetime_cols:
            params["date_column"] = profile.datetime_cols[0]
        return params

    def execute(  # type: ignore[override]
        self,
        file_path: str,
        price_column: str | None = None,
        quantity_column: str | None = None,
        product_column: str | None = None,
        date_column: str | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        if not price_column or not quantity_column:
            raise ToolExecutionError("Pass price_column and quantity_column (numeric).")
        df = _read_df(file_path)
        for col in (price_column, quantity_column, product_column, date_column):
            if col and col not in df.columns:
                raise ToolExecutionError(f"Column '{col}' not found in dataset.")
        if price_column == quantity_column:
            raise ToolExecutionError("price_column and quantity_column must be different columns.")

        frame = pd.DataFrame({
            "p": pd.to_numeric(df[price_column], errors="coerce"),
            "q": pd.to_numeric(df[quantity_column], errors="coerce"),
            "g": df[product_column].astype(str).where(df[product_column].notna()) if product_column else _ALL,
        })
        if date_column:
            dates = pd.to_datetime(df[date_column], errors="coerce", format="mixed")
            frame["m"] = dates.dt.month
        n_rows = len(frame)
        frame = frame[(frame["p"] > 0) & (frame["q"] > 0)].dropna(subset=["p", "q", "g"])
        n_positive = len(frame)
        if n_positive < _MIN_OBS:
            raise ToolExecutionError(
                f"Only {n_positive} rows have a positive '{price_column}' and '{quantity_column}'; need at least {_MIN_OBS}."
            )
        sampled = n_positive > _MAX_ROWS
        if sampled:
            frame = frame.sample(_MAX_ROWS, random_state=0)
        frame = frame.assign(lp=np.log(frame["p"]), lq=np.log(frame["q"]), rev=frame["p"] * frame["q"])
        use_months = bool(date_column) and frame["m"].notna().mean() > 0.9
        if use_months:
            frame = frame.dropna(subset=["m"])

        grouped = frame.groupby("g")
        stats_by = grouped.agg(n=("p", "size"), rev=("rev", "sum"), pmean=("p", "mean"), pstd=("p", "std"))
        stats_by["cv"] = stats_by["pstd"] / stats_by["pmean"]
        total_products = len(stats_by)
        eligible = stats_by[(stats_by["n"] >= _MIN_OBS) & (stats_by["cv"] >= _MIN_PRICE_CV)]
        skipped = total_products - len(eligible)
        chosen = eligible.sort_values("rev", ascending=False).head(_MAX_PRODUCTS)
        if chosen.empty:
            raise ToolExecutionError(
                f"No product has at least {_MIN_OBS} observations and price variation of {_MIN_PRICE_CV:.0%} or more; "
                "elasticity cannot be estimated from a price that never moves."
            )

        products: list[dict[str, Any]] = []
        for name, info in chosen.iterrows():
            sub = frame[frame["g"] == name]
            y = sub["lq"].to_numpy(dtype=float)
            ones = np.ones((len(sub), 1))
            fit = None
            seasonal = False
            if use_months:
                dummies = _month_dummies(sub["m"].astype(int))
                if dummies.shape[1] and len(sub) - 2 - dummies.shape[1] >= 3:
                    fit = _fit(y, np.hstack([sub[["lp"]].to_numpy(dtype=float), ones, dummies]))
                    seasonal = fit is not None
            if fit is None:
                fit = _fit(y, np.hstack([sub[["lp"]].to_numpy(dtype=float), ones]))
            if fit is None:
                continue
            row = {
                "product": str(name), "n": int(info["n"]), "price_cv": round(float(info["cv"]), 4),
                "revenue": round(float(info["rev"]), 2), "seasonality_adjusted": seasonal,
                "elasticity": round(fit["elasticity"], 3), "se": round(fit["se"], 4),
                "ci_low": round(fit["ci_low"], 3), "ci_high": round(fit["ci_high"], 3),
                "r2": round(fit["r2"], 3), "p_value": round(fit["p_value"], 6),
                "units_change_10pct_rise": round(_units_change(fit["elasticity"]), 4),
                "revenue_change_10pct_rise": round(_revenue_change(fit["elasticity"]), 4),
            }
            row["flags"] = _flags(row)
            products.append(row)
        if not products:
            raise ToolExecutionError("The price and quantity columns are collinear within every product; nothing to estimate.")

        pooled = self._pooled(frame, [p["product"] for p in products], use_months) if len(products) >= _POOLED_MIN_PRODUCTS else None

        result: dict[str, Any] = {
            "price_column": price_column, "quantity_column": quantity_column,
            "product_column": product_column, "date_column": date_column if use_months else None,
            "n_rows": int(n_rows), "n_used": len(frame), "sampled": sampled,
            "n_products_total": int(total_products), "n_products_skipped": int(skipped),
            "products": products, "pooled": pooled,
        }
        chart = self._chart(products, pooled)
        if chart is not None:
            result["chart"] = chart
        result["summary"] = self._summary(result)
        return result

    @staticmethod
    def _pooled(frame: pd.DataFrame, names: list[str], use_months: bool) -> dict[str, Any] | None:
        sub = frame[frame["g"].isin(names)]
        cols = sub[["lp", "lq"]].copy()
        n_dummy = 0
        if use_months:
            d = pd.get_dummies(sub["m"].astype(int), dtype=float, prefix="m").iloc[:, 1:]
            cols = pd.concat([cols, d.set_axis(sub.index)], axis=1)
            n_dummy = d.shape[1]
        demeaned = cols - cols.groupby(sub["g"]).transform("mean")
        y = demeaned["lq"].to_numpy(dtype=float)
        x_cols = ["lp"] + [c for c in demeaned.columns if c.startswith("m_")]
        fit = _fit(y, demeaned[x_cols].to_numpy(dtype=float), absorbed=len(names)) if n_dummy else None
        seasonal = fit is not None
        if fit is None:
            fit = _fit(y, demeaned[["lp"]].to_numpy(dtype=float), absorbed=len(names))
        if fit is None:
            return None
        e = fit["elasticity"]
        pooled: dict[str, Any] = {
            "n": len(sub), "n_products": len(names), "seasonality_adjusted": seasonal,
            "elasticity": round(e, 3), "se": round(fit["se"], 4),
            "ci_low": round(fit["ci_low"], 3), "ci_high": round(fit["ci_high"], 3),
            "r2_within": round(fit["r2"], 3), "p_value": round(fit["p_value"], 6),
            "units_change_10pct_rise": round(_units_change(e), 4),
            "revenue_change_10pct_rise": round(_revenue_change(e), 4),
        }
        pooled["flags"] = _flags(pooled)
        return pooled

    @staticmethod
    def _chart(products: list[dict[str, Any]], pooled: dict[str, Any] | None) -> dict[str, Any] | None:
        rows = [
            {"product": p["product"], "elasticity": p["elasticity"], "ci_low": p["ci_low"], "ci_high": p["ci_high"]}
            for p in products
        ]
        if pooled is not None:
            rows.append({
                "product": "All (pooled)", "elasticity": pooled["elasticity"],
                "ci_low": pooled["ci_low"], "ci_high": pooled["ci_high"],
            })
        if len(rows) < 2:
            return None
        rows.sort(key=lambda r: r["elasticity"])
        clean, _error = validate_chart_spec({
            "type": "dot_ci", "data": rows, "x": "product", "y": "elasticity",
            "y_lower": "ci_low", "y_upper": "ci_high",
            "title": "Price elasticity by product (95% interval)",
            "y_title": "Elasticity (% change in units per 1% price rise)", "y_format": "number",
            "annotations": [{"y": -1, "label": "-1: revenue-neutral"}, {"y": 0, "label": "0: no response"}],
            "caption": "More negative = customers react more to price. Intervals crossing 0 mean no reliable effect.",
        })
        return clean

    @staticmethod
    def _describe(label: str, r: dict[str, Any]) -> str:
        e = r["elasticity"]
        units = abs(r["units_change_10pct_rise"]) * 100
        revenue = r["revenue_change_10pct_rise"] * 100
        if "ci_spans_zero" in r["flags"]:
            return (
                f"No reliable link between price and units sold for {label}: elasticity {e:+.1f} "
                f"(plausible range {r['ci_low']:+.1f} to {r['ci_high']:+.1f})."
            )
        if e > 0:
            return (
                f"Higher prices went with more units sold for {label} (elasticity {e:+.1f}), which usually "
                "reflects promotions or stock-outs rather than a real preference for higher prices."
            )
        if "near_zero" in r["flags"]:
            return (
                f"Units sold barely respond to price for {label} (elasticity {e:+.1f}), which can also "
                "reflect promotions or stock-outs moving price and demand together."
            )
        mood = "customers are price sensitive" if e < -1 else "customers are not very price sensitive"
        direction = "fall" if revenue < 0 else "rise"
        return (
            f"A 10% price rise cuts units sold by about {units:.0f}% for {label} (elasticity {e:.1f}: {mood}); "
            f"revenue would {direction} about {abs(revenue):.1f}%."
        )

    @classmethod
    def _summary(cls, r: dict[str, Any]) -> str:
        head = r["pooled"] if r["pooled"] is not None else r["products"][0]
        label = "products overall" if r["pooled"] is not None else head["product"]
        text = cls._describe(label, head)
        tail = f" Estimated from {r['n_used']:,} rows"
        if r["sampled"]:
            tail += f" (random sample of {_MAX_ROWS:,})"
        tail += f", {len(r['products'])} product(s) with enough price variation"
        if r["n_products_skipped"]:
            tail += f", {r['n_products_skipped']} skipped"
        return text + tail + "."

    def findings(
        self,
        output: dict[str, Any],
        profile: DatasetProfile | None,
        metadata: DatasetMetadata | None,
    ) -> list[Finding]:
        price, qty = str(output.get("price_column")), str(output.get("quantity_column"))
        product_col = output.get("product_column")
        entries: list[tuple[str, dict[str, Any]]] = []
        if output.get("pooled"):
            entries.append(("products overall", output["pooled"]))
        entries.extend((p["product"], p) for p in output.get("products", []))
        results: list[Finding] = []
        for label, r in entries[:_MAX_FINDINGS]:
            n = r["n"]
            caveats = [
                "Assumes a constant elasticity across the price range; the real response can change at very high or low prices.",
                "Price moves in this data were not randomised: promotions, stock-outs and seasonality can bias the estimate toward zero or even positive.",
            ]
            if r["flags"] and ("positive" in r["flags"] or "near_zero" in r["flags"]):
                caveats.append("A positive or near-zero estimate is more likely promotion or stock-out confounding than a real preference.")
            if "ci_spans_zero" in r["flags"]:
                caveats.append("The 95% interval includes zero, so the direction of the effect is not established.")
            if n < 30:
                caveats.append(f"Only {n} observations, so the estimate is imprecise.")
            if output.get("sampled"):
                caveats.append(f"Fitted on a random sample of {output.get('n_used'):,} of the eligible rows.")
            if not r.get("seasonality_adjusted") and output.get("date_column"):
                caveats.append("Too few observations to adjust for month-of-year seasonality here.")
            if not output.get("date_column"):
                caveats.append("No date column was used, so seasonality is not removed.")
            pooled = label == "products overall"
            confidence = 0.85 if r["p_value"] < 0.01 else 0.7 if r["p_value"] < 0.05 else 0.35
            results.append(Finding(
                finding_id=f"elasticity_{price}_{label}".replace(" ", "_")[:120],
                kind="elasticity",
                headline=self._describe(label, r),
                detail=(
                    f"Log-log fit of {qty} on {price}: elasticity {r['elasticity']:+.2f}, 95% interval "
                    f"{r['ci_low']:+.2f} to {r['ci_high']:+.2f}, from {n:,} observations"
                    + (f" across {r['n_products']} products with product fixed effects." if pooled else ".")
                ),
                evidence={
                    "elasticity": r["elasticity"], "ci_low": r["ci_low"], "ci_high": r["ci_high"],
                    "units_change_10pct_rise": r["units_change_10pct_rise"],
                    "revenue_change_10pct_rise": r["revenue_change_10pct_rise"],
                    "r2": r.get("r2", r.get("r2_within")), "n": n, "product": label,
                },
                source_tool=self.name, measure=qty, dimension=None if pooled or not product_col else str(product_col),
                level=None if pooled else label,
                effect=r["elasticity"], effect_kind="elasticity", p_value=r["p_value"], confidence=confidence,
                surprise=min(1.0, abs(r["elasticity"]) / 3.0),
                caveats=caveats, chart_hint=output.get("chart"),
            ))
        return results

    def get_schema(self) -> dict[str, Any]:
        return {
            "file_path": {"type": "string", "description": "Path to the dataset.", "required": True},
            "price_column": {"type": "string", "description": "Numeric unit price column.", "required": True},
            "quantity_column": {"type": "string", "description": "Numeric units-sold column.", "required": True},
            "product_column": {
                "type": "string",
                "description": "Product / SKU column: elasticity is estimated per product (top 15 by revenue). Omit to treat all rows as one product.",
                "required": False,
            },
            "date_column": {
                "type": "string",
                "description": "Date column; adds month-of-year dummies so seasonality is not mistaken for a price effect.",
                "required": False,
            },
        }
