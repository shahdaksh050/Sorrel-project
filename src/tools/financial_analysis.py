"""
Financial Analysis Tool — Execution Layer.

Price-series measures for datasets the domain layer identified as
financial (src/core/domains.py): returns, annualised volatility, drawdown
and return autocorrelation.

Why a dedicated tool: the generic numeric summary reports the mean and
standard deviation of a *price*, which are close to meaningless — a price
level is non-stationary, so its mean describes where the series happened
to sit over the sample window and nothing more. The meaningful quantities
live in the returns, and computing them requires knowing which column is
the price and which is the date. That is exactly what the domain layer
resolves.

Every annualised figure depends on the sampling frequency, which is
inferred from the observed date spacing and reported alongside the number
so the assumption is auditable rather than buried.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from src.core.domains import domain_confidence, resolve_column
from src.core.findings import Finding
from src.tools.base import BaseTool, ToolExecutionError
from src.tools.data_processing import _read_df

if TYPE_CHECKING:
    from src.core.memory import DatasetMetadata
    from src.core.profiler import DatasetProfile

#: Median spacing between observations -> periods per year. Trading-day
#: counts (252) rather than calendar days, since price series skip weekends
#: and holidays; the label is reported so the choice is visible.
_FREQUENCY_TABLE: tuple[tuple[float, int, str], ...] = (
    (1.5, 252, "daily"),
    (8.0, 52, "weekly"),
    (16.0, 26, "fortnightly"),
    (45.0, 12, "monthly"),
    (120.0, 4, "quarterly"),
    (400.0, 1, "annual"),
)

#: Below this many return observations, annualised volatility and Sharpe
#: are too noisy to report as point estimates.
_MIN_RETURNS_FOR_ANNUALISATION = 20


def _infer_periods_per_year(dates: pd.Series) -> tuple[int, str, float]:
    """(periods_per_year, label, median_gap_days) from observed spacing."""
    ordered = dates.dropna().sort_values()
    if len(ordered) < 2:
        return 252, "assumed daily (too few dates to infer)", 0.0
    gaps = ordered.diff().dropna().dt.total_seconds() / 86400.0
    gaps = gaps[gaps > 0]
    if gaps.empty:
        return 252, "assumed daily (all timestamps identical)", 0.0
    median_gap = float(gaps.median())
    for threshold, periods, label in _FREQUENCY_TABLE:
        if median_gap <= threshold:
            return periods, label, median_gap
    return 1, "sparser than annual", median_gap


def _max_drawdown(prices: pd.Series) -> tuple[float, float]:
    """(max drawdown as a negative fraction, peak value it fell from)."""
    running_peak = prices.cummax()
    drawdown = prices / running_peak - 1.0
    trough_idx = drawdown.idxmin()
    return float(drawdown.min()), float(running_peak.loc[trough_idx])


def _series_metrics(
    frame: pd.DataFrame, date_col: str, price_col: str
) -> dict[str, Any] | None:
    """Return/risk measures for one instrument's price series."""
    ordered = frame[[date_col, price_col]].dropna().sort_values(date_col)
    prices = ordered[price_col].astype(float)
    # A non-positive price makes log returns undefined and simple returns
    # meaningless; drop rather than silently produce inf.
    ordered = ordered[prices > 0]
    prices = prices[prices > 0]
    if len(prices) < 3:
        return None

    returns = prices.pct_change().dropna()
    if returns.empty:
        return None

    periods_per_year, freq_label, median_gap = _infer_periods_per_year(ordered[date_col])
    total_return = float(prices.iloc[-1] / prices.iloc[0] - 1.0)
    span_days = float(
        (ordered[date_col].iloc[-1] - ordered[date_col].iloc[0]).total_seconds() / 86400.0
    )
    years = span_days / 365.25

    metrics: dict[str, Any] = {
        "observations": len(prices),
        "start_date": str(ordered[date_col].iloc[0].date()),
        "end_date": str(ordered[date_col].iloc[-1].date()),
        "start_price": round(float(prices.iloc[0]), 6),
        "end_price": round(float(prices.iloc[-1]), 6),
        "total_return_pct": round(total_return * 100, 4),
        "sampling_frequency": freq_label,
        "periods_per_year": periods_per_year,
        "median_gap_days": round(median_gap, 3),
        "mean_period_return_pct": round(float(returns.mean()) * 100, 6),
    }

    if years > 0 and (1.0 + total_return) > 0:
        metrics["annualised_return_pct"] = round(
            ((1.0 + total_return) ** (1.0 / years) - 1.0) * 100, 4
        )
    metrics["period_years"] = round(years, 3)

    if len(returns) >= _MIN_RETURNS_FOR_ANNUALISATION:
        volatility = float(returns.std(ddof=1)) * float(np.sqrt(periods_per_year))
        metrics["annualised_volatility_pct"] = round(volatility * 100, 4)
        if volatility > 0:
            mean_annual = float(returns.mean()) * periods_per_year
            # Sharpe at a zero risk-free rate — stated, not assumed away.
            metrics["sharpe_ratio_rf0"] = round(mean_annual / volatility, 4)
    else:
        metrics["annualised_volatility_pct"] = None
        metrics["volatility_note"] = (
            f"Only {len(returns)} return observations — too few to annualise "
            f"reliably (need {_MIN_RETURNS_FOR_ANNUALISATION})."
        )

    max_dd, peak = _max_drawdown(prices)
    metrics["max_drawdown_pct"] = round(max_dd * 100, 4)
    metrics["max_drawdown_peak_price"] = round(peak, 6)

    metrics["positive_period_share_pct"] = round(float((returns > 0).mean()) * 100, 2)
    metrics["best_period_return_pct"] = round(float(returns.max()) * 100, 4)
    metrics["worst_period_return_pct"] = round(float(returns.min()) * 100, 4)

    if len(returns) >= 3:
        lag1 = float(returns.autocorr(lag=1))
        if not np.isnan(lag1):
            metrics["return_autocorrelation_lag1"] = round(lag1, 4)
            metrics["autocorrelation_reading"] = (
                "momentum (returns persist)" if lag1 > 0.1
                else "mean-reverting (returns reverse)" if lag1 < -0.1
                else "no meaningful serial dependence"
            )
    return metrics


class FinancialAnalysisTool(BaseTool):
    """Return, volatility and drawdown analysis for price-series datasets."""

    name = "financial_analysis"
    description = (
        "Analyse a price/market series: period returns, annualised volatility, "
        "cumulative and annualised return, maximum drawdown, Sharpe ratio and "
        "return autocorrelation. Handles a single instrument or a multi-symbol "
        "panel. Use for stock/price/NAV data where the mean of the raw price "
        "is not a meaningful statistic."
    )

    def applies_to(self, profile: DatasetProfile | None, metadata: DatasetMetadata | None) -> float:
        return domain_confidence(profile, "financial")

    def execute(  # type: ignore[override]
        self,
        file_path: str,
        date_column: str | None = None,
        price_column: str | None = None,
        symbol_column: str | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        df = _read_df(file_path)

        # Sequential resolution so one column cannot fill two roles.
        claimed: set[str] = {c for c in (date_column, price_column, symbol_column) if c}
        date_column = date_column or resolve_column(
            df, ("date", "timestamp", "datetime", "time", "day"), claimed)
        claimed.add(date_column or "")
        price_column = price_column or resolve_column(
            df, ("adj_close", "close", "closing", "price", "nav"), claimed)
        if date_column is None or date_column not in df.columns:
            raise ToolExecutionError(
                "No date column found. Pass date_column explicitly."
            )
        if price_column is None or price_column not in df.columns:
            raise ToolExecutionError(
                "No price column found. Pass price_column explicitly."
            )
        if symbol_column is not None and symbol_column not in df.columns:
            raise ToolExecutionError(f"Symbol column '{symbol_column}' is not in the dataset.")

        working = df.copy()
        working[date_column] = pd.to_datetime(working[date_column], errors="coerce")
        working = working.dropna(subset=[date_column])
        if working.empty:
            raise ToolExecutionError(
                f"Column '{date_column}' contains no parseable dates."
            )
        if not pd.api.types.is_numeric_dtype(working[price_column]):
            working[price_column] = pd.to_numeric(working[price_column], errors="coerce")
        if working[price_column].notna().sum() < 3:
            raise ToolExecutionError(
                f"Column '{price_column}' has fewer than 3 usable numeric prices."
            )

        if symbol_column:
            per_symbol: dict[str, Any] = {}
            for symbol, group in working.groupby(symbol_column, sort=True):
                metrics = _series_metrics(group, date_column, price_column)
                if metrics is not None:
                    per_symbol[str(symbol)] = metrics
            if not per_symbol:
                raise ToolExecutionError(
                    "No instrument had enough usable observations to analyse."
                )
            ranked = sorted(
                per_symbol.items(),
                key=lambda kv: kv[1]["total_return_pct"],
                reverse=True,
            )
            best_name, best = ranked[0]
            worst_name, worst = ranked[-1]
            return {
                "summary": (
                    f"Analysed {len(per_symbol)} instrument(s) on '{price_column}'. "
                    f"Best: {best_name} ({best['total_return_pct']:+.2f}%); "
                    f"worst: {worst_name} ({worst['total_return_pct']:+.2f}%)."
                ),
                "mode": "panel",
                "date_column": date_column,
                "price_column": price_column,
                "symbol_column": symbol_column,
                "instrument_count": len(per_symbol),
                "per_symbol": per_symbol,
                "best_performer": {"symbol": best_name, **best},
                "worst_performer": {"symbol": worst_name, **worst},
            }

        metrics = _series_metrics(working, date_column, price_column)
        if metrics is None:
            raise ToolExecutionError(
                "Fewer than 3 usable observations with a positive price."
            )
        volatility = metrics.get("annualised_volatility_pct")
        volatility_text = (
            f"{volatility:.2f}% annualised volatility"
            if volatility is not None
            else "volatility not annualised (too few observations)"
        )
        return {
            "summary": (
                f"'{price_column}' returned {metrics['total_return_pct']:+.2f}% over "
                f"{metrics['period_years']:.2f} years ({metrics['sampling_frequency']} data); "
                f"{volatility_text}; max drawdown {metrics['max_drawdown_pct']:.2f}%."
            ),
            "mode": "single_series",
            "date_column": date_column,
            "price_column": price_column,
            **metrics,
        }

    def findings(
        self,
        output: dict[str, Any],
        profile: DatasetProfile | None,
        metadata: DatasetMetadata | None,
    ) -> list[Finding]:
        results: list[Finding] = []

        def _series_findings(metrics: dict[str, Any], label: str, prefix: str) -> list[Finding]:
            found: list[Finding] = []
            total_return = metrics.get("total_return_pct")
            if total_return is not None:
                found.append(
                    Finding(
                        finding_id=f"{prefix}_return",
                        kind="trend",
                        headline=(
                            f"{label} returned {total_return:+.2f}% over "
                            f"{metrics.get('period_years', 0):.2f} years "
                            f"({metrics.get('sampling_frequency', 'unknown')} data)"
                        ),
                        detail=(
                            f"From {metrics.get('start_date')} ({metrics.get('start_price')}) "
                            f"to {metrics.get('end_date')} ({metrics.get('end_price')})."
                        ),
                        evidence={
                            "total_return_pct": total_return,
                            "annualised_return_pct": metrics.get("annualised_return_pct"),
                            "period_years": metrics.get("period_years"),
                        },
                        measure="total_return",
                        effect=round(total_return / 100.0, 4),
                        effect_kind="pct",
                        confidence=0.7,
                        surprise=0.3,
                    )
                )

            max_dd = metrics.get("max_drawdown_pct")
            if max_dd is not None and abs(max_dd) > 0.5:
                found.append(
                    Finding(
                        finding_id=f"{prefix}_drawdown",
                        kind="financial",
                        headline=f"{label} suffered a maximum drawdown of {max_dd:.2f}%",
                        detail=f"Measured from a peak price of {metrics.get('max_drawdown_peak_price')}.",
                        evidence={
                            "max_drawdown_pct": max_dd,
                            "max_drawdown_peak_price": metrics.get("max_drawdown_peak_price"),
                        },
                        measure="max_drawdown",
                        effect=round(max_dd / 100.0, 4),
                        effect_kind="pct",
                        confidence=0.65,
                        surprise=0.4,
                    )
                )

            autocorr_reading = metrics.get("autocorrelation_reading")
            lag1 = metrics.get("return_autocorrelation_lag1")
            if (
                autocorr_reading
                and autocorr_reading != "no meaningful serial dependence"
                and lag1 is not None
            ):
                found.append(
                    Finding(
                        finding_id=f"{prefix}_autocorr",
                        kind="trend",
                        headline=(
                            f"{label} returns show {autocorr_reading} "
                            f"(lag-1 autocorrelation {lag1:+.2f})"
                        ),
                        evidence={"return_autocorrelation_lag1": lag1},
                        measure="return_autocorrelation",
                        effect=round(lag1, 4),
                        effect_kind="r",
                        confidence=0.5,
                        surprise=0.5,
                    )
                )

            vol_note = metrics.get("volatility_note")
            if vol_note:
                found.append(
                    Finding(
                        finding_id=f"{prefix}_vol_caveat",
                        kind="method_fit",
                        headline=f"{label}: volatility not annualised",
                        detail=vol_note,
                        evidence={"annualised_volatility_pct": metrics.get("annualised_volatility_pct")},
                        confidence=0.9,
                        surprise=0.0,
                        caveats=[vol_note],
                    )
                )
            return found

        mode = output.get("mode")
        if mode == "panel":
            best = output.get("best_performer") or {}
            worst = output.get("worst_performer") or {}
            best_symbol = best.get("symbol")
            worst_symbol = worst.get("symbol")
            best_return = best.get("total_return_pct")
            worst_return = worst.get("total_return_pct")
            if best_return is not None and worst_return is not None and best_symbol != worst_symbol:
                spread = best_return - worst_return
                results.append(
                    Finding(
                        finding_id=f"{self.name}_panel_spread",
                        kind="segment_lift",
                        headline=(
                            f"'{best_symbol}' returned {best_return:+.2f}% vs "
                            f"'{worst_symbol}' at {worst_return:+.2f}% "
                            f"({output.get('instrument_count')} instruments compared)"
                        ),
                        detail="Spread between the best- and worst-performing instruments in the panel.",
                        evidence={
                            "best_performer": best_symbol,
                            "best_return_pct": best_return,
                            "worst_performer": worst_symbol,
                            "worst_return_pct": worst_return,
                        },
                        measure="total_return",
                        dimension=output.get("symbol_column"),
                        level=best_symbol,
                        effect=round(spread / 100.0, 4),
                        effect_kind="pct",
                        confidence=0.7,
                        surprise=0.4,
                    )
                )
            if best:
                results.extend(
                    _series_findings(best, f"Best performer '{best_symbol}'", f"{self.name}_best")
                )
            if worst and worst_symbol != best_symbol:
                results.extend(
                    _series_findings(worst, f"Worst performer '{worst_symbol}'", f"{self.name}_worst")
                )
        else:
            price_col = output.get("price_column", "series")
            results.extend(_series_findings(output, f"'{price_col}'", self.name))

        return results

    def get_schema(self) -> dict[str, Any]:
        return {
            "file_path": {
                "type": "string",
                "description": "Path to the dataset.",
                "required": True,
            },
            "date_column": {
                "type": "string",
                "description": "Date/timestamp column. Auto-detected when omitted.",
                "required": False,
            },
            "price_column": {
                "type": "string",
                "description": (
                    "Price column to analyse (close/adjusted close/NAV). "
                    "Auto-detected when omitted."
                ),
                "required": False,
            },
            "symbol_column": {
                "type": "string",
                "description": (
                    "Instrument/ticker column. Supply for a multi-symbol panel so "
                    "each instrument is analysed separately."
                ),
                "required": False,
            },
        }

