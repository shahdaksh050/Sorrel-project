"""
Forecast Analysis Tool — Execution Layer.

"Where is this measure heading, and how sure are we?" The measure is
aggregated to a regular period series (hourly / daily / weekly / monthly /
quarterly, summed when a total is meaningful, averaged otherwise), short gaps
(3 periods or fewer) are interpolated, and a handful of cheap models compete
in a backtest on the most recent stretch of the series:

  - naive (repeat the last value) and seasonal naive;
  - a linear trend;
  - exponential smoothing (level, additive trend, damped trend, and additive
    seasonality when there are 2+ full seasons).

The model with the lowest MASE (sMAPE breaks ties) is refitted on all the
data and produces the forecast, with a 95% interval of +/- 1.96 * sigma *
sqrt(step) where sigma is the one-step residual spread. Nothing is trained
offline or stored; every call fits from scratch on at most 5,000 periods.
When the winner is no better than repeating the last value, the output says
so instead of dressing it up.
"""
from __future__ import annotations

import math
import warnings
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from src.core.findings import Finding
from src.core.profiler import profile_dataframe
from src.core.stats_utils import is_partial_final_period, measure_aggregation
from src.tools.base import BaseTool, ToolExecutionError
from src.tools.data_processing import _read_df
from src.tools.time_series import _autodetect_datetime_column

if TYPE_CHECKING:
    from src.core.memory import DatasetMetadata
    from src.core.profiler import ColumnProfile, DatasetProfile

_MAX_PERIODS = 5_000
_MIN_PERIODS = 12
_SHORT_HISTORY = 24
_MAX_HORIZON = 52
_DEFAULT_HORIZON = 12
_MAX_GAP_FILL = 3
_MAX_GROUPS = 3
_ETS_WINDOW = 1_500
_BACKTEST_SHARE = 0.2
_WIDE_INTERVAL = 0.3
_MAX_CHART_ROWS = 480
_MIN_CHART_HISTORY = 10

#: alias -> (period length in days, seasonal period, adjective, noun)
_GRAINS: dict[str, tuple[float, int, str, str]] = {
    "h": (1 / 24, 24, "Hourly", "hour"),
    "D": (1.0, 7, "Daily", "day"),
    "W": (7.0, 52, "Weekly", "week"),
    "MS": (30.44, 12, "Monthly", "month"),
    "QS": (91.31, 4, "Quarterly", "quarter"),
    "YS": (365.25, 0, "Yearly", "year"),
}
_ALIAS_ORDER = ("h", "D", "W", "MS", "QS", "YS")
_FREQ_LABEL_DAYS = {
    "hourly": 1 / 24, "daily": 1.0, "weekly": 7.0, "monthly": 30.44, "quarterly": 91.3, "yearly": 365.25,
}

Model = Callable[[np.ndarray, int], tuple[np.ndarray, np.ndarray] | None]


def _num(value: float, unit_hint: str | None = None) -> str:
    a = abs(value)
    text = f"{a:,.0f}" if a >= 100 else f"{a:,.2f}"
    text = f"${text}" if unit_hint == "currency" else text
    return f"-{text}" if value < 0 else text


def _r(value: float, digits: int = 4) -> float:
    return round(float(value), digits)


def _infer_alias(dates: pd.Series) -> str:
    unique = pd.Series(pd.to_datetime(dates.dropna().unique())).sort_values()
    if len(unique) < 3:
        return "D"
    med = float((unique.diff().dropna().dt.total_seconds() / 86_400).median())
    if med < (1 / 24) * 0.75:
        return "D"  # event-level timestamps: aggregate per day
    if med <= (1 / 24) * 1.5:
        return "h"
    if med <= 1.5:
        return "D"
    if med <= 10:
        return "W"
    if med <= 45:
        return "MS"
    if med <= 135:
        return "QS"
    return "YS"


def _coarsen(alias: str, span_days: float) -> tuple[str, bool]:
    moved = False
    while span_days / _GRAINS[alias][0] > _MAX_PERIODS and alias != "YS":
        alias = _ALIAS_ORDER[_ALIAS_ORDER.index(alias) + 1]
        moved = True
    return alias, moved


def _label(ts: pd.Timestamp, alias: str) -> str:
    fmt = {"h": "%Y-%m-%d %H:00", "MS": "%Y-%m", "YS": "%Y"}.get(alias, "%Y-%m-%d")
    return str(ts.strftime(fmt))


def _iso(ts: pd.Timestamp, alias: str) -> str:
    return str(ts.strftime("%Y-%m-%dT%H:%M:%S" if alias == "h" else "%Y-%m-%d"))


def _datetime_span_periods(profile: DatasetProfile, column: str) -> float:
    col = next((c for c in profile.columns if c.name == column), None)
    if col is None:
        return 0.0
    try:
        start = pd.Timestamp(col.stats.get("start"))
        end = pd.Timestamp(col.stats.get("end"))
    except (ValueError, TypeError):
        return 0.0
    if pd.isna(start) or pd.isna(end):
        return 0.0
    step = _FREQ_LABEL_DAYS.get(str(col.stats.get("frequency")), 1.0)
    return float((end - start).total_seconds() / 86_400 / step)


def _measure_candidates(profile: DatasetProfile, exclude: set[str]) -> list[ColumnProfile]:
    measures = [c for c in profile.measures() if c.name not in exclude and c.nunique > 2]
    return sorted(measures, key=lambda c: (c.unit_hint != "currency", c.name))


def _build_series(
    frame: pd.DataFrame, alias: str, agg: str
) -> tuple[pd.Series, list[str], int, bool] | None:
    """Regular period series (tail block without holes), notes, n gap-filled
    periods and whether a partial last period was dropped. None if empty."""
    resampler = frame.set_index("d")["v"].resample(alias, label="left", closed="left")
    per_period = resampler.sum(min_count=1) if agg == "sum" else resampler.mean()
    counts = resampler.size()
    dropped_partial = False
    if len(counts) >= 2 and is_partial_final_period(counts, frame["d"].max(), alias):
        per_period, dropped_partial = per_period.iloc[:-1], True
    if per_period.empty:
        return None
    missing = per_period.isna()
    run_id = (missing != missing.shift()).cumsum()
    run_len = missing.groupby(run_id).transform("sum")
    fillable = missing & (run_len <= _MAX_GAP_FILL)
    interpolated = per_period.interpolate(limit_area="inside")
    filled = per_period.where(~fillable, interpolated)
    notes: list[str] = []
    n_filled = int(fillable.sum())
    if n_filled:
        notes.append(f"{n_filled} short gap period(s) were filled by interpolation.")
    holes = np.flatnonzero(filled.isna().to_numpy())
    if holes.size:
        tail = filled.iloc[int(holes[-1]) + 1:]
        notes.append(
            f"A gap of more than {_MAX_GAP_FILL} periods was left unfilled, so only the {len(tail)} periods "
            "after it were used."
        )
        filled = tail
    return filled, notes, n_filled, dropped_partial


def _smape(actual: np.ndarray, pred: np.ndarray) -> float:
    denom = np.abs(actual) + np.abs(pred)
    safe = np.where(denom == 0, 1.0, denom)
    return float(200.0 * np.mean(np.where(denom == 0, 0.0, np.abs(actual - pred) / safe)))


def _models(m: int, n_train: int, n_full: int) -> list[tuple[str, Model]]:
    def naive(x: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        return np.full(k, x[-1]), np.diff(x)

    def snaive(x: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        return x[-m:][np.arange(k) % m], x[m:] - x[:-m]

    def linear(x: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        t = np.arange(len(x), dtype=float)
        slope, intercept = np.polyfit(t, x, 1)
        fitted = slope * t + intercept
        return slope * (len(x) + np.arange(k, dtype=float)) + intercept, x - fitted

    def ets(trend: str | None, damped: bool, seasonal: bool) -> Model:
        def fit(x: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray] | None:
            from statsmodels.tsa.holtwinters import ExponentialSmoothing

            window = x[-_ETS_WINDOW:]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = ExponentialSmoothing(
                    window, trend=trend, damped_trend=damped if trend else False,
                    seasonal="add" if seasonal else None, seasonal_periods=m if seasonal else None,
                    initialization_method="estimated",
                ).fit()
                return np.asarray(res.forecast(k), dtype=float), np.asarray(res.resid, dtype=float)

        return fit

    models: list[tuple[str, Model]] = [("naive", naive)]
    if m >= 2 and n_train >= m and n_full >= 2 * m:
        models.append(("seasonal_naive", snaive))
    models.append(("linear_trend", linear))
    models.append(("ets_level", ets(None, False, False)))
    models.append(("ets_trend", ets("add", False, False)))
    models.append(("ets_damped_trend", ets("add", True, False)))
    if m >= 2 and n_full >= 2 * m and n_train >= m + 3:
        models.append(("ets_seasonal", ets(None, False, True)))
        models.append(("ets_damped_seasonal", ets("add", True, True)))
    return models


def _forecast_series(y: np.ndarray, m: int, horizon: int) -> dict[str, Any]:
    n = len(y)
    hb = max(1, min(horizon, int(_BACKTEST_SHARE * n)))
    train, test = y[:-hb], y[-hb:]
    scale = float(np.mean(np.abs(np.diff(train)))) if len(train) > 1 else 0.0
    if scale <= 0 and float(np.ptp(y)) == 0:
        raise ToolExecutionError("The series is constant, so there is nothing to forecast.")
    scale = scale if scale > 0 else float(np.std(train) or 1.0)

    scored: list[dict[str, Any]] = []
    models = _models(m, len(train), n)
    for name, fn in models:
        try:
            out = fn(train, hb)
        except Exception:
            continue
        if out is None:
            continue
        pred = np.asarray(out[0], dtype=float)
        if pred.shape != test.shape or not np.isfinite(pred).all():
            continue
        mae = float(np.mean(np.abs(test - pred)))
        scored.append({"model": name, "mae": mae, "mase": mae / scale, "smape": _smape(test, pred)})
    naive_row = next((s for s in scored if s["model"] == "naive"), None)
    if naive_row is None:
        raise ToolExecutionError("The series could not be backtested.")
    scored.sort(key=lambda s: (round(s["mase"], 3), s["smape"]))
    fitters = dict(models)

    best = naive_row
    fc = np.full(horizon, float(y[-1]))
    resid = np.diff(y)
    for row in scored:
        try:
            final = fitters[row["model"]](y, horizon)
        except Exception:
            continue
        if final is not None and np.isfinite(np.asarray(final[0])).all():
            best = row
            fc, resid = np.asarray(final[0], dtype=float), np.asarray(final[1], dtype=float)
            break
    resid = resid[np.isfinite(resid)]
    sigma = float(np.std(resid, ddof=1)) if len(resid) > 2 else float(np.std(np.diff(y)))
    if not math.isfinite(sigma):
        sigma = float(np.std(y))
    half = 1.96 * sigma * np.sqrt(np.arange(1, horizon + 1, dtype=float))
    lower, upper = fc - half, fc + half
    if float(np.min(y)) >= 0:
        fc, lower = np.maximum(fc, 0.0), np.maximum(lower, 0.0)
    return {
        "model": best["model"], "mase": best["mase"], "smape": best["smape"], "mae": best["mae"],
        "naive_mae": naive_row["mae"], "backtest_periods": hb, "candidates": scored,
        "forecast": fc, "lower": lower, "upper": upper, "sigma": sigma,
    }


def _series_result(name: str, index: pd.DatetimeIndex, y: np.ndarray, alias: str, m: int, horizon: int) -> dict[str, Any]:
    fit = _forecast_series(y, m, horizon)
    future = pd.date_range(start=index[-1], periods=horizon + 1, freq=alias)[1:]
    fc = fit["forecast"]
    steps = [
        {"period": _label(ts, alias), "forecast": _r(f), "lower": _r(lo), "upper": _r(hi)}
        for ts, f, lo, hi in zip(future, fc, fit["lower"], fit["upper"], strict=True)
    ]
    last = float(y[-1])
    end = float(fc[-1])
    half = float((fit["upper"][-1] - fit["lower"][-1]) / 2)
    return {
        "series": name,
        "n_periods": len(y),
        "last_period": _label(index[-1], alias),
        "last_value": _r(last),
        "model": fit["model"],
        "mase": _r(fit["mase"], 3),
        "smape": _r(fit["smape"], 2),
        "backtest_periods": fit["backtest_periods"],
        "naive_no_better": bool(fit["model"] == "naive" or fit["mase"] >= 1.0),
        "forecast_end_period": steps[-1]["period"],
        "forecast_end_value": _r(end),
        "forecast_end_lower": steps[-1]["lower"],
        "forecast_end_upper": steps[-1]["upper"],
        "interval_half_width": _r(half),
        "pct_vs_latest": _r((end - last) / abs(last), 4) if last != 0 else None,
        "wide_interval": bool(abs(end) > 0 and half / abs(end) >= _WIDE_INTERVAL) or abs(end) == 0,
        "forecast": steps,
        "candidates": [
            {"model": c["model"], "mase": _r(c["mase"], 3), "smape": _r(c["smape"], 2)} for c in fit["candidates"]
        ],
        "_index": index, "_y": y, "_fc": fc, "_lo": fit["lower"], "_hi": fit["upper"],
    }


def _chart(series: list[dict[str, Any]], alias: str, value_column: str, date_column: str, grain: str) -> dict[str, Any] | None:
    from src.core.chart_spec import validate_chart_spec

    horizon = len(series[0]["forecast"])
    per_series = _MAX_CHART_ROWS // len(series) - (horizon + 1)
    while per_series < _MIN_CHART_HISTORY and len(series) > 1:
        series = series[:-1]
        per_series = _MAX_CHART_ROWS // len(series) - (horizon + 1)
    if per_series < _MIN_CHART_HISTORY:
        return None
    rows: list[dict[str, Any]] = []
    for s in series:
        index, y = s["_index"], s["_y"]
        picks = np.arange(len(y))
        if len(y) > per_series:
            picks = np.unique(np.linspace(0, len(y) - 1, per_series).round().astype(int))
        for i in picks:
            rows.append({
                "period": _iso(index[int(i)], alias), "series": s["series"], "part": "History",
                "value": _r(y[int(i)]), "lower": None, "upper": None,
            })
        last = float(y[-1])
        rows.append({
            "period": _iso(index[-1], alias), "series": s["series"], "part": "Forecast",
            "value": _r(last), "lower": _r(last), "upper": _r(last),
        })
        future = pd.date_range(start=index[-1], periods=horizon + 1, freq=alias)[1:]
        for ts, f, lo, hi in zip(future, s["_fc"], s["_lo"], s["_hi"], strict=True):
            rows.append({
                "period": _iso(ts, alias), "series": s["series"], "part": "Forecast",
                "value": _r(f), "lower": _r(lo), "upper": _r(hi),
            })
    multi = len(series) > 1
    x_enc = {"field": "period", "type": "temporal", "title": date_column}
    y_title = f"{value_column} per {_GRAINS[alias][3]}"
    color = {"color": {"field": "series", "type": "nominal", "title": None}} if multi else {}
    band = {
        "transform": [{"filter": {"field": "part", "equal": "Forecast"}}],
        "mark": {"type": "area", "opacity": 0.18},
        "encoding": {
            "x": x_enc,
            "y": {"field": "lower", "type": "quantitative", "title": y_title, "scale": {"zero": False}},
            "y2": {"field": "upper"},
            **color,
        },
    }
    line = {
        "mark": {"type": "line"},
        "encoding": {
            "x": x_enc,
            "y": {"field": "value", "type": "quantitative", "title": y_title, "scale": {"zero": False}},
            "strokeDash": {
                "field": "part", "type": "nominal", "legend": None,
                "scale": {"domain": ["History", "Forecast"], "range": [[1, 0], [6, 4]]},
            },
            "tooltip": [
                {"field": "period", "type": "temporal", "title": date_column},
                {"field": "value", "type": "quantitative", "title": value_column},
                {"field": "part", "type": "nominal", "title": "Type"},
            ],
            **color,
        },
    }
    spec = {
        "type": "vega_lite",
        "title": f"{grain} {value_column}: history and forecast (95% range shaded)",
        "vega_lite": {"data": {"values": rows}, "height": 260, "layer": [band, line]},
    }
    clean, _error = validate_chart_spec(spec)
    return clean


def _series_caveats(s: dict[str, Any], unit: str | None) -> list[str]:
    out: list[str] = []
    if s["naive_no_better"]:
        out.append(
            "The forecast is no better than repeating the last value in the backtest, so treat it as a "
            "flat guide, not a prediction of change."
        )
    if s["wide_interval"]:
        out.append(
            f"The range is wide (give or take {_num(s['interval_half_width'], unit)} by "
            f"{s['forecast_end_period']}), so the exact figure is uncertain."
        )
    return out


def _headline(s: dict[str, Any], grain: str, value_column: str, agg: str, unit: str | None, group: str | None) -> str:
    what = f"{grain} {'average ' if agg == 'mean' else ''}{value_column}"
    if group and s["series"] != "Total":
        what += f" for {group} {s['series']}"
    pct = s["pct_vs_latest"]
    pct_text = f" ({pct * 100:+.0f}% vs latest)" if pct is not None else ""
    return (
        f"{what} is forecast to reach {_num(s['forecast_end_value'], unit)} by {s['forecast_end_period']}"
        f"{pct_text}, give or take {_num(s['interval_half_width'], unit)}."
    )


class ForecastAnalysisTool(BaseTool):
    """Backtested short-horizon forecast of a measure over time."""

    name = "forecast_analysis"
    description = (
        "Forecast where a measure is heading: aggregates it to a regular series (day/week/month/quarter), "
        "backtests naive, seasonal-naive, linear-trend and exponential-smoothing models on the latest "
        "stretch, keeps the most accurate, and projects it forward with a 95% interval. Says plainly when "
        "the forecast is no better than repeating the last value, when history is short or there are fewer "
        "than two seasons, and when the range is wide. Optionally forecasts up to 3 groups plus the total."
    )
    requires_ml = False

    def applies_to(self, profile: DatasetProfile | None, metadata: DatasetMetadata | None) -> float:
        if profile is None or profile.row_count < _SHORT_HISTORY or not profile.datetime_cols:
            return 0.0
        date_col = profile.datetime_cols[0]
        if not _measure_candidates(profile, {date_col}):
            return 0.0
        return 0.5 if _datetime_span_periods(profile, date_col) >= _SHORT_HISTORY else 0.0

    def default_params(
        self, profile: DatasetProfile | None, metadata: DatasetMetadata | None
    ) -> dict[str, Any]:
        if profile is None or not profile.datetime_cols:
            return {}
        date_col = profile.datetime_cols[0]
        measures = _measure_candidates(profile, {date_col})
        if not measures:
            return {}
        target = getattr(metadata, "target_column", None)
        value = next((c.name for c in measures if c.name == target), measures[0].name)
        return {"date_column": date_col, "value_column": value}

    def execute(  # type: ignore[override]
        self,
        file_path: str,
        date_column: str | None = None,
        value_column: str | None = None,
        horizon: int | str | None = None,
        group_column: str | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        df = _read_df(file_path)
        if df.empty:
            raise ToolExecutionError("Dataset has no rows.")
        for given in (date_column, value_column, group_column):
            if given and given not in df.columns:
                raise ToolExecutionError(f"Column '{given}' not found in dataset.")
        profile = profile_dataframe(df)
        if date_column is None:
            date_column = (profile.datetime_cols[0] if profile.datetime_cols else None) or _autodetect_datetime_column(df)
            if date_column is None:
                raise ToolExecutionError("No date column found. Pass date_column.")
        if value_column is None:
            candidates = _measure_candidates(profile, {date_column, *([group_column] if group_column else [])})
            if not candidates:
                raise ToolExecutionError("No numeric measure found. Pass value_column.")
            value_column = candidates[0].name
        if date_column == value_column:
            raise ToolExecutionError("date_column and value_column must be different columns.")
        cp = next((c for c in profile.columns if c.name == value_column), None)
        agg = measure_aggregation(cp)
        unit = cp.unit_hint if cp else None

        dates = pd.to_datetime(df[date_column], errors="coerce", format="mixed")
        if getattr(dates.dt, "tz", None) is not None:
            dates = dates.dt.tz_localize(None)
        frame = pd.DataFrame({"d": dates, "v": pd.to_numeric(df[value_column], errors="coerce")})
        if group_column:
            frame["g"] = df[group_column].astype(str).where(df[group_column].notna(), "(missing)")
        frame = frame.dropna(subset=["d", "v"])
        if len(frame) < _MIN_PERIODS:
            raise ToolExecutionError(f"Only {len(frame)} usable (date, value) rows; need at least {_MIN_PERIODS}.")

        span_days = max((frame["d"].max() - frame["d"].min()).total_seconds() / 86_400, 1e-9)
        alias, coarsened = _coarsen(_infer_alias(frame["d"]), span_days)
        step_days, m, adjective, noun = _GRAINS[alias]
        notes: list[str] = []
        if coarsened:
            notes.append(f"The series had more than {_MAX_PERIODS:,} periods, so it was aggregated to {noun}s.")

        built = _build_series(frame, alias, agg)
        total = built
        if total is None or len(total[0]) < _MIN_PERIODS:
            usable = 0 if total is None else len(total[0])
            raise ToolExecutionError(
                f"Only {usable} usable {noun} periods after aggregating; need at least {_MIN_PERIODS} "
                f"(with gaps of no more than {_MAX_GAP_FILL} periods)."
            )
        total_series, total_notes, _n_filled, dropped_partial = total
        notes.extend(total_notes)
        if dropped_partial:
            notes.append(f"The latest {noun} was incomplete and was left out of the fit.")

        if horizon is None or (isinstance(horizon, str) and not horizon.strip()):
            h = m if m > 0 else _DEFAULT_HORIZON
        else:
            try:
                h = int(float(horizon))
            except (TypeError, ValueError) as exc:
                raise ToolExecutionError("horizon must be a whole number of periods.") from exc
        h = max(1, min(h, _MAX_HORIZON))

        results = [_series_result("Total", total_series.index, total_series.to_numpy(dtype=float), alias, m, h)]
        skipped: list[str] = []
        if group_column:
            top = frame["g"].value_counts().index[:_MAX_GROUPS]
            for level in top:
                sub = _build_series(frame[frame["g"] == level], alias, agg)
                if sub is None or len(sub[0]) < _MIN_PERIODS:
                    skipped.append(str(level))
                    continue
                try:
                    results.append(_series_result(str(level), sub[0].index, sub[0].to_numpy(dtype=float), alias, m, h))
                except ToolExecutionError:
                    skipped.append(str(level))
            if skipped:
                notes.append(f"Too little history to forecast: {', '.join(skipped)}.")
            n_levels = int(frame["g"].nunique())
            if n_levels > _MAX_GROUPS:
                notes.append(f"Only the {_MAX_GROUPS} largest of {n_levels} {group_column} groups were forecast.")

        head = results[0]
        n = head["n_periods"]
        general: list[str] = []
        if m >= 2 and n < 2 * m:
            general.append(
                f"Fewer than two full seasons of history ({n} {noun}s vs a season of {m}), so no seasonal "
                "pattern was fitted."
            )
        if n < _SHORT_HISTORY:
            general.append(f"Short history ({n} {noun}s); the range is a rough guide.")
        general.append("It assumes recent patterns continue and cannot anticipate events outside the history.")
        general.extend(notes)
        caveats = [*_series_caveats(head, unit), *general]

        summary = _headline(head, adjective, value_column, agg, unit, group_column)
        summary += (
            f" Best model: {head['model'].replace('_', ' ')} (MASE {head['mase']} over the last "
            f"{head['backtest_periods']} {noun}s of history)."
        )
        chart = _chart(results, alias, value_column, date_column, adjective)
        series_out = [{k: v for k, v in s.items() if not k.startswith("_")} for s in results]
        output: dict[str, Any] = {
            "summary": summary,
            "date_column": date_column,
            "value_column": value_column,
            "group_column": group_column,
            "aggregation": agg,
            "unit_hint": unit,
            "period_grain": noun,
            "period_adjective": adjective,
            "step_days": _r(step_days, 3),
            "seasonal_period": m,
            "horizon": h,
            "n_periods": n,
            "model": head["model"],
            "mase": head["mase"],
            "smape": head["smape"],
            "series": series_out,
            "headlines": [_headline(s, adjective, value_column, agg, unit, group_column) for s in results],
            "caveats": caveats,
            "general_caveats": general,
        }
        if chart is not None:
            output["chart"] = chart
        return output

    def findings(
        self,
        output: dict[str, Any],
        profile: DatasetProfile | None,
        metadata: DatasetMetadata | None,
    ) -> list[Finding]:
        measure = str(output.get("value_column"))
        group = output.get("group_column")
        grain = str(output.get("period_adjective") or "Period")
        agg = str(output.get("aggregation") or "sum")
        unit = output.get("unit_hint")
        general = list(output.get("general_caveats") or [])
        chart = output.get("chart")
        results: list[Finding] = []
        for s in output.get("series", []):
            pct = s.get("pct_vs_latest")
            confidence = 0.75 if s["mase"] < 0.7 else 0.55 if s["mase"] < 1.0 else 0.25
            confidence *= min(1.0, s["n_periods"] / 48.0)
            if s.get("wide_interval"):
                confidence *= 0.7
            is_total = s["series"] == "Total"
            level = None if is_total else s["series"]
            results.append(Finding(
                finding_id=f"forecast_{measure}_{s['series']}".replace(" ", "_"),
                kind="forecast",
                headline=_headline(s, grain, measure, agg, unit, str(group) if group else None),
                detail=(
                    f"Model: {str(s['model']).replace('_', ' ')}; backtest over the last {s['backtest_periods']} "
                    f"periods MASE {s['mase']}, sMAPE {s['smape']}%. Interval is 95%."
                ),
                evidence={
                    "last_value": s["last_value"], "last_period": s["last_period"],
                    "forecast_end_value": s["forecast_end_value"], "forecast_end_period": s["forecast_end_period"],
                    "forecast_end_lower": s["forecast_end_lower"], "forecast_end_upper": s["forecast_end_upper"],
                    "interval_half_width": s["interval_half_width"], "pct_vs_latest": pct,
                    "mase": s["mase"], "smape": s["smape"], "n_obs": s["n_periods"],
                    "model": s["model"],
                },
                source_tool=self.name, measure=measure,
                dimension=None if is_total else (str(group) if group else None),
                level=level,
                effect=pct, effect_kind="pct", confidence=min(1.0, confidence),
                surprise=min(1.0, abs(pct or 0.0)),
                caveats=[*_series_caveats(s, unit), *general],
                chart_hint=chart if is_total else None,
            ))
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
                "description": "Datetime column. Auto-detected when omitted.",
                "required": False,
            },
            "value_column": {
                "type": "string",
                "description": "Numeric measure to forecast. Auto-selected (currency measures first) when omitted.",
                "required": False,
            },
            "horizon": {
                "type": "integer",
                "description": "Periods to forecast ahead (max 52). Default: one season, or 12 periods.",
                "required": False,
            },
            "group_column": {
                "type": "string",
                "description": "Optional categorical column: the 3 largest groups are forecast in addition to the total.",
                "required": False,
            },
        }
