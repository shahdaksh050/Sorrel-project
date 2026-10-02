"""
Anomaly Analysis Tool — Execution Layer.

Three kinds of "that looks wrong" over a measure, each reported with its date
or segment and its size in the measure's own units:

  - spikes/dips: the measure is aggregated per period, a baseline is removed
    (the STL trend+seasonal fit when the series is clearly seasonal, else a
    centred rolling median) and residuals more than 3.5 robust SDs (median /
    MAD, Iglewicz-Hoaglin) from typical are flagged;
  - level shifts: recursive binary segmentation on the de-seasonalised,
    spike-cleaned series (a CUSUM-style mean-difference statistic scaled by
    the noise in first differences). A candidate must also be a genuine step
    (the jump at the boundary carries at least half the level change) so a
    smooth trend or random walk is not reported as a shift;
  - unusual segments: the level means of a categorical dimension are compared
    with their peers by a robust z-score, gated by each level's own sampling
    error. This is a group-level check; row-level outliers belong to
    detect_outliers.

Everything is deterministic and fitted per call; nothing is trained or stored.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from src.core.findings import Finding
from src.core.profiler import pick_measures, profile_dataframe
from src.core.run_context import current_objective
from src.core.stats_utils import is_partial_final_period, measure_aggregation
from src.tools.base import BaseTool, ToolExecutionError
from src.tools.data_processing import _read_df
from src.tools.time_series import _autodetect_datetime_column

if TYPE_CHECKING:
    from src.core.memory import DatasetMetadata
    from src.core.profiler import ColumnProfile, DatasetProfile

_MIN_PERIODS = 12
_MAX_PERIODS = 1500
_SPIKE_Z = 3.5
_SHIFT_Z = 6.0
_SHIFT_MIN_SEPARATION = 2.0
_SEASONAL_STRENGTH = 0.3
_MAX_SHIFTS_FOUND = 3
_MAX_ITEMS = 5
_MIN_GROUP_LEVELS = 5
_MAX_GROUP_LEVELS = 50
_MIN_GROUP_N = 5
_MAX_GROUPS_SCANNED = 3
#: A finding needs a practically visible deviation as well as a large z.
_MIN_PCT_FOR_FINDING = 0.10
_MIN_Z_FOR_FINDING = 4.0

#: alias, period length in days, seasonal period in samples (0 = none), noun
_GRAINS: tuple[tuple[str, float, int, str], ...] = (
    ("h", 1 / 24, 24, "hour"),
    ("D", 1.0, 7, "day"),
    ("W", 7.0, 52, "week"),
    ("MS", 30.44, 12, "month"),
    ("QS", 91.31, 4, "quarter"),
    ("YS", 365.25, 0, "year"),
)


def _choose_grain(span_days: float, n_rows: int, hint: str | None) -> tuple[str, int, str]:
    hinted = {"hourly": "h", "daily": "D", "weekly": "W", "monthly": "MS", "quarterly": "QS", "yearly": "YS"}.get(hint or "")
    fits = [g for g in _GRAINS if span_days / g[1] <= _MAX_PERIODS and n_rows >= span_days / g[1]]
    chosen = next((g for g in fits if g[0] == hinted), fits[0] if fits else _GRAINS[-1])
    return chosen[0], chosen[2], chosen[3]


def _label(ts: pd.Timestamp, alias: str) -> str:
    fmt = {"h": "%Y-%m-%d %H:00", "MS": "%Y-%m", "YS": "%Y"}.get(alias, "%Y-%m-%d")
    return str(ts.strftime(fmt))


def _robust_scale(x: np.ndarray) -> float:
    mad = 1.4826 * float(np.median(np.abs(x - np.median(x))))
    if mad > 0:
        return mad
    q75, q25 = np.percentile(x, [75, 25])
    iqr = float(q75 - q25) / 1.349
    return iqr if iqr > 0 else float(np.std(x))


def _num(value: float, unit_hint: str | None = None) -> str:
    text = f"{abs(value):,.0f}" if abs(value) >= 100 else f"{abs(value):,.2f}"
    text = f"${text}" if unit_hint == "currency" else text
    return f"-{text}" if value < 0 else text


def _pct(fraction: float | None) -> str:
    return "" if fraction is None else f" ({fraction * 100:+.0f}%)"


def _baseline(values: np.ndarray, seasonal_period: int) -> tuple[np.ndarray, np.ndarray, str]:
    """(baseline, seasonal component, method). STL only when the series has
    two full cycles and the seasonal part carries a real share of the variance."""
    n = len(values)
    if seasonal_period >= 2 and n >= 2 * seasonal_period:
        from statsmodels.tsa.seasonal import STL

        fit = STL(pd.Series(values), period=seasonal_period, robust=True).fit()
        resid, seasonal = np.asarray(fit.resid), np.asarray(fit.seasonal)
        denominator = float(np.var(resid + seasonal))
        if denominator > 0 and 1 - float(np.var(resid)) / denominator >= _SEASONAL_STRENGTH:
            return np.asarray(fit.trend) + seasonal, seasonal, "stl"
    window = max(5, min(n // 8, 31))
    window += 1 - window % 2
    rolled = pd.Series(values).rolling(window, center=True, min_periods=window // 2 + 1).median()
    return rolled.to_numpy(dtype=float), np.zeros(n), "rolling_median"


def _level_shifts(y: np.ndarray) -> list[dict[str, Any]]:
    n = len(y)
    min_seg = max(6, n // 10)
    diffs = np.diff(y)
    sigma = _robust_scale(diffs) / math.sqrt(2)
    if n < 2 * min_seg or sigma <= 0:
        return []
    found: list[dict[str, Any]] = []

    def split(lo: int, hi: int, depth: int) -> None:
        if depth >= _MAX_SHIFTS_FOUND or hi - lo < 2 * min_seg:
            return
        seg = y[lo:hi]
        m = len(seg)
        csum = np.cumsum(seg)
        ks = np.arange(min_seg, m - min_seg + 1)
        left = csum[ks - 1] / ks
        right = (csum[-1] - csum[ks - 1]) / (m - ks)
        stat = np.abs(left - right) / (sigma * np.sqrt(1 / ks + 1 / (m - ks)))
        best = int(np.argmax(stat))
        k = int(ks[best])
        delta = float(right[best] - left[best])
        if float(stat[best]) < _SHIFT_Z:
            return
        within = math.sqrt((float(np.var(seg[:k])) * k + float(np.var(seg[k:])) * (m - k)) / m)
        w = min(5, min_seg)
        local = float(np.median(seg[k:k + w]) - np.median(seg[k - w:k]))
        if abs(delta) < _SHIFT_MIN_SEPARATION * within or abs(local) < 0.5 * abs(delta):
            return
        found.append({
            "index": lo + k, "level_before": float(left[best]), "level_after": float(right[best]),
            "delta": delta, "z": float(stat[best]),
        })
        split(lo, lo + k, depth + 1)
        split(lo + k, hi, depth + 1)

    split(0, n, 0)
    return sorted(found, key=lambda s: s["z"], reverse=True)[:_MAX_SHIFTS_FOUND]


def _series_anomalies(
    df: pd.DataFrame, date_column: str, value_column: str, agg: str, time_hint: str | None
) -> dict[str, Any]:
    dates = pd.to_datetime(df[date_column], errors="coerce", format="mixed")
    if getattr(dates.dt, "tz", None) is not None:
        dates = dates.dt.tz_localize(None)
    frame = pd.DataFrame({"d": dates, "v": pd.to_numeric(df[value_column], errors="coerce")}).dropna()
    if len(frame) < _MIN_PERIODS:
        raise ToolExecutionError(f"Only {len(frame)} usable (date, value) rows; need at least {_MIN_PERIODS}.")
    span_days = max((frame["d"].max() - frame["d"].min()).total_seconds() / 86400.0, 1e-9)
    alias, seasonal_period, noun = _choose_grain(span_days, len(frame), time_hint)

    resampler = frame.set_index("d")["v"].resample(alias, label="left", closed="left")
    per_period = resampler.sum(min_count=1) if agg == "sum" else resampler.mean()
    counts = resampler.size()
    if is_partial_final_period(counts, frame["d"].max(), alias):
        per_period, counts = per_period.iloc[:-1], counts.iloc[:-1]
    full = per_period
    filled = full.isna().to_numpy()
    if len(full) < _MIN_PERIODS or filled.mean() > 0.3:
        raise ToolExecutionError(
            f"Only {int((~filled).sum())} populated {noun} periods; need at least {_MIN_PERIODS} with under 30% gaps."
        )
    values = full.interpolate(limit_direction="both").to_numpy(dtype=float)

    baseline, seasonal, method = _baseline(values, seasonal_period)
    baseline = np.where(np.isnan(baseline), np.nanmedian(baseline), baseline)
    resid = values - baseline
    scale = _robust_scale(resid)
    z = (resid - float(np.median(resid))) / scale if scale > 0 else np.zeros_like(resid)
    is_spike = (np.abs(z) >= _SPIKE_Z) & ~filled

    spikes = [
        {
            "period": _label(full.index[i], alias), "value": float(values[i]), "expected": float(baseline[i]),
            "deviation": float(resid[i]),
            "deviation_pct": float(resid[i] / abs(baseline[i])) if baseline[i] != 0 else None,
            "z": float(z[i]), "direction": "spike" if resid[i] > 0 else "dip",
        }
        for i in sorted(np.flatnonzero(is_spike), key=lambda i: -abs(z[i]))[:_MAX_ITEMS]
    ]
    cleaned = np.where(is_spike, baseline, values) - seasonal
    shifts = [
        {
            "period": _label(full.index[s["index"]], alias), "level_before": s["level_before"],
            "level_after": s["level_after"], "delta": s["delta"], "z": s["z"],
            "delta_pct": s["delta"] / abs(s["level_before"]) if s["level_before"] != 0 else None,
        }
        for s in _level_shifts(cleaned)
    ]
    return {
        "period_grain": noun, "n_periods": len(full), "n_filled_periods": int(filled.sum()),
        "seasonal_period": seasonal_period if method == "stl" else 0, "method": method,
        "spikes": spikes, "n_spikes": int(is_spike.sum()), "level_shifts": shifts,
    }


def _unusual_segments(df: pd.DataFrame, value_column: str, dimension: str) -> dict[str, Any] | None:
    work = pd.DataFrame({"g": df[dimension].astype(str), "v": pd.to_numeric(df[value_column], errors="coerce")})
    work = work[df[dimension].notna()].dropna()
    grouped = work.groupby("g")["v"].agg(["mean", "std", "size"])
    grouped = grouped[grouped["size"] >= _MIN_GROUP_N]
    if len(grouped) < _MIN_GROUP_LEVELS:
        return None
    means = grouped["mean"].to_numpy(dtype=float)
    peer_median = float(np.median(means))
    scale = _robust_scale(means)
    if scale <= 0:
        return None
    rows: list[dict[str, Any]] = []
    for level, r in grouped.iterrows():
        z = (float(r["mean"]) - peer_median) / scale
        dev = float(r["mean"]) - peer_median
        sem = float(r["std"]) / math.sqrt(float(r["size"])) if not math.isnan(float(r["std"])) else 0.0
        if abs(z) >= _SPIKE_Z and abs(dev) >= 2 * sem:
            rows.append({
                "dimension": dimension, "level": str(level), "mean": float(r["mean"]),
                "peer_median": peer_median, "deviation": dev,
                "deviation_pct": dev / abs(peer_median) if peer_median != 0 else None,
                "z": z, "n": int(r["size"]),
            })
    rows.sort(key=lambda s: -abs(s["z"]))
    return {"dimension": dimension, "levels": len(grouped), "unusual": rows[:_MAX_ITEMS]}


def _measure_candidates(profile: DatasetProfile, exclude: set[str]) -> list[ColumnProfile]:
    measures = [c for c in profile.measures() if c.name not in exclude]
    return sorted(measures, key=lambda c: (c.unit_hint != "currency", c.name))


def _dimension_candidates(profile: DatasetProfile, exclude: set[str]) -> list[str]:
    dims = [
        c for c in profile.columns_of_role("dimension")
        if _MIN_GROUP_LEVELS <= c.nunique <= _MAX_GROUP_LEVELS and c.name not in exclude
    ]
    return [c.name for c in sorted(dims, key=lambda c: c.nunique)]


def _round_dict(row: dict[str, Any]) -> dict[str, Any]:
    return {k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()}


class AnomalyAnalysisTool(BaseTool):
    """Spikes, level shifts and unusual segments in a measure."""

    name = "anomaly_analysis"
    description = (
        "Find what looks wrong in a measure: spikes and dips over time "
        "(robust z-score against a rolling or seasonal baseline), sudden "
        "level shifts / change points, and segments (groups of a "
        "categorical column) whose average is unusual relative to their "
        "peers. Reports the top few with dates, values and size in the "
        "measure's own units. Auto-selects the date, measure and dimensions "
        "when omitted."
    )

    def applies_to(self, profile: DatasetProfile | None, metadata: DatasetMetadata | None) -> float:
        if profile is None or not profile.measures():
            return 0.0
        if profile.is_time_series:
            return 0.85
        return 0.5 if _dimension_candidates(profile, set()) else 0.0

    def default_params(
        self, profile: DatasetProfile | None, metadata: DatasetMetadata | None
    ) -> dict[str, Any]:
        if profile is None:
            return {}
        date_col = profile.datetime_cols[0] if profile.datetime_cols else None
        objective = current_objective()
        measures = [c for c in pick_measures(profile, objective) if c.name != date_col]
        if not measures:
            return {}
        params: dict[str, Any] = {"value_column": measures[0].name}
        if date_col:
            params["date_column"] = date_col
        return params

    def execute(  # type: ignore[override]
        self,
        file_path: str,
        date_column: str | None = None,
        value_column: str | None = None,
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

        if date_column is None and group_column is None:
            date_column = _autodetect_datetime_column(df)
        exclude: set[str] = {date_column} if date_column else set()
        if value_column is None:
            candidates = _measure_candidates(profile, exclude)
            if not candidates:
                raise ToolExecutionError("No numeric measure found. Pass value_column.")
            value_column = candidates[0].name
        if pd.to_numeric(df[value_column], errors="coerce").notna().sum() == 0:
            raise ToolExecutionError(f"'{value_column}' is not numeric.")
        cp = next((c for c in profile.columns if c.name == value_column), None)
        agg = measure_aggregation(cp)
        unit_hint = cp.unit_hint if cp else None

        result: dict[str, Any] = {
            "date_column": date_column, "value_column": value_column, "aggregation": agg,
            "unit_hint": unit_hint, "spikes": [], "level_shifts": [], "segments": [], "groups_scanned": [],
        }
        notes: list[str] = []
        if date_column:
            try:
                series = _series_anomalies(
                    df, date_column, value_column, agg,
                    profile.time_frequency if date_column in profile.datetime_cols[:1] else None,
                )
                result.update(series)
            except ToolExecutionError as exc:
                if group_column is None and not _dimension_candidates(profile, {value_column}):
                    raise
                notes.append(str(exc))
        dims = [group_column] if group_column else _dimension_candidates(
            profile, {value_column, *exclude}
        )[:_MAX_GROUPS_SCANNED]
        for dim in dims:
            scan = _unusual_segments(df, value_column, dim)
            if scan is None:
                notes.append(f"'{dim}' has fewer than {_MIN_GROUP_LEVELS} levels with {_MIN_GROUP_N}+ rows; skipped.")
                continue
            result["groups_scanned"].append({"dimension": dim, "levels": scan["levels"]})
            result["segments"].extend(scan["unusual"])
        result["segments"].sort(key=lambda s: -abs(s["z"]))
        result["segments"] = result["segments"][:_MAX_ITEMS]
        if not date_column and not result["groups_scanned"]:
            raise ToolExecutionError(
                "Nothing to scan: no datetime column and no categorical column with "
                f"{_MIN_GROUP_LEVELS}+ levels. Pass date_column or group_column."
            )

        for key in ("spikes", "level_shifts", "segments"):
            result[key] = [_round_dict(r) for r in result[key]]
        result["notes"] = notes
        result["summary"] = self._summary(result)
        return result

    @staticmethod
    def _summary(r: dict[str, Any]) -> str:
        parts = [f"Scanned {r['value_column']}"]
        found: list[str] = []
        if r["spikes"]:
            s = r["spikes"][0]
            found.append(
                f"{r.get('n_spikes', len(r['spikes']))} spike/dip period(s), largest {s['period']} "
                f"at {_num(s['value'], r['unit_hint'])} vs about {_num(s['expected'], r['unit_hint'])} expected"
            )
        if r["level_shifts"]:
            s = r["level_shifts"][0]
            found.append(
                f"a level shift at {s['period']} from {_num(s['level_before'], r['unit_hint'])} "
                f"to {_num(s['level_after'], r['unit_hint'])}"
            )
        if r["segments"]:
            s = r["segments"][0]
            found.append(f"unusual {s['dimension']}={s['level']} ({_num(s['mean'], r['unit_hint'])} vs peer median {_num(s['peer_median'], r['unit_hint'])})")
        if not found:
            return f"{parts[0]}: no spikes, level shifts or unusual segments beyond normal variation."
        return f"{parts[0]}: found " + "; ".join(found) + "."

    def findings(
        self,
        output: dict[str, Any],
        profile: DatasetProfile | None,
        metadata: DatasetMetadata | None,
    ) -> list[Finding]:
        measure = str(output.get("value_column"))
        unit = output.get("unit_hint")
        noun = output.get("period_grain") or "period"
        per = "average" if output.get("aggregation") == "mean" else "total"
        results: list[Finding] = []
        for s in output.get("spikes", []):
            pct = s.get("deviation_pct")
            if abs(s["z"]) < _MIN_Z_FOR_FINDING or (pct is not None and abs(pct) < _MIN_PCT_FOR_FINDING):
                continue
            word = "spiked" if s["direction"] == "spike" else "dropped"
            results.append(Finding(
                finding_id=f"anomaly_spike_{measure}_{s['period']}".replace(" ", "_"),
                kind="outlier_scan",
                headline=(
                    f"{measure} {word} in {s['period']}: {_num(s['value'], unit)} vs about "
                    f"{_num(s['expected'], unit)} expected{_pct(pct)}."
                ),
                detail=f"{abs(s['z']):.1f} robust SDs from the {output.get('method', 'rolling')} baseline ({per} per {noun}).",
                evidence=s, source_tool=self.name, measure=measure, level=s["period"],
                effect=pct, effect_kind="pct", confidence=0.7, surprise=min(1.0, abs(s["z"]) / 10.0),
            ))
        for s in output.get("level_shifts", []):
            pct = s.get("delta_pct")
            if pct is not None and abs(pct) < _MIN_PCT_FOR_FINDING:
                continue
            word = "stepped up" if s["delta"] > 0 else "stepped down"
            results.append(Finding(
                finding_id=f"anomaly_shift_{measure}_{s['period']}".replace(" ", "_"),
                kind="change",
                headline=(
                    f"{measure} {word} from about {_num(s['level_before'], unit)} to "
                    f"{_num(s['level_after'], unit)} ({per} per {noun}) starting {s['period']}{_pct(pct)}."
                ),
                detail="A sustained change in level, not a one-period blip.",
                evidence=s, source_tool=self.name, measure=measure, level=s["period"],
                effect=pct, effect_kind="pct", confidence=0.75, surprise=min(1.0, abs(pct or 0.3)),
            ))
        for s in output.get("segments", []):
            pct = s.get("deviation_pct")
            if abs(s["z"]) < _MIN_Z_FOR_FINDING or (pct is not None and abs(pct) < _MIN_PCT_FOR_FINDING):
                continue
            direction = "above" if s["deviation"] > 0 else "below"
            results.append(Finding(
                finding_id=f"anomaly_segment_{measure}_{s['dimension']}_{s['level']}".replace(" ", "_"),
                kind="outlier_scan",
                headline=(
                    f"{s['dimension']} {s['level']} averages {_num(s['mean'], unit)} {measure}, "
                    f"far {direction} its peers' typical {_num(s['peer_median'], unit)}{_pct(pct)}."
                ),
                detail=f"{abs(s['z']):.1f} robust SDs from the median of {s['dimension']} levels; n={s['n']}.",
                evidence=s, source_tool=self.name, measure=measure, dimension=s["dimension"], level=s["level"],
                effect=pct, effect_kind="pct", confidence=min(1.0, s["n"] / 100.0),
                surprise=min(1.0, abs(s["z"]) / 10.0),
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
                "description": "Datetime column for the spike / level-shift scan. Auto-detected when omitted.",
                "required": False,
            },
            "value_column": {
                "type": "string",
                "description": "Numeric measure to scan. Auto-selected (currency measures first) when omitted.",
                "required": False,
            },
            "group_column": {
                "type": "string",
                "description": (
                    "Categorical column (5-50 levels) whose segments are compared "
                    "with each other for unusual levels. Auto-selected when omitted."
                ),
                "required": False,
            },
        }
