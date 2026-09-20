"""
Curve Fit Analysis Tool — Execution Layer.

"What shape does y follow as x changes, and what is the half-life / EC50 /
plateau?" for scientific-style relationships (dose-response, decay, growth,
saturation). Candidate curves are fitted by non-linear least squares
(scipy.optimize.curve_fit, bounded, data-driven starting values):

  linear a+bx | exponential a*exp(bx) | decay a*exp(-kx)+c | logistic (4
  parameter sigmoid) | hill (4-parameter dose-response, x >= 0) | power
  a*x^b | michaelis_menten Vmax*x/(Km+x) | log a+b*ln(x)

`model="auto"` fits every applicable curve, keeps the ones that converged and
picks the lowest AICc (a small-sample corrected information criterion that
penalises extra parameters; a simpler curve wins when it is within 2 AICc
points). Parameter 95% CIs come from the fit covariance (skipped when it is
singular) and are translated into plain quantities: half-life, doubling time,
EC50/IC50, plateau, Km. With a group column the best curve is fitted per
group and the key parameter is compared by confidence-interval overlap.

Everything is deterministic and fitted per call; nothing is trained or stored.
"""
from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from src.core.findings import Finding
from src.core.profiler import (
    ARCHETYPE_EXPERIMENT,
    ARCHETYPE_SENSOR,
    profile_dataframe,
)
from src.core.vocab import ROLE_TOKENS, column_role, name_tokens
from src.tools.base import BaseTool, ToolExecutionError
from src.tools.data_processing import _read_df

if TYPE_CHECKING:
    from src.core.memory import DatasetMetadata
    from src.core.profiler import ColumnProfile, DatasetProfile

_MAX_ROWS = 50_000
#: Non-linear fits run on at most this many rows (random_state=0) for speed.
_FIT_MAX_ROWS = 20_000
_MAX_GROUPS = 6
_MIN_N_AUTO = 8
_MIN_N_MODEL = 5
_MIN_APPLIES_ROWS = 10
_MIN_X_DISTINCT = 8
_MAXFEV = 2000
_AICC_TIE = 2.0
_LOW_R2 = 0.5
_SMALL_N = 20
#: Chart rows are capped at 500 by the chart contract.
_POINT_BUDGET = 240
_CURVE_BUDGET = 120

#: Axis names common in ordinary business data (weak) never suffice alone for applies_to.
_WEAK_X_TOKENS = ROLE_TOKENS["dose_x_weak"]
_X_TOKENS = ROLE_TOKENS["dose_x_strong"] | _WEAK_X_TOKENS
_NARROW_TABLE_COLS = 6
_X_UNITS = {
    "day": "days", "hour": "hours", "minute": "minutes", "week": "weeks", "month": "months",
    "year": "years", "generation": "generations", "cycle": "cycles",
}
_ALIASES = {
    "mm": "michaelis_menten", "michaelis-menten": "michaelis_menten", "michaelis menten": "michaelis_menten",
    "4pl": "hill", "dose_response": "hill", "sigmoid": "logistic", "exp": "exponential",
    "logarithmic": "log", "exponential_decay": "decay",
}


# ---------------------------------------------------------------------------
# Column picking
# ---------------------------------------------------------------------------

def _tokens(name: str) -> set[str]:
    out: set[str] = set()
    for tok in name_tokens(name):
        out.add(tok)
        if len(tok) > 3 and tok.endswith("s"):
            out.add(tok[:-1])
    return out


def _is_x_override(profile: DatasetProfile, name: str) -> bool:
    return column_role(profile, name, "dose_x_strong") or column_role(profile, name, "dose_x_weak")


def _x_candidates(profile: DatasetProfile) -> list[ColumnProfile]:
    return [
        c for c in profile.columns
        if c.kind == "numeric" and c.nunique >= _MIN_X_DISTINCT
        and (_tokens(c.name) & _X_TOKENS or _is_x_override(profile, c.name))
    ]


def _y_candidates(profile: DatasetProfile, x_name: str) -> list[ColumnProfile]:
    numeric = [c for c in profile.columns if c.kind == "numeric" and c.name != x_name and c.nunique >= 3]
    return sorted(
        numeric,
        key=lambda c: (not c.is_measure(), bool(_tokens(c.name) & _X_TOKENS or _is_x_override(profile, c.name))),
    )


def _x_unit(name: str) -> str:
    toks = _tokens(name)
    for singular, plural in _X_UNITS.items():
        if singular in toks:
            return plural
    return f"{name} units"


# ---------------------------------------------------------------------------
# Model library
# ---------------------------------------------------------------------------

def _f_linear(x: np.ndarray, a: float, b: float) -> Any:
    return a + b * x


def _f_exponential(x: np.ndarray, a: float, b: float) -> Any:
    return a * np.exp(b * x)


def _f_decay(x: np.ndarray, a: float, k: float, c: float) -> Any:
    return a * np.exp(-k * x) + c


def _f_logistic(x: np.ndarray, bottom: float, top: float, x0: float, k: float) -> Any:
    return bottom + (top - bottom) / (1.0 + np.exp(-k * (x - x0)))


def _f_hill(x: np.ndarray, bottom: float, top: float, ec50: float, hill: float) -> Any:
    u = (np.maximum(x, 0.0) / ec50) ** hill
    return bottom + (top - bottom) * u / (1.0 + u)


def _f_power(x: np.ndarray, a: float, b: float) -> Any:
    return a * np.power(x, b)


def _f_mm(x: np.ndarray, vmax: float, km: float) -> Any:
    return vmax * x / (km + x)


def _f_log(x: np.ndarray, a: float, b: float) -> Any:
    return a + b * np.log(x)


def _ends(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    order = np.argsort(x, kind="stable")
    m = max(1, int(0.2 * len(x)))
    return float(np.mean(y[order][:m])), float(np.mean(y[order][-m:]))


def _cross_x(x: np.ndarray, y: np.ndarray, level: float) -> float | None:
    """x where a moving average of y first crosses `level` (linear interpolation)."""
    order = np.argsort(x, kind="stable")
    xs, ys = x[order], y[order]
    w = max(1, len(xs) // 10)
    if w > 1:
        ys = np.convolve(ys, np.ones(w) / w, mode="valid")
        xs = np.convolve(xs, np.ones(w) / w, mode="valid")
    above = ys - level
    flips = np.flatnonzero(np.sign(above[:-1]) * np.sign(above[1:]) < 0)
    if not flips.size:
        return None
    i = int(flips[0])
    span = above[i + 1] - above[i]
    return float(xs[i] + (xs[i + 1] - xs[i]) * (-above[i] / span)) if span != 0 else float(xs[i])


def _pos_min(x: np.ndarray) -> float:
    pos = x[x > 0]
    return float(pos.min()) if pos.size else 1.0


def _ylim(y: np.ndarray) -> tuple[float, float]:
    lo, hi = float(y.min()), float(y.max())
    span = max(hi - lo, 1e-12)
    return lo - 0.5 * span, hi + 0.5 * span


def _init_linear(x: np.ndarray, y: np.ndarray) -> list[float]:
    b, a = np.polyfit(x, y, 1)
    return [float(a), float(b)]


def _bounds_free(x: np.ndarray, y: np.ndarray, k: int) -> tuple[list[float], list[float]]:
    return [-math.inf] * k, [math.inf] * k


def _bounds_linear(x: np.ndarray, y: np.ndarray) -> tuple[list[float], list[float]]:
    return _bounds_free(x, y, 2)


def _init_exponential(x: np.ndarray, y: np.ndarray) -> list[float]:
    xs = x - x.min()
    if np.all(y > 0) or np.all(y < 0):
        b, ic = np.polyfit(xs, np.log(np.abs(y)), 1)
        return [float(np.sign(np.median(y)) * math.exp(max(-50.0, min(50.0, ic)))), float(b)]
    return [float(np.mean(y)), 0.0]


def _bounds_exponential(x: np.ndarray, y: np.ndarray) -> tuple[list[float], list[float]]:
    limit = 60.0 / max(float(np.ptp(x)), 1e-12)
    return [-math.inf, -limit], [math.inf, limit]


def _init_decay(x: np.ndarray, y: np.ndarray) -> list[float]:
    xs = x - x.min()
    xr = max(float(np.ptp(x)), 1e-12)
    first, last = _ends(xs, y)
    a0 = first - last if first != last else max(float(np.ptp(y)), 1e-6)
    half = _cross_x(xs, y, (first + last) / 2.0)
    k0 = math.log(2.0) / max(half if half is not None else xr / 3.0, xr / 100.0)
    return [a0, k0, last]


def _bounds_decay(x: np.ndarray, y: np.ndarray) -> tuple[list[float], list[float]]:
    xr = max(float(np.ptp(x)), 1e-12)
    return [-math.inf, 1e-4 / xr, -math.inf], [math.inf, 500.0 / xr, math.inf]


def _init_logistic(x: np.ndarray, y: np.ndarray) -> list[float]:
    xr = max(float(np.ptp(x)), 1e-12)
    bottom, top = _ends(x, y)
    if bottom == top:
        top = bottom + max(float(np.ptp(y)), 1e-6)
    mid = _cross_x(x, y, (bottom + top) / 2.0)
    return [bottom, top, mid if mid is not None else float(np.median(x)), 4.0 / xr]


def _bounds_logistic(x: np.ndarray, y: np.ndarray) -> tuple[list[float], list[float]]:
    xr = max(float(np.ptp(x)), 1e-12)
    ylo, yhi = _ylim(y)
    return (
        [ylo, ylo, float(x.min()) - 0.5 * xr, 1e-3 / xr],
        [yhi, yhi, float(x.max()) + 0.5 * xr, 200.0 / xr],
    )


def _init_hill(x: np.ndarray, y: np.ndarray) -> list[float]:
    bottom, top = _ends(x, y)
    if bottom == top:
        top = bottom + max(float(np.ptp(y)), 1e-6)
    mid = _cross_x(x, y, (bottom + top) / 2.0)
    pos = x[x > 0]
    fallback = float(np.exp(np.mean(np.log(pos)))) if pos.size else 1.0
    return [bottom, top, mid if mid is not None and mid > 0 else fallback, 1.0]


def _bounds_hill(x: np.ndarray, y: np.ndarray) -> tuple[list[float], list[float]]:
    ylo, yhi = _ylim(y)
    return (
        [ylo, ylo, _pos_min(x) / 100.0, 0.2],
        [yhi, yhi, float(x.max()) * 100.0, 10.0],
    )


def _init_mm(x: np.ndarray, y: np.ndarray) -> list[float]:
    ymax, ymin = float(y.max()), float(y.min())
    vmax = ymax * 1.2 if abs(ymax) >= abs(ymin) else ymin * 1.2
    half = _cross_x(x, y, vmax / 2.0)
    pos = x[x > 0]
    km = half if half is not None and half > 0 else (float(np.median(pos)) if pos.size else 1.0)
    return [vmax, km]


def _bounds_mm(x: np.ndarray, y: np.ndarray) -> tuple[list[float], list[float]]:
    return [-math.inf, _pos_min(x) / 100.0], [math.inf, float(x.max()) * 100.0]


def _init_power(x: np.ndarray, y: np.ndarray) -> list[float]:
    if np.all(y > 0) or np.all(y < 0):
        b, ic = np.polyfit(np.log(x), np.log(np.abs(y)), 1)
        return [float(np.sign(np.median(y)) * math.exp(max(-50.0, min(50.0, ic)))), float(max(-10.0, min(10.0, b)))]
    return [1.0, 1.0]


def _bounds_power(x: np.ndarray, y: np.ndarray) -> tuple[list[float], list[float]]:
    return [-math.inf, -10.0], [math.inf, 10.0]


def _init_log(x: np.ndarray, y: np.ndarray) -> list[float]:
    b, a = np.polyfit(np.log(x), y, 1)
    return [float(a), float(b)]


@dataclass(frozen=True)
class _Spec:
    name: str
    label: str
    names: tuple[str, ...]
    func: Callable[..., Any]
    init: Callable[[np.ndarray, np.ndarray], list[float]]
    bounds: Callable[[np.ndarray, np.ndarray], tuple[list[float], list[float]]]
    domain: str = "any"        # any | nonneg | pos
    shift: bool = False        # fit on x - min(x); the level parameter is then the value at min(x)


_SPECS: dict[str, _Spec] = {s.name: s for s in (
    _Spec("linear", "linear", ("a", "b"), _f_linear, _init_linear, _bounds_linear),
    _Spec("exponential", "exponential growth/decay", ("a", "b"), _f_exponential, _init_exponential,
          _bounds_exponential, shift=True),
    _Spec("decay", "exponential decay to a plateau", ("a", "k", "c"), _f_decay, _init_decay,
          _bounds_decay, shift=True),
    _Spec("logistic", "S-shaped (logistic)", ("bottom", "top", "x0", "k"), _f_logistic, _init_logistic,
          _bounds_logistic),
    _Spec("hill", "dose-response (4-parameter Hill)", ("bottom", "top", "ec50", "hill"), _f_hill,
          _init_hill, _bounds_hill, domain="nonneg"),
    _Spec("power", "power law", ("a", "b"), _f_power, _init_power, _bounds_power, domain="pos"),
    _Spec("michaelis_menten", "Michaelis-Menten saturation", ("vmax", "km"), _f_mm, _init_mm,
          _bounds_mm, domain="nonneg"),
    _Spec("log", "logarithmic", ("a", "b"), _f_log, _init_log, _bounds_linear, domain="pos"),
)}


def _applicable(spec: _Spec, x: np.ndarray) -> bool:
    if spec.domain == "pos":
        return bool(np.all(x > 0))
    if spec.domain == "nonneg":
        return bool(np.all(x >= 0) and x.max() > 0)
    return True


@dataclass
class _Fit:
    spec: _Spec
    params: np.ndarray
    pcov: np.ndarray
    shift: float
    lo: np.ndarray
    hi: np.ndarray
    n: int
    r2: float
    rmse: float
    aicc: float

    def predict(self, x: np.ndarray, params: np.ndarray | None = None) -> np.ndarray:
        p = self.params if params is None else params
        with np.errstate(all="ignore"):
            return np.asarray(self.spec.func(np.asarray(x, dtype=np.float64) - self.shift, *p), dtype=np.float64)

    def intervals(self) -> list[tuple[float, float] | None]:
        from scipy import stats

        diag = np.diag(self.pcov)
        if not (np.all(np.isfinite(self.pcov)) and np.all(diag >= 0)):
            return [None] * len(self.params)
        dof = self.n - len(self.params)
        if dof <= 0:
            return [None] * len(self.params)
        t = float(stats.t.ppf(0.975, dof))
        se = np.sqrt(diag)
        return [(float(p - t * s), float(p + t * s)) for p, s in zip(self.params, se, strict=True)]

    def at_bounds(self) -> list[str]:
        hit: list[str] = []
        for name, p, lo, hi in zip(self.spec.names, self.params, self.lo, self.hi, strict=True):
            near = any(
                math.isfinite(bound) and abs(float(p) - bound) <= 1e-4 * max(abs(bound), abs(float(p)), 1e-12)
                for bound in (float(lo), float(hi))
            )
            if near:
                hit.append(name)
        return hit


def _fit_one(spec: _Spec, x: np.ndarray, y: np.ndarray) -> _Fit | None:
    from scipy.optimize import curve_fit

    n, k = len(x), len(spec.names)
    if n < k + 2:
        return None
    shift = float(x.min()) if spec.shift else 0.0
    xs = x - shift

    def func(xx: np.ndarray, *p: float) -> Any:
        with np.errstate(all="ignore"):
            return spec.func(xx, *p)

    try:
        with np.errstate(all="ignore"):
            p0 = np.asarray(spec.init(xs, y), dtype=np.float64)
        lo_l, hi_l = spec.bounds(xs, y)
        lo, hi = np.asarray(lo_l, dtype=np.float64), np.asarray(hi_l, dtype=np.float64)
        p0 = np.minimum(np.maximum(p0, lo), hi)
        if not np.all(np.isfinite(p0)):
            return None
        popt, pcov = curve_fit(func, xs, y, p0=p0, bounds=(lo, hi), max_nfev=_MAXFEV)
    except (RuntimeError, ValueError, TypeError, np.linalg.LinAlgError, FloatingPointError):
        return None
    popt = np.asarray(popt, dtype=np.float64)
    if not np.all(np.isfinite(popt)):
        return None
    with np.errstate(all="ignore"):
        pred = np.asarray(spec.func(xs, *popt), dtype=np.float64)
    if not np.all(np.isfinite(pred)):
        return None
    rss = float(np.sum((y - pred) ** 2))
    tss = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - rss / tss if tss > 0 else 0.0
    rss_eff = max(rss, 1e-12 * max(tss, 1e-12))
    aicc = (
        n * math.log(rss_eff / n) + 2 * k + 2.0 * k * (k + 1) / (n - k - 1)
        if n - k - 1 > 0 else math.inf
    )
    return _Fit(spec, popt, np.asarray(pcov, dtype=np.float64), shift, lo, hi, n, r2,
                math.sqrt(rss / n), aicc)


def _fit_best(x: np.ndarray, y: np.ndarray, model: str) -> tuple[_Fit, list[dict[str, Any]]]:
    n = len(x)
    auto = model == "auto"
    if auto and n < _MIN_N_AUTO:
        raise ToolExecutionError(
            f"Only {n} usable points; automatic model choice needs at least {_MIN_N_AUTO}. "
            "Name a model explicitly or add data."
        )
    if not auto and n < _MIN_N_MODEL:
        raise ToolExecutionError(f"Only {n} usable points; need at least {_MIN_N_MODEL}.")
    if float(np.ptp(y)) == 0:
        raise ToolExecutionError("y is constant, so there is no curve to fit.")
    fits: list[_Fit] = []
    skipped: list[str] = []
    for name in (list(_SPECS) if auto else [model]):
        spec = _SPECS[name]
        if not _applicable(spec, x):
            skipped.append(f"{name} needs {'positive' if spec.domain == 'pos' else 'non-negative'} x")
            continue
        fit = _fit_one(spec, x, y)
        if fit is None:
            skipped.append(f"{name} did not converge")
            continue
        fits.append(fit)
    if not fits:
        raise ToolExecutionError(
            "No curve could be fitted (" + "; ".join(skipped) + "). Try a different model or check x and y."
        )
    finite = [f for f in fits if math.isfinite(f.aicc)] or fits
    best = min(f.aicc for f in finite)
    contenders = [f for f in finite if f.aicc <= best + _AICC_TIE]
    chosen = min(contenders, key=lambda f: (len(f.params), f.aicc)) if auto else fits[0]
    candidates = [
        {
            "model": f.spec.name, "r2": _sig(f.r2), "rmse": _sig(f.rmse),
            "aicc": _sig(f.aicc) if math.isfinite(f.aicc) else None, "n_params": len(f.params),
            "chosen": f is chosen,
        }
        for f in sorted(fits, key=lambda f: f.aicc)
    ]
    return chosen, candidates


# ---------------------------------------------------------------------------
# Wording helpers
# ---------------------------------------------------------------------------

def _sig(v: float | None, digits: int = 4) -> float | None:
    if v is None or not math.isfinite(v):
        return None
    return float(f"{v:.{digits}g}")


def _fmt(v: float) -> str:
    a = abs(v)
    if a >= 1000:
        return f"{v:,.0f}"
    if a >= 100:
        return f"{v:.0f}"
    if a >= 10:
        return f"{v:.1f}"
    if a >= 1:
        return f"{v:.2f}"
    return f"{v:.3g}"


def _ci_text(lo: float | None, hi: float | None) -> str:
    return "" if lo is None or hi is None else f" (95% CI {_fmt(lo)} to {_fmt(hi)})"


def _inverse(k: tuple[float, float] | None, scale: float) -> tuple[float | None, float | None]:
    """CI of scale/k for a positive-valued k interval."""
    if k is None or k[0] <= 0:
        return None, None
    return scale / k[1], scale / k[0]


def _derive(fit: _Fit, x_col: str, y_col: str, xmin: float, xmax: float) -> list[dict[str, Any]]:
    names = fit.spec.names
    p = dict(zip(names, (float(v) for v in fit.params), strict=True))
    ci = dict(zip(names, fit.intervals(), strict=True))
    xr = xmax - xmin
    ln2 = math.log(2.0)
    out: list[dict[str, Any]] = []

    def add(name: str, label: str, value: float, interval: tuple[float | None, float | None] | None,
            kind: str) -> None:
        lo, hi = interval if interval is not None else (None, None)
        outside = (kind == "position" and not xmin <= value <= xmax) or (kind == "duration" and value > xr)
        out.append({
            "name": name, "label": label, "value": _sig(value), "ci_lower": _sig(lo), "ci_upper": _sig(hi),
            "kind": kind, "outside_range": bool(outside),
        })

    def raw(name: str) -> tuple[float | None, float | None] | None:
        return ci[name]

    model = fit.spec.name
    if model == "linear":
        add("slope", f"change in {y_col} per 1 unit of {x_col}", p["b"], raw("b"), "rate")
        add("intercept", f"{y_col} at {x_col} = 0", p["a"], raw("a"), "level")
    elif model == "exponential":
        b = p["b"]
        if b > 0:
            add("doubling_time", "doubling time", ln2 / b, _inverse(ci["b"], ln2), "duration")
        elif b < 0:
            neg = None if ci["b"] is None else (-ci["b"][1], -ci["b"][0])
            add("half_life", "half-life", ln2 / -b, _inverse(neg, ln2), "duration")
        add("rate", "exponential rate per unit x", b, raw("b"), "rate")
    elif model == "decay":
        add("half_life", "half-life", ln2 / p["k"], _inverse(ci["k"], ln2), "duration")
        add("plateau", "plateau", p["c"], raw("c"), "level")
        add("start_gap", "distance from plateau at the first x", p["a"], raw("a"), "level")
    elif model == "hill":
        decreasing = p["top"] < p["bottom"]
        add("ec50", "IC50" if decreasing else "EC50", p["ec50"], raw("ec50"), "position")
        add("top", "plateau at high x", p["top"], raw("top"), "level")
        add("bottom", "baseline at x = 0", p["bottom"], raw("bottom"), "level")
        add("hill", "steepness (Hill slope)", p["hill"], raw("hill"), "rate")
    elif model == "logistic":
        add("midpoint", "midpoint", p["x0"], raw("x0"), "position")
        add("top", "upper plateau", p["top"], raw("top"), "level")
        add("bottom", "lower plateau", p["bottom"], raw("bottom"), "level")
    elif model == "michaelis_menten":
        add("vmax", "plateau (Vmax)", p["vmax"], raw("vmax"), "level")
        add("km", "half-saturation constant (Km)", p["km"], raw("km"), "position")
    elif model == "power":
        add("exponent", "exponent", p["b"], raw("b"), "rate")
    elif model == "log":
        c = ci["b"]
        add("per_doubling", f"change in {y_col} per doubling of {x_col}", p["b"] * ln2,
            None if c is None else (c[0] * ln2, c[1] * ln2), "rate")
    return out


def _core_headline(fit: _Fit, d: dict[str, dict[str, Any]], x_col: str, y_col: str) -> str:
    """One plain sentence for the fitted curve; every number is in `d`."""
    xu = _x_unit(x_col)
    model = fit.spec.name
    p = dict(zip(fit.spec.names, (float(v) for v in fit.params), strict=True))

    def val(name: str) -> str:
        it = d[name]
        return f"{_fmt(it['value'])}{_ci_text(it['ci_lower'], it['ci_upper'])}"

    if model == "linear":
        word = "rises" if p["b"] >= 0 else "falls"
        return f"{y_col} {word} by {_fmt(abs(d['slope']['value']))} per 1 unit of {x_col}{_ci_text(_neg_lo(d['slope'], p['b'] < 0), _neg_hi(d['slope'], p['b'] < 0))}"
    if model == "exponential":
        if "doubling_time" in d:
            return f"{y_col} doubles every {val('doubling_time')} {xu}"
        if "half_life" in d:
            return f"{y_col} halves every {val('half_life')} {xu}"
        return f"{y_col} is essentially flat across {x_col}"
    if model == "decay":
        if abs(p["c"]) <= 0.05 * abs(p["a"]):
            return f"{y_col} {'falls' if p['a'] > 0 else 'rises'} by half every {val('half_life')} {xu}"
        return (
            f"{y_col} closes half the distance to its plateau of {_fmt(p['c'])} every "
            f"{val('half_life')} {xu}"
        )
    if model == "hill":
        name = d["ec50"]["label"]
        return (
            f"{y_col} is half-way between its baseline ({_fmt(p['bottom'])}) and plateau ({_fmt(p['top'])}) "
            f"at {x_col} = {val('ec50')} ({name})"
        )
    if model == "logistic":
        return (
            f"{y_col} switches from about {_fmt(p['bottom'])} to {_fmt(p['top'])} around "
            f"{x_col} = {val('midpoint')}"
        )
    if model == "michaelis_menten":
        return (
            f"{y_col} levels off at about {val('vmax')}; half of that is reached at "
            f"{x_col} = {val('km')}"
        )
    if model == "power":
        return (
            f"{y_col} scales with {x_col} to the power {val('exponent')}, so doubling {x_col} "
            f"multiplies {y_col} by {_fmt(2.0 ** p['b'])}"
        )
    lo, hi = d["per_doubling"]["ci_lower"], d["per_doubling"]["ci_upper"]
    word = "raises" if p["b"] >= 0 else "lowers"
    return (
        f"Each doubling of {x_col} {word} {y_col} by {_fmt(abs(d['per_doubling']['value']))}"
        f"{_ci_text(lo if p['b'] >= 0 else (-hi if hi is not None else None), hi if p['b'] >= 0 else (-lo if lo is not None else None))}"
    )


def _neg_lo(item: dict[str, Any], flip: bool) -> float | None:
    lo, hi = item["ci_lower"], item["ci_upper"]
    if not flip:
        return lo  # type: ignore[no-any-return]
    return None if hi is None else -hi


def _neg_hi(item: dict[str, Any], flip: bool) -> float | None:
    lo, hi = item["ci_lower"], item["ci_upper"]
    if not flip:
        return hi  # type: ignore[no-any-return]
    return None if lo is None else -lo


def _summarise(fit: _Fit, x_col: str, y_col: str, xmin: float, xmax: float) -> dict[str, Any]:
    ints = fit.intervals()
    at_bounds = fit.at_bounds()
    derived = _derive(fit, x_col, y_col, xmin, xmax)
    by_name = {d["name"]: d for d in derived}
    core = _core_headline(fit, by_name, x_col, y_col)
    return {
        "model": fit.spec.name,
        "model_label": fit.spec.label,
        "n": fit.n,
        "r2": _sig(fit.r2),
        "rmse": _sig(fit.rmse),
        "aicc": _sig(fit.aicc),
        "parameters": [
            {
                "name": nm, "value": _sig(float(v)),
                "ci_lower": _sig(iv[0]) if iv else None, "ci_upper": _sig(iv[1]) if iv else None,
                "at_bound": nm in at_bounds,
            }
            for nm, v, iv in zip(fit.spec.names, fit.params, ints, strict=True)
        ],
        "derived": derived,
        "headline_core": core,
        "low_fit": bool(fit.r2 < _LOW_R2),
        "ci_available": all(iv is not None for iv in ints),
        "at_bounds": at_bounds,
    }


def _key_value(fit: _Fit) -> dict[str, Any] | None:
    """The parameter to compare across groups, as (label, value, lo, hi)."""
    names = fit.spec.names
    ints = dict(zip(names, fit.intervals(), strict=True))
    p = dict(zip(names, (float(v) for v in fit.params), strict=True))
    model = fit.spec.name
    if model == "hill":
        label, key = ("IC50" if p["top"] < p["bottom"] else "EC50"), "ec50"
    elif model == "logistic":
        label, key = "midpoint", "x0"
    elif model == "michaelis_menten":
        label, key = "Km", "km"
    elif model == "decay":
        iv = ints["k"]
        lo, hi = _inverse(iv, math.log(2.0))
        return {"label": "half-life", "value": math.log(2.0) / p["k"], "lo": lo, "hi": hi}
    elif model == "power":
        label, key = "exponent", "b"
    elif model == "exponential":
        label, key = "growth rate", "b"
    elif model == "log":
        label, key = "change per ln(x)", "b"
    else:
        label, key = "slope", "b"
    iv2 = ints[key]
    return {"label": label, "value": p[key], "lo": iv2[0] if iv2 else None, "hi": iv2[1] if iv2 else None}


# ---------------------------------------------------------------------------
# Chart
# ---------------------------------------------------------------------------

def _plot_points(x: np.ndarray, y: np.ndarray, budget: int) -> tuple[np.ndarray, np.ndarray, bool]:
    if len(x) <= budget:
        return x, y, False
    ranks = pd.Series(x).rank(method="first")
    bins = pd.qcut(ranks, q=budget, labels=False, duplicates="drop")
    grouped = pd.DataFrame({"x": x, "y": y, "b": np.asarray(bins)}).groupby("b").mean()
    return grouped["x"].to_numpy(dtype=np.float64), grouped["y"].to_numpy(dtype=np.float64), True


def _band(fit: _Fit, grid: np.ndarray, y_range: float) -> tuple[np.ndarray, np.ndarray] | None:
    if not np.all(np.isfinite(fit.pcov)):
        return None
    from scipy import stats

    dof = fit.n - len(fit.params)
    if dof <= 0:
        return None
    f0 = fit.predict(grid)
    jac = np.zeros((len(grid), len(fit.params)))
    for j, pj in enumerate(fit.params):
        step = 1e-6 * max(1.0, abs(float(pj)))
        bumped = fit.params.copy()
        bumped[j] = pj + step
        jac[:, j] = (fit.predict(grid, bumped) - f0) / step
    var = np.einsum("ij,jk,ik->i", jac, fit.pcov, jac)
    if not np.all(np.isfinite(var)):
        return None
    half = float(stats.t.ppf(0.975, dof)) * np.sqrt(np.maximum(var, 0.0))
    if float(half.max()) > 5.0 * max(y_range, 1e-12):
        return None
    return f0 - half, f0 + half


def _curve_chart(
    curves: list[tuple[str | None, _Fit, np.ndarray, np.ndarray]], x_col: str, y_col: str,
    group_col: str | None, title: str, caption: str,
) -> dict[str, Any]:
    g = len(curves)
    point_budget = max(20, _POINT_BUDGET // g)
    curve_budget = max(30, min(_CURVE_BUDGET, 240 // g))
    rows: list[dict[str, Any]] = []
    any_band = False
    all_positive = all(float(x.min()) > 0 for _, _, x, _ in curves)
    for label, fit, x, y in curves:
        px, py, _binned = _plot_points(x, y, point_budget)
        for a, b in zip(px, py, strict=True):
            rows.append({"group": label or "", "kind": "observed", "x": round(float(a), 6), "y": round(float(b), 6)})
        xmin, xmax = float(x.min()), float(x.max())
        pos_min = _pos_min(x)
        if xmax / pos_min > 100 and xmin >= 0:
            grid = np.geomspace(pos_min, xmax, curve_budget)
        else:
            grid = np.linspace(xmin, xmax, curve_budget)
        fitted = fit.predict(grid)
        band = _band(fit, grid, float(np.ptp(y)))
        any_band = any_band or band is not None
        for i, (a, b) in enumerate(zip(grid, fitted, strict=True)):
            if not math.isfinite(float(b)):
                continue
            row: dict[str, Any] = {"group": label or "", "kind": "fitted", "x": round(float(a), 6), "y": round(float(b), 6)}
            if band is not None:
                row["lower"] = round(float(band[0][i]), 6)
                row["upper"] = round(float(band[1][i]), 6)
            rows.append(row)
    y_enc = {"type": "quantitative", "title": y_col, "scale": {"zero": False}}
    x_enc: dict[str, Any] = {"field": "x", "type": "quantitative", "title": x_col}
    if all_positive and any(
        float(x.max()) / _pos_min(x) > 100 for _, _, x, _ in curves
    ):
        x_enc["scale"] = {"type": "log"}
    layers: list[dict[str, Any]] = []
    if any_band:
        layers.append({
            "transform": [{"filter": {"field": "kind", "equal": "fitted"}}],
            "mark": {"type": "area", "opacity": 0.15},
            "encoding": {"y": {"field": "lower", **y_enc}, "y2": {"field": "upper"}},
        })
    layers.append({
        "transform": [{"filter": {"field": "kind", "equal": "observed"}}],
        "mark": {"type": "point", "filled": True, "opacity": 0.55, "size": 40},
        "encoding": {"y": {"field": "y", **y_enc}},
    })
    layers.append({
        "transform": [{"filter": {"field": "kind", "equal": "fitted"}}],
        "mark": {"type": "line"},
        "encoding": {"y": {"field": "y", **y_enc}},
    })
    encoding: dict[str, Any] = {"x": x_enc}
    if group_col and g > 1:
        encoding["color"] = {"field": "group", "type": "nominal", "title": group_col}
    return {
        "type": "vega_lite",
        "title": title,
        "caption": caption[:200],
        "size": "wide",
        "data": rows,
        "vega_lite": {"layer": layers, "encoding": encoding},
    }


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------

class CurveFitAnalysisTool(BaseTool):
    """Non-linear curve fitting: dose-response, decay, growth, saturation."""

    name = "curve_fit_analysis"
    description = (
        "Fit a curve of y against a scientific-style x (dose, concentration, time, temperature, "
        "age...): linear, exponential, decay-to-plateau, logistic, 4-parameter dose-response "
        "(Hill), power, Michaelis-Menten or log. model='auto' tries all and keeps the best by "
        "AICc. Reports R2, RMSE, parameter 95% CIs and plain quantities (half-life, doubling "
        "time, EC50/IC50, plateau, Km); with group_column it fits each group and compares the "
        "key parameter. Draws the points, the fitted curve and its uncertainty band."
    )

    def applies_to(self, profile: DatasetProfile | None, metadata: DatasetMetadata | None) -> float:
        if profile is None or profile.row_count < _MIN_APPLIES_ROWS:
            return 0.0
        # Generic axis names (age, time, day, distance...) exist in most business
        # tables, so on their own they must not offer this tool: they count only
        # on sensor/experiment-shaped data. Dose/concentration-style names always do.
        # A narrow all-numeric table (time, signal) is a measurement table too.
        scientific = profile.archetype in (ARCHETYPE_SENSOR, ARCHETYPE_EXPERIMENT) or (
            profile.column_count <= _NARROW_TABLE_COLS and all(c.kind == "numeric" for c in profile.columns)
        )
        for xc in _x_candidates(profile):
            if not scientific and not (
                (_tokens(xc.name) & _X_TOKENS) - _WEAK_X_TOKENS or column_role(profile, xc.name, "dose_x_strong")
            ):
                continue
            if _y_candidates(profile, xc.name):
                return 0.4
        return 0.0

    def default_params(
        self, profile: DatasetProfile | None, metadata: DatasetMetadata | None
    ) -> dict[str, Any]:
        if profile is None:
            return {}
        for xc in _x_candidates(profile):
            ys = _y_candidates(profile, xc.name)
            if ys:
                return {"x_column": xc.name, "y_column": ys[0].name}
        return {}

    def execute(  # type: ignore[override]
        self,
        file_path: str,
        x_column: str | None = None,
        y_column: str | None = None,
        group_column: str | None = None,
        model: str = "auto",
        **_: Any,
    ) -> dict[str, Any]:
        df = _read_df(file_path)
        if df.empty:
            raise ToolExecutionError("Dataset has no rows.")
        for given in (x_column, y_column, group_column):
            if given and given not in df.columns:
                raise ToolExecutionError(f"Column '{given}' not found in dataset.")
        model_key = str(model or "auto").strip().lower()
        model_key = _ALIASES.get(model_key, model_key.replace("-", "_").replace(" ", "_"))
        if model_key != "auto" and model_key not in _SPECS:
            raise ToolExecutionError(
                f"Unknown model '{model}'. Choose one of: auto, {', '.join(_SPECS)}."
            )
        if x_column is None or y_column is None:
            profile = profile_dataframe(df)
            if x_column is None:
                xs = _x_candidates(profile)
                if not xs:
                    raise ToolExecutionError(
                        "No numeric x column with a scientific name (dose, concentration, time, "
                        "temperature...) found. Pass x_column."
                    )
                x_column = xs[0].name
            if y_column is None:
                ys = _y_candidates(profile, x_column)
                if not ys:
                    raise ToolExecutionError("No numeric y column found. Pass y_column.")
                y_column = ys[0].name
        if x_column == y_column:
            raise ToolExecutionError("x_column and y_column must differ.")
        if group_column in (x_column, y_column):
            group_column = None

        notes: list[str] = []
        if len(df) > _MAX_ROWS:
            df = df.sample(n=_MAX_ROWS, random_state=0)
            notes.append(f"Analysed a random sample of {_MAX_ROWS:,} rows for speed.")
        xv = pd.to_numeric(df[x_column], errors="coerce")
        yv = pd.to_numeric(df[y_column], errors="coerce")
        if xv.notna().sum() == 0:
            raise ToolExecutionError(f"'{x_column}' is not numeric.")
        if yv.notna().sum() == 0:
            raise ToolExecutionError(f"'{y_column}' is not numeric.")
        keep = np.isfinite(xv.to_numpy(dtype=np.float64)) & np.isfinite(yv.to_numpy(dtype=np.float64))
        if group_column:
            keep &= df[group_column].notna().to_numpy()
        work = pd.DataFrame({"x": xv.to_numpy(dtype=np.float64), "y": yv.to_numpy(dtype=np.float64)})[keep]
        if group_column:
            work["g"] = df[group_column].astype(str).to_numpy()[keep]
        dropped = int(len(df) - keep.sum())
        if dropped:
            notes.append(f"{dropped:,} rows with a missing x or y were left out.")
        if work["x"].nunique() < 3:
            raise ToolExecutionError(f"'{x_column}' has fewer than 3 distinct values; nothing to fit a curve to.")
        if len(work) > _FIT_MAX_ROWS:
            work = work.sample(n=_FIT_MAX_ROWS, random_state=0)
            notes.append(f"Curves were fitted on a random {_FIT_MAX_ROWS:,}-row subset for speed.")

        x_all = work["x"].to_numpy(dtype=np.float64)
        y_all = work["y"].to_numpy(dtype=np.float64)
        xmin, xmax = float(x_all.min()), float(x_all.max())
        pooled, candidates = _fit_best(x_all, y_all, model_key)
        result: dict[str, Any] = {
            "x_column": x_column, "y_column": y_column, "group_column": None,
            "model_requested": model_key, "n_points": len(work),
            "x_range": [_sig(xmin), _sig(xmax)], "candidates": candidates,
        }
        result.update(_summarise(pooled, x_column, y_column, xmin, xmax))

        caveats: list[str] = []
        group_fits: list[tuple[str, _Fit, np.ndarray, np.ndarray]] = []
        if group_column:
            group_fits = self._fit_groups(work, model_key, pooled, group_column, notes, result)
        chart_curves: list[tuple[str | None, _Fit, np.ndarray, np.ndarray]] = (
            [(lvl, f, gx, gy) for lvl, f, gx, gy in group_fits] if group_fits
            else [(None, pooled, x_all, y_all)]
        )

        self._caveats(result, pooled, group_fits, caveats, xmin, xmax, x_column, candidates)
        title = (
            f"{y_column} vs {x_column}: {pooled.spec.label} fit"
            if not group_fits else f"{y_column} vs {x_column} by {group_column}"
        )
        result["chart"] = _curve_chart(
            chart_curves, x_column, y_column, result["group_column"], title,
            (result["headline_core"] + ".") if not group_fits else result["comparison"]["sentence"] + ".",
        )
        result["notes"] = notes
        result["caveats"] = caveats
        result["summary"] = self._summary(result)
        return result

    # ---- groups ----

    @staticmethod
    def _fit_groups(
        work: pd.DataFrame, model_key: str, pooled: _Fit, group_column: str, notes: list[str],
        result: dict[str, Any],
    ) -> list[tuple[str, _Fit, np.ndarray, np.ndarray]]:
        min_n = _MIN_N_AUTO if model_key == "auto" else _MIN_N_MODEL
        counts = work["g"].value_counts()
        eligible = [str(k) for k, v in counts.items() if int(v) >= min_n]
        levels = eligible[:_MAX_GROUPS]
        if len(counts) - len(levels) > 0:
            notes.append(
                f"Only the {len(levels)} largest '{group_column}' groups with {min_n}+ points were fitted; "
                f"{len(counts) - len(levels)} other group(s) were ignored."
            )
        data = {
            lvl: (
                work.loc[work["g"] == lvl, "x"].to_numpy(dtype=np.float64),
                work.loc[work["g"] == lvl, "y"].to_numpy(dtype=np.float64),
            )
            for lvl in levels
        }
        fits: dict[str, _Fit] = {}
        for lvl, (gx, gy) in data.items():
            if len(np.unique(gx)) < 3:
                notes.append(f"{group_column} '{lvl}' has fewer than 3 distinct x values; skipped.")
                continue
            try:
                fits[lvl] = _fit_best(gx, gy, model_key)[0]
            except ToolExecutionError as exc:
                notes.append(f"{group_column} '{lvl}': {exc}")
        if len(fits) < 2:
            notes.append(f"'{group_column}' has fewer than 2 fittable groups; no group comparison.")
            return []
        harmonised = False
        if len({f.spec.name for f in fits.values()}) > 1:
            common: dict[str, _Fit] = {}
            for lvl in fits:
                gx, gy = data[lvl]
                try:
                    common[lvl] = _fit_best(gx, gy, pooled.spec.name)[0]
                except ToolExecutionError:
                    break
            if len(common) == len(fits):
                fits, harmonised = common, True
                notes.append(
                    f"The best curve differed by group, so every group was refitted with the "
                    f"overall best model ({pooled.spec.label}) to make parameters comparable."
                )
        result["group_column"] = group_column
        rows: list[dict[str, Any]] = []
        keys: dict[str, dict[str, Any]] = {}
        for lvl, f in fits.items():
            gx, gy = data[lvl]
            summary = _summarise(f, result["x_column"], result["y_column"], float(gx.min()), float(gx.max()))
            summary["group"] = lvl
            rows.append(summary)
            kv = _key_value(f)
            if kv is not None:
                keys[lvl] = kv
        result["groups"] = rows
        same_model = len({f.spec.name for f in fits.values()}) == 1
        result["comparison"] = _compare(keys, same_model, group_column, harmonised)
        return [(lvl, f, *data[lvl]) for lvl, f in fits.items()]

    @staticmethod
    def _caveats(
        result: dict[str, Any], pooled: _Fit, group_fits: list[tuple[str, _Fit, np.ndarray, np.ndarray]],
        caveats: list[str], xmin: float, xmax: float, x_col: str, candidates: list[dict[str, Any]],
    ) -> None:
        summaries = result.get("groups") or [result]
        for s in summaries:
            tag = f" ({result['group_column']} {s['group']})" if "group" in s else ""
            if s["low_fit"]:
                caveats.append(f"No clear curve{tag}: the best fit explains only {(s['r2'] or 0) * 100:.0f}% of the variation.")
            if s["at_bounds"]:
                caveats.append(
                    f"Parameter(s) {', '.join(s['at_bounds'])}{tag} sit at the limit the fit allows, so they are "
                    "not well determined by this data."
                )
            if not s["ci_available"]:
                caveats.append(f"Parameter uncertainty could not be estimated{tag} (the fit is poorly determined).")
            for d in s["derived"]:
                if d["outside_range"]:
                    caveats.append(
                        f"The {d['label']}{tag} lies outside the observed {x_col} range "
                        f"({_fmt(xmin)} to {_fmt(xmax)}), so it is an extrapolation."
                    )
            if s["n"] < _SMALL_N:
                caveats.append(f"Only {s['n']} points{tag}; the choice of curve and its parameters are fragile.")
        caveats.append(
            f"The curve describes {x_col} between {_fmt(xmin)} and {_fmt(xmax)}; do not extrapolate beyond it."
        )
        close = [c for c in candidates if not c["chosen"] and c["aicc"] is not None]
        chosen = next((c for c in candidates if c["chosen"]), None)
        if result["model_requested"] == "auto" and chosen and chosen["aicc"] is not None:
            rivals = [c["model"] for c in close if c["aicc"] - chosen["aicc"] <= _AICC_TIE]
            if rivals:
                caveats.append(
                    f"{', '.join(rivals)} fit(s) about as well as {chosen['model']}; the curve shape is not decisive."
                )
        if group_fits:
            caveats.append("Group differences are associations; other differences between the groups may explain them.")
        caveats[:] = list(dict.fromkeys(caveats))

    @staticmethod
    def _summary(r: dict[str, Any]) -> str:
        if r.get("comparison"):
            return str(r["comparison"]["sentence"]) + "."
        if r["low_fit"]:
            return (
                f"No clear curve: the best fit ({r['model_label']}) explains only {r['r2'] * 100:.0f}% of "
                f"the variation in {r['y_column']} across {r['x_column']}."
            )
        return f"{r['headline_core']}; the {r['model_label']} curve explains {r['r2'] * 100:.0f}% of the variation."

    # ---- findings ----

    def findings(
        self,
        output: dict[str, Any],
        profile: DatasetProfile | None,
        metadata: DatasetMetadata | None,
    ) -> list[Finding]:
        x_col, y_col = str(output.get("x_column")), str(output.get("y_column"))
        group = output.get("group_column")
        caveats = list(output.get("caveats") or [])
        results: list[Finding] = []
        comparison = output.get("comparison")
        if comparison:
            best = comparison.get("featured")
            results.append(Finding(
                finding_id=f"curve_fit_compare_{y_col}_{x_col}_{group}",
                kind="curve_fit",
                headline=comparison["sentence"] + ".",
                detail=(
                    f"Best curve per {group}: "
                    + ", ".join(f"{g['group']} = {g['model_label']} (R2 {(g['r2'] or 0):.2f})" for g in output["groups"])
                    + "."
                ),
                evidence={
                    "parameter": comparison["parameter"], "pairs": comparison["pairs"][:6],
                    "n": sum(g["n"] for g in output["groups"]),
                },
                source_tool=self.name,
                measure=y_col,
                dimension=str(group),
                effect=best["effect"] if best else 0.0,
                effect_kind="lift",
                confidence=0.7 if best else 0.5,
                surprise=0.5 if best else 0.2,
                caveats=caveats,
                chart_hint=output.get("chart"),
            ))
            for g in sorted(output["groups"], key=lambda s: -(s["r2"] or 0.0))[:2]:
                results.append(self._fit_finding(g, output, x_col, y_col, str(group), caveats, level=g["group"]))
            return results
        results.append(self._fit_finding(output, output, x_col, y_col, None, caveats, level=None))
        return results

    def _fit_finding(
        self, s: dict[str, Any], output: dict[str, Any], x_col: str, y_col: str, group: str | None,
        caveats: list[str], level: str | None,
    ) -> Finding:
        r2 = float(s["r2"] or 0.0)
        where = f" for {group} {level}" if level is not None else ""
        if s["low_fit"]:
            headline = (
                f"No clear curve{where}: the best fit ({s['model_label']}) explains only "
                f"{r2 * 100:.0f}% of the variation in {y_col} across {x_col}."
            )
        else:
            headline = f"{s['headline_core']}{where}, explaining {r2 * 100:.0f}% of the variation in {y_col}."
        primary = s["derived"][0] if s["derived"] else {}
        n = int(s["n"])
        return Finding(
            finding_id=f"curve_fit_{y_col}_{x_col}_{level if level is not None else 'all'}".replace(" ", "_"),
            kind="curve_fit",
            headline=headline,
            detail=(
                f"{s['model_label']} fitted by non-linear least squares; R2 {r2:.3f}, RMSE {s['rmse']}. "
                + "; ".join(f"{p['name']} = {p['value']}" for p in s["parameters"])
            ),
            evidence={
                "model": s["model"], "r2": s["r2"], "rmse": s["rmse"], "n": n,
                "value": primary.get("value"), "ci_lower": primary.get("ci_lower"), "ci_upper": primary.get("ci_upper"),
                "parameters": s["parameters"],
            },
            source_tool=self.name,
            measure=y_col,
            dimension=group,
            level=level,
            effect=r2,
            effect_kind="r2",
            confidence=round(min(1.0, n / 50.0) * (0.5 + 0.5 * max(r2, 0.0)), 3),
            surprise=0.3,
            caveats=caveats,
            chart_hint=output.get("chart"),
        )

    def get_schema(self) -> dict[str, Any]:
        return {
            "file_path": {
                "type": "string",
                "description": "Path to the dataset.",
                "required": True,
            },
            "x_column": {
                "type": "string",
                "description": "Numeric predictor (dose, concentration, time, temperature...). Auto-detected when omitted.",
                "required": False,
            },
            "y_column": {
                "type": "string",
                "description": "Numeric response to model. Auto-selected when omitted.",
                "required": False,
            },
            "group_column": {
                "type": "string",
                "description": "Optional categorical column (up to 6 groups): fit the curve per group and compare the key parameter.",
                "required": False,
            },
            "model": {
                "type": "string",
                "description": (
                    "auto (default: best of all by AICc) | linear | exponential | decay | logistic | hill | "
                    "power | michaelis_menten | log."
                ),
                "required": False,
            },
        }


def _compare(
    keys: dict[str, dict[str, Any]], same_model: bool, group_column: str, harmonised: bool
) -> dict[str, Any]:
    levels = list(keys)
    label = keys[levels[0]]["label"] if levels else "parameter"
    if not same_model:
        return {
            "parameter": "not comparable", "pairs": [], "featured": None,
            "sentence": (
                f"The best curve differs across {group_column} ({', '.join(levels)}), so a single "
                "parameter cannot be compared directly"
            ),
        }
    pairs: list[dict[str, Any]] = []
    for i, a in enumerate(levels):
        for b in levels[i + 1:]:
            ka, kb = keys[a], keys[b]
            overlap: bool | None = None
            if None not in (ka["lo"], ka["hi"], kb["lo"], kb["hi"]):
                overlap = bool(ka["lo"] <= kb["hi"] and kb["lo"] <= ka["hi"])
            ratio = ka["value"] / kb["value"] if kb["value"] not in (0, 0.0) and ka["value"] * kb["value"] > 0 else None
            pairs.append({
                "a": a, "b": b, "value_a": _sig(ka["value"]), "value_b": _sig(kb["value"]),
                "ci_a": [_sig(ka["lo"]), _sig(ka["hi"])], "ci_b": [_sig(kb["lo"]), _sig(kb["hi"])],
                "ratio": _sig(ratio), "intervals_overlap": overlap,
            })
    distinct = [p for p in pairs if p["intervals_overlap"] is False]
    parameter = label
    if distinct:
        top = max(distinct, key=lambda p: abs(math.log(p["ratio"])) if p["ratio"] else 0.0)
        ka, kb = keys[top["a"]], keys[top["b"]]
        sentence = (
            f"{parameter} differs by {group_column}: {top['a']} {_fmt(ka['value'])}{_ci_text(ka['lo'], ka['hi'])} "
            f"vs {top['b']} {_fmt(kb['value'])}{_ci_text(kb['lo'], kb['hi'])}, and the ranges do not overlap"
        )
        effect = (top["ratio"] - 1.0) if top["ratio"] else None
        featured: dict[str, Any] | None = {"pair": [top["a"], top["b"]], "effect": _sig(effect) if effect is not None else 0.1}
    else:
        parts = ", ".join(f"{lvl} {_fmt(keys[lvl]['value'])}" for lvl in levels[:4])
        sentence = f"{parameter} is not clearly different across {group_column} ({parts}); the ranges overlap"
        featured = None
    return {
        "parameter": parameter, "pairs": pairs, "featured": featured, "sentence": sentence,
        "refitted_with_common_model": harmonised,
    }
