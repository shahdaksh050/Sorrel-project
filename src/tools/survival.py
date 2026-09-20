"""
Survival Analysis Tool — Execution Layer.

"How long until it happens, and who gets there sooner?" for time-to-event
data (customer tenure until churn, time to failure, time to relapse) where
some rows have not had the event yet (censored):

  - Kaplan-Meier survival curve (vectorised cumulative counts) with
    Greenwood log-log 95% confidence bands, the median time to event with its
    interval (or the honest "more than half still survive to the end of
    follow-up"), and survival at three milestones (25/50/75% of the longest
    follow-up);
  - with a group column (top 6 by size): one curve per group, an overall
    log-rank test, and Cox proportional hazards (statsmodels PHReg) hazard
    ratios with 95% CIs, phrased as "B leaves 1.8x faster than A";
  - with covariate columns (<= 5): the same Cox model adjusts for them, each
    numeric covariate reported per 1 standard deviation.

Everything is deterministic and fitted per call; nothing is trained or stored.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from src.core.findings import Finding
from src.core.profiler import profile_dataframe
from src.core.vocab import ROLE_TOKENS, column_role, name_tokens
from src.tools.base import BaseTool, ToolExecutionError
from src.tools.data_processing import _read_df

if TYPE_CHECKING:
    from src.core.memory import DatasetMetadata
    from src.core.profiler import ColumnProfile, DatasetProfile

_MAX_ROWS = 50_000
#: PHReg is O(unique event times x rows) per iteration; a random subset this
#: size gives hazard ratios as precise as the analyst can use.
_COX_MAX_ROWS = 8_000
_MAX_GROUPS = 6
_MAX_COVARIATES = 5
_MAX_COVARIATE_LEVELS = 6
_MAX_DESIGN_COLUMNS = 12
_MIN_GROUP_N = 5
_MIN_ROWS = 10
_MIN_APPLIES_ROWS = 30
_Z = 1.959964
_HEAVY_CENSORING = 0.80
_MIN_EVENTS_PER_GROUP = 10
_MILESTONES = (0.25, 0.5, 0.75)
_CHART_ROW_BUDGET = 440
_MAX_BAND_GROUPS = 3
#: A hazard ratio this close to 1 is not worth a headline even when significant.
_MIN_HR_EFFECT = 0.10
_MAX_HR_FINDINGS = 3

_DURATION_TOKENS = ROLE_TOKENS["duration"]
_EVENT_TOKENS = ROLE_TOKENS["event"]
_TIME_UNITS = {
    "day": "days", "week": "weeks", "month": "months", "year": "years", "hour": "hours",
    "minute": "minutes",
}
#: (verb, base verb, noun) used to phrase the event in plain words.
_EVENT_WORDS: tuple[tuple[frozenset[str], tuple[str, str, str]], ...] = (
    (frozenset({"churn", "churned", "attrition", "left", "terminated", "dropout"}), ("leaves", "leave", "leaving")),
    (frozenset({"died", "death", "dead", "deceased"}), ("dies", "die", "dying")),
    (frozenset({"failure", "failed"}), ("fails", "fail", "failing")),
    (frozenset({"converted"}), ("converts", "convert", "converting")),
    (frozenset({"relapse"}), ("relapses", "relapse", "relapsing")),
)
_DEFAULT_EVENT_WORDS = ("has the event", "have had the event", "the event")

_PLAIN_WORDS = frozenset({"1", "0", "1.0", "0.0", "yes", "no", "y", "n", "true", "false", "t", "f"})
_WORD_MAP: dict[str, float] = {
    **dict.fromkeys(
        ("1", "1.0", "yes", "y", "true", "t", "event", "observed", "died", "dead", "death", "deceased",
         "churned", "churn", "left", "failed", "failure", "converted", "relapsed", "relapse",
         "terminated", "dropped", "dropout", "attrited", "attrition"),
        1.0,
    ),
    **dict.fromkeys(
        ("0", "0.0", "no", "n", "false", "f", "alive", "active", "retained", "stayed", "censored",
         "survived", "ongoing", "current", "none"),
        0.0,
    ),
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


def _is_duration_name(name: str, profile: DatasetProfile | None = None) -> bool:
    toks = _tokens(name)
    return column_role(profile, name, "duration") or bool(
        toks & _DURATION_TOKENS
        or {"age", "at"} <= toks
        or {"follow", "up"} <= toks
    )


def _is_event_name(name: str, profile: DatasetProfile | None = None) -> bool:
    return column_role(profile, name, "event") or bool(_tokens(name) & _EVENT_TOKENS)


def _mappable(top_values: dict[str, int]) -> bool:
    return bool(top_values) and all(str(k).strip().lower() in _WORD_MAP for k in top_values)


def _duration_candidates(profile: DatasetProfile) -> list[ColumnProfile]:
    return [
        c for c in profile.columns
        if c.kind == "numeric" and c.nunique >= 3 and _is_duration_name(c.name, profile)
    ]


def _event_candidates(profile: DatasetProfile, exclude: set[str]) -> list[ColumnProfile]:
    out: list[ColumnProfile] = []
    for c in profile.columns:
        if c.name in exclude or not _is_event_name(c.name, profile):
            continue
        if c.kind in ("boolean", "numeric") and c.nunique == 2:
            out.append(c)
        elif c.kind == "categorical" and 2 <= c.nunique <= 3 and _mappable(c.top_values):
            out.append(c)
    return out


def _event_words(name: str) -> tuple[str, str, str]:
    toks = _tokens(name)
    for keys, words in _EVENT_WORDS:
        if toks & keys:
            return words
    return _DEFAULT_EVENT_WORDS


def _time_unit(name: str) -> str:
    toks = _tokens(name)
    for singular, plural in _TIME_UNITS.items():
        if singular in toks:
            return plural
    return f"{name} units"


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def _encode_event(series: pd.Series, column: str) -> tuple[pd.Series, str]:
    """(1.0 = event / 0.0 = censored / NaN = missing, coding note)."""
    plain = True
    if pd.api.types.is_bool_dtype(series):
        coded = series.astype("float64")
    elif pd.api.types.is_numeric_dtype(series):
        values = sorted({float(v) for v in series.dropna().unique()})
        if not values:
            raise ToolExecutionError(f"Event column '{column}' has no values.")
        if len(values) > 2:
            raise ToolExecutionError(
                f"Event column '{column}' has {len(values)} distinct values; it must be two-valued "
                "(1 = event happened, 0 = censored)."
            )
        top = 1.0 if set(values) <= {0.0, 1.0} else values[-1]
        coded = (series.astype("float64") == top).astype("float64").where(series.notna())
    else:
        present = series.notna()
        text = series[present].astype(str).str.strip().str.lower()
        lookup = {u: _WORD_MAP.get(u, float("nan")) for u in text.unique()}
        mapped = text.map(lookup).astype("float64")
        unknown = text[mapped.isna()]
        if len(unknown):
            shown = ", ".join(repr(str(v)) for v in list(unknown.unique())[:4])
            raise ToolExecutionError(
                f"Event column '{column}' has values I cannot read as event/no-event ({shown}). "
                "Recode it as 1 = event happened, 0 = censored."
            )
        coded = pd.Series(np.nan, index=series.index, dtype="float64")
        coded.loc[text.index] = mapped
        plain = set(text.unique()) <= _PLAIN_WORDS
    note = "1 / yes = event happened"
    if plain and _tokens(column) & {"censor", "censored"}:
        coded = 1.0 - coded
        note = f"column '{column}' records censoring, so 1 / yes was read as censored (no event)"
    return coded, note


def _clean_covariates(raw: Any, df: pd.DataFrame, exclude: set[str]) -> tuple[list[str], list[str]]:
    if raw is None:
        return [], []
    names = [s.strip() for s in raw.split(",")] if isinstance(raw, str) else [str(s) for s in raw]
    covs: list[str] = []
    notes: list[str] = []
    for name in names:
        if not name or name in covs:
            continue
        if name not in df.columns:
            raise ToolExecutionError(f"Covariate column '{name}' not found in dataset.")
        if name in exclude:
            continue
        covs.append(name)
    if len(covs) > _MAX_COVARIATES:
        notes.append(f"Only the first {_MAX_COVARIATES} covariates were used; ignored: {', '.join(covs[_MAX_COVARIATES:])}.")
        covs = covs[:_MAX_COVARIATES]
    return covs, notes


# ---------------------------------------------------------------------------
# Kaplan-Meier, log-rank, Cox
# ---------------------------------------------------------------------------

def _km(t: np.ndarray, e: np.ndarray) -> dict[str, Any]:
    order = np.argsort(t, kind="stable")
    ts = t[order]
    es = e[order].astype(np.float64)
    times, start = np.unique(ts, return_index=True)
    n_risk = (len(ts) - start).astype(np.float64)
    d = np.add.reduceat(es, start)
    with np.errstate(all="ignore"):
        surv = np.cumprod(1.0 - d / n_risk)
        terms = np.where(n_risk > d, d / (n_risk * (n_risk - d)), 0.0)
        var = np.cumsum(terms)
        log_s = np.log(surv)
        se = np.sqrt(var) / np.abs(log_s)
        valid = (surv > 0) & (surv < 1) & np.isfinite(se)
        factor = np.exp(_Z * np.where(valid, se, 0.0))
        lo = np.where(valid, surv ** factor, surv)
        hi = np.where(valid, surv ** (1.0 / factor), surv)
    return {
        "times": times, "surv": surv, "lo": lo, "hi": hi, "d": d, "n": len(ts),
        "events": int(es.sum()), "tmax": float(times[-1]),
    }


def _first_time(times: np.ndarray, cond: np.ndarray) -> float | None:
    idx = np.flatnonzero(cond)
    return float(times[idx[0]]) if idx.size else None


def _median(km: dict[str, Any]) -> dict[str, float | None]:
    times, surv = km["times"], km["surv"]
    return {
        "median": _first_time(times, surv <= 0.5),
        "ci_lower": _first_time(times, km["hi"] <= 0.5),
        "ci_upper": _first_time(times, km["lo"] <= 0.5),
    }


def _step_at(km: dict[str, Any], at: np.ndarray, key: str = "surv") -> np.ndarray:
    idx = np.searchsorted(km["times"], at, side="right") - 1
    vals = km[key][np.clip(idx, 0, None)]
    return np.asarray(np.where(idx >= 0, vals, 1.0), dtype=np.float64)


def _milestones(km: dict[str, Any], at: np.ndarray) -> list[dict[str, float]]:
    s, lo, hi = _step_at(km, at), _step_at(km, at, "lo"), _step_at(km, at, "hi")
    return [
        {"time": float(a), "survival": float(x), "ci_lower": float(l_), "ci_upper": float(h_)}
        for a, x, l_, h_ in zip(at, s, lo, hi, strict=True)
    ]


def _log_rank(groups: list[tuple[np.ndarray, np.ndarray]]) -> dict[str, float] | None:
    from scipy import stats

    all_t = np.unique(np.concatenate([g[0] for g in groups]))
    k, m = len(groups), len(all_t)
    at_risk = np.zeros((k, m))
    events = np.zeros((k, m))
    for i, (t, e) in enumerate(groups):
        order = np.argsort(t, kind="stable")
        ts, es = t[order], e[order].astype(np.float64)
        cum = np.concatenate([[0.0], np.cumsum(es)])
        left = np.searchsorted(ts, all_t, side="left")
        right = np.searchsorted(ts, all_t, side="right")
        at_risk[i] = len(ts) - left
        events[i] = cum[right] - cum[left]
    n_tot = at_risk.sum(axis=0)
    d_tot = events.sum(axis=0)
    if d_tot.sum() == 0 or k < 2:
        return None
    with np.errstate(all="ignore"):
        expected = at_risk * (d_tot / n_tot)
        c = np.where(n_tot > 1, d_tot * (n_tot - d_tot) / (n_tot - 1.0), 0.0)
        share = at_risk / n_tot
    diff = (events.sum(axis=1) - np.nan_to_num(expected).sum(axis=1))[:-1]
    weighted = np.nan_to_num(share * c)
    cov = (np.diag(weighted.sum(axis=1)) - weighted @ np.nan_to_num(share).T)[:-1, :-1]
    try:
        chi2 = float(diff @ np.linalg.solve(cov, diff))
    except np.linalg.LinAlgError:
        return None
    if not math.isfinite(chi2) or chi2 < 0:
        return None
    return {"chi2": chi2, "df": float(k - 1), "p_value": float(stats.chi2.sf(chi2, k - 1))}


def _design(
    work: pd.DataFrame, covs: list[str], covdf: pd.DataFrame, ref: str | None, levels: list[str]
) -> tuple[np.ndarray, list[dict[str, Any]], list[str]]:
    cols: list[np.ndarray] = []
    terms: list[dict[str, Any]] = []
    notes: list[str] = []
    if ref is not None:
        for lvl in levels:
            if lvl == ref:
                continue
            cols.append((work["g"] == lvl).to_numpy(dtype=np.float64))
            terms.append({"kind": "group", "label": lvl, "reference": ref})
    for c in covs:
        s = covdf[c]
        if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s) and s.nunique() > 2:
            x = s.to_numpy(dtype=np.float64)
            sd = float(np.std(x))
            if sd <= 0:
                notes.append(f"Covariate '{c}' is constant; skipped.")
                continue
            cols.append((x - float(np.mean(x))) / sd)
            terms.append({"kind": "numeric", "label": c, "sd": sd})
            continue
        text = s.astype(str)
        counts = text.value_counts()
        if len(counts) > _MAX_COVARIATE_LEVELS:
            notes.append(f"Covariate '{c}' has more than {_MAX_COVARIATE_LEVELS} levels; skipped.")
            continue
        base = str(counts.index[0])
        for lvl in counts.index[1:]:
            if int(counts[lvl]) < _MIN_GROUP_N:
                continue
            cols.append((text == lvl).to_numpy(dtype=np.float64))
            terms.append({"kind": "category", "label": c, "level": str(lvl), "reference": base})
    if len(cols) > _MAX_DESIGN_COLUMNS:
        notes.append(f"Model limited to {_MAX_DESIGN_COLUMNS} predictors.")
        cols, terms = cols[:_MAX_DESIGN_COLUMNS], terms[:_MAX_DESIGN_COLUMNS]
    matrix = np.column_stack(cols) if cols else np.empty((len(work), 0))
    return matrix, terms, notes


def _cox(
    work: pd.DataFrame, covs: list[str], covdf: pd.DataFrame, ref: str | None, levels: list[str]
) -> tuple[dict[str, Any] | None, list[str]]:
    notes: list[str] = []
    keep = pd.Series(True, index=work.index)
    for c in covs:
        keep &= covdf[c].notna()
    dropped = int((~keep).sum())
    if dropped:
        notes.append(f"{dropped:,} rows with a missing covariate were left out of the hazard-ratio model.")
    sub, subcov = work[keep], covdf[keep]
    if len(sub) > _COX_MAX_ROWS:
        picked = sub.sample(n=_COX_MAX_ROWS, random_state=0).index
        notes.append(f"Hazard ratios were fitted on a random {_COX_MAX_ROWS:,}-row subset for speed.")
        sub, subcov = sub.loc[picked], subcov.loc[picked]
    matrix, terms, design_notes = _design(sub, covs, subcov, ref, levels)
    notes.extend(design_notes)
    if not terms:
        return None, notes
    n_events = int(sub["e"].sum())
    if n_events < 5:
        notes.append("Too few events to fit hazard ratios.")
        return None, notes
    try:
        from statsmodels.duration.hazard_regression import PHReg

        model = PHReg(
            sub["t"].to_numpy(dtype=np.float64), matrix,
            status=sub["e"].to_numpy(dtype=np.float64), ties="breslow",
        )
        res = model.fit(disp=False)
        params = np.asarray(res.params, dtype=np.float64)
        bse = np.asarray(res.bse, dtype=np.float64)
        pvals = np.asarray(res.pvalues, dtype=np.float64)
    except Exception as exc:
        notes.append(f"Hazard-ratio model could not be fitted ({type(exc).__name__}).")
        return None, notes
    rows: list[dict[str, Any]] = []
    for term, b, s, p in zip(terms, params, bse, pvals, strict=True):
        if not (math.isfinite(b) and math.isfinite(s)) or abs(b) > 15 or s > 5:
            notes.append(f"Hazard ratio for '{term['label']}' is unstable (near-perfect separation); left out.")
            continue
        rows.append({
            **term,
            "hr": float(math.exp(b)),
            "ci_lower": float(math.exp(b - _Z * s)),
            "ci_upper": float(math.exp(b + _Z * s)),
            "p_value": float(p) if math.isfinite(p) else None,
        })
    if not rows:
        return None, notes
    return {"n": len(sub), "events": n_events, "hazard_ratios": rows}, notes


# ---------------------------------------------------------------------------
# Wording helpers
# ---------------------------------------------------------------------------

def _fmt(x: float) -> str:
    return f"{x:,.0f}" if abs(x) >= 100 else f"{x:.1f}" if abs(x) >= 10 else f"{x:.2f}"


def _ratio(x: float) -> str:
    return f"{x:.1f}" if x >= 1.15 else f"{x:.2f}"


def _hr_sentence(term: dict[str, Any], words: tuple[str, str, str], col: str) -> str:
    hr, lo, hi = term["hr"], term["ci_lower"], term["ci_upper"]
    faster = hr >= 1.0
    if not faster:
        hr, lo, hi = 1.0 / hr, 1.0 / hi, 1.0 / lo
    pace = "faster" if faster else "slower"
    ci = f"(95% CI {_ratio(lo)}-{_ratio(hi)})"
    verb, _base, noun = words
    if term["kind"] == "group":
        return f"{col} {term['label']} {verb} {_ratio(hr)}x {pace} than {term['reference']} {ci}"
    if term["kind"] == "numeric":
        action = "speeds up" if faster else "slows"
        return f"each 1-SD higher {term['label']} ({_fmt(term['sd'])}) {action} {noun} {_ratio(hr)}x {ci}"
    return (
        f"{term['label']} {term['level']} {verb} {_ratio(hr)}x {pace} than {term['label']} "
        f"{term['reference']} {ci}"
    )


def _median_sentence(out: dict[str, Any]) -> str:
    unit = out["time_unit"]
    med = out["median_survival"]
    if med["median"] is None:
        end = out["survival_at_end"]
        return (
            f"More than half still survive to the end of follow-up "
            f"({_fmt(out['max_follow_up'])} {unit}; {end * 100:.0f}% remain)"
        )
    text = f"Half {out['event_words'][1]} within {_fmt(med['median'])} {unit}"
    if med["ci_lower"] is not None and med["ci_upper"] is not None:
        return f"{text} (95% CI {_fmt(med['ci_lower'])}-{_fmt(med['ci_upper'])})"
    if med["ci_lower"] is not None:
        return f"{text} (95% CI from {_fmt(med['ci_lower'])}, upper bound not reached)"
    return text


def _crossing(kms: list[dict[str, Any]], grid: np.ndarray) -> bool:
    curves = np.vstack([_step_at(k, grid) for k in kms])
    diff = curves[1:] - curves[0]
    return bool(np.any((diff.max(axis=1) > 0.05) & (diff.min(axis=1) < -0.05)))


# ---------------------------------------------------------------------------
# Chart
# ---------------------------------------------------------------------------

def _km_chart(
    curves: list[tuple[str, dict[str, Any]]], unit: str, group_column: str | None, title: str,
    caption: str,
) -> dict[str, Any]:
    budget = max(20, _CHART_ROW_BUDGET // len(curves))
    bands = len(curves) <= _MAX_BAND_GROUPS
    rows: list[dict[str, Any]] = []
    for label, km in curves:
        sel = np.flatnonzero(km["d"] > 0)
        t = np.concatenate([[0.0], km["times"][sel]])
        s = np.concatenate([[1.0], km["surv"][sel]])
        lo = np.concatenate([[1.0], km["lo"][sel]])
        hi = np.concatenate([[1.0], km["hi"][sel]])
        if t[-1] != km["tmax"]:
            t = np.append(t, km["tmax"])
            s = np.append(s, km["surv"][-1])
            lo = np.append(lo, km["lo"][-1])
            hi = np.append(hi, km["hi"][-1])
        if len(t) > budget:
            keep = np.unique(np.linspace(0, len(t) - 1, budget).round().astype(int))
            t, s, lo, hi = t[keep], s[keep], lo[keep], hi[keep]
        for a, b, c, d in zip(t, s, lo, hi, strict=True):
            row: dict[str, Any] = {"group": label, "time": round(float(a), 4), "survival": round(float(b), 4)}
            if bands:
                row["lower"] = round(float(c), 4)
                row["upper"] = round(float(d), 4)
            rows.append(row)
    y_axis = {"format": "%", "title": "Share still surviving"}
    y_scale = {"domain": [0, 1]}
    layers: list[dict[str, Any]] = []
    if bands:
        layers.append({
            "mark": {"type": "area", "interpolate": "step-after", "opacity": 0.15},
            "encoding": {
                "y": {"field": "lower", "type": "quantitative", "scale": y_scale, "axis": y_axis},
                "y2": {"field": "upper"},
            },
        })
    layers.append({
        "mark": {"type": "line", "interpolate": "step-after"},
        "encoding": {"y": {"field": "survival", "type": "quantitative", "scale": y_scale, "axis": y_axis}},
    })
    encoding: dict[str, Any] = {
        "x": {"field": "time", "type": "quantitative", "title": f"Time ({unit})"},
    }
    if group_column and len(curves) > 1:
        encoding["color"] = {"field": "group", "type": "nominal", "title": group_column}
    return {
        "type": "vega_lite",
        "title": title,
        "caption": caption[:200],
        "size": "wide",
        "data": rows,
        "vega_lite": {"layer": layers, "encoding": encoding},
    }


def _r4(x: float | None) -> float | None:
    return None if x is None else round(float(x), 4)


def _round_median(m: dict[str, float | None]) -> dict[str, float | None]:
    return {k: _r4(v) for k, v in m.items()}


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------

class SurvivalAnalysisTool(BaseTool):
    """Kaplan-Meier survival / retention curves, log-rank and Cox hazard ratios."""

    name = "survival_analysis"
    description = (
        "Time-to-event analysis with censoring (customer tenure until churn, time to failure "
        "or relapse): Kaplan-Meier survival curve with confidence bands, median time to event "
        "(or 'more than half survive'), survival at milestones, and - with group_column and/or "
        "covariate_columns - a log-rank comparison of groups and Cox hazard ratios in plain "
        "words ('B leaves 1.8x faster'). Needs a numeric duration column and a 0/1 or yes/no "
        "event column (a column named 'censored' is read as inverted)."
    )

    def applies_to(self, profile: DatasetProfile | None, metadata: DatasetMetadata | None) -> float:
        if profile is None or profile.row_count < _MIN_APPLIES_ROWS:
            return 0.0
        durations = _duration_candidates(profile)
        if not durations:
            return 0.0
        for d in durations:
            if _event_candidates(profile, {d.name}):
                return 0.6
        return 0.0

    def default_params(
        self, profile: DatasetProfile | None, metadata: DatasetMetadata | None
    ) -> dict[str, Any]:
        if profile is None:
            return {}
        for d in _duration_candidates(profile):
            events = _event_candidates(profile, {d.name})
            if events:
                params: dict[str, Any] = {"duration_column": d.name, "event_column": events[0].name}
                taken = {d.name, events[0].name}
                groups = [
                    c for c in profile.columns
                    if c.kind == "categorical" and 2 <= c.nunique <= _MAX_GROUPS and c.name not in taken
                    and not _is_event_name(c.name, profile)
                ]
                if groups:
                    params["group_column"] = groups[0].name
                return params
        return {}

    def execute(  # type: ignore[override]
        self,
        file_path: str,
        duration_column: str | None = None,
        event_column: str | None = None,
        group_column: str | None = None,
        covariate_columns: Any = None,
        **_: Any,
    ) -> dict[str, Any]:
        df = _read_df(file_path)
        if df.empty:
            raise ToolExecutionError("Dataset has no rows.")
        for given in (duration_column, event_column, group_column):
            if given and given not in df.columns:
                raise ToolExecutionError(f"Column '{given}' not found in dataset.")
        if duration_column is None or event_column is None:
            profile = profile_dataframe(df)
            if duration_column is None:
                cands = _duration_candidates(profile)
                if not cands:
                    raise ToolExecutionError(
                        "No duration column found (a numeric column such as tenure, days, time). "
                        "Pass duration_column."
                    )
                duration_column = cands[0].name
            if event_column is None:
                ev = _event_candidates(profile, {duration_column})
                if not ev:
                    raise ToolExecutionError(
                        "No event column found (a 0/1 or yes/no column such as churned, event, status). "
                        "Pass event_column."
                    )
                event_column = ev[0].name
        if duration_column == event_column:
            raise ToolExecutionError("duration_column and event_column must differ.")
        if group_column in (duration_column, event_column):
            group_column = None
        covs, notes = _clean_covariates(
            covariate_columns, df, {duration_column, event_column, group_column or ""}
        )

        if len(df) > _MAX_ROWS:
            df = df.sample(n=_MAX_ROWS, random_state=0)
            notes.append(f"Analysed a random sample of {_MAX_ROWS:,} rows for speed.")
        duration = pd.to_numeric(df[duration_column], errors="coerce")
        if duration.notna().sum() == 0:
            raise ToolExecutionError(f"'{duration_column}' is not numeric.")
        event, coding = _encode_event(df[event_column], event_column)
        keep = duration.notna() & event.notna() & (duration > 0)
        if group_column:
            keep &= df[group_column].notna()
        n_dropped = int(len(df) - keep.sum())
        if n_dropped:
            notes.append(f"{n_dropped:,} rows with a missing or non-positive duration/event were left out.")
        work = pd.DataFrame({"t": duration, "e": event})[keep]
        work["e"] = work["e"].astype(bool)
        if group_column:
            work["g"] = df.loc[work.index, group_column].astype(str)
        covdf = df.loc[work.index, covs] if covs else pd.DataFrame(index=work.index)
        if len(work) < _MIN_ROWS:
            raise ToolExecutionError(f"Only {len(work)} usable rows; need at least {_MIN_ROWS}.")
        n_events = int(work["e"].sum())
        if n_events == 0:
            raise ToolExecutionError(
                f"'{event_column}' has no events after coding ({coding}); every row is censored. "
                "Check the event coding."
            )

        t_all = work["t"].to_numpy(dtype=np.float64)
        e_all = work["e"].to_numpy(dtype=bool)
        km_all = _km(t_all, e_all)
        tmax = float(t_all.max())
        at = np.array(_MILESTONES) * tmax
        words = _event_words(event_column)
        unit = _time_unit(duration_column)
        n = len(work)
        censored_pct = 1.0 - n_events / n

        result: dict[str, Any] = {
            "duration_column": duration_column, "event_column": event_column,
            "group_column": None, "covariate_columns": covs, "event_coding": coding,
            "time_unit": unit, "event_words": list(words), "n_rows": n, "n_events": n_events,
            "n_censored": n - n_events, "censored_share": round(censored_pct, 4),
            "max_follow_up": round(tmax, 4),
            "median_survival": _round_median(_median(km_all)),
            "survival_at_end": round(float(km_all["surv"][-1]), 4),
            "milestones": [
                {k: round(v, 4) for k, v in m.items()} for m in _milestones(km_all, at)
            ],
            "groups": [], "log_rank": None, "cox": None,
        }

        curves: list[tuple[str, dict[str, Any]]] = [("All", km_all)]
        caveats: list[str] = []
        levels: list[str] = []
        ref: str | None = None
        group_kms: list[dict[str, Any]] = []
        if group_column:
            counts = work["g"].value_counts()
            eligible = [str(k) for k, v in counts.items() if int(v) >= _MIN_GROUP_N]
            levels = eligible[:_MAX_GROUPS]
            extra = len(counts) - len(levels)
            if extra > 0:
                notes.append(
                    f"Only the {len(levels)} largest '{group_column}' groups are compared; "
                    f"{extra} smaller group(s) were ignored."
                )
            if len(levels) < 2:
                notes.append(f"'{group_column}' has fewer than 2 groups with {_MIN_GROUP_N}+ rows; no group comparison.")
                levels = []
            else:
                result["group_column"] = group_column
                ref = levels[0]
                work = work[work["g"].isin(levels)]
                covdf = covdf.loc[work.index]
                curves = []
                for lvl in levels:
                    sub = work[work["g"] == lvl]
                    km = _km(sub["t"].to_numpy(dtype=np.float64), sub["e"].to_numpy(dtype=bool))
                    group_kms.append(km)
                    curves.append((lvl, km))
                    result["groups"].append({
                        "group": lvl, "n": km["n"], "events": km["events"],
                        "median_survival": _round_median(_median(km)),
                        "survival_at_end": round(float(km["surv"][-1]), 4),
                        "milestones": [
                            {k: round(v, 4) for k, v in m.items()} for m in _milestones(km, at)
                        ],
                    })
                lr = _log_rank([
                    (work.loc[work["g"] == lvl, "t"].to_numpy(dtype=np.float64),
                     work.loc[work["g"] == lvl, "e"].to_numpy(dtype=bool))
                    for lvl in levels
                ])
                if lr is not None:
                    result["log_rank"] = {k: round(v, 6) for k, v in lr.items()}
                few = [g["group"] for g in result["groups"] if g["events"] < _MIN_EVENTS_PER_GROUP]
                if few:
                    caveats.append(
                        f"Fewer than {_MIN_EVENTS_PER_GROUP} events in {', '.join(str(f) for f in few[:4])}: "
                        "their curves and hazard ratios are unreliable."
                    )
        if len(group_kms) >= 2:
            grid = np.unique(np.concatenate([k["times"] for k in group_kms]))
            if grid.size > 2000:
                grid = grid[np.linspace(0, grid.size - 1, 2000).astype(int)]
            result["curves_cross"] = _crossing(group_kms, grid)
            if result["curves_cross"]:
                caveats.append(
                    "The group curves cross, so a single hazard ratio oversimplifies: one group "
                    "leads early and the other later."
                )

        if levels or covs:
            cox, cox_notes = _cox(work, covs, covdf, ref, levels)
            notes.extend(cox_notes)
            result["cox"] = cox
            if cox is not None:
                for term in cox["hazard_ratios"]:
                    term["sentence"] = _hr_sentence(term, words, group_column or "")
                    for key in ("hr", "ci_lower", "ci_upper"):
                        term[key] = round(term[key], 4)
                    if term.get("sd") is not None:
                        term["sd"] = round(term["sd"], 4)
                    if term["p_value"] is not None:
                        term["p_value"] = round(term["p_value"], 6)

        if censored_pct > _HEAVY_CENSORING:
            caveats.append(
                f"{censored_pct * 100:.0f}% of rows are censored (the event has not happened yet), "
                "so survival estimates rest on few events and the tail is uncertain."
            )
        if n_events < _MIN_EVENTS_PER_GROUP:
            caveats.append(f"Only {n_events} events in total; estimates are very uncertain.")
        caveats.append(
            "Assumes censoring is unrelated to the outcome (rows drop out at random with respect "
            "to how likely they were to have the event)."
        )
        if levels:
            caveats.append("Group differences are associations; they do not show the group causes the difference.")

        title = f"Survival over {unit}" if not levels else f"Survival over {unit} by {group_column}"
        result["chart"] = _km_chart(
            curves, unit, result["group_column"], title,
            _median_sentence(result) + ".",
        )
        result["notes"] = notes
        result["caveats"] = caveats
        result["summary"] = self._summary(result)
        return result

    @staticmethod
    def _summary(r: dict[str, Any]) -> str:
        parts = [
            f"{_median_sentence(r)}, based on {r['n_rows']:,} rows "
            f"({r['n_events']:,} events, {r['censored_share'] * 100:.0f}% censored)."
        ]
        cox = r.get("cox") or {}
        group_terms = [t for t in cox.get("hazard_ratios", []) if t["kind"] == "group"]
        if group_terms:
            top = max(group_terms, key=lambda t: abs(math.log(t["hr"])))
            parts.append(f"Largest group gap: {top['sentence']}.")
        return " ".join(parts)

    def findings(
        self,
        output: dict[str, Any],
        profile: DatasetProfile | None,
        metadata: DatasetMetadata | None,
    ) -> list[Finding]:
        duration = str(output.get("duration_column"))
        event = str(output.get("event_column"))
        group = output.get("group_column")
        caveats = list(output.get("caveats") or [])
        events = int(output.get("n_events") or 0)
        n = int(output.get("n_rows") or 0)
        shrink = 0.6 if (output.get("censored_share") or 0) > _HEAVY_CENSORING else 1.0
        confidence = round(min(1.0, events / 60.0) * shrink, 3)
        med = output.get("median_survival") or {}
        results: list[Finding] = []
        results.append(Finding(
            finding_id=f"survival_median_{duration}_{event}",
            kind="survival",
            headline=_median_sentence(output) + ".",
            detail=(
                f"Kaplan-Meier estimate from {n:,} rows with {events:,} events "
                f"({output.get('event_coding')}). Milestones: "
                + "; ".join(
                    f"{m['survival'] * 100:.0f}% still surviving at {_fmt(m['time'])} {output.get('time_unit')}"
                    for m in output.get("milestones", [])
                )
                + "."
            ),
            evidence={
                "median": med.get("median"), "ci_lower": med.get("ci_lower"), "ci_upper": med.get("ci_upper"),
                "max_follow_up": output.get("max_follow_up"), "survival_at_end": output.get("survival_at_end"),
                "n": n, "events": events, "milestones": output.get("milestones"),
            },
            source_tool=self.name,
            measure=duration,
            dimension=None,
            effect=None,
            confidence=confidence,
            surprise=0.3,
            caveats=caveats,
            chart_hint=output.get("chart"),
        ))
        cox = output.get("cox") or {}
        p_log_rank = (output.get("log_rank") or {}).get("p_value")
        picked: list[dict[str, Any]] = []
        for term in sorted(cox.get("hazard_ratios", []), key=lambda t: -abs(math.log(t["hr"]))):
            if term["ci_lower"] <= 1.0 <= term["ci_upper"] or abs(term["hr"] - 1.0) < _MIN_HR_EFFECT:
                continue
            picked.append(term)
            if len(picked) >= _MAX_HR_FINDINGS:
                break
        for term in picked:
            is_group = term["kind"] == "group"
            label = str(term["label"] if is_group else f"{term['label']}_{term.get('level', 'sd')}")
            results.append(Finding(
                finding_id=f"survival_hr_{duration}_{label}".replace(" ", "_"),
                kind="survival",
                headline=_capitalise(term["sentence"]) + ".",
                detail=(
                    "Cox proportional hazards estimate"
                    + (f", adjusted for {', '.join(output.get('covariate_columns') or [])}" if output.get("covariate_columns") else "")
                    + f"; {cox.get('events', 0):,} events in {cox.get('n', 0):,} rows."
                ),
                evidence={
                    "hazard_ratio": term["hr"], "ci_lower": term["ci_lower"], "ci_upper": term["ci_upper"],
                    "term": term["label"], "n": cox.get("n"), "events": cox.get("events"),
                },
                source_tool=self.name,
                measure=duration,
                dimension=str(group) if is_group and group else (None if is_group else str(term["label"])),
                level=str(term["label"]) if is_group else term.get("level"),
                effect=round(float(term["hr"]) - 1.0, 4),
                effect_kind="lift",
                p_value=term.get("p_value"),
                confidence=confidence,
                surprise=min(1.0, abs(math.log(term["hr"]))),
                caveats=caveats,
                chart_hint=output.get("chart"),
            ))
        if not picked and group and isinstance(p_log_rank, float) and p_log_rank < 0.05 and len(output.get("groups", [])) >= 2:
            groups = [g for g in output["groups"] if g["milestones"]]
            mid = 1
            best = max(groups, key=lambda g: g["milestones"][mid]["survival"])
            worst = min(groups, key=lambda g: g["milestones"][mid]["survival"])
            t_mid = best["milestones"][mid]["time"]
            results.append(Finding(
                finding_id=f"survival_groups_{duration}_{group}",
                kind="survival",
                headline=(
                    f"By {_fmt(t_mid)} {output.get('time_unit')}, {best['milestones'][mid]['survival'] * 100:.0f}% "
                    f"of {group} {best['group']} still survive vs {worst['milestones'][mid]['survival'] * 100:.0f}% "
                    f"of {group} {worst['group']}."
                ),
                detail="The survival curves differ across groups over the whole follow-up.",
                evidence={
                    "time": t_mid, "best_group": best["group"], "worst_group": worst["group"],
                    "best_survival": best["milestones"][mid]["survival"],
                    "worst_survival": worst["milestones"][mid]["survival"], "n": n,
                },
                source_tool=self.name,
                measure=duration,
                dimension=str(group),
                effect=round(best["milestones"][mid]["survival"] - worst["milestones"][mid]["survival"], 4),
                effect_kind="pct",
                p_value=p_log_rank,
                confidence=confidence,
                surprise=0.4,
                caveats=caveats,
                chart_hint=output.get("chart"),
            ))
        return results

    def get_schema(self) -> dict[str, Any]:
        return {
            "file_path": {
                "type": "string",
                "description": "Path to the dataset.",
                "required": True,
            },
            "duration_column": {
                "type": "string",
                "description": "Numeric time-to-event / follow-up column (tenure, days, time). Auto-detected when omitted.",
                "required": False,
            },
            "event_column": {
                "type": "string",
                "description": (
                    "0/1, boolean or yes/no column: did the event happen (churned, died, failed)? "
                    "A column named 'censored' is read inverted (1 = censored). Auto-detected when omitted."
                ),
                "required": False,
            },
            "group_column": {
                "type": "string",
                "description": "Optional categorical column: one survival curve per group (top 6 by size) with a log-rank comparison and hazard ratios.",
                "required": False,
            },
            "covariate_columns": {
                "type": "array",
                "description": "Optional list (max 5) of columns to adjust the Cox hazard ratios for.",
                "required": False,
            },
        }


def _capitalise(text: str) -> str:
    return text[:1].upper() + text[1:]
