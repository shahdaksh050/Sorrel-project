"""
Equity Analysis Tool — Execution Layer.

Does an outcome differ between groups defined by a protected attribute
(gender, race, age group ...), and is the difference still there after
allowing for legitimate factors such as level, tenure and department?

  - numeric outcome (pay): the unadjusted gap in median and mean against the
    largest group, and an adjusted gap from an OLS regression of log(pay) on
    the group plus the control columns (HC3 robust standard errors), with a
    95% confidence interval and the share of the raw gap the controls explain;
  - binary outcome (hired / promoted / attrited / approved): each group's
    selection rate with a Wilson interval, the adverse-impact ratio against
    the highest-rate group (below 0.8 the "four-fifths rule" says the pattern
    warrants review) and a Fisher exact / chi-square test.

A statistical gap is never presented as proof of discrimination, and groups
under the minimum cell size (src.core.privacy) are never reported.
Deterministic, fitted per call, numpy only (no heavy imports at module load).
"""
from __future__ import annotations

import math
import re
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from src.core.findings import Finding
from src.core.privacy import is_small, min_cell_size, suppression_note
from src.tools.base import BaseTool, ToolExecutionError
from src.tools.data_processing import _read_df

if TYPE_CHECKING:
    from src.core.memory import DatasetMetadata
    from src.core.profiler import ColumnProfile, DatasetProfile

_MAX_ROWS = 50_000
_MIN_ROWS = 100
_MAX_GROUPS = 8
_MAX_CONTROLS = 6
_MAX_CONTROL_LEVELS = 8
_MAX_CONTROL_CARDINALITY = 60
_MAX_FINDINGS = 3
_Z95 = 1.959964
_MIN_GAP_FOR_FINDING_PCT = 2.0
_FOUR_FIFTHS = 0.8

_PROTECTED_TOKENS = frozenset({
    "gender", "sex", "race", "ethnicity", "ethnic", "agegroup", "disability",
    "nationality", "religion", "marital", "veteran",
})
_PAY_TOKENS = frozenset({
    "salary", "pay", "wage", "income", "compensation", "bonus", "earnings", "earning",
})
_BINARY_OUTCOME_TOKENS = frozenset({
    "hired", "promoted", "attrition", "terminated", "approved", "rejected",
    "selected", "churn", "left",
})
#: Outcomes where the affirmative value is the adverse one.
_ADVERSE_TOKENS = frozenset({
    "attrition", "terminated", "rejected", "churn", "left", "attrited", "fired",
    "resigned", "quit", "churned",
})
_POSITIVE_VALUES = frozenset({
    "1", "yes", "y", "true", "t", "hired", "promoted", "approved", "selected", "accepted",
    "left", "attrited", "terminated", "churned", "rejected", "resigned", "quit", "fired",
})
#: Legitimate-factor name tokens used to pick control columns when none are given.
_CONTROL_TOKENS = frozenset({
    "level", "grade", "band", "tenure", "department", "dept", "division", "role", "job",
    "title", "experience", "seniority", "location", "region", "education", "hours",
    "years", "function", "team",
})
_SLUG_RE = re.compile(r"[^A-Za-z0-9]+")
_CAMEL_RE = re.compile(r"([a-z0-9])([A-Z])")
_SPLIT_RE = re.compile(r"[^a-z0-9]+")


def _tokens(name: object) -> set[str]:
    return {t for t in _SPLIT_RE.split(_CAMEL_RE.sub(r"\1_\2", str(name)).lower()) if t}


def _slug(text: object) -> str:
    return _SLUG_RE.sub("_", str(text)).strip("_").lower() or "na"


def _is_protected(name: object) -> bool:
    tokens = _tokens(name)
    return bool(tokens & _PROTECTED_TOKENS) or {"age", "group"} <= tokens


def _is_pay(name: object) -> bool:
    return bool(_tokens(name) & _PAY_TOKENS)


def _is_binary_outcome_name(name: object) -> bool:
    return bool(_tokens(name) & _BINARY_OUTCOME_TOKENS)


def _label(name: object) -> str:
    return str(name).replace("_", " ").strip()


def _pct_text(value: float) -> str:
    magnitude = abs(value)
    return f"{magnitude:.1f}%" if magnitude < 2 else f"{magnitude:.0f}%"


def _join_words(items: list[str]) -> str:
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def _candidate_groups(profile: DatasetProfile) -> list[ColumnProfile]:
    cands = [
        c for c in profile.columns
        if _is_protected(c.name) and 2 <= c.nunique <= 30 and c.kind not in ("identifier", "constant")
    ]
    return sorted(cands, key=lambda c: (c.nunique, c.name))


def _candidate_pay(profile: DatasetProfile) -> list[ColumnProfile]:
    return [c for c in profile.columns if _is_pay(c.name) and c.kind == "numeric" and c.nunique > 2]


def _candidate_binary(profile: DatasetProfile) -> list[ColumnProfile]:
    return [
        c for c in profile.columns
        if _is_binary_outcome_name(c.name) and c.nunique == 2 and c.kind != "constant"
    ]


def _binary_outcome(series: pd.Series) -> pd.Series:
    """Series of float 0/1 (NaN kept) with 1 = the affirmative outcome."""
    values = series.dropna()
    if values.nunique() != 2:
        raise ToolExecutionError(
            f"'{series.name}' must have exactly 2 distinct values to be a yes/no outcome; "
            f"it has {values.nunique()}."
        )
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().sum() == values.size:
        positive = float(numeric.max())
        return (numeric == positive).astype(float).where(numeric.notna())
    lowered = series.astype("string").str.strip().str.lower()
    distinct = [str(v) for v in lowered.dropna().unique()]
    positives = [v for v in distinct if v in _POSITIVE_VALUES]
    if len(positives) != 1:
        raise ToolExecutionError(
            f"Cannot tell which value of '{series.name}' is the positive outcome "
            f"({', '.join(sorted(distinct))}). Recode it to 1/0 or yes/no."
        )
    return (lowered == positives[0]).astype(float).where(lowered.notna())


def _wilson(successes: float, n: float) -> tuple[float, float]:
    p = successes / n
    z2 = _Z95 ** 2
    denom = 1 + z2 / n
    centre = (p + z2 / (2 * n)) / denom
    half = _Z95 * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def _normal_p(z: float) -> float:
    return math.erfc(abs(z) / math.sqrt(2.0))


def _ols_hc3(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """OLS coefficients and their HC3 robust covariance; None when the design
    is rank deficient (collinear controls)."""
    xtx = x.T @ x
    if np.linalg.matrix_rank(xtx) < xtx.shape[0]:
        return None
    inv = np.linalg.inv(xtx)
    beta = inv @ (x.T @ y)
    resid = y - x @ beta
    leverage = ((x @ inv) * x).sum(axis=1)
    scaled = resid / np.clip(1.0 - leverage, 1e-6, None)
    meat = (x * (scaled ** 2)[:, None]).T @ x
    return beta, inv @ meat @ inv


def _control_columns(df: pd.DataFrame, profile: DatasetProfile, exclude: set[str], pay_outcome: bool) -> list[str]:
    picked: list[str] = []
    for c in profile.columns:
        if c.name in exclude or c.kind in ("identifier", "constant", "datetime"):
            continue
        if not _tokens(c.name) & _CONTROL_TOKENS or _is_protected(c.name):
            continue
        if pay_outcome and _is_pay(c.name):
            continue
        picked.append(c.name)
    return picked[:_MAX_CONTROLS]


def _design(
    work: pd.DataFrame, controls: list[str], notes: list[str]
) -> tuple[np.ndarray, list[str]]:
    """Control columns -> float matrix (numeric z-scored; categorical top
    levels dummy-coded against the most common level) and the columns used."""
    blocks: list[np.ndarray] = []
    used: list[str] = []
    for name in controls:
        series = work[name]
        numeric = pd.to_numeric(series, errors="coerce")
        if pd.api.types.is_numeric_dtype(series) or numeric.notna().mean() > 0.9:
            if numeric.isna().mean() > 0.3:
                notes.append(f"Control '{name}' was left out: more than 30% of its values are missing.")
                continue
            values = numeric.to_numpy(dtype=float)
            std = float(np.nanstd(values))
            if not math.isfinite(std) or std == 0:
                continue
            blocks.append(((values - np.nanmean(values)) / std)[:, None])
            used.append(name)
            continue
        text = series.astype("string").fillna("(missing)")
        if text.nunique() > _MAX_CONTROL_CARDINALITY:
            notes.append(f"Control '{name}' was left out: too many distinct values ({text.nunique()}).")
            continue
        counts = text.value_counts()
        levels = [str(lvl) for lvl in counts.index[1:_MAX_CONTROL_LEVELS + 1] if not is_small(counts[lvl])]
        if not levels:
            continue
        blocks.append(np.column_stack([(text == lvl).to_numpy(dtype=float) for lvl in levels]))
        used.append(name)
    if not blocks:
        return np.empty((len(work), 0)), []
    return np.hstack(blocks), used


class EquityAnalysisTool(BaseTool):
    """Group gaps in pay or selection rates, raw and adjusted for controls."""

    name = "equity_analysis"
    description = (
        "Check whether an outcome differs between groups of a protected attribute "
        "(gender, race, age group ...). For pay-like numbers: the raw gap in median "
        "and mean, the gap after allowing for control columns such as level, tenure "
        "and department (robust regression, 95% interval) and how much of the raw "
        "gap the controls explain. For yes/no outcomes (hired, promoted, left): "
        "each group's rate, the adverse-impact ratio against the highest-rate group "
        "(four-fifths rule) and a significance test. Small groups are suppressed. "
        "A gap is not proof of discrimination."
    )

    def applies_to(self, profile: DatasetProfile | None, metadata: DatasetMetadata | None) -> float:
        if profile is None or profile.row_count < _MIN_ROWS:
            return 0.0
        if not _candidate_groups(profile):
            return 0.0
        return 0.6 if (_candidate_pay(profile) or _candidate_binary(profile)) else 0.0

    def default_params(
        self, profile: DatasetProfile | None, metadata: DatasetMetadata | None
    ) -> dict[str, Any]:
        if profile is None:
            return {}
        groups, pay, binary = _candidate_groups(profile), _candidate_pay(profile), _candidate_binary(profile)
        outcome = (pay or binary)[0].name if (pay or binary) else None
        if not groups or outcome is None:
            return {}
        return {"group_column": groups[0].name, "outcome_column": outcome}

    def execute(  # type: ignore[override]
        self,
        file_path: str,
        outcome_column: str | None = None,
        group_column: str | None = None,
        control_columns: list[str] | None = None,
        outcome_kind: str = "auto",
        **_: Any,
    ) -> dict[str, Any]:
        from src.core.profiler import profile_dataframe

        df = _read_df(file_path)
        if df.empty:
            raise ToolExecutionError("Dataset has no rows.")
        if outcome_kind not in ("auto", "numeric", "binary"):
            raise ToolExecutionError("outcome_kind must be 'auto', 'numeric' or 'binary'.")
        for given in (outcome_column, group_column, *(control_columns or [])):
            if given and given not in df.columns:
                raise ToolExecutionError(f"Column '{given}' not found in dataset.")
        notes: list[str] = []
        sampled = len(df) > _MAX_ROWS
        if sampled:
            df = df.sample(n=_MAX_ROWS, random_state=0)
            notes.append(f"Analysed a random sample of {_MAX_ROWS:,} rows.")
        profile = profile_dataframe(df)

        if group_column is None:
            groups = _candidate_groups(profile)
            if not groups:
                raise ToolExecutionError(
                    "No protected-attribute column found (gender, race, age group ...). Pass group_column."
                )
            group_column = groups[0].name
        if outcome_column is None:
            pay, binary = _candidate_pay(profile), _candidate_binary(profile)
            if outcome_kind == "binary":
                pool = binary
            elif outcome_kind == "numeric":
                pool = pay
            else:
                pool = pay or binary
            pool = [c for c in pool if c.name != group_column]
            if not pool:
                raise ToolExecutionError("No pay-like or yes/no outcome column found. Pass outcome_column.")
            outcome_column = pool[0].name
        if outcome_column == group_column:
            raise ToolExecutionError("outcome_column and group_column must differ.")

        if outcome_kind == "auto":
            numeric_y = pd.to_numeric(df[outcome_column], errors="coerce")
            distinct = df[outcome_column].dropna().nunique()
            if distinct == 2:
                outcome_kind = "binary"
            elif numeric_y.notna().sum() > 0 and distinct > 2:
                outcome_kind = "numeric"
            else:
                raise ToolExecutionError(f"'{outcome_column}' has too few distinct values to compare.")

        controls = [c for c in (control_columns or []) if c not in (group_column, outcome_column)][:_MAX_CONTROLS]
        if control_columns is None:
            controls = _control_columns(
                df, profile, {group_column, outcome_column}, outcome_kind == "numeric" and _is_pay(outcome_column)
            )
        elif len(control_columns) > _MAX_CONTROLS:
            notes.append(f"Only the first {_MAX_CONTROLS} control columns were used.")

        if outcome_kind == "binary":
            result = self._binary(df, outcome_column, group_column, notes)
        else:
            result = self._numeric(df, outcome_column, group_column, controls, notes)
        result.update({
            "outcome_column": outcome_column, "group_column": group_column,
            "outcome_kind": outcome_kind, "sampled": sampled,
        })
        return result

    # ------------------------------------------------------------------
    def _kept_groups(
        self, frame: pd.DataFrame, group_column: str, notes: list[str], key: str = "g"
    ) -> tuple[list[str], int]:
        """Groups big enough to report (largest first, at most `_MAX_GROUPS`)
        and how many were left out for being under the minimum size."""
        counts = frame[key].value_counts()
        big = [str(g) for g, n in counts.items() if not is_small(n)]
        suppressed = int(len(counts) - len(big))
        if len(big) > _MAX_GROUPS:
            notes.append(f"Only the {_MAX_GROUPS} largest groups of '{group_column}' were compared.")
            big = big[:_MAX_GROUPS]
        if len(big) < 2:
            raise ToolExecutionError(
                f"Need at least 2 groups of {min_cell_size()}+ people in '{group_column}'; "
                f"found {len(big)}."
            )
        return big, suppressed

    def _caveats(self, suppressed: int, notes: list[str], adjusted: bool) -> list[str]:
        out = [
            "A statistical gap is not proof of discrimination: it shows a difference between groups, "
            "not why it exists.",
            "Factors that were not measured (for example role scope, negotiation, performance or "
            "career breaks) can explain or hide a gap"
            + ("; controls can themselves reflect earlier unequal treatment." if adjusted else "."),
            suppression_note(suppressed)
            if suppressed
            else f"Groups with fewer than {min_cell_size()} people are never reported, to avoid "
                 "identifying individuals.",
        ]
        return out + notes

    def _numeric(
        self, df: pd.DataFrame, outcome: str, group: str, controls: list[str], notes: list[str]
    ) -> dict[str, Any]:
        cols = list(dict.fromkeys([group, outcome, *controls]))
        frame = df[cols].copy()
        frame["_eq_y"] = pd.to_numeric(frame[outcome], errors="coerce")
        frame["_eq_g"] = frame[group].astype("string")
        frame = frame[frame["_eq_y"].notna() & frame["_eq_g"].notna()]
        if frame.empty:
            raise ToolExecutionError(f"No rows have both '{outcome}' and '{group}'.")
        kept, suppressed = self._kept_groups(frame, group, notes, "_eq_g")
        frame = frame[frame["_eq_g"].isin(kept)]
        ref = kept[0]
        levels = list(kept)

        y_all = frame["_eq_y"].to_numpy(dtype=float)
        use_log = bool((y_all > 0).all())
        stats: dict[str, dict[str, float]] = {}
        for lvl in levels:
            vals = frame.loc[frame["_eq_g"] == lvl, "_eq_y"]
            stats[lvl] = {"n": int(vals.size), "median": float(vals.median()), "mean": float(vals.mean())}

        # Adjusted model: intercept + group dummies + controls.
        design, used = _design(frame, controls, notes)
        target = np.log(y_all) if use_log else y_all
        g_values = frame["_eq_g"].to_numpy(dtype=object)
        dummies = np.column_stack([(g_values == lvl).astype(float) for lvl in levels[1:]])
        x = np.hstack([np.ones((len(frame), 1)), dummies, design])
        valid = np.isfinite(x).all(axis=1) & np.isfinite(target)
        x, target, g_valid = x[valid], target[valid], g_values[valid]
        adjusted_controls = used
        fit = _ols_hc3(x, target) if len(target) > x.shape[1] + 10 else None
        if fit is None and used:
            notes.append("Controls were dropped because they were collinear or too numerous for the data; "
                         "the gap shown is unadjusted.")
            adjusted_controls = []
            x = x[:, : 1 + dummies.shape[1]]
            fit = _ols_hc3(x, target) if len(target) > x.shape[1] + 10 else None
        if fit is None:
            raise ToolExecutionError("Too few rows to estimate the gap between groups.")
        beta, cov = fit
        ref_mean_model = float(target[g_valid == ref].mean())
        ref_mean_raw_scale = stats[ref]["mean"]

        rows: list[dict[str, Any]] = []
        for i, lvl in enumerate(levels[1:], start=1):
            b, se = float(beta[i]), math.sqrt(max(float(cov[i, i]), 0.0))
            raw_b = float(target[g_valid == lvl].mean()) - ref_mean_model
            if use_log:
                adj, lo, hi = math.expm1(b), math.expm1(b - _Z95 * se), math.expm1(b + _Z95 * se)
            else:
                scale = ref_mean_raw_scale if ref_mean_raw_scale != 0 else float("nan")
                adj, lo, hi = b / scale, (b - _Z95 * se) / scale, (b + _Z95 * se) / scale
            explained = None
            if adjusted_controls and abs(raw_b) > (0.005 if use_log else 1e-9):
                explained = 1.0 - b / raw_b
            s, r = stats[lvl], stats[ref]
            rows.append({
                "group": lvl, "n": s["n"],
                "median": round(s["median"], 2), "mean": round(s["mean"], 2),
                "unadjusted_median_gap_pct": _pct_change(s["median"], r["median"]),
                "unadjusted_mean_gap_pct": _pct_change(s["mean"], r["mean"]),
                "adjusted_gap_pct": _r2(adj), "ci_lower_pct": _r2(lo), "ci_upper_pct": _r2(hi),
                "p_value": _normal_p(b / se) if se > 0 else None,
                "share_explained": None if explained is None else round(explained, 3),
            })
        rows = [r for r in rows if r["adjusted_gap_pct"] is not None]
        rows.sort(key=lambda r: abs(r["adjusted_gap_pct"]), reverse=True)
        ref_row = {"group": ref, "n": stats[ref]["n"], "median": round(stats[ref]["median"], 2),
                   "mean": round(stats[ref]["mean"], 2)}
        if not rows:
            raise ToolExecutionError(f"Could not compute a gap: the reference group's average {outcome} is zero.")

        caveats = self._caveats(suppressed, notes, bool(adjusted_controls))
        chart = self._chart_numeric(outcome, group, ref, rows, adjusted_controls)
        top = rows[0]
        summary = self._sentence_numeric(top, ref, outcome, adjusted_controls)
        return {
            "summary": summary,
            "reference_group": ref,
            "reference": ref_row,
            "groups": rows,
            "control_columns": adjusted_controls,
            "model_scale": "log" if use_log else "raw",
            "n_model": len(target),
            "suppressed_groups": suppressed,
            "caveats": caveats,
            "chart": chart,
        }

    @staticmethod
    def _sentence_numeric(row: dict[str, Any], ref: str, outcome: str, controls: list[str]) -> str:
        pay = _is_pay(outcome)
        raw = row["unadjusted_mean_gap_pct"]
        adj, lo, hi = row["adjusted_gap_pct"], row["ci_lower_pct"], row["ci_upper_pct"]
        less, more = ("less", "more") if pay else ("lower", "higher")

        def amount(v: float) -> str:
            return f"{_pct_text(v)} {less if v < 0 else more}"

        unclear = lo < 0 < hi
        if unclear:
            interval = f"{_pct_text(lo)} {less} to {_pct_text(hi)} {more}"
        else:
            interval = f"{_pct_text(min(abs(lo), abs(hi)))} to {_pct_text(max(abs(lo), abs(hi)))}"
        allowing = (
            f"after allowing for {_join_words([_label(c) for c in controls])}"
            if controls else "with no adjustment available"
        )
        lead = f"{row['group']} {'are paid' if pay else 'have ' + _label(outcome)} "
        if raw is None:
            first = f"{lead}{amount(adj)} on average than {ref}"
        else:
            first = f"{lead}{amount(raw)} on average than {ref}"
        if unclear:
            second = f"the gap is not clearly different from zero {allowing}"
            joiner = "but"
        else:
            second = f"{amount(adj)} {allowing}"
            joiner = "and"
        if raw is None:
            return f"{first} {allowing} (95% CI {interval})."
        return f"{first}, {joiner} {second} (95% CI {interval})."

    def _chart_numeric(
        self, outcome: str, group: str, ref: str, rows: list[dict[str, Any]], controls: list[str]
    ) -> dict[str, Any] | None:
        data = [{"group": ref, "gap": 0.0, "lower": 0.0, "upper": 0.0}] + [
            {"group": r["group"], "gap": r["adjusted_gap_pct"] / 100, "lower": r["ci_lower_pct"] / 100,
             "upper": r["ci_upper_pct"] / 100}
            for r in rows
        ]
        kind = "Adjusted" if controls else "Unadjusted"
        spec = {
            "type": "dot_ci", "data": data, "x": "group", "y": "gap", "y_lower": "lower", "y_upper": "upper",
            "title": f"{kind} {_label(outcome)} gap by {_label(group)}",
            "x_title": _label(group), "y_title": f"Gap vs {ref}", "y_format": "percent",
            "annotations": [{"y": 0.0, "label": f"{ref} (reference)"}],
            "caption": f"Difference in {_label(outcome)} relative to {ref}, with 95% confidence intervals; "
                       "a gap is not proof of discrimination.",
        }
        return _validated(spec)

    # ------------------------------------------------------------------
    def _binary(self, df: pd.DataFrame, outcome: str, group: str, notes: list[str]) -> dict[str, Any]:
        from scipy import stats as sps

        frame = pd.DataFrame({"y": _binary_outcome(df[outcome]), "g": df[group].astype("string")})
        frame = frame[frame["y"].notna() & frame["g"].notna()]
        if frame.empty:
            raise ToolExecutionError(f"No rows have both '{outcome}' and '{group}'.")
        kept, suppressed = self._kept_groups(frame, group, notes)
        frame = frame[frame["g"].isin(kept)]
        adverse = bool(_tokens(outcome) & _ADVERSE_TOKENS)
        agg = frame.groupby("g")["y"].agg(["sum", "size"])
        rows: list[dict[str, Any]] = []
        for lvl in kept:
            succ, n = float(agg.loc[lvl, "sum"]), int(agg.loc[lvl, "size"])
            lo, hi = _wilson(succ, n)
            rows.append({"group": lvl, "n": n, "selected": int(succ), "rate": succ / n,
                         "ci_lower": lo, "ci_upper": hi})
        for r in rows:
            r["favourable_rate"] = 1.0 - r["rate"] if adverse else r["rate"]
        ref_row = max(rows, key=lambda r: r["favourable_rate"])
        ref = ref_row["group"]
        fav_ref = ref_row["favourable_rate"]
        for r in rows:
            r["air"] = None if fav_ref <= 0 else r["favourable_rate"] / fav_ref
            r["flag_four_fifths"] = bool(r["air"] is not None and r["air"] < _FOUR_FIFTHS and r["group"] != ref)
            r["p_value"] = None
            if r["group"] != ref:
                table = [[r["selected"], r["n"] - r["selected"]],
                         [ref_row["selected"], ref_row["n"] - ref_row["selected"]]]
                r["p_value"] = float(sps.fisher_exact(table)[1])
        overall_p: float | None = None
        test = "fisher_exact"
        if len(rows) > 2:
            test = "chi_square"
            contingency = np.array([[r["selected"], r["n"] - r["selected"]] for r in rows], dtype=float)
            if (contingency.sum(axis=0) > 0).all():
                overall_p = float(sps.chi2_contingency(contingency)[1])
        else:
            overall_p = next((r["p_value"] for r in rows if r["group"] != ref), None)

        for r in rows:
            for key in ("rate", "ci_lower", "ci_upper", "favourable_rate"):
                r[key] = round(r[key], 4)
            r["air"] = None if r["air"] is None else round(r["air"], 3)
            if r["p_value"] is not None:
                r["p_value"] = round(r["p_value"], 6)
        others = sorted((r for r in rows if r["group"] != ref), key=lambda r: (r["air"] is None, r["air"] or 0.0))
        flagged = [r for r in others if r["flag_four_fifths"]]
        worst = others[0]
        if flagged:
            summary = (
                f"{len(flagged)} group(s) fall below four-fifths of {ref}'s {_label(outcome)} "
                f"({'favourable-outcome ' if adverse else ''}rate): lowest ratio {worst['air']:.2f} for "
                f"{worst['group']}, which warrants review."
            )
        else:
            summary = (
                f"No group's {_label(outcome)} rate falls below four-fifths of {ref}'s "
                f"(lowest ratio {worst['air']:.2f} for {worst['group']})."
                if worst["air"] is not None else f"Rates of {_label(outcome)} by {_label(group)} computed."
            )
        caveats = self._caveats(suppressed, notes, False)
        if adverse:
            caveats.append(
                f"'{outcome}' is an adverse outcome, so the ratio is computed on the favourable side "
                f"(not {_label(outcome)})."
            )
        return {
            "summary": summary,
            "reference_group": ref,
            "adverse_outcome": adverse,
            "groups": rows,
            "test": test,
            "overall_p_value": None if overall_p is None else round(overall_p, 6),
            "suppressed_groups": suppressed,
            "caveats": caveats,
            "chart": self._chart_binary(outcome, group, rows, ref_row, adverse),
        }

    def _chart_binary(
        self, outcome: str, group: str, rows: list[dict[str, Any]], ref_row: dict[str, Any], adverse: bool
    ) -> dict[str, Any] | None:
        data = [
            {"group": r["group"], "rate": r["rate"], "lower": r["ci_lower"], "upper": r["ci_upper"]}
            for r in sorted(rows, key=lambda r: r["rate"], reverse=True)
        ]
        spec: dict[str, Any] = {
            "type": "dot_ci", "data": data, "x": "group", "y": "rate", "y_lower": "lower", "y_upper": "upper",
            "title": f"{_label(outcome).capitalize()} rate by {_label(group)}",
            "x_title": _label(group), "y_title": f"{_label(outcome).capitalize()} rate", "y_format": "percent",
            "caption": "Each dot is a group's rate with a 95% confidence interval; a gap is not proof of discrimination.",
        }
        threshold = _FOUR_FIFTHS * ref_row["favourable_rate"]
        if adverse:
            threshold = 1.0 - threshold  # the line sits on the adverse side of the rate axis
        if 0.0 < threshold < 1.0:
            spec["annotations"] = [{"y": round(threshold, 4), "label": "Four-fifths threshold"}]
        return _validated(spec)

    # ------------------------------------------------------------------
    def findings(
        self,
        output: dict[str, Any],
        profile: DatasetProfile | None,
        metadata: DatasetMetadata | None,
    ) -> list[Finding]:
        outcome, group = str(output.get("outcome_column")), str(output.get("group_column"))
        ref = str(output.get("reference_group"))
        caveats = [str(c) for c in output.get("caveats") or []]
        chart = output.get("chart") if isinstance(output.get("chart"), dict) else None
        rows = list(output.get("groups") or [])
        results: list[Finding] = []
        if output.get("outcome_kind") == "binary":
            rows = [r for r in rows if r.get("group") != ref]
            picked = [r for r in rows if r.get("flag_four_fifths")][:_MAX_FINDINGS] or rows[:1]
            ref_rate = next((r["rate"] for r in output.get("groups", []) if r.get("group") == ref), None)
            adverse = bool(output.get("adverse_outcome"))
            for r in picked:
                air = r.get("air")
                if air is None or ref_rate is None:
                    continue
                flagged = bool(r.get("flag_four_fifths"))
                basis = "measured on the favourable outcome, " if adverse else ""
                verdict = (
                    "below the 0.8 four-fifths guideline, which warrants review"
                    if flagged else "not below the 0.8 four-fifths guideline"
                )
                headline = (
                    f"{r['group']} have a {_label(outcome)} rate of {r['rate'] * 100:.0f}% vs "
                    f"{ref_rate * 100:.0f}% for {ref} ({basis}ratio {air:.2f}, {verdict})."
                )
                results.append(Finding(
                    finding_id=f"equity_{_slug(outcome)}_{_slug(group)}_{_slug(r['group'])}",
                    kind="equity", headline=headline,
                    detail=f"Highest-rate group is {ref}; {output.get('test', 'test')} p={r.get('p_value')}.",
                    evidence={
                        "group": r["group"], "reference_group": ref, "rate": r["rate"], "reference_rate": ref_rate,
                        "adverse_impact_ratio": air, "n": r["n"], "selected": r["selected"],
                    },
                    source_tool=self.name, measure=outcome, dimension=group, level=str(r["group"]),
                    effect=round(air - 1.0, 4), effect_kind="lift", p_value=r.get("p_value"),
                    confidence=min(1.0, r["n"] / 300.0), surprise=0.7 if flagged else 0.3,
                    caveats=caveats, chart_hint=chart,
                ))
            return results

        controls = list(output.get("control_columns") or [])
        rows.sort(key=lambda r: abs(r["adjusted_gap_pct"]), reverse=True)
        picked = [
            r for r in rows
            if abs(r["unadjusted_mean_gap_pct"] or 0.0) >= _MIN_GAP_FOR_FINDING_PCT
            or abs(r["adjusted_gap_pct"]) >= _MIN_GAP_FOR_FINDING_PCT
        ][:_MAX_FINDINGS] or rows[:1]
        ref_n = (output.get("reference") or {}).get("n")
        for r in picked:
            headline = self._sentence_numeric(r, ref, outcome, controls)
            results.append(Finding(
                finding_id=f"equity_{_slug(outcome)}_{_slug(group)}_{_slug(r['group'])}",
                kind="equity", headline=headline,
                detail=(
                    f"Gap in median {outcome}: {r['unadjusted_median_gap_pct']}%; "
                    f"controls explain {round(r['share_explained'] * 100)}% of the raw gap."
                    if r.get("share_explained") is not None
                    else f"Gap in median {outcome}: {r['unadjusted_median_gap_pct']}%."
                ),
                evidence={
                    "group": r["group"], "reference_group": ref,
                    "unadjusted_mean_gap_pct": r["unadjusted_mean_gap_pct"],
                    "unadjusted_median_gap_pct": r["unadjusted_median_gap_pct"],
                    "adjusted_gap_pct": r["adjusted_gap_pct"], "ci_lower_pct": r["ci_lower_pct"],
                    "ci_upper_pct": r["ci_upper_pct"], "share_explained": r.get("share_explained"),
                    "controls": controls, "n": r["n"], "n_reference": ref_n,
                },
                source_tool=self.name, measure=outcome, dimension=group, level=str(r["group"]),
                effect=round(r["adjusted_gap_pct"] / 100.0, 4), effect_kind="pct", p_value=r.get("p_value"),
                confidence=min(1.0, min(r["n"], ref_n or r["n"]) / 300.0), surprise=0.6,
                caveats=caveats, chart_hint=chart,
            ))
        return results

    def get_schema(self) -> dict[str, Any]:
        return {
            "file_path": {"type": "string", "description": "Path to the dataset.", "required": True},
            "outcome_column": {
                "type": "string",
                "description": (
                    "Outcome to compare: a pay-like number (salary, wage, bonus) or a yes/no column "
                    "(hired, promoted, left). Auto-detected when omitted."
                ),
                "required": False,
            },
            "group_column": {
                "type": "string",
                "description": (
                    "Protected-attribute column defining the groups (gender, race, age group ...). "
                    "Auto-detected when omitted."
                ),
                "required": False,
            },
            "control_columns": {
                "type": "array",
                "description": (
                    "Up to 6 legitimate-factor columns (level, tenure, department ...) to allow for "
                    "when comparing pay. Auto-picked from column names when omitted."
                ),
                "required": False,
            },
            "outcome_kind": {
                "type": "string",
                "description": "'auto' (default), 'numeric' or 'binary'.",
                "required": False,
            },
        }


def _r2(value: float) -> float | None:
    """Fraction -> percent rounded to 2 decimals; None when not finite."""
    return round(value * 100, 2) if math.isfinite(value) else None


def _pct_change(value: float, reference: float) -> float | None:
    return round((value / reference - 1.0) * 100, 2) if reference else None


def _validated(spec: dict[str, Any]) -> dict[str, Any] | None:
    from src.core.chart_spec import validate_chart_spec

    clean, _reason = validate_chart_spec(spec)
    return clean
