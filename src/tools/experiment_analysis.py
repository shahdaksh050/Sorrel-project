"""
Experiment Analysis Tool — Execution Layer.

"Did the variant move the metric, by how much, and could this test have seen
a real effect at all?" — for an A/B-style split of rows into arms. Each
treatment arm is compared with a control arm:

  - a rate (0/1 outcome) uses Fisher's exact test when any expected cell is
    below 5, otherwise a two-proportion z-test; the absolute-difference CI is
    Newcombe's and the relative-lift CI is a score interval;
  - a numeric metric uses Welch's t-test, or Mann-Whitney when the sample is
    small and skewed; the mean-difference CI is Welch's and the relative-lift
    CI is a log-ratio delta interval.

Every comparison also reports the minimum detectable effect at 80% power
(the smallest lift the test could reliably have seen), so a "no significant
difference" is read as "inconclusive below X" rather than "no effect". Arm
sizes are checked against an equal split (sample-ratio mismatch), and the
p-values of all treatment-vs-control comparisons are corrected together with
Benjamini-Hochberg.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.proportion import confint_proportions_2indep, proportions_ztest

from src.core.findings import Finding
from src.core.multiple_testing import apply_benjamini_hochberg
from src.core.profiler import find_experiment_arm, profile_dataframe
from src.core.stats_utils import aggregate_to_entity, repeated_entity
from src.tools.base import BaseTool, ToolExecutionError
from src.tools.data_processing import _read_df

if TYPE_CHECKING:
    from src.core.memory import DatasetMetadata
    from src.core.profiler import DatasetProfile

_ALPHA = 0.05
_POWER = 0.8
_MIN_ARM_N = 10
_MAX_ARMS = 6
#: A sample-ratio mismatch this unlikely under an equal split means the
#: assignment mechanism, not the treatment, is producing the size gap.
_SRM_P = 0.001
_MIN_LIFT_FOR_FINDING = 0.02
_MAX_REPORTED_N_NEEDED = 100_000_000

_TRUE_WORDS = ("yes", "true", "y", "1", "converted", "success")
_FALSE_WORDS = ("no", "false", "n", "0", "not converted", "failure")
_BINARY_MAP = {**dict.fromkeys(_TRUE_WORDS, 1.0), **dict.fromkeys(_FALSE_WORDS, 0.0)}
_CONTROL_NAMES = ("control", "ctrl", "baseline", "a", "0", "false", "no", "off", "existing", "holdout")


def _coerce_metric(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series) or pd.api.types.is_numeric_dtype(series):
        return series.astype("float64")
    text = series.astype(str).str.strip().str.lower()
    mapped = text.map(_BINARY_MAP)
    present = series.notna()
    if present.any() and mapped[present].notna().all():
        return mapped
    return pd.to_numeric(series, errors="coerce")


def _pick_outcome(profile: DatasetProfile, metadata: DatasetMetadata | None, arm: str) -> str | None:
    target = getattr(metadata, "target_column", None)
    names = {c.name for c in profile.columns}
    if target and target in names and target != arm:
        return str(target)
    flags = [c for c in profile.columns_of_role("flag") if c.name != arm]
    if flags:
        return flags[0].name
    measures = [c for c in profile.measures() if c.name != arm]
    measures.sort(key=lambda c: (c.unit_hint != "currency", c.name))
    return measures[0].name if measures else None


def _choose_control(levels: list[str], requested: str | None) -> str:
    if requested is not None:
        if str(requested) not in levels:
            raise ToolExecutionError(
                f"control_level '{requested}' is not one of the arms: {', '.join(levels)}."
            )
        return str(requested)
    for level in levels:
        if level.strip().lower() in _CONTROL_NAMES:
            return level
    return levels[0]


def _use_rank_test(v1: np.ndarray, v0: np.ndarray) -> bool:
    n_min = min(len(v1), len(v0))
    skews = [abs(float(stats.skew(v))) for v in (v1, v0) if float(np.std(v)) > 0]
    skew = max(skews, default=0.0)
    return (n_min < 30 and skew > 1.0) or (n_min < 100 and skew > 2.0)


def _z_sum(alpha: float, power: float, df: float | None = None) -> float:
    if df is None:
        return float(stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power))
    return float(stats.t.ppf(1 - alpha / 2, df) + stats.t.ppf(power, df))


def _mde_rate(p0: float, n1: int, n0: int) -> float:
    """Absolute rate increase over `p0` the test would detect with 80% power
    (arcsine effect size)."""
    h = _z_sum(_ALPHA, _POWER) * math.sqrt(1 / n1 + 1 / n0)
    angle = min(math.asin(math.sqrt(min(max(p0, 0.0), 1.0))) + h / 2, math.pi / 2)
    return math.sin(angle) ** 2 - p0


def _n_needed_rate(p1: float, p0: float) -> float | None:
    h = abs(2 * math.asin(math.sqrt(p1)) - 2 * math.asin(math.sqrt(p0)))
    if h == 0:
        return None
    return 2 * (_z_sum(_ALPHA, _POWER) / h) ** 2


def _rate_comparison(v1: np.ndarray, v0: np.ndarray) -> dict[str, Any]:
    n1, n0 = len(v1), len(v0)
    x1, x0 = round(float(v1.sum())), round(float(v0.sum()))
    p1, p0 = x1 / n1, x0 / n0
    table = np.array([[x1, n1 - x1], [x0, n0 - x0]])
    expected = np.outer(table.sum(axis=1), table.sum(axis=0)) / table.sum()
    p_value: float | None
    if expected.min() < 5:
        test = "fisher_exact"
        p_value = float(stats.fisher_exact(table)[1])
    else:
        test = "two_proportion_z"
        p_value = float(proportions_ztest([x1, x0], [n1, n0])[1])
    ci_abs: tuple[float | None, float | None] = (None, None)
    ci_rel: tuple[float | None, float | None] = (None, None)
    try:
        lo, hi = confint_proportions_2indep(x1, n1, x0, n0, method="newcomb", compare="diff", alpha=_ALPHA)
        ci_abs = (float(lo), float(hi))
    except (ValueError, ZeroDivisionError):
        pass
    if x1 > 0 and x0 > 0:
        try:
            lo, hi = confint_proportions_2indep(x1, n1, x0, n0, method="score", compare="ratio", alpha=_ALPHA)
            ci_rel = (float(lo) - 1.0, float(hi) - 1.0)
        except (ValueError, ZeroDivisionError):
            pass
    mde_abs = _mde_rate(p0, n1, n0)
    needed = _n_needed_rate(p1, p0) if p0 > 0 and p1 > 0 else None
    return {
        "value": p1, "control_value": p0, "abs_diff": p1 - p0,
        "ci_abs": ci_abs, "rel_lift": (p1 / p0 - 1.0) if p0 > 0 else None, "ci_rel": ci_rel,
        "cohens_d": None, "test": test, "p_value": None if math.isnan(p_value) else p_value,
        "mde_abs": mde_abs, "mde_rel": mde_abs / p0 if p0 > 0 else None, "n_needed": needed,
    }


def _mean_comparison(v1: np.ndarray, v0: np.ndarray) -> dict[str, Any]:
    n1, n0 = len(v1), len(v0)
    m1, m0 = float(v1.mean()), float(v0.mean())
    s1, s0 = float(v1.std(ddof=1)), float(v0.std(ddof=1))
    diff = m1 - m0
    var1, var0 = s1 ** 2 / n1, s0 ** 2 / n0
    se = math.sqrt(var1 + var0)
    p_value: float | None = None
    test = "welch_t"
    ci_abs: tuple[float | None, float | None] = (None, None)
    if se > 0:
        df_w = (var1 + var0) ** 2 / (var1 ** 2 / (n1 - 1) + var0 ** 2 / (n0 - 1))
        t_crit = float(stats.t.ppf(1 - _ALPHA / 2, df_w))
        ci_abs = (diff - t_crit * se, diff + t_crit * se)
        if _use_rank_test(v1, v0):
            test = "mann_whitney"
            p_value = float(stats.mannwhitneyu(v1, v0, alternative="two-sided").pvalue)
        else:
            p_value = float(stats.ttest_ind(v1, v0, equal_var=False).pvalue)
    pooled_sd = math.sqrt(((n1 - 1) * s1 ** 2 + (n0 - 1) * s0 ** 2) / (n1 + n0 - 2))
    d = diff / pooled_sd if pooled_sd > 0 else None
    ratio_scale = float(min(v1.min(), v0.min())) >= 0 and m0 > 0
    rel_lift = m1 / m0 - 1.0 if ratio_scale else None
    ci_rel: tuple[float | None, float | None] = (None, None)
    if ratio_scale and m1 > 0 and se > 0:
        se_log = math.sqrt(var1 / m1 ** 2 + var0 / m0 ** 2)
        z = float(stats.norm.ppf(1 - _ALPHA / 2))
        log_ratio = math.log(m1 / m0)
        ci_rel = (math.exp(log_ratio - z * se_log) - 1.0, math.exp(log_ratio + z * se_log) - 1.0)
    mde_abs = _z_sum(_ALPHA, _POWER, n1 + n0 - 2) * math.sqrt(1 / n1 + 1 / n0) * pooled_sd
    needed = 2 * (_z_sum(_ALPHA, _POWER) / abs(d)) ** 2 if d else None
    return {
        "value": m1, "control_value": m0, "abs_diff": diff, "ci_abs": ci_abs,
        "rel_lift": rel_lift, "ci_rel": ci_rel, "cohens_d": d, "test": test,
        "p_value": p_value, "mde_abs": mde_abs,
        "mde_rel": mde_abs / m0 if ratio_scale else None, "n_needed": needed,
    }


def _round(value: float | None, digits: int = 6) -> float | None:
    return None if value is None or not math.isfinite(value) else round(float(value), digits)


def _fmt(value: float, binary: bool) -> str:
    return f"{value * 100:.2f}%" if binary else f"{value:,.2f}"


def _fmt_diff(value: float, binary: bool) -> str:
    return f"{value * 100:+.2f} percentage points" if binary else f"{value:+,.2f}"


def _compare(v1: np.ndarray, v0: np.ndarray, binary: bool) -> dict[str, Any]:
    raw = _rate_comparison(v1, v0) if binary else _mean_comparison(v1, v0)
    lo, hi = raw["ci_abs"]
    rel_lo, rel_hi = raw["ci_rel"]
    rel = raw["rel_lift"]
    if rel is not None:
        effect, effect_kind = rel, "lift"
    elif raw["cohens_d"] is not None:
        effect, effect_kind = raw["cohens_d"], "cohens_d"
    else:
        effect, effect_kind = raw["abs_diff"], "pct"
    return {
        "kind": "rate" if binary else "mean",
        "n": len(v1), "n_control": len(v0),
        "value": _round(raw["value"]), "control_value": _round(raw["control_value"]),
        "abs_diff": _round(raw["abs_diff"]), "ci_lower": _round(lo), "ci_upper": _round(hi),
        "rel_lift": _round(rel, 4), "rel_lift_ci_lower": _round(rel_lo, 4), "rel_lift_ci_upper": _round(rel_hi, 4),
        "cohens_d": _round(raw["cohens_d"], 4),
        "effect": _round(effect, 4), "effect_kind": effect_kind,
        "test": raw["test"], "p_value": raw["p_value"],
        "mde_abs": _round(raw["mde_abs"]), "mde_rel": _round(raw["mde_rel"], 4),
        "n_per_arm_needed": (
            math.ceil(raw["n_needed"])
            if raw["n_needed"] is not None and raw["n_needed"] < _MAX_REPORTED_N_NEEDED else None
        ),
    }


def _plain_lift(c: dict[str, Any], metric: str, as_pct: bool) -> str:
    """One sentence a reader can act on, with the verdict spelled out."""
    pct = as_pct or c["kind"] == "rate"
    diff = _fmt_diff(c["abs_diff"], pct)
    if c["rel_lift"] is not None:
        word = "lifts" if c["rel_lift"] > 0 else "lowers"
        move = f"{word} {metric} by {abs(c['rel_lift']) * 100:.1f}% relative to control"
    elif c["effect_kind"] == "cohens_d":
        move = f"moves {metric} by d={c['effect']:+.2f} versus control"
    else:
        move = f"moves {metric} by {diff} versus control"
    interval = ""
    if c["ci_lower"] is not None:
        interval = f", 95% CI {_fmt_diff(c['ci_lower'], pct)} to {_fmt_diff(c['ci_upper'], pct)}"
    sentence = (
        f"{c['level']} {move} ({_fmt(c['value'], pct)} vs {_fmt(c['control_value'], pct)}; "
        f"{diff}{interval}; n={c['n']:,} vs {c['n_control']:,})"
    )
    if c.get("significant_after_correction"):
        return f"{sentence} - a real difference, not chance."
    return f"{sentence} - not distinguishable from no change.{_detectable(c, pct)}"


def _detectable(c: dict[str, Any], pct: bool) -> str:
    if c.get("mde_abs") is None:
        return ""
    text = f" This test could only reliably detect a change of {_fmt_diff(c['mde_abs'], pct).lstrip('+')} or more"
    if c.get("mde_rel") is not None:
        return f"{text} ({c['mde_rel'] * 100:.0f}% relative)."
    return f"{text}."


class ExperimentAnalysisTool(BaseTool):
    """Treatment-vs-control comparison of a rate or numeric metric across the
    arms of an experiment, with lift, CI, power and an assignment check."""

    name = "experiment_analysis"
    description = (
        "Analyse an A/B-style experiment: compare a metric between a control "
        "arm and each treatment arm. Picks the right test automatically "
        "(two-proportion z / Fisher for rates, Welch t / Mann-Whitney for "
        "means), reports absolute and relative lift with confidence "
        "intervals, the minimum detectable effect (is the test powered?), a "
        "sample-ratio-mismatch check on arm sizes, and a multiple-arm "
        "correction. Use when rows are split into groups such as "
        "control/variant."
    )

    def applies_to(self, profile: DatasetProfile | None, metadata: DatasetMetadata | None) -> float:
        if profile is None:
            return 0.0
        arm = find_experiment_arm(profile.columns, profile.row_count)
        if arm is None or _pick_outcome(profile, metadata, arm.name) is None:
            return 0.0
        return 0.95 if profile.archetype == "experiment" else 0.85

    def default_params(
        self, profile: DatasetProfile | None, metadata: DatasetMetadata | None
    ) -> dict[str, Any]:
        if profile is None:
            return {}
        arm = find_experiment_arm(profile.columns, profile.row_count)
        if arm is None:
            return {}
        metric = _pick_outcome(profile, metadata, arm.name)
        return {"group_column": arm.name, "metric_column": metric} if metric else {}

    def execute(  # type: ignore[override]
        self,
        file_path: str,
        group_column: str | None = None,
        metric_column: str | None = None,
        control_level: str | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        df = _read_df(file_path)
        if df.empty:
            raise ToolExecutionError("Dataset has no rows.")
        profile = profile_dataframe(df)

        if group_column is None:
            arm = find_experiment_arm(profile.columns, profile.row_count)
            if arm is None:
                raise ToolExecutionError(
                    "No experiment arm column found. Pass group_column (the control/variant column)."
                )
            group_column = arm.name
        if group_column not in df.columns:
            raise ToolExecutionError(f"Column '{group_column}' not found in dataset.")
        if metric_column is None:
            metric_column = _pick_outcome(profile, None, group_column)
            if metric_column is None:
                raise ToolExecutionError("No outcome metric found. Pass metric_column.")
        if metric_column not in df.columns:
            raise ToolExecutionError(f"Column '{metric_column}' not found in dataset.")
        if metric_column == group_column:
            raise ToolExecutionError("metric_column and group_column must differ.")

        metric = _coerce_metric(df[metric_column])
        if metric.notna().sum() == 0:
            raise ToolExecutionError(f"'{metric_column}' is not numeric and cannot be compared across arms.")
        is_rate_metric = bool(set(metric.dropna().unique()) <= {0.0, 1.0})

        entity = repeated_entity(profile, df)
        if entity in (group_column, metric_column):
            entity = None
        work = pd.DataFrame({"arm": df[group_column].astype(str), "y": metric})
        if entity:
            work["entity"] = df[entity]
        work = work[df[group_column].notna()].dropna()
        contaminated = 0
        if entity:
            contaminated = int((work.groupby("entity")["arm"].nunique() > 1).sum())
            work = aggregate_to_entity(work, "entity", "y", "mean", by="arm")
        if work.empty:
            raise ToolExecutionError("No rows have both an arm and a metric value.")

        counts = work["arm"].value_counts()
        kept = [lvl for lvl, n in counts.items() if n >= _MIN_ARM_N][:_MAX_ARMS]
        skipped = [str(lvl) for lvl in counts.index if lvl not in kept]
        if len(kept) < 2:
            raise ToolExecutionError(
                f"Need at least 2 arms with {_MIN_ARM_N}+ units each; '{group_column}' has "
                f"{len(counts)} level(s) and {len(kept)} large enough."
            )
        control = _choose_control([str(k) for k in kept], control_level)
        binary = bool(set(work["y"].unique()) <= {0.0, 1.0})
        arm_values = {lvl: work.loc[work["arm"] == lvl, "y"].to_numpy(dtype=float) for lvl in kept}

        comparisons: list[dict[str, Any]] = []
        for lvl in kept:
            if lvl == control:
                continue
            row = _compare(arm_values[lvl], arm_values[control], binary)
            row.update({"level": lvl, "control_level": control})
            comparisons.append(row)

        testable = [c for c in comparisons if c["p_value"] is not None]
        corrected = apply_benjamini_hochberg(testable, alpha=_ALPHA)
        for c in comparisons:
            if c["p_value"] is None:
                c["p_adjusted"], c["significant_after_correction"] = None, False
        comparisons = corrected + [c for c in comparisons if c["p_value"] is None]
        for c in comparisons:
            c["underpowered"] = (
                not c["significant_after_correction"] and c["mde_abs"] is not None
                and abs(c["abs_diff"] or 0.0) < c["mde_abs"]
            )
        comparisons.sort(key=lambda c: abs(c["effect"] or 0.0), reverse=True)

        sizes = np.array([len(arm_values[lvl]) for lvl in kept], dtype=float)
        srm_p = float(stats.chisquare(sizes).pvalue)
        omnibus = self._omnibus(arm_values, binary) if len(kept) >= 3 else None

        caveats: list[str] = []
        if srm_p < _SRM_P:
            caveats.append(
                f"Sample-ratio mismatch: arm sizes ({', '.join(f'{k}={len(arm_values[k]):,}' for k in kept)}) "
                f"differ from an equal split (p={srm_p:.2g}); if an equal split was intended, "
                "assignment is broken and the comparison may be biased."
            )
        if contaminated:
            caveats.append(f"{contaminated:,} {entity} value(s) appear in more than one arm.")
        if skipped:
            caveats.append(f"Arms with fewer than {_MIN_ARM_N} units were left out: {', '.join(skipped[:5])}.")

        best = comparisons[0]
        summary = _plain_lift(best, metric_column, is_rate_metric)
        if caveats and srm_p < _SRM_P:
            summary += f" Warning: {caveats[0]}"

        return {
            "summary": summary,
            "group_column": group_column,
            "metric_column": metric_column,
            "metric_kind": "rate" if binary else "mean",
            "as_percent": is_rate_metric,
            "control_level": control,
            "arms": [
                {"level": str(k), "n": len(arm_values[k]), "value": _round(float(arm_values[k].mean()))}
                for k in kept
            ],
            "comparisons": comparisons,
            "omnibus": omnibus,
            "srm": {"p_value": _round(srm_p, 6), "flagged": srm_p < _SRM_P, "expected": "equal split"},
            "alpha": _ALPHA,
            "power": _POWER,
            "unit_of_analysis": entity or "row",
            "caveats": caveats,
        }

    @staticmethod
    def _omnibus(arm_values: dict[str, np.ndarray], binary: bool) -> dict[str, Any] | None:
        groups = list(arm_values.values())
        try:
            if binary:
                table = np.array([[g.sum(), len(g) - g.sum()] for g in groups])
                if (table.sum(axis=0) == 0).any():
                    return None
                return {"test": "chi_square", "p_value": _round(float(stats.chi2_contingency(table)[1]), 6)}
            if any(float(np.std(g)) == 0 for g in groups):
                return None
            return {"test": "kruskal_wallis", "p_value": _round(float(stats.kruskal(*groups).pvalue), 6)}
        except ValueError:
            return None

    def findings(
        self,
        output: dict[str, Any],
        profile: DatasetProfile | None,
        metadata: DatasetMetadata | None,
    ) -> list[Finding]:
        metric = str(output.get("metric_column"))
        group = str(output.get("group_column"))
        as_pct = bool(output.get("as_percent"))
        out: list[Finding] = []
        for c in output.get("comparisons", []):
            effect = c.get("effect")
            if not c.get("significant_after_correction") or effect is None:
                continue
            if abs(effect) < (0.2 if c["effect_kind"] == "cohens_d" else _MIN_LIFT_FOR_FINDING):
                continue
            pct = as_pct or c["kind"] == "rate"
            out.append(Finding(
                finding_id=f"experiment_{metric}_{group}_{c['level']}".replace(" ", "_"),
                kind="test",
                headline=_plain_lift(c, metric, as_pct),
                detail=(
                    f"{c['test']}, p={c['p_value']:.4g}, adjusted p={c['p_adjusted']:.4g}."
                    f"{_detectable(c, pct).replace('could only reliably detect', 'had power to detect')}"
                ),
                evidence=c,
                source_tool=self.name,
                measure=metric,
                dimension=group,
                level=str(c["level"]),
                effect=effect,
                effect_kind=c["effect_kind"],
                p_value=c["p_value"],
                p_adjusted=c["p_adjusted"],
                confidence=min(1.0, min(c["n"], c["n_control"]) / 200.0),
                surprise=min(1.0, abs(effect)),
                caveats=list(output.get("caveats") or []),
            ))
        srm = output.get("srm") or {}
        if srm.get("flagged"):
            message = next((m for m in output.get("caveats", []) if m.startswith("Sample-ratio")), "")
            out.append(Finding(
                finding_id=f"experiment_srm_{group}",
                kind="method_fit",
                headline=message,
                evidence={"srm_p_value": srm.get("p_value"), "arms": output.get("arms")},
                source_tool=self.name,
                dimension=group,
                confidence=0.9,
            ))
        return out

    def get_schema(self) -> dict[str, Any]:
        return {
            "file_path": {
                "type": "string",
                "description": "Path to the dataset.",
                "required": True,
            },
            "group_column": {
                "type": "string",
                "description": (
                    "Column holding the experiment arm (control/variant). "
                    "Auto-detected from arm-like names when omitted."
                ),
                "required": False,
            },
            "metric_column": {
                "type": "string",
                "description": (
                    "Outcome to compare: a 0/1 or yes/no column (a rate) or a "
                    "numeric measure (a mean). Auto-selected when omitted."
                ),
                "required": False,
            },
            "control_level": {
                "type": "string",
                "description": (
                    "The arm every other arm is compared against. Defaults to a "
                    "level named control/baseline/A, else the first level."
                ),
                "required": False,
            },
        }
