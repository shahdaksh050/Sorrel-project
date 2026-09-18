"""
Segment Comparison Tool — Execution Layer (IMPROVEMENTS.md item 7.2 #1).

The one analysis missing everywhere in the codebase: "does this measure (or
target rate) differ by segment, and by how much?" Generic column statistics
answer "what is the mean of `amount`?"; this tool answers "is the West
region's `amount` higher than everyone else's, and is that difference real
or noise?" — the segment-vs-rest-of-data comparison with lift, a
significance test, sample size and a confidence interval that recovers
sentences like:

    "Month-to-month customers churn at 42% vs 27% for everyone else (1.6x, n=812 vs n=2,340)."
    "West-region orders run 25% above every other region (1.25x, n=340 vs n=980)."

The baseline in `ratio`/`lift`/the headline is always the REST of the data
(every row NOT in that level) — the same population the significance test
(`proportions_ztest`/`ttest_ind`) compares the level against — never the
dataset-wide mean, which would include the segment itself and drift from
what the p-value is actually testing. The dataset-wide mean is still kept
as `overall_mean` in each comparison's evidence for reference.

Two vocabularies:
  - `ratio`  = level_value / baseline_value — the "1.6x" multiplier a human
    would say out loud.
  - `lift`   = ratio - 1.0 — the same comparison re-centred on zero, so "no
    difference" is 0.0 and triviality/ranking thresholds behave sanely
    (a ratio is centred on 1.0, which a zero-floor threshold can't filter).

A dimension is only ever compared against a *family* of tests — every level
of every (measure, dimension) pair considered — and the whole family is
corrected together with Benjamini-Hochberg (src.core.multiple_testing),
never one ad hoc pair at a time.
"""
from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

import pandas as pd
from scipy import stats
from statsmodels.stats.proportion import proportion_confint, proportions_ztest

from src.core.findings import Finding
from src.core.multiple_testing import apply_benjamini_hochberg
from src.core.profiler import profile_dataframe
from src.tools.base import BaseTool, ToolExecutionError
from src.tools.data_processing import _read_df

if TYPE_CHECKING:
    from src.core.memory import DatasetMetadata
    from src.core.profiler import ColumnProfile, DatasetProfile

#: Dimension cardinality window for auto-selection — below 2 there is
#: nothing to compare, above 20 a level-by-level comparison is no longer a
#: "segment", it's every value of an ID column.
_MIN_DIM_CARD = 2
_MAX_DIM_CARD = 20

#: Even for an explicitly-passed dimension with more levels than that, only
#: the this many most frequent levels are tested — guards against a caller
#: pointing this at a near-identifier column.
_MAX_LEVELS_TESTED = 20

#: Ceiling on how many (measure, dimension) pairs get swept when neither
#: column is pinned by the caller — combinatorial guard (IMPROVEMENTS.md 7.2).
_MAX_PAIRS = 12

#: A level needs at least this many rows, and the rest of the data needs at
#: least this many too, before a test/CI is attempted at all.
_MIN_LEVEL_N = 5

_ALPHA = 0.05

#: A |lift| (relative deviation from baseline) below this is "close enough
#: to baseline to not be a story" — the triviality-suppression floor for
#: this tool specifically (T5).
_MIN_LIFT_FOR_FINDING = 0.05

_SLUG_RE = re.compile(r"[^A-Za-z0-9]+")

#: Column-name fragments that make the generated sentence say "customers"
#: instead of the generic "rows".
_ENTITY_NOUNS = {
    "customer": "customers", "client": "clients", "user": "users",
    "account": "accounts", "patient": "patients", "member": "members",
    "employee": "employees", "subscriber": "subscribers", "order": "orders",
    "transaction": "transactions", "product": "products",
}


def _slug(text: str) -> str:
    return _SLUG_RE.sub("_", str(text)).strip("_").lower() or "na"


def _row_noun(df: pd.DataFrame) -> str:
    lower_cols = " ".join(str(c).lower() for c in df.columns)
    for hint, noun in _ENTITY_NOUNS.items():
        if hint in lower_cols:
            return noun
    return "rows"


def _pick_measure_columns(profile: DatasetProfile) -> list[ColumnProfile]:
    """Measures, preferring currency, plus binary flags (a target like
    `churn`) since those are explicitly in scope as "rate" measures.

    Flags are included unconditionally alongside continuous measures, not
    only as a last-resort fallback when no continuous measure exists — a
    classification-target flag (e.g. `churn`) must never be crowded out of
    the family sweep just because the dataset also happens to have
    unrelated continuous features (`monthly_charges`, `tenure_months`...).
    `default_params()` below pins only the *first* measure this function
    returns, so a flag excluded here would never be tested against any
    dimension at all on a classification-target dataset — which is exactly
    the bug this fixes.
    """
    measures = list(profile.measures()) + list(profile.columns_of_role("flag"))
    return sorted(measures, key=lambda c: (c.unit_hint != "currency", c.name))


def _pick_dimension_columns(profile: DatasetProfile) -> list[ColumnProfile]:
    dims = [
        c for c in profile.columns_of_role("dimension", "flag")
        if _MIN_DIM_CARD <= c.nunique <= _MAX_DIM_CARD
    ]
    return sorted(dims, key=lambda c: c.nunique)


def _is_rate_measure(df: pd.DataFrame, measure_column: str, profile: DatasetProfile) -> bool:
    cp = next((c for c in profile.columns if c.name == measure_column), None)
    if cp is not None and cp.semantic_role == "flag":
        return True
    vals = pd.to_numeric(df[measure_column], errors="coerce").dropna().unique()
    if len(vals) == 0:
        return False
    rounded = {round(float(v), 6) for v in vals}
    return rounded <= {0.0, 1.0}


def _format_value(value: float, is_rate: bool, unit_hint: str | None) -> str:
    if is_rate:
        return f"{value * 100:.1f}%"
    if unit_hint == "currency":
        return f"${value:,.2f}"
    if unit_hint == "percent":
        return f"{value:.1f}%"
    return f"{value:,.2f}"


class SegmentComparisonTool(BaseTool):
    """Per-level mean/rate vs. baseline, with lift, CI, n and a significance
    test corrected across the whole family of comparisons tested."""

    name = "segment_comparison"
    description = (
        "Compare a measure (or a binary target treated as a rate) across the "
        "levels of a dimension against the dataset-wide baseline. Reports "
        "lift, a confidence interval, sample size and a significance test "
        "per level, Benjamini-Hochberg corrected across every level/pair "
        "tested. Auto-selects measure and dimension when omitted."
    )

    def applies_to(self, profile: DatasetProfile | None, metadata: DatasetMetadata | None) -> float:
        if profile is None:
            return 0.0
        if _pick_measure_columns(profile) and _pick_dimension_columns(profile):
            return 0.9
        return 0.0

    def default_params(
        self, profile: DatasetProfile | None, metadata: DatasetMetadata | None
    ) -> dict[str, Any]:
        # Deliberately pin ONLY the measure, not the dimension: `execute()`'s
        # `_build_pairs()` runs the full family sweep (up to `_MAX_PAIRS`
        # (measure, dimension) pairs, BH-corrected together) only when
        # `dimension_column` is left empty. Pinning both here — as an
        # earlier version of this method did — collapses every no-LLM run to
        # a single arbitrary pair and is exactly how a planted effect on a
        # dimension that isn't "first" (e.g. `region` when `product_category`
        # sorts first) gets silently missed. See IMPROVEMENTS.md 7.6's same
        # warning about `default_params` over-specifying and pre-empting the
        # family/agenda mechanism it exists to feed.
        if profile is None:
            return {}
        measures = _pick_measure_columns(profile)
        dims = _pick_dimension_columns(profile)
        if not measures or not dims:
            return {}
        # When the dataset has a known classification/regression target,
        # that target is the one measure a user actually cares about seeing
        # broken out by segment ("does churn differ by contract type?").
        # `measures[0]` prefers a currency-flavoured continuous column
        # (e.g. `monthly_charges`) purely on unit_hint, which would pin the
        # sweep to an unrelated feature and never test the target itself —
        # since only ONE measure gets pinned here (see the comment above),
        # that silently drops every (target, dimension) pair from the
        # no-LLM plan on every classification-target dataset.
        target = getattr(metadata, "target_column", None)
        target_measure = next((m for m in measures if m.name == target), None)
        chosen = target_measure or measures[0]
        return {"measure_column": chosen.name}

    def _build_pairs(
        self,
        profile: DatasetProfile,
        measure_column: str | None,
        dimension_column: str | None,
    ) -> list[tuple[str, str]]:
        if measure_column and dimension_column:
            return [(measure_column, dimension_column)]
        measures = [measure_column] if measure_column else [c.name for c in _pick_measure_columns(profile)]
        dims = [dimension_column] if dimension_column else [c.name for c in _pick_dimension_columns(profile)]
        if not measures or not dims:
            return []
        # A column can legitimately appear in both pools now that flags
        # (e.g. `churn`) are always eligible measures — but a column can
        # never be meaningfully compared against itself (every "level"
        # trivially has a mean/rate equal to itself, baseline included).
        pairs = [(m, d) for m in measures for d in dims if m != d]
        return pairs[:_MAX_PAIRS]

    def _compare_one(
        self, df: pd.DataFrame, measure_column: str, dimension_column: str, profile: DatasetProfile
    ) -> list[dict[str, Any]]:
        is_rate = _is_rate_measure(df, measure_column, profile)
        cp = next((c for c in profile.columns if c.name == measure_column), None)
        unit_hint = cp.unit_hint if cp else None

        work = df[[dimension_column, measure_column]].copy()
        work[measure_column] = pd.to_numeric(work[measure_column], errors="coerce")
        work = work.dropna(subset=[dimension_column, measure_column])
        if work.empty:
            return []

        # Kept for reference only (evidence["overall_mean"]) — the tested
        # comparison, and therefore ratio/lift/headline, is level vs the
        # REST of the data (`rest_vals` below), not this overall figure.
        # Using the overall mean as the divisor mixes the segment into its
        # own baseline and drifts from what the p-value actually tests.
        overall_mean = float(work[measure_column].mean())

        counts = work[dimension_column].value_counts()
        levels = counts.head(_MAX_LEVELS_TESTED).index.tolist()

        out: list[dict[str, Any]] = []
        for level in levels:
            mask = work[dimension_column] == level
            level_vals = work.loc[mask, measure_column]
            rest_vals = work.loc[~mask, measure_column]
            n, n_rest = len(level_vals), len(rest_vals)
            if n < _MIN_LEVEL_N or n_rest < _MIN_LEVEL_N:
                continue

            baseline_mean = float(rest_vals.mean())
            if baseline_mean == 0:
                continue  # ratio/lift undefined vs a zero rest-of-data baseline

            level_mean = float(level_vals.mean())
            ratio = level_mean / baseline_mean
            lift = ratio - 1.0

            p_value: float | None
            ci_lo: float | None
            ci_hi: float | None

            if is_rate:
                succ_level = int(level_vals.sum())
                succ_rest = int(rest_vals.sum())
                try:
                    _stat, p_value = proportions_ztest(
                        [succ_level, succ_rest], [n, n_rest]
                    )
                    p_value = float(p_value)
                except (ValueError, ZeroDivisionError):
                    p_value = None
                try:
                    ci_lo, ci_hi = proportion_confint(succ_level, n, alpha=_ALPHA, method="wilson")
                except (ValueError, ZeroDivisionError):
                    ci_lo, ci_hi = None, None
            else:
                p_value = None
                if level_vals.nunique() >= 2 and rest_vals.nunique() >= 2:
                    try:
                        _stat, p_value = stats.ttest_ind(level_vals, rest_vals, equal_var=False)
                        p_value = float(p_value)
                    except (ValueError, ZeroDivisionError):
                        p_value = None
                ci_lo, ci_hi = None, None
                if n >= 2:
                    sd = float(level_vals.std(ddof=1))
                    if sd > 0:
                        sem = sd / (n ** 0.5)
                        t_crit = float(stats.t.ppf(1 - _ALPHA / 2, df=n - 1))
                        ci_lo, ci_hi = level_mean - t_crit * sem, level_mean + t_crit * sem
                    else:
                        ci_lo, ci_hi = level_mean, level_mean

            out.append({
                "measure": measure_column,
                "dimension": dimension_column,
                "level": str(level),
                "is_rate": is_rate,
                "unit_hint": unit_hint,
                "level_value": round(level_mean, 6),
                "baseline_value": round(baseline_mean, 6),
                "overall_mean": round(overall_mean, 6),
                "ratio": round(ratio, 4),
                "lift": round(lift, 4),
                "n": n,
                "n_rest": n_rest,
                "p_value": p_value,
                "ci_lower": round(ci_lo, 6) if ci_lo is not None else None,
                "ci_upper": round(ci_hi, 6) if ci_hi is not None else None,
            })
        return out

    def execute(  # type: ignore[override]
        self,
        file_path: str,
        measure_column: str | None = None,
        dimension_column: str | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        df = _read_df(file_path)
        if df.empty:
            raise ToolExecutionError("Dataset has no rows.")

        if measure_column and measure_column not in df.columns:
            raise ToolExecutionError(f"Column '{measure_column}' not found in dataset.")
        if dimension_column and dimension_column not in df.columns:
            raise ToolExecutionError(f"Column '{dimension_column}' not found in dataset.")
        if measure_column:
            numeric = pd.to_numeric(df[measure_column], errors="coerce")
            if numeric.notna().sum() == 0:
                raise ToolExecutionError(
                    f"'{measure_column}' is not numeric and cannot be used as a measure."
                )
        if dimension_column and df[dimension_column].nunique(dropna=True) < 2:
            raise ToolExecutionError(
                f"'{dimension_column}' has fewer than 2 distinct values — nothing to compare."
            )

        profile = profile_dataframe(df)
        pairs = self._build_pairs(profile, measure_column, dimension_column)
        if not pairs:
            raise ToolExecutionError(
                "No usable (measure, dimension) pair: need at least one numeric "
                "measure (or binary rate) and one categorical/flag dimension "
                f"with {_MIN_DIM_CARD}-{_MAX_DIM_CARD} levels. Pass measure_column "
                "and/or dimension_column explicitly."
            )

        comparisons: list[dict[str, Any]] = []
        for measure, dimension in pairs:
            comparisons.extend(self._compare_one(df, measure, dimension, profile))

        if not comparisons:
            raise ToolExecutionError(
                f"No segment had at least {_MIN_LEVEL_N} rows on both sides of "
                "the comparison — nothing to test."
            )

        testable = [c for c in comparisons if c["p_value"] is not None]
        untestable = [c for c in comparisons if c["p_value"] is None]
        corrected = apply_benjamini_hochberg(testable, alpha=_ALPHA) if testable else []
        for c in untestable:
            c["p_adjusted"] = None
            c["significant_after_correction"] = False

        all_rows = corrected + untestable
        all_rows.sort(key=lambda c: abs(c["lift"]) * c["n"], reverse=True)

        row_noun = _row_noun(df)
        best = all_rows[0]
        summary = self._headline(best, row_noun)

        return {
            "summary": summary,
            "pairs_tested": pairs,
            "n_comparisons": len(all_rows),
            "alpha": _ALPHA,
            "comparisons": all_rows,
            "row_noun": row_noun,
        }

    @staticmethod
    def _headline(c: dict[str, Any], row_noun: str) -> str:
        level_str = _format_value(c["level_value"], c["is_rate"], c["unit_hint"])
        baseline_str = _format_value(c["baseline_value"], c["is_rate"], c["unit_hint"])
        if c["is_rate"]:
            middle = f"show a {c['measure']} rate of {level_str}"
        else:
            middle = f"average {c['measure']} of {level_str}"
        return (
            f"{c['level']} {c['dimension']} {row_noun} {middle} vs {baseline_str} "
            f"for everyone else ({c['ratio']:.2f}x, n={c['n']:,} vs {c['n_rest']:,})."
        )

    def findings(
        self,
        output: dict[str, Any],
        profile: DatasetProfile | None,
        metadata: DatasetMetadata | None,
    ) -> list[Finding]:
        comparisons = output.get("comparisons", [])
        # Reuse execute()'s own entity-noun pick (persisted in the output so
        # findings() — which has no dataframe, only `output` — doesn't fall
        # back to a generic "rows" when the tool already knew to say
        # "customers"/"orders". Old cached output without the field still
        # degrades gracefully to "rows".
        row_noun = output.get("row_noun") or "rows"
        results: list[Finding] = []
        for i, c in enumerate(comparisons):
            if not c.get("significant_after_correction"):
                continue
            if c.get("lift") is None or abs(c["lift"]) < _MIN_LIFT_FOR_FINDING:
                continue
            headline = self._headline({**c, "unit_hint": c.get("unit_hint")}, row_noun)
            results.append(Finding(
                finding_id=f"segment_lift_{i}_{_slug(c['measure'])}_{_slug(c['dimension'])}_{_slug(c['level'])}",
                kind="segment_lift",
                headline=headline,
                detail=(
                    f"n={c['n']} in segment vs {c['n_rest']} elsewhere; "
                    f"p={c['p_value']:.4g}, p_adjusted={c['p_adjusted']:.4g}; "
                    f"95% CI [{c['ci_lower']}, {c['ci_upper']}]."
                    if c.get("ci_lower") is not None
                    else f"n={c['n']} in segment vs {c['n_rest']} elsewhere."
                ),
                evidence=c,
                source_tool=self.name,
                measure=c["measure"],
                dimension=c["dimension"],
                level=c["level"],
                effect=c["lift"],
                effect_kind="lift",
                p_value=c["p_value"],
                p_adjusted=c["p_adjusted"],
                confidence=min(1.0, c["n"] / 200.0),
                surprise=min(1.0, abs(c["lift"])),
            ))
        return results

    def get_schema(self) -> dict[str, Any]:
        return {
            "file_path": {
                "type": "string",
                "description": "Path to the dataset.",
                "required": True,
            },
            "measure_column": {
                "type": "string",
                "description": (
                    "Numeric measure to compare, or a binary column treated as a "
                    "rate. Auto-selected (preferring a currency measure, falling "
                    "back to a binary flag) when omitted."
                ),
                "required": False,
            },
            "dimension_column": {
                "type": "string",
                "description": (
                    "Categorical/flag column to segment by (2-20 levels). "
                    "Auto-selected when omitted."
                ),
                "required": False,
            },
        }
