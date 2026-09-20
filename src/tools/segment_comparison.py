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

A ratio only means something on a ratio scale. Ordinal scores and
non-additive measures that can be zero or negative (temperature, a
balance, a z-score) are compared by `difference` (level - rest) and
Cohen's d instead, with `effect_kind` saying which one a row carries.

A dimension is only ever compared against a *family* of tests — every level
of every (measure, dimension) pair considered — and the whole family is
corrected together with Benjamini-Hochberg (src.core.multiple_testing),
never one ad hoc pair at a time.
"""
from __future__ import annotations

import re
from itertools import zip_longest
from typing import TYPE_CHECKING, Any

import pandas as pd
from scipy import stats
from statsmodels.stats.proportion import proportion_confint, proportions_ztest

from src.core.findings import Finding
from src.core.multiple_testing import apply_benjamini_hochberg
from src.core.privacy import min_cell_size, redact_small_level, suppression_note
from src.core.profiler import profile_dataframe
from src.core.stats_utils import aggregate_to_entity, measure_aggregation, repeated_entity
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

#: A level needs at least `min_cell_size()` rows (src.core.privacy; default 5),
#: and the rest of the data needs at least that many too, before a test/CI is
#: attempted or anything is reported about it. Smaller levels are omitted from
#: every output and counted in a caveat instead.

_ALPHA = 0.05

#: A |lift| (relative deviation from baseline) below this is "close enough
#: to baseline to not be a story" — the triviality-suppression floor for
#: this tool specifically (T5).
_MIN_LIFT_FOR_FINDING = 0.05
#: The same floor for difference-scale comparisons: a "small" Cohen's d.
_MIN_D_FOR_FINDING = 0.2
_EFFECT_FLOORS = {"cohens_d": _MIN_D_FOR_FINDING, "lift": _MIN_LIFT_FOR_FINDING}

#: Simpson's-paradox check: a stratum counts only when both the level and
#: the rest have at least this many units in it, and an auto-picked
#: stratifier must be a small dimension (2-8 levels).
_MIN_STRATUM_N = 20
_MAX_STRATIFIER_CARD = 8

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
    measures = list(profile.measures()) + list(profile.columns_of_role("flag", "ordinal"))
    return sorted(measures, key=lambda c: (c.unit_hint != "currency", c.name))


def _pick_dimension_columns(profile: DatasetProfile) -> list[ColumnProfile]:
    dims = [
        c for c in profile.columns_of_role("dimension", "flag")
        if _MIN_DIM_CARD <= c.nunique <= _MAX_DIM_CARD
    ]
    return sorted(dims, key=lambda c: c.nunique)


def _is_rate_measure(df: pd.DataFrame, measure_column: str, profile: DatasetProfile) -> bool:
    # A "flag" role alone is not enough: a 1/2-coded column is an ordinal
    # measure, not a rate — the values must be {0, 1} (or booleans).
    vals =pd.to_numeric(df[measure_column], errors="coerce").dropna().unique()
    if len(vals) == 0:
        return False
    rounded = {round(float(v), 6) for v in vals}
    return rounded <= {0.0, 1.0}


def _column(profile: DatasetProfile, name: str) -> ColumnProfile | None:
    return next((c for c in profile.columns if c.name == name), None)


def _entity_agg(is_rate: bool, cp: ColumnProfile | None) -> str:
    """How rows combine into one value per entity: a flag becomes the
    entity's rate and a score its average; only a true measure may sum."""
    if is_rate or cp is None or cp.semantic_role != "measure":
        return "mean"
    return measure_aggregation(cp)


def _level_and_rest(
    work: pd.DataFrame, measure: str, dimension: str, level: Any, entity_col: str | None, agg: str
) -> tuple[pd.Series, pd.Series]:
    """Measure values inside `level` and in the rest of the data — one value
    per entity on each side when rows repeat per entity. An entity present
    on both sides contributes to both, so the two samples are then only
    approximately independent."""
    mask = work[dimension] == level
    if entity_col is None:
        return work.loc[mask, measure], work.loc[~mask, measure]
    return (
        aggregate_to_entity(work[mask], entity_col, measure, agg)[measure],
        aggregate_to_entity(work[~mask], entity_col, measure, agg)[measure],
    )


def _effect(level_vals: pd.Series, rest_vals: pd.Series, by_difference: bool) -> float | None:
    """Cohen's d (difference scale) or lift (ratio scale) of level vs rest;
    None when undefined (no variance, or a zero baseline)."""
    level_mean, rest_mean = float(level_vals.mean()), float(rest_vals.mean())
    if by_difference:
        n, n_rest = len(level_vals), len(rest_vals)
        pooled_var = (
            (n - 1) * float(level_vals.var(ddof=1)) + (n_rest - 1) * float(rest_vals.var(ddof=1))
        ) / (n + n_rest - 2)
        return (level_mean - rest_mean) / pooled_var ** 0.5 if pooled_var > 0 else None
    return level_mean / rest_mean - 1.0 if rest_mean != 0 else None


def _cramers_v(df: pd.DataFrame, a: str, b: str) -> float:
    table = pd.crosstab(df[a], df[b])
    if min(table.shape) < 2:
        return 0.0
    chi2 = float(stats.chi2_contingency(table)[0])
    return float((chi2 / (table.to_numpy().sum() * (min(table.shape) - 1))) ** 0.5)


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
        # Round-robin across measures before capping — measure-major order
        # would spend the whole budget on the first measure or two.
        per_measure = [[(m, d) for d in dims if m != d] for m in measures]
        pairs = [p for tier in zip_longest(*per_measure) for p in tier if p]
        return pairs[:_MAX_PAIRS]

    def _compare_one(
        self,
        df: pd.DataFrame,
        measure_column: str,
        dimension_column: str,
        profile: DatasetProfile,
        entity_col: str | None = None,
        suppressed: set[tuple[str, str]] | None = None,
    ) -> list[dict[str, Any]]:
        k = min_cell_size()
        is_rate = _is_rate_measure(df, measure_column, profile)
        cp = _column(profile, measure_column)
        unit_hint = cp.unit_hint if cp else None
        # Repeated rows per entity are not independent evidence: test one
        # value per entity instead (a flag becomes the entity's rate). Not
        # when the dimension IS the entity — each level is then one unit.
        entity = entity_col if entity_col not in (measure_column, dimension_column) else None
        agg = _entity_agg(is_rate, cp)

        work = df[[dimension_column, measure_column] + ([entity] if entity else [])].copy()
        work[measure_column] = pd.to_numeric(work[measure_column], errors="coerce")
        work = work.dropna()
        if work.empty:
            return []

        # Ratios need a ratio scale: an ordinal score, or a non-additive
        # measure that can be zero/negative, is compared by difference.
        by_difference = not is_rate and (
            (cp is not None and cp.semantic_role == "ordinal")
            or (measure_aggregation(cp) == "mean" and float(work[measure_column].min()) <= 0)
        )

        # Kept for reference only (evidence["overall_mean"]) — the tested
        # comparison, and therefore ratio/lift/headline, is level vs the
        # REST of the data (`rest_vals` below), not this overall figure.
        # Using the overall mean as the divisor mixes the segment into its
        # own baseline and drifts from what the p-value actually tests.
        overall_mean = float(
            aggregate_to_entity(work, entity, measure_column, agg)[measure_column].mean()
            if entity
            else work[measure_column].mean()
        )

        counts = work[dimension_column].value_counts()
        levels = counts.head(_MAX_LEVELS_TESTED).index.tolist()
        if suppressed is not None:
            # Levels beyond the tested head that are also under the minimum.
            suppressed.update(
                (dimension_column, str(lvl)) for lvl in counts[counts < k].index
            )

        out: list[dict[str, Any]] = []
        for level in levels:
            level_vals, rest_vals = _level_and_rest(
                work, measure_column, dimension_column, level, entity, agg
            )
            n, n_rest = len(level_vals), len(rest_vals)
            if n < k:
                if suppressed is not None:
                    suppressed.add((dimension_column, str(level)))
                continue
            if n_rest < k:
                continue

            baseline_mean = float(rest_vals.mean())
            level_mean = float(level_vals.mean())
            # None: no variance (difference scale) or a zero rest-of-data
            # baseline (ratio/lift undefined).
            maybe_effect = _effect(level_vals, rest_vals, by_difference)
            if maybe_effect is None:
                continue
            effect = maybe_effect
            ratio: float | None
            lift: float | None
            if by_difference:
                ratio, lift, effect_kind = None, None, "cohens_d"
            else:
                ratio, lift, effect_kind = level_mean / baseline_mean, effect, "lift"

            p_value: float | None
            ci_lo: float | None
            ci_hi: float | None

            # Per-entity rates are fractions, not 0/1 outcomes, so they take
            # the t-test/t-interval path below rather than the proportion test.
            if is_rate and entity is None:
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
                "ratio": round(ratio, 4) if ratio is not None else None,
                "lift": round(lift, 4) if lift is not None else None,
                "difference": round(level_mean - baseline_mean, 6),
                "effect": round(effect, 4),
                "effect_kind": effect_kind,
                "n": n,
                "n_rest": n_rest,
                "p_value": p_value,
                "ci_lower": round(ci_lo, 6) if ci_lo is not None else None,
                "ci_upper": round(ci_hi, 6) if ci_hi is not None else None,
                "unit_of_analysis": entity or "row",
            })
        return out

    def _pick_stratifier(
        self, df: pd.DataFrame, profile: DatasetProfile, c: dict[str, Any], entity_col: str | None
    ) -> str | None:
        """The small dimension most associated with the compared dimension
        (Cramér's V) — the likeliest confounder of a level-vs-rest effect."""
        excluded = {c["measure"], c["dimension"], entity_col}
        candidates = [
            col.name for col in _pick_dimension_columns(profile)
            if col.name not in excluded and col.nunique <= _MAX_STRATIFIER_CARD
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda s: _cramers_v(df, c["dimension"], s))

    def _stratify(
        self,
        df: pd.DataFrame,
        c: dict[str, Any],
        stratify_by: str,
        profile: DatasetProfile,
        entity_col: str | None,
    ) -> dict[str, Any] | None:
        """Simpson's-paradox check: the same level-vs-rest effect computed
        within each stratum of `stratify_by`. None when fewer than two
        strata have enough data on both sides to judge."""
        measure, dimension = c["measure"], c["dimension"]
        if stratify_by in (measure, dimension):
            return None
        entity = entity_col if entity_col not in (measure, dimension, stratify_by) else None
        agg = _entity_agg(c["is_rate"], _column(profile, measure))
        work = df[[dimension, measure, stratify_by] + ([entity] if entity else [])].copy()
        work[measure] = pd.to_numeric(work[measure], errors="coerce")
        work = work.dropna()
        # Comparison rows carry the level as a string.
        work[dimension] = work[dimension].astype(str)
        by_difference = c["effect_kind"] == "cohens_d"

        per_stratum: list[dict[str, Any]] = []
        for stratum, cell in work.groupby(stratify_by, observed=True):
            level_vals, rest_vals = _level_and_rest(cell, measure, dimension, c["level"], entity, agg)
            n, n_rest = len(level_vals), len(rest_vals)
            if n < max(_MIN_STRATUM_N, min_cell_size()) or n_rest < max(_MIN_STRATUM_N, min_cell_size()):
                continue
            effect = _effect(level_vals, rest_vals, by_difference)
            if effect is not None:
                per_stratum.append(
                    {"stratum": str(stratum), "effect": round(effect, 4), "n": n, "n_rest": n_rest}
                )
        if len(per_stratum) < 2:
            return None

        # Weighted by effective sample size n*n_rest/(n+n_rest): a stratum
        # with a tiny level or a tiny rest carries little information.
        weights = [s["n"] * s["n_rest"] / (s["n"] + s["n_rest"]) for s in per_stratum]
        pooled = sum(w * s["effect"] for w, s in zip(weights, per_stratum, strict=True)) / sum(weights)
        consistent = all((s["effect"] > 0) == (c["effect"] > 0) for s in per_stratum)
        vanishes = abs(pooled) < _EFFECT_FLOORS[c["effect_kind"]]
        result: dict[str, Any] = {
            "stratify_by": stratify_by,
            "per_stratum": per_stratum,
            "consistent": consistent,
            "pooled_effect": round(pooled, 4),
        }
        if not consistent or vanishes:
            result["caveat"] = (
                f"Effect does not hold within {stratify_by} strata "
                f"({'direction reverses' if not consistent else 'it shrinks to negligible'}) "
                "— possible confounding."
            )
        return result

    def execute(  # type: ignore[override]
        self,
        file_path: str,
        measure_column: str | None = None,
        dimension_column: str | None = None,
        stratify_by: str | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        df = _read_df(file_path)
        if df.empty:
            raise ToolExecutionError("Dataset has no rows.")
        if stratify_by and (stratify_by not in df.columns or df[stratify_by].nunique(dropna=True) < 2):
            raise ToolExecutionError(
                f"stratify_by '{stratify_by}' must be a column with at least 2 distinct values."
            )

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

        entity_col = repeated_entity(profile, df)
        comparisons: list[dict[str, Any]] = []
        suppressed: set[tuple[str, str]] = set()
        for measure, dimension in pairs:
            comparisons.extend(self._compare_one(df, measure, dimension, profile, entity_col, suppressed))

        if not comparisons:
            raise ToolExecutionError(
                f"No segment had at least {min_cell_size()} rows on both sides of "
                "the comparison — nothing to test."
            )
        # Levels under the minimum are left out entirely (small-cell suppression).
        caveats = [suppression_note(len(suppressed))] if suppressed else []

        testable = [c for c in comparisons if c["p_value"] is not None]
        untestable = [c for c in comparisons if c["p_value"] is None]
        corrected = apply_benjamini_hochberg(testable, alpha=_ALPHA) if testable else []
        for c in untestable:
            c["p_adjusted"] = None
            c["significant_after_correction"] = False

        all_rows = corrected + untestable
        # Lift and Cohen's d are on different scales; rank each as a multiple
        # of its own triviality floor so neither kind wins by units alone.
        all_rows.sort(
            key=lambda c: abs(c["effect"]) / _EFFECT_FLOORS[c["effect_kind"]] * c["n"], reverse=True
        )

        # Simpson's-paradox check on the comparisons that can become
        # findings: all of them when the caller names a stratifier,
        # otherwise the top one against the small dimension most associated
        # with its segmenting dimension.
        reportable = [
            c for c in all_rows
            if c["significant_after_correction"]
            and abs(c["effect"]) >= _EFFECT_FLOORS[c["effect_kind"]]
        ]
        if stratify_by:
            to_stratify = [(c, stratify_by) for c in reportable]
        else:
            top = reportable[0] if reportable else None
            auto = self._pick_stratifier(df, profile, top, entity_col) if top else None
            to_stratify = [(top, auto)] if top and auto else []
        for c, stratum_col in to_stratify:
            stratified = self._stratify(df, c, stratum_col, profile, entity_col)
            if stratified:
                c["stratified"] = stratified

        # n counts entities when rows were aggregated per entity.
        row_noun = _row_noun(df[[entity_col]] if entity_col else df)
        if entity_col and row_noun == "rows":
            row_noun = "entities"
        best = all_rows[0]
        summary = self._headline(best, row_noun)
        if best.get("stratified", {}).get("caveat"):
            summary += f" {best['stratified']['caveat']}"

        return {
            "summary": summary,
            "pairs_tested": pairs,
            "n_comparisons": len(all_rows),
            "alpha": _ALPHA,
            "comparisons": all_rows,
            "row_noun": row_noun,
            "unit_of_analysis": entity_col or "row",
            "stratify_by": stratify_by or (to_stratify[0][1] if to_stratify else None),
            "suppressed_levels": len(suppressed),
            "caveats": caveats,
        }

    @staticmethod
    def _headline(c: dict[str, Any], row_noun: str) -> str:
        level_str = _format_value(c["level_value"], c["is_rate"], c["unit_hint"])
        baseline_str = _format_value(c["baseline_value"], c["is_rate"], c["unit_hint"])
        if c["is_rate"]:
            middle = f"show a {c['measure']} rate of {level_str}"
        else:
            middle = f"average {c['measure']} of {level_str}"
        if c.get("effect_kind") == "cohens_d":
            comparison = f"difference {c['difference']:+,.2f}, d={c['effect']:.2f}"
        else:
            comparison = f"{c['ratio']:.2f}x"
        return (
            f"{redact_small_level(c['level'], c.get('n'))} {c['dimension']} {row_noun} {middle} vs {baseline_str} "
            f"for everyone else ({comparison}, n={c['n']:,} vs {c['n_rest']:,})."
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
        privacy_caveats = [str(m) for m in output.get("caveats") or []]
        results: list[Finding] = []
        for i, c in enumerate(comparisons):
            if not c.get("significant_after_correction"):
                continue
            # Old cached output predates `effect`/`effect_kind`: it is all lift.
            effect_kind = c.get("effect_kind", "lift")
            effect = c.get("effect", c.get("lift"))
            floor = _MIN_D_FOR_FINDING if effect_kind == "cohens_d" else _MIN_LIFT_FOR_FINDING
            if effect is None or abs(effect) < floor:
                continue
            headline = self._headline({**c, "unit_hint": c.get("unit_hint")}, row_noun)
            stratum_caveat = (c.get("stratified") or {}).get("caveat")
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
                effect=effect,
                effect_kind=effect_kind,
                p_value=c["p_value"],
                p_adjusted=c["p_adjusted"],
                confidence=min(1.0, c["n"] / 200.0),
                surprise=min(1.0, abs(effect)),
                caveats=([stratum_caveat] if stratum_caveat else []) + privacy_caveats,
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
            "stratify_by": {
                "type": "string",
                "description": (
                    "Optional dimension to check each level-vs-rest effect within "
                    "(Simpson's-paradox / confounding check). Reported per comparison "
                    "as 'stratified'. When omitted, the top significant comparison is "
                    "checked against the most related 2-8 level dimension automatically."
                ),
                "required": False,
            },
        }
