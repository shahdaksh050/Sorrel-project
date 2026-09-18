"""
Statistical Analysis Tools — Execution Layer.

Stage 3: Statistical hypothesis test selection and execution.

Decision logic (auto-selection):
  - Categorical feature               → Chi-Square test (or Fisher's exact
                                         for a 2x2 table with low expected counts)
  - 2 groups, normal, equal variance  → Independent T-test (Cohen's d)
  - 2 groups, normal, unequal         → Welch's T-test (Hedges' g)
  - 2 groups, non-normal              → Mann-Whitney U (rank-biserial r)
  - 3+ groups, normal                 → One-Way ANOVA (eta-squared)
  - 3+ groups, non-normal             → Kruskal-Wallis (epsilon-squared)

Every branch reports an effect size and a `practical_significance` flag
distinct from `significant` — a p-value alone cannot tell a reader whether
a difference is real-but-tiny (the "large-n trap") or a genuine effect.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
import pandas as pd
from scipy import stats

from src.core.findings import Finding
from src.core.io import DatasetReadError, read_any
from src.core.multiple_testing import apply_benjamini_hochberg
from src.core.profiler import is_identifier_like, profile_dataframe
from src.tools.base import BaseTool, ToolExecutionError

if TYPE_CHECKING:
    from src.core.memory import DatasetMetadata
    from src.core.profiler import DatasetProfile

#: Effect-size magnitude below which a result is "not practically
#: significant" even when p < alpha — conventional small-effect cutoffs.
PRACTICAL_THRESHOLD_D = 0.2        # Cohen's d / Hedges' g
PRACTICAL_THRESHOLD_R = 0.1        # rank-biserial correlation
PRACTICAL_THRESHOLD_V = 0.1        # Cramér's V
PRACTICAL_THRESHOLD_ETA2 = 0.01    # eta-squared / epsilon-squared

#: A group smaller than this is underpowered for the tests this tool runs.
MIN_RELIABLE_GROUP_SIZE = 20

#: Above this total sample size, a significant-but-below-threshold effect
#: is flagged as the large-n trap: p<alpha with n=100,000 can correspond to
#: a difference too small to matter.
LARGE_N_TRAP_THRESHOLD = 10_000

#: Every random draw in this tool (Shapiro-Wilk sub-sampling) is seeded —
#: AGENTS.md tool contract #6: every tool must be deterministic.
_RNG_SEED = 42

#: Expected contingency-table cell count below which chi-square is
#: considered unreliable (the standard rule of thumb).
_MIN_EXPECTED_CELL_COUNT = 5

#: A grouping column with more distinct levels than this is treated as
#: identifier-like for hypothesis-testing purposes: too many groups to
#: interpret and, for the family-mode path (7.6), too expensive to test
#: against every measure. Reused both for an explicit group_column (the
#: original n_groups check) and to filter family-mode dimension candidates,
#: so both paths agree on what counts as "too many groups".
MAX_DIMENSION_CARDINALITY = 20

#: Maps each effect-size metric this tool computes to the practical-
#: significance threshold used for it elsewhere in this file (PRACTICAL_
#: THRESHOLD_*). Family mode (7.6) needs to judge "negligible effect" per
#: pairing from just the effect_kind string carried in family_results,
#: since each pairing can use a different test/effect metric.
_EFFECT_KIND_THRESHOLDS: dict[str, float] = {
    "cohens_d": PRACTICAL_THRESHOLD_D,
    "hedges_g": PRACTICAL_THRESHOLD_D,
    "rank_biserial": PRACTICAL_THRESHOLD_R,
    "cramers_v": PRACTICAL_THRESHOLD_V,
    "eta_squared": PRACTICAL_THRESHOLD_ETA2,
    "epsilon_squared": PRACTICAL_THRESHOLD_ETA2,
}

#: Findings below this p_adjusted are candidates for the finding bus (7.6
#: item 4) — matches the tool's own default alpha.
_FINDING_P_THRESHOLD = 0.05


def _read_df(file_path: str) -> pd.DataFrame:
    """Read a dataset via the unified reader (src.core.io.read_any)."""
    try:
        df, _report = read_any(file_path)
    except DatasetReadError as exc:
        raise ToolExecutionError(str(exc)) from exc
    return df


def _seeded_sample(values: np.ndarray, max_n: int = 5000) -> np.ndarray:
    """Random (not first-N) sub-sample for Shapiro-Wilk, which errors above
    ~5000 points. Taking the first N of sorted/exported data silently
    truncates the distribution's tail and can flip the normality verdict;
    a seeded random sample doesn't have that bias."""
    if len(values) <= max_n:
        return values
    rng = np.random.default_rng(_RNG_SEED)
    idx = rng.choice(len(values), size=max_n, replace=False)
    return values[idx]


def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Pooled-SD Cohen's d — for the equal-variance (Independent T-Test) case."""
    n1, n2 = len(a), len(b)
    v1, v2 = float(np.var(a, ddof=1)), float(np.var(b, ddof=1))
    pooled_sd = (((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2)) ** 0.5
    if pooled_sd == 0:
        return 0.0
    return float((float(np.mean(a)) - float(np.mean(b))) / pooled_sd)


def _hedges_g(a: np.ndarray, b: np.ndarray) -> float:
    """Effect size for the unequal-variance (Welch's T-Test) case: Cohen's d
    with the unpooled average-variance denominator (appropriate when
    variances differ), then bias-corrected for small samples."""
    n1, n2 = len(a), len(b)
    v1, v2 = float(np.var(a, ddof=1)), float(np.var(b, ddof=1))
    avg_sd = ((v1 + v2) / 2) ** 0.5
    if avg_sd == 0:
        return 0.0
    d = (float(np.mean(a)) - float(np.mean(b))) / avg_sd
    correction = 1 - 3 / (4 * (n1 + n2) - 9)
    return float(d * correction)


def _mean_diff_ci(a: np.ndarray, b: np.ndarray, equal_var: bool, alpha: float) -> tuple[float, float]:
    """(1-alpha) CI on the difference of means, Welch-Satterthwaite df when
    variances are unequal."""
    n1, n2 = len(a), len(b)
    diff = float(np.mean(a)) - float(np.mean(b))
    v1, v2 = float(np.var(a, ddof=1)), float(np.var(b, ddof=1))
    if equal_var:
        pooled_var = ((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2)
        se = (pooled_var * (1 / n1 + 1 / n2)) ** 0.5
        df = float(n1 + n2 - 2)
    else:
        se = (v1 / n1 + v2 / n2) ** 0.5
        denom = (v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1)
        df = (v1 / n1 + v2 / n2) ** 2 / denom if denom > 0 else float(n1 + n2 - 2)
    t_crit = float(stats.t.ppf(1 - alpha / 2, df))
    margin = t_crit * se
    return (diff - margin, diff + margin)


def _rank_biserial(u_stat: float, n1: int, n2: int) -> float:
    return 1.0 - (2.0 * u_stat) / (n1 * n2)


def _eta_squared(group_arrays: list[np.ndarray]) -> float:
    all_values = np.concatenate(group_arrays)
    grand_mean = float(np.mean(all_values))
    ss_total = float(np.sum((all_values - grand_mean) ** 2))
    if ss_total == 0:
        return 0.0
    ss_between = sum(len(g) * (float(np.mean(g)) - grand_mean) ** 2 for g in group_arrays)
    return ss_between / ss_total


def _epsilon_squared(h_stat: float, n: int) -> float:
    denom = (n**2 - 1) / (n + 1)
    if denom <= 0:
        return 0.0
    return h_stat / denom


def _cramers_v(chi2: float, n: int, n_rows: int, n_cols: int) -> float:
    denom = n * min(n_rows - 1, n_cols - 1)
    if denom <= 0:
        return 0.0
    return float((chi2 / denom) ** 0.5)


def _sample_size_note(
    group_sizes: list[int], significant: bool, practical: bool
) -> str | None:
    """Flags both directions: too little data to trust the test, and too
    much data for statistical significance to mean anything on its own."""
    notes: list[str] = []
    n_small = sum(1 for n in group_sizes if n < MIN_RELIABLE_GROUP_SIZE)
    if n_small:
        notes.append(
            f"{n_small} of {len(group_sizes)} group(s) have fewer than "
            f"{MIN_RELIABLE_GROUP_SIZE} observations — the test may be underpowered."
        )
    total_n = sum(group_sizes)
    if significant and not practical and total_n >= LARGE_N_TRAP_THRESHOLD:
        notes.append(
            f"With n={total_n:,}, this p-value corresponds to a negligible "
            "effect size (see effect_size) — statistically significant, not "
            "necessarily practically important."
        )
    return " ".join(notes) if notes else None


def _lead_with_effect(
    effect_label: str,
    effect_value: float,
    threshold: float,
    practical: bool,
    significant: bool,
    p_val: float,
    alpha: float,
) -> str:
    """Interpretation text that leads with the effect, not the p-value —
    the p-value alone can't distinguish a real effect from a large-n artifact."""
    magnitude = "a meaningful" if practical else "a negligible"
    parts = [f"{effect_label}={effect_value:.3f} — {magnitude} effect (threshold {threshold})."]
    if significant and practical:
        parts.append(f"Statistically significant (p={p_val:.4f} < α={alpha}) and practically meaningful.")
    elif significant and not practical:
        parts.append(
            f"Statistically significant (p={p_val:.4f} < α={alpha}) but the effect size is below the "
            "practical-significance threshold — likely a large-sample artifact, not a meaningful difference."
        )
    elif not significant and practical:
        parts.append(
            f"Not statistically significant (p={p_val:.4f} ≥ α={alpha}) despite an effect size above the "
            "practical threshold — likely underpowered."
        )
    else:
        parts.append(f"Not statistically significant (p={p_val:.4f} ≥ α={alpha}).")
    return " ".join(parts)


class SelectStatisticalTestTool(BaseTool):
    """
    Autonomously selects and executes the appropriate statistical test.

    Returns the test name, statistic, p-value, effect size, and a
    plain-English interpretation — ready for inclusion in the final report.
    """

    name = "select_statistical_test"
    description = (
        "Automatically select and run the correct statistical hypothesis test "
        "based on data characteristics (normality, group count, data types). "
        "Returns test name, statistic, p-value, effect size, and interpretation."
    )
    requires_context: ClassVar[dict[str, str]] = {"target_column": "group_column"}

    def default_params(
        self, profile: DatasetProfile | None, metadata: DatasetMetadata | None
    ) -> dict[str, Any]:
        """Pick a measure column for the deterministic (no-LLM) planner.
        Deliberately leaves `group_column` unset (7.6).

        Only `feature_column` is filled here, and it prefers a column typed
        `semantic_role == "measure"` over an arbitrary positional numeric
        pick — this is the literal fix for the "tests quantity when the real
        question was amount" bug (7.6 item 3).

        `group_column` is intentionally left empty rather than guessing a
        small categorical column, for an ordering reason: BaseTool.
        prepare_params only fills a param via `requires_context` when the
        planner left it empty, so pre-filling group_column here would win
        the race against the target_column -> group_column injection every
        time, regardless of which one is the better choice. Leaving it
        empty lets that injection apply when a real target exists, and
        otherwise `execute()` falls into family mode — testing this measure
        against every eligible dimension and ranking by effect size instead
        of committing to one arbitrary pairing before the tool ever runs."""
        if profile is None:
            return {}
        measures = [c.name for c in profile.measures() if c.nunique > 1]
        if not measures:
            # No column typed as a measure yet (e.g. an older profile) —
            # fall back to the previous plain-numeric heuristic.
            measures = [
                c.name for c in profile.columns
                if c.kind == "numeric" and c.nunique > 1
            ]
        if not measures:
            return {}
        return {"feature_column": measures[0]}

    def applies_to(self, profile: DatasetProfile | None, metadata: DatasetMetadata | None) -> float:
        if metadata and metadata.target_column and metadata.task_type == "classification":
            return 1.0
        if profile is None:
            return 0.6
        groupable = [c for c in profile.columns if 2 <= c.nunique <= 20 and c.kind in ("categorical", "boolean")]
        return 0.6 if groupable else 0.0

    def execute(  # type: ignore[override]
        self,
        file_path: str,
        feature_column: str | None = None,
        group_column: str | None = None,
        alpha: float = 0.05,
        test_family: bool = False,
        **_: Any,
    ) -> dict[str, Any]:
        df = _read_df(file_path)

        # Needed to pick a typed measure/dimension rather than an arbitrary
        # positional column (7.6 item 3), and to enumerate family-mode
        # candidates (item 1). execute() never receives the controller's
        # DatasetProfile directly, so it is recomputed here; a failure here
        # (unexpected data shape) must not break the explicit-pair path
        # below, hence the broad except.
        try:
            profile = profile_dataframe(df)
        except Exception:
            profile = None

        group_supplied = bool(group_column)

        if feature_column is None:
            feature_column = self._pick_measure_column(df, profile)
            if feature_column is None:
                raise ToolExecutionError(
                    "No numeric measure column found to test. Pass feature_column explicitly."
                )
        elif feature_column not in df.columns:
            raise ToolExecutionError(f"Feature column '{feature_column}' not found.")

        # Reject ID-like columns. The old heuristic here (monotonic + >95%
        # unique) rejected any sorted continuous measurement — a genuine
        # feature exported grouped/sorted by key is sorted by construction,
        # and continuous measurements are ~100% unique regardless of order.
        # profiler.is_identifier_like requires a name hint alongside
        # near-uniqueness (or exact uniqueness on a non-float dtype), so a
        # sorted float measurement no longer trips it — only a real key does.
        feat_series = df[feature_column].dropna()
        if is_identifier_like(feature_column, feat_series, len(feat_series)):
            raise ToolExecutionError(
                f"Column '{feature_column}' appears to be a row ID or index. "
                "Pass a meaningful feature instead."
            )

        # Family mode (7.6): test_family=True forces it explicitly; leaving
        # group_column empty (the deterministic planner's default_params no
        # longer guesses one — see default_params) infers it. An explicit
        # feature_column/group_column pair supplied together keeps testing
        # exactly that pair, unchanged, for backward compatibility.
        family_mode = test_family or not group_supplied
        if not family_mode:
            if group_column not in df.columns:
                raise ToolExecutionError(f"Group column '{group_column}' not found.")
            result, _n = self._run_single_test(df, feature_column, group_column, alpha)
            return result

        return self._execute_family(
            df, profile, feature_column, group_column if group_supplied else None, alpha
        )

    def _pick_measure_column(self, df: pd.DataFrame, profile: DatasetProfile | None) -> str | None:
        """Prefer a column typed `semantic_role == 'measure'` over an
        arbitrary positional numeric pick (7.6 item 3 — the literal fix for
        testing `quantity` when the real question was `amount`)."""
        if profile is not None:
            for c in profile.measures():
                if c.name in df.columns and df[c.name].dropna().nunique() > 1:
                    return c.name
        for col in df.columns:
            series = df[col]
            if not pd.api.types.is_numeric_dtype(series):
                continue
            clean = series.dropna()
            if clean.nunique() <= 1:
                continue
            if is_identifier_like(col, clean, len(clean)):
                continue
            return col
        return None

    def _eligible_dimensions(
        self,
        df: pd.DataFrame,
        profile: DatasetProfile | None,
        feature_column: str,
        explicit_group_column: str | None,
    ) -> list[str]:
        """Every admissible grouping column for family mode: typed
        `semantic_role in ("dimension", "flag")`, with a level count between
        2 and MAX_DIMENSION_CARDINALITY (7.6 item 1). An explicitly-supplied
        group_column (present only when test_family=True overrides an
        explicit pair) is always included — that's the caller's decision to
        honor (7.6 item 3)."""
        candidates: list[str] = []
        if profile is not None:
            for c in profile.dimensions():
                if c.name == feature_column or c.name not in df.columns:
                    continue
                if 2 <= c.nunique <= MAX_DIMENSION_CARDINALITY:
                    candidates.append(c.name)
        else:
            # Defensive fallback if profiling itself failed above — mirrors
            # the categorical/boolean heuristic this tool used before 7.6.
            for col in df.columns:
                if col == feature_column:
                    continue
                series = df[col]
                nunique = int(series.nunique(dropna=True))
                if not (2 <= nunique <= MAX_DIMENSION_CARDINALITY):
                    continue
                if pd.api.types.is_numeric_dtype(series) and nunique > 2:
                    continue  # looks like a continuous/count measure, not a category
                candidates.append(col)
        if (
            explicit_group_column
            and explicit_group_column in df.columns
            and explicit_group_column not in candidates
        ):
            candidates.append(explicit_group_column)
        return candidates

    def _execute_family(
        self,
        df: pd.DataFrame,
        profile: DatasetProfile | None,
        feature_column: str,
        explicit_group_column: str | None,
        alpha: float,
    ) -> dict[str, Any]:
        """Test `feature_column` against every eligible dimension column,
        rank by effect size, and BH-correct p-values within the family
        (7.6 items 1-2)."""
        candidate_dims = self._eligible_dimensions(df, profile, feature_column, explicit_group_column)
        if not candidate_dims:
            raise ToolExecutionError(
                f"No eligible dimension columns (semantic_role in 'dimension'/'flag', "
                f"2-{MAX_DIMENSION_CARDINALITY} levels) found to test '{feature_column}' against."
            )

        raw_results: dict[str, dict[str, Any]] = {}
        totals: dict[str, int] = {}
        for dim in candidate_dims:
            try:
                r, n = self._run_single_test(df, feature_column, dim, alpha)
            except ToolExecutionError:
                continue
            raw_results[dim] = r
            totals[dim] = n

        if not raw_results:
            raise ToolExecutionError(
                f"None of the {len(candidate_dims)} candidate dimension(s) produced a valid "
                f"test against '{feature_column}'."
            )

        # Built from the (feature_column, dim) the loop above already knows
        # rather than reading r["feature_column"]/r["group_column"] back —
        # the categorical-feature branch (_chi_square) doesn't put those
        # keys in its result dict, matching this tool's pre-7.6 output
        # shape for that path.
        pvalue_tests = [
            {
                "feature_column": feature_column,
                "group_column": dim,
                "test_name": r["test_name"],
                "effect_size": r["effect_size"],
                "effect_kind": r["effect_size_metric"],
                "p_value": r["p_value"],
                "n": totals[dim],
            }
            for dim, r in raw_results.items()
        ]
        corrected = apply_benjamini_hochberg(pvalue_tests, alpha=alpha)
        # Rank by effect size (magnitude), not p-value — a family of tests
        # can include several significant-but-negligible pairings once BH
        # correction is applied to enough of them; effect size is what
        # actually answers "which pairing matters" (7.6 item 2).
        ranked = sorted(corrected, key=lambda e: (-abs(e["effect_size"]), e["p_adjusted"]))

        top_group = ranked[0]["group_column"]
        result = dict(raw_results[top_group])
        result["feature_column"] = feature_column
        result["group_column"] = top_group
        result["p_adjusted"] = ranked[0]["p_adjusted"]
        result["significant_after_correction"] = ranked[0]["significant_after_correction"]
        result["family_results"] = ranked
        result["family_size"] = len(ranked)
        result["test_family"] = True
        result["summary"] = (
            f"Tested '{feature_column}' against {len(ranked)} dimension(s); strongest pairing — "
            f"{result['summary']}"
        )
        return result

    def _run_single_test(
        self, df: pd.DataFrame, feature_column: str, group_column: str, alpha: float
    ) -> tuple[dict[str, Any], int]:
        """Run the one auto-selected test for this (feature, group) pair.

        Returns (result, total_n). `result` is exactly the dict this tool
        has always returned for a single explicit pairing — unchanged shape
        for backward compatibility. `total_n` is extra bookkeeping consumed
        only by family mode (7.6) and never exposed to a single-pair caller.
        """
        if group_column not in df.columns:
            raise ToolExecutionError(f"Group column '{group_column}' not found.")

        df_clean = df[[feature_column, group_column]].dropna()
        # For continuous group columns, bin into quartiles automatically
        if str(df_clean[group_column].dtype).startswith("float"):
            df_clean = df_clean.copy()
            df_clean[group_column] = pd.qcut(df_clean[group_column], q=4,
                                              labels=["Q1","Q2","Q3","Q4"],
                                              duplicates="drop")

        groups = df_clean.groupby(group_column)[feature_column].apply(list)
        group_arrays_raw = [pd.array(g) for g in groups]
        group_sizes = [len(g) for g in group_arrays_raw]
        n_groups = len(group_arrays_raw)

        if n_groups < 2:
            raise ToolExecutionError("At least 2 groups are required for hypothesis testing.")
        if n_groups > MAX_DIMENSION_CARDINALITY:
            raise ToolExecutionError(
                f"Too many groups ({n_groups}) for hypothesis testing. "
                f"Pass a categorical group_column with ≤{MAX_DIMENSION_CARDINALITY} unique values."
            )

        # Categorical feature → Chi-Square / Fisher's exact
        # (is_numeric_dtype, not `dtype == object`: pandas 3 strings are `str` dtype)
        if not pd.api.types.is_numeric_dtype(df_clean[feature_column]):
            return self._chi_square(df_clean, feature_column, group_column, alpha)

        group_arrays = [np.asarray(g, dtype=float) for g in group_arrays_raw]

        # Normality (Shapiro-Wilk, seeded sub-sample for large groups). Shapiro
        # requires n>=3 and raises otherwise; a group smaller than that can't
        # be tested for normality, so treat it as non-normal and fall back to
        # the non-parametric branch below rather than crashing the whole test.
        is_normal = all(
            len(g) >= 3 and stats.shapiro(_seeded_sample(g))[1] > alpha
            for g in group_arrays
        )

        mean_diff_ci: tuple[float, float] | None = None

        if n_groups == 2:
            g1, g2 = group_arrays[0], group_arrays[1]
            n1, n2 = len(g1), len(g2)
            if is_normal:
                _, p_lev = stats.levene(g1, g2)
                equal_var = p_lev > alpha
                stat, p_val = stats.ttest_ind(g1, g2, equal_var=equal_var)
                if equal_var:
                    test_name = "Independent T-Test"
                    effect_value = _cohens_d(g1, g2)
                    effect_metric = "cohens_d"
                else:
                    test_name = "Welch's T-Test"
                    effect_value = _hedges_g(g1, g2)
                    effect_metric = "hedges_g"
                mean_diff_ci = _mean_diff_ci(g1, g2, equal_var, alpha)
                threshold = PRACTICAL_THRESHOLD_D
            else:
                stat, p_val = stats.mannwhitneyu(g1, g2, alternative="two-sided")
                test_name = "Mann-Whitney U"
                effect_value = _rank_biserial(float(stat), n1, n2)
                effect_metric = "rank_biserial"
                threshold = PRACTICAL_THRESHOLD_R
        else:
            if is_normal:
                stat, p_val = stats.f_oneway(*group_arrays)
                test_name = "One-Way ANOVA"
                effect_value = _eta_squared(group_arrays)
                effect_metric = "eta_squared"
            else:
                stat, p_val = stats.kruskal(*group_arrays)
                test_name = "Kruskal-Wallis"
                effect_value = _epsilon_squared(float(stat), sum(group_sizes))
                effect_metric = "epsilon_squared"
            threshold = PRACTICAL_THRESHOLD_ETA2

        significant = bool(p_val < alpha)
        practical = abs(effect_value) >= threshold
        interpretation = _lead_with_effect(
            effect_label=effect_metric, effect_value=effect_value, threshold=threshold,
            practical=practical, significant=significant, p_val=float(p_val), alpha=alpha,
        )
        sample_size_note = _sample_size_note(group_sizes, significant, practical)

        result: dict[str, Any] = {
            "summary": f"{test_name}: stat={stat:.4f}, p={p_val:.4f}. {interpretation}",
            "test_name": test_name,
            "statistic": round(float(stat), 6),
            "p_value": round(float(p_val), 6),
            "alpha": alpha,
            "significant": significant,
            "n_groups": n_groups,
            "normality_assumed": is_normal,
            "interpretation": interpretation,
            "feature_column": feature_column,
            "group_column": group_column,
            "effect_size": round(float(effect_value), 4),
            "effect_size_metric": effect_metric,
            "practical_significance": practical,
            "practical_threshold": threshold,
        }
        if mean_diff_ci is not None:
            result["mean_diff_ci_95"] = [round(mean_diff_ci[0], 6), round(mean_diff_ci[1], 6)]
        if sample_size_note:
            result["sample_size_note"] = sample_size_note
        return result, sum(group_sizes)

    def _chi_square(
        self, df: pd.DataFrame, feature_col: str, group_col: str, alpha: float
    ) -> tuple[dict[str, Any], int]:
        contingency = pd.crosstab(df[feature_col], df[group_col])
        chi2_stat, chi2_p, dof, expected = stats.chi2_contingency(contingency)
        n = int(contingency.to_numpy().sum())
        n_rows, n_cols = contingency.shape
        cramers_v = _cramers_v(float(chi2_stat), n, n_rows, n_cols)

        stat, p_val, test_name = float(chi2_stat), float(chi2_p), "Chi-Square Test of Independence"
        expected_frequency_warning: str | None = None
        if (expected < _MIN_EXPECTED_CELL_COUNT).any():
            if n_rows == 2 and n_cols == 2:
                odds_ratio, fisher_p = stats.fisher_exact(contingency.to_numpy())
                stat, p_val, test_name = float(odds_ratio), float(fisher_p), "Fisher's Exact Test"
                expected_frequency_warning = (
                    f"Expected cell count(s) below {_MIN_EXPECTED_CELL_COUNT} — chi-square is unreliable "
                    "at this cell size, switched to Fisher's exact test."
                )
            else:
                expected_frequency_warning = (
                    f"Expected cell count(s) below {_MIN_EXPECTED_CELL_COUNT} in this "
                    f"{n_rows}x{n_cols} table — the chi-square statistic may be unreliable; "
                    "interpret with caution."
                )

        significant = bool(p_val < alpha)
        practical = cramers_v >= PRACTICAL_THRESHOLD_V
        interpretation = _lead_with_effect(
            effect_label="cramers_v", effect_value=cramers_v, threshold=PRACTICAL_THRESHOLD_V,
            practical=practical, significant=significant, p_val=p_val, alpha=alpha,
        )
        sample_size_note = _sample_size_note([n], significant, practical)

        result: dict[str, Any] = {
            "summary": f"{test_name}: stat={stat:.4f}, p={p_val:.4f}. {interpretation}",
            "test_name": test_name,
            "statistic": round(stat, 6),
            "p_value": round(p_val, 6),
            "degrees_of_freedom": int(dof),
            "alpha": alpha,
            "significant": significant,
            "interpretation": interpretation,
            "effect_size": round(cramers_v, 4),
            "effect_size_metric": "cramers_v",
            "practical_significance": practical,
            "practical_threshold": PRACTICAL_THRESHOLD_V,
        }
        if expected_frequency_warning:
            result["expected_frequency_warning"] = expected_frequency_warning
        if sample_size_note:
            result["sample_size_note"] = sample_size_note
        return result, n

    def get_schema(self) -> dict[str, Any]:
        return {
            "file_path": {"type": "string", "description": "Path to dataset.", "required": True},
            "feature_column": {
                "type": "string",
                "description": (
                    "The continuous or categorical measure to test. If omitted, the "
                    "best available measure column (semantic_role == 'measure') is "
                    "picked automatically."
                ),
                "required": False,
            },
            "group_column": {
                "type": "string",
                "description": (
                    "The column defining comparison groups. If omitted, or if "
                    "test_family=True, feature_column is tested against every "
                    "eligible dimension/flag column instead of just this one, "
                    "ranked by effect size in output['family_results']."
                ),
                "required": False,
            },
            "alpha": {
                "type": "float",
                "description": "Significance level. Default: 0.05.",
                "required": False,
            },
            "test_family": {
                "type": "boolean",
                "description": (
                    "Force family mode — test feature_column against every eligible "
                    "dimension column and rank by effect size — even when "
                    "group_column is also supplied. Default: False."
                ),
                "required": False,
            },
        }

    def findings(
        self,
        output: dict[str, Any],
        profile: DatasetProfile | None,
        metadata: DatasetMetadata | None,
    ) -> list[Finding]:
        """Derive Finding objects (7.6 item 4) from either the single
        pairing this tool ran, or every surviving pairing in family_results.

        A pairing only becomes a Finding when it clears both bars this file
        already uses to distinguish a real result from a large-n artifact:
        p_adjusted < 0.05 (BH-corrected within the family when in family
        mode; this tool's own p_value otherwise — no run-wide correction is
        available at this layer) AND a non-negligible effect size, per the
        same PRACTICAL_THRESHOLD_* this tool already reports as
        practical_significance/"negligible effect" in its interpretation
        text. A statistically-significant-but-negligible-effect pairing
        (the large-n trap) is deliberately skipped.
        """
        if not output:
            return []

        family = output.get("family_results")
        if family:
            candidates = [
                {
                    "feature_column": p.get("feature_column"),
                    "group_column": p.get("group_column"),
                    "test_name": p.get("test_name"),
                    "effect_size": p.get("effect_size"),
                    "effect_kind": p.get("effect_kind"),
                    "p_value": p.get("p_value"),
                    "p_adjusted": p.get("p_adjusted"),
                    "n": p.get("n"),
                }
                for p in family
            ]
        else:
            candidates = [{
                "feature_column": output.get("feature_column"),
                "group_column": output.get("group_column"),
                "test_name": output.get("test_name"),
                "effect_size": output.get("effect_size"),
                "effect_kind": output.get("effect_size_metric"),
                "p_value": output.get("p_value"),
                # No run-wide BH correction is visible at this layer for a
                # single explicit pairing — that happens once, run-wide, in
                # src.core.multiple_testing off the accumulated p-values.
                # Falling back to the raw p_value keeps this tool's own
                # significance bar consistent with `significant` above.
                "p_adjusted": output.get("p_value"),
                "n": None,
            }]

        results: list[Finding] = []
        for c in candidates:
            p_adj, effect, effect_kind = c["p_adjusted"], c["effect_size"], c["effect_kind"]
            if p_adj is None or effect is None or effect_kind is None:
                continue
            if p_adj >= _FINDING_P_THRESHOLD:
                continue
            threshold = _EFFECT_KIND_THRESHOLDS.get(effect_kind, PRACTICAL_THRESHOLD_D)
            if abs(effect) < threshold:
                continue  # negligible effect — the large-n trap, not a finding
            measure, dimension = c["feature_column"], c["group_column"]
            headline = (
                f"'{measure}' differs meaningfully across '{dimension}' "
                f"({c['test_name']}, {effect_kind}={effect:.3f}, p_adj={p_adj:.4f})."
            )
            results.append(Finding(
                finding_id=f"{self.name}:{measure}:{dimension}",
                kind="test",
                headline=headline,
                detail=output.get("interpretation", ""),
                evidence={
                    "test_name": c["test_name"],
                    "p_value": c["p_value"],
                    "p_adjusted": p_adj,
                    "n": c["n"],
                },
                source_tool=self.name,
                measure=measure,
                dimension=dimension,
                effect=effect,
                effect_kind=effect_kind,
                p_value=c["p_value"],
                p_adjusted=p_adj,
                confidence=0.7,
            ))
        return results
