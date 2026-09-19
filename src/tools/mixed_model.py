"""
Mixed-Model Analysis Tool — Execution Layer.

"What drives this measure once I account for rows that share a site / patient /
batch?" For repeated-measures or nested data (replicates, patients, sites,
batches, plots, schools) rows from the same group resemble each other, so
treating them as independent overstates how much evidence there is.

The fit is a linear mixed model (statsmodels MixedLM, REML, L-BFGS) with a
random intercept per group and a small, lean set of fixed effects. Numeric
predictors are standardised for the fit; every effect is reported per natural
unit and per 1 SD (of the predictor) with a 95% interval. A random slope is
added only when exactly one numeric feature is requested and there are 20+
groups. If the mixed model does not converge or is singular, the fit falls back
to a random-intercept-only model and then to plain OLS with cluster-robust
standard errors, and says so.

The intraclass correlation (ICC, the share of variance that sits between
groups) comes from a one-way variance-components (ANOVA) estimate, and the
design effect 1 + (mean group size - 1) * ICC says how much naive
independent-rows inference would understate the uncertainty.
"""
from __future__ import annotations

import math
import re
import warnings
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from src.core.findings import Finding
from src.core.profiler import profile_dataframe
from src.tools.base import BaseTool, ToolExecutionError
from src.tools.data_processing import _read_df

if TYPE_CHECKING:
    from src.core.memory import DatasetMetadata
    from src.core.profiler import ColumnProfile, DatasetProfile

_MAX_GROUPS = 2_000
_MAX_ROWS = 50_000
_MAX_FEATURES = 6
_AUTO_FEATURES = 5
_MIN_GROUPS = 5
_MIN_GROUP_SIZE = 3.0
_MIN_ROWS = 30
_SLOPE_MIN_GROUPS = 20
_MAX_LEVELS = 8
_MAX_CAT_UNIQUE = 30
_MAX_MISSING_AUTO = 0.2
_MAX_FINDINGS = 3
_MIN_STD_EFFECT = 0.05
_MIN_ICC_FOR_FINDING = 0.05
_MIN_DEFF_FOR_NOTE = 1.1
_GROUP_ROLES = ("dimension", "identifier", "ordinal")
_SKIP_FEATURE_ROLES = ("identifier", "text", "time", "constant")
_ID_WORDS = frozenset({"id", "ids", "code", "no", "num", "number", "key"})
_GROUP_TOKENS = frozenset({
    "subject", "patient", "participant", "site", "batch", "plot", "replicate", "cluster",
    "school", "plate", "sample", "station", "unit",
})
_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def _tokens(name: str) -> list[str]:
    return [t for t in _NON_ALNUM.split(_CAMEL.sub("_", str(name)).lower()) if t]


def _is_group_name(name: str) -> bool:
    return any(t in _GROUP_TOKENS or (t.endswith("s") and t[:-1] in _GROUP_TOKENS) for t in _tokens(name))


def _group_candidates(profile: DatasetProfile) -> list[ColumnProfile]:
    n = profile.row_count
    out: list[ColumnProfile] = []
    for c in profile.columns:
        if c.semantic_role not in _GROUP_ROLES or c.nunique < _MIN_GROUPS or c.nunique > n / 2:
            continue
        if n / c.nunique < _MIN_GROUP_SIZE:
            continue
        if c.name == profile.entity_col or _is_group_name(c.name):
            out.append(c)
    return out


def _best_group(profile: DatasetProfile) -> str | None:
    candidates = _group_candidates(profile)
    if not candidates:
        return None
    entity = next((c for c in candidates if c.name == profile.entity_col), None)
    return (entity or candidates[0]).name


def _group_nouns(column: str) -> tuple[str, str]:
    words = [t for t in _tokens(column) if t not in _ID_WORDS]
    word = words[-1] if words else str(column)
    plural = word if word.endswith("s") else (word + "es" if word.endswith(("x", "ch", "sh")) else word + "s")
    return word, plural


def _num(value: float) -> str:
    a = abs(value)
    if a >= 1000:
        return f"{value:,.0f}"
    if a >= 100:
        return f"{value:.0f}"
    if a >= 10:
        return f"{value:.1f}"
    if a >= 1:
        return f"{value:.2f}"
    if a >= 0.01:
        return f"{value:.3f}"
    return f"{value:.2g}" if value else "0"


def _pct_text(share: float) -> str:
    pct = share * 100
    return "under 1%" if 0 < pct < 1 else f"{pct:.0f}%"


def _r(value: float, digits: int = 4) -> float:
    return round(float(value), digits)


def _coerce_target(raw: pd.Series, target: str) -> pd.Series:
    y = pd.to_numeric(raw, errors="coerce")
    non_null = raw.dropna()
    if int(y.notna().sum()) < 0.5 * len(non_null) or y.notna().sum() == 0:
        if 0 < non_null.nunique() <= 2:
            raise ToolExecutionError(_binary_message(target))
        raise ToolExecutionError(f"Target '{target}' is not numeric; a mixed model needs a numeric measure.")
    distinct = int(y.dropna().nunique())
    if distinct <= 2:
        raise ToolExecutionError(_binary_message(target) if distinct == 2 else f"Target '{target}' is constant - nothing to explain.")
    return y.astype("float64")


def _binary_message(target: str) -> str:
    return (
        f"Target '{target}' is a two-value (yes/no) outcome; a linear mixed model needs a numeric target. "
        "Use regression_analysis, which fits a logistic regression for yes/no outcomes."
    )


def _classify(df: pd.DataFrame, column: str, role: str | None) -> str | None:
    """'numeric', 'categorical' or None (not usable as a predictor)."""
    s = df[column]
    if role in _SKIP_FEATURE_ROLES or pd.api.types.is_datetime64_any_dtype(s):
        return None
    nunique = int(s.nunique(dropna=True))
    if nunique < 2:
        return None
    if pd.api.types.is_bool_dtype(s):
        return "categorical"
    if pd.api.types.is_numeric_dtype(s):
        return "numeric" if nunique > 2 else "categorical"
    return "categorical" if nunique <= _MAX_CAT_UNIQUE else None


def _association(y: pd.Series, s: pd.Series, kind: str) -> float:
    """|Pearson r| for a numeric feature, eta for a categorical one."""
    if kind == "numeric":
        r = pd.to_numeric(s, errors="coerce").corr(y)
        return abs(float(r)) if r == r else 0.0
    frame = pd.DataFrame({"y": y.to_numpy(dtype=float), "s": s.astype(str).to_numpy()}).dropna()
    if len(frame) < 3:
        return 0.0
    total = float(((frame["y"] - frame["y"].mean()) ** 2).sum())
    if total <= 0:
        return 0.0
    grouped = frame.groupby("s")["y"].agg(["mean", "size"])
    between = float((grouped["size"] * (grouped["mean"] - frame["y"].mean()) ** 2).sum())
    return math.sqrt(max(between / total, 0.0))


def _select_features(
    df: pd.DataFrame,
    y: pd.Series,
    exclude: set[str],
    requested: list[str] | None,
    roles: dict[str, str],
    notes: list[str],
) -> list[tuple[str, str]]:
    if requested:
        chosen: list[tuple[str, str]] = []
        for name in requested:
            if name in exclude:
                notes.append(f"'{name}' is the target or grouping column and was left out of the predictors.")
                continue
            kind = _classify(df, name, roles.get(name))
            if kind is None:
                notes.append(f"'{name}' cannot be used as a predictor (constant, identifier, date or free text) and was skipped.")
                continue
            chosen.append((name, kind))
        if len(chosen) > _MAX_FEATURES:
            notes.append(f"Only the first {_MAX_FEATURES} predictors were used.")
            chosen = chosen[:_MAX_FEATURES]
        if not chosen:
            raise ToolExecutionError("None of the requested feature_columns can be used as predictors.")
        return chosen
    scored: list[tuple[float, str, str]] = []
    for name in df.columns:
        if name in exclude or float(df[name].isna().mean()) > _MAX_MISSING_AUTO:
            continue
        kind = _classify(df, str(name), roles.get(str(name)))
        if kind is None:
            continue
        scored.append((_association(y, df[name], kind), str(name), kind))
    scored.sort(key=lambda t: (-t[0], t[1]))
    return [(name, kind) for score, name, kind in scored[:_AUTO_FEATURES] if score > 0]


def _build_terms(df: pd.DataFrame, feats: list[tuple[str, str]], notes: list[str]) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Standardised design columns x0, x1, ... plus a description of each."""
    columns: dict[str, np.ndarray] = {}
    terms: list[dict[str, Any]] = []

    def add(feature: str, kind: str, values: np.ndarray, level: str | None, reference: str | None) -> None:
        sd = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        if not math.isfinite(sd) or sd <= 0:
            notes.append(f"'{feature}' has no variation in the analysed rows and was dropped.")
            return
        key = f"x{len(terms)}"
        mean = float(np.mean(values))
        columns[key] = (values - mean) / sd
        label = feature if level is None else f"{feature}: {level} vs {reference}"
        terms.append({
            "key": key, "feature": feature, "kind": kind, "level": level, "reference": reference,
            "label": label, "sd": sd,
        })

    for feature, kind in feats:
        s = df[feature]
        if kind == "numeric":
            add(feature, kind, pd.to_numeric(s, errors="coerce").to_numpy(dtype=float), None, None)
            continue
        text = s.astype(str).where(s.notna(), "(missing)")
        counts = text.value_counts()
        keep = list(counts.index[:_MAX_LEVELS])
        if len(counts) > _MAX_LEVELS:
            text = text.where(text.isin(keep), "Other")
            notes.append(f"'{feature}' has {len(counts)} levels; the {_MAX_LEVELS} most common are kept and the rest are grouped as 'Other'.")
        ordered = list(text.value_counts().index)
        reference = ordered[0]
        for level in ordered[1:]:
            add(feature, kind, (text == level).to_numpy(dtype=float), str(level), str(reference))
    return pd.DataFrame(columns), terms


def _icc_anova(y: np.ndarray, groups: np.ndarray) -> float:
    """One-way random-effects ICC(1) from variance components, in [0, 1]."""
    codes, _ = pd.factorize(groups)
    k = int(codes.max()) + 1
    n = len(y)
    sizes = np.bincount(codes, minlength=k).astype(float)
    sums = np.bincount(codes, weights=y, minlength=k)
    means = sums / sizes
    grand = float(y.mean())
    ss_between = float((sizes * (means - grand) ** 2).sum())
    ss_within = float(((y - means[codes]) ** 2).sum())
    if k < 2 or n <= k:
        return 0.0
    msb = ss_between / (k - 1)
    msw = ss_within / (n - k)
    n0 = (n - float((sizes**2).sum()) / n) / (k - 1)
    var_between = max((msb - msw) / n0, 0.0) if n0 > 0 else 0.0
    total = var_between + msw
    return float(var_between / total) if total > 0 else 0.0


def _fit_mixed(work: pd.DataFrame, xcols: list[str], slope: str | None) -> tuple[dict[str, Any] | None, str]:
    import statsmodels.formula.api as smf
    from statsmodels.tools.sm_exceptions import ConvergenceWarning

    formula = "y ~ " + " + ".join(xcols)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            model = smf.mixedlm(formula, work, groups=work["g"], re_formula=f"~{slope}" if slope else None)
            res = model.fit(reml=True, method="lbfgs", maxiter=200)
        except (np.linalg.LinAlgError, ValueError, FloatingPointError, IndexError) as exc:
            return None, str(exc)[:80] or type(exc).__name__
    if any(issubclass(w.category, ConvergenceWarning) or "singular" in str(w.message).lower() for w in caught):
        return None, "convergence warning or singular fit"
    if not bool(getattr(res, "converged", True)):
        return None, "did not converge"
    names = [str(n) for n in res.fe_params.index]
    coefs = np.asarray(res.fe_params, dtype=float)
    ses = np.asarray(res.bse_fe, dtype=float)
    if not (np.isfinite(coefs).all() and np.isfinite(ses).all() and (ses > 0).all()):
        return None, "non-finite standard errors"
    var_group = float(np.asarray(res.cov_re)[0, 0])
    var_resid = float(res.scale)
    if not math.isfinite(var_group) or var_group <= 1e-8 * max(var_resid, 1e-12):
        return None, "group variance estimated at zero (singular fit)"
    return {
        "names": names, "coefs": coefs, "ses": ses, "var_group": var_group, "var_resid": var_resid,
    }, ""


def _fit_ols_cluster(work: pd.DataFrame, xcols: list[str]) -> dict[str, Any]:
    import statsmodels.formula.api as smf

    codes = pd.factorize(work["g"])[0]
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = smf.ols("y ~ " + " + ".join(xcols), work).fit(cov_type="cluster", cov_kwds={"groups": codes})
    except (np.linalg.LinAlgError, ValueError) as exc:
        raise ToolExecutionError(f"The model could not be fitted ({exc}); try fewer or different feature_columns.") from exc
    return {
        "names": [str(n) for n in res.params.index], "coefs": np.asarray(res.params, dtype=float),
        "ses": np.asarray(res.bse, dtype=float), "var_group": None, "var_resid": float(res.scale),
    }


def _effect_sentence(t: dict[str, Any], target: str) -> str:
    direction = "higher" if t["per_sd"] >= 0 else "lower"
    evidence = (
        "the pattern is unlikely to be chance"
        if t.get("significant_after_correction") else "too uncertain to tell apart from no effect"
    )
    ci = f"95% range {_num(t['per_sd_low'])} to {_num(t['per_sd_high'])} per 1 SD"
    if t["level"] is not None:
        return (
            f"{t['feature']} = {t['level']} goes with {_num(abs(t['per_unit']))} {direction} {target} than "
            f"{t['reference']} (95% range {_num(t['per_unit_low'])} to {_num(t['per_unit_high'])}); {evidence}."
        )
    return (
        f"Each 1 SD higher {t['feature']} (about {_num(t['sd'])} units) goes with {_num(abs(t['per_sd']))} "
        f"{direction} {target} ({ci}), which is {_num(t['per_unit'])} per unit; {evidence}."
    )


def _chart(terms: list[dict[str, Any]], target: str) -> dict[str, Any] | None:
    from src.core.chart_spec import validate_chart_spec

    shown = sorted(terms, key=lambda t: -abs(t["per_sd"]))[:12]
    rows = [
        {"effect_on": t["label"], "change": t["per_sd"], "lower": t["per_sd_low"], "upper": t["per_sd_high"]}
        for t in shown
    ]
    spec = {
        "type": "dot_ci", "data": rows, "x": "effect_on", "y": "change", "y_lower": "lower", "y_upper": "upper",
        "title": f"Change in {target} per 1 SD of each factor (95% range)",
        "x_title": "Factor", "y_title": f"Change in {target}",
        "annotations": [{"y": 0, "label": "No effect"}],
        "caption": "Effects within the same group; intervals that cross zero are not distinguishable from no effect.",
    }
    clean, _error = validate_chart_spec(spec)
    return clean


class MixedModelAnalysisTool(BaseTool):
    """Random-intercept mixed model for repeated-measures / nested data."""

    name = "mixed_model_analysis"
    description = (
        "For repeated measures or nested data (replicates, patients, sites, batches, plots, schools): "
        "fit a mixed model with a random intercept per group so effects are not overstated by treating "
        "rows from the same group as independent. Reports the share of variation that is between groups "
        "(ICC), each factor's effect within a group per natural unit and per 1 SD with a 95% interval, "
        "and how much naive independent-rows inference would understate the uncertainty (design effect). "
        "Needs a numeric target and a grouping column; refuses yes/no targets (use regression_analysis)."
    )
    requires_ml = True

    def applies_to(self, profile: DatasetProfile | None, metadata: DatasetMetadata | None) -> float:
        if profile is None or profile.row_count < _MIN_ROWS or not profile.measures():
            return 0.0
        return 0.5 if _group_candidates(profile) else 0.0

    def default_params(
        self, profile: DatasetProfile | None, metadata: DatasetMetadata | None
    ) -> dict[str, Any]:
        if profile is None:
            return {}
        group = _best_group(profile)
        if group is None:
            return {}
        measures = [c for c in profile.measures() if c.name != group and c.nunique > 2]
        target = getattr(metadata, "target_column", None)
        if not (target and any(c.name == target for c in measures)):
            measures.sort(key=lambda c: (c.unit_hint != "currency", c.name))
            target = measures[0].name if measures else None
        if target is None:
            return {}
        return {"target_column": target, "group_column": group}

    def execute(  # type: ignore[override]
        self,
        file_path: str,
        target_column: str | None = None,
        group_column: str | None = None,
        feature_columns: list[str] | str | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        from scipy import stats

        from src.core.multiple_testing import apply_benjamini_hochberg

        df = _read_df(file_path)
        if df.empty:
            raise ToolExecutionError("Dataset has no rows.")
        if isinstance(feature_columns, str):
            feature_columns = [c.strip() for c in feature_columns.split(",") if c.strip()]
        for given in (target_column, group_column, *(feature_columns or [])):
            if given and given not in df.columns:
                raise ToolExecutionError(f"Column '{given}' not found in dataset.")
        profile = profile_dataframe(df)
        if group_column is None:
            group_column = _best_group(profile)
            if group_column is None:
                raise ToolExecutionError(
                    "No grouping column found (need a subject/patient/site/batch-like column with 5+ groups "
                    "and 3+ rows each). Pass group_column."
                )
        if target_column is None:
            options = [c for c in profile.measures() if c.name != group_column and c.nunique > 2]
            if not options:
                raise ToolExecutionError("No numeric measure found. Pass target_column.")
            options.sort(key=lambda c: (c.unit_hint != "currency", c.name))
            target_column = options[0].name
        if target_column == group_column:
            raise ToolExecutionError("target_column and group_column must be different columns.")

        notes: list[str] = []
        y_all = _coerce_target(df[target_column], target_column)
        keep = (y_all.notna() & df[group_column].notna()).to_numpy()
        df = df.loc[keep].reset_index(drop=True)
        y = y_all.loc[keep].reset_index(drop=True)
        group_text = df[group_column].astype(str)

        counts = group_text.value_counts()
        n_groups_total = len(counts)
        if n_groups_total > _MAX_GROUPS:
            chosen = np.random.default_rng(0).choice(counts.index.to_numpy(), _MAX_GROUPS, replace=False)
            mask = group_text.isin(chosen).to_numpy()
            df, y, group_text = df.loc[mask].reset_index(drop=True), y.loc[mask].reset_index(drop=True), group_text.loc[mask].reset_index(drop=True)
            notes.append(f"{n_groups_total:,} {group_column} groups found; a random {_MAX_GROUPS:,} were analysed.")

        roles = {c.name: c.semantic_role for c in profile.columns}
        feats = _select_features(
            df, y, {target_column, group_column}, list(feature_columns) if feature_columns else None, roles, notes
        )
        if not feats:
            raise ToolExecutionError("No usable predictor columns found. Pass feature_columns.")

        complete = np.ones(len(df), dtype=bool)
        for name, kind in feats:
            if kind == "numeric":
                complete &= pd.to_numeric(df[name], errors="coerce").notna().to_numpy()
        if not complete.all():
            notes.append(f"{int((~complete).sum()):,} rows with a missing predictor value were left out.")
            df, y, group_text = df.loc[complete].reset_index(drop=True), y.loc[complete].reset_index(drop=True), group_text.loc[complete].reset_index(drop=True)
        sampled = False
        if len(df) > _MAX_ROWS:
            pick = np.sort(np.random.default_rng(0).choice(len(df), _MAX_ROWS, replace=False))
            df, y, group_text = df.iloc[pick].reset_index(drop=True), y.iloc[pick].reset_index(drop=True), group_text.iloc[pick].reset_index(drop=True)
            sampled = True
            notes.append(f"Analysed a random sample of {_MAX_ROWS:,} rows.")

        n_obs = len(df)
        sizes = group_text.value_counts()
        n_groups = len(sizes)
        if n_obs < _MIN_ROWS or n_groups < _MIN_GROUPS:
            raise ToolExecutionError(
                f"Only {n_obs} usable rows in {n_groups} '{group_column}' groups; a mixed model needs at "
                f"least {_MIN_ROWS} rows and {_MIN_GROUPS} groups."
            )
        mean_size = n_obs / n_groups

        design, terms = _build_terms(df, feats, notes)
        if not terms:
            raise ToolExecutionError("No predictor had usable variation. Pass different feature_columns.")
        work = design.copy()
        work["y"] = y.to_numpy(dtype=float)
        work["g"] = group_text.to_numpy()
        xcols = [t["key"] for t in terms]
        target_sd = float(np.std(work["y"], ddof=1))

        slope_key = (
            xcols[0]
            if feature_columns and len(terms) == 1 and terms[0]["kind"] == "numeric" and n_groups >= _SLOPE_MIN_GROUPS
            else None
        )
        attempts: list[tuple[str, str | None]] = []
        if slope_key:
            attempts.append(("mixed_random_slope", slope_key))
        attempts.append(("mixed_random_intercept", None))
        fit: dict[str, Any] | None = None
        method = ""
        for label, slope in attempts:
            fit, why = _fit_mixed(work, xcols, slope)
            if fit is not None:
                method = label
                break
            nxt = "a random-intercept-only model" if slope else "plain OLS with cluster-robust standard errors"
            notes.append(f"The {label.replace('_', ' ')} fit was not stable ({why}); fell back to {nxt}.")
        if fit is None:
            fit = _fit_ols_cluster(work, xcols)
            method = "ols_cluster_robust"

        z95 = float(stats.norm.ppf(0.975))
        by_key = {t["key"]: t for t in terms}
        tests: list[dict[str, Any]] = []
        for name, coef, se in zip(fit["names"], fit["coefs"], fit["ses"], strict=True):
            t = by_key.get(name)
            if t is None or not math.isfinite(se) or se <= 0:
                continue
            sd = t["sd"]
            p = float(2 * stats.norm.sf(abs(coef / se)))
            tests.append({
                "term": name, "feature": t["feature"], "kind": t["kind"], "level": t["level"],
                "reference": t["reference"], "label": t["label"], "sd": _r(sd),
                "per_unit": _r(coef / sd), "per_unit_low": _r((coef - z95 * se) / sd),
                "per_unit_high": _r((coef + z95 * se) / sd),
                "per_sd": _r(coef), "per_sd_low": _r(coef - z95 * se), "per_sd_high": _r(coef + z95 * se),
                "std_effect": _r(coef / target_sd) if target_sd > 0 else None,
                "p_value": _r(p, 6),
            })
        if not tests:
            raise ToolExecutionError("The model produced no usable effect estimates.")
        effects = apply_benjamini_hochberg(tests, alpha=0.05)
        for e in effects:
            e["sentence"] = _effect_sentence(e, target_column)
        effects.sort(key=lambda e: abs(e["std_effect"] or 0.0), reverse=True)

        icc = _icc_anova(work["y"].to_numpy(dtype=float), work["g"].to_numpy())
        deff = 1.0 + (mean_size - 1.0) * icc
        inflation = math.sqrt(max(deff, 1.0))
        singular, plural = _group_nouns(group_column)

        caveats = [
            "These are associations within the data, not proof of cause and effect.",
            *notes,
        ]
        if n_groups < 20:
            caveats.append(f"Only {n_groups} groups; the between-group share is imprecise.")
        if method == "ols_cluster_robust":
            caveats.append("Ordinary regression with cluster-robust standard errors was used instead of a mixed model.")
        if sampled:
            caveats.append(f"Results come from a random sample of {_MAX_ROWS:,} rows.")
        design_note = None
        if deff >= _MIN_DEFF_FOR_NOTE:
            design_note = (
                f"Treating rows as independent would understate uncertainty about {inflation:.1f}x "
                f"for effects that differ between {plural}."
            )

        icc_text = _pct_text(icc)
        summary = (
            f"{icc_text} of the variation in {target_column} is between {plural}, the rest is within the same "
            f"{singular} ({n_groups:,} {plural}, {mean_size:.1f} rows each on average)."
        )
        if design_note:
            summary += f" {design_note}"
        if effects:
            summary += f" {effects[0]['sentence']}"

        chart = _chart(effects, target_column)
        result: dict[str, Any] = {
            "summary": summary,
            "target_column": target_column,
            "group_column": group_column,
            "feature_columns": list(dict.fromkeys(t["feature"] for t in terms)),
            "method": method,
            "random_slope": by_key[slope_key]["feature"] if slope_key and method == "mixed_random_slope" else None,
            "n_obs": n_obs,
            "n_groups": n_groups,
            "mean_obs_per_group": _r(mean_size, 2),
            "min_obs_per_group": int(sizes.min()),
            "max_obs_per_group": int(sizes.max()),
            "target_sd": _r(target_sd),
            "icc": _r(icc),
            "between_groups_text": f"{icc_text} of the variation is between {plural}, the rest is within.",
            "design_effect": _r(deff, 3),
            "se_inflation": _r(inflation, 2),
            "design_effect_note": design_note,
            "group_variance": _r(fit["var_group"], 6) if fit["var_group"] is not None else None,
            "residual_variance": _r(fit["var_resid"], 6),
            "fixed_effects": effects,
            "caveats": caveats,
        }
        if chart is not None:
            result["chart"] = chart
        return result

    def findings(
        self,
        output: dict[str, Any],
        profile: DatasetProfile | None,
        metadata: DatasetMetadata | None,
    ) -> list[Finding]:
        target = str(output.get("target_column"))
        group = str(output.get("group_column"))
        n_groups = int(output.get("n_groups") or 0)
        n_obs = int(output.get("n_obs") or 0)
        caveats = list(output.get("caveats") or [])
        chart = output.get("chart")
        confidence_base = min(1.0, n_groups / 30.0) * min(1.0, n_obs / 200.0)
        singular, plural = _group_nouns(group)
        results: list[Finding] = []

        icc = float(output.get("icc") or 0.0)
        if icc >= _MIN_ICC_FOR_FINDING:
            results.append(Finding(
                finding_id=f"mixed_icc_{target}_{group}".replace(" ", "_"),
                kind="mixed_model",
                headline=(
                    f"{_pct_text(icc)} of the variation in {target} is between {plural}; "
                    f"the rest is within the same {singular}."
                ),
                detail=str(output.get("design_effect_note") or "") or f"Rows from the same {singular} are similar to each other.",
                evidence={
                    "icc": icc, "n_groups": n_groups, "n_obs": n_obs,
                    "mean_obs_per_group": output.get("mean_obs_per_group"),
                    "design_effect": output.get("design_effect"), "se_inflation": output.get("se_inflation"),
                },
                source_tool=self.name, measure=target, dimension=group,
                effect=icc, effect_kind="share", confidence=confidence_base, surprise=min(1.0, icc),
                caveats=caveats, chart_hint=chart,
            ))

        for e in output.get("fixed_effects", []):
            std = e.get("std_effect")
            if not e.get("significant_after_correction") or std is None or abs(std) < _MIN_STD_EFFECT:
                continue
            direction = "higher" if e["per_sd"] >= 0 else "lower"
            if e.get("level") is not None:
                headline = (
                    f"Within the same {singular}, {e['feature']} = {e['level']} goes with "
                    f"{_num(abs(e['per_unit']))} {direction} {target} than {e['reference']} "
                    f"(95% range {_num(e['per_unit_low'])} to {_num(e['per_unit_high'])})."
                )
            else:
                headline = (
                    f"Within the same {singular}, each 1 SD higher {e['feature']} (about {_num(e['sd'])} units) "
                    f"goes with {_num(abs(e['per_sd']))} {direction} {target} "
                    f"(95% range {_num(e['per_sd_low'])} to {_num(e['per_sd_high'])})."
                )
            results.append(Finding(
                finding_id=f"mixed_effect_{target}_{e['term']}_{e['feature']}".replace(" ", "_"),
                kind="mixed_model",
                headline=headline,
                detail=str(e.get("sentence") or ""),
                evidence={**{k: e[k] for k in (
                    "feature", "level", "reference", "sd", "per_unit", "per_unit_low", "per_unit_high",
                    "per_sd", "per_sd_low", "per_sd_high", "std_effect",
                ) if k in e}, "n_groups": n_groups, "n_obs": n_obs},
                source_tool=self.name, measure=target, dimension=e["feature"],
                level=e.get("level"), effect=std, effect_kind="r",
                p_value=e["p_value"], p_adjusted=e.get("p_adjusted"),
                confidence=confidence_base, surprise=min(1.0, abs(std)),
                caveats=caveats, chart_hint=chart,
            ))
            if len(results) >= _MAX_FINDINGS + 1:
                break
        return results

    def get_schema(self) -> dict[str, Any]:
        return {
            "file_path": {
                "type": "string",
                "description": "Path to the dataset.",
                "required": True,
            },
            "target_column": {
                "type": "string",
                "description": (
                    "Numeric outcome to explain (not yes/no). Auto-selected from the measures when omitted."
                ),
                "required": False,
            },
            "group_column": {
                "type": "string",
                "description": (
                    "Column identifying the repeated unit / cluster (subject, patient, site, batch, plot, "
                    "school...). 5+ groups with 3+ rows each. Auto-detected when omitted."
                ),
                "required": False,
            },
            "feature_columns": {
                "type": "array",
                "description": (
                    "Up to 6 predictor columns. Default: the columns most associated with the target. "
                    "A random slope is added only when exactly one numeric feature is given and there are 20+ groups."
                ),
                "required": False,
            },
        }
