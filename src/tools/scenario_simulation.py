"""
Scenario Simulation Tool — Execution Layer.

Models "What-If" intervention scenarios on continuous drivers and segment equalizations,
producing projected baseline shifts, total population impact, and 95% confidence intervals.
"""
from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from scipy import stats

from src.core.findings import Finding
from src.tools.base import BaseTool, ToolExecutionError
from src.tools.data_processing import _read_df

if TYPE_CHECKING:
    from src.core.memory import DatasetMetadata
    from src.core.profiler import DatasetProfile

logger = logging.getLogger(__name__)


_PROTECTED_ATTRIBUTES: frozenset[str] = frozenset({
    "gender", "sex", "race", "ethnicity", "religion", "disability", "sexual_orientation", "age",
})

_TARGET_GOALS: frozenset[str] = frozenset({"minimize", "maximize", "hold", "sensitivity_only"})
_ADVERSE_TARGET_TERMS: frozenset[str] = frozenset({
    "churn", "churned", "default", "defaulted", "defaults", "fraud", "fraudulent",
    "failure", "failures", "failed", "mortality", "pollution", "emission", "emissions",
    "defect", "defects", "error", "errors", "incident", "incidents", "risk", "risks",
    "complaint", "complaints",
})
_BENEFICIAL_TARGET_TERMS: frozenset[str] = frozenset({
    "revenue", "revenues", "profit", "profits", "sales", "conversion", "conversions",
    "retention", "growth", "margin", "margins",
})


def _slug(text: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in str(text)).strip("_").lower() or "na"


def _resolve_target_goal(target_column: str, requested_goal: str | None) -> tuple[str, str, str]:
    """Resolve an explicitly requested or conservatively inferred outcome direction."""
    normalized_target = _slug(target_column)
    target_terms = set(normalized_target.split("_"))
    adverse = sorted(target_terms & _ADVERSE_TARGET_TERMS)
    beneficial = sorted(target_terms & _BENEFICIAL_TARGET_TERMS)

    if requested_goal is not None:
        normalized_goal = requested_goal.strip().lower()
        if normalized_goal not in _TARGET_GOALS:
            allowed = ", ".join(sorted(_TARGET_GOALS))
            raise ToolExecutionError(f"target_goal must be one of: {allowed}.")
        if normalized_goal == "maximize" and adverse and not beneficial:
            raise ToolExecutionError(
                f"Cannot maximize adverse target '{target_column}' (detected adverse term(s): {', '.join(adverse)}). "
                f"Target goal must be 'minimize', 'hold', or 'sensitivity_only'."
            )
        return normalized_goal, "user", "Provided by the scenario request."

    if adverse and not beneficial:
        return "minimize", "domain_policy", f"Column name contains adverse outcome term(s): {', '.join(adverse)}."
    if beneficial and not adverse:
        return "maximize", "domain_policy", f"Column name contains beneficial outcome term(s): {', '.join(beneficial)}."
    return (
        "sensitivity_only",
        "inferred_ambiguous",
        "No unambiguous outcome direction was supplied or inferred; reporting sensitivity only.",
    )


_INTENSIVE_TOKENS = frozenset({
    "avg", "average", "mean", "median", "rate", "ratio", "pct", "percent", "percentage",
    "share", "score", "index", "price", "margin", "std", "stdev", "var", "min", "max",
    "per", "density", "age", "rank", "rating", "temp", "temperature", "celsius",
    "fahrenheit", "satisfaction", "humidity", "pressure", "voltage", "speed", "frequency",
})


def _is_additive_target(column: str) -> bool:
    tokens = set(re.split(r"[^a-z0-9]+", column.lower()))
    return not bool(tokens & _INTENSIVE_TOKENS)


class ScenarioSimulationTool(BaseTool):
    """Simulate parametric 'What-If' interventions on drivers and segments."""

    name = "scenario_simulation"
    is_concurrent_safe = True
    description = (
        "Model 'What-If' scenarios for a target outcome variable given intervention shifts "
        "(e.g. +5%, +10%, +25% on a key driver feature, or closing segment performance gaps). "
        "Calculates expected impact, population-level gains, and 95% confidence intervals."
    )

    def applies_to(self, profile: DatasetProfile | None, metadata: DatasetMetadata | None) -> float:
        if profile is None or profile.row_count < 20:
            return 0.0
        measures = profile.measures()
        if len(measures) >= 2:
            return 0.85
        # Single-measure datasets can still use segment equalization mode
        # if there is at least one discrete categorical dimension with 2-8 levels.
        has_segment_dim = any(
            c.kind in ("categorical", "string") and 2 <= c.nunique <= 8
            for c in profile.columns
        )
        if len(measures) == 1 and has_segment_dim:
            return 0.55
        return 0.0

    def default_params(
        self, profile: DatasetProfile | None, metadata: DatasetMetadata | None
    ) -> dict[str, Any]:
        if profile is None:
            return {}
        target = getattr(metadata, "target_column", None)
        if target:
            return {"target_column": target}
        measures = profile.measures()
        if measures:
            return {"target_column": measures[0].name}
        return {}

    def execute(  # type: ignore[override]
        self,
        file_path: str,
        target_column: str | None = None,
        driver_column: str | None = None,
        shifts: list[float] | None = None,
        segment_column: str | None = None,
        joint_optimization: bool = True,
        target_goal: str | None = None,
        max_shift_pct: float = 20.0,
        projection_population: int | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        df = _read_df(file_path)
        if df.empty:
            raise ToolExecutionError("Dataset has no rows.")

        numeric_cols = [
            c for c in df.columns
            if pd.api.types.is_numeric_dtype(df[c]) and df[c].nunique(dropna=True) > 2
        ]

        if not target_column:
            if numeric_cols:
                target_column = numeric_cols[0]
            else:
                raise ToolExecutionError("No numeric target column available for simulation.")

        if target_column not in df.columns:
            raise ToolExecutionError(f"Target column '{target_column}' not found.")
        if not pd.api.types.is_numeric_dtype(df[target_column]):
            raise ToolExecutionError(f"Target column '{target_column}' must be numeric.")

        resolved_goal, goal_source, goal_rationale = _resolve_target_goal(target_column, target_goal)

        target_s = pd.to_numeric(df[target_column], errors="coerce").dropna()
        if len(target_s) < 20:
            raise ToolExecutionError(f"Target column '{target_column}' has fewer than 20 numeric observations.")

        n_proj = int(projection_population) if projection_population is not None else len(target_s)
        n_est = len(target_s)
        base_mean = float(target_s.mean())
        base_total = float(target_s.sum())
        is_binary_target = bool(target_s.isin([0, 1]).all())
        is_additive = is_binary_target or _is_additive_target(target_column)

        # Select driver column if omitted
        raw_candidate_drivers = [c for c in numeric_cols if c != target_column]
        excluded_levers: list[dict[str, str]] = []
        candidate_drivers: list[str] = []
        for column in raw_candidate_drivers:
            normalized = _slug(column)
            tokens = set(normalized.split("_"))
            reason = None
            if normalized in _PROTECTED_ATTRIBUTES or bool(tokens & _PROTECTED_ATTRIBUTES):
                reason = "protected attribute"
            elif tokens & {"id", "key", "uuid", "date", "time", "timestamp"}:
                reason = "identifier or temporal field"
            elif tokens & {"sensor", "reading", "token"}:
                reason = "measurement or token, not a controllable lever"
            if reason:
                excluded_levers.append({"column": str(column), "reason": reason})
            else:
                candidate_drivers.append(str(column))
        if not driver_column and candidate_drivers:
            # Pick highest Pearson correlation
            best_r = -1.0
            best_col = candidate_drivers[0]
            for c in candidate_drivers:
                sub = df[[c, target_column]].dropna()
                if len(sub) >= 20:
                    try:
                        r, _ = stats.pearsonr(sub[c], sub[target_column])
                        if abs(r) > best_r:
                            best_r = abs(r)
                            best_col = c
                    except Exception:
                        pass
            driver_column = best_col

        scenarios: list[dict[str, Any]] = []
        driver_model: dict[str, Any] | None = None

        if driver_column and driver_column in df.columns and pd.api.types.is_numeric_dtype(df[driver_column]):
            sub = df[[driver_column, target_column]].dropna()
            if len(sub) >= 20:
                n_est = len(sub)
                x = sub[driver_column].to_numpy(dtype=float)
                y = sub[target_column].to_numpy(dtype=float)
                x_mean = float(x.mean())
                x_min = float(x.min())
                x_max = float(x.max())
                x_var = float(np.var(x, ddof=1))
                if x_var > 1e-9:
                    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
                    driver_model = {
                        "driver": driver_column,
                        "slope": float(slope),
                        "intercept": float(intercept),
                        "r_squared": float(r_value ** 2),
                        "p_value": float(p_value),
                        "std_err": float(std_err),
                        "x_mean": x_mean,
                        "x_min": x_min,
                        "x_max": x_max,
                    }

                    sim_shifts = shifts or [5.0, 10.0, 25.0, -10.0]
                    for shift_pct in sim_shifts:
                        delta_x = x_mean * (shift_pct / 100.0)
                        simulated_x = x_mean + delta_x
                        is_extrap = bool(simulated_x < x_min or simulated_x > x_max)
                        delta_y = float(slope * delta_x)
                        ci_margin = float(1.96 * std_err * abs(delta_x)) if delta_x else 0.0
                        new_mean = base_mean + delta_y
                        if is_binary_target:
                            new_mean = float(np.clip(new_mean, 0.0, 1.0))
                            delta_y = new_mean - base_mean
                            ci_lower = max(-base_mean, delta_y - ci_margin)
                            ci_upper = min(1.0 - base_mean, delta_y + ci_margin)
                        elif float(target_s.min()) >= 0.0:
                            new_mean = max(0.0, new_mean)
                            delta_y = new_mean - base_mean
                            ci_lower = max(-base_mean, delta_y - ci_margin)
                            ci_upper = delta_y + ci_margin
                        else:
                            ci_lower = delta_y - ci_margin
                            ci_upper = delta_y + ci_margin
                        rel_change_pct = (delta_y / base_mean * 100.0) if abs(base_mean) > 1e-9 else 0.0
                        tot_impact = round(delta_y * n_proj, 2) if is_additive else None

                        scenarios.append({
                            "type": "driver_shift",
                            "driver": driver_column,
                            "shift_pct": shift_pct,
                            "delta_driver": round(delta_x, 4),
                            "simulated_driver_value": round(simulated_x, 4),
                            "driver_min": round(x_min, 4),
                            "driver_max": round(x_max, 4),
                            "is_extrapolation": is_extrap,
                            "delta_target": round(delta_y, 4),
                            "relative_change_pct": round(rel_change_pct, 2),
                            "ci_lower": round(ci_lower, 4),
                            "ci_upper": round(ci_upper, 4),
                            "projected_mean": round(new_mean, 4),
                            "projected_total_impact": tot_impact,
                            "sensitivity_total": tot_impact,
                        })

        # Multi-variable joint optimization across top continuous drivers
        joint_package: dict[str, Any] | None = None
        prescriptive_optimization_allowed = (
            joint_optimization
            and resolved_goal in {"minimize", "maximize"}
            and not is_binary_target
        )
        if prescriptive_optimization_allowed and len(candidate_drivers) >= 2:
            top_drivers = candidate_drivers[:4]
            sub_mv = df[[target_column, *top_drivers]].dropna()
            if len(sub_mv) >= 25:
                try:
                    x_mv = sub_mv[top_drivers].to_numpy(dtype=float)
                    y_mv = sub_mv[target_column].to_numpy(dtype=float)
                    x_design = np.column_stack([np.ones(len(x_mv)), x_mv])
                    coeffs_mv, _, rank_mv, _ = np.linalg.lstsq(x_design, y_mv, rcond=None)
                    if rank_mv == len(top_drivers) + 1:
                        slopes_mv = coeffs_mv[1:]
                        mv_interventions: list[dict[str, Any]] = []
                        total_mv_delta_y = 0.0
                        for idx, drv in enumerate(top_drivers):
                            s_drv = float(slopes_mv[idx])
                            mean_drv = float(np.mean(sub_mv[drv]))
                            direction = 1.0 if s_drv >= 0 else -1.0
                            if resolved_goal == "minimize":
                                direction = -direction
                            shift_applied = direction * abs(max_shift_pct)
                            dx = mean_drv * (shift_applied / 100.0)
                            dy_contrib = float(s_drv * dx)
                            total_mv_delta_y += dy_contrib
                            mv_interventions.append({
                                "driver": drv,
                                "slope": round(s_drv, 4),
                                "mean": round(mean_drv, 4),
                                "recommended_shift_pct": shift_applied,
                                "unit_shift": round(dx, 4),
                                "target_impact": round(dy_contrib, 4),
                            })

                        rel_opt_change = (total_mv_delta_y / base_mean * 100.0) if abs(base_mean) > 1e-9 else 0.0
                        joint_tot_impact = round(total_mv_delta_y * n_proj, 2) if is_additive else None
                        joint_package = {
                            "target_goal": resolved_goal,
                            "goal_source": goal_source,
                            "goal_rationale": goal_rationale,
                            "drivers_optimized": top_drivers,
                            "interventions": mv_interventions,
                            "predicted_delta_target": round(total_mv_delta_y, 4),
                            "relative_change_pct": round(rel_opt_change, 2),
                            "projected_mean": round(base_mean + total_mv_delta_y, 4),
                            "projected_total_impact": joint_tot_impact,
                            "sensitivity_total": joint_tot_impact,
                        }
                except Exception:
                    joint_package = None

        # Segment equalization opportunity (skips demographic/protected attributes)
        segment_opportunity: dict[str, Any] | None = None
        def _is_protected_col(col_name: Any) -> bool:
            slugged = _slug(col_name)
            return slugged in _PROTECTED_ATTRIBUTES or bool(set(slugged.split("_")) & _PROTECTED_ATTRIBUTES)

        cat_candidates = [
            c for c in df.columns
            if c != target_column
            and 2 <= df[c].nunique(dropna=True) <= 8
            and not pd.api.types.is_numeric_dtype(df[c])
            and not _is_protected_col(c)
        ]
        seg_col = segment_column or (cat_candidates[0] if cat_candidates else None)
        if resolved_goal in {"minimize", "maximize"} and seg_col and seg_col in df.columns and not _is_protected_col(seg_col):
            seg_work = df[[seg_col, target_column]].dropna()
            group_stats = seg_work.groupby(seg_col)[target_column].agg(["count", "mean"])
            if len(group_stats) >= 2:
                eligible = group_stats[group_stats["count"] >= 15]
                if not eligible.empty:
                    segment = eligible["mean"].idxmax() if resolved_goal == "minimize" else eligible["mean"].idxmin()
                    segment_mean = float(eligible.loc[segment, "mean"])
                    segment_n = int(eligible.loc[segment, "count"])
                    rest_mean = float(seg_work.loc[seg_work[seg_col] != segment, target_column].mean())
                    gap = rest_mean - segment_mean
                    if (resolved_goal == "minimize" and gap < 0) or (resolved_goal == "maximize" and gap > 0):
                        segment_opportunity = {
                            "dimension": seg_col,
                            "segment": str(segment),
                            "segment_n": segment_n,
                            "segment_mean": round(segment_mean, 4),
                            "rest_mean": round(rest_mean, 4),
                            "gap_per_unit": round(gap, 4),
                            "gap_pct": round((gap / abs(rest_mean)) * 100.0, 2) if rest_mean else None,
                            "recoverable_total": None,
                            "cohort_gap_total": round(gap * segment_n, 2) if is_additive else None,
                        }

        # Build Markdown decision table
        lines = [
            f"| Sensitivity Benchmark / Lever | Reference Cohort | Statistical Shift (Unadjusted) | 95% Confidence Interval | Linear Extrapolation ({n_proj:,} units) |",
            "| :--- | :--- | :--- | :--- | :--- |",
        ]
        if joint_optimization and not prescriptive_optimization_allowed:
            if is_binary_target:
                lines.append("| Joint Optimization | Not generated | Binary target; unbounded prescriptive optimization is disabled | N/A | N/A |")
            else:
                lines.append(f"| Joint Optimization | Not generated | target_goal={resolved_goal}; descriptive sensitivity only | N/A | N/A |")
        if joint_package:
            interv_summary = ", ".join(f"{it['driver']} {it['recommended_shift_pct']:+0.0f}%" for it in joint_package["interventions"])
            joint_tot_str = f"{joint_package['projected_total_impact']:+,.2f}" if joint_package["projected_total_impact"] is not None else "N/A (Intensive Measure)"
            lines.append(
                f"| Multi-Driver Sensitivity Simulation ({interv_summary}) | All Population | "
                f"{joint_package['predicted_delta_target']:+0.2f} ({joint_package['relative_change_pct']:+0.1f}%) | "
                f"[Joint multivariate model] | {joint_tot_str} |"
            )
        for s in scenarios:
            s_tot_str = f"{s['projected_total_impact']:+,.2f}" if s["projected_total_impact"] is not None else "N/A (Intensive Measure)"
            extrap_badge = " [Extrapolation]" if s.get("is_extrapolation") else ""
            lines.append(
                f"| Shift {s['driver']} {s['shift_pct']:+0.0f}% | All Population | {s['delta_target']:+0.2f} ({s['relative_change_pct']:+0.1f}%) | "
                f"[{s['ci_lower']:+0.2f}, {s['ci_upper']:+0.2f}] | {s_tot_str}{extrap_badge} |"
            )
        if segment_opportunity:
            observed_total = segment_opportunity["cohort_gap_total"]
            seg_tot_str = f"{observed_total:+,.2f} across the cohort" if observed_total is not None else "N/A (Intensive Measure)"
            lines.append(
                f"| Difference to other records | {segment_opportunity['segment']} (n={segment_opportunity['segment_n']:,}) | "
                f"{segment_opportunity['gap_per_unit']:+0.2f} per record (others minus cohort) | Observed comparison | "
                f"{seg_tot_str} |"
            )
        decision_table = "\n".join(lines)

        # Build summary
        summary_parts: list[str] = []
        summary_parts.append(f"Scenario goal: {resolved_goal} ({goal_source}; {goal_rationale})")
        if joint_optimization and not prescriptive_optimization_allowed:
            if is_binary_target:
                summary_parts.append("Joint optimization was not generated because the target is binary.")
            else:
                summary_parts.append("Joint optimization was not generated because this scenario is non-prescriptive.")
        if not is_additive:
            summary_parts.append(f"Total impact omitted: '{target_column}' is an intensive/non-additive measure.")
        if joint_package:
            if joint_package["projected_total_impact"] is not None:
                joint_tot_phrase = f"; total impact: {joint_package['projected_total_impact']:+,.2f}"
            else:
                joint_tot_phrase = " (intensive measure; total omitted)"
            summary_parts.append(
                f"Multi-driver joint sensitivity simulation ({', '.join(joint_package['drivers_optimized'])}) projects a "
                f"{joint_package['relative_change_pct']:+0.1f}% cumulative shift in '{target_column}' "
                f"({joint_package['predicted_delta_target']:+0.2f} per unit{joint_tot_phrase})."
            )
        if scenarios:
            top_s = next((s for s in scenarios if s["shift_pct"] == 10.0), scenarios[0])
            if top_s["projected_total_impact"] is not None:
                tot_phrase = f"extrapolating to {top_s['projected_total_impact']:+,.2f} across {n_proj:,} records"
            else:
                tot_phrase = f"total impact omitted (intensive measure; projection n={n_proj:,}, estimation n={n_est:,})"
            extrap_note = " [Extrapolation beyond observed driver support]" if top_s.get("is_extrapolation") else ""
            summary_parts.append(
                f"A {top_s['shift_pct']:+0.0f}% shift in '{top_s['driver']}' corresponds to an unadjusted linear change in "
                f"'{target_column}' of {top_s['relative_change_pct']:+0.1f}% ({top_s['delta_target']:+0.2f} per unit, "
                f"95% CI: [{top_s['ci_lower']:+0.2f}, {top_s['ci_upper']:+0.2f}]), {tot_phrase}{extrap_note} "
                "[observational sensitivity benchmark under stated counterfactual assumption; unadjusted for confounding]."
            )
        if segment_opportunity:
            summary_parts.append(
                f"Mean '{target_column}' was {segment_opportunity['segment_mean']:,.2f} for "
                f"{segment_opportunity['dimension']}='{segment_opportunity['segment']}' and "
                f"{segment_opportunity['rest_mean']:,.2f} for other records. This is an unadjusted cohort comparison."
            )

        summary = " ".join(summary_parts) if summary_parts else f"Baseline scenario analysis complete for '{target_column}' across {n_proj:,} records."

        return {
            "summary": summary,
            "target_column": target_column,
            "target_goal": resolved_goal,
            "goal_source": goal_source,
            "goal_rationale": goal_rationale,
            "is_binary_target": is_binary_target,
            "bounded_target": is_binary_target,
            "is_additive_target": is_additive,
            "model_family": (
                "bounded_probability_sensitivity" if is_binary_target
                else "linear_sensitivity"
            ),
            "actionable_levers": candidate_drivers,
            "excluded_levers": excluded_levers,
            "estimation_population": n_est,
            "projection_population": n_proj,
            "population_size": n_proj,
            "baseline_mean": round(base_mean, 4),
            "baseline_total": round(base_total, 2),
            "driver_model": driver_model,
            "scenarios": scenarios,
            "joint_package": joint_package,
            "segment_opportunity": segment_opportunity,
            "decision_table": decision_table,
        }

    def findings(
        self,
        output: dict[str, Any],
        profile: DatasetProfile | None,
        metadata: DatasetMetadata | None,
    ) -> list[Finding]:
        results: list[Finding] = []
        target = output.get("target_column") or "target"
        scenarios = output.get("scenarios") or []
        n_proj = int(output.get("projection_population", output.get("population_size", 0)))
        n_est = int(output.get("estimation_population", output.get("population_size", 0)))
        for i, s in enumerate(scenarios):
            if s.get("shift_pct") in (10.0, -10.0, 25.0) or len(scenarios) <= 3 or s.get("is_extrapolation"):
                if abs(s.get("relative_change_pct", 0.0)) / 100.0 < 0.02 and not s.get("is_extrapolation"):
                    continue
                extrap_flag = " [Extrapolation]" if s.get("is_extrapolation") else ""
                headline = (
                    f"Sensitivity Projection: {s['shift_pct']:+0.0f}% shift in '{s['driver']}' associates with a "
                    f"{s['relative_change_pct']:+0.1f}% change in '{target}' ({s['delta_target']:+0.2f} per unit, "
                    f"95% CI: [{s['ci_lower']:+0.2f}, {s['ci_upper']:+0.2f}]){extrap_flag}."
                )
                if s.get("projected_total_impact") is not None:
                    tot_desc = f"Linear extrapolation across population: {s['projected_total_impact']:+,.2f}"
                else:
                    tot_desc = f"Total impact omitted: '{target}' is an intensive/non-additive measure"

                results.append(Finding(
                    finding_id=f"decision_scenario_{i}_{_slug(target)}_{_slug(s['driver'])}_{int(s['shift_pct'])}",
                    kind="decision_scenario",
                    headline=headline,
                    detail=(
                        f"{tot_desc}; baseline mean: {output.get('baseline_mean')}; "
                        f"projection n={n_proj:,}; estimation n={n_est:,}. "
                        "Observational linear sensitivity benchmark under stated counterfactual assumption (unadjusted for confounding)."
                    ),
                    evidence={
                        **s,
                        "estimation_population": n_est,
                        "projection_population": n_proj,
                        "is_additive": output.get("is_additive_target", True),
                    },
                    source_tool=self.name,
                    measure=target,
                    dimension=s["driver"],
                    effect=s["relative_change_pct"] / 100.0,
                    effect_kind="lift",
                    confidence=0.75,
                    surprise=0.3,
                    caveats=[
                        "Observational linear extrapolation unadjusted for unmeasured confounding; "
                        "descriptive sensitivity only, not guaranteed causal impact."
                    ],
                ))

        seg_opp = output.get("segment_opportunity")
        if seg_opp:
            observed_difference = seg_opp["segment_mean"] - seg_opp["rest_mean"]
            headline = (
                f"Mean '{target}' for {seg_opp['dimension']}='{seg_opp['segment']}' was "
                f"{abs(observed_difference):,.2f} {'higher' if observed_difference > 0 else 'lower'} than for other records."
            )
            results.append(Finding(
                finding_id=f"recoverable_gap_{_slug(target)}_{_slug(seg_opp['dimension'])}_{_slug(seg_opp['segment'])}",
                kind="decision_scenario",
                headline=headline,
                detail=f"Segment n={seg_opp['segment_n']:,}; segment mean: {seg_opp['segment_mean']}; rest mean: {seg_opp['rest_mean']}.",
                evidence=seg_opp,
                source_tool=self.name,
                measure=target,
                dimension=seg_opp["dimension"],
                level=seg_opp["segment"],
                effect=observed_difference / abs(seg_opp["rest_mean"]) if seg_opp["rest_mean"] else observed_difference,
                effect_kind="lift" if seg_opp["rest_mean"] else "difference",
                confidence=0.70,
                surprise=0.4,
                caveats=["Observational cohort difference; descriptive benchmark unadjusted for confounding or selection effects, not guaranteed causal recovery."],
            ))

        joint_pkg = output.get("joint_package")
        if joint_pkg and abs(joint_pkg.get("relative_change_pct", 0.0)) >= 1.0:
            drvs_str = ", ".join(joint_pkg["drivers_optimized"])
            if joint_pkg.get("projected_total_impact") is not None:
                joint_tot_str = f", total impact {joint_pkg['projected_total_impact']:+,.2f}"
            else:
                joint_tot_str = " (intensive measure; total omitted)"
            headline = (
                f"Multi-Driver Joint Sensitivity Simulation: Coordinated shift across {len(joint_pkg['drivers_optimized'])} levers ({drvs_str}) "
                f"projects a {joint_pkg['relative_change_pct']:+0.1f}% cumulative shift in '{target}' "
                f"({joint_pkg['predicted_delta_target']:+0.2f} per unit{joint_tot_str})."
            )
            results.append(Finding(
                finding_id=f"joint_opt_{_slug(target)}",
                kind="decision_scenario",
                headline=headline,
                detail=(
                    f"Multivariate sensitivity simulation identified coordinated shifts across {drvs_str}. "
                    f"Projected mean moves to {joint_pkg['projected_mean']} across {output.get('projection_population', output.get('population_size'))} units."
                ),
                evidence=joint_pkg,
                source_tool=self.name,
                measure=target,
                dimension=joint_pkg["drivers_optimized"][0] if joint_pkg["drivers_optimized"] else None,
                effect=abs(joint_pkg["relative_change_pct"]) / 100.0,
                effect_kind="lift",
                confidence=0.80,
                surprise=0.5,
                objective_fit=0.90,
                caveats=["Multivariate linear sensitivity optimization; descriptive OLS shift unadjusted for unmeasured confounding, not guaranteed causal impact."],
            ))

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
                "description": "Target metric to project (e.g. revenue, cost, score).",
                "required": False,
            },
            "driver_column": {
                "type": "string",
                "description": "Continuous feature to shift parametrically.",
                "required": False,
            },
            "shifts": {
                "type": "array",
                "description": "List of percentage shifts to evaluate (default: [+5, +10, +25, -10]).",
                "required": False,
            },
            "segment_column": {
                "type": "string",
                "description": "Categorical column to evaluate for gap recovery.",
                "required": False,
            },
            "target_goal": {
                "type": "string",
                "enum": ["minimize", "maximize", "hold", "sensitivity_only"],
                "description": (
                    "Outcome direction. If omitted, only unambiguous target names are inferred; "
                    "ambiguous targets default to sensitivity_only."
                ),
                "required": False,
            },
            "joint_optimization": {
                "type": "boolean",
                "description": "Generate prescriptive multi-driver shifts only for non-binary minimize/maximize goals.",
                "required": False,
            },
            "projection_population": {
                "type": "integer",
                "description": "Eligible population count to project totals across (defaults to dataset rows).",
                "required": False,
            },
        }
