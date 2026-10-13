"""
Scenario Simulation Tool — Execution Layer.

Models "What-If" intervention scenarios on continuous drivers and segment equalizations,
producing projected baseline shifts, total population impact, and 95% confidence intervals.
"""
from __future__ import annotations

import logging
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
    "gender", "sex", "race", "ethnicity", "religion", "disability", "sexual_orientation",
})


def _slug(text: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in str(text)).strip("_").lower() or "na"


class ScenarioSimulationTool(BaseTool):
    """Simulate parametric 'What-If' interventions on drivers and segments."""

    name = "scenario_simulation"
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
        target_goal: str = "maximize",
        max_shift_pct: float = 20.0,
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

        target_s = pd.to_numeric(df[target_column], errors="coerce").dropna()
        if len(target_s) < 20:
            raise ToolExecutionError(f"Target column '{target_column}' has fewer than 20 numeric observations.")

        n_pop = len(target_s)
        base_mean = float(target_s.mean())
        base_total = float(target_s.sum())

        # Select driver column if omitted
        candidate_drivers = [c for c in numeric_cols if c != target_column]
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
                x = sub[driver_column].to_numpy(dtype=float)
                y = sub[target_column].to_numpy(dtype=float)
                x_mean = float(x.mean())
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
                    }

                    sim_shifts = shifts or [5.0, 10.0, 25.0, -10.0]
                    for shift_pct in sim_shifts:
                        delta_x = x_mean * (shift_pct / 100.0)
                        delta_y = float(slope * delta_x)
                        ci_margin = float(1.96 * std_err * abs(delta_x))
                        new_mean = base_mean + delta_y
                        rel_change_pct = (delta_y / base_mean * 100.0) if abs(base_mean) > 1e-9 else 0.0
                        tot_impact = delta_y * n_pop

                        scenarios.append({
                            "type": "driver_shift",
                            "driver": driver_column,
                            "shift_pct": shift_pct,
                            "delta_driver": round(delta_x, 4),
                            "delta_target": round(delta_y, 4),
                            "relative_change_pct": round(rel_change_pct, 2),
                            "ci_lower": round(delta_y - ci_margin, 4),
                            "ci_upper": round(delta_y + ci_margin, 4),
                            "projected_mean": round(new_mean, 4),
                            "projected_total_impact": round(tot_impact, 2),
                        })

        # Multi-variable joint optimization across top continuous drivers
        joint_package: dict[str, Any] | None = None
        if joint_optimization and len(candidate_drivers) >= 2:
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
                            if target_goal == "minimize":
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
                        joint_package = {
                            "target_goal": target_goal,
                            "drivers_optimized": top_drivers,
                            "interventions": mv_interventions,
                            "predicted_delta_target": round(total_mv_delta_y, 4),
                            "relative_change_pct": round(rel_opt_change, 2),
                            "projected_mean": round(base_mean + total_mv_delta_y, 4),
                            "projected_total_impact": round(total_mv_delta_y * n_pop, 2),
                        }
                except Exception:
                    joint_package = None

        # Segment equalization opportunity (skips demographic/protected attributes)
        segment_opportunity: dict[str, Any] | None = None
        cat_candidates = [
            c for c in df.columns
            if c != target_column
            and 2 <= df[c].nunique(dropna=True) <= 8
            and not pd.api.types.is_numeric_dtype(df[c])
            and str(c).lower().strip() not in _PROTECTED_ATTRIBUTES
        ]
        seg_col = segment_column or (cat_candidates[0] if cat_candidates else None)
        if seg_col and seg_col in df.columns and str(seg_col).lower().strip() not in _PROTECTED_ATTRIBUTES:
            seg_work = df[[seg_col, target_column]].dropna()
            group_stats = seg_work.groupby(seg_col)[target_column].agg(["count", "mean"])
            if len(group_stats) >= 2:
                # Find lowest performing segment with at least 15 rows
                eligible = group_stats[group_stats["count"] >= 15]
                if not eligible.empty:
                    lowest_seg = eligible["mean"].idxmin()
                    lowest_mean = float(eligible.loc[lowest_seg, "mean"])
                    lowest_n = int(eligible.loc[lowest_seg, "count"])
                    rest_mean = float(seg_work[seg_work[seg_col] != lowest_seg][target_column].mean())
                    gap = rest_mean - lowest_mean
                    if gap > 0:
                        recoverable_total = gap * lowest_n
                        segment_opportunity = {
                            "dimension": seg_col,
                            "segment": str(lowest_seg),
                            "segment_n": lowest_n,
                            "segment_mean": round(lowest_mean, 4),
                            "rest_mean": round(rest_mean, 4),
                            "gap_per_unit": round(gap, 4),
                            "gap_pct": round((gap / rest_mean) * 100.0, 2) if rest_mean > 0 else 0.0,
                            "recoverable_total": round(recoverable_total, 2),
                        }

        # Build Markdown decision table
        lines = [
            f"| Sensitivity Benchmark / Lever | Reference Cohort | Statistical Shift (Unadjusted) | 95% Confidence Interval | Linear Extrapolation ({n_pop:,} units) |",
            "| :--- | :--- | :--- | :--- | :--- |",
        ]
        if joint_package:
            interv_summary = ", ".join(f"{it['driver']} {it['recommended_shift_pct']:+0.0f}%" for it in joint_package["interventions"])
            lines.append(
                f"| Multi-Driver Optimization ({interv_summary}) | All Population | "
                f"{joint_package['predicted_delta_target']:+0.2f} ({joint_package['relative_change_pct']:+0.1f}%) | "
                f"[Joint multivariate model] | {joint_package['projected_total_impact']:+,.2f} |"
            )
        for s in scenarios:
            lines.append(
                f"| Shift {s['driver']} {s['shift_pct']:+0.0f}% | All Population | {s['delta_target']:+0.2f} ({s['relative_change_pct']:+0.1f}%) | "
                f"[{s['ci_lower']:+0.2f}, {s['ci_upper']:+0.2f}] | {s['projected_total_impact']:+,.2f} |"
            )
        if segment_opportunity:
            lines.append(
                f"| Benchmark '{segment_opportunity['segment']}' to average | {segment_opportunity['segment']} (n={segment_opportunity['segment_n']:,}) | "
                f"+{segment_opportunity['gap_per_unit']:0.2f} (+{segment_opportunity['gap_pct']:0.1f}%) | [Audited baseline gap] | "
                f"+{segment_opportunity['recoverable_total']:,.2f} (Cohort difference) |"
            )
        decision_table = "\n".join(lines)

        # Build summary
        summary_parts: list[str] = []
        if joint_package:
            summary_parts.append(
                f"Multi-driver joint optimization ({', '.join(joint_package['drivers_optimized'])}) projects a "
                f"{joint_package['relative_change_pct']:+0.1f}% cumulative shift in '{target_column}' "
                f"({joint_package['predicted_delta_target']:+0.2f} per unit; total impact: {joint_package['projected_total_impact']:+,.2f})."
            )
        if scenarios:
            top_s = next((s for s in scenarios if s["shift_pct"] == 10.0), scenarios[0])
            summary_parts.append(
                f"A {top_s['shift_pct']:+0.0f}% shift in '{top_s['driver']}' corresponds to an unadjusted linear change in "
                f"'{target_column}' of {top_s['relative_change_pct']:+0.1f}% ({top_s['delta_target']:+0.2f} per unit, "
                f"95% CI: [{top_s['ci_lower']:+0.2f}, {top_s['ci_upper']:+0.2f}]), extrapolating to "
                f"{top_s['projected_total_impact']:+,.2f} across {n_pop:,} records [descriptive sensitivity benchmark; unadjusted for confounding]."
            )
        if segment_opportunity:
            summary_parts.append(
                f"The observed performance gap for cohort '{segment_opportunity['segment']}' in '{segment_opportunity['dimension']}' "
                f"represents an unadjusted cohort difference of +{segment_opportunity['recoverable_total']:,.2f}."
            )

        summary = " ".join(summary_parts) if summary_parts else f"Baseline scenario analysis complete for '{target_column}' across {n_pop:,} records."

        return {
            "summary": summary,
            "target_column": target_column,
            "population_size": n_pop,
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
        for i, s in enumerate(scenarios):
            if s.get("shift_pct") in (10.0, -10.0, 25.0):
                if abs(s.get("relative_change_pct", 0.0)) / 100.0 < 0.02:
                    continue
                headline = (
                    f"Sensitivity Projection: {s['shift_pct']:+0.0f}% shift in '{s['driver']}' associates with a "
                    f"{s['relative_change_pct']:+0.1f}% change in '{target}' ({s['delta_target']:+0.2f} per unit, "
                    f"95% CI: [{s['ci_lower']:+0.2f}, {s['ci_upper']:+0.2f}])."
                )
                results.append(Finding(
                    finding_id=f"decision_scenario_{i}_{_slug(target)}_{_slug(s['driver'])}_{int(s['shift_pct'])}",
                    kind="decision_scenario",
                    headline=headline,
                    detail=(
                        f"Linear extrapolation across population: {s['projected_total_impact']:+,.2f}; "
                        f"baseline mean: {output.get('baseline_mean')}; population n={output.get('population_size')}."
                    ),
                    evidence=s,
                    source_tool=self.name,
                    measure=target,
                    dimension=s["driver"],
                    effect=s["relative_change_pct"] / 100.0,
                    effect_kind="lift",
                    confidence=0.75,
                    surprise=0.3,
                    caveats=["Observational linear extrapolation unadjusted for unmeasured confounding."],
                ))

        seg_opp = output.get("segment_opportunity")
        if seg_opp:
            headline = (
                f"Cohort Benchmark: Segment '{seg_opp['segment']}' in '{seg_opp['dimension']}' trails baseline by "
                f"{seg_opp['gap_per_unit']:0.2f} per unit ({seg_opp['gap_pct']:0.1f}%), an aggregate gap of "
                f"+{seg_opp['recoverable_total']:,.2f}."
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
                effect=seg_opp["gap_pct"] / 100.0,
                effect_kind="lift",
                confidence=0.70,
                surprise=0.4,
                caveats=["Observational cohort difference; unadjusted for confounding or selection effects."],
            ))

        joint_pkg = output.get("joint_package")
        if joint_pkg and abs(joint_pkg.get("relative_change_pct", 0.0)) >= 1.0:
            drvs_str = ", ".join(joint_pkg["drivers_optimized"])
            headline = (
                f"Multi-Driver Joint Optimization: Coordinated shift across {len(joint_pkg['drivers_optimized'])} levers ({drvs_str}) "
                f"projects a {joint_pkg['relative_change_pct']:+0.1f}% cumulative shift in '{target}' "
                f"({joint_pkg['predicted_delta_target']:+0.2f} per unit, total impact {joint_pkg['projected_total_impact']:+,.2f})."
            )
            results.append(Finding(
                finding_id=f"joint_opt_{_slug(target)}",
                kind="decision_scenario",
                headline=headline,
                detail=(
                    f"Multivariate optimization identified coordinated interventions across {drvs_str}. "
                    f"Projected mean moves to {joint_pkg['projected_mean']} across {output.get('population_size')} units."
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
                caveats=["Multivariate linear optimization; assumes independent additive driver interventions."],
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
        }
