"""
Threshold Analysis Tool — Execution Layer.

Detects non-linear tipping points, piecewise breakpoints, and regime shifts in
continuous relationships (e.g. electricity grid load, sensor drift, game balance,
pricing thresholds, fatigue curves).
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, ClassVar

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

_MIN_N = 30
_PERCENTILE_LOW = 10.0
_PERCENTILE_HIGH = 90.0
_MIN_R2_GAIN = 0.02
_ALPHA = 0.05


class ThresholdAnalysisTool(BaseTool):
    """Detect non-linear breakpoints and tipping points where relationships shift."""

    name: str = "threshold_analysis"
    description: str = (
        "Detect non-linear tipping points and piecewise breakpoints in continuous features "
        "(e.g. temperature, load, power, level, duration) where the target outcome's slope or "
        "rate shifts abruptly. Identifies the exact critical threshold value, pre- and post-slopes, "
        "and tests statistical significance against a linear baseline."
    )
    requires_ml: ClassVar[bool] = False

    def applies_to(self, profile: DatasetProfile | None, metadata: DatasetMetadata | None) -> float:
        if profile is None or profile.row_count < _MIN_N:
            return 0.0
        measures = profile.measures()
        if len(measures) >= 2:
            return 0.80
        return 0.0

    def default_params(
        self, profile: DatasetProfile | None, metadata: DatasetMetadata | None
    ) -> dict[str, Any]:
        if profile is None:
            return {}
        target = getattr(metadata, "target_column", None)
        measures = [m.name for m in profile.measures()]
        if not measures:
            return {}
        if target and target in measures:
            feature_candidates = [m for m in measures if m != target]
            if feature_candidates:
                return {"target_column": target, "feature_column": feature_candidates[0]}
        elif len(measures) >= 2:
            return {"target_column": measures[0], "feature_column": measures[1]}
        return {}

    def get_schema(self) -> dict[str, Any]:
        return {
            "file_path": {
                "type": "string",
                "description": "Path to the dataset file.",
                "required": True,
            },
            "target_column": {
                "type": "string",
                "description": "Continuous dependent variable (Y) to analyze.",
                "required": False,
            },
            "feature_column": {
                "type": "string",
                "description": "Continuous independent feature (X) suspected of having a tipping point.",
                "required": False,
            },
            "max_candidates": {
                "type": "int",
                "description": "Maximum number of candidate threshold percentiles to evaluate (default: 40).",
                "required": False,
            },
        }

    def execute(  # type: ignore[override]
        self,
        file_path: str,
        target_column: str | None = None,
        feature_column: str | None = None,
        max_candidates: int = 40,
        **_: Any,
    ) -> dict[str, Any]:
        df = _read_df(file_path)
        if df.empty:
            raise ToolExecutionError("Dataset has no rows.")

        numeric_cols = [
            c for c in df.columns
            if pd.api.types.is_numeric_dtype(df[c]) and df[c].nunique(dropna=True) >= 5
        ]

        if len(numeric_cols) < 2:
            raise ToolExecutionError("Threshold analysis requires at least two continuous numeric columns.")

        if target_column not in df.columns:
            target_column = numeric_cols[0]
        if feature_column not in df.columns or feature_column == target_column:
            remaining = [c for c in numeric_cols if c != target_column]
            if not remaining:
                raise ToolExecutionError("No distinct continuous feature column available for threshold analysis.")
            feature_column = remaining[0]

        valid = df[[feature_column, target_column]].dropna()
        if len(valid) < _MIN_N:
            raise ToolExecutionError(
                f"Insufficient complete observations ({len(valid)}) for threshold analysis (minimum {_MIN_N})."
            )

        x = pd.to_numeric(valid[feature_column], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(valid[target_column], errors="coerce").to_numpy(dtype=float)

        mask = np.isfinite(x) & np.isfinite(y)
        x = x[mask]
        y = y[mask]
        n = len(x)
        if n < _MIN_N:
            raise ToolExecutionError("Insufficient finite values after cleaning.")

        order = np.argsort(x)
        x = x[order]
        y = y[order]

        # 1. Fit baseline simple linear model: Y = a + b*X
        x_mean = float(np.mean(x))
        y_mean = float(np.mean(y))
        ss_xx = float(np.sum((x - x_mean) ** 2))
        if ss_xx <= 1e-12:
            raise ToolExecutionError(f"Feature column '{feature_column}' has zero or near-zero variance.")

        ss_xy = float(np.sum((x - x_mean) * (y - y_mean)))
        b_linear = ss_xy / ss_xx
        a_linear = y_mean - b_linear * x_mean
        y_pred_linear = a_linear + b_linear * x
        rss_linear = float(np.sum((y - y_pred_linear) ** 2))
        tss = float(np.sum((y - y_mean) ** 2))
        r2_linear = max(0.0, 1.0 - rss_linear / tss) if tss > 1e-12 else 0.0

        # 2. Search candidate thresholds across interior percentiles
        low_val = float(np.percentile(x, _PERCENTILE_LOW))
        high_val = float(np.percentile(x, _PERCENTILE_HIGH))
        if low_val >= high_val:
            raise ToolExecutionError(f"Feature '{feature_column}' lacks adequate interior distribution range.")

        num_pts = min(max_candidates, 50)
        candidates = np.linspace(low_val, high_val, num_pts)

        best_c: float | None = None
        best_rss = rss_linear
        best_params: tuple[float, float, float] | None = None  # (b0, b1, b2)

        # Segmented model: Y = b0 + b1*X + b2*(X - c)+
        for c in candidates:
            x_hinge = np.maximum(0.0, x - c)
            # Design matrix: [1, x, (x-c)+]
            x_mat = np.column_stack([np.ones(n), x, x_hinge])
            try:
                # OLS via least squares
                coeffs, residuals, rank, _ = np.linalg.lstsq(x_mat, y, rcond=None)
                if rank < 3:
                    continue
                rss_cand = float(residuals[0]) if len(residuals) > 0 else float(np.sum((y - x_mat @ coeffs) ** 2))
                if rss_cand < best_rss:
                    best_rss = rss_cand
                    best_c = float(c)
                    best_params = (float(coeffs[0]), float(coeffs[1]), float(coeffs[2]))
            except Exception:
                continue

        r2_piecewise = max(0.0, 1.0 - best_rss / tss) if tss > 1e-12 else 0.0
        r2_gain = float(r2_piecewise - r2_linear)

        # 3. Statistical test for structural break (F-test comparing models)
        p_val = 1.0
        f_stat = 0.0
        has_breakpoint = False
        pre_slope = b_linear
        post_slope = b_linear
        pct_slope_shift = 0.0

        if best_c is not None and best_params is not None and r2_gain >= _MIN_R2_GAIN:
            # Degrees of freedom: full model has 3 params, reduced has 2
            df_diff = 1
            df_residual = n - 3
            if df_residual > 0 and best_rss > 1e-12:
                f_stat = float(((rss_linear - best_rss) / df_diff) / (best_rss / df_residual))
                p_val = float(1.0 - stats.f.cdf(f_stat, df_diff, df_residual))

            if p_val < _ALPHA:
                has_breakpoint = True
                pre_slope = best_params[1]
                post_slope = best_params[1] + best_params[2]
                if abs(pre_slope) > 1e-6:
                    pct_slope_shift = float((post_slope - pre_slope) / abs(pre_slope) * 100.0)
                else:
                    pct_slope_shift = float(post_slope * 100.0)

        # Format summary
        if has_breakpoint and best_c is not None:
            summary = (
                f"Identified significant non-linear tipping point in '{feature_column}' at {best_c:g} "
                f"affecting '{target_column}': slope shifts from {pre_slope:.4g} to {post_slope:.4g} "
                f"({pct_slope_shift:+.1f}%, F={f_stat:.2f}, p={p_val:.3g}, ΔR²={r2_gain:+.3f}, n={n})."
            )
        else:
            summary = (
                f"Evaluated relationship between '{feature_column}' and '{target_column}' across {n} rows: "
                f"relationship is continuous linear (slope={b_linear:.4g}, R²={r2_linear:.3f}) with no "
                f"statistically significant tipping point or structural break (p={p_val:.3g})."
            )

        return {
            "summary": summary,
            "feature_column": feature_column,
            "target_column": target_column,
            "has_breakpoint": has_breakpoint,
            "breakpoint": best_c if has_breakpoint else None,
            "pre_breakpoint_slope": pre_slope,
            "post_breakpoint_slope": post_slope,
            "slope_shift_pct": pct_slope_shift,
            "linear_slope": b_linear,
            "linear_r2": round(r2_linear, 4),
            "piecewise_r2": round(r2_piecewise, 4),
            "r2_gain": round(r2_gain, 4),
            "f_statistic": round(f_stat, 3),
            "p_value": round(p_val, 4),
            "sample_size": n,
        }

    def findings(
        self,
        output: dict[str, Any],
        profile: DatasetProfile | None = None,
        metadata: DatasetMetadata | None = None,
    ) -> list[Finding]:
        if not output.get("has_breakpoint") or output.get("breakpoint") is None:
            return []

        feat = output["feature_column"]
        target = output["target_column"]
        c = output["breakpoint"]
        pre_m = output["pre_breakpoint_slope"]
        post_m = output["post_breakpoint_slope"]
        shift_pct = output.get("slope_shift_pct", 0.0)
        p_val = output.get("p_value", 0.0)
        r2_gain = output.get("r2_gain", 0.0)
        n = output.get("sample_size", 0)

        headline = (
            f"Critical threshold at {feat} = {c:g}: outcome '{target}' slope shifts by {shift_pct:+.1f}% "
            f"(from {pre_m:.3g} to {post_m:.3g}, ΔR²={r2_gain:+.2f}, p={p_val:.3g})."
        )
        detail = (
            f"Piecewise regression identified a statistically significant structural breakpoint in {feat} at {c:g}. "
            f"Prior to this threshold the marginal relationship with {target} is {pre_m:.4g}, while beyond it the "
            f"slope transitions to {post_m:.4g} (F={output.get('f_statistic', 0.0):.2f}, n={n})."
        )

        finding = Finding(
            finding_id=f"threshold_{feat}_{target}",
            kind="threshold",
            headline=headline,
            detail=detail,
            evidence={
                "feature": feat,
                "target": target,
                "breakpoint": c,
                "pre_slope": pre_m,
                "post_slope": post_m,
                "slope_shift_pct": shift_pct,
                "p_value": p_val,
                "r2_gain": r2_gain,
                "n": n,
            },
            source_tool=self.name,
            measure=target,
            dimension=feat,
            effect=abs(shift_pct) / 100.0,
            effect_kind="pct",
            p_value=p_val,
            confidence=0.85 if p_val < 0.01 else 0.70,
            surprise=0.75,
            objective_fit=0.80,
            chart_hint={
                "kind": "scatter",
                "x": feat,
                "y": target,
                "title": f"Tipping Point in {feat} vs {target}",
            },
        )
        return [finding]
