"""
Variable-aware statistical methods tool.

FutureScope Phase 3:
Provides scale-appropriate statistical methods:
- Circular statistics for periodic / angular variables (wind, time-of-day).
- Compositional data transforms (Centered Log-Ratio) for shares summing to 1 or 100.
- Benford's Law and digit heaping forensics for accounting, financial, and survey data.
- Non-parametric ordinal effect sizes (Cliff's delta).
- Count model diagnostics (Poisson vs Negative Binomial vs zero-inflation).
"""
from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
import pandas as pd

from src.core.findings import Finding
from src.core.stats_utils import (
    benford_analysis,
    circular_statistics,
    cliffs_delta,
    compositional_clr,
    count_target_diagnostics,
    digit_heaping_test,
    robust_location_dispersion,
)
from src.tools.base import BaseTool, ToolExecutionError
from src.tools.data_processing import _read_df

if TYPE_CHECKING:
    from src.core.memory import DatasetMetadata
    from src.core.profiler import DatasetProfile


class VariableScaleAnalysisTool(BaseTool):
    """
    Scale-appropriate statistical analysis for circular, compositional, count,
    ordinal, and heavy-tailed data, plus Benford and digit-heaping forensics.
    """

    name = "variable_scale_analysis"
    description = (
        "Run scale-appropriate statistical analysis for non-standard variables: "
        "circular statistics for periodic/angular data (wind direction, time-of-day), "
        "centered log-ratio (CLR) for compositional data (parts of a whole), "
        "Benford's Law analysis for financial/accounting digit integrity, "
        "digit heaping/rounding tests (Whipple's index), ordinal rank effect size (Cliff's delta), "
        "and count target diagnostics (Poisson vs Negative Binomial vs zero-inflation)."
    )

    requires_context: ClassVar[dict[str, str]] = {"target_column": "target_column"}

    def applies_to(self, profile: DatasetProfile | None, metadata: DatasetMetadata | None) -> float:
        if profile is None or profile.row_count < 10:
            return 0.0
        # Check if circular, count, financial, or compositional candidates exist
        col_names = [c.name.lower() for c in profile.columns]
        circular_hints = {"angle", "deg", "degree", "direction", "bearing", "hour", "wind_dir"}
        financial_hints = {"amount", "price", "cost", "revenue", "expense", "spend", "sales", "balance"}

        score = 0.3
        if any(any(h in c for h in circular_hints) for c in col_names):
            score = max(score, 0.8)
        if any(any(h in c for h in financial_hints) for c in col_names):
            score = max(score, 0.7)
        if getattr(metadata, "target_column", None):
            score = max(score, 0.6)
        return score

    def default_params(
        self, profile: DatasetProfile | None, metadata: DatasetMetadata | None
    ) -> dict[str, Any]:
        return {"analysis_type": "auto"}

    def get_schema(self) -> dict[str, Any]:
        return {
            "file_path": {
                "type": "string",
                "description": "Path to the dataset file.",
                "required": True,
            },
            "analysis_type": {
                "type": "string",
                "description": "Type of variable analysis: 'auto', 'circular', 'compositional', 'benford', 'heaping', 'ordinal', 'count_diagnostics', 'robust'.",
                "required": False,
            },
            "columns": {
                "type": "list[str]",
                "description": "Specific column(s) to analyze.",
                "required": False,
            },
            "group_column": {
                "type": "string",
                "description": "Grouping column for ordinal/group comparisons.",
                "required": False,
            },
            "target_column": {
                "type": "string",
                "description": "Target column for count model diagnostics.",
                "required": False,
            },
            "high": {
                "type": "float",
                "description": "Upper period bound for circular statistics (e.g. 360 for degrees, 24 for hours).",
                "required": False,
            },
        }

    def execute(  # type: ignore[override]
        self,
        file_path: str,
        analysis_type: str = "auto",
        columns: list[str] | str | None = None,
        group_column: str | None = None,
        target_column: str | None = None,
        high: float = 360.0,
        **_: Any,
    ) -> dict[str, Any]:
        df = _read_df(file_path)
        if df.empty:
            raise ToolExecutionError("Dataset is empty.")

        if isinstance(columns, str):
            columns = [c.strip() for c in columns.split(",") if c.strip()]

        findings: list[Finding] = []
        results: dict[str, Any] = {}
        executed_types: list[str] = []

        # 1. Circular Statistics
        run_circular = analysis_type == "circular" or (
            analysis_type == "auto"
            and any(
                re.search(r"(?:^|_)(angle|deg|degree|direction|bearing|hour|wind_dir)(?:$|_)", c, re.I)
                for c in (columns or df.columns)
            )
        )
        if run_circular:
            circ_cols = [
                c for c in (columns or df.columns)
                if re.search(r"(?:^|_)(angle|deg|degree|direction|bearing|hour|wind_dir)(?:$|_)", c, re.I)
                and pd.api.types.is_numeric_dtype(df[c])
            ]
            if circ_cols:
                executed_types.append("circular")
                circ_results = {}
                for c in circ_cols:
                    col_high = 24.0 if "hour" in c.lower() else high
                    c_stat = circular_statistics(df[c], high=col_high)
                    circ_results[c] = c_stat
                    if not c_stat["is_uniform"] and c_stat["circular_mean"] is not None:
                        findings.append(
                            Finding(
                                finding_id=f"circ_{c}",
                                kind="test",
                                headline=f"Significant directional clustering in '{c}' around {c_stat['circular_mean']} (p={c_stat['rayleigh_p']})",
                                detail=f"Resultant length R={c_stat['resultant_length']}, circular dispersion={c_stat['circular_dispersion']}.",
                                importance=0.75,
                                effect=float(c_stat["resultant_length"]),
                                effect_kind="r",
                                p_value=c_stat["rayleigh_p"],
                                evidence=c_stat,
                                source_tool=self.name,
                            )
                        )
                results["circular"] = circ_results

        # 2. Compositional Analysis
        run_comp = analysis_type == "compositional" or (
            analysis_type == "auto"
            and len(df.select_dtypes(include=[np.number]).columns) >= 3
        )
        if run_comp:
            num_cols = list(df.select_dtypes(include=[np.number]).columns)
            # Check if any subset sums to ~100 or ~1
            if len(num_cols) >= 3:
                row_sums = df[num_cols].sum(axis=1).dropna()
                is_pct = bool(np.isclose(row_sums.median(), 100.0, atol=5.0))
                is_prop = bool(np.isclose(row_sums.median(), 1.0, atol=0.05))
                if (is_pct or is_prop) or analysis_type == "compositional":
                    executed_types.append("compositional")
                    clr_df = compositional_clr(df, num_cols)
                    results["compositional"] = {
                        "columns": num_cols,
                        "clr_columns": list(clr_df.columns),
                        "row_sum_median": float(row_sums.median()),
                        "scale": "percentage" if is_pct else ("proportion" if is_prop else "custom"),
                    }
                    findings.append(
                        Finding(
                            finding_id="comp_parts",
                            kind="test",
                            headline=f"Detected compositional parts of a whole ({', '.join(num_cols[:4])}…) summing to {round(float(row_sums.median()), 1)}",
                            detail="Applied Centered Log-Ratio (CLR) transform to prevent spurious negative correlation bias.",
                            importance=0.7,
                            evidence={"columns": num_cols, "median_sum": float(row_sums.median())},
                            source_tool=self.name,
                        )
                    )

        # 3. Benford's Law Analysis
        run_benford = analysis_type == "benford" or (
            analysis_type == "auto"
            and any(
                re.search(r"\b(amount|price|cost|revenue|expense|spend|sales|balance)\b", c, re.I)
                for c in (columns or df.columns)
            )
        )
        if run_benford:
            fin_cols = [
                c for c in (columns or df.columns)
                if pd.api.types.is_numeric_dtype(df[c])
                and (
                    analysis_type == "benford"
                    or re.search(r"\b(amount|price|cost|revenue|expense|spend|sales|balance)\b", c, re.I)
                )
            ]
            if fin_cols:
                executed_types.append("benford")
                benford_res = {}
                for c in fin_cols:
                    b_stat = benford_analysis(df[c])
                    benford_res[c] = b_stat
                    if b_stat["conformity"] == "non_conforming" and b_stat["mad"] is not None:
                        findings.append(
                            Finding(
                                finding_id=f"benford_flag_{c}",
                                kind="test",
                                headline=f"Column '{c}' deviates from Benford's Law (MAD={b_stat['mad']}, p={b_stat['p_value']})",
                                detail="First-digit distribution shows non-conformity. Flagged for review (not an accusation of anomaly).",
                                importance=0.8,
                                effect=float(b_stat["mad"]),
                                effect_kind="pct",
                                p_value=b_stat["p_value"],
                                evidence=b_stat,
                                source_tool=self.name,
                            )
                        )
                    elif b_stat["conformity"] in ("close", "acceptable"):
                        findings.append(
                            Finding(
                                finding_id=f"benford_pass_{c}",
                                kind="test",
                                headline=f"Column '{c}' conforms to Benford's Law (MAD={b_stat['mad']}, {b_stat['conformity']} conformity)",
                                detail="First-digit distribution matches natural logarithmic proportions.",
                                importance=0.5,
                                effect=float(b_stat["mad"]) if b_stat["mad"] is not None else 0.0,
                                effect_kind="pct",
                                evidence=b_stat,
                                source_tool=self.name,
                            )
                        )
                results["benford"] = benford_res

        # 4. Digit Heaping / Rounding
        run_heaping = analysis_type == "heaping" or (
            analysis_type == "auto"
            and any(
                re.search(r"\b(age|score|weight|height|count|qty|quantity)\b", c, re.I)
                for c in (columns or df.columns)
            )
        )
        if run_heaping:
            heap_cols = [
                c for c in (columns or df.columns)
                if pd.api.types.is_numeric_dtype(df[c])
                and (
                    analysis_type == "heaping"
                    or re.search(r"\b(age|score|weight|height|count|qty|quantity)\b", c, re.I)
                )
            ]
            if heap_cols:
                executed_types.append("heaping")
                heap_res = {}
                for c in heap_cols:
                    h_stat = digit_heaping_test(df[c])
                    heap_res[c] = h_stat
                    if h_stat["heaping_detected"]:
                        findings.append(
                            Finding(
                                finding_id=f"heaping_{c}",
                                kind="test",
                                headline=f"Digit heaping / rounding detected in '{c}' (Whipple index={h_stat['whipples_index']})",
                                detail=f"Excess of terminal digits 0 and 5 observed. {h_stat['recommendation']}.",
                                importance=0.7,
                                effect=float(h_stat["whipples_index"]) if h_stat["whipples_index"] else 0.0,
                                effect_kind="pct",
                                evidence=h_stat,
                                source_tool=self.name,
                            )
                        )
                results["heaping"] = heap_res

        # 5. Count Target Diagnostics
        target = target_column or (columns[0] if columns and len(columns) == 1 else None)
        if target and target in df.columns and (analysis_type in ("count_diagnostics", "auto")):
            cnt_stat = count_target_diagnostics(df[target])
            if cnt_stat["is_count"]:
                executed_types.append("count_diagnostics")
                results["count_diagnostics"] = cnt_stat
                findings.append(
                    Finding(
                        finding_id=f"count_diag_{target}",
                        kind="test",
                        headline=f"Target '{target}' is a count variable: recommended model is '{cnt_stat['recommended_model']}'",
                        detail=(
                            f"Mean={cnt_stat['mean']}, variance={cnt_stat['variance']}, "
                            f"dispersion ratio={cnt_stat['dispersion_ratio']}, zero proportion={cnt_stat['zero_proportion']}."
                        ),
                        importance=0.75,
                        effect=float(cnt_stat["dispersion_ratio"]),
                        effect_kind="lift",
                        evidence=cnt_stat,
                        source_tool=self.name,
                    )
                )

        # 6. Ordinal Effect Size (Cliff's Delta)
        if (analysis_type == "ordinal" or group_column) and group_column and group_column in df.columns:
            target_cols = [c for c in (columns or df.select_dtypes(include=[np.number]).columns) if c != group_column]
            groups = df[group_column].dropna().unique()
            if len(groups) == 2 and target_cols:
                executed_types.append("ordinal")
                ord_res = {}
                g1, g2 = groups[0], groups[1]
                for c in target_cols:
                    x = df[df[group_column] == g1][c]
                    y = df[df[group_column] == g2][c]
                    cd_stat = cliffs_delta(x, y)
                    ord_res[c] = cd_stat
                    if cd_stat["magnitude"] in ("medium", "large"):
                        findings.append(
                            Finding(
                                finding_id=f"ordinal_{c}",
                                kind="test",
                                headline=f"Substantial ordinal difference in '{c}' between {g1} and {g2} (Cliff's delta={cd_stat['delta']}, {cd_stat['magnitude']})",
                                detail=f"Non-parametric rank test p-value: {cd_stat['p_value']}.",
                                importance=0.8,
                                effect=float(cd_stat["delta"]) if cd_stat["delta"] is not None else 0.0,
                                effect_kind="rank_biserial",
                                p_value=cd_stat["p_value"],
                                evidence=cd_stat,
                                source_tool=self.name,
                            )
                        )
                results["ordinal"] = ord_res

        # 7. Robust Location & Dispersion
        run_robust = analysis_type in ("robust", "auto")
        if run_robust:
            num_cols = list(df.select_dtypes(include=[np.number]).columns)
            robust_res = {}
            for c in num_cols:
                r_stat = robust_location_dispersion(df[c])
                if r_stat.get("is_heavy_tailed"):
                    robust_res[c] = r_stat
                    findings.append(
                        Finding(
                            finding_id=f"robust_{c}",
                            kind="test",
                            headline=f"Heavy tails in '{c}' (skew={r_stat['skewness']}, kurtosis={r_stat['kurtosis']}): use median/IQR",
                            detail=f"Median={r_stat['median']}, IQR={r_stat['iqr']}, 5% trimmed mean={r_stat['trimmed_mean_5pct']}.",
                            importance=0.65,
                            effect=abs(float(r_stat["skewness"])),
                            effect_kind="lift",
                            evidence=r_stat,
                            source_tool=self.name,
                        )
                    )
            if robust_res:
                executed_types.append("robust")
                results["robust"] = robust_res

        summary = (
            f"Variable-aware scale analysis executed for: {', '.join(executed_types) or 'general'}. "
            f"Identified {len(findings)} finding(s)."
        )

        return {
            "summary": summary,
            "executed_types": executed_types,
            "results": results,
            "findings": [f.to_dict() for f in findings],
            "finding_payloads": [f.to_dict() for f in findings],
        }
