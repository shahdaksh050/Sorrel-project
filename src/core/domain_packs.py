"""
Domain Packs: Contextual benchmarks, regulatory limits, and domain KPIs.

FutureScope Phase 5 (FutureScope.md §5.7):
Turns raw metrics into actionable domain findings:
- Evaluates domain-specific regulatory and reference thresholds (e.g. WHO air quality limits, clinical reference ranges).
- Computes sector-standard KPIs (e.g. Sharpe ratio, max drawdown, AOV).
- Generates rich, contextual findings explaining exceedance rates and benchmark comparisons.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.core.findings import Finding


@dataclass
class LimitSpec:
    threshold: float
    direction: str  # "max" or "min"
    unit: str
    standard_name: str
    description: str


@dataclass
class DomainPack:
    name: str
    display_name: str
    activation_keywords: list[str]
    limits: dict[str, LimitSpec] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Built-in Domain Packs
# ---------------------------------------------------------------------------

#: Word-boundary helper for limit-key regexes. `\b` alone is unreliable here
#: because real-world sensor column names carry unit/method suffixes in
#: parentheses with no underscore separator — e.g. the project's own
#: reference dataset (AirQualityUCI.csv) names its columns "CO(GT)",
#: "NO2(GT)", "C6H6(GT)". A boundary of "start/end or a literal underscore"
#: (the original pattern) requires the token to be immediately followed by
#: "_" or end-of-string, so it never matches "CO(GT)" at all — silently
#: disabling every air-quality limit on the exact dataset this feature was
#: built to demonstrate.
#:
#: The fix is asymmetric on purpose: the token may be *followed* by any
#: non-alphanumeric character (so "CO(GT)" matches), but may only be
#: *preceded* by start-of-string, "_", space or "-" — never "(". That
#: excludes columns where the token is a parenthetical qualifier on a
#: different base name, which AirQualityUCI also has right next to the real
#: ones: "PT08.S1(CO)" and "PT08.S4(NO2)" are tin-oxide sensor responses
#: (unitless resistance-ish values in the hundreds/thousands), not CO/NO2
#: concentrations on the mg/m3 or ug/m3 scale these limits assume — matching
#: them would produce a fabricated exceedance claim against the wrong unit.
def _bounded(token: str) -> str:
    return rf"(?:^|[_\s-]){token}(?:$|[^a-z0-9])"


_AIR_QUALITY_PACK = DomainPack(
    name="air_quality",
    display_name="Air Quality & Environmental Health",
    activation_keywords=["pollut", "air", "quality", "co", "no2", "nox", "pm2.5", "pm10", "o3", "benzene", "c6h6"],
    limits={
        _bounded("co"): LimitSpec(
            threshold=10.0,
            direction="max",
            unit="mg/m³",
            standard_name="WHO Air Quality Guideline (8-hour)",
            description="Carbon Monoxide exceedance above health guideline",
        ),
        _bounded("no2"): LimitSpec(
            threshold=200.0,
            direction="max",
            unit="µg/m³",
            standard_name="WHO Air Quality Guideline (1-hour)",
            description="Nitrogen Dioxide exceedance above health guideline",
        ),
        _bounded(r"c6h6") + "|benzene": LimitSpec(
            threshold=5.0,
            direction="max",
            unit="µg/m³",
            standard_name="EU Annual Air Quality Standard",
            description="Benzene exceedance above annual exposure target",
        ),
        _bounded(r"pm2\.?5"): LimitSpec(
            threshold=15.0,
            direction="max",
            unit="µg/m³",
            standard_name="WHO Air Quality Guideline (24-hour)",
            description="Fine particulate matter exceedance",
        ),
        _bounded("pm10"): LimitSpec(
            threshold=45.0,
            direction="max",
            unit="µg/m³",
            standard_name="WHO Air Quality Guideline (24-hour)",
            description="Coarse particulate matter exceedance",
        ),
        _bounded("o3") + "|ozone": LimitSpec(
            threshold=100.0,
            direction="max",
            unit="µg/m³",
            standard_name="WHO Air Quality Guideline (8-hour)",
            description="Ozone exceedance above health guideline",
        ),
    },
)

_FINANCE_PACK = DomainPack(
    name="finance",
    display_name="Financial & Portfolio Analytics",
    activation_keywords=["return", "stock", "price", "portfolio", "sharpe", "drawdown", "volatility", "asset"],
    limits={},
)

_HEALTHCARE_PACK = DomainPack(
    name="healthcare",
    display_name="Clinical & Health Indicators",
    activation_keywords=[
        "patient", "glucose", "blood_pressure", "blood pressure", "systolic",
        "diastolic", "bmi", "cholesterol", "heart_rate", "clinical", "hypertension",
    ],
    limits={
        _bounded("systolic"): LimitSpec(
            threshold=140.0,
            direction="max",
            unit="mmHg",
            standard_name="AHA/ACC Hypertension Stage 2",
            description="Systolic blood pressure above hypertension threshold",
        ),
        _bounded("diastolic"): LimitSpec(
            threshold=90.0,
            direction="max",
            unit="mmHg",
            standard_name="AHA/ACC Hypertension Stage 2",
            description="Diastolic blood pressure above hypertension threshold",
        ),
        _bounded("glucose"): LimitSpec(
            threshold=126.0,
            direction="max",
            unit="mg/dL",
            standard_name="ADA Fasting Glucose Diagnostic Threshold",
            description="Fasting blood glucose above diabetic threshold",
        ),
        _bounded("bmi"): LimitSpec(
            threshold=30.0,
            direction="max",
            unit="kg/m²",
            standard_name="WHO Obesity Class I Threshold",
            description="Body Mass Index in obesity category",
        ),
    },
)

BUILTIN_PACKS: list[DomainPack] = [
    _AIR_QUALITY_PACK,
    _FINANCE_PACK,
    _HEALTHCARE_PACK,
]


def detect_domain_pack(
    df: pd.DataFrame,
    objective: str = "",
) -> DomainPack | None:
    """
    Select the most applicable domain pack based on objective keywords
    and column name matching.
    """
    cols_text = " ".join(df.columns).lower()
    obj_text = objective.lower()
    combined_text = f"{obj_text} {cols_text}"

    best_pack: DomainPack | None = None
    best_score = 0

    for pack in BUILTIN_PACKS:
        score = sum(1 for kw in pack.activation_keywords if re.search(r"\b" + re.escape(kw) + r"\b", combined_text))
        if score > best_score and score >= 2:
            best_score = score
            best_pack = pack

    return best_pack


def evaluate_domain_pack(
    df: pd.DataFrame,
    pack: DomainPack | None = None,
    objective: str = "",
) -> list[Finding]:
    """
    Evaluate domain limits and KPI benchmarks against dataframe columns.
    Returns domain-specific Finding objects with exceedance rates and context.
    """
    active_pack = pack or detect_domain_pack(df, objective)
    if active_pack is None:
        return []

    findings: list[Finding] = []

    # Evaluate limits
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        col_clean = df[col].dropna()
        if col_clean.empty:
            continue

        for pattern, spec in active_pack.limits.items():
            if re.search(pattern, str(col), re.I):
                n_total = len(col_clean)
                if spec.direction == "max":
                    violating = col_clean[col_clean > spec.threshold]
                else:
                    violating = col_clean[col_clean < spec.threshold]

                count_viol = len(violating)
                if count_viol > 0:
                    rate = float(count_viol / n_total)
                    rate_pct = round(rate * 100.0, 1)
                    max_val = round(float(violating.max()), 2)
                    findings.append(
                        Finding(
                            finding_id=f"pack_{active_pack.name}_{col}_exceed",
                            kind="test",
                            headline=(
                                f"'{col}' exceeded {spec.standard_name} ({spec.threshold} {spec.unit}) "
                                f"in {rate_pct}% of observations (peak: {max_val} {spec.unit})"
                            ),
                            detail=(
                                f"{count_viol} of {n_total} readings violated the guideline threshold. "
                                f"{spec.description}."
                            ),
                            importance=0.85,
                            effect=rate,
                            effect_kind="pct",
                            evidence={
                                "column": str(col),
                                "threshold": spec.threshold,
                                "exceedance_rate": rate,
                                "violations": count_viol,
                                "total": n_total,
                                "peak_value": max_val,
                                "standard": spec.standard_name,
                            },
                            source_tool="domain_pack",
                        )
                    )

    # Evaluate Finance KPIs if active
    if active_pack.name == "finance":
        num_cols = list(df.select_dtypes(include=[np.number]).columns)
        for c in num_cols:
            if re.search(r"return|change|pct", c, re.I):
                rets = df[c].dropna()
                if len(rets) >= 20:
                    mean_ret = float(rets.mean())
                    std_ret = float(rets.std())
                    if std_ret > 0:
                        ann_sharpe = (mean_ret / std_ret) * np.sqrt(252)
                        cum_rets = (1.0 + rets).cumprod()
                        rolling_max = cum_rets.cummax()
                        drawdowns = (cum_rets - rolling_max) / rolling_max
                        max_dd = abs(float(drawdowns.min())) if not drawdowns.empty else 0.0

                        findings.append(
                            Finding(
                                finding_id=f"fin_kpi_{c}",
                                kind="financial",
                                headline=(
                                    f"Financial performance for '{c}': Annualized Sharpe={round(ann_sharpe, 2)}, "
                                    f"Max Drawdown={round(max_dd * 100, 1)}%"
                                ),
                                detail=f"Annualized volatility: {round(std_ret * np.sqrt(252) * 100, 1)}%.",
                                importance=0.8,
                                effect=max_dd,
                                effect_kind="pct",
                                evidence={
                                    "sharpe": round(ann_sharpe, 3),
                                    "max_drawdown": round(max_dd, 4),
                                    "volatility": round(std_ret * np.sqrt(252), 4),
                                },
                                source_tool="domain_pack",
                            )
                        )

    return findings
