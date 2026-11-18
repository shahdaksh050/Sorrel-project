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
from src.core.profiler import is_transaction_value_name, is_unit_rate_name


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
        "hba1c", "ldl", "tachycardia",
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
        _bounded("hba1c"): LimitSpec(
            threshold=6.5,
            direction="max",
            unit="%",
            standard_name="ADA Diabetic Diagnostic HbA1c Threshold",
            description="Glycated hemoglobin above diabetic threshold",
        ),
        _bounded("ldl"): LimitSpec(
            threshold=130.0,
            direction="max",
            unit="mg/dL",
            standard_name="AHA Borderline High LDL Cholesterol Threshold",
            description="Low-density lipoprotein above cardiovascular risk threshold",
        ),
        _bounded("bmi"): LimitSpec(
            threshold=30.0,
            direction="max",
            unit="kg/m²",
            standard_name="WHO Obesity Class I Threshold",
            description="Body Mass Index in obesity category",
        ),
        _bounded("heart_rate"): LimitSpec(
            threshold=100.0,
            direction="max",
            unit="bpm",
            standard_name="Clinical Tachycardia Threshold",
            description="Resting heart rate above adult normal range",
        ),
    },
)

_RETAIL_ECOMMERCE_PACK = DomainPack(
    name="retail_ecommerce",
    display_name="Retail & E-Commerce Operations",
    activation_keywords=[
        "retail", "order", "sales", "revenue", "basket", "aov", "customer",
        "product", "sku", "ecommerce", "e-commerce", "quantity", "price", "checkout",
    ],
    limits={
        _bounded("discount"): LimitSpec(
            threshold=0.50,
            direction="max",
            unit="ratio",
            standard_name="Commercial Discount Safety Ceiling",
            description="Promotional discount depth exceeding 50% margin-erosion threshold",
        ),
        _bounded("return_rate"): LimitSpec(
            threshold=0.15,
            direction="max",
            unit="ratio",
            standard_name="E-Commerce Benchmark Return Ceiling",
            description="Product return rate exceeding 15% operational risk threshold",
        ),
    },
)

_SAAS_SUBSCRIPTION_PACK = DomainPack(
    name="saas_subscription",
    display_name="SaaS & Recurring Revenue Metrics",
    activation_keywords=[
        "churn", "mrr", "arr", "subscription", "plan", "saas", "renewal",
        "retention", "ltv", "cac", "tenure", "seats", "contract",
    ],
    limits={
        _bounded("churn_rate") + "|" + _bounded("churn"): LimitSpec(
            threshold=0.05,
            direction="max",
            unit="ratio",
            standard_name="SaaS Industry Median Monthly Churn Ceiling",
            description="Monthly customer churn rate exceeding 5% high-risk threshold",
        ),
    },
)

BUILTIN_PACKS: list[DomainPack] = [
    _AIR_QUALITY_PACK,
    _FINANCE_PACK,
    _HEALTHCARE_PACK,
    _RETAIL_ECOMMERCE_PACK,
    _SAAS_SUBSCRIPTION_PACK,
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
                                "n": n_total,
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
                        downside_rets = rets[rets < 0]
                        downside_std = float(downside_rets.std()) if len(downside_rets) > 1 else std_ret
                        ann_sortino = (mean_ret / downside_std) * np.sqrt(252) if downside_std > 0 else ann_sharpe
                        var_95 = abs(float(np.percentile(rets, 5)))
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
                                    f"Sortino={round(ann_sortino, 2)}, Max Drawdown={round(max_dd * 100, 1)}%"
                                ),
                                detail=(
                                    f"Annualized volatility: {round(std_ret * np.sqrt(252) * 100, 1)}%, "
                                    f"Historical 95% Daily VaR: {round(var_95 * 100, 2)}%."
                                ),
                                importance=0.8,
                                effect=max_dd,
                                effect_kind="pct",
                                evidence={
                                    "sharpe": round(ann_sharpe, 3),
                                    "sortino": round(ann_sortino, 3),
                                    "max_drawdown": round(max_dd, 4),
                                    "volatility": round(std_ret * np.sqrt(252), 4),
                                    "var_95": round(var_95, 4),
                                },
                                source_tool="domain_pack",
                            )
                        )

    # Evaluate Retail / E-Commerce KPIs if active
    if active_pack.name == "retail_ecommerce":
        num_cols = list(df.select_dtypes(include=[np.number]).columns)

        # Tier 1: Explicit transaction value
        val_col = next((c for c in num_cols if is_transaction_value_name(str(c))), None)
        if not val_col:
            val_col = next(
                (c for c in num_cols if re.search(r"(?:revenue|sales|total_amount|spend|line_total|total_price|gross)", str(c), re.I)
                 and not re.search(r"(?:unit|rate|price_per)", str(c), re.I)),
                None,
            )

        # Candidate unit rate and quantity columns
        rate_col = next((c for c in num_cols if is_unit_rate_name(str(c)) or re.search(r"(?:unit_price|unitprice|price)", str(c), re.I)), None)
        qty_col = next((c for c in num_cols if re.search(r"(?:quantity|qty|units|volume)", str(c), re.I)), None)
        id_col = next((c for c in df.columns if re.search(r"(?:order|basket|invoice|transaction)", str(c), re.I)), None)
        cust_col = next((c for c in df.columns if re.search(r"(?:customer|client|user|buyer|account)", str(c), re.I)), None)

        is_synthesized = False
        if val_col:
            clean_amounts = pd.to_numeric(df[val_col], errors="coerce").dropna()
            if not clean_amounts.empty:
                tot_rev: float | None = float(clean_amounts.sum())
                amount_col: str | None = str(val_col)
            else:
                tot_rev = None
                amount_col = None
        elif rate_col and qty_col:
            clean_amounts = (
                pd.to_numeric(df[rate_col], errors="coerce")
                * pd.to_numeric(df[qty_col], errors="coerce")
            ).dropna()
            if not clean_amounts.empty:
                tot_rev = float(clean_amounts.sum())
                amount_col = f"{qty_col} * {rate_col}"
                is_synthesized = True
            else:
                clean_amounts = pd.to_numeric(df[rate_col], errors="coerce").dropna()
                tot_rev = None
                amount_col = str(rate_col) if not clean_amounts.empty else None
        elif rate_col:
            clean_amounts = pd.to_numeric(df[rate_col], errors="coerce").dropna()
            tot_rev = None
            amount_col = str(rate_col) if not clean_amounts.empty else None
        else:
            clean_amounts = pd.Series(dtype=float)
            tot_rev = None
            amount_col = None

        if tot_rev is not None and amount_col:
            if id_col:
                n_orders = max(1, df[id_col].nunique())
                aov = round(tot_rev / n_orders, 2)
                calc_note = f" (synthesized from {amount_col})" if is_synthesized else ""
                findings.append(
                    Finding(
                        finding_id=f"retail_aov_{val_col or rate_col}",
                        kind="financial",
                        headline=f"Retail KPI for '{amount_col}': Average Order Value (AOV) is ${aov:,.2f} across {n_orders:,} orders{calc_note}.",
                        detail=f"Total gross volume: ${tot_rev:,.2f} across {n_orders:,} orders.",
                        importance=0.82,
                        effect=aov,
                        effect_kind="numeric",
                        evidence={"aov": aov, "total_revenue": tot_rev, "order_count": n_orders, "amount_column": amount_col},
                        source_tool="domain_pack",
                    )
                )
        elif rate_col and not clean_amounts.empty:
            avg_unit_price = round(float(clean_amounts.mean()), 2)
            findings.append(
                Finding(
                    finding_id=f"retail_unit_rate_{rate_col}",
                    kind="financial",
                    headline=f"Retail Pricing KPI for '{rate_col}': Average Unit Price is ${avg_unit_price:,.2f}.",
                    detail=f"Mean unit rate across {len(clean_amounts):,} item records. Gross order volume requires a quantity or transaction total column.",
                    importance=0.75,
                    effect=avg_unit_price,
                    effect_kind="numeric",
                    evidence={"avg_unit_price": avg_unit_price, "rate_column": rate_col},
                    source_tool="domain_pack",
                )
            )

        if cust_col:
            cust_order_counts = df.groupby(cust_col).size()
            n_cust = len(cust_order_counts)
            n_repeat = int((cust_order_counts >= 2).sum())
            repeat_rate = round(float(n_repeat / max(1, n_cust)), 3)
            repeat_pct = round(repeat_rate * 100, 1)
            findings.append(
                Finding(
                    finding_id=f"retail_repeat_cust_{cust_col}",
                    kind="segment_lift",
                    headline=f"Customer Retention: {repeat_pct}% repeat purchase rate ({n_repeat:,} of {n_cust:,} customers).",
                    detail=f"{n_repeat:,} customers made 2 or more purchases.",
                    importance=0.80,
                    effect=repeat_rate,
                    effect_kind="pct",
                    evidence={"repeat_rate": repeat_rate, "repeat_customers": n_repeat, "total_customers": n_cust},
                    source_tool="domain_pack",
                )
            )
        if amount_col and not clean_amounts.empty:
            neg_rows = int((clean_amounts < 0).sum())
            if neg_rows > 0:
                neg_rate = round(float(neg_rows / len(clean_amounts)), 3)
                findings.append(
                    Finding(
                        finding_id=f"retail_refunds_{val_col or rate_col}",
                        kind="data_integrity",
                        headline=f"Commercial Integrity: {neg_rows:,} negative transactions ({round(neg_rate * 100, 1)}%) in '{amount_col}' representing refunds/reversals.",
                        detail=f"Quarantined/flagged return transactions totaling ${abs(float(clean_amounts[clean_amounts < 0].sum())):,.2f}.",
                        importance=0.75,
                        effect=neg_rate,
                        effect_kind="pct",
                        evidence={"negative_count": neg_rows, "negative_rate": neg_rate},
                        source_tool="domain_pack",
                    )
                )

    # Evaluate SaaS Subscription KPIs if active
    if active_pack.name == "saas_subscription":
        churn_col = next((c for c in df.columns if re.search(r"(?:churn|cancelled|canceled)", str(c), re.I)), None)
        tenure_col = next((c for c in df.columns if re.search(r"(?:tenure|duration|months_active)", str(c), re.I) and pd.api.types.is_numeric_dtype(df[c])), None)
        mrr_col = next((c for c in df.columns if re.search(r"(?:mrr|arr|monthly_charges|recurring_revenue)", str(c), re.I) and pd.api.types.is_numeric_dtype(df[c])), None)

        if churn_col:
            churn_s = pd.to_numeric(df[churn_col], errors="coerce").dropna()
            if len(churn_s) >= 10:
                churn_rate = float(churn_s.mean())
                churn_pct = round(churn_rate * 100, 1)
                est_lifetime = round(1.0 / churn_rate, 1) if churn_rate > 0.005 else None
                detail_str = f"Observed {int(churn_s.sum()):,} churn events out of {len(churn_s):,} subscriptions."
                if est_lifetime:
                    detail_str += f" Implied customer lifetime: {est_lifetime} cycles."
                if tenure_col:
                    med_tenure = round(float(df[tenure_col].median()), 1)
                    detail_str += f" Median tenure: {med_tenure}."
                findings.append(
                    Finding(
                        finding_id=f"saas_churn_{churn_col}",
                        kind="driver",
                        headline=f"SaaS Cohort Performance: {churn_pct}% observed churn rate.",
                        detail=detail_str,
                        importance=0.88,
                        effect=churn_rate,
                        effect_kind="pct",
                        evidence={
                            "churn_rate": round(churn_rate, 4),
                            "churned_count": int(churn_s.sum()),
                            "total_count": len(churn_s),
                            "implied_lifetime": est_lifetime,
                        },
                        source_tool="domain_pack",
                    )
                )
        if mrr_col:
            mrr_s = df[mrr_col].dropna()
            avg_mrr = round(float(mrr_s.mean()), 2)
            tot_mrr = round(float(mrr_s.sum()), 2)
            findings.append(
                Finding(
                    finding_id=f"saas_mrr_{mrr_col}",
                    kind="financial",
                    headline=f"SaaS Revenue: Average Monthly Recurring Revenue (ARPU/MRR) is ${avg_mrr:,.2f}.",
                    detail=f"Total monthly revenue base: ${tot_mrr:,.2f} across {len(mrr_s):,} accounts.",
                    importance=0.84,
                    effect=avg_mrr,
                    effect_kind="numeric",
                    evidence={"average_mrr": avg_mrr, "total_mrr": tot_mrr, "n": len(mrr_s)},
                    source_tool="domain_pack",
                )
            )

    return findings
