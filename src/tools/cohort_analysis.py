"""
Cohort / Transaction Analysis Tool — Execution Layer.

Customer-behaviour measures for datasets the domain layer identified as
transactional (src/core/domains.py): RFM segmentation, repeat-purchase
behaviour, revenue concentration and basket composition.

Why a dedicated tool: a transaction log's rows are not independent
observations — they are repeated events per customer. Generic column
statistics therefore answer the wrong question ("what is the mean of
`amount`?" describes an average *line item*, not an average *customer*).
The questions a shop actually has are about customers and orders, which
means aggregating to those grains first. The domain layer is what
identifies which column carries which grain.

Recency is measured against the dataset's own last transaction date, not
today's date, so the result is reproducible and does not silently drift
as the file ages on disk.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from src.core.domains import domain_confidence, normalize_name, resolve_column
from src.core.findings import Finding
from src.tools.base import BaseTool, ToolExecutionError
from src.tools.data_processing import _read_df

if TYPE_CHECKING:
    from src.core.memory import DatasetMetadata
    from src.core.profiler import DatasetProfile

#: RFM uses quintiles when the customer base supports it. Below this many
#: customers, quintile cuts produce unstable, near-empty segments.
_MIN_CUSTOMERS_FOR_RFM = 20

#: Number of quantile buckets for each RFM dimension.
_RFM_BUCKETS = 5

#: Top-N lists (products, segments) reported in the output.
_TOP_N = 10


def _score_quantiles(series: pd.Series, ascending: bool) -> pd.Series:
    """
    Bucket a series into 1..5 by quantile.

    `ascending=True` means a higher raw value earns a higher score (spend,
    order count). Recency inverts it: fewer days since the last order is
    better. Falls back to a rank-based cut when the distribution is too
    tied for qcut to find distinct edges (common with small integer
    frequency counts, where most customers have exactly one order).
    """
    labels = list(range(1, _RFM_BUCKETS + 1))
    if not ascending:
        labels = labels[::-1]
    try:
        scored = pd.qcut(series, _RFM_BUCKETS, labels=labels)
    except (ValueError, IndexError):
        ranked = series.rank(method="first", ascending=True)
        try:
            scored = pd.qcut(ranked, _RFM_BUCKETS, labels=labels)
        except (ValueError, IndexError):
            # Degenerate (e.g. every customer identical) — score everyone mid.
            return pd.Series(3, index=series.index, dtype=int)
    return scored.astype(int)


def _segment(recency_score: int, frequency_score: int, monetary_score: int) -> str:
    """Conventional RFM segment labels from the three component scores."""
    value = (frequency_score + monetary_score) / 2
    if recency_score >= 4 and value >= 4:
        return "Champions"
    if recency_score >= 3 and value >= 3:
        return "Loyal"
    if recency_score >= 4 and value < 3:
        return "New / promising"
    if recency_score <= 2 and value >= 4:
        return "At risk (high value, lapsing)"
    if recency_score <= 2 and value >= 3:
        return "Needs attention"
    if recency_score <= 2:
        return "Lost / dormant"
    return "Occasional"


class CohortAnalysisTool(BaseTool):
    """RFM segmentation and repeat-purchase analysis for transaction data."""

    name = "cohort_analysis"
    description = (
        "Analyse customer transaction data: RFM (recency/frequency/monetary) "
        "segmentation, repeat-purchase rate, average order value, revenue "
        "concentration across the customer base, revenue by month and top "
        "products. Use for retail/e-commerce order logs where rows are "
        "repeated purchase events rather than independent observations."
    )

    def applies_to(self, profile: DatasetProfile | None, metadata: DatasetMetadata | None) -> float:
        return domain_confidence(profile, "transactional")

    def execute(  # type: ignore[override]
        self,
        file_path: str,
        date_column: str | None = None,
        amount_column: str | None = None,
        customer_column: str | None = None,
        order_column: str | None = None,
        product_column: str | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        df = _read_df(file_path)

        # Resolved in sequence, each claimed column removed from the pool, so
        # one column can never fill two roles at different grains.
        claimed: set[str] = {c for c in (date_column, amount_column, customer_column,
                                         order_column, product_column) if c}
        date_column = date_column or resolve_column(
            df, ("order_date", "invoice_date", "transaction_date", "purchase_date",
                 "date", "timestamp"), claimed)
        claimed.add(date_column or "")
        amount_column = amount_column or resolve_column(
            df, ("amount", "revenue", "sales", "total", "unit_price", "price",
                 "value", "spend"), claimed)
        claimed.add(amount_column or "")
        customer_column = customer_column or resolve_column(
            df, ("customer_id", "customerid", "customer", "client", "buyer",
                 "user", "account"), claimed)
        claimed.add(customer_column or "")
        order_column = order_column or resolve_column(
            df, ("order_id", "orderid", "invoice_no", "invoice", "transaction_id",
                 "transaction", "receipt", "basket", "order"), claimed)
        claimed.add(order_column or "")
        product_column = product_column or resolve_column(
            df, ("product", "item", "sku", "article", "category"), claimed)
        claimed.add(product_column or "")

        # A per-unit price is not line revenue: summing it ignores how many
        # units each line sold. Multiply by a quantity column when one exists.
        quantity_column: str | None = None
        amount_tokens = normalize_name(amount_column or "").split("_")
        is_unit_price = bool(
            {"price", "unitprice", "rate", "cost"} & set(amount_tokens)
        ) and not {"total", "amount", "revenue", "sales", "line"} & set(amount_tokens)
        if is_unit_price:
            quantity_column = resolve_column(df, ("quantity", "qty", "units"), claimed)

        if date_column is None or date_column not in df.columns:
            raise ToolExecutionError("No date column found. Pass date_column explicitly.")
        if amount_column is None or amount_column not in df.columns:
            raise ToolExecutionError("No amount/revenue column found. Pass amount_column explicitly.")

        work = df.copy()
        work[date_column] = pd.to_datetime(work[date_column], errors="coerce")
        work[amount_column] = pd.to_numeric(work[amount_column], errors="coerce")
        if quantity_column:
            work[amount_column] = work[amount_column] * pd.to_numeric(
                work[quantity_column], errors="coerce"
            )
            revenue_basis = f"{amount_column} x {quantity_column}"
        elif is_unit_price:
            revenue_basis = (
                f"approximated: sum of unit price '{amount_column}' (no quantity column "
                "found), so revenue ignores units per line"
            )
        else:
            revenue_basis = amount_column
        work = work.dropna(subset=[date_column, amount_column])
        if work.empty:
            raise ToolExecutionError(
                "No rows have both a parseable date and a numeric amount."
            )

        last_date = work[date_column].max()
        total_revenue = float(work[amount_column].sum())

        result: dict[str, Any] = {
            "date_column": date_column,
            "amount_column": amount_column,
            "quantity_column": quantity_column,
            "revenue_basis": revenue_basis,
            "transaction_count": len(work),
            "date_range": {
                "start": str(work[date_column].min().date()),
                "end": str(last_date.date()),
            },
            "total_revenue": round(total_revenue, 2),
            "mean_line_item_value": round(float(work[amount_column].mean()), 2),
            "median_line_item_value": round(float(work[amount_column].median()), 2),
            # Recency is measured against the data's own last date so the
            # result does not drift as the file ages.
            "recency_reference_date": str(last_date.date()),
        }

        if order_column and order_column in work.columns:
            order_totals = work.groupby(order_column)[amount_column].sum()
            result["order_count"] = int(order_totals.size)
            result["average_order_value"] = round(float(order_totals.mean()), 2)
            result["median_order_value"] = round(float(order_totals.median()), 2)
            # Lines with a missing order id belong to no order (groupby drops
            # them), so they must not inflate the lines-per-order numerator.
            result["mean_lines_per_order"] = round(
                int(work[order_column].notna().sum()) / order_totals.size, 3
            ) if order_totals.size else 0.0
            result["order_column"] = order_column

        # ---- Revenue by month ----
        monthly = (
            work.set_index(date_column)[amount_column].resample("MS").sum().dropna()
        )
        if not monthly.empty:
            result["revenue_by_month"] = {
                str(idx.date()): round(float(val), 2) for idx, val in monthly.items()
            }
            peak_month = monthly.idxmax()
            result["peak_month"] = {
                "month": str(peak_month.date()),
                "revenue": round(float(monthly.max()), 2),
            }

        # ---- Products ----
        if product_column and product_column in work.columns:
            by_product = (
                work.groupby(product_column)[amount_column]
                .agg(["sum", "count"])
                .sort_values("sum", ascending=False)
            )
            result["product_column"] = product_column
            result["distinct_products"] = int(by_product.shape[0])
            result["top_products_by_revenue"] = [
                {
                    "product": str(name),
                    "revenue": round(float(row["sum"]), 2),
                    "line_items": int(row["count"]),
                    "revenue_share_pct": round(float(row["sum"]) / total_revenue * 100, 2)
                    if total_revenue
                    else 0.0,
                }
                for name, row in by_product.head(_TOP_N).iterrows()
            ]

        # ---- Customer grain: RFM, repeat rate, concentration ----
        if customer_column and customer_column in work.columns:
            result["customer_column"] = customer_column
            grouped = work.groupby(customer_column)
            # Without an order id, a purchase is a distinct calendar day per
            # customer: counting rows would make one multi-line basket look
            # like a repeat customer.
            has_orders = bool(order_column and order_column in work.columns)
            order_grain = (
                # Every customer here has at least one purchase row; one whose
                # order ids are all missing would otherwise count 0 orders.
                grouped[order_column].nunique().clip(lower=1)
                if has_orders
                else work[date_column].dt.normalize().groupby(work[customer_column]).nunique()
            )
            result["frequency_basis"] = (
                f"distinct {order_column}" if has_orders else f"distinct days in {date_column}"
            )
            rfm = pd.DataFrame(
                {
                    "recency_days": (last_date - grouped[date_column].max()).dt.days,
                    "frequency": order_grain,
                    "monetary": grouped[amount_column].sum(),
                }
            )

            customer_count = int(rfm.shape[0])
            repeat_customers = int((rfm["frequency"] > 1).sum())
            result["customer_count"] = customer_count
            result["repeat_customer_count"] = repeat_customers
            result["repeat_purchase_rate_pct"] = round(
                repeat_customers / customer_count * 100, 2
            ) if customer_count else 0.0
            result["mean_revenue_per_customer"] = round(float(rfm["monetary"].mean()), 2)
            result["median_revenue_per_customer"] = round(float(rfm["monetary"].median()), 2)
            result["mean_orders_per_customer"] = round(float(rfm["frequency"].mean()), 3)

            # Revenue concentration — how much of the business rests on the
            # best customers (the practical read on Pareto). Shares are of
            # customer-attributed revenue: rows with a missing customer id
            # belong to no customer and must not dilute the denominator.
            customer_revenue = float(rfm["monetary"].sum())
            result["customer_attributed_transaction_count"] = int(
                work[customer_column].notna().sum()
            )
            ranked_revenue = rfm["monetary"].sort_values(ascending=False)
            for share in (10, 20):
                cutoff = max(1, int(np.ceil(customer_count * share / 100)))
                top_sum = float(ranked_revenue.head(cutoff).sum())
                result[f"revenue_share_of_top_{share}pct_customers"] = round(
                    top_sum / customer_revenue * 100, 2
                ) if customer_revenue else 0.0

            if customer_count >= _MIN_CUSTOMERS_FOR_RFM:
                rfm["r_score"] = _score_quantiles(rfm["recency_days"], ascending=False)
                rfm["f_score"] = _score_quantiles(rfm["frequency"], ascending=True)
                rfm["m_score"] = _score_quantiles(rfm["monetary"], ascending=True)
                rfm["segment"] = [
                    _segment(int(r), int(f), int(m))
                    for r, f, m in zip(
                        rfm["r_score"], rfm["f_score"], rfm["m_score"], strict=True
                    )
                ]
                seg = (
                    rfm.groupby("segment")
                    .agg(
                        customers=("monetary", "size"),
                        revenue=("monetary", "sum"),
                        mean_recency_days=("recency_days", "mean"),
                        mean_frequency=("frequency", "mean"),
                    )
                    .sort_values("revenue", ascending=False)
                )
                result["rfm_segments"] = [
                    {
                        "segment": str(name),
                        "customers": int(row["customers"]),
                        "customer_share_pct": round(
                            int(row["customers"]) / customer_count * 100, 2
                        ),
                        "revenue": round(float(row["revenue"]), 2),
                        "revenue_share_pct": round(
                            float(row["revenue"]) / customer_revenue * 100, 2
                        ) if customer_revenue else 0.0,
                        "mean_recency_days": round(float(row["mean_recency_days"]), 1),
                        "mean_frequency": round(float(row["mean_frequency"]), 2),
                    }
                    for name, row in seg.iterrows()
                ]
                result["rfm_method"] = (
                    f"Quintile scoring (1-{_RFM_BUCKETS}) on recency (lower is "
                    "better), frequency and monetary value; recency measured "
                    f"from {last_date.date()}, the last date in the data."
                )
            else:
                result["rfm_note"] = (
                    f"Only {customer_count} customers — below the {_MIN_CUSTOMERS_FOR_RFM} "
                    "needed for stable quintile segmentation, so RFM was skipped."
                )

        parts = [
            f"{result['transaction_count']:,} transactions totalling "
            f"{total_revenue:,.2f} over {result['date_range']['start']} to "
            f"{result['date_range']['end']}"
        ]
        if "customer_count" in result:
            parts.append(
                f"{result['customer_count']:,} customers, "
                f"{result['repeat_purchase_rate_pct']:.1f}% repeat"
            )
        if "revenue_share_of_top_10pct_customers" in result:
            parts.append(
                f"top 10% of customers drive "
                f"{result['revenue_share_of_top_10pct_customers']:.1f}% of revenue"
            )
        if is_unit_price and not quantity_column:
            parts.append(f"revenue is {revenue_basis}")
        result["summary"] = "; ".join(parts) + "."
        return result

    def findings(
        self,
        output: dict[str, Any],
        profile: DatasetProfile | None,
        metadata: DatasetMetadata | None,
    ) -> list[Finding]:
        results: list[Finding] = []

        summary = output.get("summary")
        if summary:
            results.append(
                Finding(
                    finding_id=f"{self.name}_summary",
                    kind="cohort",
                    headline=summary.rstrip("."),
                    detail=(
                        f"{output.get('transaction_count', 0):,} transactions, "
                        f"{output.get('customer_count', 0):,} customers, total revenue "
                        f"{output.get('total_revenue', 0):,.2f} over "
                        f"{output.get('date_range', {}).get('start')} to "
                        f"{output.get('date_range', {}).get('end')}."
                    ),
                    evidence={
                        "transaction_count": output.get("transaction_count"),
                        "total_revenue": output.get("total_revenue"),
                        "customer_count": output.get("customer_count"),
                        "revenue_basis": output.get("revenue_basis"),
                    },
                    measure="revenue",
                    confidence=0.75,
                    surprise=0.3,
                    objective_fit=0.3,
                    layer="exec",
                )
            )

        top10_share = output.get("revenue_share_of_top_10pct_customers")
        if top10_share is not None and top10_share > 15.0:
            excess = (top10_share - 10.0) / 100.0
            results.append(
                Finding(
                    finding_id=f"{self.name}_top10_concentration",
                    kind="concentration",
                    headline=f"Top 10% of customers drive {top10_share:.1f}% of total revenue",
                    detail=(
                        "Revenue concentration measured by ranking customers by total "
                        "spend and summing the top decile's share of total revenue."
                    ),
                    evidence={
                        "revenue_share_of_top_10pct_customers": top10_share,
                        "revenue_share_of_top_20pct_customers": output.get(
                            "revenue_share_of_top_20pct_customers"
                        ),
                        "customer_count": output.get("customer_count"),
                    },
                    measure="revenue_share",
                    dimension="customer",
                    effect=round(excess, 4),
                    effect_kind="share",
                    confidence=0.7,
                    surprise=0.5,
                    objective_fit=0.3,
                )
            )

        repeat_rate = output.get("repeat_purchase_rate_pct")
        if repeat_rate is not None:
            results.append(
                Finding(
                    finding_id=f"{self.name}_repeat_rate",
                    kind="cohort",
                    headline=f"{repeat_rate:.1f}% of customers are repeat buyers",
                    detail=(
                        f"{output.get('repeat_customer_count', 0):,} of "
                        f"{output.get('customer_count', 0):,} customers placed more than "
                        f"one order (orders counted as {output.get('frequency_basis', 'rows')})."
                    ),
                    evidence={
                        "repeat_customer_count": output.get("repeat_customer_count"),
                        "customer_count": output.get("customer_count"),
                        "frequency_basis": output.get("frequency_basis"),
                    },
                    measure="repeat_purchase_rate",
                    dimension="customer",
                    effect=round(repeat_rate / 100.0, 4),
                    effect_kind="pct",
                    confidence=0.7,
                    surprise=0.3,
                )
            )

        rfm_segments = output.get("rfm_segments")
        if rfm_segments:
            top_segment = rfm_segments[0]
            cust_share = top_segment.get("customer_share_pct") or 0.0
            rev_share = top_segment.get("revenue_share_pct") or 0.0
            if cust_share > 0:
                lift = rev_share / cust_share
                if lift > 1.15:
                    results.append(
                        Finding(
                            finding_id=f"{self.name}_rfm_{top_segment.get('segment', 'top')}",
                            kind="segment_lift",
                            headline=(
                                f"'{top_segment.get('segment')}' customers are "
                                f"{cust_share:.1f}% of the base but drive "
                                f"{rev_share:.1f}% of revenue"
                            ),
                            detail=(
                                f"Mean recency {top_segment.get('mean_recency_days')} days, "
                                f"mean frequency {top_segment.get('mean_frequency')} orders."
                            ),
                            evidence=dict(top_segment),
                            measure="revenue_share",
                            dimension="rfm_segment",
                            level=str(top_segment.get("segment")),
                            effect=round(lift - 1.0, 4),
                            effect_kind="lift",
                            confidence=0.65,
                            surprise=0.45,
                            objective_fit=0.2,
                        )
                    )

        return results

    def get_schema(self) -> dict[str, Any]:
        return {
            "file_path": {
                "type": "string",
                "description": "Path to the dataset.",
                "required": True,
            },
            "date_column": {
                "type": "string",
                "description": "Transaction/order date column. Auto-detected when omitted.",
                "required": False,
            },
            "amount_column": {
                "type": "string",
                "description": "Revenue/amount per row. Auto-detected when omitted.",
                "required": False,
            },
            "customer_column": {
                "type": "string",
                "description": (
                    "Customer identifier. Required for RFM, repeat-rate and "
                    "concentration measures. Auto-detected when omitted."
                ),
                "required": False,
            },
            "order_column": {
                "type": "string",
                "description": (
                    "Order/invoice identifier, used to aggregate line items into "
                    "orders. Auto-detected when omitted."
                ),
                "required": False,
            },
            "product_column": {
                "type": "string",
                "description": "Product/SKU column for the top-product breakdown.",
                "required": False,
            },
        }

