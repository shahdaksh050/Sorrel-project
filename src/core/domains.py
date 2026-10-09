"""
Domain inference — what a dataset is *about*, not just how it is shaped.

src/core/profiler.py answers structural questions (which columns are
numeric, is there a time axis, is it a panel). That is enough to pick
between a histogram and a line chart, but not enough to know that
`open/high/low/close` means a price series whose returns and drawdown are
the interesting quantities, or that `customer_id + order_id + amount`
means repeat-purchase behaviour is.

This module adds that semantic layer. A DomainSpec names the column roles
a domain needs; a structural check then confirms the *shape* agrees,
because names alone do not separate these domains — `date`, `amount`,
`price` and `id` appear in retail and finance alike. What separates them
is cardinality: one row per date (a price series) versus many rows per
customer (a transaction log) versus one row per person (a roster).

Inference returns a *ranked* list, not a winner. Real files are mixed — a
shop's employee roster, a portfolio with a transaction history — and a
tool gates itself with a float score, so a plausible second domain still
contributes its analyses instead of being discarded.

Extensible by design: register_domain() adds a domain without touching
this module's logic or any tool. Pure inspection — no LLM calls, no I/O.
"""
from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    from src.core.profiler import DatasetProfile

#: A domain must score at least this to be reported at all. Below it, the
#: evidence is too thin to justify running domain-specific analyses.
MIN_CONFIDENCE = 0.55

#: Non-alphanumerics collapse to "_" so "Close Price", "close-price" and
#: "ClosePrice" all normalise to the same token stream.
_NON_ALNUM = re.compile(r"[^a-z0-9]+")
_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


def normalize_name(name: str) -> str:
    """Column name -> underscore-delimited lowercase tokens."""
    spaced = _CAMEL_BOUNDARY.sub("_", str(name))
    return _NON_ALNUM.sub("_", spaced.lower()).strip("_")


def resolve_column(
    df: pd.DataFrame, candidates: tuple[str, ...], exclude: set[str] | None = None
) -> str | None:
    """
    Best column for a role, by normalised name. Shared by the domain tools
    for the case where the planner did not pass an explicit column.

    `exclude` holds columns already claimed by another role, and callers are
    expected to resolve roles in sequence, adding each result. Without that,
    a column like `order_date` satisfies both a "date" role and an "order"
    role, and whichever resolves second aggregates at the wrong grain —
    average order value silently becomes revenue per day. Candidates are
    tried most-specific first for the same reason, and a whole-token match
    always beats a substring match.
    """
    blocked = exclude or set()
    normalised = {
        str(c): normalize_name(str(c)) for c in df.columns if str(c) not in blocked
    }
    for candidate in candidates:
        for original, norm in normalised.items():
            if candidate == norm or candidate in norm.split("_"):
                return original
    for candidate in candidates:
        for original, norm in normalised.items():
            if candidate in norm:
                return original
    return None


@dataclass(frozen=True)
class RoleSpec:
    """One semantic slot a domain needs filled, e.g. "the close price"."""

    role: str
    #: Name tokens that identify this role. Matched whole-token first, then
    #: as a substring — "close" matches "close", "adj_close", "closing".
    patterns: tuple[str, ...]
    #: Acceptable ColumnProfile.kind values; empty means any kind.
    kinds: tuple[str, ...] = ()
    #: A domain cannot match at all unless every required role is filled.
    required: bool = False


@dataclass
class DomainMatch:
    """One domain's claim on a dataset, with the evidence behind it."""

    domain: str
    confidence: float
    #: role name -> actual column name in the dataframe.
    roles: dict[str, str]
    evidence: list[str] = field(default_factory=list)

    def column_for(self, role: str) -> str | None:
        return self.roles.get(role)

    def to_dict(self) -> dict[str, Any]:
        return {
            "domain": self.domain,
            "confidence": round(self.confidence, 3),
            "roles": dict(self.roles),
            "evidence": list(self.evidence),
        }


#: Signature of a domain's structural check. Receives the dataframe, its
#: profile and the resolved roles; returns (score_delta, evidence). A
#: negative delta is how a check vetoes a name-only coincidence.
StructuralCheck = Callable[
    [pd.DataFrame, "DatasetProfile", dict[str, str]], "tuple[float, list[str]]"
]


@dataclass(frozen=True)
class DomainSpec:
    """A registered domain: the roles it needs and the shape it expects."""

    name: str
    description: str
    roles: tuple[RoleSpec, ...]
    structural: StructuralCheck


_REGISTRY: list[DomainSpec] = []


def register_domain(spec: DomainSpec) -> None:
    """Add a domain to the registry, replacing any spec of the same name."""
    global _REGISTRY
    _REGISTRY = [s for s in _REGISTRY if s.name != spec.name]
    _REGISTRY.append(spec)


def registered_domains() -> list[DomainSpec]:
    return list(_REGISTRY)


# --------------------------------------------------------------------------
# Role resolution
# --------------------------------------------------------------------------


def _kind_by_column(profile: DatasetProfile) -> dict[str, str]:
    return {c.name: c.kind for c in profile.columns}


def _match_score(tokens: list[str], pattern: str) -> float:
    """How well one column's tokens match one pattern. 0.0 = no match."""
    if pattern in tokens:
        # Whole-token hit, e.g. "close" in ["adj", "close"].
        return 1.0
    joined = "_".join(tokens)
    if joined == pattern:
        return 1.0
    if pattern in joined:
        # Substring hit, e.g. "close" in "closingprice". Weaker, so a real
        # token match on another column always wins.
        return 0.6
    return 0.0


def _resolve_roles(
    df: pd.DataFrame, profile: DatasetProfile, specs: tuple[RoleSpec, ...]
) -> tuple[dict[str, str], set[str]]:
    """
    Fill each role with its best-matching unused column.

    Returns (role -> column, set of roles left unfilled). Columns are
    consumed greedily by descending match strength so two roles never claim
    the same column (an `order_date`/`ship_date` pair resolves sensibly).
    """
    kinds = _kind_by_column(profile)
    tokens_by_col = {str(c): normalize_name(str(c)).split("_") for c in df.columns}

    scored: list[tuple[float, str, str]] = []
    for spec in specs:
        for col, tokens in tokens_by_col.items():
            if spec.kinds and kinds.get(col, "") not in spec.kinds:
                continue
            best = max((_match_score(tokens, p) for p in spec.patterns), default=0.0)
            if best > 0.0:
                scored.append((best, spec.role, col))

    scored.sort(key=lambda item: (-item[0], item[1], item[2]))
    roles: dict[str, str] = {}
    taken: set[str] = set()
    for _score, role, col in scored:
        if role in roles or col in taken:
            continue
        roles[role] = col
        taken.add(col)

    missing = {s.role for s in specs if s.required and s.role not in roles}
    return roles, missing


# --------------------------------------------------------------------------
# Structural checks — what names alone cannot tell apart
# --------------------------------------------------------------------------


def _rows_per_unique(df: pd.DataFrame, column: str | None) -> float:
    """Mean rows per distinct value. 1.0 means the column is one-per-row."""
    if not column or column not in df.columns:
        return 0.0
    nunique = int(df[column].nunique(dropna=True))
    return len(df) / nunique if nunique else 0.0


def _financial_structure(
    df: pd.DataFrame, profile: DatasetProfile, roles: dict[str, str]
) -> tuple[float, list[str]]:
    """
    A price series is one row per date, per instrument.

    The strongest signal is a full OHLC set: four columns named for the
    same bar is not something a retail or HR export produces by accident.
    """
    evidence: list[str] = []
    score = 0.0

    ohlc = [r for r in ("open", "high", "low", "close") if r in roles]
    if len(ohlc) == 4:
        score += 0.35
        evidence.append("Full OHLC column set present (open/high/low/close).")
    elif len(ohlc) >= 2:
        score += 0.15
        evidence.append(f"Partial OHLC columns present ({', '.join(ohlc)}).")

    if "volume" in roles:
        score += 0.05
        evidence.append("Trading volume column present.")

    date_col = roles.get("date")
    symbol_col = roles.get("symbol")
    if date_col:
        rows_per_date = _rows_per_unique(df, date_col)
        if symbol_col:
            # Panel of instruments: one row per (date, symbol).
            score += 0.15
            evidence.append(
                f"Multi-instrument panel: {df[symbol_col].nunique()} symbols over "
                f"{df[date_col].nunique()} dates."
            )
        elif rows_per_date <= 1.05:
            score += 0.20
            evidence.append("One row per date — a single instrument's price series.")
        else:
            # Many rows per date with no instrument column looks like a
            # transaction log, not a price series.
            score -= 0.25
            evidence.append(
                f"{rows_per_date:.1f} rows per date with no instrument column — "
                "not a price series."
            )

    if profile.is_time_series:
        score += 0.05

    return score, evidence


def _transactional_structure(
    df: pd.DataFrame, profile: DatasetProfile, roles: dict[str, str]
) -> tuple[float, list[str]]:
    """
    A transaction log has many rows per customer — that repetition is the
    whole point, and it is what separates it from a customer roster.
    """
    evidence: list[str] = []
    score = 0.0

    customer_col = roles.get("customer_id")
    if customer_col:
        per_customer = _rows_per_unique(df, customer_col)
        if per_customer >= 1.5:
            score += 0.30
            evidence.append(
                f"{per_customer:.1f} rows per customer — repeat purchases present."
            )
        else:
            # One row per customer is a roster, not a transaction log.
            score -= 0.20
            evidence.append(
                "One row per customer — a customer list rather than a transaction log."
            )

    if "order_id" in roles:
        score += 0.15
        evidence.append("Order/transaction identifier present.")
    if "product" in roles:
        score += 0.10
        evidence.append("Product column present.")
    if "quantity" in roles:
        score += 0.05

    return score, evidence


def _workforce_structure(
    df: pd.DataFrame, profile: DatasetProfile, roles: dict[str, str]
) -> tuple[float, list[str]]:
    """
    A roster is one row per person. Compensation plus person-grain
    uniqueness is the combination no price series or sales log produces.
    """
    evidence: list[str] = []
    score = 0.0

    if "salary" in roles:
        score += 0.30
        evidence.append("Compensation column present.")

    employee_col = roles.get("employee_id")
    if employee_col:
        per_employee = _rows_per_unique(df, employee_col)
        if per_employee <= 1.05:
            score += 0.25
            evidence.append("One row per employee — a headcount roster.")
        else:
            score += 0.05
            evidence.append(
                f"{per_employee:.1f} rows per employee — a longitudinal HR record."
            )

    for role, label in (
        ("department", "Department"),
        ("job_title", "Job title"),
        ("hire_date", "Hire date"),
    ):
        if role in roles:
            score += 0.05
            evidence.append(f"{label} column present.")

    return score, evidence


def _product_catalog_structure(
    df: pd.DataFrame, profile: DatasetProfile, roles: dict[str, str]
) -> tuple[float, list[str]]:
    """Confirm a one-row-per-product catalogue rather than a transaction log.

    Product and price names alone also occur in order-line data.  Catalogue
    structure requires the product key to be almost unique and no dominant
    time axis; descriptive/specification columns provide corroboration but
    are deliberately optional so tidy catalogues still match.
    """
    evidence: list[str] = []
    score = 0.0
    product = roles["product"]
    unique_share = float(df[product].nunique(dropna=True)) / max(1, len(df))
    if unique_share >= 0.90:
        score += 0.25
        evidence.append(f"One row per product ({unique_share:.0%} unique product keys).")
    else:
        score -= 0.25
        evidence.append(
            f"Product keys repeat ({unique_share:.0%} unique), which resembles transactions rather than a catalogue."
        )

    price = pd.to_numeric(df[roles["price"]], errors="coerce").dropna()
    if not price.empty and float((price > 0).mean()) >= 0.90:
        score += 0.10
        evidence.append("Price is numeric and predominantly positive.")
    else:
        score -= 0.10

    if not profile.datetime_cols:
        score += 0.05
        evidence.append("No dominant time axis — consistent with a point-in-time catalogue.")

    descriptive = [
        role for role in ("brand", "category", "rating", "specification") if role in roles
    ]
    if descriptive:
        score += min(0.10, 0.025 * len(descriptive))
        evidence.append(f"Product attributes present: {', '.join(descriptive)}.")
    return score, evidence


# --------------------------------------------------------------------------
# Built-in domains
# --------------------------------------------------------------------------

FINANCIAL = DomainSpec(
    name="financial",
    description=(
        "Market price/return data — instrument prices over time. Returns, "
        "volatility, drawdown and autocorrelation are the meaningful measures."
    ),
    roles=(
        RoleSpec("date", ("date", "timestamp", "time", "datetime", "day"),
                 ("datetime",), required=True),
        RoleSpec("close", ("close", "adj_close", "closing", "price", "nav"),
                 ("numeric",), required=True),
        RoleSpec("open", ("open", "opening"), ("numeric",)),
        RoleSpec("high", ("high",), ("numeric",)),
        RoleSpec("low", ("low",), ("numeric",)),
        # "identifier" is accepted because a near-all-unique integer volume
        # column trips the profiler's uniqueness rule and gets labelled an
        # identifier — it is still the volume series we want.
        RoleSpec("volume", ("volume", "vol", "turnover"), ("numeric", "identifier")),
        RoleSpec("symbol", ("symbol", "ticker", "instrument", "security", "isin"),
                 ("categorical", "identifier")),
    ),
    structural=_financial_structure,
)

TRANSACTIONAL = DomainSpec(
    name="transactional",
    description=(
        "Retail/commerce transactions — orders, customers and revenue. "
        "Repeat-purchase behaviour, RFM segmentation and basket composition "
        "are the meaningful measures."
    ),
    roles=(
        RoleSpec("date", ("date", "order_date", "timestamp", "purchase_date",
                          "invoice_date", "transaction_date"),
                 ("datetime",), required=True),
        RoleSpec("amount", ("amount", "revenue", "sales", "total", "price",
                            "unit_price", "value", "spend"),
                 ("numeric",), required=True),
        RoleSpec("customer_id", ("customer", "customer_id", "client", "buyer",
                                 "user", "account"),
                 ("identifier", "categorical", "numeric")),
        RoleSpec("order_id", ("order", "order_id", "invoice", "transaction",
                              "receipt", "basket"),
                 ("identifier", "categorical", "numeric")),
        RoleSpec("product", ("product", "item", "sku", "article", "category"),
                 ("categorical", "identifier")),
        RoleSpec("quantity", ("quantity", "qty", "units", "count"), ("numeric",)),
    ),
    structural=_transactional_structure,
)

WORKFORCE = DomainSpec(
    name="workforce",
    description=(
        "Employee/HR records — headcount, compensation and tenure. Pay "
        "distribution, tenure and attrition are the meaningful measures."
    ),
    roles=(
        RoleSpec("employee_id", ("employee", "employee_id", "emp", "staff",
                                 "person", "worker"),
                 ("identifier", "categorical", "numeric"), required=True),
        RoleSpec("salary", ("salary", "compensation", "pay", "wage", "income",
                            "ctc", "remuneration"),
                 ("numeric",), required=True),
        RoleSpec("department", ("department", "dept", "division", "team", "function"),
                 ("categorical",)),
        RoleSpec("job_title", ("title", "job_title", "role", "position", "designation",
                               "grade", "level"),
                 ("categorical",)),
        RoleSpec("hire_date", ("hire_date", "joining_date", "start_date", "joined",
                               "hire", "doj"),
                 ("datetime",)),
        RoleSpec("exit_date", ("exit_date", "termination_date", "leave_date",
                               "end_date", "resignation_date"),
                 ("datetime",)),
        RoleSpec("status", ("status", "attrition", "active", "employment_status",
                            "left", "churn"),
                 ("categorical", "boolean")),
        RoleSpec("gender", ("gender", "sex"), ("categorical",)),
    ),
    structural=_workforce_structure,
)

PRODUCT_CATALOG = DomainSpec(
    name="product_catalog",
    description=(
        "One-row-per-product catalogue — prices, ratings, categories and "
        "specifications. Price tiers, feature coverage, value relationships "
        "and catalogue completeness are the meaningful analyses."
    ),
    roles=(
        RoleSpec(
            "product",
            ("product", "product_id", "item", "item_id", "sku", "model", "title", "name"),
            ("identifier", "categorical"),
            required=True,
        ),
        RoleSpec(
            "price",
            ("price", "list_price", "sale_price", "cost", "amount", "mrp"),
            ("numeric",),
            required=True,
        ),
        RoleSpec("brand", ("brand", "manufacturer", "make"), ("categorical",)),
        RoleSpec("category", ("category", "type", "family", "segment"), ("categorical",)),
        RoleSpec("rating", ("rating", "score", "review_score", "quality"), ("numeric",)),
        RoleSpec(
            "specification",
            ("specification", "spec", "features", "description", "details"),
            ("text", "categorical"),
        ),
    ),
    structural=_product_catalog_structure,
)

for _spec in (FINANCIAL, TRANSACTIONAL, WORKFORCE, PRODUCT_CATALOG):
    register_domain(_spec)


# --------------------------------------------------------------------------
# Inference
# --------------------------------------------------------------------------


def infer_domains(
    df: pd.DataFrame, profile: DatasetProfile, min_confidence: float = MIN_CONFIDENCE
) -> list[DomainMatch]:
    """
    Rank the registered domains against this dataset.

    Returns matches at or above `min_confidence`, best first. An empty list
    means no domain claimed the data — the generic profile-driven tools
    still apply, which is the correct outcome for a dataset that genuinely
    isn't one of these shapes.
    """
    matches: list[DomainMatch] = []
    for spec in _REGISTRY:
        roles, missing = _resolve_roles(df, profile, spec.roles)
        if missing:
            continue

        optional = [s.role for s in spec.roles if not s.required]
        filled_optional = sum(1 for r in optional if r in roles)
        # Required roles are the price of entry; the score comes from how
        # much corroborating structure sits on top of them.
        base = 0.40 + 0.20 * (filled_optional / len(optional) if optional else 0.0)

        try:
            delta, evidence = spec.structural(df, profile, roles)
        except Exception:
            # A structural check must never take the pipeline down — a
            # domain that cannot be confirmed simply does not match.
            continue

        confidence = max(0.0, min(1.0, base + delta))
        if confidence >= min_confidence:
            matches.append(
                DomainMatch(
                    domain=spec.name,
                    confidence=confidence,
                    roles=dict(roles),
                    evidence=evidence,
                )
            )

    matches.sort(key=lambda m: (-m.confidence, m.domain))
    return matches


def domain_confidence(profile: DatasetProfile | None, domain: str) -> float:
    """
    Score for `domain` on this profile, for use in BaseTool.applies_to.

    Returns 0.0 when there is no profile or the domain did not match, which
    gates the domain's tools out entirely.
    """
    if profile is None:
        return 0.0
    for match in profile.domains:
        if match.domain == domain:
            return match.confidence
    return 0.0


def domains_prompt_string(matches: list[DomainMatch]) -> str:
    """One line per matched domain for LLM context injection."""
    if not matches:
        return "Data domain: none identified — general-purpose analysis."
    from src.core.security import sanitize_for_prompt as _sp

    lines: list[str] = []
    for match in matches:
        roles = ", ".join(f"{r}={_sp(c)}" for r, c in sorted(match.roles.items()))
        lines.append(
            f"Data domain: {match.domain} (confidence {match.confidence:.2f}); "
            f"roles: {roles}."
        )
    return "\n".join(lines)
