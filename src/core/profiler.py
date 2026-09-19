"""
Data Profiler — the 'first look' a data scientist takes at a dataset.

Runs automatically at ingestion (Stage 1) and produces a structured profile:
  - Per-column semantics: numeric / categorical / datetime / boolean /
    identifier / constant, with the statistics that matter for each kind.
  - Dataset-level health: duplicates, missingness, memory footprint.
  - A 0-100 quality score plus human-readable warnings.
  - A compact prompt summary so the planning LLM reasons from the same
    profile the user sees in the UI.

Pure computation — no file I/O, no LLM calls, deterministic.
"""
from __future__ import annotations

import copy
import re
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import pandas as pd

from src.core.security import CARD_RE, EMAIL_RE, IPV4_RE, PHONE_RE, SSN_RE, luhn_valid

if TYPE_CHECKING:
    from src.core.domains import DomainMatch

#: A categorical column with more unique values than this is "high cardinality".
HIGH_CARDINALITY_THRESHOLD = 50

#: Missing-fraction above which a column is flagged as a data-quality problem.
HIGH_MISSING_FRACTION = 0.20

#: Absolute skewness above which a numeric column is flagged as skewed.
SEVERE_SKEW_THRESHOLD = 2.0

#: Identifier-style column name fragments.
_ID_NAME_HINTS = ("id", "uuid", "guid", "index", "key", "code", "number", "no")

#: A string column averaging at least this many words per value reads as
#: free text (reviews, comments, descriptions) rather than category labels
#: — routes it to text-analysis tooling instead of one-hot encoding.
FREE_TEXT_AVG_WORDS = 6.0

#: Minimum rows before a datetime column is treated as a usable time axis.
MIN_TIME_SERIES_ROWS = 20

#: Below this row count, no analysis can produce a meaningful result at all —
#: the data-sufficiency hard floor.
MIN_ROWS_HARD_FLOOR = 2

#: Below this row count (but at/above the hard floor), an inferential result
#: (hypothesis test, trained model) should not be trusted — a soft warning,
#: not a block; tools still run, but the caveat must travel with the result.
MIN_ROWS_RELIABLE = 30

#: Quality-score ceiling once the hard floor trips — distinguishes "0 rows"
#: from "1 row" from "99 rows", which a flat 10-point penalty could not.
INSUFFICIENT_QUALITY_CAP = 20

#: Numeric feature count at/above which dimensionality-reduction tooling
#: (PCA, multicollinearity) starts to pay off.
HIGH_DIMENSIONAL_THRESHOLD = 8

#: Categorical cardinality range considered a plausible panel/group key
#: (too few = boolean-like, too many = identifier-like).
_PANEL_GROUP_MIN_CARD = 2
_PANEL_GROUP_MAX_CARD = 50

#: Column-name fragments that hint at geographic coordinates.
_LAT_NAME_HINTS = ("lat", "latitude")
_LON_NAME_HINTS = ("lon", "lng", "longitude")

# ---------------------------------------------------------------------------
# T1 — semantic role layer (Round 7, item 7.3).
#
# `kind` (numeric/categorical/datetime/...) is structural — it says how the
# column is stored. `semantic_role` is what the column *is for*: whether a
# 0/1 int is a flag or a measure decides whether log1p or IQR should ever
# touch it, whether a categorical is a dimension worth grouping by, and
# whether a numeric column should be summed (a measure) or never aggregated
# at all (an ordinal or identifier). Almost every Round 7 defect traced back
# to this distinction not existing.
# ---------------------------------------------------------------------------

SEMANTIC_MEASURE = "measure"
SEMANTIC_DIMENSION = "dimension"
SEMANTIC_FLAG = "flag"
SEMANTIC_ORDINAL = "ordinal"
SEMANTIC_IDENTIFIER = "identifier"
SEMANTIC_TIME = "time"
SEMANTIC_TEXT = "text"
SEMANTIC_CONSTANT = "constant"

#: A numeric column with cardinality in this (inclusive) range, made up of
#: consecutive small integers, reads as an ordinal scale (rating 1-5, a
#: 1-10 NPS score) rather than a continuous measure — it should never be
#: summed or fed to log1p/IQR, and a segment comparison against it should
#: treat each level as a group, not a number to average blindly.
_ORDINAL_CARD_MIN = 3
_ORDINAL_CARD_MAX = 10
#: Column-name fragments that reinforce an ordinal read on a small-cardinality
#: integer column (distinguishes "star_rating" from "num_children", which is
#: the same shape but is a genuine count/measure).
_ORDINAL_NAME_HINTS = (
    "rating", "score", "grade", "level", "tier", "stars", "priority",
    "satisfaction", "nps", "rank", "severity", "star",
)

#: Whole name tokens (see _name_tokens) that hint a measure is money.
#: "value" is deliberately absent — "measured_value", "sensor_value" and
#: "p_value" are not currency.
_CURRENCY_NAME_HINTS = (
    "price", "amount", "revenue", "cost", "salary", "income", "sales",
    "profit", "margin", "spend", "fee", "charge", "payment", "balance",
    "wage", "expense", "discount", "fare", "budget", "arpu", "usd", "eur",
)
#: Tokens that always mean a percentage, and tokens that do only when the
#: values sit on a 0-1 / 0-100 scale ("heart_rate" is a rate, not a percent).
_PERCENT_NAME_HINTS = ("pct", "percent", "percentage", "perc")
_RATIO_NAME_HINTS = ("rate", "ratio", "share", "proportion", "fraction")
#: Qualifiers that make a "rate" a physical/price quantity, not a percent.
_NON_PERCENT_RATE_QUALIFIERS = (
    "heart", "pulse", "respiratory", "respiration", "breathing", "resp",
    "exchange", "hourly", "daily", "flow", "sampling", "frame", "bit",
)
#: Tokens that hint a measure is a plain count.
_COUNT_NAME_HINTS = ("count", "qty", "quantity", "units", "orders", "visits", "clicks", "views")

#: Tokens that mark a measure additive (a total is meaningful) versus a
#: level/intensity whose total is meaningless (summing temperatures). Unknown
#: measures default to mean — the safe choice for science/sensor data.
_SUM_NAME_HINTS = (
    "amount", "revenue", "sales", "cost", "quantity", "qty", "units", "volume",
    "count", "orders", "visits", "clicks", "views", "spend", "total", "sessions",
    "transactions", "purchases", "downloads", "installs", "impressions",
)
_DURATION_NAME_HINTS = ("duration", "hours", "minutes", "seconds", "mins", "secs")
_MEAN_NAME_HINTS = (
    "temperature", "temp", "age", "score", "pressure", "bp", "heart", "pulse",
    "bmi", "height", "weight", "speed", "velocity", "level", "index", "avg",
    "average", "mean", "median", "latitude", "longitude", "lat", "lon", "lng",
    "humidity", "ph", "density", "concentration", "rating", "price", "rate",
    "ratio", "pct", "percent", "percentage", "glucose", "cholesterol", "std",
)
#: Share of negative values above which a non-currency measure is treated as
#: a signed level (temperature anomaly, return, residual), not a total.
_NEGATIVE_SHARE_FOR_MEAN = 0.05

#: Whole name tokens that mark an integer column as a coded category (a
#: calendar part, a class/group code) rather than a quantity to sum or
#: average. Identifier tokens (id/code/zip...) come from _ID_NAME_HINTS via
#: has_identifier_name_hint plus the zip/postal extras below.
_CODED_DIMENSION_HINTS = (
    "year", "month", "day", "week", "quarter", "hour", "weekday", "dow",
    "zip", "zipcode", "postal", "postcode", "category", "class", "group",
    "type", "region", "segment", "cluster", "cohort",
)
#: Tokens that, next to a coded-dimension token, say the column is a
#: quantity after all ("class_size", "group_count", "total_days").
_QUANTITY_NAME_TOKENS = ("size", "count", "num", "n", "total", "amount", "avg", "mean", "qty")
#: Cardinality ceiling for an integer column to be read as a coded category.
_CODED_DIMENSION_MAX_CARD = 100

#: Tokens that hint a column identifies an "entity" whose repeat-row
#: structure defines the dataset's grain (many rows per customer, per
#: sensor, per player). Whole-token match. Mirrors the entity roles
#: src.core.domains looks for, kept independent to avoid a profiler ->
#: domains import cycle (domains.py imports DatasetProfile from this module).
_ENTITY_NAME_HINTS = (
    "customer", "client", "user", "account", "patient", "member", "employee",
    "subscriber", "sensor", "device", "station", "subject", "participant",
    "player", "team", "store", "shop", "site", "machine", "vehicle", "school",
    "hospital", "company", "firm", "ticker", "symbol", "product", "sku",
    "household", "person", "respondent", "meter", "asset", "fund", "branch",
    "clinic", "farm", "animal",
)
#: Geographic tokens that only name an entity when the table repeats them
#: over time (country-year panels); otherwise they are just dimensions.
_GEO_ENTITY_NAME_HINTS = ("country", "state", "city", "county", "province", "district")

#: Rows-per-unique-value above which a candidate entity column is treated as
#: this dataset's grain-defining entity (i.e. genuinely repeats, not just
#: incidentally non-unique).
_ENTITY_REPEAT_THRESHOLD = 1.5

#: Row budget for the panel-structure probe (head of the frame — a random
#: sample would break the timestamp overlap the probe measures).
_PANEL_PROBE_ROWS = 20_000

_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_NON_ALNUM = re.compile(r"[^a-z0-9]+")

# ---------------------------------------------------------------------------
# Data archetypes — what kind of study/table this is (ImpPlan Phase 1). Unlike
# src.core.domains (what the data is *about*), an archetype is decided from
# structure: time axis, entity repetition, scale columns, an assignment arm.
# ---------------------------------------------------------------------------

ARCHETYPE_EVENT_LOG = "event_log"
ARCHETYPE_PANEL = "panel"
ARCHETYPE_SENSOR = "sensor_timeseries"
ARCHETYPE_SURVEY = "survey"
ARCHETYPE_EXPERIMENT = "experiment"
ARCHETYPE_CROSS_SECTION = "cross_section"

_REGULAR_FREQUENCIES = ("hourly", "daily", "weekly", "monthly", "quarterly", "yearly")

#: Whole tokens that name an experiment assignment column. The column's
#: tokens must ALL come from this set, so "group"/"test_group"/"ab_variant"
#: qualify but "age_group"/"product_group" (plain dimensions) do not.
_ARM_NAME_TOKENS = ("group", "variant", "arm", "treatment", "control", "cohort")
_ARM_QUALIFIER_TOKENS = (
    "test", "ab", "experiment", "exp", "assigned", "assignment", "condition", "bucket",
)
_ARM_MAX_LEVELS = 5
#: Smallest arm's share of rows — below it the column is a rare category,
#: not an assignment.
_ARM_MIN_LEVEL_SHARE = 0.05

#: Ordinal columns needed to call a table a survey, and the low-cardinality
#: categorical count/share that does so alongside a respondent key.
_SURVEY_MIN_ORDINALS = 3
_SURVEY_MIN_CATEGORICALS = 6
_SURVEY_MIN_CATEGORICAL_SHARE = 0.6
_SURVEY_MAX_LEVELS = 10
#: Top of a rating scale (1-5, 0-10, 1-7 ...) for unnamed Likert items.
_LIKERT_SCALE_MAX = (4, 5, 6, 7, 9, 10, 11)

# ---------------------------------------------------------------------------
# PII detection (ImpPlan Phase 4, data egress) — flags columns whose values
# identify a person so to_prompt_string can withhold them from the LLM.
# ---------------------------------------------------------------------------

PII_EMAIL = "email"
PII_PHONE = "phone"
PII_PERSON_NAME = "person_name"
PII_ADDRESS = "address"
PII_GOV_ID = "gov_id"
PII_CARD = "card_number"
PII_IP = "ip_address"

#: Values sampled per column for the pattern probe, and the share that must
#: match for a column to be flagged.
_PII_SAMPLE_ROWS = 200
_PII_MATCH_SHARE = 0.8

_PERSON_NAME_RE = re.compile(r"[A-Z][A-Za-z'-]+(?: [A-Z][A-Za-z'.-]+){1,2}")

_PHONE_NAME_TOKENS = ("phone", "mobile", "telephone", "tel", "fax", "cellphone")
_GOV_ID_NAME_TOKENS = ("ssn", "passport", "aadhaar", "nino")
_ADDRESS_NAME_TOKENS = ("address", "street", "addr")
_NAME_QUALIFIER_TOKENS = ("first", "last", "full", "given", "middle", "family", "sur")
_NAME_WORD_TOKENS = ("firstname", "lastname", "fullname", "surname", "forename")
#: Entities whose "<entity>_name" column holds a person's name ("product_name"
#: and "store_name" do not).
_PERSON_ENTITY_TOKENS = (
    "customer", "client", "user", "patient", "member", "employee", "subscriber",
    "participant", "respondent", "person", "contact", "owner", "author", "student",
)


def _name_tokens(name: str) -> list[str]:
    """Column name -> lowercase tokens, split on camelCase and any
    non-alphanumeric ("heartRate", "heart-rate", "heart_rate" -> heart, rate)."""
    return [t for t in _NON_ALNUM.split(_CAMEL_BOUNDARY.sub("_", str(name)).lower()) if t]


def _has_token(tokens: list[str], hints: tuple[str, ...]) -> bool:
    """Whole-token match, tolerating a plural "s" ("charges" -> charge)."""
    return any(t in hints or (t.endswith("s") and t[:-1] in hints) for t in tokens)


def _rows_per_value(row_count: int, nunique: int) -> float:
    """Mean rows per distinct value of a column, given its row count and
    nunique. 1.0 = one row per value.

    Takes the already-computed `nunique` (every ColumnProfile has one, from
    `_profile_column`'s own `series.nunique(dropna=True)`) instead of a
    `(df, column)` pair that would recompute it — the entity-detection block
    below used to re-scan every entity-name-hinted column a second time
    purely to redo a count `_profile_column` had already done."""
    return row_count / nunique if nunique else 0.0


def _infer_semantic_role(
    name: str, series: pd.Series, kind: str, nunique: int, flags: list[str]
) -> str:
    """Assign a semantic role given the structural `kind` already decided by
    `_profile_column`. Only numeric/categorical/boolean columns need real
    reasoning here — the rest map onto a role one-to-one."""
    if kind == "constant":
        return SEMANTIC_CONSTANT
    if kind == "identifier":
        return SEMANTIC_IDENTIFIER
    if kind == "datetime":
        return SEMANTIC_TIME
    if kind == "text":
        return SEMANTIC_TEXT
    if kind == "boolean":
        return SEMANTIC_FLAG
    if kind == "categorical":
        return SEMANTIC_DIMENSION
    if kind == "numeric":
        is_integer_valued = False
        try:
            clean = series.dropna()
            is_integer_valued = bool(len(clean)) and bool((clean % 1 == 0).all())
        except TypeError:
            is_integer_valued = False
        if is_integer_valued and nunique == 2:
            return SEMANTIC_FLAG
        if is_integer_valued:
            coded_role = _coded_integer_role(name, len(series), nunique)
            if coded_role:
                return coded_role
        name_l = name.lower()
        if (
            is_integer_valued
            and _ORDINAL_CARD_MIN <= nunique <= _ORDINAL_CARD_MAX
            and any(h in name_l for h in _ORDINAL_NAME_HINTS)
        ):
            return SEMANTIC_ORDINAL
        return SEMANTIC_MEASURE
    return SEMANTIC_DIMENSION


#: Last-token identifier words for integer-coded keys. Narrower than
#: _ID_NAME_HINTS on purpose: "index"/"number"/"no" also end genuine
#: measures ("uv_index", "room_number" counts) and must not demote them.
_CODED_KEY_TOKENS = ("id", "ids", "uuid", "guid", "key", "code", "zip", "zipcode", "postal", "postcode")


def _coded_integer_role(name: str, row_count: int, nunique: int) -> str | None:
    """Role for an integer column whose *name* says it is a code, not a
    quantity — a calendar part (year, month, hour), a class/group/region
    code, or a repeating entity key (an int customer_id). Such columns must
    never be summed or averaged; returns None when the name gives no such
    signal and the column stays a measure/ordinal candidate."""
    tokens = _name_tokens(name)
    if not tokens:
        return None
    if tokens[-1] in _CODED_KEY_TOKENS or tokens[-1] in _ENTITY_NAME_HINTS:
        repeats = _rows_per_value(row_count, nunique) >= _ENTITY_REPEAT_THRESHOLD
        if repeats or nunique <= _CODED_DIMENSION_MAX_CARD:
            return SEMANTIC_IDENTIFIER if nunique > _PANEL_GROUP_MAX_CARD else SEMANTIC_DIMENSION
    if (
        any(t in _CODED_DIMENSION_HINTS for t in tokens)
        and not any(t in _QUANTITY_NAME_TOKENS for t in tokens)
        and nunique <= _CODED_DIMENSION_MAX_CARD
    ):
        return SEMANTIC_DIMENSION
    return None


def _infer_unit_hint(name: str, semantic_role: str, stats: dict[str, Any]) -> str | None:
    """Cheap name-based unit inference for measures — a fuller version would
    thread src.core.coercion's currency/percent detection through, but that
    detection happens at read time and is discarded before profiling runs
    today; this heuristic gets the common cases without that plumbing.
    Whole-token matching only, and "rate"/"ratio"/"share" count as a percent
    only on a 0-1 or 0-100 scale without a physical qualifier — a
    heart_rate of 72 bpm is not 72%."""
    if semantic_role != SEMANTIC_MEASURE:
        return None
    tokens = _name_tokens(name)
    if _has_token(tokens, _PERCENT_NAME_HINTS) or name.rstrip().endswith("%"):
        return "percent"
    lo, hi = stats.get("min"), stats.get("max")
    if (
        _has_token(tokens, _RATIO_NAME_HINTS)
        and not any(t in _NON_PERCENT_RATE_QUALIFIERS for t in tokens)
        and lo is not None and hi is not None
        and ((-1.0 <= lo and hi <= 1.0) or (0.0 <= lo and hi <= 100.0))
    ):
        return "percent"
    if _has_token(tokens, _CURRENCY_NAME_HINTS):
        return "currency"
    if _has_token(tokens, _COUNT_NAME_HINTS):
        return "count"
    return None


def _infer_aggregation(
    name: str, semantic_role: str, unit_hint: str | None, stats: dict[str, Any]
) -> str | None:
    """"sum" when a total of this measure is meaningful (revenue, units,
    visits), "mean" when only an average is (temperature, age, a rate).
    Name lists are checked before unit_hint so a mis-tagged unit can never
    make a level summable; unknown measures default to "mean", the safe
    choice for science/sensor data. None for non-measures."""
    if semantic_role != SEMANTIC_MEASURE:
        return None
    tokens = _name_tokens(name)
    if unit_hint == "percent" or _has_token(tokens, _MEAN_NAME_HINTS):
        return "mean"
    if unit_hint != "currency" and stats.get("negative_pct", 0.0) > _NEGATIVE_SHARE_FOR_MEAN * 100:
        return "mean"
    if unit_hint in ("currency", "count") or _has_token(tokens, _SUM_NAME_HINTS):
        return "sum"
    if _has_token(tokens, _DURATION_NAME_HINTS) and stats.get("min", -1.0) >= 0:
        return "sum"
    return "mean"


#: Tokens that make a PII-named column a quantity about it ("email_count",
#: "address_length"), not the personal data itself.
_PII_QUANTITY_TOKENS = ("count", "total", "amount", "avg", "mean", "qty", "size", "length", "verified")


def _pii_name_hint(tokens: list[str]) -> str | None:
    """PII kind a column *name* announces (whole tokens), or None. A bare
    "name" column returns None here — it needs value evidence (it may hold
    product names)."""
    if "email" in tokens or ("e" in tokens and "mail" in tokens):
        return PII_EMAIL
    if any(t in ("ip", "ipv4", "ipv6") for t in tokens):
        return PII_IP
    if _has_token(tokens, _PHONE_NAME_TOKENS):
        return PII_PHONE
    if (
        any(t in _GOV_ID_NAME_TOKENS for t in tokens)
        or (any(t in ("national", "social", "tax") for t in tokens) and any(t in ("id", "number", "no") for t in tokens))
        or (any(t in ("driver", "driving") for t in tokens) and any(t in ("license", "licence") for t in tokens))
    ):
        return PII_GOV_ID
    if "card" in tokens and any(t in ("number", "no", "num") for t in tokens):
        return PII_CARD
    if _has_token(tokens, _ADDRESS_NAME_TOKENS):
        return PII_ADDRESS
    if any(t in _NAME_WORD_TOKENS for t in tokens) or (
        "name" in tokens
        and any(t in _NAME_QUALIFIER_TOKENS or t in _PERSON_ENTITY_TOKENS for t in tokens)
    ):
        return PII_PERSON_NAME
    return None


def _detect_pii(name: str, series: pd.Series, kind: str, semantic_role: str) -> str | None:
    """PII kind held by a column, from its name (whole tokens) or, failing
    that, from value patterns on a head sample of string-like columns
    (email, IPv4, SSN-style id, Luhn-valid card, separated phone number).
    Flags, constants and quantities about PII ("email_count") are never PII."""
    if kind in ("constant", "boolean") or semantic_role == SEMANTIC_FLAG:
        return None
    tokens = _name_tokens(name)
    if any(t in _PII_QUANTITY_TOKENS for t in tokens):
        return None
    hint = _pii_name_hint(tokens)
    if hint:
        return hint
    if kind not in ("categorical", "identifier", "text"):
        return None
    sample = series.dropna().astype(str).str.strip().head(_PII_SAMPLE_ROWS)
    if sample.empty:
        return None
    probes: list[tuple[str, Any]] = [
        (PII_EMAIL, EMAIL_RE.fullmatch),
        (PII_IP, IPV4_RE.fullmatch),
        (PII_GOV_ID, SSN_RE.fullmatch),
        (PII_CARD, lambda v: CARD_RE.fullmatch(v) is not None and luhn_valid(v)),
        (PII_PHONE, PHONE_RE.fullmatch),
    ]
    if tokens == ["name"]:
        probes.append((PII_PERSON_NAME, _PERSON_NAME_RE.fullmatch))
    for label, probe in probes:
        if float(sample.map(lambda v, p=probe: bool(p(v))).mean()) >= _PII_MATCH_SHARE:
            return label
    return None


@dataclass
class ColumnProfile:
    """Profile of a single column."""

    name: str
    dtype: str
    kind: str                 # numeric | categorical | datetime | boolean | identifier | constant
    missing_count: int
    missing_pct: float
    nunique: int
    stats: dict[str, Any] = field(default_factory=dict)        # numeric: mean/std/min/max/median/q1/q3/skew/zero_pct/negative_pct; datetime: start/end/frequency; boolean: mean
    top_values: dict[str, int] = field(default_factory=dict)   # categorical columns
    flags: list[str] = field(default_factory=list)

    # ---- T1 semantic layer (7.3) ----
    semantic_role: str = SEMANTIC_DIMENSION   # measure | dimension | flag | ordinal | identifier | time | text | constant
    unit_hint: str | None = None              # currency | percent | count | None
    aggregation: str | None = None            # sum | mean for measures (is a total meaningful?); None otherwise
    pii: str | None = None                    # email | phone | person_name | address | gov_id | card_number | ip_address

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "dtype": self.dtype,
            "kind": self.kind,
            "missing_count": self.missing_count,
            "missing_pct": self.missing_pct,
            "nunique": self.nunique,
            "stats": self.stats,
            "top_values": self.top_values,
            "flags": self.flags,
            "semantic_role": self.semantic_role,
            "unit_hint": self.unit_hint,
            "aggregation": self.aggregation,
            "pii": self.pii,
        }

    def is_measure(self) -> bool:
        return self.semantic_role == SEMANTIC_MEASURE

    def is_flag(self) -> bool:
        return self.semantic_role == SEMANTIC_FLAG

    def is_dimension(self) -> bool:
        return self.semantic_role in (SEMANTIC_DIMENSION, SEMANTIC_FLAG)


@dataclass
class DatasetProfile:
    """Full structured profile of a dataset."""

    row_count: int
    column_count: int
    duplicate_rows: int
    memory_mb: float
    columns: list[ColumnProfile]
    warnings: list[str]
    quality_score: int        # 0-100

    # ---- Dataset-nature facts (drive tool selection, not just display) ----
    datetime_cols: list[str] = field(default_factory=list)
    is_time_series: bool = False
    text_cols: list[str] = field(default_factory=list)
    geo_lat_col: str | None = None
    geo_lon_col: str | None = None
    is_high_dimensional: bool = False
    panel_group_cols: list[str] = field(default_factory=list)

    # ---- Data-sufficiency gate (U1.3) — row_count < 100 used to cost a flat
    # 10 quality points regardless of whether that meant 99 rows or 0 rows.
    # This makes "not enough data to trust a result" an explicit, checkable
    # fact instead of a number embedded in the score. ----
    is_sufficient: bool = True
    sufficiency_reason: str | None = None

    # ---- Semantic domain (src/core/domains.py) — what the data is *about*,
    # populated by the controller after profiling because inference needs the
    # dataframe as well as this profile. Defaulted to empty so every
    # DatasetProfile always carries the attribute: BaseTool.applies_to reads
    # it directly, and a missing attribute would silently gate every
    # domain tool out (the failure mode U0.5 documented for time-series). ----
    domains: list[DomainMatch] = field(default_factory=list)

    # ---- T1 semantic layer (7.3) — what one row *is*. `grain` is the
    # column set that makes a row unique; `entity_col`/`rows_per_entity`
    # describe repeat-row structure (e.g. 10 order rows per customer) even
    # when no single column is a clean key. Populated by profile_dataframe;
    # defaulted so every DatasetProfile carries the attributes. ----
    grain: list[str] = field(default_factory=list)
    entity_col: str | None = None
    rows_per_entity: float | None = None
    #: Sampling frequency of the first datetime column (hourly | daily |
    #: weekly | monthly | quarterly | yearly | irregular), None without one.
    time_frequency: str | None = None
    #: Structural archetype (event_log | panel | sensor_timeseries | survey |
    #: experiment | cross_section; None when data is insufficient) and the
    #: short facts that decided it.
    archetype: str | None = None
    archetype_evidence: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "row_count": self.row_count,
            "column_count": self.column_count,
            "duplicate_rows": self.duplicate_rows,
            "memory_mb": self.memory_mb,
            "columns": [c.to_dict() for c in self.columns],
            "warnings": self.warnings,
            "quality_score": self.quality_score,
            "datetime_cols": self.datetime_cols,
            "is_time_series": self.is_time_series,
            "text_cols": self.text_cols,
            "geo_lat_col": self.geo_lat_col,
            "geo_lon_col": self.geo_lon_col,
            "is_high_dimensional": self.is_high_dimensional,
            "panel_group_cols": self.panel_group_cols,
            "is_sufficient": self.is_sufficient,
            "sufficiency_reason": self.sufficiency_reason,
            "domains": [d.to_dict() for d in self.domains],
            "grain": self.grain,
            "entity_col": self.entity_col,
            "rows_per_entity": self.rows_per_entity,
            "time_frequency": self.time_frequency,
            "archetype": self.archetype,
            "archetype_evidence": self.archetype_evidence,
        }

    def columns_of_kind(self, *kinds: str) -> list[ColumnProfile]:
        return [c for c in self.columns if c.kind in kinds]

    def columns_of_role(self, *roles: str) -> list[ColumnProfile]:
        return [c for c in self.columns if c.semantic_role in roles]

    def measures(self) -> list[ColumnProfile]:
        return self.columns_of_role(SEMANTIC_MEASURE)

    def dimensions(self) -> list[ColumnProfile]:
        return self.columns_of_role(SEMANTIC_DIMENSION, SEMANTIC_FLAG)

    def has_geo(self) -> bool:
        return bool(self.geo_lat_col and self.geo_lon_col)

    def to_prompt_string(self, max_warnings: int = 8, max_columns: int = 40) -> str:
        """
        Dense profile summary for LLM context injection: dataset-level facts,
        then one short line per column (up to `max_columns`) so a planner
        sees every column of a wide dataset, not a sample of them.

        Column names and category values are dataset-derived (untrusted) and
        are sanitised before they reach a prompt. Warnings embed column
        names too, so they pass through the same sanitiser.
        """
        from src.core.security import pii_redaction_enabled
        from src.core.security import sanitize_for_prompt as _sp

        redact = pii_redaction_enabled()
        lines = [
            f"Data profile: {self.row_count:,} rows x {self.column_count} columns; "
            f"quality score {self.quality_score}/100; {self.duplicate_rows} duplicate rows.",
        ]
        if not self.is_sufficient:
            lines.append(f"INSUFFICIENT DATA: {self.sufficiency_reason or 'too few rows.'}")
        nature: list[str] = []
        if self.is_time_series:
            freq = f", {self.time_frequency}" if self.time_frequency else ""
            nature.append(f"time-series ({', '.join(_sp(c) for c in self.datetime_cols[:3])}{freq})")
        if self.text_cols:
            nature.append(f"free text ({', '.join(_sp(c) for c in self.text_cols[:3])})")
        if self.has_geo():
            nature.append(f"geo coordinates ({_sp(self.geo_lat_col or '')}, {_sp(self.geo_lon_col or '')})")
        if self.is_high_dimensional:
            nature.append("high-dimensional (watch multicollinearity)")
        if self.panel_group_cols:
            nature.append(f"entity x time panel via {', '.join(_sp(c) for c in self.panel_group_cols[:3])}")
        if self.entity_col and self.rows_per_entity:
            nature.append(f"{self.rows_per_entity:.1f} rows per '{_sp(self.entity_col)}'")
        if self.grain:
            nature.append(f"row grain: {', '.join(_sp(c) for c in self.grain)}")
        if nature:
            lines.append("Data nature: " + "; ".join(nature) + ".")
        if self.archetype:
            evidence = "; ".join(_sp(e, max_len=80) for e in self.archetype_evidence[:3])
            lines.append(f"Data archetype: {self.archetype}" + (f" ({evidence})" if evidence else "") + ".")
        for match in self.domains:
            roles = ", ".join(f"{r}={_sp(c)}" for r, c in sorted(match.roles.items()))
            lines.append(
                f"Data domain: {match.domain} (confidence {match.confidence:.2f}) — roles: {roles}."
            )
        lines.append(
            "Roles: measure=quantity (agg=sum: totals meaningful; agg=mean: average only); "
            "dimension=group-by; flag=0/1 indicator (rate, never sum); ordinal=ranked scale; "
            "identifier=key (never model on it)."
        )
        lines.append("Columns:")
        for c in self.columns[:max_columns]:
            head = c.semantic_role + (f"/{c.unit_hint}" if c.unit_hint else "")
            if c.aggregation:
                head += f", agg={c.aggregation}"
            facts = self._column_facts(c, _sp, redact)
            lines.append(f"- {_sp(c.name)}: {head}" + (f" — {facts}" if facts else ""))
        rest = self.columns[max_columns:]
        if rest:
            role_counts: dict[str, int] = {}
            for c in rest:
                role_counts[c.semantic_role] = role_counts.get(c.semantic_role, 0) + 1
            summary = ", ".join(f"{v} {k}" for k, v in sorted(role_counts.items()))
            lines.append(f"- ...and {len(rest)} more columns ({summary}).")
        if self.warnings:
            lines.append(
                "Warnings: " + " | ".join(_sp(w, max_len=160) for w in self.warnings[:max_warnings])
            )
        return "\n".join(lines)

    def _column_facts(self, c: ColumnProfile, _sp: Any, redact: bool = True) -> str:
        """Key facts for one column's prompt line, chosen by semantic role.
        A PII column's values (top levels, ranges, dates) are withheld when
        `redact` is on — only its name, kind of PII and missingness remain."""
        s = c.stats
        facts: list[str] = []
        role = c.semantic_role
        if c.pii and redact:
            facts.append(f"[redacted: {c.pii}]")
        elif role == SEMANTIC_TIME:
            if "start" in s:
                span = f"{_short_ts(s['start'])}..{_short_ts(s['end'])}"
                facts.append(span + (f", {s['frequency']}" if s.get("frequency") else ""))
        elif role == SEMANTIC_FLAG:
            rate = s.get("mean")
            if rate is not None and (c.kind == "boolean" or (s.get("min") == 0 and s.get("max") == 1)):
                facts.append(f"{rate:.0%} positive")
            elif c.top_values:
                facts.append(self._level_shares(c, _sp))
        elif role in (SEMANTIC_MEASURE, SEMANTIC_ORDINAL) and "median" in s:
            facts.append(f"median {_num(s['median'])}")
            if "q1" in s:
                facts.append(f"IQR {_num(s['q1'])}–{_num(s['q3'])}")
            facts.append(f"range {_num(s['min'])}–{_num(s['max'])}")
            if "severe_skew" in c.flags:
                facts.append("skewed")
            if s.get("zero_pct", 0) >= 10:
                facts.append(f"{s['zero_pct']:.0f}% zeros")
        elif role in (SEMANTIC_DIMENSION, SEMANTIC_IDENTIFIER):
            if c.top_values:
                facts.append(self._level_shares(c, _sp))
            elif "min" in s:
                facts.append(f"{c.nunique:,} codes {_num(s['min'])}–{_num(s['max'])}")
            else:
                facts.append(f"{c.nunique:,} unique")
        elif role == SEMANTIC_TEXT and "avg_word_count" in s:
            facts.append(f"free text, ~{s['avg_word_count']:.0f} words")
        if c.missing_pct > 0:
            facts.append(f"{c.missing_pct:g}% missing")
        return ", ".join(facts)

    def _level_shares(self, c: ColumnProfile, _sp: Any) -> str:
        """"N levels: a 50%, b 30%, c 12%" from a column's top values."""
        present = max(1, self.row_count - c.missing_count)
        top = ", ".join(
            f"{_sp(k, max_len=30)} {v / present:.0%}" for k, v in list(c.top_values.items())[:3]
        )
        return f"{c.nunique:,} levels: {top}"


def _num(value: float) -> str:
    """Compact number for prompt text: 12,345 / 54.6 / 0.0312."""
    v = float(value)
    if abs(v) >= 1000:
        return f"{v:,.0f}"
    return f"{v:.3g}"


def _short_ts(value: str) -> str:
    """Drop a midnight time part from a timestamp string."""
    return value[:-9] if value.endswith(" 00:00:00") else value


def _is_datetime_like(series: pd.Series) -> bool:
    """True for datetime dtypes or string columns that parse as dates."""
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
        return False
    sample = series.dropna().head(20)
    if sample.empty:
        return False
    # Ponytail: a pure time column like "00:00:00" has no date separators;
    # pd.to_datetime prepends today's date, creating artificial timestamps.
    sample_str = sample.astype(str)
    if not sample_str.str.contains(r"[-/.]|^\d{8}$").any():
        return False
    try:
        parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
    except (ValueError, TypeError):
        return False
    return bool(parsed.notna().mean() >= 0.9)


#: (label, nominal step in days) for time_frequency; a median gap within
#: _FREQ_TOLERANCE of a step names the series' frequency.
_FREQUENCIES = (
    ("hourly", 1 / 24), ("daily", 1.0), ("weekly", 7.0), ("monthly", 30.44),
    ("quarterly", 91.3), ("yearly", 365.25),
)
_FREQ_TOLERANCE = (0.75, 1.3)


def _datetime_stats(series: pd.Series) -> dict[str, Any]:
    """Range and inferred frequency of a datetime(-like) column: the median
    gap between sorted distinct timestamps names the frequency, "irregular"
    when fewer than half the gaps agree with it (a gappy business-day series
    still reads daily). Parses distinct values only, so it stays cheap."""
    values = series.dropna().unique()
    if not pd.api.types.is_datetime64_any_dtype(series):
        values = pd.to_datetime(pd.Series(values), errors="coerce", format="mixed").dropna().unique()
    ts = pd.Series(pd.to_datetime(values)).sort_values()
    if ts.empty:
        return {}
    stats: dict[str, Any] = {"start": str(ts.iloc[0]), "end": str(ts.iloc[-1])}
    if len(ts) >= 3:
        gaps = ts.diff().dropna().dt.total_seconds() / 86_400
        median_gap = float(gaps.median())
        frequency = "irregular"
        for label, step in _FREQUENCIES:
            lo, hi = step * _FREQ_TOLERANCE[0], step * _FREQ_TOLERANCE[1]
            if lo <= median_gap <= hi:
                if float(gaps.between(lo, hi).mean()) >= 0.5:
                    frequency = label
                break
        stats["frequency"] = frequency
    return stats


def has_identifier_name_hint(name: str) -> bool:
    """True when a column name reads as an identifier (id, code, zipcode...).

    Public so other modules can apply the same "this name says identifier"
    rule without duplicating it — e.g. src.core.coercion uses it to avoid
    numeric-coercing a zero-padded zipcode, and
    src.tools.statistical_analysis reuses is_identifier_like wholesale
    instead of keeping a second, looser ID heuristic.
    """
    name_l = name.lower()
    return any(
        name_l == h or name_l.endswith(f"_{h}") or name_l.endswith(h)
        for h in _ID_NAME_HINTS
    )


def is_identifier_like(name: str, series: pd.Series, row_count: int) -> bool:
    """Heuristic: near-unique column whose name hints at an identifier, or a
    fully-unique non-float column. A sorted *float* measurement (a
    continuous value that happens to be 100% unique) does NOT qualify —
    only a genuine key does. Public: reused by
    src.tools.statistical_analysis instead of a duplicate local heuristic.
    """
    if row_count == 0:
        return False
    nunique = int(series.nunique(dropna=True))
    uniqueness = nunique / row_count
    name_hit = has_identifier_name_hint(name)
    return (uniqueness >= 0.98 and name_hit) or (
        uniqueness == 1.0 and not pd.api.types.is_float_dtype(series)
    )


def _profile_column(name: str, series: pd.Series, row_count: int) -> ColumnProfile:
    missing = int(series.isna().sum())
    missing_pct = round(100.0 * missing / row_count, 2) if row_count else 0.0
    nunique = int(series.nunique(dropna=True))
    flags: list[str] = []
    stats: dict[str, Any] = {}
    top_values: dict[str, int] = {}

    # Free-text probe, computed early: a review/comment column is very often
    # *exactly* unique per row (no two reviews are word-for-word identical),
    # which would otherwise satisfy the identifier check's uniqueness==1.0
    # branch below and misfile prose as a row ID. Prose length is decided
    # entirely by content, not by dtype or cardinality, so it must be
    # checked before identifier — a 100%-unique text column is text, not an ID.
    free_text_avg_words: float | None = None
    if (
        not pd.api.types.is_numeric_dtype(series)
        and not pd.api.types.is_bool_dtype(series)
        and nunique > 1
    ):
        sample = series.dropna().astype(str).head(200)
        if not sample.empty:
            free_text_avg_words = float(sample.str.split().str.len().mean())

    if nunique <= 1:
        kind = "constant"
        flags.append("constant")
    elif pd.api.types.is_bool_dtype(series):
        kind = "boolean"
        clean = series.dropna()
        if not clean.empty:
            stats["mean"] = round(float(clean.astype(float).mean()), 4)
    # Datetime check must precede the identifier check: a daily time index is
    # 100% unique but is a time axis, not an ID.
    elif _is_datetime_like(series):
        kind = "datetime"
        stats = _datetime_stats(series)
    elif free_text_avg_words is not None and free_text_avg_words >= FREE_TEXT_AVG_WORDS:
        kind = "text"
        flags.append("free_text")
        stats["avg_word_count"] = round(free_text_avg_words, 2)
    elif is_identifier_like(name, series, row_count):
        kind = "identifier"
        flags.append("id_like")
    elif pd.api.types.is_numeric_dtype(series):
        kind = "numeric"
        clean = series.dropna()
        if not clean.empty:
            stats = {
                "mean": round(float(clean.mean()), 4),
                "std": round(float(clean.std()), 4) if len(clean) > 1 else 0.0,
                "min": round(float(clean.min()), 4),
                "max": round(float(clean.max()), 4),
                "median": round(float(clean.median()), 4),
                "q1": round(float(clean.quantile(0.25)), 4),
                "q3": round(float(clean.quantile(0.75)), 4),
                "zero_pct": round(100.0 * float((clean == 0).mean()), 2),
                "negative_pct": round(100.0 * float((clean < 0).mean()), 2),
            }
            if len(clean) > 2:
                skew = float(clean.skew())
                stats["skew"] = round(skew, 4)
                if abs(skew) >= SEVERE_SKEW_THRESHOLD:
                    flags.append("severe_skew")
    else:
        kind = "categorical"
        counts = series.dropna().astype(str).value_counts().head(5)
        top_values = {str(k): int(v) for k, v in counts.items()}
        if nunique > HIGH_CARDINALITY_THRESHOLD:
            flags.append("high_cardinality")
        if free_text_avg_words is not None:
            stats["avg_word_count"] = round(free_text_avg_words, 2)

    if missing_pct > HIGH_MISSING_FRACTION * 100:
        flags.append("high_missing")

    semantic_role = _infer_semantic_role(name, series, kind, nunique, flags)
    unit_hint = _infer_unit_hint(name, semantic_role, stats)
    aggregation = _infer_aggregation(name, semantic_role, unit_hint, stats)
    pii = _detect_pii(name, series, kind, semantic_role)

    return ColumnProfile(
        name=name,
        dtype=str(series.dtype),
        kind=kind,
        missing_count=missing,
        missing_pct=missing_pct,
        nunique=nunique,
        stats=stats,
        top_values=top_values,
        flags=flags,
        semantic_role=semantic_role,
        unit_hint=unit_hint,
        aggregation=aggregation,
        pii=pii,
    )


#: Bounded memoisation cache for profile_dataframe(). A single run now calls
#: profile_dataframe on the *same* dataset up to 8 times (twice in
#: controller.load_dataset, once for the dashboard, once each inside five
#: tools), and profiling a wide dataset is not free — this trades a little
#: memory for skipping the repeat work. Small (8 entries): a run touches at
#: most a couple of distinct frames (raw + cleaned), this is headroom, not a
#: general-purpose cache. Values are deep-copied on the way in and out (see
#: profile_dataframe) because callers (controller.py) mutate the returned
#: profile in place (`profile.domains = ...`, `.grain = ...`, `.entity_col
#: = ...`) — sharing the cached object directly would leak one caller's
#: mutation into every other caller's "fresh" profile.
_PROFILE_CACHE_MAX_ENTRIES = 8
_PROFILE_CACHE: OrderedDict[tuple[Any, ...], DatasetProfile] = OrderedDict()
_PROFILE_CACHE_LOCK = threading.Lock()

#: Row-count ceiling for computing a cache-key content fingerprint via
#: pd.util.hash_pandas_object. That hash is a full pass over every cell, so
#: fingerprinting a frame this large would cost more than just re-profiling
#: it — past this size, profile_dataframe skips caching entirely.
_PROFILE_CACHE_MAX_ROWS_FOR_FINGERPRINT = 200_000


def _profile_cache_key(df: pd.DataFrame, target_column: str | None) -> tuple[Any, ...] | None:
    """
    Cache key for a profiling call, or None when the call should not be
    cached at all.

    Keyed on content, not object identity: every tool reads its own copy of
    the dataset, so id(df) would differ on each call and the cache would
    never hit across tools (the repeat-profiling this exists to remove).
    Shape, columns, dtypes and a full-content fingerprint
    (pd.util.hash_pandas_object summed to one int) identify the data itself;
    a hash collision *and* every structural field matching is not a risk
    worth guarding further here. Returns None (caller
    must skip caching) when the frame is over the fingerprint row budget, or
    when hashing fails outright — e.g. unhashable cells (lists/dicts) in an
    object column, which hash_pandas_object cannot handle.
    """
    if len(df) > _PROFILE_CACHE_MAX_ROWS_FOR_FINGERPRINT:
        return None
    try:
        fingerprint = int(pd.util.hash_pandas_object(df, index=True).sum())
    except (TypeError, ValueError):
        return None
    return (
        df.shape,
        tuple(str(c) for c in df.columns),
        tuple(str(dt) for dt in df.dtypes),
        target_column,
        fingerprint,
    )


def clear_profile_cache() -> None:
    """Drop every cached profile. For tests and long-lived processes."""
    with _PROFILE_CACHE_LOCK:
        _PROFILE_CACHE.clear()


def profile_dataframe(df: pd.DataFrame, target_column: str | None = None) -> DatasetProfile:
    """
    Build a full DatasetProfile from a DataFrame.

    Memoised on (shape, columns, dtypes, target_column, content
    fingerprint) — see _profile_cache_key — because the same frame is
    profiled repeatedly within one run (controller.load_dataset does it
    twice, the dashboard once more, and five tools profile their own input
    on every call). Every call still gets its own DatasetProfile instance
    (deep-copied off the cache) since callers mutate the object they get
    back.

    Args:
        df:            The raw (uncleaned) dataset.
        target_column: Optional target — enables class-imbalance checks.
    """
    key = _profile_cache_key(df, target_column)
    if key is not None:
        with _PROFILE_CACHE_LOCK:
            cached = _PROFILE_CACHE.get(key)
            if cached is not None:
                _PROFILE_CACHE.move_to_end(key)
                return copy.deepcopy(cached)

    profile = _profile_dataframe_uncached(df, target_column)

    if key is not None:
        with _PROFILE_CACHE_LOCK:
            _PROFILE_CACHE[key] = copy.deepcopy(profile)
            while len(_PROFILE_CACHE) > _PROFILE_CACHE_MAX_ENTRIES:
                _PROFILE_CACHE.popitem(last=False)
    return profile


def _panel_group_cols(
    df: pd.DataFrame,
    columns: list[ColumnProfile],
    datetime_cols: list[str],
    entity_col: str | None,
) -> list[str]:
    """Columns that are genuine entity x time panel keys: each (level,
    timestamp) pair is near-unique AND the typical level is observed at
    most of the timestamps (every station reports at the same instants). A
    gender column in a transaction log spans the dates too, but carries many
    rows per (gender, date) — a segment, not a panel, and splitting a model
    by it would be wrong; a customer_id that buys on a handful of dates
    fails the coverage test. The detected entity is exempt from
    the cardinality ceiling (a 5,000-sensor panel is still a panel) but not
    from the probe. Probed on the head of the frame to stay cheap."""
    if not datetime_cols:
        return []
    dt = datetime_cols[0]
    sample = df.head(_PANEL_PROBE_ROWS)
    result: list[str] = []
    for col in columns:
        if (
            col.name == dt
            or col.semantic_role not in (SEMANTIC_DIMENSION, SEMANTIC_FLAG, SEMANTIC_IDENTIFIER)
            or (
                col.name != entity_col
                and not _PANEL_GROUP_MIN_CARD <= col.nunique <= _PANEL_GROUP_MAX_CARD
            )
        ):
            continue
        pairs = sample[[col.name, dt]].dropna()
        if pairs.empty or float(pairs.duplicated().mean()) > 0.1:
            continue
        coverage = pairs.groupby(col.name)[dt].nunique() / max(1, pairs[dt].nunique())
        if float(coverage.median()) >= 0.5:
            result.append(col.name)
    return result


def ordinal_scale_columns(columns: list[ColumnProfile]) -> list[ColumnProfile]:
    """Rating-scale items: ordinal-role columns plus unnamed Likert-shaped
    ones (a "q7" holding integers 1..5 is profiled as a measure). Likert
    shape = integer range starting at 0/1 and topping out at a common scale
    maximum, with no more levels than the range allows — and, since a lone
    0..5 column may just be a small count, at least _SURVEY_MIN_ORDINALS
    columns must share that exact scale."""
    named = [c for c in columns if c.semantic_role == SEMANTIC_ORDINAL]
    by_scale: dict[tuple[float, float], list[ColumnProfile]] = {}
    for c in columns:
        lo, hi = c.stats.get("min"), c.stats.get("max")
        if (
            c.semantic_role == SEMANTIC_MEASURE
            and c.unit_hint is None
            and lo in (0, 1)
            and hi in _LIKERT_SCALE_MAX
            and _ORDINAL_CARD_MIN <= c.nunique <= hi - lo + 1
        ):
            by_scale.setdefault((lo, hi), []).append(c)
    shaped = [c for group in by_scale.values() if len(group) >= _SURVEY_MIN_ORDINALS for c in group]
    return named + shaped


def find_experiment_arm(columns: list[ColumnProfile], row_count: int) -> ColumnProfile | None:
    """The experiment assignment column, if any: a dimension/flag whose name
    tokens are all arm words (group, variant, arm, treatment, control,
    cohort — plus qualifiers like test/ab), with 2-5 levels and no level
    under 5% of rows. Shared by archetype detection and the agenda."""
    triggers = (*_ARM_NAME_TOKENS, "ab", "experiment")
    allowed = (*_ARM_NAME_TOKENS, *_ARM_QUALIFIER_TOKENS)
    for c in columns:
        tokens = _name_tokens(c.name)
        if (
            c.semantic_role not in (SEMANTIC_DIMENSION, SEMANTIC_FLAG)
            or not 2 <= c.nunique <= _ARM_MAX_LEVELS
            or not any(t in triggers for t in tokens)
            or not all(t in allowed for t in tokens)
        ):
            continue
        present = row_count - c.missing_count
        if c.top_values and present > 0 and min(c.top_values.values()) / present < _ARM_MIN_LEVEL_SHARE:
            continue
        return c
    return None


def _infer_archetype(
    columns: list[ColumnProfile],
    row_count: int,
    datetime_cols: list[str],
    time_frequency: str | None,
    entity_col: str | None,
    rows_per_entity: float | None,
    panel_group_cols: list[str],
) -> tuple[str, list[str]]:
    """Structural archetype plus the facts that decided it. Checked most
    specific first: experiment, survey (scales), panel, event log, sensor
    series, survey (categorical questionnaire), else cross-section."""
    arm = find_experiment_arm(columns, row_count)
    if arm is not None:
        outcomes = [
            c for c in columns
            if c.name != arm.name and c.semantic_role in (SEMANTIC_MEASURE, SEMANTIC_FLAG, SEMANTIC_ORDINAL)
        ]
        if outcomes:
            return ARCHETYPE_EXPERIMENT, [
                f"assignment column '{arm.name}' with {arm.nunique} levels",
                f"{len(outcomes)} outcome candidates (e.g. '{outcomes[0].name}')",
            ]

    scales = ordinal_scale_columns(columns)
    if len(scales) >= _SURVEY_MIN_ORDINALS:
        return ARCHETYPE_SURVEY, [
            f"{len(scales)} ordinal/rating-scale columns",
            f"e.g. {', '.join(repr(c.name) for c in scales[:3])}",
        ]

    time_col = datetime_cols[0] if datetime_cols else None
    regular = time_frequency in _REGULAR_FREQUENCIES
    if time_col and panel_group_cols and regular:
        return ARCHETYPE_PANEL, [
            f"'{panel_group_cols[0]}' observed at most '{time_col}' timestamps",
            f"{time_frequency} frequency",
        ]

    if time_col and entity_col and rows_per_entity and rows_per_entity > _ENTITY_REPEAT_THRESHOLD:
        return ARCHETYPE_EVENT_LOG, [
            f"{rows_per_entity:.1f} rows per {entity_col}",
            f"'{time_col}' timestamps {time_frequency or 'sparse'}, not a shared grid across entities",
        ]

    dims = [c for c in columns if c.semantic_role == SEMANTIC_DIMENSION and c.kind == "categorical"]
    measures = [c for c in columns if c.semantic_role == SEMANTIC_MEASURE]
    time_profile = next((c for c in columns if c.name == time_col), None)
    unique_instants = time_profile is not None and time_profile.nunique >= 0.95 * row_count
    if (
        time_col
        and row_count >= MIN_TIME_SERIES_ROWS
        and entity_col is None
        and (regular or (unique_instants and not dims))
        and len(dims) <= 2
        and len(measures) >= max(1, (len(columns) - 1) // 2)
    ):
        return ARCHETYPE_SENSOR, [
            f"'{time_col}' {time_frequency or 'timestamped'} axis",
            f"{len(measures)} numeric measures, {len(dims)} categorical columns",
        ]

    low_card = [c for c in dims if 2 <= c.nunique <= _SURVEY_MAX_LEVELS]
    respondent = next((c for c in columns if c.kind == "identifier"), None)
    if (
        respondent is not None
        and len(low_card) >= _SURVEY_MIN_CATEGORICALS
        and len(low_card) >= _SURVEY_MIN_CATEGORICAL_SHARE * len(columns)
    ):
        return ARCHETYPE_SURVEY, [
            f"{len(low_card)} of {len(columns)} columns are low-cardinality categoricals",
            f"respondent key '{respondent.name}'",
        ]

    key = next((c.name for c in columns if c.kind == "identifier" and c.nunique == row_count), None)
    evidence = [f"one row per '{key}'"] if key else []
    evidence.append("no dominant time axis" if not time_col else f"'{time_col}' is not a regular series axis")
    return ARCHETYPE_CROSS_SECTION, evidence


def _profile_dataframe_uncached(df: pd.DataFrame, target_column: str | None = None) -> DatasetProfile:
    """The actual profiling work — see profile_dataframe() for the memoised
    public entry point every caller should use instead of this."""
    row_count = len(df)
    duplicate_rows = int(df.duplicated().sum())
    memory_mb = round(float(df.memory_usage(deep=True).sum()) / 1_048_576, 2)

    columns = [_profile_column(str(c), df[c], row_count) for c in df.columns]

    warnings: list[str] = []
    penalty = 0

    total_cells = max(1, row_count * max(1, len(df.columns)))
    missing_frac = float(df.isna().sum().sum()) / total_cells
    if missing_frac > 0:
        penalty += min(25, int(missing_frac * 100))
        if missing_frac > 0.05:
            warnings.append(f"{missing_frac:.0%} of all cells are missing.")

    if duplicate_rows:
        dup_frac = duplicate_rows / max(1, row_count)
        penalty += min(15, int(dup_frac * 100))
        warnings.append(f"{duplicate_rows} duplicate rows ({dup_frac:.1%}).")

    for col in columns:
        if col.kind == "constant":
            penalty += 3
            warnings.append(f"Column '{col.name}' is constant — carries no signal.")
        if "high_missing" in col.flags:
            penalty += 4
            warnings.append(f"Column '{col.name}' is {col.missing_pct:.0f}% missing.")
        if "high_cardinality" in col.flags:
            penalty += 2
            warnings.append(
                f"Column '{col.name}' has {col.nunique} categories — "
                "one-hot encoding would explode; consider dropping or target-encoding."
            )

    if target_column and target_column in df.columns:
        tgt = df[target_column].dropna()
        if not tgt.empty and not pd.api.types.is_float_dtype(tgt) and tgt.nunique() <= 20:
            counts = tgt.value_counts()
            if len(counts) >= 2:
                minority_frac = float(counts.iloc[-1]) / float(counts.sum())
                if minority_frac < 0.10:
                    penalty += 5
                    warnings.append(
                        f"Target '{target_column}' is imbalanced — minority class is "
                        f"{minority_frac:.1%}. Prefer F1/recall over accuracy."
                    )

    if row_count < 100:
        penalty += 10
        warnings.append(f"Only {row_count} rows — results will have high variance.")

    is_sufficient = True
    sufficiency_reason: str | None = None
    if row_count < MIN_ROWS_HARD_FLOOR:
        is_sufficient = False
        sufficiency_reason = (
            f"Only {row_count} row(s) — not enough data for any analysis "
            "to produce a meaningful result."
        )
        warnings.append(sufficiency_reason)
    elif row_count < MIN_ROWS_RELIABLE:
        warnings.append(
            f"Only {row_count} rows — below {MIN_ROWS_RELIABLE}, no "
            "inferential result (hypothesis test, trained model) should be trusted."
        )

    quality_score = max(0, 100 - penalty)
    if not is_sufficient:
        quality_score = min(quality_score, INSUFFICIENT_QUALITY_CAP)

    # ---- Dataset-nature facts — structured, not prose, so gating logic
    #      (ToolRegistry.candidate_tools) can branch on them directly ----
    datetime_cols = [c.name for c in columns if c.kind == "datetime"]
    is_time_series = bool(datetime_cols) and row_count >= MIN_TIME_SERIES_ROWS

    text_cols = [c.name for c in columns if c.kind == "text"]

    # Integer-coded dimensions/keys (year, store_id) are stored as numbers
    # but are not features in the multicollinearity sense.
    numeric_cols = [
        c.name for c in columns
        if c.kind == "numeric" and c.semantic_role not in (SEMANTIC_DIMENSION, SEMANTIC_IDENTIFIER)
    ]
    geo_lat_col: str | None = None
    geo_lon_col: str | None = None
    for col in columns:
        if col.kind != "numeric":
            continue
        name_l = col.name.lower()
        lo, hi = col.stats.get("min"), col.stats.get("max")
        if lo is None or hi is None:
            continue
        if geo_lat_col is None and any(h in name_l for h in _LAT_NAME_HINTS) and -90.0 <= lo and hi <= 90.0:
            geo_lat_col = col.name
        elif geo_lon_col is None and any(h in name_l for h in _LON_NAME_HINTS) and -180.0 <= lo and hi <= 180.0:
            geo_lon_col = col.name

    is_high_dimensional = len(numeric_cols) >= HIGH_DIMENSIONAL_THRESHOLD

    # ---- Grain / entity structure (T1) — one row *is* what? An identifier
    # column that is 100% unique is a clean row key; failing that, a
    # column whose name says it is an entity (customer, sensor, player,
    # station...) and whose rows-per-value ratio is well above 1 describes a
    # repeat-row grain (e.g. "10 rows per customer") even with no
    # single-column key. The most-repeating candidate wins (as before the
    # hint list was generalised); hint order only breaks ties. ----
    grain: list[str] = []
    entity_col: str | None = None
    rows_per_entity: float | None = None
    id_key = next((c.name for c in columns if c.kind == "identifier" and c.nunique == row_count and row_count > 0), None)
    if id_key:
        grain = [id_key]
    entity_hints = _ENTITY_NAME_HINTS + (_GEO_ENTITY_NAME_HINTS if datetime_cols else ())
    ranked: list[tuple[float, int, ColumnProfile]] = []
    for c in columns:
        if c.semantic_role not in (SEMANTIC_IDENTIFIER, SEMANTIC_DIMENSION) or not 1 < c.nunique < row_count:
            continue
        ratio = _rows_per_value(row_count, c.nunique)
        if ratio < _ENTITY_REPEAT_THRESHOLD:
            continue
        tokens = _name_tokens(c.name)
        if tokens and tokens[-1] in entity_hints:
            hint = tokens[-1]
        elif tokens and tokens[-1] in (*_CODED_KEY_TOKENS, "name"):
            hint = next((t for t in tokens[:-1] if t in entity_hints), "")
        else:
            hint = ""
        if hint:
            ranked.append((-ratio, entity_hints.index(hint), c))
    if ranked:
        neg_ratio, _, best = min(ranked, key=lambda item: item[:2])
        entity_col = best.name
        rows_per_entity = round(-neg_ratio, 2)
        if not grain:
            grain = [best.name] + (datetime_cols[:1] if datetime_cols else [])

    panel_group_cols = _panel_group_cols(df, columns, datetime_cols, entity_col)
    time_frequency = next(
        (c.stats.get("frequency") for c in columns if c.kind == "datetime"), None
    )

    archetype: str | None = None
    archetype_evidence: list[str] = []
    if is_sufficient:
        archetype, archetype_evidence = _infer_archetype(
            columns, row_count, datetime_cols, time_frequency,
            entity_col, rows_per_entity, panel_group_cols,
        )

    pii_cols = [c for c in columns if c.pii]
    if pii_cols:
        listed = ", ".join(f"'{c.name}' ({c.pii})" for c in pii_cols[:6])
        more = f" and {len(pii_cols) - 6} more" if len(pii_cols) > 6 else ""
        warnings.append(
            f"Possible personal data (PII) in {listed}{more} — values are withheld "
            "from LLM prompts unless REDACT_PII=false."
        )

    return DatasetProfile(
        row_count=row_count,
        column_count=len(df.columns),
        duplicate_rows=duplicate_rows,
        memory_mb=memory_mb,
        columns=columns,
        warnings=warnings,
        quality_score=quality_score,
        datetime_cols=datetime_cols,
        is_time_series=is_time_series,
        text_cols=text_cols,
        geo_lat_col=geo_lat_col,
        geo_lon_col=geo_lon_col,
        is_high_dimensional=is_high_dimensional,
        panel_group_cols=panel_group_cols,
        is_sufficient=is_sufficient,
        sufficiency_reason=sufficiency_reason,
        grain=grain,
        entity_col=entity_col,
        rows_per_entity=rows_per_entity,
        time_frequency=time_frequency,
        archetype=archetype,
        archetype_evidence=archetype_evidence,
    )
