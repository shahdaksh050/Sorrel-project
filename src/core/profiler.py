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

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import pandas as pd

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

#: Column-name fragments that hint a measure is money.
_CURRENCY_NAME_HINTS = (
    "price", "amount", "revenue", "cost", "salary", "income", "sales",
    "profit", "margin", "spend", "fee", "charge", "payment", "balance",
    "value", "wage", "expense", "discount", "fare", "budget", "arpu",
)
#: Column-name fragments that hint a measure is a percentage/rate already
#: expressed on a 0-100 or 0-1 scale.
_PERCENT_NAME_HINTS = ("pct", "percent", "rate", "ratio", "share", "margin_pct")
#: Column-name fragments that hint a measure is a plain count.
_COUNT_NAME_HINTS = ("count", "qty", "quantity", "units", "orders", "visits", "clicks", "views")

#: Column-name fragments that hint a column identifies an "entity" whose
#: repeat-row structure defines the dataset's grain (one row per order line,
#: many rows per customer). Mirrors the entity roles src.core.domains looks
#: for, kept independent to avoid a profiler -> domains import cycle
#: (domains.py imports DatasetProfile from this module).
_ENTITY_NAME_HINTS = ("customer", "client", "user", "account", "patient", "member", "employee", "subscriber")

#: Rows-per-unique-value above which a candidate entity column is treated as
#: this dataset's grain-defining entity (i.e. genuinely repeats, not just
#: incidentally non-unique).
_ENTITY_REPEAT_THRESHOLD = 1.5


def _rows_per_value(df: pd.DataFrame, column: str) -> float:
    """Mean rows per distinct value of `column`. 1.0 = one row per value."""
    nunique = int(df[column].nunique(dropna=True))
    return len(df) / nunique if nunique else 0.0


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
        name_l = name.lower()
        if (
            is_integer_valued
            and _ORDINAL_CARD_MIN <= nunique <= _ORDINAL_CARD_MAX
            and any(h in name_l for h in _ORDINAL_NAME_HINTS)
        ):
            return SEMANTIC_ORDINAL
        return SEMANTIC_MEASURE
    return SEMANTIC_DIMENSION


def _infer_unit_hint(name: str, semantic_role: str) -> str | None:
    """Cheap name-based unit inference for measures — a fuller version would
    thread src.core.coercion's currency/percent detection through, but that
    detection happens at read time and is discarded before profiling runs
    today; this heuristic gets the common cases without that plumbing."""
    if semantic_role != SEMANTIC_MEASURE:
        return None
    name_l = name.lower()
    if any(h in name_l for h in _PERCENT_NAME_HINTS):
        return "percent"
    if any(h in name_l for h in _CURRENCY_NAME_HINTS):
        return "currency"
    if any(h in name_l for h in _COUNT_NAME_HINTS):
        return "count"
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
    stats: dict[str, float] = field(default_factory=dict)      # numeric columns
    top_values: dict[str, int] = field(default_factory=dict)   # categorical columns
    flags: list[str] = field(default_factory=list)

    # ---- T1 semantic layer (7.3) ----
    semantic_role: str = SEMANTIC_DIMENSION   # measure | dimension | flag | ordinal | identifier | time | text | constant
    unit_hint: str | None = None              # currency | percent | count | None

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

    def to_prompt_string(self, max_warnings: int = 8) -> str:
        """
        Compact profile summary (~150 tokens) for LLM context injection.

        Column names are dataset-derived (untrusted) and are sanitised before
        they reach a prompt. Warnings embed column names too, so they pass
        through the same sanitiser.
        """
        from src.core.security import sanitize_for_prompt as _sp

        kind_counts: dict[str, int] = {}
        for col in self.columns:
            kind_counts[col.kind] = kind_counts.get(col.kind, 0) + 1
        kinds = ", ".join(f"{v} {k}" for k, v in sorted(kind_counts.items()))
        lines = [
            f"Data profile: quality score {self.quality_score}/100; "
            f"{self.duplicate_rows} duplicate rows; column kinds: {kinds}.",
        ]
        if not self.is_sufficient:
            lines.append(f"INSUFFICIENT DATA: {self.sufficiency_reason or 'too few rows.'}")
        skewed = [_sp(c.name) for c in self.columns if "severe_skew" in c.flags]
        if skewed:
            lines.append(f"Severely skewed numerics: {', '.join(skewed[:6])}.")
        ids = [_sp(c.name) for c in self.columns if c.kind == "identifier"]
        if ids:
            lines.append(f"Identifier columns (exclude from modelling): {', '.join(ids[:6])}.")
        nature: list[str] = []
        if self.is_time_series:
            nature.append(f"time-series (datetime column(s): {', '.join(_sp(c) for c in self.datetime_cols[:3])})")
        if self.text_cols:
            nature.append(f"free-text column(s): {', '.join(_sp(c) for c in self.text_cols[:3])}")
        if self.has_geo():
            nature.append(f"geographic coordinates ({_sp(self.geo_lat_col or '')}, {_sp(self.geo_lon_col or '')})")
        if self.is_high_dimensional:
            nature.append("high-dimensional (many numeric features — watch multicollinearity)")
        if self.panel_group_cols:
            nature.append(f"grouped/panel structure via: {', '.join(_sp(c) for c in self.panel_group_cols[:3])}")
        if self.entity_col and self.rows_per_entity:
            nature.append(
                f"repeat-row grain via '{_sp(self.entity_col)}' "
                f"({self.rows_per_entity:.1f} rows/entity)"
            )
        if nature:
            lines.append("Data nature: " + "; ".join(nature) + ".")
        measures = [_sp(c.name) for c in self.measures()]
        flags = [_sp(c.name) for c in self.columns if c.semantic_role == SEMANTIC_FLAG]
        if measures:
            lines.append(f"Measures (summable quantities): {', '.join(measures[:8])}.")
        if flags:
            lines.append(f"Binary flags (not measures — never sum/log-transform): {', '.join(flags[:8])}.")
        for match in self.domains:
            roles = ", ".join(f"{r}={_sp(c)}" for r, c in sorted(match.roles.items()))
            lines.append(
                f"Data domain: {match.domain} (confidence {match.confidence:.2f}) "
                f"— roles: {roles}."
            )
        if self.warnings:
            lines.append(
                "Warnings: " + " | ".join(_sp(w, max_len=160) for w in self.warnings[:max_warnings])
            )
        return "\n".join(lines)


def _is_datetime_like(series: pd.Series) -> bool:
    """True for datetime dtypes or string columns that parse as dates."""
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
        return False
    sample = series.dropna().head(20)
    if sample.empty:
        return False
    try:
        parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
    except (ValueError, TypeError):
        return False
    return bool(parsed.notna().mean() >= 0.9)


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
    stats: dict[str, float] = {}
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
    # Datetime check must precede the identifier check: a daily time index is
    # 100% unique but is a time axis, not an ID.
    elif _is_datetime_like(series):
        kind = "datetime"
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
    unit_hint = _infer_unit_hint(name, semantic_role)

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
    )


def profile_dataframe(df: pd.DataFrame, target_column: str | None = None) -> DatasetProfile:
    """
    Build a full DatasetProfile from a DataFrame.

    Args:
        df:            The raw (uncleaned) dataset.
        target_column: Optional target — enables class-imbalance checks.
    """
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

    numeric_cols = [c.name for c in columns if c.kind == "numeric"]
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

    panel_group_cols: list[str] = []
    if datetime_cols:
        for col in columns:
            if (
                col.kind in ("categorical", "boolean")
                and _PANEL_GROUP_MIN_CARD <= col.nunique <= _PANEL_GROUP_MAX_CARD
            ):
                panel_group_cols.append(col.name)

    # ---- Grain / entity structure (T1) — one row *is* what? An identifier
    # column that is 100% unique is a clean row key; failing that, a
    # dimension whose name hints at an entity (customer, order, patient...)
    # and whose rows-per-value ratio is well above 1 describes a repeat-row
    # grain (e.g. "10 rows per customer") even with no single-column key. ----
    grain: list[str] = []
    entity_col: str | None = None
    rows_per_entity: float | None = None
    id_key = next((c.name for c in columns if c.kind == "identifier" and c.nunique == row_count and row_count > 0), None)
    if id_key:
        grain = [id_key]
    entity_candidates = [
        c for c in columns
        if c.semantic_role in (SEMANTIC_IDENTIFIER, SEMANTIC_DIMENSION)
        and any(h in c.name.lower() for h in _ENTITY_NAME_HINTS)
        and 0 < c.nunique < row_count
    ]
    if entity_candidates:
        best = max(entity_candidates, key=lambda c: _rows_per_value(df, c.name))
        ratio = _rows_per_value(df, best.name)
        if ratio >= _ENTITY_REPEAT_THRESHOLD:
            entity_col = best.name
            rows_per_entity = round(ratio, 2)
            if not grain:
                grain = [best.name] + (datetime_cols[:1] if datetime_cols else [])

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
    )
