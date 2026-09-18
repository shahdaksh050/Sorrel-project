"""
Type coercion / repair pass — numerics trapped in strings (U0.7).

Real-world exports routinely put numeric values inside strings: "$123.45",
"45.3%", "1,234,567". Left alone these become `identifier` or
high-cardinality `categorical` columns and are silently dropped from every
numeric analysis while also being counted against the quality score.

Runs once, at ingestion, before profiling — coerce_types() never runs
against already-profiled dtypes, so profile_dataframe() always sees the
repaired frame.

Every coercion is recorded and reported, never silent — the same
discipline ml_pipeline.py applies to `treatments_applied`.
"""
from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from src.core.profiler import has_identifier_name_hint

#: Fraction of non-null values that must match a rule before a column is coerced.
COERCE_MATCH_THRESHOLD = 0.95

_CURRENCY_RE = re.compile(r"^[\$€£¥]\s*-?[\d,]+(\.\d+)?$|^-?[\d,]+(\.\d+)?\s*[\$€£¥]$")
_PERCENT_RE = re.compile(r"^-?[\d,]+(\.\d+)?\s*%$")
_THOUSANDS_RE = re.compile(r"^-?\d{1,3}(,\d{3})+$")
_BOOL_TRUE = {"y", "yes", "true", "t"}
_BOOL_FALSE = {"n", "no", "false", "f"}
_COMMON_SENTINELS: frozenset[str] = frozenset({
    "?", "-", "--", "na", "n/a", "#n/a", "null", "none", "nan", "", "missing", "unknown"
})


@dataclass
class Coercion:
    """One column's type repair — what changed, and what didn't parse."""

    column: str
    from_kind: str          # "string"
    to_kind: str             # "numeric" | "boolean" | "datetime"
    rule: str                 # currency|percent|thousands|yes_no|numeric|date_*
    n_converted: int
    n_failed: int
    failed_examples: list[str] = field(default_factory=list)
    is_sentinel_only: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "column": self.column,
            "from_kind": self.from_kind,
            "to_kind": self.to_kind,
            "rule": self.rule,
            "n_converted": self.n_converted,
            "n_failed": self.n_failed,
            "failed_examples": self.failed_examples,
            "is_sentinel_only": self.is_sentinel_only,
        }


def _parse_currency(s: str) -> float | None:
    if not _CURRENCY_RE.match(s):
        return None
    cleaned = re.sub(r"[\$€£¥,]", "", s)
    try:
        return float(cleaned)
    except ValueError:
        return None


def _parse_percent(s: str) -> float | None:
    if not _PERCENT_RE.match(s):
        return None
    cleaned = s.rstrip("%").replace(",", "")
    try:
        return float(cleaned) / 100.0
    except ValueError:
        return None


def _parse_thousands(s: str) -> float | None:
    if not _THOUSANDS_RE.match(s):
        return None
    try:
        return float(s.replace(",", ""))
    except ValueError:
        return None


def _parse_bool(s: str) -> bool | None:
    t = s.lower()
    if t in _BOOL_TRUE:
        return True
    if t in _BOOL_FALSE:
        return False
    return None


def _make_numeric_generic_parser(decimal_comma: bool) -> Callable[[str], float | None]:
    def _parse(s: str) -> float | None:
        # European decimals ("4,5" meaning 4.5) collide with US thousands
        # grouping ("4,500"). Only read ',' as a decimal point when the
        # file's sniffed delimiter was ';' (src.core.io.ReadReport) — the
        # signal that this file is EU-locale in the first place.
        t = s.replace(",", ".") if decimal_comma else s.replace(",", "")
        try:
            return float(t)
        except ValueError:
            return None

    return _parse


#: (rule name, target kind, parser) — checked in this order per column.
_RULES: list[tuple[str, str, Callable[[str], Any]]] = [
    ("currency", "numeric", _parse_currency),
    ("percent", "numeric", _parse_percent),
    ("thousands", "numeric", _parse_thousands),
    ("yes_no", "boolean", _parse_bool),
]

#: A date written with separators: 09/02/2023, 2023-02-09, 9.2.2023, with an
#: optional time part. Deliberately narrow — arbitrary prose must not be fed
#: to the date parser, which is both slow and prone to false positives.
_DATE_LIKE_RE = re.compile(
    r"^\s*\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}"
    r"([ T]\d{1,2}:\d{2}(:\d{2})?(\.\d+)?\s*(Z|[+-]\d{2}:?\d{2})?)?\s*$"
)

#: The first two numeric components of a separator-style date.
_DATE_PARTS_RE = re.compile(r"^\s*(\d{1,4})[-/.](\d{1,2})[-/.](\d{1,4})")


def _detect_date_convention(values: pd.Series) -> tuple[str, bool] | None:
    """
    Decide whether a date column is day-first, month-first, or ISO.

    pandas defaults to month-first, so a `dd/mm/yyyy` export — the norm
    across most of the world — silently turns every day>12 into NaT and
    *silently swaps day and month on the rows that survive*. On a real
    1500-row shop export that dropped 63% of rows and shifted the date
    range by five months, with nothing reported.

    The convention is read off the data: a first component above 12 can
    only be a day, a second component above 12 can only be a day in
    month-first order. When every row is ambiguous (all components <= 12)
    there is genuinely no way to tell, so pandas' default is kept and the
    caller records the ambiguity rather than implying certainty.

    Returns (rule_name, dayfirst) or None when this is not a date column.
    """
    # Ponytail: sample unique dates across the series so high-frequency
    # (minute/hourly) datasets reveal days 13..31 even when head(2000) only covers Day 1.
    sample = values.drop_duplicates().head(2000)
    if sample.empty:
        return None
    matches = sample.str.match(_DATE_LIKE_RE)
    if matches.mean() < COERCE_MATCH_THRESHOLD:
        return None

    parts = sample.str.extract(_DATE_PARTS_RE).dropna()
    if parts.empty:
        return None
    first = pd.to_numeric(parts[0], errors="coerce")
    second = pd.to_numeric(parts[1], errors="coerce")

    # A 4-digit leading component is ISO (yyyy-mm-dd) — unambiguous.
    if bool((first > 31).any()):
        return "date_iso", False
    if bool((first > 12).any()):
        return "date_dayfirst", True
    if bool((second > 12).any()):
        return "date_monthfirst", False
    return "date_ambiguous", False


def _parse_dates(values: pd.Series, dayfirst: bool) -> pd.Series:
    # Ponytail: format="mixed" parses with pandas C engine and suppresses the dateutil UserWarning
    return pd.to_datetime(values, errors="coerce", dayfirst=dayfirst, format="mixed")


def _try_coerce_column(
    values: pd.Series, decimal_comma: bool
) -> tuple[str, str, pd.Series] | None:
    total = len(values)
    if total == 0:
        return None
    for rule, to_kind, parser in _RULES:
        parsed = values.map(parser)
        if parsed.notna().sum() / total >= COERCE_MATCH_THRESHOLD:
            return rule, to_kind, parsed
    generic_parsed = values.map(_make_numeric_generic_parser(decimal_comma))
    if generic_parsed.notna().sum() / total >= COERCE_MATCH_THRESHOLD:
        return "numeric", "numeric", generic_parsed
    return None


def coerce_types(df: pd.DataFrame, delimiter: str | None = None) -> tuple[pd.DataFrame, list[Coercion]]:
    """
    Repair numeric/boolean values trapped in string columns.

    Args:
        df:        Raw (uncoerced) dataset, straight from src.core.io.read_any.
        delimiter: The delimiter src.core.io.ReadReport sniffed/assumed for
                   this file — ';' signals a European-locale export, which
                   changes how a lone ',' inside a number is interpreted.

    Returns:
        (repaired_df, coercions) — coercions is empty when nothing changed.
        Every entry is a real, reportable change; nothing here is silent.
    """
    out = df.copy()
    coercions: list[Coercion] = []
    decimal_comma = delimiter == ";"

    for col in out.columns:
        series = out[col]
        if (
            pd.api.types.is_numeric_dtype(series)
            or pd.api.types.is_bool_dtype(series)
            or pd.api.types.is_datetime64_any_dtype(series)
        ):
            continue
        if has_identifier_name_hint(str(col)):
            continue

        non_null = series.dropna().astype(str).str.strip()
        non_null = non_null[non_null != ""]
        if non_null.empty:
            continue

        # Dates first: a date column must never reach the numeric rules, and
        # parsing it once here means no downstream tool re-parses it with
        # pandas' month-first default and quietly disagrees.
        date_rule = _detect_date_convention(non_null)
        if date_rule is not None:
            rule_name, dayfirst = date_rule
            parsed_dates = _parse_dates(non_null, dayfirst)
            converted = int(parsed_dates.notna().sum())
            if converted and converted / len(non_null) >= COERCE_MATCH_THRESHOLD:
                failed = non_null[parsed_dates.isna()]
                new_dates = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")
                new_dates.loc[non_null.index] = parsed_dates
                out[col] = new_dates
                date_failed = [str(x) for x in dict.fromkeys(failed.tolist())][:5]
                coercions.append(
                    Coercion(
                        column=str(col),
                        from_kind="string",
                        to_kind="datetime",
                        rule=rule_name,
                        n_converted=converted,
                        n_failed=len(non_null) - converted,
                        failed_examples=date_failed,
                        is_sentinel_only=bool(date_failed and all(str(x).strip().lower() in _COMMON_SENTINELS for x in date_failed)),
                    )
                )
                continue

        result = _try_coerce_column(non_null, decimal_comma)
        if result is None:
            continue
        rule, to_kind, parsed = result

        matched_mask = parsed.notna()
        n_converted = int(matched_mask.sum())
        n_failed = int((~matched_mask).sum())
        if n_converted == 0:
            continue

        failed_examples = list(dict.fromkeys(non_null[~matched_mask].tolist()))[:5]
        failed_examples_str = [str(x) for x in failed_examples]
        is_sentinel = bool(failed_examples and all(str(x).strip().lower() in _COMMON_SENTINELS for x in failed_examples))

        new_col = pd.Series(pd.NA, index=series.index, dtype=object)
        new_col.loc[non_null.index] = parsed.to_numpy()

        if to_kind == "boolean":
            out[col] = new_col.astype("boolean")
        else:
            out[col] = pd.to_numeric(new_col, errors="coerce")

        coercions.append(
            Coercion(
                column=str(col),
                from_kind="string",
                to_kind=to_kind,
                rule=rule,
                n_converted=n_converted,
                n_failed=n_failed,
                failed_examples=failed_examples_str,
                is_sentinel_only=is_sentinel,
            )
        )

    # Date + Time fusion: if there is a datetime column with midnight timestamps
    # and a companion column containing time-of-day strings (HH:MM:SS), fuse them
    # into a single comprehensive timestamp series so time-series tools have full
    # sub-daily resolution.
    time_cols = [
        c for c in out.columns
        if not pd.api.types.is_numeric_dtype(out[c])
        and not pd.api.types.is_datetime64_any_dtype(out[c])
        and not pd.api.types.is_bool_dtype(out[c])
    ]
    date_cols = [
        c for c in out.columns
        if pd.api.types.is_datetime64_any_dtype(out[c])
    ]
    if date_cols and time_cols:
        for d_col in date_cols:
            d_series = out[d_col].dropna()
            if d_series.empty:
                continue
            sample_d = d_series.head(100)
            if not ((sample_d.dt.hour == 0).all() and (sample_d.dt.minute == 0).all() and (sample_d.dt.second == 0).all()):
                continue
            for t_col in time_cols:
                t_non_null = out[t_col].dropna().astype(str).str.strip()
                if t_non_null.empty:
                    continue
                sample_t = t_non_null.head(100)
                if (sample_t.str.match(r"^\d{1,2}:\d{2}(?::\d{2}(?:\.\d+)?)?$")).mean() >= 0.9:
                    timedeltas = pd.to_timedelta(t_non_null, errors="coerce")
                    valid_td = timedeltas.notna()
                    if valid_td.mean() >= COERCE_MATCH_THRESHOLD:
                        out[d_col] = out[d_col].add(timedeltas, fill_value=pd.Timedelta(0))
                        coercions.append(
                            Coercion(
                                column=str(d_col),
                                from_kind="date",
                                to_kind="datetime",
                                rule=f"date_time_fusion:{t_col}",
                                n_converted=int(valid_td.sum()),
                                n_failed=int((~valid_td).sum()),
                                failed_examples=[str(x) for x in t_non_null[~valid_td].head(5)],
                                is_sentinel_only=False,
                            )
                        )
                        break

    return out, coercions
