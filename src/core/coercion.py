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

import numpy as np
import pandas as pd

from src.core.profiler import has_identifier_name_hint
from src.core.sentinels import format_value, null_sentinels

#: Fraction of non-null values that must match a rule before a column is coerced.
COERCE_MATCH_THRESHOLD = 0.95

_CURRENCY_RE = re.compile(r"^[\$€£¥]\s*-?[\d,]+(\.\d+)?$|^-?[\d,]+(\.\d+)?\s*[\$€£¥]$")
_PERCENT_RE = re.compile(r"^-?[\d,]+(\.\d+)?\s*%$")
_THOUSANDS_RE = re.compile(r"^-?\d{1,3}(,\d{3})+$")
#: A comma NOT followed by exactly three digits can never be a thousands separator.
_NON_THOUSANDS_COMMA_RE = re.compile(r",(?!\d{3}(?:\D|$))")
#: "0,7578" / "-0,5": thousands groups never start with zero.
_LEADING_ZERO_COMMA_RE = re.compile(r"^-?0,")
#: "1,234.50": comma before the dot is the US thousands+decimal layout.
_THOUSANDS_DECIMAL_RE = re.compile(r",.*\.")
_LEADING_ZERO_RE = re.compile(r"^0\d")
_BOOL_TRUE ={"y", "yes", "true", "t"}
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
    detail: str = ""        # human-readable disclosure (rule "sentinel")

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
            "detail": self.detail,
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


def _is_decimal_comma(values: pd.Series, semicolon_file: bool) -> bool:
    """
    Data-driven: does ',' act as the decimal separator in this column?

    Yes when any comma is followed by a digit count other than exactly three
    ("2,6", "0,7578") or a value starts "0," / "-0,"; a ';'-delimited file
    also counts when a comma value is not a valid thousands grouping. No when
    every comma is a three-digit group ("1,234", "1,234,567") or a value is
    laid out "1,234.50".
    """
    commas = values[values.str.contains(",", regex=False)]
    if commas.empty or bool(commas.str.contains(_THOUSANDS_DECIMAL_RE).any()):
        return False
    if bool(commas.str.contains(_NON_THOUSANDS_COMMA_RE).any()) or bool(
        commas.str.contains(_LEADING_ZERO_COMMA_RE).any()
    ):
        return True
    return semicolon_file and not bool(commas.str.match(_THOUSANDS_RE).all())


def _make_numeric_generic_parser(decimal_comma: bool) -> Callable[[str], float | None]:
    def _parse(s: str) -> float | None:
        # European decimals ("4,5" meaning 4.5) collide with US thousands
        # grouping ("4,500"); _is_decimal_comma decides per column.
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


#: Time of day: HH:MM, HH:MM:SS, HH.MM.SS (dotted only with seconds, so a
#: decimal such as "12.30" is never read as a time).
_TIME_OF_DAY_RE = re.compile(r"^([01]?\d|2[0-3])(?::([0-5]\d)(?::([0-5]\d))?|\.([0-5]\d)\.([0-5]\d))$")


def _parse_time_of_day(values: pd.Series) -> pd.Series:
    """Vectorised time-of-day strings to timedeltas (NaT where not a time)."""
    ex = values.str.extract(_TIME_OF_DAY_RE)
    hh = pd.to_numeric(ex[0], errors="coerce")
    mm = pd.to_numeric(ex[1].fillna(ex[3]), errors="coerce")
    ss = pd.to_numeric(ex[2].fillna(ex[4]), errors="coerce").fillna(0)
    secs = (hh * 3600 + mm * 60 + ss).where(hh.notna() & mm.notna())
    return pd.to_timedelta(secs, unit="s")


def _try_coerce_column(
    values: pd.Series, decimal_comma: bool
) -> tuple[str, str, pd.Series] | None:
    total = len(values)
    if total == 0:
        return None
    for rule, to_kind, parser in _RULES:
        if rule == "thousands" and decimal_comma:
            continue
        parsed = values.map(parser)
        if parsed.notna().sum() / total >= COERCE_MATCH_THRESHOLD:
            return rule, to_kind, parsed
    generic_parsed = values.map(_make_numeric_generic_parser(decimal_comma))
    if generic_parsed.notna().sum() / total >= COERCE_MATCH_THRESHOLD:
        return "numeric", "numeric", generic_parsed
    return None


# --- Censored / scientific values ---------------------------------------
#: Lab and environmental exports put non-numeric markers in numeric columns:
#: "<0.5" (below a detection limit), "ND", "BDL", "trace". Numbers may also
#: carry a unicode minus, a decimal-comma exponent ("1,2E-3"), a "± error"
#: and a trailing unit ("12.5 mg/L").
_NUMBER_PAT = r"[-+]?(?:\d+(?:[.,]\d+)?|[.,]\d+)(?:[eE][-+]?\d+)?"
_VALUE_RE = re.compile(
    rf"^\s*(?P<op><=?|≤|≦)?\s*(?P<num>{_NUMBER_PAT})\s*"
    rf"(?:(?:±|\+/-|\+-)\s*{_NUMBER_PAT}\s*)?"
    r"(?P<unit>[A-Za-zµμ°][A-Za-z0-9µμ°/^*·.\-²³%]{0,15})?\s*$"
)
_ND_RE = re.compile(
    r"(?i)^\s*(?:n\.?\s?d\.?|n/d|b\.?d\.?l\.?|<\s*(?:lod|loq|lloq|dl|mdl|mql|rl)"
    r"|below\s+(?:the\s+)?(?:detection|quantitation|quantification|reporting)\s+limit"
    r"|below\s+(?:lod|loq|dl)|trace|not\s+detected|non-?detects?)\s*$"
)
#: A column must look numeric-ish (first unique values) before the censored
#: pass does any full-column work — keeps free-text columns free.
_NUMERICISH_RE = re.compile(r"^\s*(?:[<≤≦]\s*)?[-−+]?[.,]?\d")
_CENSORED_MIN_PARSED = 0.80
_MAX_DISTINCT_UNITS = 3


@dataclass
class _CensoredResult:
    values: np.ndarray            # float per row of the non-null series
    below: np.ndarray             # bool per row: value was censored / ND
    n_below: int
    n_failed: int
    failed_examples: list[str]
    marker_examples: list[str]
    units: list[str]
    n_nd_unfilled: int


def _try_censored_numeric(values: pd.Series, decimal_comma: bool) -> _CensoredResult | None:
    """
    Parse numerics with censored markers / unicode minus / ± / trailing units.

    Works on the *unique* strings only (factorize + vectorised `str` ops) and
    maps back through the integer codes — no per-row Python. Returns None when
    the column is not a mostly-numeric one, so it costs nothing on text.
    """
    codes, uniques = pd.factorize(values)
    uniq = pd.Series(uniques)
    if uniq.empty:
        return None
    if float(uniq.head(2000).str.match(_NUMERICISH_RE).mean()) < 0.5:
        return None
    counts = np.bincount(codes, minlength=len(uniq))
    total = int(counts.sum())
    t = uniq.str.replace("−", "-", regex=False).str.strip()

    ex = t.str.extract(_VALUE_RE)
    num = ex["num"].fillna("")
    has_comma = num.str.contains(",", regex=False)
    if bool(has_comma.any()):
        decimal_mask = has_comma & (
            num.str.contains("[eE]", regex=True)
            | decimal_comma
            | ~num.str.match(_THOUSANDS_RE)
        )
        thousands_mask = has_comma & ~decimal_mask
        num = num.where(~decimal_mask, num.str.replace(",", ".", regex=False))
        num = num.where(~thousands_mask, num.str.replace(",", "", regex=False))
    val = pd.to_numeric(num, errors="coerce").to_numpy(dtype=float, copy=True)

    unit = ex["unit"].str.lower()
    has_unit = unit.notna().to_numpy(dtype=bool)
    units = sorted({str(u) for u in unit.dropna().unique()})
    if len(units) > _MAX_DISTINCT_UNITS:
        # Too many distinct suffixes to be one measurement unit ("3rd", "12ab").
        val[has_unit] = np.nan
        units = []
    op = ex["op"].notna().to_numpy(dtype=bool)
    nd = t.str.match(_ND_RE).fillna(False).to_numpy(dtype=bool) & ~op
    ok = ~np.isnan(val)
    detected = ok & ~op & ~nd
    cens = ok & op
    n_det = int(counts[detected].sum())
    n_marker = int(counts[cens | nd].sum())

    if n_marker == 0:
        if n_det / total < COERCE_MATCH_THRESHOLD:
            return None
    elif n_det / total < _CENSORED_MIN_PARSED or (n_det + n_marker) / total < COERCE_MATCH_THRESHOLD:
        return None

    out_u = np.where(cens, val / 2.0, val)
    positive = val[detected & (val > 0)]
    fill = float(positive.min()) / 2.0 if positive.size else float("nan")
    out_u[nd] = fill
    below_u = cens | nd
    failed_u = np.isnan(out_u)
    return _CensoredResult(
        values=out_u[codes],
        below=below_u[codes],
        n_below=int(counts[below_u].sum()),
        n_failed=int(counts[failed_u].sum()),
        failed_examples=[str(x) for x in uniq[failed_u].head(5).tolist()],
        marker_examples=[str(x) for x in t[below_u].head(5).tolist()],
        units=units,
        n_nd_unfilled=int(counts[nd].sum()) if np.isnan(fill) else 0,
    )


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
    semicolon_file = delimiter == ";"
    flag_columns: list[tuple[str, str, pd.Series]] = []  # (after column, flag name, flag)

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

        # Zero-padded codes (ZIP "02134", phone) are identifiers, not numbers.
        if non_null.str.match(_LEADING_ZERO_RE).any():
            continue

        decimal_comma = _is_decimal_comma(non_null, semicolon_file)
        result = _try_coerce_column(non_null, decimal_comma)

        # Failed-coercion path only: a clean column (all values parsed) costs
        # nothing extra. Currency/percent/thousands hits are already exact.
        if result is None or (result[0] == "numeric" and bool(result[2].isna().any())):
            cen = _try_censored_numeric(non_null, decimal_comma)
            if cen is not None:
                new_num = pd.Series(np.nan, index=series.index, dtype="float64")
                new_num.loc[non_null.index] = cen.values
                out[col] = new_num
                if cen.n_below:
                    flag = pd.Series(False, index=series.index, dtype="bool")
                    flag.loc[non_null.index] = cen.below
                    flag_columns.append((str(col), f"{col}__below_limit", flag))
                    detail = (
                        f"{col}: {cen.n_below} values below detection limit "
                        f"({', '.join(cen.marker_examples[:3])} etc) set to half the limit; "
                        "flag column added"
                    )
                    if cen.n_nd_unfilled:
                        detail += f"; {cen.n_nd_unfilled} bare ND values left missing (no detected value)"
                    rule_name = "censored"
                else:
                    detail = f"{col}: unicode minus / ± / exponent / unit formats normalised"
                    rule_name = "numeric"
                if cen.units:
                    detail += f"; unit suffix stripped ({', '.join(cen.units)})"
                coercions.append(
                    Coercion(
                        column=str(col),
                        from_kind="string",
                        to_kind="numeric",
                        rule=rule_name,
                        n_converted=int((~np.isnan(cen.values)).sum()),
                        n_failed=cen.n_failed,
                        failed_examples=cen.failed_examples,
                        is_sentinel_only=bool(
                            cen.failed_examples
                            and all(str(x).strip().lower() in _COMMON_SENTINELS for x in cen.failed_examples)
                        ),
                        detail=detail,
                    )
                )
                continue
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

    for after_col, flag_name, flag in flag_columns:
        while flag_name in out.columns:
            flag_name += "_"
        out.insert(int(out.columns.get_loc(after_col)) + 1, flag_name, flag)

    # Date + Time fusion: a date column (midnight only) plus a companion
    # time-of-day string column ("18.00.00", "18:00:00", "18:00") becomes one
    # datetime so time-series tools see the sub-daily resolution. The time
    # column is kept as is.
    date_cols = [c for c in out.columns if pd.api.types.is_datetime64_any_dtype(out[c])]
    time_cols = [
        c for c in out.columns
        if not pd.api.types.is_numeric_dtype(out[c])
        and not pd.api.types.is_datetime64_any_dtype(out[c])
        and not pd.api.types.is_bool_dtype(out[c])
    ]
    for d_col in date_cols:
        d_series = out[d_col].dropna()
        if d_series.empty or not bool((d_series == d_series.dt.normalize()).all()):
            continue
        for t_col in time_cols:
            t_non_null = out[t_col].dropna().astype(str).str.strip()
            t_non_null = t_non_null[t_non_null != ""]
            if t_non_null.empty:
                continue
            td = _parse_time_of_day(t_non_null)
            valid_td = td.notna()
            if valid_td.mean() < COERCE_MATCH_THRESHOLD:
                continue
            out[d_col] = out[d_col] + td.reindex(out.index).fillna(pd.Timedelta(0))
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
                    detail=f"{d_col}: combined with time-of-day column {t_col} into one datetime",
                )
            )
            break

    # Numeric placeholders (-200, -999...) — including columns just repaired
    # from strings above, which read_any could not yet see as numeric.
    out, sentinel_records = null_sentinels(out)
    for rec in sentinel_records:
        coercions.append(
            Coercion(
                column=rec["column"],
                from_kind="numeric",
                to_kind="numeric",
                rule="sentinel",
                n_converted=0,
                n_failed=rec["count"],
                failed_examples=[format_value(rec["value"])],
                detail=rec["note"],
            )
        )

    return out, coercions
