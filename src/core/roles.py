"""Validate LLM-proposed column roles against the data.

Tools pick columns by name (`src.core.vocab`); an unusual name (`dias_hasta_baja`
is covered, `estado_x` is not) falls through. The first reply may propose roles
for such columns. A proposal is only ever *believed* after a cheap deterministic
check on the data itself; accepted ones land in `DatasetProfile.role_overrides`
and behave exactly like a name match (`vocab.column_role`).
"""
from __future__ import annotations

import warnings
from collections.abc import Callable

import pandas as pd

from src.core.vocab import ROLE_TOKENS

MAX_PROPOSALS = 12
_SAMPLE_ROWS = 20_000
_POSITIVE_SHARE = 0.9
_TIME_OF_DAY = r"^(?:[01]?\d|2[0-3]):[0-5]\d(?::[0-5]\d)?$"


def _numeric(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)


def _mostly_positive(s: pd.Series) -> str | None:
    if not _numeric(s):
        return "not numeric"
    if float((s > 0).mean()) < _POSITIVE_SHARE:
        return "fewer than 90% of values are positive"
    return None


def _duration(s: pd.Series) -> str | None:
    return _mostly_positive(s) or (None if s.nunique() >= 10 else "fewer than 10 distinct values")


def _event(s: pd.Series) -> str | None:
    return None if pd.api.types.is_bool_dtype(s) or s.nunique() == 2 else "not a two-valued column"


def _order(s: pd.Series) -> str | None:
    n, k = len(s), s.nunique()
    return None if n / k >= 1.5 and k / n < _POSITIVE_SHARE else "values are not repeated across rows"


def _item(s: pd.Series) -> str | None:
    return None if 2 <= s.nunique() <= len(s) / 2 else "needs 2 to n/2 distinct values"


def _entity_group(s: pd.Series) -> str | None:
    n, k = len(s), s.nunique()
    if not 5 <= k <= n / 2:
        return "needs 5 to n/2 groups"
    return None if n / k >= 3 else "groups average fewer than 3 rows"


def _protected(s: pd.Series) -> str | None:
    if _numeric(s):
        return "not categorical"
    return None if 2 <= s.nunique() <= 12 else "needs 2 to 12 levels"


def _binary(s: pd.Series) -> str | None:
    return None if s.nunique() == 2 else "needs exactly 2 distinct values"


def _dose(s: pd.Series) -> str | None:
    if not _numeric(s):
        return "not numeric"
    return None if s.nunique() >= 8 else "fewer than 8 distinct values"


def _id(s: pd.Series) -> str | None:
    return None if s.nunique() / len(s) >= 0.95 else "fewer than 95% of values are unique"


def _date(s: pd.Series) -> str | None:
    if pd.api.types.is_datetime64_any_dtype(s):
        return None
    if _numeric(s) or pd.api.types.is_bool_dtype(s):
        return "not a date column"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        parsed = pd.to_datetime(s.astype(str), errors="coerce", format="mixed")
    return None if float(parsed.notna().mean()) >= _POSITIVE_SHARE else "fewer than 90% parse as dates"


def _time_of_day(s: pd.Series) -> str | None:
    if _numeric(s):
        whole = (s % 1 == 0) & s.between(0, 23)
        return None if bool(whole.all()) else "numbers are not whole hours 0-23"
    ok = s.astype(str).str.strip().str.match(_TIME_OF_DAY)
    return None if float(ok.mean()) >= _POSITIVE_SHARE else "not HH:MM(:SS) times"


_CHECKS: dict[str, Callable[[pd.Series], str | None]] = {
    "duration": _duration,
    "event": _event,
    "price": _mostly_positive,
    "quantity": _mostly_positive,
    "pay": _mostly_positive,
    "order": _order,
    "item": _item,
    "entity_group": _entity_group,
    "protected_attribute": _protected,
    "binary_outcome": _binary,
    "adverse_outcome": _binary,
    "dose_x_strong": _dose,
    "dose_x_weak": _dose,
    "id": _id,
    "date": _date,
    "time_of_day": _time_of_day,
}
assert set(_CHECKS) <= set(ROLE_TOKENS)

#: Role names the model may propose (the prompt lists exactly these).
ALLOWED_ROLES: tuple[str, ...] = tuple(_CHECKS)


def validate_roles(
    df: pd.DataFrame, proposals: dict[str, str]
) -> tuple[dict[str, str], dict[str, str]]:
    """(accepted {column: role}, rejected {column: reason}). Exact, case-sensitive
    column names only; one role per column; at most `MAX_PROPOSALS` proposals."""
    accepted: dict[str, str] = {}
    rejected: dict[str, str] = {}
    if not isinstance(proposals, dict):
        return accepted, rejected
    sample = df if len(df) <= _SAMPLE_ROWS else df.sample(n=_SAMPLE_ROWS, random_state=0)
    for i, (column, role) in enumerate(proposals.items()):
        column = str(column)
        if i >= MAX_PROPOSALS:
            rejected[column] = f"more than {MAX_PROPOSALS} proposals"
        elif column not in df.columns:
            rejected[column] = "no such column"
        elif not isinstance(role, str) or role not in _CHECKS:
            rejected[column] = f"unknown role {role!r}"
        else:
            values = sample[column].dropna()
            reason = "no values" if values.empty else _CHECKS[role](values)
            if reason is None:
                accepted[column] = role
            else:
                rejected[column] = f"{role}: {reason}"
    return accepted, rejected
