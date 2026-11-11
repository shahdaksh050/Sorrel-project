"""Strict, loss-aware conversion of common data values into JSON values.

This is deliberately a small boundary utility.  It lets persistence and
presentation layers opt into RFC-compliant JSON without each writer inventing
slightly different handling for pandas missing values or IEEE non-finite
numbers.  Callers must opt in; this module does not change existing writers.
"""
from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from datetime import date, datetime, time
from typing import Any, TypeAlias, cast

import numpy as np
import pandas as pd

JSONScalar: TypeAlias = str | int | float | bool | None
JSONValue: TypeAlias = JSONScalar | list["JSONValue"] | dict[str, "JSONValue"]


def normalize_json(value: Any) -> JSONValue:
    """Return *value* as recursively JSON-safe data.

    Missing values and non-finite floats become ``None``. Datetime-like
    values use ISO-8601 strings. Mapping keys are stringified because JSON
    objects only permit string keys. Unsupported objects fail loudly so a
    caller cannot accidentally persist their opaque representation.
    """
    return _normalize_json(value, ancestors=set())


def strict_json_dumps(
    value: Any,
    *,
    indent: int | None = None,
    sort_keys: bool = False,
) -> str:
    """Serialize normalized data with JSON's non-finite-number extension off."""
    return json.dumps(
        normalize_json(value),
        allow_nan=False,
        indent=indent,
        sort_keys=sort_keys,
    )


def _normalize_json(value: Any, ancestors: set[int]) -> JSONValue:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, np.generic):
        return _normalize_json(value.item(), ancestors)
    if isinstance(value, np.datetime64):
        if np.isnat(value):
            return None
        return cast(str, pd.Timestamp(value).isoformat())
    if isinstance(value, np.timedelta64):
        if np.isnat(value):
            return None
        return cast(str, pd.Timedelta(value).isoformat())
    if isinstance(value, pd.Timestamp):
        return None if pd.isna(value) else cast(str, value.isoformat())
    if isinstance(value, pd.Timedelta):
        return None if pd.isna(value) else cast(str, value.isoformat())
    if isinstance(value, datetime | date | time):
        return value.isoformat()
    if isinstance(value, Mapping):
        return _normalize_mapping(value, ancestors)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return _normalize_sequence(value, ancestors)
    raise TypeError(f"Unsupported value for strict JSON normalization: {type(value).__name__}")


def _normalize_mapping(value: Mapping[Any, Any], ancestors: set[int]) -> dict[str, JSONValue]:
    _guard_cycle(value, ancestors)
    try:
        return {str(key): _normalize_json(item, ancestors) for key, item in value.items()}
    finally:
        ancestors.remove(id(value))


def _normalize_sequence(value: Sequence[Any], ancestors: set[int]) -> list[JSONValue]:
    _guard_cycle(value, ancestors)
    try:
        return [_normalize_json(item, ancestors) for item in value]
    finally:
        ancestors.remove(id(value))


def _guard_cycle(value: object, ancestors: set[int]) -> None:
    identity = id(value)
    if identity in ancestors:
        raise ValueError("Cannot normalize a cyclic value as JSON")
    ancestors.add(identity)
