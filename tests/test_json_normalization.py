"""Tests for the strict JSON boundary utility."""
from __future__ import annotations

import json
from datetime import UTC, date, datetime, time

import numpy as np
import pandas as pd
import pytest

from src.core.json_normalization import normalize_json, strict_json_dumps


def test_normalize_json_handles_nested_data_scalars_and_missing_values() -> None:
    value = {
        "numpy": [np.int64(7), np.float32(2.5), np.float64(float("nan"))],
        "missing": {"na": pd.NA, "nat": pd.NaT, "infinity": float("inf")},
        "nested": ({"negative": float("-inf")},),
    }

    assert normalize_json(value) == {
        "numpy": [7, 2.5, None],
        "missing": {"na": None, "nat": None, "infinity": None},
        "nested": [{"negative": None}],
    }


def test_normalize_json_uses_iso_datetime_representations() -> None:
    timestamp = datetime(2026, 10, 16, 9, 30, tzinfo=UTC)
    assert normalize_json(
        {
            "datetime": timestamp,
            "date": date(2026, 10, 16),
            "time": time(9, 30, 1),
            "np_datetime": np.datetime64("2026-10-16T09:30:00"),
            "timedelta": pd.Timedelta("2 days 03:04:05"),
        }
    ) == {
        "datetime": "2026-10-16T09:30:00+00:00",
        "date": "2026-10-16",
        "time": "09:30:01",
        "np_datetime": "2026-10-16T09:30:00",
        "timedelta": "P2DT3H4M5S",
    }


def test_strict_json_dumps_is_parseable_and_never_emits_non_finite_literals() -> None:
    text = strict_json_dumps({1: np.float64(float("nan")), "good": [True, None]})
    assert text == '{"1": null, "good": [true, null]}'
    assert json.loads(text) == {"1": None, "good": [True, None]}
    assert "NaN" not in text and "Infinity" not in text


def test_normalize_json_rejects_opaque_objects_and_cycles() -> None:
    with pytest.raises(TypeError, match="Unsupported value"):
        normalize_json(object())

    cyclic: list[object] = []
    cyclic.append(cyclic)
    with pytest.raises(ValueError, match="cyclic"):
        normalize_json(cyclic)
