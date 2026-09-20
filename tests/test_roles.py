"""Tests for src/core/roles.py: LLM-proposed column roles are validated against the data."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.core.profiler import profile_dataframe
from src.core.roles import ALLOWED_ROLES, MAX_PROPOSALS, validate_roles
from src.core.vocab import column_role, roles_for
from src.tools.survival import SurvivalAnalysisTool

N = 400
_RNG = np.random.default_rng(7)


def _fit_and_misfit() -> dict[str, tuple[pd.Series, pd.Series]]:
    """role -> (a column that fits, a column that does not)."""
    positive = pd.Series(_RNG.uniform(1, 100, N).round(2))
    categorical = pd.Series(_RNG.choice(list("abcdef"), N))
    five_valued = pd.Series(_RNG.integers(0, 5, N))
    unique_ids = pd.Series([f"r{i}" for i in range(N)])
    times = pd.Series([f"{h:02d}:{m:02d}" for h, m in zip(_RNG.integers(0, 24, N), _RNG.integers(0, 60, N), strict=True)])
    return {
        "duration": (positive, categorical),
        "event": (pd.Series(_RNG.integers(0, 2, N)), five_valued),
        "price": (positive, categorical),
        "quantity": (pd.Series(_RNG.integers(1, 30, N)), pd.Series(-positive)),
        "pay": (positive, categorical),
        "order": (pd.Series(np.repeat(np.arange(N // 4), 4)), unique_ids),
        "item": (categorical, pd.Series(np.zeros(N, dtype=int))),
        "entity_group": (pd.Series(np.repeat(np.arange(N // 8), 8)), unique_ids),
        "protected_attribute": (pd.Series(_RNG.choice(["f", "m", "x"], N)), positive),
        "binary_outcome": (pd.Series(_RNG.choice(["yes", "no"], N)), five_valued),
        "adverse_outcome": (pd.Series(_RNG.integers(0, 2, N)), categorical),
        "dose_x_strong": (positive, categorical),
        "dose_x_weak": (positive, pd.Series(_RNG.integers(0, 3, N))),
        "id": (unique_ids, categorical),
        "date": (pd.Series(pd.date_range("2024-01-01", periods=N).astype(str)), positive),
        "time_of_day": (times, categorical),
    }


CASES = _fit_and_misfit()


def test_every_allowed_role_is_covered() -> None:
    assert set(CASES) == set(ALLOWED_ROLES)


@pytest.mark.parametrize("role", sorted(CASES))
def test_role_accepted_on_fitting_data_rejected_on_misfit(role: str) -> None:
    fit, misfit = CASES[role]
    accepted, rejected = validate_roles(pd.DataFrame({"good": fit, "bad": misfit}), {"good": role, "bad": role})
    assert accepted == {"good": role}
    assert list(rejected) == ["bad"] and rejected["bad"].startswith(f"{role}:")


def test_unknown_column_and_unknown_role_rejected() -> None:
    df = pd.DataFrame({"Price": _RNG.uniform(1, 9, N), "x": _RNG.uniform(1, 9, N)})
    accepted, rejected = validate_roles(df, {"price": "price", "x": "wizard", "Price": "price"})
    assert accepted == {"Price": "price"}
    assert rejected["price"] == "no such column"
    assert "unknown role" in rejected["x"]


def test_more_than_twelve_proposals_capped() -> None:
    df = pd.DataFrame({f"c{i}": _RNG.uniform(1, 100, N) for i in range(15)})
    accepted, rejected = validate_roles(df, {f"c{i}": "price" for i in range(15)})
    assert len(accepted) == MAX_PROPOSALS
    assert sorted(rejected) == ["c12", "c13", "c14"]


def test_null_only_column_rejected() -> None:
    accepted, rejected = validate_roles(pd.DataFrame({"a": [np.nan] * 50}), {"a": "price"})
    assert not accepted and rejected["a"].endswith("no values")


def test_vocab_helpers_honour_overrides() -> None:
    profile = profile_dataframe(pd.DataFrame({"blob": np.arange(30), "price": np.arange(30) + 1.0}))
    assert not column_role(profile, "blob", "price")
    profile.role_overrides = {"blob": "price"}
    assert column_role(profile, "blob", "price") and column_role(profile, "price", "price")
    assert roles_for(profile, "price") == ["blob", "price"]
    assert profile.to_dict()["role_overrides"] == {"blob": "price"}


def test_survival_applies_only_with_validated_overrides() -> None:
    rng = np.random.default_rng(3)
    n = 300
    df = pd.DataFrame({
        "vida_util_k": rng.exponential(12, n) + 0.5,
        "marca_z": rng.integers(0, 2, n),
        "segmento": rng.choice(["a", "b", "c"], n),
    })
    tool = SurvivalAnalysisTool()
    profile = profile_dataframe(df)
    assert tool.applies_to(profile, None) == 0.0
    accepted, _ = validate_roles(df, {"vida_util_k": "duration", "marca_z": "event"})
    assert accepted == {"vida_util_k": "duration", "marca_z": "event"}
    profile.role_overrides = accepted
    assert tool.applies_to(profile, None) > 0
    assert tool.default_params(profile, None)["duration_column"] == "vida_util_k"
    assert tool.default_params(profile, None)["event_column"] == "marca_z"
