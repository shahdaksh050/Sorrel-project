"""Neutral column-name pattern helpers shared by reasoning and tools."""
from __future__ import annotations

from collections.abc import Sequence


def find_benchmark_pairs(columns: Sequence[str]) -> list[tuple[str, str, str]]:
    """Return observed/expected column pairs and their normalized stem."""
    lookup = {column.lower(): column for column in columns}
    pairs: list[tuple[str, str, str]] = []
    seen: set[tuple[str, str]] = set()
    for column in columns:
        lowered = column.lower()
        for observed_prefix in ("actual_", "observed_"):
            if lowered.startswith(observed_prefix):
                stem = lowered[len(observed_prefix):]
                for prefix in ("target_", "expected_", "budget_", "predicted_"):
                    expected = lookup.get(prefix + stem)
                    if expected and (column, expected) not in seen and column != expected:
                        seen.add((column, expected))
                        pairs.append((column, expected, stem))
        for observed_suffix in ("_actual", "_observed", "_obs"):
            if lowered.endswith(observed_suffix):
                stem = lowered[:-len(observed_suffix)]
                for suffix in ("_target", "_expected", "_budget", "_predicted", "_exp"):
                    expected = lookup.get(stem + suffix)
                    if expected and (column, expected) not in seen and column != expected:
                        seen.add((column, expected))
                        pairs.append((column, expected, stem))
    for observed_name, expected_name, stem in (
        ("goals", "xgoals", "goals"), ("goals", "xg", "goals"),
        ("home_goals", "home_xg", "home_goals"),
        ("home_goals", "home_xgoals", "home_goals"),
        ("away_goals", "away_xg", "away_goals"),
        ("away_goals", "away_xgoals", "away_goals"),
        ("duration", "eta", "duration"), ("revenue", "quota", "revenue"),
        ("sales", "budget", "sales"), ("cost", "budget", "cost"),
        ("actual", "target", "target"), ("actual", "budget", "budget"),
        ("observed", "expected", "expected"),
    ):
        observed = lookup.get(observed_name)
        expected = lookup.get(expected_name)
        if observed and expected and observed != expected and (observed, expected) not in seen:
            seen.add((observed, expected))
            pairs.append((observed, expected, stem))
    return pairs
