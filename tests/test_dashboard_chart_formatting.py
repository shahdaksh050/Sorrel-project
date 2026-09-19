"""
Round 8, item 8.4 — end-to-end confirmation that chart_theme's unit-format
intelligence actually reaches real chart specs produced by build_dashboard,
not just that axis_format()/humanize_axis_title() work in isolation
(see tests/test_chart_theme.py for those). Uses columns whose names trigger
profiler.py's name-based unit_hint inference (_infer_unit_hint) so the
currency/percent/count paths are genuinely exercised, not assumed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.core.dashboard import build_dashboard
from src.core.profiler import profile_dataframe

RNG = np.random.default_rng(11)


@pytest.fixture
def measures_df() -> pd.DataFrame:
    n = 300
    return pd.DataFrame({
        "customer_id": range(1, n + 1),
        "amount": RNG.normal(200, 60, n).clip(min=5),          # -> unit_hint "currency"
        "churn_rate": RNG.uniform(0.05, 0.35, n),               # -> unit_hint "percent"
        "plan": RNG.choice(["basic", "pro", "enterprise"], n, p=[0.6, 0.3, 0.1]),
    })


def _findings(*columns: str) -> list[dict]:
    """Strict curation charts a distribution only for columns a finding names."""
    return [{"kind": "distribution", "measure": c} for c in columns]


def _find_chart(charts: list, chart_id_prefix: str):
    for c in charts:
        if c.chart_id.startswith(chart_id_prefix):
            return c
    return None


class TestCurrencyFormatting:
    def test_histogram_axis_carries_dollar_format(self, measures_df: pd.DataFrame) -> None:
        profile = profile_dataframe(measures_df)
        charts = build_dashboard(measures_df, profile, findings=_findings("amount"))
        hist = _find_chart(charts, "hist_amount")
        assert hist is not None, "expected a histogram for the currency-hinted 'amount' column"
        x_axis = hist.spec["encoding"]["x"].get("axis", {})
        assert x_axis.get("format") == "$,.0f"

    def test_histogram_title_is_humanized_with_dollar_suffix(self, measures_df: pd.DataFrame) -> None:
        profile = profile_dataframe(measures_df)
        charts = build_dashboard(measures_df, profile, findings=_findings("amount"))
        hist = _find_chart(charts, "hist_amount")
        assert hist is not None
        assert hist.spec["encoding"]["x"]["title"] == "Amount ($)"

    def test_histogram_caption_is_plain_language_and_dollar_formatted(
        self, measures_df: pd.DataFrame
    ) -> None:
        profile = profile_dataframe(measures_df)
        charts = build_dashboard(measures_df, profile, findings=_findings("amount"))
        hist = _find_chart(charts, "hist_amount")
        assert hist is not None
        assert hist.caption is not None
        assert "$" in hist.caption
        assert "amount" in hist.caption.lower()


class TestPercentFormatting:
    def test_histogram_axis_carries_percent_format(self, measures_df: pd.DataFrame) -> None:
        profile = profile_dataframe(measures_df)
        charts = build_dashboard(measures_df, profile, findings=_findings("churn_rate"))
        hist = _find_chart(charts, "hist_churn_rate")
        assert hist is not None, "expected a histogram for the percent-hinted 'churn_rate' column"
        x_axis = hist.spec["encoding"]["x"].get("axis", {})
        assert x_axis.get("format") == ".0%"

    def test_histogram_caption_uses_percent_not_raw_fraction(self, measures_df: pd.DataFrame) -> None:
        profile = profile_dataframe(measures_df)
        charts = build_dashboard(measures_df, profile, findings=_findings("churn_rate"))
        hist = _find_chart(charts, "hist_churn_rate")
        assert hist is not None
        assert "%" in hist.caption
        assert "0.1" not in hist.caption  # never a bare fraction leaking through


class TestNoUnitHintUnaffected:
    def test_plain_numeric_column_gets_no_format_override(self, measures_df: pd.DataFrame) -> None:
        df = measures_df.copy()
        df["tenure_months"] = RNG.integers(1, 72, len(df))
        profile = profile_dataframe(df)
        charts = build_dashboard(df, profile, findings=_findings("tenure_months"))
        hist = _find_chart(charts, "hist_tenure_months")
        assert hist is not None
        x_axis = hist.spec["encoding"]["x"].get("axis", {})
        assert "format" not in x_axis
        assert hist.spec["encoding"]["x"]["title"] == "Tenure Months"


class TestCategoricalBarsSortedByValue:
    def test_class_balance_chart_sorts_bars_descending(self, measures_df: pd.DataFrame) -> None:
        df = measures_df.copy()
        df["churn"] = RNG.choice([0, 1], len(df), p=[0.8, 0.2])
        profile = profile_dataframe(df, target_column="churn")
        charts = build_dashboard(df, profile, target_column="churn", task_type="classification")
        class_balance = _find_chart(charts, "class_balance")
        assert class_balance is not None
        assert class_balance.spec["encoding"]["x"].get("sort") == "-y"

    def test_class_balance_count_axis_uses_integer_format(self, measures_df: pd.DataFrame) -> None:
        df = measures_df.copy()
        df["churn"] = RNG.choice([0, 1], len(df), p=[0.8, 0.2])
        profile = profile_dataframe(df, target_column="churn")
        charts = build_dashboard(df, profile, target_column="churn", task_type="classification")
        class_balance = _find_chart(charts, "class_balance")
        assert class_balance is not None
        y_axis = class_balance.spec["encoding"]["y"].get("axis", {})
        assert y_axis.get("format") == ",d"


class TestRegressionAgainstPriorRounds:
    """8.4 must not disturb panel selection/suppression/theme tokens from
    earlier rounds — a coarse smoke check that the dashboard still produces
    a normal set of panels, not a rewrite of test_dashboard.py's own
    coverage of that behavior."""

    def test_dashboard_still_produces_multiple_charts(self, measures_df: pd.DataFrame) -> None:
        profile = profile_dataframe(measures_df)
        charts = build_dashboard(measures_df, profile, findings=_findings("amount", "plan"))
        assert len(charts) >= 2

    def test_every_chart_is_json_serializable(self, measures_df: pd.DataFrame) -> None:
        import json

        profile = profile_dataframe(measures_df)
        charts = build_dashboard(measures_df, profile)
        for chart in charts:
            json.dumps(chart.to_dict(), default=str)  # must not raise
