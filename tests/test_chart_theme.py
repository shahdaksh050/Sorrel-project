"""Tests for src/core/chart_theme.py (Round 8, item 8.4 — chart unit-format
intelligence: currency/percent/count axis formats and human-readable titles,
so a chart shows "$67" and "Amount ($)" instead of "67.39" and "amount")."""
from __future__ import annotations

from src.core.chart_theme import axis_format, humanize_axis_title, humanize_label, vega_config


class TestAxisFormat:
    def test_currency_gets_dollar_format(self) -> None:
        assert axis_format("currency") == {"format": "$,.0f"}

    def test_percent_gets_percent_format(self) -> None:
        # Percent columns in this codebase are stored as 0-1 fractions
        # (src/core/coercion.py's _parse_percent divides by 100 at parse
        # time), and Vega-Lite's ".0%" format multiplies by 100 itself —
        # these must line up, or every percent chart is silently 100x off.
        assert axis_format("percent") == {"format": ".0%"}

    def test_count_gets_integer_format(self) -> None:
        assert axis_format("count") == {"format": ",d"}

    def test_none_gets_no_override(self) -> None:
        assert axis_format(None) == {}

    def test_unknown_hint_gets_no_override(self) -> None:
        assert axis_format("something_unrecognized") == {}

    def test_returned_dict_is_a_copy_not_shared_mutable_state(self) -> None:
        a = axis_format("currency")
        b = axis_format("currency")
        a["format"] = "mutated"
        assert b == {"format": "$,.0f"}


class TestHumanizeAxisTitle:
    def test_currency_column_gets_dollar_suffix(self) -> None:
        assert humanize_axis_title("monthly_charges", "currency") == "Monthly Charges ($)"

    def test_percent_column_gets_percent_suffix(self) -> None:
        assert humanize_axis_title("churn_rate", "percent") == "Churn Rate (%)"

    def test_count_column_gets_no_suffix(self) -> None:
        assert humanize_axis_title("row_count", "count") == "Row Count"

    def test_no_unit_hint_gets_no_suffix(self) -> None:
        assert humanize_axis_title("tenure_months") == "Tenure Months"

    def test_underscores_replaced_with_spaces(self) -> None:
        assert "_" not in humanize_axis_title("total_charges_usd")

    def test_single_word_column(self) -> None:
        assert humanize_axis_title("amount", "currency") == "Amount ($)"


class TestHumanizeLabel:
    def test_underscores_replaced_and_capitalized(self) -> None:
        assert humanize_label("monthly_charges") == "Monthly Charges"

    def test_acronyms_uppercased(self) -> None:
        assert humanize_label("revenue_usd") == "Revenue USD"
        assert humanize_label("total_kpi") == "Total KPI"
        assert humanize_label("user_id") == "User ID"

    def test_mixed_case_preserved(self) -> None:
        assert humanize_label("iPhone_sales") == "iPhone Sales"
        assert humanize_label("MRR") == "MRR"

    def test_non_string_handled_safely(self) -> None:
        assert humanize_label(123) == "123"  # type: ignore[arg-type]


class TestVegaConfigUnaffected:
    """8.4 must not touch the existing theme tokens (colors) — regression
    guard that the new axis_format work landed alongside, not inside,
    vega_config()."""

    def test_vega_config_still_returns_expected_top_level_keys(self) -> None:
        config = vega_config()
        for key in ("background", "axis", "legend", "range", "mark"):
            assert key in config

    def test_dark_and_light_configs_differ(self) -> None:
        assert vega_config(dark=False) != vega_config(dark=True)
