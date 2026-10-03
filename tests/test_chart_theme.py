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


# ── Sorrel palette: the pen is green, so no other slot may be ─────────────────────────────────


def _channel(c: int) -> float:
    v = c / 255
    return v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4


def _luminance(hex_color: str) -> float:
    r, g, b = (_channel(int(hex_color[i : i + 2], 16)) for i in (1, 3, 5))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _contrast(a: str, b: str) -> float:
    hi, lo = sorted((_luminance(a), _luminance(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def _hue(hex_color: str) -> float:
    import colorsys

    r, g, b = (int(hex_color[i : i + 2], 16) / 255 for i in (1, 3, 5))
    return colorsys.rgb_to_hsv(r, g, b)[0] * 360


class TestCategoryRanges:
    """The six-slot ranges were validated with a script (CIEDE2000, all pairs, normal and
    protan/deutan/tritan vision); these pin the properties that script established so a
    hand-edited hex cannot quietly undo them."""

    def test_slot_one_is_the_pen_family_in_both_modes(self) -> None:
        from src.core import design_tokens
        from src.core.chart_theme import CATEGORY_RANGE_DAY, CATEGORY_RANGE_NIGHT, CHART_PEN_NIGHT

        assert CATEGORY_RANGE_DAY[0] == design_tokens.PEN_DAY
        assert CATEGORY_RANGE_NIGHT[0] == CHART_PEN_NIGHT
        assert abs(_hue(CHART_PEN_NIGHT) - _hue(design_tokens.PEN_NIGHT)) < 12  # same hue, lighter

    def test_six_distinct_slots_and_the_risk_colour_is_never_a_series(self) -> None:
        from src.core import design_tokens
        from src.core.chart_theme import CATEGORY_RANGE_DAY, CATEGORY_RANGE_NIGHT

        for mode, ranges in (("day", CATEGORY_RANGE_DAY), ("night", CATEGORY_RANGE_NIGHT)):
            assert len(ranges) == 6 and len(set(ranges)) == 6
            assert design_tokens.palette(mode)["risk"] not in ranges  # type: ignore[arg-type]

    def test_no_second_green_beside_the_pen(self) -> None:
        from src.core.chart_theme import CATEGORY_RANGE_DAY, CATEGORY_RANGE_NIGHT

        for ranges in (CATEGORY_RANGE_DAY, CATEGORY_RANGE_NIGHT):
            for hex_color in ranges[1:]:
                assert not 90 <= _hue(hex_color) <= 170, f"{hex_color} reads as green, like slot 1"

    def test_every_mark_reaches_three_to_one_on_the_chart_surface(self) -> None:
        from src.core import design_tokens
        from src.core.chart_theme import CATEGORY_RANGE_DAY, CATEGORY_RANGE_NIGHT

        for mode, ranges in (("day", CATEGORY_RANGE_DAY), ("night", CATEGORY_RANGE_NIGHT)):
            surface = design_tokens.palette(mode)["sheet"]  # type: ignore[arg-type]
            for hex_color in ranges:
                assert _contrast(hex_color, surface) >= 3.0, f"{mode} {hex_color}"

    def test_default_mark_colours_also_reach_three_to_one(self) -> None:
        from src.core import design_tokens

        for dark, mode in ((False, "day"), (True, "night")):
            cfg = vega_config(dark=dark)
            surface = design_tokens.palette(mode)["sheet"]  # type: ignore[arg-type]
            assert cfg["background"] == surface
            for key in ("mark", "bar", "area", "circle", "point"):
                assert _contrast(cfg[key]["color"], surface) >= 3.0, f"{mode} {key}"

    def test_chart_text_is_set_in_geist_not_the_old_faces(self) -> None:
        cfg = vega_config()
        for font in (cfg["font"], cfg["axis"]["labelFont"], cfg["axis"]["titleFont"], cfg["legend"]["labelFont"]):
            assert font.startswith("Geist,")
            for old in ("Public Sans", "Bricolage", "Baloo", "Mukta", "Plex"):
                assert old not in font
