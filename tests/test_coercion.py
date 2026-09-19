"""Tests for src.core.coercion — numerics trapped in strings (U0.7)."""
from __future__ import annotations

import pandas as pd
import pytest

from src.core.coercion import coerce_types


class TestCurrency:
    def test_dollar_with_thousands_separator(self) -> None:
        df = pd.DataFrame({"amount": ["$1,234.56"] * 20})
        out, coercions = coerce_types(df)
        assert out["amount"].iloc[0] == 1234.56
        assert pd.api.types.is_numeric_dtype(out["amount"])
        assert coercions[0].rule == "currency"
        assert coercions[0].column == "amount"


class TestPercent:
    def test_percent_stored_as_fraction(self) -> None:
        df = pd.DataFrame({"rate": ["45.3%"] * 20})
        out, coercions = coerce_types(df)
        assert out["rate"].iloc[0] == pytest.approx(0.453)
        assert coercions[0].rule == "percent"


class TestBoolean:
    def test_yes_no_becomes_bool(self) -> None:
        df = pd.DataFrame({"active": (["Y", "N"] * 10)})
        out, coercions = coerce_types(df)
        assert out["active"].iloc[0] == True  # noqa: E712
        assert out["active"].iloc[1] == False  # noqa: E712
        assert coercions[0].rule == "yes_no"
        assert coercions[0].to_kind == "boolean"


class TestIdentifierGuard:
    def test_zipcode_stays_string_not_coerced(self) -> None:
        """A numeric-looking but zero-padded zipcode must never be coerced —
        it would silently strip the leading zero."""
        df = pd.DataFrame({"zipcode": ["04521", "04522", "04523"] * 10})
        out, coercions = coerce_types(df)
        assert out["zipcode"].tolist() == df["zipcode"].tolist()
        assert coercions == []


class TestMixedTypeThreshold:
    def test_below_95_percent_not_coerced(self) -> None:
        """A 50/50 mixed column must not be silently coerced."""
        values = (["1", "2", "not_a_number", "also_bad"] * 10)
        df = pd.DataFrame({"value": values})
        out, coercions = coerce_types(df)
        assert not pd.api.types.is_numeric_dtype(out["value"])
        assert coercions == []

    def test_above_95_percent_coerced_with_failures_recorded(self) -> None:
        values = ["1.5"] * 95 + ["garbage"] * 5
        df = pd.DataFrame({"value": values})
        out, coercions = coerce_types(df)
        assert pd.api.types.is_numeric_dtype(out["value"])
        assert coercions[0].rule == "numeric"
        assert coercions[0].n_converted == 95
        assert coercions[0].n_failed == 5
        assert "garbage" in coercions[0].failed_examples


class TestEveryCoercionReported:
    def test_multiple_columns_all_appear_in_returned_list(self) -> None:
        df = pd.DataFrame({
            "amount": ["$10.00"] * 20,
            "rate": ["5.0%"] * 20,
            "active": ["yes", "no"] * 10,
            "plain_text": ["hello world"] * 20,
        })
        out, coercions = coerce_types(df)
        coerced_cols = {c.column for c in coercions}
        assert coerced_cols == {"amount", "rate", "active"}
        assert not pd.api.types.is_numeric_dtype(out["plain_text"])


class TestEuropeanDecimal:
    def test_comma_as_decimal_only_when_semicolon_delimited(self) -> None:
        df = pd.DataFrame({"value": ["4,5", "10,25", "3,0"] * 10})
        out, _coercions = coerce_types(df, delimiter=";")
        assert out["value"].iloc[0] == 4.5
        assert out["value"].iloc[1] == 10.25

    def test_comma_as_thousands_when_not_semicolon_delimited(self) -> None:
        df = pd.DataFrame({"value": ["4,500", "10,250"] * 10})
        out, coercions = coerce_types(df, delimiter=",")
        assert out["value"].iloc[0] == 4500
        assert coercions[0].rule == "thousands"


class TestHighFrequencyDateConvention:
    def test_minute_frequency_dayfirst_detected(self) -> None:
        """High-frequency minute data (1440 rows/day) has days <= 2 in first 2000 rows.
        sampling from unique dates must correctly discover day 13+ and infer date_dayfirst."""
        dates = ["01/01/2020"] * 1500 + ["02/01/2020"] * 1500 + ["15/01/2020"] * 100
        df = pd.DataFrame({"record_date": dates})
        out, coercions = coerce_types(df)
        assert pd.api.types.is_datetime64_any_dtype(out["record_date"])
        assert coercions[0].rule == "date_dayfirst"
        # January 15th 2020
        assert out["record_date"].iloc[-1] == pd.Timestamp("2020-01-15")


class TestSentinelDetection:
    def test_question_mark_sentinel_flagged_as_sentinel_only(self) -> None:
        df = pd.DataFrame({"metric": ["12.5"] * 96 + ["?"] * 4})
        out, coercions = coerce_types(df)
        assert pd.api.types.is_numeric_dtype(out["metric"])
        assert coercions[0].is_sentinel_only is True
        assert coercions[0].n_failed == 4


class TestDateTimeFusion:
    def test_companion_date_and_time_fused(self) -> None:
        df = pd.DataFrame({
            "Date": ["16/12/2006", "17/12/2006"] * 20,
            "Time": ["17:24:00", "08:30:15"] * 20,
        })
        out, coercions = coerce_types(df)
        assert pd.api.types.is_datetime64_any_dtype(out["Date"])
        # Should be fused into 2006-12-16 17:24:00
        assert out["Date"].iloc[0] == pd.Timestamp("2006-12-16 17:24:00")
        assert out["Date"].iloc[1] == pd.Timestamp("2006-12-17 08:30:15")
        rules = [c.rule for c in coercions]
        assert any("date_time_fusion" in r for r in rules)


    def test_dotted_time_and_hh_mm_fused(self) -> None:
        for times in (["18.00.00", "19.30.10"], ["18:00", "19:30"]):
            df = pd.DataFrame({
                "Date": ["16/12/2006", "17/12/2006"] * 20,
                "Time": times * 20,
            })
            out, _ = coerce_types(df)
            assert out["Date"].iloc[0].hour == 18
            assert out["Date"].iloc[1].hour == 19
            assert list(out["Time"].iloc[:2]) == times  # time column kept as is

    def test_date_with_time_components_not_refused(self) -> None:
        df = pd.DataFrame({
            "Date": ["2006-12-16 10:00:00", "2006-12-17 11:00:00"] * 20,
            "Time": ["17:24:00", "08:30:15"] * 20,
        })
        out, coercions = coerce_types(df)
        assert out["Date"].iloc[0] == pd.Timestamp("2006-12-16 10:00:00")
        assert not any("date_time_fusion" in c.rule for c in coercions)

    def test_non_time_column_not_fused(self) -> None:
        df = pd.DataFrame({
            "Date": ["16/12/2006", "17/12/2006"] * 20,
            "note": ["hello", "world"] * 20,
        })
        out, _ = coerce_types(df)
        assert out["Date"].iloc[0] == pd.Timestamp("2006-12-16")


class TestDecimalComma:
    @pytest.mark.parametrize("delimiter", [None, ",", ";"])
    def test_leading_zero_decimal_comma(self, delimiter: str | None) -> None:
        vals = ["0,7578", "1,0656", "0,7255", "1,2"] * 10
        out, _ = coerce_types(pd.DataFrame({"AH": vals}), delimiter=delimiter)
        assert out["AH"].iloc[0] == pytest.approx(0.7578)
        assert out["AH"].max() < 2.5

    def test_one_or_two_digit_decimals(self) -> None:
        out, _ = coerce_types(pd.DataFrame({"T": ["13,6", "-2,5", "10"] * 12}))
        assert out["T"].iloc[0] == pytest.approx(13.6)
        assert out["T"].iloc[1] == pytest.approx(-2.5)

    def test_four_digit_decimals_without_leading_zero(self) -> None:
        out, _ = coerce_types(pd.DataFrame({"x": ["1,2345", "2,5", "3,14159"] * 12}))
        assert out["x"].iloc[0] == pytest.approx(1.2345)

    def test_all_three_digit_groups_stay_thousands(self) -> None:
        out, _ = coerce_types(pd.DataFrame({"n": ["1,234", "12,345", "1,234,567"] * 12}))
        assert out["n"].iloc[0] == 1234
        assert out["n"].iloc[2] == 1234567

    def test_thousands_with_decimal_kept(self) -> None:
        out, _ = coerce_types(pd.DataFrame({"n": ["1,234.50", "2,000.25", "12,345.10"] * 12}))
        assert out["n"].iloc[0] == pytest.approx(1234.5)

    def test_leading_zero_three_digit_is_decimal(self) -> None:
        out, _ = coerce_types(pd.DataFrame({"n": ["0,750", "0,125", "-0,500"] * 12}))
        assert out["n"].iloc[0] == pytest.approx(0.75)
        assert out["n"].iloc[2] == pytest.approx(-0.5)


class TestSentinelAtoms:
    @staticmethod
    def _sensor_frame() -> pd.DataFrame:
        import numpy as np

        rng = np.random.default_rng(0)
        n = 600
        cols = {}
        for name, lo, hi in (("s1", 400, 2400), ("s2", 300, 2600), ("s3", 200, 2700)):
            x = rng.integers(lo, hi, n).astype(float)
            x[rng.choice(n, 30, replace=False)] = -200.0
            cols[name] = x
        # Mostly-missing column: -200 is ~85% of rows.
        y = rng.integers(10, 1100, n).astype(float)
        y[rng.choice(n, 510, replace=False)] = -200.0
        cols["mostly"] = y
        return pd.DataFrame(cols)

    def test_wide_spread_sensor_placeholder_nulled(self) -> None:
        out, coercions = coerce_types(self._sensor_frame())
        for c in out.columns:
            assert not (out[c] == -200).any(), c
        assert out["mostly"].isna().mean() > 0.8
        assert {c.column for c in coercions if c.rule == "sentinel"} == set(out.columns)

    def test_zero_inflated_frame_untouched(self) -> None:
        import numpy as np

        rng = np.random.default_rng(1)
        n = 600
        df = pd.DataFrame({
            f"income{i}": np.where(
                rng.random(n) < 0.4, 0.0, rng.integers(5000, 90000, n).astype(float)
            )
            for i in range(4)
        })
        # A heavy non-zero atom sitting right next to the bulk (tiny gap vs IQR).
        df["capped"] = np.where(rng.random(n) < 0.2, 500.0, rng.integers(400, 500, n).astype(float))
        out, coercions = coerce_types(df)
        assert out.equals(df)
        assert not [c for c in coercions if c.rule == "sentinel"]
