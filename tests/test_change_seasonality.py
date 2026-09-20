"""change_analysis: same-period-last-year check and short-history caveat."""
from __future__ import annotations

import pandas as pd

from src.tools.change_analysis import ChangeAnalysisTool


def _run(tmp_path, dates, values):
    path = tmp_path / "d.csv"
    pd.DataFrame({"Date": dates, "AH": values}).to_csv(path, index=False)
    tool = ChangeAnalysisTool()
    out = tool.execute(str(path), date_column="Date", measure_column="AH")
    return out, tool.findings(out, None, None)


def _monthly(start, end, value_for):
    dates = pd.date_range(start, end, freq="D")
    return dates, [value_for(d) for d in dates]


def test_planted_seasonal_dip_is_flagged_and_demoted(tmp_path):
    # Flat 10 except every February = 5; latest month (March) rebounds +100%.
    dates, vals = _monthly("2023-01-01", "2024-03-31", lambda d: 5.0 if d.month == 2 else 10.0)
    out, findings = _run(tmp_path, dates, vals)
    assert out["latest_period"] == "2024-03"
    assert out["yoy_period"] == "2023-03"
    assert out["yoy_change_pct"] == 0.0
    assert out["seasonal_move"] is True
    (f,) = findings
    assert f.layer == "appendix" and f.confidence < 0.6
    assert "2023-03" in f.headline and "seasonal" in f.headline
    assert any("seasonal" in c for c in f.caveats)
    assert f.evidence["yoy_change_pct"] == 0.0


def test_genuine_level_shift_is_confirmed(tmp_path):
    dates, vals = _monthly("2023-01-01", "2024-03-31", lambda d: 15.0 if d >= pd.Timestamp("2024-03-01") else 10.0)
    out, findings = _run(tmp_path, dates, vals)
    assert out["seasonal_move"] is False
    assert out["yoy_change_pct"] == 0.5
    (f,) = findings
    assert f.layer == "analyst" and f.confidence == 0.6
    assert "vs the prior period" in f.headline and "50% above 2023-03" in f.headline
    assert not f.caveats


def test_short_history_gets_caveat(tmp_path):
    dates, vals = _monthly("2024-01-01", "2024-09-30", lambda d: 20.0 if d >= pd.Timestamp("2024-09-23") else 10.0)
    out, findings = _run(tmp_path, dates, vals)
    assert out["short_history"] is True
    assert out["yoy_change_pct"] is None
    (f,) = findings
    assert "less than a year of history" in f.caveats[0]
