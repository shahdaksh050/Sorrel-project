from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.tools.threshold_analysis import ThresholdAnalysisTool


def test_threshold_analysis_detects_piecewise_shift(tmp_path: Path) -> None:
    # Construct synthetic data with an abrupt slope break at x = 50.0
    rng = np.random.default_rng(42)
    n = 100
    x = np.linspace(10, 90, n)
    c_true = 50.0
    # Before 50: slope = 1.0; After 50: slope = 5.0 (delta = 4.0)
    y = 10.0 + 1.0 * x + 4.0 * np.maximum(0.0, x - c_true) + rng.normal(0, 0.5, n)

    df = pd.DataFrame({"load_factor": x, "failure_rate": y})
    csv_path = tmp_path / "load_data.csv"
    df.to_csv(csv_path, index=False)

    tool = ThresholdAnalysisTool()
    res = tool.run(file_path=str(csv_path), feature_column="load_factor", target_column="failure_rate")
    assert res.status == "success"
    out: dict[str, Any] = res.output
    assert "threshold" in out
    # Estimated threshold should be very close to the true break at 50.0
    assert abs(out["threshold"] - 50.0) <= 6.0
    assert out["f_pvalue"] < 0.01
    assert out["post_slope"] > out["pre_slope"]

    # Verify findings generation
    findings = tool.findings(out)
    assert len(findings) >= 1
    f = findings[0]
    assert f.kind == "threshold"
    assert f.chart_hint == "scatter"
    assert "critical threshold" in f.headline.lower()


def test_threshold_analysis_linear_data_no_false_break(tmp_path: Path) -> None:
    # Pure linear relationship with no structural breakpoint
    rng = np.random.default_rng(42)
    n = 80
    x = np.linspace(1, 100, n)
    y = 5.0 + 2.0 * x + rng.normal(0, 1.0, n)

    df = pd.DataFrame({"x": x, "y": y})
    csv_path = tmp_path / "linear_data.csv"
    df.to_csv(csv_path, index=False)

    tool = ThresholdAnalysisTool()
    res = tool.run(file_path=str(csv_path), feature_column="x", target_column="y")
    assert res.status == "success"
    out: dict[str, Any] = res.output
    # When no significant structural break exists, findings should be empty
    findings = tool.findings(out)
    assert len(findings) == 0
