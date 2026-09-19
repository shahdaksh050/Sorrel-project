"""Unit tests for ExperimentAnalysisTool (src/tools/experiment_analysis.py)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.tools.experiment_analysis import ExperimentAnalysisTool


def _run(tmp_path: Path, df: pd.DataFrame, **params: Any) -> dict[str, Any]:
    path = tmp_path / "exp.csv"
    df.to_csv(path, index=False)
    result = ExperimentAnalysisTool().run(file_path=str(path), **params)
    assert result.status == "success", result.error_message
    return result.output


def _rate_frame(n_control: int, n_variant: int, p_control: float, p_variant: float) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "variant": ["control"] * n_control + ["treatment"] * n_variant,
        "converted": np.concatenate([
            rng.binomial(1, p_control, n_control), rng.binomial(1, p_variant, n_variant),
        ]),
    })


def test_rate_lift_is_significant_with_plain_wording(tmp_path: Path) -> None:
    out = _run(tmp_path, _rate_frame(2000, 2000, 0.10, 0.14), group_column="variant", metric_column="converted")
    comp = out["comparisons"][0]
    assert out["metric_kind"] == "rate" and out["control_level"] == "control"
    assert comp["test"] == "two_proportion_z"
    assert comp["significant_after_correction"] is True
    assert comp["rel_lift"] == pytest.approx(0.4, abs=0.15)
    assert comp["ci_lower"] < comp["abs_diff"] < comp["ci_upper"]
    assert "lifts converted" in out["summary"] and "not chance" in out["summary"]
    assert not out["srm"]["flagged"]


def test_small_counts_use_fisher_and_report_mde(tmp_path: Path) -> None:
    out = _run(tmp_path, _rate_frame(60, 60, 0.05, 0.05), group_column="variant", metric_column="converted")
    comp = out["comparisons"][0]
    assert comp["test"] == "fisher_exact"
    assert comp["mde_abs"] > 0 and comp["underpowered"] in (True, False)


def test_sample_ratio_mismatch_is_flagged(tmp_path: Path) -> None:
    out = _run(tmp_path, _rate_frame(3000, 1000, 0.10, 0.10), group_column="variant", metric_column="converted")
    assert out["srm"]["flagged"] is True
    assert any(f.kind == "method_fit" for f in ExperimentAnalysisTool().findings(out, None, None))


def test_mean_metric_uses_welch_and_multiple_arms_are_corrected(tmp_path: Path) -> None:
    rng = np.random.default_rng(7)
    frames = [
        pd.DataFrame({"arm": name, "revenue": rng.normal(mu, 5.0, 300)})
        for name, mu in (("control", 50.0), ("B", 50.5), ("C", 56.0))
    ]
    out = _run(tmp_path, pd.concat(frames, ignore_index=True), group_column="arm", metric_column="revenue")
    by_level = {c["level"]: c for c in out["comparisons"]}
    assert by_level["C"]["test"] == "welch_t" and by_level["C"]["significant_after_correction"]
    assert not by_level["B"]["significant_after_correction"]
    assert out["omnibus"] is not None
    assert all("p_adjusted" in c for c in out["comparisons"])


def test_findings_only_for_significant_comparisons(tmp_path: Path) -> None:
    tool = ExperimentAnalysisTool()
    out = _run(tmp_path, _rate_frame(2000, 2000, 0.10, 0.14), group_column="variant", metric_column="converted")
    found = tool.findings(out, None, None)
    assert found and found[0].kind == "test" and found[0].effect_kind == "lift"
    assert found[0].p_adjusted is not None and found[0].source_tool == "experiment_analysis"


def test_errors_on_single_arm(tmp_path: Path) -> None:
    df = pd.DataFrame({"variant": ["control"] * 50, "converted": [0, 1] * 25})
    path = tmp_path / "one.csv"
    df.to_csv(path, index=False)
    result = ExperimentAnalysisTool().run(file_path=str(path), group_column="variant", metric_column="converted")
    assert result.status == "error"


def test_schema_uses_column_param_names() -> None:
    assert {"group_column", "metric_column", "control_level"} <= set(ExperimentAnalysisTool().get_schema())
