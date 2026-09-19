"""Unit tests for RegressionAnalysisTool (src/tools/regression.py)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.tools.regression import RegressionAnalysisTool


def _frame(n: int = 800) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    x1, x2, noise = rng.normal(0, 1, n), rng.normal(0, 1, n), rng.normal(0, 1, n)
    plan = rng.choice(["basic", "plus", "pro"], n)
    y = 3.0 * x1 + 1.0 * x2 + np.where(plan == "pro", 2.0, 0.0) + rng.normal(0, 1.0, n)
    return pd.DataFrame({
        "spend": y, "tenure": x1, "visits": x2, "shoe_size": noise, "plan": plan,
        "visits_copy": x2 * 1.01 + rng.normal(0, 0.01, n),
    })


def _run(tmp_path: Path, df: pd.DataFrame, **params: Any) -> dict[str, Any]:
    path = tmp_path / "reg.csv"
    df.to_csv(path, index=False)
    result = RegressionAnalysisTool().run(file_path=str(path), **params)
    assert result.status == "success", result.error_message
    return result.output


def test_ols_recovers_drivers_and_is_out_of_sample_honest(tmp_path: Path) -> None:
    out = _run(tmp_path, _frame(), target_column="spend")
    assert out["model_type"] == "ols"
    by_feature = {c["feature"]: c for c in out["coefficients"] if c["term_kind"] != "level"}
    assert by_feature["tenure"]["effect"] == pytest.approx(3.0, abs=0.4)
    assert by_feature["tenure"]["significant_after_correction"] is True
    assert "each +1 SD in tenure" in by_feature["tenure"]["sentence"]
    assert out["cv_mean"] > 0.8 and out["train_test_gap"] < 0.05 and out["overfit_warnings"] == []
    assert out["importance"][0]["feature"] == "tenure"
    assert out["cv_scheme"] == "k_fold" and out["cv_folds"] == 5


def test_near_duplicate_predictor_is_pruned_for_collinearity(tmp_path: Path) -> None:
    out = _run(tmp_path, _frame(), target_column="spend")
    assert not {"visits", "visits_copy"} <= set(out["features_used"])
    assert out["diagnostics"]["max_vif"] < 5


def test_logistic_reports_odds_ratios_and_auc(tmp_path: Path) -> None:
    df = _frame()
    prob = 1.0 / (1.0 + np.exp(-1.5 * df["tenure"]))
    df["churned"] = np.random.default_rng(3).binomial(1, prob)
    out = _run(tmp_path, df.drop(columns=["spend"]), target_column="churned")
    assert out["model_type"] == "logistic" and out["metric"] == "auc"
    top = next(c for c in out["coefficients"] if c["feature"] == "tenure")
    assert top["odds_ratio"] > 1.5 and top["ci_lower"] < top["odds_ratio"] < top["ci_upper"]
    assert out["cv_mean"] > 0.7 and out["cv_scheme"] == "stratified_k_fold"


def test_skewed_positive_target_is_modelled_on_log_scale(tmp_path: Path) -> None:
    df = _frame()
    df["spend"] = np.exp(0.6 * df["tenure"] + np.random.default_rng(5).normal(0, 0.3, len(df)))
    out = _run(tmp_path, df, target_column="spend")
    assert out["target_transform"] == "log"
    tenure = next(c for c in out["coefficients"] if c["feature"] == "tenure")
    assert tenure["effect_scale"] == "percent" and "%" in tenure["sentence"]


def test_findings_are_driver_kind_with_corrected_p(tmp_path: Path) -> None:
    tool = RegressionAnalysisTool()
    out = _run(tmp_path, _frame(), target_column="spend")
    found = tool.findings(out, None, None)
    assert found and all(f.kind == "driver" and f.p_adjusted is not None for f in found)
    assert found[0].effect_kind == "r" and found[0].dimension == "tenure"


def test_unknown_target_is_an_error(tmp_path: Path) -> None:
    path = tmp_path / "reg.csv"
    _frame().to_csv(path, index=False)
    assert RegressionAnalysisTool().run(file_path=str(path), target_column="nope").status == "error"


def test_is_gated_as_a_model_fitting_tool_and_writes_nothing() -> None:
    tool = RegressionAnalysisTool()
    assert tool.requires_ml is True and tool.output_subdir is None and tool.executes_code is False
