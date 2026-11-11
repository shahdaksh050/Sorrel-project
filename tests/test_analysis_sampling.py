"""Exploratory statistics use deterministic, workload-aware correlation bounds."""
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd

from src.core.compute_policy import BackendDecision, ComputePolicy
from src.tools import data_processing
from src.tools.data_processing import CorrelationAnalysisTool, DetectOutliersTool, pearson_matrix


def _csv(tmp_path, n=3000):
    rng = np.random.default_rng(0)
    a = rng.normal(size=n)
    df = pd.DataFrame({"a": a, "b": a * 0.8 + rng.normal(scale=0.5, size=n), "c": rng.normal(size=n)})
    p = tmp_path / "d.csv"
    df.to_csv(p, index=False)
    return str(p)


def test_correlation_sampled_when_pair_work_exceeds_budget_with_full_n(tmp_path, monkeypatch):
    path = _csv(tmp_path)
    monkeypatch.setenv("DSA_CORR_WORK_BUDGET", "5000")
    out = CorrelationAnalysisTool().execute(path)
    assert (out["sampled_from"], out["sampled_to"]) == (3000, 1666)
    assert "random sample of 1,666 of 3,000" in out["sample_caveat"]
    assert out["top_correlations"][0]["n"] == 3000  # tests use every row
    assert out["sampling"] == {"applied": True, "seed": 0, "reason": "correlation work budget"}
    assert out["approximation"] == "seeded_sample"


def test_below_cap_untouched(tmp_path, monkeypatch):
    path = _csv(tmp_path)
    monkeypatch.setenv("DSA_ANALYSIS_SAMPLE_ROWS", "1000")
    out = CorrelationAnalysisTool().execute(path)
    assert "sampled_from" not in out
    assert out["approximation"] == "none"
    assert out["excluded_columns"] == []
    assert out["compute_backend"] == out["compute_device"] == "cpu"


def test_complete_dense_pearson_uses_admitted_array_backend(monkeypatch):
    class _Pool:
        def free_all_blocks(self) -> None:
            return None

    fake_cupy = SimpleNamespace(
        float64=np.float64,
        asarray=np.asarray,
        corrcoef=np.corrcoef,
        asnumpy=np.asarray,
        get_default_memory_pool=lambda: _Pool(),
        get_default_pinned_memory_pool=lambda: _Pool(),
    )
    monkeypatch.setitem(sys.modules, "cupy", fake_cupy)
    monkeypatch.setattr(
        data_processing,
        "select_array_backend",
        lambda *args, **kwargs: BackendDecision("cuda", "Test GPU", "verified_array_cuda", 10),
    )
    frame = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [2.0, 4.0, 8.0]})

    observed, decision = data_processing._accelerated_pearson_matrix(
        frame, ComputePolicy(mode="cuda")
    )

    np.testing.assert_allclose(observed.to_numpy(), frame.corr().to_numpy(), atol=1e-12)
    assert decision.backend == "cuda"


def test_reported_pair_p_values_have_their_own_row_guard(tmp_path, monkeypatch):
    path = _csv(tmp_path)
    monkeypatch.setenv("DSA_CORR_PVALUE_ROWS", "1000")
    out = CorrelationAnalysisTool().execute(path)
    pair = out["top_correlations"][0]
    assert pair["n"] == 1000
    assert pair["p_value_sampled_from"] == 3000
    assert out["p_value_row_budget"] == 1000


def test_wide_spearman_is_column_bounded_and_seeded_before_pairwise_work(tmp_path, monkeypatch):
    rng = np.random.default_rng(3)
    rows, columns = 1500, 80
    frame = pd.DataFrame(rng.lognormal(size=(rows, columns)), columns=[f"feature_{i:03}" for i in range(columns)])
    frame.iloc[::7, 2] = np.nan
    frame["target"] = rng.normal(size=rows)
    path = tmp_path / "wide.csv"
    frame.to_csv(path, index=False)
    monkeypatch.setenv("DSA_CORR_WORK_BUDGET", "1000")
    monkeypatch.setenv("DSA_CORR_MAX_COLUMNS", "12")

    observed_shapes: list[tuple[int, int]] = []
    original = data_processing._accelerated_spearman_matrix

    def recording_spearman(
        df: pd.DataFrame, policy: ComputePolicy,
    ) -> tuple[pd.DataFrame, BackendDecision]:
        observed_shapes.append(df.shape)
        return original(df, policy)

    monkeypatch.setattr(data_processing, "_accelerated_spearman_matrix", recording_spearman)
    out = CorrelationAnalysisTool().execute(str(path), method="spearman", target_column="target")

    assert observed_shapes == [(15, 12)]
    assert out["n_features"] == 12
    assert out["excluded_columns"]
    assert "target" in out["features_analyzed"]
    assert out["correlation_budget"]["estimated_work_after"] <= 1_000


def test_isolation_forest_scores_every_row(tmp_path, monkeypatch):
    path = _csv(tmp_path)
    monkeypatch.setenv("DSA_ANALYSIS_SAMPLE_ROWS", "1000")
    out = DetectOutliersTool().execute(path, method="isolation_forest", output_dir=str(tmp_path))
    assert out["n_rows_scored"] == 3000 and out["sampled_to"] == 1000


def test_pearson_matrix_matches_pandas_with_missing_values():
    rng = np.random.default_rng(1)
    df = pd.DataFrame(rng.normal(size=(400, 40)))
    df.iloc[::7, 3] = np.nan
    df.iloc[::11, 5] = np.nan
    np.testing.assert_allclose(pearson_matrix(df).to_numpy(), df.corr().to_numpy(), atol=1e-10)
