"""Exploratory statistics are sampled above DSA_ANALYSIS_SAMPLE_ROWS, never below."""
import numpy as np
import pandas as pd

from src.tools.data_processing import CorrelationAnalysisTool, DetectOutliersTool, pearson_matrix


def _csv(tmp_path, n=3000):
    rng = np.random.default_rng(0)
    a = rng.normal(size=n)
    df = pd.DataFrame({"a": a, "b": a * 0.8 + rng.normal(scale=0.5, size=n), "c": rng.normal(size=n)})
    p = tmp_path / "d.csv"
    df.to_csv(p, index=False)
    return str(p)


def test_correlation_sampled_above_cap_with_caveat_and_full_n(tmp_path, monkeypatch):
    path = _csv(tmp_path)
    monkeypatch.setenv("DSA_ANALYSIS_SAMPLE_ROWS", "1000")
    out = CorrelationAnalysisTool().execute(path)
    assert (out["sampled_from"], out["sampled_to"]) == (3000, 1000)
    assert "random sample of 1,000 of 3,000" in out["sample_caveat"]
    assert out["top_correlations"][0]["n"] == 3000  # tests use every row


def test_below_cap_untouched(tmp_path, monkeypatch):
    path = _csv(tmp_path)
    monkeypatch.setenv("DSA_ANALYSIS_SAMPLE_ROWS", "5000")
    out = CorrelationAnalysisTool().execute(path)
    assert "sampled_from" not in out


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
