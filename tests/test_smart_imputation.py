"""
Unit tests for MICE and KNN Smart Imputation in CleanDataTool (Phase 12).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.tools.data_processing import CleanDataTool


def test_mice_imputation(tmp_path: Path) -> None:
    csv_path = tmp_path / "test_data_mice.csv"
    np.random.seed(42)
    x1 = np.random.normal(10, 2, 50)
    x2 = 2 * x1 + np.random.normal(0, 1, 50)
    # Introduce NaNs
    x1[5] = np.nan
    x2[12] = np.nan

    df = pd.DataFrame({"x1": x1, "x2": x2, "cat": ["A", "B"] * 25})
    df.to_csv(csv_path, index=False)

    tool = CleanDataTool()
    result = tool.run(file_path=str(csv_path), strategy="mice", output_dir=str(tmp_path))

    assert result.status == "success"
    assert result.output["missing_after"] == 0
    cleaned_df = pd.read_csv(result.output["cleaned_file_path"])
    assert not cleaned_df["x1"].isnull().any()
    assert not cleaned_df["x2"].isnull().any()


def test_knn_imputation(tmp_path: Path) -> None:
    csv_path = tmp_path / "test_data_knn.csv"
    np.random.seed(42)
    x1 = np.random.normal(5, 1, 40)
    x2 = np.random.normal(50, 5, 40)
    x1[2] = np.nan
    x2[10] = np.nan

    df = pd.DataFrame({"x1": x1, "x2": x2, "label": ["C", "D"] * 20})
    df.to_csv(csv_path, index=False)

    tool = CleanDataTool()
    result = tool.run(file_path=str(csv_path), strategy="knn", output_dir=str(tmp_path))

    assert result.status == "success"
    assert result.output["missing_after"] == 0
    cleaned_df = pd.read_csv(result.output["cleaned_file_path"])
    assert not cleaned_df["x1"].isnull().any()
    assert not cleaned_df["x2"].isnull().any()
