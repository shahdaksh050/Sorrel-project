"""Unit tests for src/tools/clustering.py — the cluster_data tool."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.tools.clustering import PCA_POINT_CAP, ClusterDataTool

RNG = np.random.default_rng(42)


@pytest.fixture
def blobs_csv(tmp_path: Path) -> str:
    """Three well-separated gaussian blobs — clustering should find k=3."""
    centers = [(0, 0), (10, 10), (-10, 10)]
    frames = []
    for i, (cx, cy) in enumerate(centers):
        frames.append(pd.DataFrame({
            "feat_x": RNG.normal(cx, 1.0, 100),
            "feat_y": RNG.normal(cy, 1.0, 100),
            "noise": RNG.normal(0, 1, 100),
            "row_id": range(i * 100, (i + 1) * 100),
        }))
    df = pd.concat(frames, ignore_index=True)
    p = tmp_path / "blobs.csv"
    df.to_csv(p, index=False)
    return str(p)


class TestClusterDataTool:
    def test_finds_three_blobs(self, blobs_csv: str, tmp_path: Path) -> None:
        result = ClusterDataTool().run(file_path=blobs_csv, output_dir=str(tmp_path / "m"))
        assert result.status == "success"
        out = result.output
        assert out["n_clusters"] == 3
        assert out["silhouette_score"] > 0.5
        assert out["separation_quality"] in ("strong", "reasonable")

    def test_identifier_column_excluded(self, blobs_csv: str, tmp_path: Path) -> None:
        result = ClusterDataTool().run(file_path=blobs_csv, output_dir=str(tmp_path / "m"))
        assert "row_id" not in result.output["features_used"]

    def test_cluster_sizes_sum_to_rows(self, blobs_csv: str, tmp_path: Path) -> None:
        result = ClusterDataTool().run(file_path=blobs_csv, output_dir=str(tmp_path / "m"))
        assert sum(result.output["cluster_sizes"].values()) == 300

    def test_profiles_cover_every_cluster(self, blobs_csv: str, tmp_path: Path) -> None:
        out = ClusterDataTool().run(file_path=blobs_csv, output_dir=str(tmp_path / "m")).output
        assert set(out["cluster_profiles"].keys()) == set(out["cluster_sizes"].keys())
        for profile in out["cluster_profiles"].values():
            assert "feat_x" in profile

    def test_fixed_n_clusters_respected(self, blobs_csv: str, tmp_path: Path) -> None:
        out = ClusterDataTool().run(
            file_path=blobs_csv, n_clusters=2, output_dir=str(tmp_path / "m")
        ).output
        assert out["n_clusters"] == 2

    def test_pca_points_capped_and_labelled(self, blobs_csv: str, tmp_path: Path) -> None:
        out = ClusterDataTool().run(file_path=blobs_csv, output_dir=str(tmp_path / "m")).output
        points = out["pca_points"]
        assert 0 < len(points) <= PCA_POINT_CAP
        assert all({"x", "y", "cluster"} <= set(p.keys()) for p in points)

    def test_model_saved(self, blobs_csv: str, tmp_path: Path) -> None:
        out = ClusterDataTool().run(file_path=blobs_csv, output_dir=str(tmp_path / "m")).output
        assert Path(out["model_path"]).exists()

    def test_too_few_features_errors(self, tmp_path: Path) -> None:
        p = tmp_path / "one_col.csv"
        pd.DataFrame({"only": RNG.normal(0, 1, 50)}).to_csv(p, index=False)
        result = ClusterDataTool().run(file_path=str(p))
        assert result.status == "error"
        assert "2 usable numeric features" in str(result.error_message)

    def test_too_few_rows_errors(self, tmp_path: Path) -> None:
        p = tmp_path / "tiny.csv"
        pd.DataFrame({"a": range(5), "b": RNG.normal(0, 1, 5)}).to_csv(p, index=False)
        result = ClusterDataTool().run(file_path=str(p))
        assert result.status == "error"

    def test_deterministic(self, blobs_csv: str, tmp_path: Path) -> None:
        a = ClusterDataTool().run(file_path=blobs_csv, output_dir=str(tmp_path / "a")).output
        b = ClusterDataTool().run(file_path=blobs_csv, output_dir=str(tmp_path / "b")).output
        assert a["silhouette_score"] == b["silhouette_score"]
        assert a["cluster_sizes"] == b["cluster_sizes"]
