"""CUDA PCA routing stays numerically compatible with the CPU path."""
from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd

from src.core.compute_policy import BackendDecision, ComputePolicy
from src.tools import dimensionality
from src.tools.dimensionality import DimensionalityAnalysisTool, _gpu_pca_explained_variance


class _Pool:
    def free_all_blocks(self) -> None:
        return None


def _fake_cupy() -> SimpleNamespace:
    """Small NumPy-backed CuPy facade sufficient for the SVD execution path."""
    return SimpleNamespace(
        float64=np.float64,
        asarray=np.asarray,
        asnumpy=np.asarray,
        linalg=SimpleNamespace(svd=np.linalg.svd),
        get_default_memory_pool=lambda: _Pool(),
        get_default_pinned_memory_pool=lambda: _Pool(),
    )


def test_gpu_pca_variance_matches_cpu_svd(monkeypatch) -> None:
    values = np.random.default_rng(42).normal(size=(40, 6))
    monkeypatch.setitem(sys.modules, "cupy", _fake_cupy())
    monkeypatch.setattr(
        dimensionality,
        "select_array_backend",
        lambda *args, **kwargs: BackendDecision(
            "cuda", "Test GPU", "verified_array_cuda", 64
        ),
    )

    observed, decision = _gpu_pca_explained_variance(values, ComputePolicy(mode="cuda"))
    singular = np.linalg.svd(values, compute_uv=False, full_matrices=False)
    expected = singular**2 / np.sum(singular**2)

    assert decision.backend == "cuda"
    assert observed is not None
    np.testing.assert_allclose(observed, expected, rtol=1e-12, atol=1e-12)


def test_gpu_pca_cuda_failure_falls_back_to_cpu(monkeypatch) -> None:
    values = np.ones((8, 3), dtype=float)
    broken = _fake_cupy()
    broken.linalg = SimpleNamespace(
        svd=lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("CUDA OOM"))
    )
    monkeypatch.setitem(sys.modules, "cupy", broken)
    monkeypatch.setattr(
        dimensionality,
        "select_array_backend",
        lambda *args, **kwargs: BackendDecision(
            "cuda", "Test GPU", "verified_array_cuda", 64
        ),
    )

    observed, decision = _gpu_pca_explained_variance(values, ComputePolicy(mode="cuda"))

    assert observed is None
    assert decision.backend == "cpu"
    assert decision.reason == "cuda_pca_failed_cpu_fallback"


def test_gpu_pca_respects_cpu_policy_without_importing_cupy(monkeypatch) -> None:
    values = np.eye(3, dtype=float)
    monkeypatch.setattr(
        dimensionality,
        "select_array_backend",
        lambda *args, **kwargs: BackendDecision("cpu", "cpu", "forced_cpu", 64),
    )

    observed, decision = _gpu_pca_explained_variance(values, ComputePolicy(mode="cpu"))

    assert observed is None
    assert decision.backend == "cpu"
    assert decision.reason == "forced_cpu"


def test_dimensionality_cpu_telemetry_and_cuda_parity(tmp_path, monkeypatch) -> None:
    rng = np.random.default_rng(7)
    a = rng.normal(size=60)
    frame = pd.DataFrame(
        {"a": a, "b": a * 0.8 + rng.normal(scale=0.2, size=60), "c": rng.normal(size=60)}
    )
    path = tmp_path / "wide.csv"
    frame.to_csv(path, index=False)

    cpu = DimensionalityAnalysisTool().execute(str(path), compute_backend="cpu")
    assert cpu["compute_backend"] == "cpu"
    assert cpu["backend_reason"] == "forced_cpu"
    assert "component_top_features" in cpu
    assert len(cpu["component_top_features"]) == 3
    assert all(f in ("a", "b", "c") for f in cpu["component_top_features"])

    monkeypatch.setitem(sys.modules, "cupy", _fake_cupy())
    monkeypatch.setattr(
        dimensionality,
        "select_array_backend",
        lambda *args, **kwargs: BackendDecision(
            "cuda", "Test GPU", "verified_array_cuda", 64
        ),
    )
    gpu = DimensionalityAnalysisTool().execute(str(path), compute_backend="cuda")

    assert gpu["compute_backend"] == "cuda"
    assert gpu["compute_device"] == "Test GPU"
    assert gpu["estimated_vram_bytes"] == 64
    np.testing.assert_allclose(
        gpu["explained_variance_ratio"], cpu["explained_variance_ratio"], atol=0.0001
    )
