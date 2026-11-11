"""Decision-table coverage for optional CUDA routing."""
from __future__ import annotations

import pytest

from src.core.compute_policy import (
    ComputePolicy,
    GpuCapability,
    estimate_array_vram_bytes,
    estimate_xgboost_vram_bytes,
    select_array_backend,
    select_xgboost_backend,
)


def _cap(**changes: object) -> GpuCapability:
    values: dict[str, object] = {
        "available": True,
        "device_name": "Test GPU",
        "free_vram_bytes": 2_000_000_000,
        "total_vram_bytes": 4_000_000_000,
        "xgboost_cuda": True,
        "reason": "cuda_capable",
    }
    values.update(changes)
    return GpuCapability(**values)  # type: ignore[arg-type]


def test_cpu_mode_never_probes_or_selects_cuda() -> None:
    decision = select_xgboost_backend(ComputePolicy(mode="cpu"), 1_000_000, 20, _cap())
    assert decision.backend == "cpu"
    assert decision.reason == "forced_cpu"


def test_auto_keeps_small_xgboost_workloads_on_cpu() -> None:
    decision = select_xgboost_backend(ComputePolicy(), 100, 10, _cap())
    assert decision.backend == "cpu"
    assert decision.reason == "workload_below_gpu_min_cells"


def test_requested_cuda_falls_back_when_xgboost_is_not_verified() -> None:
    decision = select_xgboost_backend(
        ComputePolicy(mode="cuda"), 100_000, 10, _cap(available=False, xgboost_cuda=False, reason="smoke_failed")
    )
    assert decision.backend == "cpu"
    assert decision.reason == "smoke_failed"


def test_cuda_is_selected_for_verified_safe_workload() -> None:
    decision = select_xgboost_backend(ComputePolicy(mode="cuda"), 100_000, 10, _cap())
    assert decision.backend == "cuda"
    assert decision.device == "Test GPU"
    assert decision.reason == "verified_xgboost_cuda"


def test_insufficient_free_vram_is_a_deterministic_cpu_fallback() -> None:
    decision = select_xgboost_backend(
        ComputePolicy(mode="cuda"), 100_000, 10, _cap(free_vram_bytes=1),
    )
    assert decision.backend == "cpu"
    assert decision.reason == "insufficient_free_vram"
    assert decision.estimated_vram_bytes == estimate_xgboost_vram_bytes(100_000, 10)


def test_policy_rejects_invalid_mode() -> None:
    with pytest.raises(ValueError, match="mode"):
        ComputePolicy(mode="metal")  # type: ignore[arg-type]


def test_array_backend_requires_complete_matrix() -> None:
    decision = select_array_backend(
        ComputePolicy(mode="cuda"), 20_000, 40, has_missing=True,
        capability=_cap(array_cuda=True),
    )
    assert decision.backend == "cpu"
    assert decision.reason == "pairwise_missing_gpu_unsupported"


def test_array_backend_uses_verified_cuda_with_safe_vram() -> None:
    decision = select_array_backend(
        ComputePolicy(mode="cuda"), 20_000, 40, has_missing=False,
        capability=_cap(array_cuda=True),
    )
    assert decision.backend == "cuda"
    assert decision.estimated_vram_bytes == estimate_array_vram_bytes(20_000, 40)
