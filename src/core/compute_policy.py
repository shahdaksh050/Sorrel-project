"""Typed, optional CUDA capability checks and deterministic backend routing.

The module deliberately has no dependency on tools or controllers.  CUDA is an
optimization: all decisions safely select CPU when an optional runtime is not
installed, cannot execute its smoke test, or does not have a safe amount of
free memory.
"""
from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from typing import Literal, cast

ComputeBackend = Literal["auto", "cpu", "cuda"]
SelectedBackend = Literal["cpu", "cuda"]


@dataclass(frozen=True)
class ComputePolicy:
    """Per-run compute limits.  ``from_env`` is the CLI-safe default."""

    mode: ComputeBackend = "auto"
    gpu_min_cells: int = 200_000
    gpu_memory_fraction: float = 0.60
    max_stat_rows: int = 50_000
    max_corr_rows: int = 20_000
    max_corr_columns: int = 64
    tool_timeout_s: float = 120.0

    def __post_init__(self) -> None:
        if self.mode not in {"auto", "cpu", "cuda"}:
            raise ValueError("mode must be one of: auto, cpu, cuda")
        if self.gpu_min_cells < 1 or not 0.0 < self.gpu_memory_fraction <= 1.0:
            raise ValueError("GPU thresholds must be positive and memory fraction in (0, 1]")

    @classmethod
    def from_env(cls) -> ComputePolicy:
        """Build a policy from public, non-secret environment configuration."""
        mode = os.getenv("COMPUTE_BACKEND", "auto").strip().lower()
        if mode not in {"auto", "cpu", "cuda"}:
            mode = "auto"

        def integer(name: str, default: int, minimum: int = 1) -> int:
            try:
                return max(minimum, int(os.getenv(name, str(default))))
            except ValueError:
                return default

        def fraction(name: str, default: float) -> float:
            try:
                return min(1.0, max(0.05, float(os.getenv(name, str(default)))))
            except ValueError:
                return default

        return cls(
            mode=cast(ComputeBackend, mode),
            gpu_min_cells=integer("GPU_MIN_CELLS", 200_000),
            gpu_memory_fraction=fraction("GPU_MEMORY_FRACTION", 0.60),
            max_stat_rows=integer("MAX_STAT_ROWS", 50_000),
            max_corr_rows=integer("MAX_CORR_ROWS", 20_000),
            max_corr_columns=integer("MAX_CORR_COLUMNS", 64, minimum=2),
            tool_timeout_s=float(integer("TOOL_TIMEOUT_S", 120)),
        )


@dataclass(frozen=True)
class GpuCapability:
    """A probe result; memory facts are refreshed for every selection."""

    available: bool
    device_name: str = "cpu"
    free_vram_bytes: int | None = None
    total_vram_bytes: int | None = None
    xgboost_cuda: bool = False
    array_cuda: bool = False
    reason: str = "cuda_not_available"


@dataclass(frozen=True)
class BackendDecision:
    """Serializable routing result returned by eligible execution tools."""

    backend: SelectedBackend
    device: str
    reason: str
    estimated_vram_bytes: int = 0

    def as_telemetry(self) -> dict[str, str | int]:
        return {
            "compute_backend": self.backend,
            "compute_device": self.device,
            "backend_reason": self.reason,
            "estimated_vram_bytes": self.estimated_vram_bytes,
        }


@lru_cache(maxsize=1)
def _hardware_facts() -> tuple[str, int] | None:
    """Return stable device name/total bytes without requiring CUDA Python libs."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, check=False, text=True, timeout=3,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0 or not result.stdout.strip():
        return None
    name, _, total = result.stdout.splitlines()[0].partition(",")
    try:
        return name.strip(), int(float(total.strip())) * 1024 * 1024
    except ValueError:
        return None


def _free_vram_bytes() -> int | None:
    """Read volatile free VRAM; deliberately do not cache this value."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, check=False, text=True, timeout=3,
        )
        if result.returncode == 0 and result.stdout.strip():
            return int(float(result.stdout.splitlines()[0].strip())) * 1024 * 1024
    except (OSError, subprocess.SubprocessError, ValueError):
        pass
    return None


def _probe_xgboost_cuda() -> bool:
    """Verify XGBoost can allocate and run a tiny CUDA training operation."""
    try:
        import numpy as np
        import xgboost as xgb

        matrix = xgb.DMatrix(np.array([[0.0], [1.0]], dtype=np.float32), label=np.array([0, 1]))
        xgb.train({"tree_method": "hist", "device": "cuda", "objective": "binary:logistic"}, matrix, 1)
    except Exception:
        return False
    return True


def _probe_array_cuda() -> bool:
    """Verify optional CuPy can allocate and execute on CUDA."""
    try:
        import cupy as cp

        return bool(cp.asarray([1.0], dtype=cp.float32).sum().item() == 1.0)
    except Exception:
        return False


def probe_gpu_capability(*, require_xgboost: bool = False, require_array: bool = False) -> GpuCapability:
    """Probe physical memory plus the requested *actual* CUDA implementation."""
    facts = _hardware_facts()
    if facts is None:
        return GpuCapability(available=False, reason="nvidia_gpu_not_detected")
    name, total = facts
    free = _free_vram_bytes()
    xgboost_ok = _probe_xgboost_cuda() if require_xgboost else False
    array_ok = _probe_array_cuda() if require_array else False
    supported = xgboost_ok or array_ok or (not require_xgboost and not require_array)
    reason = "cuda_capable" if supported else "requested_cuda_library_unavailable_or_failed_smoke_test"
    return GpuCapability(
        available=supported, device_name=name, free_vram_bytes=free, total_vram_bytes=total,
        xgboost_cuda=xgboost_ok, array_cuda=array_ok, reason=reason,
    )


def estimate_xgboost_vram_bytes(rows: int, columns: int) -> int:
    """Conservative DMatrix + histogram workspace estimate for admission control."""
    cells = max(rows, 0) * max(columns, 0)
    return cells * 16 + max(rows, 0) * 128 + 32 * 1024 * 1024


def estimate_array_vram_bytes(rows: int, columns: int) -> int:
    """Conservative float64 input, centered workspace, and square output estimate."""
    safe_rows = max(rows, 0)
    safe_columns = max(columns, 0)
    return safe_rows * safe_columns * 24 + safe_columns * safe_columns * 16 + 16 * 1024 * 1024


def select_array_backend(
    policy: ComputePolicy,
    rows: int,
    columns: int,
    *,
    has_missing: bool,
    capability: GpuCapability | None = None,
) -> BackendDecision:
    """Route a dense numeric matrix to optional CuPy when semantics and memory permit."""
    if policy.mode == "cpu":
        return BackendDecision("cpu", "cpu", "forced_cpu")
    estimate = estimate_array_vram_bytes(rows, columns)
    if has_missing:
        return BackendDecision("cpu", "cpu", "pairwise_missing_gpu_unsupported", estimate)
    cells = max(rows, 0) * max(columns, 0)
    if policy.mode == "auto" and cells < policy.gpu_min_cells:
        return BackendDecision("cpu", "cpu", "workload_below_gpu_min_cells", estimate)
    cap = capability or probe_gpu_capability(require_array=True)
    if not cap.available or not cap.array_cuda:
        return BackendDecision("cpu", "cpu", cap.reason, estimate)
    if cap.free_vram_bytes is None:
        return BackendDecision("cpu", "cpu", "free_vram_unavailable", estimate)
    if estimate > int(cap.free_vram_bytes * policy.gpu_memory_fraction):
        return BackendDecision("cpu", "cpu", "insufficient_free_vram", estimate)
    return BackendDecision("cuda", cap.device_name, "verified_array_cuda", estimate)


def select_xgboost_backend(
    policy: ComputePolicy, rows: int, columns: int, capability: GpuCapability | None = None
) -> BackendDecision:
    """Select CUDA only when a verified XGBoost runtime has safe free VRAM."""
    if policy.mode == "cpu":
        return BackendDecision("cpu", "cpu", "forced_cpu")
    cells = max(rows, 0) * max(columns, 0)
    if policy.mode == "auto" and cells < policy.gpu_min_cells:
        return BackendDecision("cpu", "cpu", "workload_below_gpu_min_cells")
    cap = capability or probe_gpu_capability(require_xgboost=True)
    estimate = estimate_xgboost_vram_bytes(rows, columns)
    if not cap.available or not cap.xgboost_cuda:
        return BackendDecision("cpu", "cpu", cap.reason, estimate)
    if cap.free_vram_bytes is None:
        return BackendDecision("cpu", "cpu", "free_vram_unavailable", estimate)
    if estimate > int(cap.free_vram_bytes * policy.gpu_memory_fraction):
        return BackendDecision("cpu", "cpu", "insufficient_free_vram", estimate)
    return BackendDecision("cuda", cap.device_name, "verified_xgboost_cuda", estimate)
