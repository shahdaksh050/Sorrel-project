"""
Dimensionality Analysis Tool — Execution Layer.

Stage 3: PCA variance structure and multicollinearity screening for
datasets with many numeric features (DatasetProfile.is_high_dimensional).

Answers two questions a data scientist asks before modelling wide data:
  - How many components capture most of the variance? (PCA)
  - Which feature pairs are redundant enough to cause instability? (|r| > 0.9)
"""
from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any, ClassVar, cast

import numpy as np

from src.core.compute_policy import (
    BackendDecision,
    ComputeBackend,
    ComputePolicy,
    select_array_backend,
)
from src.core.findings import Finding
from src.core.profiler import profile_dataframe
from src.tools.base import BaseTool, ToolExecutionError
from src.tools.clustering import _per_entity_features, _select_cluster_features
from src.tools.data_processing import _read_df, analysis_sample_rows, pearson_matrix, sample_note

if TYPE_CHECKING:
    from src.core.memory import DatasetMetadata
    from src.core.profiler import DatasetProfile

#: |correlation| at/above this between two features is reported as redundant.
HIGH_CORRELATION_THRESHOLD = 0.9


def _gpu_pca_explained_variance(
    values: np.ndarray,
    policy: ComputePolicy,
) -> tuple[np.ndarray | None, BackendDecision]:
    """Return PCA variance ratios via CuPy only when the dense matrix is safe.

    ``values`` has already been standardized with the CPU implementation used
    by the historical sklearn PCA path.  This keeps imputation and scaling
    semantics identical, while moving the bounded SVD to CUDA when available.
    A CUDA error is deliberately non-fatal: PCA is exploratory and the CPU
    implementation remains the authoritative fallback.
    """
    decision = select_array_backend(
        policy,
        int(values.shape[0]),
        int(values.shape[1]),
        has_missing=bool(np.isnan(values).any()),
    )
    if decision.backend != "cuda":
        return None, decision
    cupy_module: Any | None = None
    try:
        import cupy as cp

        cupy_module = cp
        gpu_values = cp.asarray(values, dtype=cp.float64)
        singular_values = cp.linalg.svd(gpu_values, compute_uv=False, full_matrices=False)
        squared = singular_values * singular_values
        total = float(cp.asnumpy(squared.sum()))
        if not np.isfinite(total) or total <= 0.0:
            raise ValueError("PCA variance is not positive")
        ratios = np.asarray(cp.asnumpy(squared / total), dtype=float)
        return ratios, decision
    except Exception:
        return None, BackendDecision(
            "cpu",
            "cpu",
            "cuda_pca_failed_cpu_fallback",
            decision.estimated_vram_bytes,
        )
    finally:
        # Release only CuPy's caching pools; CPU arrays stay owned by NumPy.
        try:
            if cupy_module is not None:
                cupy_module.get_default_memory_pool().free_all_blocks()
                cupy_module.get_default_pinned_memory_pool().free_all_blocks()
        except Exception:
            pass


class DimensionalityAnalysisTool(BaseTool):
    """PCA explained-variance structure plus a multicollinearity screen."""

    requires_ml = True

    name = "dimensionality_analysis"
    description = (
        "Analyse a wide numeric feature space: PCA explained variance per component "
        "(and how many components reach the variance_threshold), plus feature pairs "
        "with |correlation| > 0.9 (multicollinearity risk). Use when the data profile "
        "shows is_high_dimensional (many numeric features)."
    )
    requires_context: ClassVar[dict[str, str]] = {
        "target_column": "target_column",
        "ui_compute_backend": "compute_backend",
    }

    def applies_to(self, profile: DatasetProfile | None, metadata: DatasetMetadata | None) -> float:
        return 1.0 if profile is not None and profile.is_high_dimensional else 0.0

    def execute(  # type: ignore[override]
        self,
        file_path: str,
        target_column: str | None = None,
        variance_threshold: float = 0.95,
        compute_backend: str | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler

        df = _read_df(file_path)
        if target_column and target_column in df.columns:
            df = df.drop(columns=[target_column])
        profile = profile_dataframe(df)
        features, unit_of_analysis = _per_entity_features(
            _select_cluster_features(df, profile, None), df, profile
        )

        if features.shape[1] < 2:
            raise ToolExecutionError(
                "Dimensionality analysis needs at least 2 usable numeric features; "
                f"found {features.shape[1]}."
            )

        sampled: dict[str, Any] = {}
        if len(features) > analysis_sample_rows():
            sampled = sample_note(len(features), analysis_sample_rows())
            features = features.sample(n=analysis_sample_rows(), random_state=0)
        filled = features.fillna(features.median(numeric_only=True)).fillna(0.0)
        policy = ComputePolicy.from_env()
        if compute_backend is not None:
            try:
                policy = replace(policy, mode=cast(ComputeBackend, compute_backend))
            except ValueError as exc:
                raise ToolExecutionError(f"Invalid compute_backend: {compute_backend!r}") from exc

        # ---- Multicollinearity screen ----
        # Pairwise-complete on the raw values: median-filled rows would
        # dilute every correlation toward zero.
        corr = pearson_matrix(features)
        cols = corr.columns.tolist()
        values_r = corr.to_numpy(dtype=float)
        with np.errstate(invalid="ignore"):
            strong = np.triu(np.abs(values_r) >= HIGH_CORRELATION_THRESHOLD, 1)
        high_corr_pairs: list[dict[str, Any]] = [
            {"col_a": cols[i], "col_b": cols[j], "correlation": round(float(values_r[i, j]), 4)}
            for i, j in np.argwhere(strong)
        ]
        high_corr_pairs.sort(key=lambda x: abs(x["correlation"]), reverse=True)

        # ---- PCA ----
        scaler = StandardScaler()
        X = scaler.fit_transform(filled)
        n_components = min(X.shape[0], X.shape[1])
        gpu_explained, backend_decision = _gpu_pca_explained_variance(X, policy)
        if gpu_explained is None:
            pca = PCA(n_components=n_components, random_state=42)
            pca.fit(X)
            explained_values = pca.explained_variance_ratio_
        else:
            explained_values = gpu_explained

        explained = [round(float(v), 4) for v in explained_values]
        cumulative = np.cumsum(explained)
        n_for_threshold = int(np.searchsorted(cumulative, variance_threshold) + 1)
        n_for_threshold = min(n_for_threshold, len(explained))

        return {
            **sampled,
            "summary": (
                f"{features.shape[1]} numeric features → {n_for_threshold} PCA component(s) "
                f"explain {variance_threshold:.0%} of variance. "
                f"{len(high_corr_pairs)} feature pair(s) with |r| ≥ {HIGH_CORRELATION_THRESHOLD}."
                + (f" PCA and correlations {sampled['sample_caveat']}." if sampled else "")
            ),
            "n_features": int(features.shape[1]),
            **backend_decision.as_telemetry(),
            "features_used": list(features.columns),
            "explained_variance_ratio": explained,
            "cumulative_variance": [round(float(v), 4) for v in cumulative],
            "n_components_for_threshold": n_for_threshold,
            "variance_threshold": variance_threshold,
            "high_correlation_pairs": high_corr_pairs,
            "multicollinearity_risk": bool(high_corr_pairs),
            "unit_of_analysis": unit_of_analysis,
        }

    def findings(
        self,
        output: dict[str, Any],
        profile: DatasetProfile | None,
        metadata: DatasetMetadata | None,
    ) -> list[Finding]:
        results: list[Finding] = []

        n_features = output.get("n_features")
        n_for_threshold = output.get("n_components_for_threshold")
        variance_threshold = output.get("variance_threshold")
        if n_features and n_for_threshold:
            reduction = 1.0 - (n_for_threshold / n_features)
            first = (output.get("explained_variance_ratio") or [0.0])[0]
            main_story = n_features >= 30 or first > 0.6
            if reduction > 0.15:
                results.append(
                    Finding(
                        finding_id=f"{self.name}_variance",
                        kind="dimensionality",
                        headline=(
                            f"{n_for_threshold} of {n_features} features capture "
                            f"{variance_threshold:.0%} of the variance"
                        ),
                        detail=(
                            "PCA on standardised features; components ranked by "
                            "explained_variance_ratio."
                        ),
                        evidence={
                            "n_features": n_features,
                            "n_components_for_threshold": n_for_threshold,
                            "explained_variance_ratio": output.get("explained_variance_ratio"),
                        },
                        measure="explained_variance",
                        effect=round(reduction, 4),
                        effect_kind="share",
                        confidence=0.6,
                        surprise=0.3,
                        layer="analyst" if main_story else "appendix",
                    )
                )

        high_corr = output.get("high_correlation_pairs")
        if high_corr:
            top_pair = high_corr[0]
            results.append(
                Finding(
                    finding_id=f"{self.name}_multicollinearity",
                    kind="method_fit",
                    headline=(
                        f"{len(high_corr)} feature pair(s) are highly correlated "
                        f"(top: '{top_pair.get('col_a')}' & '{top_pair.get('col_b')}', "
                        f"r={top_pair.get('correlation')})"
                    ),
                    detail=(
                        "|r| >= 0.9 between these features — treat as redundant for "
                        "modelling; consider dropping one of each pair."
                    ),
                    evidence={"high_correlation_pairs": high_corr},
                    measure="correlation",
                    effect=top_pair.get("correlation"),
                    effect_kind="r",
                    confidence=0.8,
                    surprise=0.2,
                    caveats=[
                        f"{len(high_corr)} redundant feature pair(s) detected "
                        "(|r| >= 0.9) — multicollinearity risk for linear models."
                    ],
                )
            )

        return results

    def get_schema(self) -> dict[str, Any]:
        return {
            "file_path": {"type": "string", "description": "Path to the (cleaned) dataset.", "required": True},
            "target_column": {
                "type": "string",
                "description": "Excluded from the feature space if given.",
                "required": False,
            },
            "variance_threshold": {
                "type": "float",
                "description": "Cumulative variance fraction to report component count for. Default: 0.95.",
                "required": False,
            },
            "compute_backend": {
                "type": "string",
                "enum": ["auto", "cpu", "cuda"],
                "description": "Optional PCA backend preference. CUDA remains bounded and falls back to CPU.",
                "required": False,
            },
        }
