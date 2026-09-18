"""
Dimensionality Analysis Tool — Execution Layer.

Stage 3: PCA variance structure and multicollinearity screening for
datasets with many numeric features (DatasetProfile.is_high_dimensional).

Answers two questions a data scientist asks before modelling wide data:
  - How many components capture most of the variance? (PCA)
  - Which feature pairs are redundant enough to cause instability? (|r| > 0.9)
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from src.core.findings import Finding
from src.tools.base import BaseTool, ToolExecutionError
from src.tools.clustering import _select_cluster_features
from src.tools.data_processing import _read_df

if TYPE_CHECKING:
    from src.core.memory import DatasetMetadata
    from src.core.profiler import DatasetProfile

#: |correlation| at/above this between two features is reported as redundant.
HIGH_CORRELATION_THRESHOLD = 0.9


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

    def applies_to(self, profile: DatasetProfile | None, metadata: DatasetMetadata | None) -> float:
        return 1.0 if profile is not None and profile.is_high_dimensional else 0.0

    def execute(  # type: ignore[override]
        self,
        file_path: str,
        target_column: str | None = None,
        variance_threshold: float = 0.95,
        **_: Any,
    ) -> dict[str, Any]:
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler

        df = _read_df(file_path)
        if target_column and target_column in df.columns:
            df = df.drop(columns=[target_column])
        features = _select_cluster_features(df)

        if features.shape[1] < 2:
            raise ToolExecutionError(
                "Dimensionality analysis needs at least 2 usable numeric features; "
                f"found {features.shape[1]}."
            )

        filled = features.fillna(features.median(numeric_only=True))

        # ---- Multicollinearity screen ----
        corr = filled.corr(method="pearson")
        cols = corr.columns.tolist()
        high_corr_pairs: list[dict[str, Any]] = []
        for i, ca in enumerate(cols):
            for cb in cols[i + 1:]:
                val = float(corr.loc[ca, cb])
                if not np.isnan(val) and abs(val) >= HIGH_CORRELATION_THRESHOLD:
                    high_corr_pairs.append({"col_a": ca, "col_b": cb, "correlation": round(val, 4)})
        high_corr_pairs.sort(key=lambda x: abs(x["correlation"]), reverse=True)

        # ---- PCA ----
        scaler = StandardScaler()
        X = scaler.fit_transform(filled)
        n_components = min(X.shape[0], X.shape[1])
        pca = PCA(n_components=n_components, random_state=42)
        pca.fit(X)

        explained = [round(float(v), 4) for v in pca.explained_variance_ratio_]
        cumulative = np.cumsum(explained)
        n_for_threshold = int(np.searchsorted(cumulative, variance_threshold) + 1)
        n_for_threshold = min(n_for_threshold, len(explained))

        return {
            "summary": (
                f"{features.shape[1]} numeric features → {n_for_threshold} PCA component(s) "
                f"explain {variance_threshold:.0%} of variance. "
                f"{len(high_corr_pairs)} feature pair(s) with |r| ≥ {HIGH_CORRELATION_THRESHOLD}."
            ),
            "n_features": int(features.shape[1]),
            "features_used": list(features.columns),
            "explained_variance_ratio": explained,
            "cumulative_variance": [round(float(v), 4) for v in cumulative],
            "n_components_for_threshold": n_for_threshold,
            "variance_threshold": variance_threshold,
            "high_correlation_pairs": high_corr_pairs,
            "multicollinearity_risk": bool(high_corr_pairs),
        }

    def findings(  # type: ignore[override]
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
        }
