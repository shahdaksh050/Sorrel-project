"""
Clustering Tool — Execution Layer.

Stage 3: Unsupervised segmentation (ClusterDataTool).

Implements the path the rest of the system only promised: when no target
column exists (or the user asks about segments), the agent can discover
natural groups in the data:

  - KMeans with automatic k selection via silhouette score (k = 2..max_k)
  - Standard-scaled numeric features; identifiers/constants/datetimes excluded
  - Per-cluster profiles (feature means) so clusters are interpretable
  - 2-D PCA coordinates (sampled) for the dashboard scatter
  - Deterministic throughout (random_state=42)
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from src.core.findings import Finding
from src.tools.base import BaseTool, ToolExecutionError
from src.tools.ml_pipeline import _read_df

if TYPE_CHECKING:
    from src.core.memory import DatasetMetadata
    from src.core.profiler import DatasetProfile

#: Default upper bound for the automatic k search.
DEFAULT_MAX_K = 8

#: Rows sampled for silhouette scoring and PCA scatter (keeps big data fast).
SILHOUETTE_SAMPLE = 2_000
PCA_POINT_CAP = 1_000

#: Cluster profiles report at most this many features (highest variance first).
MAX_PROFILE_FEATURES = 8


def _select_cluster_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Numeric feature matrix suitable for distance-based clustering.

    Excludes datetimes, constants, and identifier-like columns (near-unique
    integers — cluster geometry on row IDs is meaningless).
    """
    keep: list[str] = []
    n = max(len(df), 1)
    for col in df.columns:
        series = df[col]
        if pd.api.types.is_datetime64_any_dtype(series):
            continue
        if not pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
            continue
        nunique = series.nunique(dropna=True)
        if nunique <= 1:
            continue
        if pd.api.types.is_integer_dtype(series) and nunique / n >= 0.98:
            continue  # identifier-like
        keep.append(str(col))
    return df[keep]


class ClusterDataTool(BaseTool):
    """Discover natural segments in the data with auto-tuned KMeans."""

    requires_ml = True

    name = "cluster_data"
    description = (
        "Segment the dataset into natural groups using KMeans clustering. "
        "Automatically selects the number of clusters (k) by silhouette score "
        "unless n_clusters is given. Returns cluster sizes, per-cluster feature "
        "profiles, the silhouette quality score, and 2-D PCA coordinates for "
        "visualisation. Use when there is no target column or when the user "
        "asks about segments/groups/personas."
    )
    output_subdir = "models"

    def applies_to(self, profile: DatasetProfile | None, metadata: DatasetMetadata | None) -> float:
        if metadata and metadata.target_column:
            # Still occasionally useful (segmenting features regardless of
            # label), but modelling the target is almost always the priority.
            return 0.4
        return 1.0

    def execute(  # type: ignore[override]
        self,
        file_path: str,
        n_clusters: int | None = None,
        max_k: int = DEFAULT_MAX_K,
        output_dir: str = "output/models",
        **_: Any,
    ) -> dict[str, Any]:
        import numpy as np
        from sklearn.cluster import KMeans
        from sklearn.decomposition import PCA
        from sklearn.metrics import silhouette_score
        from sklearn.preprocessing import StandardScaler

        df = _read_df(file_path)
        features = _select_cluster_features(df)

        if features.shape[1] < 2:
            raise ToolExecutionError(
                "Clustering needs at least 2 usable numeric features; "
                f"found {features.shape[1]}. Identifier, constant, datetime and "
                "non-numeric columns are excluded."
            )
        if len(features) < 20:
            raise ToolExecutionError(
                f"Clustering needs at least 20 rows; dataset has {len(features)}."
            )

        # Median-impute then standardise — KMeans is distance-based
        filled = features.fillna(features.median(numeric_only=True))
        scaler = StandardScaler()
        X = scaler.fit_transform(filled)

        rng = np.random.default_rng(42)
        sil_idx = (
            rng.choice(len(X), size=SILHOUETTE_SAMPLE, replace=False)
            if len(X) > SILHOUETTE_SAMPLE
            else np.arange(len(X))
        )

        # ---- k selection ----
        k_scores: dict[int, float] = {}
        if n_clusters is not None:
            if n_clusters < 2:
                raise ToolExecutionError("n_clusters must be >= 2.")
            candidates = [int(n_clusters)]
        else:
            upper = min(max(2, int(max_k)), len(features) - 1)
            candidates = list(range(2, upper + 1))

        best_k, best_score, best_model = -1, -2.0, None
        for k in candidates:
            model = KMeans(n_clusters=k, random_state=42, n_init="auto")
            labels = model.fit_predict(X)
            if len(set(labels[sil_idx])) < 2:
                continue
            score = float(silhouette_score(X[sil_idx], labels[sil_idx]))
            k_scores[k] = round(score, 4)
            if score > best_score:
                best_k, best_score, best_model = k, score, model

        if best_model is None:
            raise ToolExecutionError("KMeans failed to produce 2+ distinct clusters.")

        labels = best_model.predict(X)
        sizes = pd.Series(labels).value_counts().sort_index()
        cluster_sizes = {f"cluster_{int(c)}": int(v) for c, v in sizes.items()}

        # ---- per-cluster profiles on the most variable features ----
        profile_cols = (
            filled.var().sort_values(ascending=False).head(MAX_PROFILE_FEATURES).index
        )
        profiled = filled[profile_cols].assign(_cluster=labels)
        cluster_profiles: dict[str, dict[str, float]] = {}
        for c, group in profiled.groupby("_cluster"):
            cluster_profiles[f"cluster_{int(c)}"] = {
                str(col): round(float(group[col].mean()), 4) for col in profile_cols
            }

        # ---- 2-D PCA coordinates for the dashboard scatter ----
        pca = PCA(n_components=2, random_state=42)
        coords = pca.fit_transform(X)
        point_idx = (
            rng.choice(len(coords), size=PCA_POINT_CAP, replace=False)
            if len(coords) > PCA_POINT_CAP
            else np.arange(len(coords))
        )
        pca_points = [
            {
                "x": round(float(coords[i, 0]), 4),
                "y": round(float(coords[i, 1]), 4),
                "cluster": f"cluster_{int(labels[i])}",
            }
            for i in point_idx
        ]

        Path(output_dir).mkdir(parents=True, exist_ok=True)
        model_path = Path(output_dir) / "kmeans.pkl"
        with open(model_path, "wb") as f:
            pickle.dump({"scaler": scaler, "kmeans": best_model, "features": list(features.columns)}, f)

        quality = (
            "strong" if best_score >= 0.5
            else "moderate" if best_score >= 0.25
            else "weak"
        )
        return {
            "summary": (
                f"Found {best_k} clusters (silhouette={best_score:.3f}, {quality} separation) "
                f"across {len(features)} rows × {features.shape[1]} numeric features."
            ),
            "n_clusters": int(best_k),
            "silhouette_score": round(best_score, 4),
            "separation_quality": quality,
            "k_scores": k_scores,
            "cluster_sizes": cluster_sizes,
            "cluster_profiles": cluster_profiles,
            "features_used": list(features.columns),
            "pca_points": pca_points,
            "pca_explained_variance": [round(float(v), 4) for v in pca.explained_variance_ratio_],
            "model_path": str(model_path),
        }

    def findings(
        self,
        output: dict[str, Any],
        profile: DatasetProfile | None,
        metadata: DatasetMetadata | None,
    ) -> list[Finding]:
        results: list[Finding] = []

        n_clusters = output.get("n_clusters")
        silhouette = output.get("silhouette_score")
        quality = output.get("separation_quality")
        if n_clusters and n_clusters >= 2 and silhouette is not None:
            results.append(
                Finding(
                    finding_id=f"{self.name}_segments",
                    kind="cluster",
                    headline=(
                        f"Data splits into {n_clusters} segments with {quality} "
                        f"separation (silhouette={silhouette:.3f})"
                    ),
                    detail=(
                        f"KMeans over {len(output.get('features_used', []))} feature(s); "
                        "k selected by silhouette score."
                    ),
                    evidence={
                        "n_clusters": n_clusters,
                        "silhouette_score": silhouette,
                        "cluster_sizes": output.get("cluster_sizes"),
                    },
                    source_tool=self.name,
                    measure="silhouette_score",
                    effect=round(silhouette, 4),
                    effect_kind="r",
                    confidence=0.6 if quality != "weak" else 0.35,
                    surprise=0.3,
                    layer="exec" if quality != "weak" else "analyst",
                )
            )

        profiles = output.get("cluster_profiles") or {}
        sizes = output.get("cluster_sizes") or {}
        if len(profiles) >= 2 and sizes:
            largest_name = max(sizes, key=lambda k: sizes[k])
            largest_profile = profiles.get(largest_name) or {}
            feature_names = list(largest_profile.keys())
            if feature_names:
                overall_mean = {
                    f: sum(p.get(f, 0.0) for p in profiles.values()) / len(profiles)
                    for f in feature_names
                }

                def _rel_dev(feat: str) -> float:
                    om = float(overall_mean[feat])
                    v = float(largest_profile[feat])
                    denom = abs(om) if abs(om) > 1e-9 else 1.0
                    return abs(v - om) / denom

                defining_feature = max(feature_names, key=_rel_dev)
                deviation = _rel_dev(defining_feature)
                if deviation > 0.15:
                    total_rows = sum(sizes.values())
                    share = sizes[largest_name] / total_rows if total_rows else 0.0
                    # This finding rides on the same cluster geometry as the
                    # "segments" finding above, so it can't be more trustworthy
                    # than its parent: cap its confidence at the parent's
                    # (0.6 if separation isn't weak, else 0.35), and within
                    # that ceiling scale down with the actual silhouette
                    # rather than a flat constant. A weak silhouette (e.g.
                    # 0.087) now lands at/near the floor instead of a
                    # hard-coded 0.55 that ignored cluster quality entirely.
                    parent_confidence = 0.6 if quality != "weak" else 0.35
                    sil = float(silhouette) if silhouette is not None else 0.0
                    profile_confidence = round(min(parent_confidence, max(0.2, sil)), 3)
                    # `surprise` is gated the same way as confidence: a
                    # profile deviation drawn from weakly-separated clusters
                    # is not a noteworthy discovery the way the same
                    # deviation would be inside genuinely distinct groups,
                    # so it shouldn't get the sibling finding's full 0.3-0.4
                    # surprise weight in the ranking.
                    profile_surprise = 0.4 if quality != "weak" else 0.15
                    profile_caveats: list[str] = []
                    if quality == "weak":
                        profile_caveats.append(
                            f"Cluster separation is weak (silhouette={sil:.3f}) — this "
                            "segment profile may not reflect a genuinely distinct group "
                            "and should be treated as a soft pattern, not a hard segment."
                        )
                    results.append(
                        Finding(
                            finding_id=f"{self.name}_{largest_name}_profile",
                            kind="cluster",
                            headline=(
                                f"Largest segment ({largest_name}, {sizes[largest_name]:,} rows, "
                                f"{share * 100:.1f}% of data) stands out on '{defining_feature}' "
                                f"(mean {largest_profile[defining_feature]:.3g} vs "
                                f"{overall_mean[defining_feature]:.3g} across segments)"
                            ),
                            evidence={
                                "cluster": largest_name,
                                "size": sizes[largest_name],
                                "defining_feature": defining_feature,
                                "cluster_mean": largest_profile[defining_feature],
                                "overall_mean": overall_mean[defining_feature],
                                "silhouette_score": silhouette,
                                "separation_quality": quality,
                            },
                            source_tool=self.name,
                            measure=defining_feature,
                            dimension="cluster",
                            level=largest_name,
                            effect=round(min(deviation, 2.0), 4),
                            effect_kind="pct",
                            confidence=profile_confidence,
                            surprise=profile_surprise,
                            caveats=profile_caveats,
                        )
                    )

        return results

    def get_schema(self) -> dict[str, Any]:
        return {
            "file_path": {"type": "string", "description": "Path to the (cleaned) dataset.", "required": True},
            "n_clusters": {
                "type": "int",
                "description": "Fixed number of clusters. Omit to auto-select by silhouette.",
                "required": False,
            },
            "max_k": {
                "type": "int",
                "description": f"Upper bound for the automatic k search. Default: {DEFAULT_MAX_K}.",
                "required": False,
            },
            "output_dir": {
                "type": "string",
                "description": "Directory for the saved clustering model.",
                "required": False,
            },
        }
