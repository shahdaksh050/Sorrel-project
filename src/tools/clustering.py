"""
Clustering Tool — Execution Layer.

Stage 3: Unsupervised segmentation (ClusterDataTool).

Implements the path the rest of the system only promised: when no target
column exists (or the user asks about segments), the agent can discover
natural groups in the data:

  - KMeans with automatic k selection via silhouette score (k = 2..max_k)
  - Standard-scaled continuous measures only: the target, identifiers, flags,
    ordinal codes, coordinates and year/time columns are excluded, and
    severely right-skewed non-negative features are log1p'd first
  - Per-cluster profiles (feature means) so clusters are interpretable
  - 2-D PCA coordinates (sampled) for the dashboard scatter
  - Deterministic throughout (random_state=42)
"""
from __future__ import annotations

import pickle
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import pandas as pd

from src.core.findings import Finding
from src.core.profiler import SEVERE_SKEW_THRESHOLD, profile_dataframe
from src.core.stats_utils import measure_aggregation, repeated_entity
from src.tools.base import BaseTool, ToolExecutionError
from src.tools.ml_pipeline import _read_df

if TYPE_CHECKING:
    from src.core.memory import DatasetMetadata
    from src.core.profiler import DatasetProfile

#: Default upper bound for the automatic k search.
DEFAULT_MAX_K = 8

#: Fewest units (rows, or entities after aggregation) worth clustering.
MIN_CLUSTER_UNITS = 20

#: Rows sampled for silhouette scoring and PCA scatter (keeps big data fast).
SILHOUETTE_SAMPLE = 2_000
PCA_POINT_CAP = 1_000

#: Cluster profiles report at most this many features (highest variance first).
MAX_PROFILE_FEATURES = 8


#: Name fragments marking a column as a coordinate or calendar field —
#: numeric, but distance on them is not a similarity between rows.
_NON_FEATURE_NAME_TOKENS = frozenset({
    "lat", "latitude", "lon", "lng", "long", "longitude",
    "year", "yr", "month", "day", "week", "hour", "quarter", "zip", "zipcode",
})


def _select_cluster_features(
    df: pd.DataFrame, profile: DatasetProfile | None, target_column: str | None
) -> pd.DataFrame:
    """
    Numeric feature matrix suitable for distance-based clustering.

    Only continuous measures qualify: the target (clustering on it just
    rediscovers the label), identifiers, 0/1 flags, ordinal codes,
    coordinates and year/time columns all carry numbers whose distances mean
    nothing. Uses the profile's semantic roles when available, and the
    near-unique-integer identifier check either way.
    """
    roles = {c.name: c.semantic_role for c in profile.columns} if profile else {}
    geo = {profile.geo_lat_col, profile.geo_lon_col} if profile else set()
    keep: list[str] = []
    n = max(len(df), 1)
    for col in df.columns:
        series = df[col]
        name_l = str(col).lower()
        if col == target_column or col in geo:
            continue
        if roles and roles.get(str(col)) != "measure":
            continue
        if pd.api.types.is_datetime64_any_dtype(series):
            continue
        if not pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
            continue
        if _NON_FEATURE_NAME_TOKENS & set(re.split(r"[^a-z]+", name_l)):
            continue
        nunique = series.nunique(dropna=True)
        if nunique <= 2:
            continue  # constant or flag
        if pd.api.types.is_integer_dtype(series) and nunique / n >= 0.98:
            continue  # identifier-like
        if pd.api.types.is_integer_dtype(series) and series.between(1900, 2100).all():
            continue  # calendar year stored as a number
        keep.append(str(col))
    return df[keep]


def _per_entity_features(
    features: pd.DataFrame, df: pd.DataFrame, profile: DatasetProfile | None
) -> tuple[pd.DataFrame, str]:
    """One feature row per entity when rows are repeated measurements of
    the same entity (orders per customer, readings per sensor) — otherwise
    heavy entities dominate the geometry and segments describe rows, not
    customers. Each feature combines by its own measure_aggregation (sum for
    spend, mean for a temperature). Returns (features, unit_of_analysis);
    falls back to rows when too few entities remain to cluster."""
    entity_col = repeated_entity(profile, df)
    if not entity_col or entity_col in features.columns or profile is None or features.empty:
        return features, "row"
    by_name = {c.name: c for c in profile.columns}
    aggs = {str(c): measure_aggregation(by_name.get(str(c))) for c in features.columns}
    grouped = features.groupby(df[entity_col], dropna=True)
    # An entity with no observed value stays missing rather than summing to 0.
    per_entity = grouped.agg(aggs).where(grouped.count() > 0)
    if len(per_entity) < MIN_CLUSTER_UNITS:
        return features, "row"
    return per_entity, entity_col


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
    requires_context: ClassVar[dict[str, str]] = {"target_column": "target_column"}

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
        target_column: str | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        import numpy as np
        from sklearn.cluster import KMeans
        from sklearn.decomposition import PCA
        from sklearn.metrics import silhouette_score
        from sklearn.preprocessing import StandardScaler

        df = _read_df(file_path)
        try:
            profile = profile_dataframe(df)
        except Exception:
            profile = None
        features = _select_cluster_features(df, profile, target_column)
        features, unit_of_analysis = _per_entity_features(features, df, profile)
        unit_noun = "rows" if unit_of_analysis == "row" else f"'{unit_of_analysis}' entities"

        if features.shape[1] < 2:
            raise ToolExecutionError(
                "Clustering needs at least 2 usable numeric features; "
                f"found {features.shape[1]}. Identifier, constant, datetime and "
                "non-numeric columns are excluded, as are the target, flags, ordinal codes, "
                "coordinates and year/time columns."
            )
        if len(features) < MIN_CLUSTER_UNITS:
            raise ToolExecutionError(
                f"Clustering needs at least {MIN_CLUSTER_UNITS} rows; dataset has {len(features)}."
            )

        # Median-impute then standardise — KMeans is distance-based
        filled = features.fillna(features.median(numeric_only=True))
        # A heavy right tail (income, spend) otherwise dominates the
        # Euclidean distance and KMeans just splits off the outliers.
        log_features = [
            c for c in filled.columns
            if float(filled[c].min()) >= 0 and float(filled[c].skew()) >= SEVERE_SKEW_THRESHOLD
        ]
        model_input = filled.copy()
        model_input[log_features] = np.log1p(model_input[log_features])
        scaler = StandardScaler()
        X = scaler.fit_transform(model_input)

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
            pickle.dump({
                "scaler": scaler, "kmeans": best_model, "features": list(features.columns),
                "log1p_features": log_features,
            }, f)

        # Kaufman & Rousseeuw: 0.71+ strong, 0.51-0.70 reasonable,
        # 0.26-0.50 weak (possibly artificial), <= 0.25 no real structure.
        quality = (
            "strong" if best_score >= 0.71
            else "reasonable" if best_score >= 0.51
            else "weak" if best_score > 0.25
            else "none"
        )
        return {
            "summary": (
                f"Found {best_k} clusters (silhouette={best_score:.3f}, "
                + ("no real structure" if quality == "none" else f"{quality} separation")
                + ") "
                f"across {len(features)} {unit_noun} × {features.shape[1]} numeric features."
            ),
            "unit_of_analysis": unit_of_analysis,
            "n_clusters": int(best_k),
            "silhouette_score": round(best_score, 4),
            "separation_quality": quality,
            "k_scores": k_scores,
            "cluster_sizes": cluster_sizes,
            "cluster_profiles": cluster_profiles,
            "features_used": list(features.columns),
            "log1p_features": log_features,
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
        if n_clusters and n_clusters >= 2 and silhouette is not None and quality == "none":
            # Nothing to segment on: a caveat, not a headline, and no
            # segment-profile finding built on the same geometry.
            return [
                Finding(
                    finding_id=f"{self.name}_segments",
                    kind="cluster",
                    headline=(
                        f"No distinct segments found (silhouette={silhouette:.3f}); "
                        "clusters would be artificial"
                    ),
                    detail="KMeans over the numeric features found no real cluster structure.",
                    evidence={"n_clusters": n_clusters, "silhouette_score": silhouette},
                    source_tool=self.name,
                    measure="silhouette_score",
                    effect=round(silhouette, 4),
                    effect_kind="r",
                    confidence=0.35,
                    surprise=0.1,
                    layer="appendix",
                )
            ]
        if n_clusters and n_clusters >= 2 and silhouette is not None:
            if quality == "weak":
                headline = (
                    f"Data may split into {n_clusters} segments, but separation is weak "
                    f"and may be an artefact (silhouette={silhouette:.3f})"
                )
            else:
                headline = (
                    f"Data splits into {n_clusters} segments with {quality} "
                    f"separation (silhouette={silhouette:.3f})"
                )
            results.append(
                Finding(
                    finding_id=f"{self.name}_segments",
                    kind="cluster",
                    headline=headline,
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
                    layer="exec" if quality in ("strong", "reasonable") else "analyst",
                )
            )

        profiles = output.get("cluster_profiles") or {}
        sizes = output.get("cluster_sizes") or {}
        if len(profiles) >= 2 and sizes:
            largest_name = max(sizes, key=lambda k: sizes[k])
            largest_profile = profiles.get(largest_name) or {}
            feature_names = list(largest_profile.keys())
            total = sum(sizes.get(name, 0) for name in profiles)
            if feature_names and total:
                # Row-weighted: the data's actual mean, not an average of
                # cluster means that over-weights small clusters.
                overall_mean = {
                    f: sum(p.get(f, 0.0) * sizes.get(name, 0) for name, p in profiles.items()) / total
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
                    unit = output.get("unit_of_analysis", "row")
                    unit_noun = "rows" if unit == "row" else f"'{unit}' entities"
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
                                f"Largest segment ({largest_name}, {sizes[largest_name]:,} {unit_noun}, "
                                f"{share * 100:.1f}% of data) stands out on '{defining_feature}' "
                                f"(mean {largest_profile[defining_feature]:.3g} vs "
                                f"{overall_mean[defining_feature]:.3g} overall)"
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
            "target_column": {
                "type": "string",
                "description": "Target/label column to exclude from the clustering features.",
                "required": False,
            },
        }
