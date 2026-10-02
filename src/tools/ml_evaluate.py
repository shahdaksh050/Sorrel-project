"""
ML Pipeline — EvaluateModelTool — held-out evaluation of a saved model.

Split out of ml_pipeline.py; `src.tools.ml_pipeline` re-exports every name here.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
import pandas as pd

from src.core.findings import Finding
from src.core.model_io import ModelIntegrityError, load_model
from src.tools.base import BaseTool, ToolExecutionError
from src.tools.data_processing import (
    _read_df as _read_df,  # re-exported: src.tools.clustering imports it from here
)

if TYPE_CHECKING:
    from src.core.memory import DatasetMetadata, MemorySystem
    from src.core.profiler import DatasetProfile
from src.tools.ml_common import (
    OVERFIT_THRESHOLD,
    _cap_train_rows,
    _encode_target,
    _prepare_features,
    _resolve_split_strategy,
    _split_train_test,
)

logger = logging.getLogger(__name__)


#: evaluate_model scores the train side (for the train-test gap) on at most
#: this many rows, and ranks drivers by permutation importance (one full
#: prediction pass per feature per repeat) on at most _PERMUTATION_MAX_ROWS
#: held-out rows; a seeded random sample above those sizes.
_EVAL_TRAIN_MAX_ROWS = 100_000


_PERMUTATION_MAX_ROWS = 25_000


def _seeded_rows(n: int, cap: int) -> np.ndarray | None:
    """Sorted positions of a seeded random sample of `cap` of `n` rows, or None
    when n <= cap (nothing is sampled)."""
    if n <= cap:
        return None
    return np.sort(np.random.default_rng(0).choice(n, cap, replace=False))


class EvaluateModelTool(BaseTool):
    """
    Load a saved model and run detailed evaluation on a held-out split.

    Recreates the same train/test split used by TrainModelTool
    (random_state=42) so the reported metrics describe generalisation,
    not memorisation. Produces a classification report or regression
    metrics plus the train-test gap as an overfitting diagnostic.
    """

    requires_ml = True

    name = "evaluate_model"
    description = (
        "Load a saved .pkl model and evaluate it on the held-out test split "
        "of a dataset (same random_state=42 split as train_model). "
        "Produces a full classification report (or regression metrics) "
        "plus an overfitting diagnostic (train_test_gap)."
    )
    requires_context: ClassVar[dict[str, str]] = {"target_column": "target_column"}

    def applies_to(self, profile: DatasetProfile | None, metadata: DatasetMetadata | None) -> float:
        return 1.0 if metadata and metadata.target_column and metadata.task_type in ("classification", "regression") else 0.0

    def prepare_params(
        self, params: dict[str, Any], memory: MemorySystem, output_root: str
    ) -> dict[str, Any]:
        params = super().prepare_params(params, memory, output_root)
        # best_model_path only overrides when the planner's model_path is
        # missing or doesn't exist — it may legitimately name a different
        # saved model on a re-evaluation step.
        best_path = memory.get_context("best_model_path")
        if best_path:
            raw_mp = params.get("model_path", "")
            if not raw_mp or not Path(raw_mp).exists():
                params["model_path"] = best_path
        # Pin model_path under the run's output directory before it reaches
        # execute()'s unsandboxed pickle.load() — a planner step naming any
        # other existing file path (e.g. from prompt-injected dataset
        # content) must not be honoured as-is; fall back to the trusted
        # best_model_path, or fail closed via a path execute() will reject.
        from src.core.security import UploadValidationError, resolve_output_path
        raw_mp = params.get("model_path", "")
        if raw_mp:
            try:
                params["model_path"] = str(resolve_output_path(output_root, raw_mp))
            except UploadValidationError:
                params["model_path"] = best_path or ""
        # evaluate_model's whole purpose is to recreate train_model's exact
        # split ("held-out data only") — a different test_size, or a
        # different split_strategy/time_column/group_column, produces a
        # different partition, so these are forced overrides, never a
        # fill-if-absent: a plan step naming a stale value must still lose
        # to what train_model actually used.
        trained_test_size = memory.get_context("train_test_size")
        if trained_test_size is not None:
            params["test_size"] = trained_test_size
        split_strategy = memory.get_context("split_strategy")
        if split_strategy is not None:
            params["split_strategy"] = split_strategy
            params["time_column"] = memory.get_context("split_time_column")
            params["group_column"] = memory.get_context("split_group_column")
        return params

    def findings(
        self,
        output: dict[str, Any],
        profile: DatasetProfile | None,
        metadata: DatasetMetadata | None,
    ) -> list[Finding]:
        """
        Project `_explain_drivers`'s upgraded output — per-driver level
        effects/directions, plus the overfit gap this tool itself measured —
        onto the finding bus (IMPROVEMENTS.md 7.2 item 4).
        """
        found: list[Finding] = []
        target_column = metadata.target_column if metadata else None
        task_type = output.get("task_type")

        for drv in output.get("top_drivers") or []:
            feature = drv.get("feature")
            importance = drv.get("importance")
            level_effect = drv.get("level_effect")
            headline = drv.get("headline") or f"'{feature}' is a top driver of {target_column or 'the outcome'}"
            if level_effect:
                level = level_effect.get("level")
                lift = level_effect.get("lift")
                effect = lift if lift is not None else level_effect.get("diff")
                effect_kind = "lift" if lift is not None else "pct"
                found.append(
                    Finding(
                        finding_id=f"{self.name}_driver_{feature}_{level}",
                        kind="driver",
                        headline=headline,
                        detail=f"Permutation importance {importance}.",
                        evidence={"level_effect": level_effect, "importance": importance},
                        source_tool=self.name,
                        measure=target_column,
                        dimension=feature,
                        level=str(level) if level is not None else None,
                        effect=round(float(effect), 4) if effect is not None else None,
                        effect_kind=effect_kind,
                        confidence=0.6,
                        surprise=0.4,
                        layer="analyst",
                    )
                )
            else:
                corr = drv.get("correlation")
                effect = corr if corr is not None else importance
                found.append(
                    Finding(
                        finding_id=f"{self.name}_driver_{feature}",
                        kind="driver",
                        headline=headline,
                        detail=f"Permutation importance {importance}.",
                        evidence={"importance": importance, "correlation": corr},
                        source_tool=self.name,
                        measure=target_column,
                        dimension=feature,
                        effect=round(float(effect), 4) if effect is not None else None,
                        effect_kind="r" if corr is not None else None,
                        confidence=0.5,
                        layer="analyst",
                    )
                )

        gap = output.get("train_test_gap")
        if isinstance(gap, (int, float)) and gap > OVERFIT_THRESHOLD:
            found.append(
                Finding(
                    finding_id=f"{self.name}_overfit_gap",
                    kind="method_fit",
                    headline=f"Train-test gap of {gap:+.3f} suggests possible overfitting",
                    detail=output.get("summary", ""),
                    evidence={"train_test_gap": gap, "task_type": task_type},
                    source_tool=self.name,
                    measure=target_column,
                    caveats=[f"train_test_gap={gap:+.4f} exceeds the {OVERFIT_THRESHOLD} threshold."],
                    confidence=0.6,
                    layer="analyst",
                )
            )

        return found

    def execute(  # type: ignore[override]
        self,
        model_path: str,
        file_path: str,
        target_column: str,
        task_type: str = "classification",
        test_size: float = 0.2,
        split_strategy: str = "random",
        time_column: str | None = None,
        group_column: str | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        from sklearn.metrics import classification_report

        if not Path(model_path).exists():
            raise ToolExecutionError(f"Model file not found: {model_path}")

        df = _read_df(file_path)
        target_words: dict[str, str] | None = (df.attrs.get("boolean_labels") or {}).get(target_column)
        if target_column not in df.columns:
            raise ToolExecutionError(f"Target column '{target_column}' not in dataset.")

        df, split_strategy, _notes = _resolve_split_strategy(
            df, split_strategy, time_column, group_column
        )
        X, y, _treatments = _prepare_features(df, target_column)
        class_labels: list[str] = []
        if task_type == "classification":
            y, class_labels = _encode_target(y, target_words)

        try:
            model = load_model(model_path)
        except ModelIntegrityError as exc:
            raise ToolExecutionError(str(exc)) from exc

        # Recreate train_model's exact split so evaluation runs on rows the
        # model never trained on, whichever strategy produced them.
        X_train, X_test, y_train, y_test, _cv, groups_train = _split_train_test(
            X, y, df, split_strategy, group_column, task_type, test_size
        )
        # Reapply the identical MAX_TRAIN_SAMPLES cap train_model applied
        # before fitting (same helper, same seed/ordering) — the persisted
        # model was fit on this subset, not the full pre-cap train split, so
        # the train-side score below must be measured on the same rows or
        # train_test_gap silently understates overfitting.
        X_train, y_train, _groups_train, cap_orig_len = _cap_train_rows(
            X_train, y_train, groups_train, split_strategy, task_type
        )
        # One predict_proba() pass on X_test serves three consumers below
        # (y_pred_test, _held_out_curves' ROC curve, _explain_drivers'
        # baseline outcome) instead of each calling predict()/predict_proba()
        # on the full held-out split separately — same Pipeline transform,
        # computed once. predict() == classes_[argmax(predict_proba())] for
        # every classifier this file trains (RF, XGBoost, LogisticRegression).
        y_prob_test: np.ndarray | None = None
        if task_type == "classification" and hasattr(model, "predict_proba"):
            y_prob_test = model.predict_proba(X_test)
            y_pred_test = model.classes_[np.argmax(y_prob_test, axis=1)]
        else:
            y_pred_test = model.predict(X_test)
        n_train_full = len(X_train)
        train_rows = _seeded_rows(n_train_full, _EVAL_TRAIN_MAX_ROWS)
        if train_rows is not None:
            X_train, y_train = X_train.iloc[train_rows], y_train.iloc[train_rows]
        y_pred_train = model.predict(X_train)

        perm_rows = _seeded_rows(len(X_test), _PERMUTATION_MAX_ROWS)
        drivers, driver_narrative = self._explain_drivers(
            model,
            X_test if perm_rows is None else X_test.iloc[perm_rows],
            y_test if perm_rows is None else y_test.iloc[perm_rows],
            task_type, target_column, class_labels,
            precomputed_proba=(
                y_prob_test if perm_rows is None or y_prob_test is None else y_prob_test[perm_rows]
            ),
        )
        sampling: dict[str, Any] = {}
        caveats = []
        # train_sample_from/to describe what the train-side score above was
        # actually computed on, so the caveat stays accurate whether that was
        # the model's full fit set, a MAX_TRAIN_SAMPLES-capped subset of it,
        # or (rarely, when the cap is raised/disabled) a further seeded
        # sample of a still-huge fit set.
        train_sample_from: int | None = None
        train_sample_to: int | None = None
        if train_rows is not None:
            train_sample_from, train_sample_to = n_train_full, len(X_train)
            caveats.append(
                f"train score computed on a random sample of {train_sample_to:,} of "
                f"{train_sample_from:,} rows actually used to fit the model"
                + (
                    f" (itself capped from {cap_orig_len:,} by MAX_TRAIN_SAMPLES)"
                    if cap_orig_len is not None
                    else ""
                )
            )
        elif cap_orig_len is not None:
            train_sample_from, train_sample_to = cap_orig_len, n_train_full
            caveats.append(
                f"train score computed on the {n_train_full:,} rows actually used to fit the "
                f"model (capped from {cap_orig_len:,} by MAX_TRAIN_SAMPLES)"
            )
        if perm_rows is not None:
            caveats.append(
                f"driver ranking computed on a random sample of {len(perm_rows):,} of {len(X_test):,} held-out rows"
            )
        if caveats:
            sampling = {
                "sampled_from": train_sample_from if train_sample_from is not None else len(X_test),
                "sampled_to": train_sample_to if train_sample_to is not None else _PERMUTATION_MAX_ROWS,
                "sample_caveat": "; ".join(caveats),
            }

        if task_type == "classification":
            from sklearn.metrics import accuracy_score

            report = classification_report(y_test, y_pred_test, output_dict=True, zero_division=0)
            test_acc = float(accuracy_score(y_test, y_pred_test))
            train_acc = float(accuracy_score(y_train, y_pred_train))
            gap = round(train_acc - test_acc, 4)
            # Map encoded integer class keys back to original label names
            if class_labels:
                report = {
                    (class_labels[int(k)] if k.isdigit() and int(k) < len(class_labels) else k): v
                    for k, v in report.items()
                }
            return {
                "summary": (
                    f"Held-out evaluation complete. Test accuracy: {test_acc:.4f} "
                    f"(train: {train_acc:.4f}, gap: {gap:+.4f})."
                ),
                "classification_report": report,
                "task_type": task_type,
                "accuracy": round(test_acc, 4),
                "train_accuracy": round(train_acc, 4),
                "train_test_gap": gap,
                "class_labels": class_labels,
                "top_drivers": drivers,
                "driver_narrative": driver_narrative,
                **sampling,
                **self._held_out_curves(model, X_test, y_test, y_pred_test, class_labels, y_prob_test),
            }
        else:
            from sklearn.metrics import mean_squared_error, r2_score

            rmse = float(np.sqrt(mean_squared_error(y_test, y_pred_test)))
            r2_test = float(r2_score(y_test, y_pred_test))
            r2_train = float(r2_score(y_train, y_pred_train))
            gap = round(r2_train - r2_test, 4)
            return {
                "summary": (
                    f"Held-out evaluation complete. RMSE: {rmse:.4f}, "
                    f"R²: {r2_test:.4f} (train R²: {r2_train:.4f}, gap: {gap:+.4f})."
                ),
                "rmse": round(rmse, 4),
                "r2": round(r2_test, 4),
                "train_r2": round(r2_train, 4),
                "train_test_gap": gap,
                "task_type": task_type,
                "top_drivers": drivers,
                "driver_narrative": driver_narrative,
                **sampling,
            }

    #: ROC points kept for the dashboard — enough for a smooth curve, small
    #: enough to inline into dashboard.json and the HTML report.
    _MAX_ROC_POINTS = 100

    @classmethod
    def _held_out_curves(
        cls,
        model: Any,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        y_pred_test: Any,
        class_labels: list[str],
        y_prob_test: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Confusion matrix (rows = actual, columns = predicted, ordered like
        `class_labels`) and, for a binary target, ROC points + AUC — all on the
        held-out split. Never fails evaluation.

        ``y_prob_test`` is the caller's already-computed `predict_proba(X_test)`
        (execute() needs it anyway for y_pred_test) — reused here instead of
        transforming/predicting X_test a second time; falls back to computing
        it when the caller didn't have one (e.g. a model without predict_proba).
        """
        from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

        out: dict[str, Any] = {}
        try:
            labels = list(range(len(class_labels))) if class_labels else sorted(pd.Series(y_test).unique())
            out["confusion_matrix"] = confusion_matrix(y_test, y_pred_test, labels=labels).tolist()
        except Exception:
            logger.debug("confusion matrix skipped", exc_info=True)
        try:
            if pd.Series(y_test).nunique() == 2 and (y_prob_test is not None or hasattr(model, "predict_proba")):
                proba = y_prob_test[:, 1] if y_prob_test is not None else model.predict_proba(X_test)[:, 1]
                positive = sorted(pd.Series(y_test).unique())[-1]
                fpr, tpr, _ = roc_curve(y_test, proba, pos_label=positive)
                if len(fpr) > cls._MAX_ROC_POINTS:
                    keep = np.linspace(0, len(fpr) - 1, cls._MAX_ROC_POINTS).astype(int)
                    fpr, tpr = fpr[keep], tpr[keep]
                out["roc_curve"] = {"fpr": [round(float(v), 4) for v in fpr],
                                    "tpr": [round(float(v), 4) for v in tpr]}
                out["roc_auc"] = round(float(roc_auc_score(y_test == positive, proba)), 4)
        except Exception:
            logger.debug("ROC curve skipped", exc_info=True)
        return out

    @staticmethod
    def _explain_drivers(
        model: Any,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        task_type: str,
        target_column: str,
        class_labels: list[str],
        precomputed_proba: np.ndarray | None = None,
    ) -> tuple[list[dict[str, Any]], list[str]]:
        """
        Model-agnostic explainability: permutation importance on the held-out
        split ranks the top drivers; how each is then explained depends on
        its shape (IMPROVEMENTS.md 7.2 item 4):

          - numeric/measure driver: direction (increases/decreases) plus the
            raw feature-target correlation, as before — just made explicit
            in the returned dict instead of living only in the ranking.
          - categorical/dimension driver: a "level effect" — the per-level
            mean of the outcome (predicted probability of the positive class
            for classification, the target itself for regression), compared
            to the overall baseline, so a driver like `contract` is reported
            as "month-to-month customers churn at 3.1x the base rate"
            instead of "a categorical feature" with no direction named.

        X_test carries raw (post-P0.1) columns, including string categoricals
        for a Pipeline-wrapped model, which is what makes the per-level
        groupby possible here without re-deriving the encoding.

        ``precomputed_proba`` is the caller's `predict_proba(X_test)` (already
        row-aligned to this X_test), reused for the baseline/outcome instead
        of transforming and predicting X_test again inside this method.

        Failure here must never fail evaluation — returns empty results instead.
        """
        try:
            from sklearn.inspection import permutation_importance

            perm = permutation_importance(
                model, X_test, y_test, n_repeats=5, random_state=42, n_jobs=1
            )
            order = perm.importances_mean.argsort()[::-1][:5]
            positive_label: str | None = None
            if len(class_labels) == 2:
                positive_label = class_labels[-1]
            elif task_type == "classification" and pd.Series(y_test).nunique() == 2:
                # Numeric binary target — name the positive class by its value
                positive_label = f"{target_column}={sorted(pd.Series(y_test).unique())[-1]}"

            # The "outcome" per-row used for level-effect / correlation
            # comparisons: predicted probability of the positive class when
            # available (classification), else the actual target — a
            # constant, reused for every driver rather than recomputed.
            outcome = pd.Series(y_test).astype(float)
            is_rate = False
            if task_type == "classification" and (precomputed_proba is not None or hasattr(model, "predict_proba")):
                try:
                    proba = (
                        precomputed_proba if precomputed_proba is not None
                        else model.predict_proba(X_test)
                    )
                    if proba.shape[1] == 2 and len(proba) == len(X_test):
                        outcome = pd.Series(proba[:, 1], index=X_test.index)
                        is_rate = True
                except Exception:
                    logger.debug("probability outcome unavailable; using point predictions", exc_info=True)
            baseline = float(outcome.mean())

            drivers: list[dict[str, Any]] = []
            narrative: list[str] = []
            for rank, idx in enumerate(order, 1):
                importance = float(perm.importances_mean[idx])
                if importance <= 0:
                    continue
                feature = str(X_test.columns[idx])
                col = X_test.iloc[:, idx]

                if pd.api.types.is_numeric_dtype(col):
                    corr = float(col.corr(pd.Series(y_test).astype(float)))
                    direction = "increases" if corr >= 0 else "decreases"
                    entry: dict[str, Any] = {
                        "feature": feature,
                        "importance": round(importance, 4),
                        "importance_std": round(float(perm.importances_std[idx]), 4),
                        "kind": "numeric",
                        "direction": direction,
                        "correlation": round(corr, 4),
                        "level_effect": None,
                    }
                    if task_type == "classification":
                        toward = f"'{positive_label}'" if positive_label else "the higher-encoded class"
                        headline = (
                            f"'{feature}' — higher values "
                            f"{'push predictions toward ' + toward if direction == 'increases' else 'push predictions away from ' + toward}"
                            f" (r={corr:.2f})"
                        )
                    else:
                        headline = (
                            f"'{feature}' — higher values {direction} "
                            f"predicted '{target_column}' (r={corr:.2f})"
                        )
                    entry["headline"] = headline
                    drivers.append(entry)
                    narrative.append(
                        f"#{rank} driver: {headline} (permutation importance {importance:.3f})."
                    )
                    continue

                # Categorical/dimension driver — per-level mean of the
                # outcome vs. the overall baseline, so the direction and
                # magnitude are named instead of collapsing to "categorical".
                level_frame = pd.DataFrame({"level": col.astype(str), "outcome": outcome})
                level_means = level_frame.groupby("level", observed=True)["outcome"].mean()
                if level_means.empty:
                    continue
                deviations = (level_means - baseline).abs()
                best_level = str(deviations.idxmax())
                level_value = float(level_means.loc[best_level])
                diff = level_value - baseline
                lift = (level_value / baseline) if abs(baseline) > 1e-9 else None
                direction_word = "up" if diff >= 0 else "down"
                level_effect = {
                    "level": best_level,
                    "level_value": round(level_value, 4),
                    "baseline": round(baseline, 4),
                    "diff": round(diff, 4),
                    "lift": round(lift, 4) if lift is not None else None,
                    "direction": direction_word,
                }
                entry = {
                    "feature": feature,
                    "importance": round(importance, 4),
                    "importance_std": round(float(perm.importances_std[idx]), 4),
                    "kind": "categorical",
                    "direction": None,
                    "level_effect": level_effect,
                }

                if is_rate:
                    if lift is not None:
                        headline = (
                            f"'{feature}' = '{best_level}': predicted {target_column} rate is "
                            f"{lift:.2f}x the baseline ({level_value:.1%} vs {baseline:.1%})"
                        )
                    else:
                        headline = (
                            f"'{feature}' = '{best_level}': predicted {target_column} rate is "
                            f"{level_value:.1%}, {abs(diff) * 100:.1f} pts {direction_word} "
                            f"vs baseline {baseline:.1%}"
                        )
                elif lift is not None:
                    headline = (
                        f"'{feature}' = '{best_level}': average {target_column} is {lift:.2f}x "
                        f"the overall baseline ({level_value:.3g} vs {baseline:.3g})"
                    )
                else:
                    headline = (
                        f"'{feature}' = '{best_level}': average {target_column} is "
                        f"{level_value:.3g}, {diff:+.3g} vs baseline {baseline:.3g}"
                    )
                entry["headline"] = headline
                drivers.append(entry)
                narrative.append(
                    f"#{rank} driver: {headline} (permutation importance {importance:.3f})."
                )
            return drivers, narrative
        except Exception:
            return [], []

    def get_schema(self) -> dict[str, Any]:
        return {
            "model_path": {"type": "string", "description": "Path to .pkl model file.", "required": True},
            "file_path": {"type": "string", "description": "Path to evaluation dataset.", "required": True},
            "target_column": {"type": "string", "description": "Target column name.", "required": True},
            "task_type": {
                "type": "string",
                "description": "classification | regression.",
                "required": False,
            },
            "test_size": {
                "type": "float",
                "description": "Held-out fraction — must match train_model. Default: 0.2.",
                "required": False,
            },
            "split_strategy": {
                "type": "string",
                "description": (
                    "'random' | 'time_series' | 'panel' — must match the train_model "
                    "call that produced model_path. Auto-filled from that step's result."
                ),
                "required": False,
            },
            "time_column": {"type": "string", "description": "Must match train_model. Auto-filled.", "required": False},
            "group_column": {"type": "string", "description": "Must match train_model. Auto-filled.", "required": False},
        }
