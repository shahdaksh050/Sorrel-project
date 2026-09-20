"""
Standalone Validation Script — no rich/xgboost required.

Validates every stage of the 7-stage agent workflow using a mock LLM
that returns pre-scripted JSON responses.

Usage:
    python scripts/validate.py

Exit code 0 on success, 1 on failure.
"""
from __future__ import annotations

import csv
import json
import os
import sys
import tempfile
import types
from pathlib import Path
from typing import Any

# Project root on path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Windows consoles default to cp1252, which cannot print ✓/✗ — force UTF-8
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]

# Minimal console shim so src/ code that imports rich doesn't crash here
# (the real src files use rich; here we just run them in subprocess or mock them)

PASS_ICON = "✓"
FAIL_ICON = "✗"

results: list[tuple[str, str, bool]] = []


def check(stage: str, label: str, condition: bool, detail: str = "") -> None:
    results.append((stage, label, condition))
    icon = PASS_ICON if condition else FAIL_ICON
    suffix = f" ({detail})" if detail else ""
    print(f"  {icon} [{stage}] {label}{suffix}")
    if not condition:
        print(f"      FAILED: {detail or 'condition was False'}")


# ---------------------------------------------------------------------------
# Mock rich so src modules can import without error
# ---------------------------------------------------------------------------

def _make_rich_mock() -> None:
    """Inject stub rich modules so src imports don't blow up."""
    for mod in [
        "rich", "rich.console", "rich.panel", "rich.table",
        "rich.tree", "rich.progress",
    ]:
        if mod not in sys.modules:
            sys.modules[mod] = types.ModuleType(mod)

    class _Console:
        def print(self, *a: Any, **kw: Any) -> None: pass

    class _Panel:
        def __init__(self, *a: Any, **kw: Any): pass

    class _Table:
        def __init__(self, *a: Any, **kw: Any): pass
        def add_column(self, *a: Any, **kw: Any) -> None: pass
        def add_row(self, *a: Any, **kw: Any) -> None: pass

    class _Tree:
        def __init__(self, *a: Any, **kw: Any): pass
        def add(self, *a: Any, **kw: Any) -> _Tree: return self

    class _Progress:
        def __init__(self, *a: Any, **kw: Any): pass
        def __enter__(self) -> _Progress: return self
        def __exit__(self, *a: Any) -> None: pass
        def add_task(self, *a: Any, **kw: Any) -> int: return 0
        def update(self, *a: Any, **kw: Any) -> None: pass

    class _SpinnerColumn:
        def __init__(self, *a: Any, **kw: Any): pass

    class _TextColumn:
        def __init__(self, *a: Any, **kw: Any): pass

    sys.modules["rich"].Console = _Console  # type: ignore[attr-defined]
    sys.modules["rich.console"].Console = _Console  # type: ignore[attr-defined]
    sys.modules["rich.panel"].Panel = _Panel  # type: ignore[attr-defined]
    sys.modules["rich.table"].Table = _Table  # type: ignore[attr-defined]
    sys.modules["rich.tree"].Tree = _Tree  # type: ignore[attr-defined]
    sys.modules["rich.progress"].Progress = _Progress  # type: ignore[attr-defined]
    sys.modules["rich.progress"].SpinnerColumn = _SpinnerColumn  # type: ignore[attr-defined]
    sys.modules["rich.progress"].TextColumn = _TextColumn  # type: ignore[attr-defined]

_make_rich_mock()


# ---------------------------------------------------------------------------
# Mock LLM responses
# ---------------------------------------------------------------------------

class MockLLMClient:
    def __init__(self, cleaned_path: str) -> None:
        self._cleaned_path = cleaned_path
        self._call_count = 0

    def call(self, _sys: str, _usr: str) -> dict[str, Any]:
        self._call_count += 1
        if self._call_count == 1:
            return {
                "status": "in_progress",
                "reasoning": "Mock Stage 2: initial analysis plan.",
                "steps": [
                    {
                        "step_number": 1,
                        "tool_name": "clean_data",
                        "parameters": {"file_path": self._cleaned_path, "strategy": "median"},
                        "rationale": "Impute missing values.",
                    },
                    {
                        "step_number": 2,
                        "tool_name": "detect_outliers",
                        "parameters": {"file_path": self._cleaned_path, "method": "iqr"},
                        "rationale": "Flag anomalous rows.",
                    },
                    {
                        "step_number": 3,
                        "tool_name": "correlation_analysis",
                        "parameters": {"file_path": self._cleaned_path, "target_column": "label"},
                        "rationale": "Understand feature relationships.",
                    },
                ],
            }
        if self._call_count == 2:
            return {
                "status": "in_progress",
                "reasoning": "Mock Stage 4: add statistical test and training.",
                "steps": [
                    {
                        "step_number": 4,
                        "tool_name": "select_statistical_test",
                        "parameters": {
                            "file_path": self._cleaned_path,
                            "feature_column": "feature_a",
                            "group_column": "label",
                        },
                        "rationale": "Test statistical significance.",
                    },
                    {
                        "step_number": 5,
                        "tool_name": "train_model",
                        "parameters": {
                            "file_path": self._cleaned_path,
                            "target_column": "label",
                            "task_type": "classification",
                            "models": ["logistic_regression"],
                            "n_cv_folds": 3,
                            "max_depth": 4,
                        },
                        "rationale": "Train with CV and regularisation.",
                    },
                ],
            }
        return {
            "status": "complete",
            "reasoning": "Mock Stage 7: analysis complete.",
            "insights": ["feature_a correlated with label.", "No overfitting detected."],
            "recommendations": ["Deploy logistic_regression as baseline."],
            "best_model": "logistic_regression",
            "key_metrics": {"cv_mean": "0.74", "train_test_gap": "0.04"},
        }


# ---------------------------------------------------------------------------
# Sample CSV
# ---------------------------------------------------------------------------

def make_sample_csv(path: Path) -> None:
    import random
    rng = random.Random(42)
    rows = [["feature_a", "feature_b", "feature_c", "label"]]
    for i in range(80):
        fa: Any = round(rng.gauss(0, 1), 3)
        fb = round(rng.gauss(5, 2), 3)
        fc = round(rng.uniform(0, 10), 3)
        label = 1 if float(fa) + rng.gauss(0, 0.5) > 0 else 0
        if i % 15 == 0:
            fa = ""
        rows.append([fa, fb, fc, label])
    with open(path, "w", newline="") as f:
        csv.writer(f).writerows(rows)


# ---------------------------------------------------------------------------
# Main validation
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("  Agentic Data Analysis — 7-Stage Dry-Run Validation")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        csv_path = tmp / "sample.csv"
        make_sample_csv(csv_path)
        output_dir = tmp / "output"
        os.environ["OUTPUT_DIR"] = str(output_dir)
        os.environ["ENABLE_RLM_INFERENCE"] = "true"

        # ----------------------------------------------------------
        # STAGE 1 — Dataset Ingestion
        # ----------------------------------------------------------
        print("\n--- Stage 1: Dataset Ingestion ---")
        from src.tools.data_processing import IngestDatasetTool
        ingest = IngestDatasetTool().run(file_path=str(csv_path), target_column="label")
        check("S1", "Ingest returns success", ingest.status == "success")
        meta_dict = ingest.output.get("metadata", {})
        check("S1", "Row count == 80", meta_dict.get("row_count") == 80)
        check("S1", "4 columns detected", meta_dict.get("column_count") == 4)
        check("S1", "Task type inferred as classification", meta_dict.get("task_type") == "classification")
        check("S1", "Missing values detected", len(meta_dict.get("missing_values", {})) > 0)
        check("S1", "'summary' key present", "summary" in ingest.output)

        # ----------------------------------------------------------
        # STAGE 1 — Memory System
        # ----------------------------------------------------------
        print("\n--- Stage 1: Memory System ---")
        from src.core.memory import DatasetMetadata, MemorySystem
        mem = MemorySystem()
        metadata = DatasetMetadata(**meta_dict)
        mem.store_dataset_metadata(metadata)
        check("S1", "Metadata stored", mem.dataset_metadata is not None)
        prompt = mem.get_metadata_prompt()
        check("S1", "Metadata prompt is str", isinstance(prompt, str))
        check("S1", "Prompt references row count", "80" in prompt)
        check("S1", "Prompt references target column", "label" in prompt)

        # ----------------------------------------------------------
        # STAGE 3 — Tool Execution
        # ----------------------------------------------------------
        print("\n--- Stage 3: Data Cleaning ---")
        from src.tools.data_processing import CleanDataTool
        clean = CleanDataTool().run(file_path=str(csv_path), strategy="median")
        check("S3", "Clean returns success", clean.status == "success")
        check("S3", "missing_after == 0", clean.output.get("missing_after") == 0)
        cleaned_path = clean.output.get("cleaned_file_path", "")
        check("S3", "cleaned_file_path exists on disk", bool(cleaned_path) and Path(cleaned_path).exists())
        mem.append_tool_result(clean)
        mem.set_context("cleaned_file_path", cleaned_path)

        print("\n--- Stage 3: Outlier Detection ---")
        from src.tools.data_processing import DetectOutliersTool
        outlier = DetectOutliersTool().run(file_path=cleaned_path, method="iqr")
        check("S3", "Outlier detection success", outlier.status == "success")
        check("S3", "total_outliers key present", "total_outliers" in outlier.output)
        pct = outlier.output.get("outlier_percentage", -1)
        check("S3", "outlier_percentage in [0,100]", 0.0 <= pct <= 100.0)
        check("S3", "flagged_file_path present", "flagged_file_path" in outlier.output)
        mem.append_tool_result(outlier)

        print("\n--- Stage 3: Correlation Analysis ---")
        from src.tools.data_processing import CorrelationAnalysisTool
        corr = CorrelationAnalysisTool().run(file_path=cleaned_path, target_column="label")
        check("S3", "Correlation analysis success", corr.status == "success")
        check("S3", "top_correlations non-empty", len(corr.output.get("top_correlations", [])) > 0)
        check("S3", "target_correlations excludes self", "label" not in corr.output.get("target_correlations", {}))
        mem.append_tool_result(corr)

        print("\n--- Stage 3: Statistical Test ---")
        from src.tools.statistical_analysis import SelectStatisticalTestTool
        stat = SelectStatisticalTestTool().run(
            file_path=cleaned_path,
            feature_column="feature_a",
            group_column="label",
        )
        check("S3", "Statistical test success", stat.status == "success")
        p = stat.output.get("p_value", -1)
        check("S3", "p_value in [0,1]", 0.0 <= p <= 1.0)
        check("S3", "test_name non-empty", bool(stat.output.get("test_name")))
        check("S3", "interpretation non-empty", bool(stat.output.get("interpretation")))
        mem.append_tool_result(stat)

        print("\n--- Stage 3: Model Training (anti-overfitting) ---")
        from src.tools.ml_pipeline import TrainModelTool
        model_dir = str(output_dir / "models")
        train = TrainModelTool().run(
            file_path=cleaned_path,
            target_column="label",
            task_type="classification",
            models=["logistic_regression"],
            n_cv_folds=3,
            max_depth=4,
            output_dir=model_dir,
        )
        check("S3", "Training success", train.status == "success")
        lr_metrics = train.output.get("models_trained", {}).get("logistic_regression", {})
        check("S3", "cv_mean present", "cv_mean" in lr_metrics)
        check("S3", "cv_std present", "cv_std" in lr_metrics)
        check("S3", "train_test_gap present", "train_test_gap" in lr_metrics)
        check("S3", "train_metrics present", "train_metrics" in lr_metrics)
        check("S3", "test_metrics present", "test_metrics" in lr_metrics)
        gap = lr_metrics.get("train_test_gap", 1.0)
        check("S3", "train_test_gap sane (<0.5)", gap < 0.5, f"gap={gap:.3f}")
        check("S3", "best_model set", bool(train.output.get("best_model")))
        model_path = lr_metrics.get("model_path", "")
        check("S3", "model .pkl file saved", bool(model_path) and Path(model_path).exists())
        mem.append_tool_result(train)

        print("\n--- Stage 3: Model Evaluation ---")
        if model_path and Path(model_path).exists():
            from src.tools.ml_pipeline import EvaluateModelTool
            eval_r = EvaluateModelTool().run(
                model_path=model_path,
                file_path=cleaned_path,
                target_column="label",
                task_type="classification",
            )
            check("S3", "Evaluation success", eval_r.status == "success")
            check("S3", "accuracy key present", "accuracy" in eval_r.output)
            check("S3", "classification_report present", "classification_report" in eval_r.output)
            mem.append_tool_result(eval_r)

        print("\n--- Stage 3: Visualization ---")
        from src.tools.visualization import GenerateVisualizationsTool
        viz_dir = str(output_dir / "visualizations")
        viz = GenerateVisualizationsTool().run(
            file_path=cleaned_path,
            chart_type="correlation_heatmap",
            output_dir=viz_dir,
        )
        check("S3", "Visualization success", viz.status == "success")
        charts = viz.output.get("charts", [])  # the tool returns Vega-Lite specs, not image files
        check("S3", "At least one chart spec produced", len(charts) > 0)
        if charts:
            check("S3", "Chart specs carry inline data", all(isinstance(c, dict) and c.get("data") for c in charts))
        mem.append_tool_result(viz)

        # ----------------------------------------------------------
        # STAGES 2/4/5 — RLM Engine & Prompt Manager
        # ----------------------------------------------------------
        print("\n--- Stages 2/4/5: RLM Engine & Prompt Manager ---")
        from src.core.controller import ToolRegistry
        from src.core.prompt_manager import PromptManager
        from src.rlm.engine import RLMEngine

        mock_llm = MockLLMClient(cleaned_path=cleaned_path)
        registry = ToolRegistry()
        pm = PromptManager(mem, registry.get_all_descriptions())

        sys_prompt = pm.get_system_prompt()
        check("S2", "System prompt non-empty", len(sys_prompt) > 100)
        check("S2", "System prompt has tool descriptions", "tool_name" in sys_prompt)

        init_prompt = pm.get_initial_user_prompt()
        check("S2", "Initial prompt references metadata", "80" in init_prompt or "label" in init_prompt)

        iter_prompt = pm.get_iteration_user_prompt()
        check("S4", "Iteration prompt non-empty", len(iter_prompt) > 50)
        check("S4", "Iteration prompt has results summary", "Tool" in iter_prompt or "clean_data" in iter_prompt)

        engine = RLMEngine(llm_callable=mock_llm.call, system_prompt=sys_prompt, max_depth=3)
        engine.set_iteration(1)
        resp1 = engine.invoke(init_prompt, depth=0, stage="stage2")
        check("S2", "Stage 2 LLM response is dict", isinstance(resp1, dict))
        check("S2", "Stage 2 has steps list", isinstance(resp1.get("steps"), list))
        check("S2", "Stage 2 has 3 steps", len(resp1.get("steps", [])) == 3)

        engine.set_iteration(2)
        resp2 = engine.invoke(iter_prompt, depth=0, stage="stage4")
        check("S4", "Stage 4 response is dict", isinstance(resp2, dict))
        check("S5", "Stage 5 status is in_progress", resp2.get("status") == "in_progress")

        engine.set_iteration(3)
        final_prompt = pm.get_final_interpretation_prompt()
        resp3 = engine.invoke(final_prompt, depth=0, stage="stage7")
        check("S7-pre", "Stage 7 response status complete", resp3.get("status") == "complete")
        check("S7-pre", "Stage 7 has insights", len(resp3.get("insights", [])) > 0)

        # ----------------------------------------------------------
        # STAGE 6 — RLM Decomposition
        # ----------------------------------------------------------
        print("\n--- Stage 6: RLM Task Decomposition ---")
        from src.rlm.engine import RLMSubTask

        sub_tasks = [
            RLMSubTask("grp_num", "Numerical features", {"columns": ["feature_a", "feature_b"]}),
            RLMSubTask("grp_cat", "Categorical features", {"columns": ["label"]}),
        ]

        def build_prompt(task: RLMSubTask) -> str:
            return pm.get_rlm_subtask_prompt(
                task_id=task.task_id,
                description=task.description,
                context_summary=json.dumps(task.context),
            )

        engine.set_iteration(4)
        sub_results = engine.decompose_and_invoke(sub_tasks, build_prompt, depth=1)
        check("S6", "Sub-results count == 2", len(sub_results) == 2)
        check("S6", "All sub-results are dicts", all(isinstance(v, dict) for v in sub_results.values()))
        mem.set_context("rlm_sub_results", sub_results)
        check("S6", "Sub-results stored in memory", mem.get_context("rlm_sub_results") is not None)
        check("S6", "REPL env has sub-task context keys",
              any("subtask_ctx_" in k for k in engine.repl_env.variables))

        # ----------------------------------------------------------
        # STAGE 7 — Report Generation
        # ----------------------------------------------------------
        print("\n--- Stage 7: Report Generation ---")
        from src.tools.report_generator import GenerateReportTool
        report_dir = str(output_dir / "reports")
        report = GenerateReportTool().run(
            dataset_name="dry_run_sample",
            tool_results_json=json.dumps(
                [r.to_dict() for r in mem.tool_results], default=str
            ),
            llm_insights=resp3,
            output_dir=report_dir,
        )
        check("S7", "GenerateReportTool success", report.status == "success")
        md_path = Path(report.output.get("markdown_path", ""))
        check("S7", "Markdown file exists", md_path.exists())
        check("S7", "Markdown file non-empty (>200 bytes)", md_path.exists() and md_path.stat().st_size > 200)
        json_path = Path(report.output.get("json_path", ""))
        check("S7", "JSON file exists", json_path.exists())
        if md_path.exists():
            content = md_path.read_text()
            check("S7", "Report has Insights section", "## Key Insights" in content)
            check("S7", "Report has Recommendations", "## Recommendations" in content)
            check("S7", "Report has Tool Log", "## Tool Execution Log" in content)
            check("S7", "Report lists clean_data in log", "clean_data" in content)

        # ----------------------------------------------------------
        # RLM Trace
        # ----------------------------------------------------------
        print("\n--- Reasoning Trace ---")
        n_calls = len(engine.trace)
        check("Trace", "Call log has entries", n_calls > 0)
        check("Trace", "All calls recorded", n_calls >= 4, f"got {n_calls}")

        # ----------------------------------------------------------
        # Memory System integrity
        # ----------------------------------------------------------
        print("\n--- Memory Integrity ---")
        check("Mem", "tool_results list non-empty", len(mem.tool_results) > 0)
        check("Mem", "cleaned_file_path in context", bool(mem.get_context("cleaned_file_path")))
        check("Mem", "rlm_sub_results in context", mem.get_context("rlm_sub_results") is not None)

    # ----------------------------------------------------------
    # Summary
    # ----------------------------------------------------------
    passed = sum(1 for _, _, ok in results if ok)
    failed = sum(1 for _, _, ok in results if not ok)
    total = len(results)
    print()
    print("=" * 60)
    print(f"  Results: {passed}/{total} passed, {failed} failed")
    print("=" * 60)
    if failed > 0:
        print("\n  FAILED checks:")
        for stage, label, ok in results:
            if not ok:
                print(f"    {FAIL_ICON} [{stage}] {label}")
        print()
        sys.exit(1)
    else:
        print("\n  All 7 workflow stages validated successfully.")
        print("  The system is ready for a live LLM run via main.py")
        print()
        sys.exit(0)


if __name__ == "__main__":
    main()
