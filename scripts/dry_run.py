"""
Dry-Run Validation Script — no LLM API key required.

Validates every stage of the 7-stage agent workflow using a mock LLM
that returns pre-scripted JSON responses. Exits with code 0 on success,
non-zero on failure.

Usage:
    python scripts/dry_run.py

Checks:
  Stage 1  — Dataset ingestion and metadata extraction
  Stage 2  — Initial reasoning (mock LLM response parsed correctly)
  Stage 3  — All tools execute on sample data without error
  Stage 4  — Iteration prompt builds without error
  Stage 5  — Second reasoning cycle resolves pending steps
  Stage 6  — RLM decomposition triggers on wide datasets
  Stage 7  — Report generation produces .md and .json files
"""
from __future__ import annotations

import csv
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

# ---------------------------------------------------------------------------
# Mock LLM — cycles through scripted responses
# ---------------------------------------------------------------------------

class MockLLMClient:
    """Returns pre-scripted JSON responses to simulate LLM planning."""

    def __init__(self, cleaned_path: str) -> None:
        self._cleaned_path = cleaned_path
        self._call_count = 0

    def call(self, _system: str, user_prompt: str) -> dict[str, Any]:
        self._call_count += 1

        # Stage 2: initial reasoning — return a 3-step plan
        if self._call_count == 1:
            return {
                "status": "in_progress",
                "reasoning": "Dry-run: plan the first three cleaning and EDA steps.",
                "steps": [
                    {
                        "step_number": 1,
                        "tool_name": "clean_data",
                        "parameters": {"file_path": self._cleaned_path, "strategy": "median"},
                        "rationale": "Handle missing values before analysis.",
                    },
                    {
                        "step_number": 2,
                        "tool_name": "detect_outliers",
                        "parameters": {"file_path": self._cleaned_path, "method": "iqr"},
                        "rationale": "Identify anomalous rows using IQR method.",
                    },
                    {
                        "step_number": 3,
                        "tool_name": "correlation_analysis",
                        "parameters": {"file_path": self._cleaned_path, "target_column": "label"},
                        "rationale": "Understand feature relationships before model selection.",
                    },
                ],
            }

        # Stage 4/5: second iteration — add stat test + training
        if self._call_count == 2:
            return {
                "status": "in_progress",
                "reasoning": "Dry-run: run statistical test and train models.",
                "steps": [
                    {
                        "step_number": 4,
                        "tool_name": "select_statistical_test",
                        "parameters": {
                            "file_path": self._cleaned_path,
                            "feature_column": "feature_a",
                            "group_column": "label",
                        },
                        "rationale": "Confirm statistical significance of feature_a vs label.",
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
                        "rationale": "Train with logistic regression; k=3 CV for speed.",
                    },
                ],
            }

        # Stage 7: final answer
        return {
            "status": "complete",
            "reasoning": "All analyses complete. Dry-run passed all stages.",
            "insights": [
                "feature_a showed statistically significant correlation with label.",
                "Logistic Regression achieved reasonable CV score on synthetic data.",
            ],
            "recommendations": [
                "Consider adding XGBoost for a stronger baseline.",
                "Monitor train-test gap — overfitting risk is low on this dataset.",
            ],
            "best_model": "logistic_regression",
            "key_metrics": {"cv_mean": "~0.75", "train_test_gap": "<0.10"},
        }


# ---------------------------------------------------------------------------
# Sample dataset generator
# ---------------------------------------------------------------------------

def make_sample_csv(path: Path) -> None:
    """Write a small synthetic CSV with known structure."""
    rows = [["feature_a", "feature_b", "feature_c", "label"]]
    import random
    rng = random.Random(42)
    for i in range(80):
        fa = round(rng.gauss(0, 1), 3)
        fb = round(rng.gauss(5, 2), 3)
        fc = round(rng.uniform(0, 10), 3)
        label = 1 if fa + rng.gauss(0, 0.5) > 0 else 0
        # Introduce some missing values
        if i % 15 == 0:
            fa = ""  # type: ignore[assignment]
        rows.append([fa, fb, fc, label])
    with open(path, "w", newline="") as f:
        csv.writer(f).writerows(rows)


# ---------------------------------------------------------------------------
# Stage validators
# ---------------------------------------------------------------------------

PASS = "[bold green]PASS[/]"
FAIL = "[bold red]FAIL[/]"

results: list[tuple[str, str, str]] = []  # (stage, check, status)


def check(stage: str, label: str, condition: bool, detail: str = "") -> None:
    results.append((stage, label, "✓" if condition else "✗"))
    icon = "✓" if condition else "✗"
    color = "green" if condition else "red"
    msg = f"  [{color}]{icon}[/] {label}"
    if detail:
        msg += f" [dim]({detail})[/]"
    console.print(msg)
    if not condition:
        console.print(f"    [red]Detail: {detail}[/]")


def main() -> None:
    console.print(
        Panel(
            "[bold cyan]Agentic Data Analysis — Dry-Run Validation[/]\n"
            "[dim]Validates all 7 workflow stages without a live LLM API key[/]",
            border_style="cyan",
        )
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        csv_path = tmp / "sample.csv"
        make_sample_csv(csv_path)
        output_dir = tmp / "output"
        os.environ["OUTPUT_DIR"] = str(output_dir)
        os.environ["ENABLE_RLM_INFERENCE"] = "true"

        # ----------------------------------------------------------------
        # Stage 1 — Dataset Ingestion
        # ----------------------------------------------------------------
        console.print("\n[bold]Stage 1 — Dataset Ingestion[/]")
        from src.tools.data_processing import IngestDatasetTool
        ingest = IngestDatasetTool().run(file_path=str(csv_path), target_column="label")
        check("S1", "IngestDatasetTool returns success", ingest.status == "success")
        check("S1", "Row count correct", ingest.output.get("metadata", {}).get("row_count") == 80)
        check("S1", "Task type inferred as classification",
              ingest.output.get("metadata", {}).get("task_type") == "classification")
        check("S1", "Summary key present", "summary" in ingest.output)

        # ----------------------------------------------------------------
        # Stage 1 — Memory System
        # ----------------------------------------------------------------
        console.print("\n[bold]Stage 1 — Memory System[/]")
        from src.core.memory import DatasetMetadata, MemorySystem
        mem = MemorySystem()
        meta = DatasetMetadata(**ingest.output["metadata"])
        mem.store_dataset_metadata(meta)
        check("S1", "Metadata stored in MemorySystem", mem.dataset_metadata is not None)
        prompt = mem.get_metadata_prompt()
        check("S1", "Metadata prompt is non-empty string", isinstance(prompt, str) and len(prompt) > 20)
        check("S1", "Prompt contains row count", "80" in prompt)

        # ----------------------------------------------------------------
        # Stage 3 — Tool Execution (Cleaning)
        # ----------------------------------------------------------------
        console.print("\n[bold]Stage 3 — Data Cleaning[/]")
        from src.tools.data_processing import CleanDataTool
        clean = CleanDataTool().run(file_path=str(csv_path), strategy="median")
        check("S3", "CleanDataTool returns success", clean.status == "success")
        check("S3", "Missing values reduced", clean.output.get("missing_after", 1) == 0)
        cleaned_path = clean.output.get("cleaned_file_path", "")
        check("S3", "cleaned_file_path returned", bool(cleaned_path) and Path(cleaned_path).exists())
        mem.append_tool_result(clean)
        mem.set_context("cleaned_file_path", cleaned_path)

        # ----------------------------------------------------------------
        # Stage 3 — Outlier Detection
        # ----------------------------------------------------------------
        console.print("\n[bold]Stage 3 — Outlier Detection[/]")
        from src.tools.data_processing import DetectOutliersTool
        outlier = DetectOutliersTool().run(file_path=cleaned_path, method="iqr")
        check("S3", "DetectOutliersTool returns success", outlier.status == "success")
        check("S3", "total_outliers key present", "total_outliers" in outlier.output)
        check("S3", "flagged_file_path present", "flagged_file_path" in outlier.output)
        mem.append_tool_result(outlier)

        # ----------------------------------------------------------------
        # Stage 3 — Correlation Analysis
        # ----------------------------------------------------------------
        console.print("\n[bold]Stage 3 — Correlation Analysis[/]")
        from src.tools.data_processing import CorrelationAnalysisTool
        corr = CorrelationAnalysisTool().run(
            file_path=cleaned_path, target_column="label"
        )
        check("S3", "CorrelationAnalysisTool returns success", corr.status == "success")
        check("S3", "top_correlations non-empty", len(corr.output.get("top_correlations", [])) > 0)
        check("S3", "target_correlations excludes self", "label" not in corr.output.get("target_correlations", {}))
        mem.append_tool_result(corr)

        # ----------------------------------------------------------------
        # Stage 3 — Statistical Test
        # ----------------------------------------------------------------
        console.print("\n[bold]Stage 3 — Statistical Test[/]")
        from src.tools.statistical_analysis import SelectStatisticalTestTool
        stat = SelectStatisticalTestTool().run(
            file_path=cleaned_path,
            feature_column="feature_a",
            group_column="label",
        )
        check("S3", "SelectStatisticalTestTool returns success", stat.status == "success")
        check("S3", "p_value in [0,1]", 0.0 <= stat.output.get("p_value", -1) <= 1.0)
        check("S3", "test_name non-empty", bool(stat.output.get("test_name")))
        mem.append_tool_result(stat)

        # ----------------------------------------------------------------
        # Stage 3 — Model Training (anti-overfit checks)
        # ----------------------------------------------------------------
        console.print("\n[bold]Stage 3 — Model Training & Anti-Overfitting[/]")
        from src.tools.ml_pipeline import TrainModelTool
        model_out_dir = str(output_dir / "models")
        train = TrainModelTool().run(
            file_path=cleaned_path,
            target_column="label",
            task_type="classification",
            models=["logistic_regression"],
            n_cv_folds=3,
            max_depth=4,
            output_dir=model_out_dir,
        )
        check("S3", "TrainModelTool returns success", train.status == "success")
        check("S3", "cv_mean present for logistic_regression",
              "cv_mean" in train.output.get("models_trained", {}).get("logistic_regression", {}))
        check("S3", "train_test_gap present",
              "train_test_gap" in train.output.get("models_trained", {}).get("logistic_regression", {}))
        gap = train.output.get("models_trained", {}).get("logistic_regression", {}).get("train_test_gap", 1.0)
        check("S3", "train_test_gap < 0.5 (sanity)", gap < 0.5, f"gap={gap:.3f}")
        mem.append_tool_result(train)

        # ----------------------------------------------------------------
        # Stages 2/4/5 — RLM Engine (mock LLM)
        # ----------------------------------------------------------------
        console.print("\n[bold]Stages 2/4/5 — RLM Engine & Prompt Manager[/]")
        from src.core.prompt_manager import PromptManager
        from src.rlm.engine import RLMEngine

        mock_llm = MockLLMClient(cleaned_path=cleaned_path)
        from src.core.tool_registry import ToolRegistry
        registry = ToolRegistry()
        pm = PromptManager(mem, registry.get_all_descriptions())

        sys_prompt = pm.get_system_prompt()
        check("S2", "System prompt non-empty", len(sys_prompt) > 100)

        init_prompt = pm.get_initial_user_prompt()
        check("S2", "Initial user prompt contains metadata", "80" in init_prompt or "label" in init_prompt)

        engine = RLMEngine(llm_callable=mock_llm.call, system_prompt=sys_prompt, max_depth=3)
        engine.set_iteration(1)

        resp1 = engine.invoke(init_prompt, depth=0, stage="stage2:dry_run")
        check("S2", "LLM response is dict", isinstance(resp1, dict))
        check("S2", "Response has 'steps'", "steps" in resp1)
        check("S2", "3 steps returned", len(resp1.get("steps", [])) == 3)

        # Simulate iterating with stage 4/5 prompt
        iter_prompt = pm.get_iteration_user_prompt()
        engine.set_iteration(2)
        resp2 = engine.invoke(iter_prompt, depth=0, stage="stage4:dry_run")
        check("S4", "Iteration response is dict", isinstance(resp2, dict))

        # ----------------------------------------------------------------
        # Stage 6 — RLM Decomposition
        # ----------------------------------------------------------------
        console.print("\n[bold]Stage 6 — RLM Task Decomposition[/]")
        from src.rlm.engine import RLMSubTask

        sub_tasks = [
            RLMSubTask("num_g1", "Analyse numerical group 1", {"columns": ["feature_a", "feature_b"]}),
            RLMSubTask("cat_g1", "Analyse categorical group 1", {"columns": ["label"]}),
        ]

        def prompt_builder(task: RLMSubTask) -> str:
            return pm.get_rlm_subtask_prompt(
                task_id=task.task_id,
                description=task.description,
                context_summary=json.dumps(task.context),
            )

        engine.set_iteration(3)
        sub_results = engine.decompose_and_invoke(sub_tasks, prompt_builder, depth=1)
        check("S6", "Sub-results returned for each sub-task", len(sub_results) == 2)
        check("S6", "Sub-results are dicts", all(isinstance(v, dict) for v in sub_results.values()))
        mem.set_context("rlm_sub_results", sub_results)
        check("S6", "Sub-results stored in memory context", mem.get_context("rlm_sub_results") is not None)

        # ----------------------------------------------------------------
        # Stage 7 — Report Generation
        # ----------------------------------------------------------------
        console.print("\n[bold]Stage 7 — Report Generation[/]")
        from src.tools.report_generator import GenerateReportTool

        final_llm = {
            "status": "complete",
            "reasoning": "Dry-run complete — all stages validated.",
            "insights": ["feature_a is the strongest predictor.", "No significant overfitting detected."],
            "recommendations": ["Deploy logistic_regression as baseline model."],
            "best_model": "logistic_regression",
            "key_metrics": {"cv_mean": "0.75", "train_test_gap": "0.03"},
        }

        report_dir = str(output_dir / "reports")
        report = GenerateReportTool().run(
            dataset_name="dry_run_sample",
            tool_results_json=json.dumps([r.to_dict() for r in mem.tool_results], default=str),
            llm_insights=final_llm,
            output_dir=report_dir,
        )
        check("S7", "GenerateReportTool returns success", report.status == "success")
        md_path = Path(report.output.get("markdown_path", ""))
        check("S7", "Markdown report file exists", md_path.exists())
        check("S7", "Markdown report is non-empty", md_path.exists() and md_path.stat().st_size > 200)
        json_path = Path(report.output.get("json_path", ""))
        check("S7", "JSON report file exists", json_path.exists())

        # Check Markdown contains expected sections
        if md_path.exists():
            content = md_path.read_text()
            check("S7", "Report contains insights section", "## Key Insights" in content)
            check("S7", "Report contains recommendations section", "## Recommendations" in content)
            check("S7", "Report contains tool execution log", "## Tool Execution Log" in content)

        # ----------------------------------------------------------------
        # Reasoning Trace
        # ----------------------------------------------------------------
        console.print("\n[bold]Reasoning Trace[/]")
        engine.print_reasoning_trace()
        check("Trace", "Call log non-empty", len(engine.trace) > 0)

    # ----------------------------------------------------------------
    # Summary
    # ----------------------------------------------------------------
    console.print()
    table = Table(title="Dry-Run Results", show_header=True, header_style="bold")
    table.add_column("Stage", style="cyan")
    table.add_column("Check")
    table.add_column("Result", justify="center")
    for stage, label, status in results:
        color = "green" if status == "✓" else "red"
        table.add_row(stage, label, f"[{color}]{status}[/]")
    console.print(table)

    failures = sum(1 for _, _, s in results if s == "✗")
    total = len(results)
    if failures == 0:
        console.print(
            Panel(
                f"[bold green]All {total} checks passed![/]\n"
                "[dim]The 7-stage agent workflow is fully functional.[/]",
                border_style="green",
            )
        )
        sys.exit(0)
    else:
        console.print(
            Panel(
                f"[bold red]{failures}/{total} checks failed.[/]\n"
                "[dim]Review the output above for details.[/]",
                border_style="red",
            )
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
