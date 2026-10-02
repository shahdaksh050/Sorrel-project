"""
AgentController mixin: Stage 7 — final synthesis, dashboard, HTML/JSON report and claim verification.

Split out of controller.py; `AgentController` inherits it, so the methods keep
their `self.*` state. Shared constants live in controller_common.
"""
from __future__ import annotations

import json
import logging
import os
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from rich.panel import Panel

from src.core.chart_designer import design_from_reply
from src.core.claim_verification import flag_unverified_claims, verified_number_pools
from src.core.controller_common import (
    _MIN_CHART_DESIGN_CONTEXT,
    ControllerState,
    _read_dataframe,
    console,
)
from src.core.controller_planning import PlanMixin
from src.core.dashboard import (
    ChartSpec,
    build_dashboard,
    dashboard_to_json,
    designed_chart,
    merge_designed,
)
from src.core.dependence import check_simpsons_paradox, intraclass_correlation
from src.core.findings import Finding
from src.core.profiler import DatasetProfile, profile_dataframe
from src.core.prompt_manager import (
    CHART_DESIGN_PROMPT,
)
from src.core.stats_utils import repeated_entity

logger = logging.getLogger(__name__)


class ReportMixin(ControllerState):
    """Stage 7 — final synthesis, dashboard, HTML/JSON report and claim verification."""

    def _final_synthesis(self, stage: str) -> dict[str, Any]:
        """One LLM call for the final interpretation of everything found so
        far; the deterministic synthesis when the LLM is off, capped, failing,
        or does not return a usable Form 2 reply."""
        if not self.use_llm or self._prompt_manager is None or self._rlm_engine is None:
            return self._deterministic_final()
        try:
            if self._llm_budget_exhausted():
                raise RuntimeError("LLM token cap reached")
            self.llm_client.stage = stage
            result: dict[str, Any] = PlanMixin._normalise_reply(self._rlm_engine.invoke(
                self._prompt_manager.get_final_interpretation_prompt(), depth=0, stage=stage
            ))
            if result.get("status") == "error":
                raise RuntimeError(str(result.get("error", "Unknown LLM error")))
            if result.get("status") != "complete" or not result.get("insights"):
                raise RuntimeError("final interpretation reply had no insights")
            return result
        except Exception as exc:
            self.memory.set_context("llm_error", f"{type(exc).__name__}: {exc}")
            return self._deterministic_final()

    def _generate_dashboard(self) -> None:
        """
        Build the dynamic dashboard from the final state of the analysis and
        save it as JSON next to the reports. Non-fatal on any failure.
        """
        meta = self.memory.dataset_metadata
        if meta is None:
            return
        source = str(self.memory.get_context("cleaned_file_path") or meta.file_path)
        try:
            df = _read_dataframe(source)
            profile = profile_dataframe(df, target_column=meta.target_column)
            charts = build_dashboard(
                df,
                profile,
                target_column=meta.target_column,
                task_type=meta.task_type,
                tool_results=[r.to_dict() for r in self.memory.tool_results],
                findings=[f.to_dict() for f in self.memory.ranked_findings()],
            )
            charts = self._design_charts(df, profile, charts)
            out_path = Path(self._output_dir) / "reports" / "dashboard.json"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(dashboard_to_json(charts), encoding="utf-8")
            self._last_charts = [c.to_dict() for c in charts]
            self.memory.set_context("dashboard_path", str(out_path))
            console.print(
                f"  [green]📊 Dashboard generated: {len(charts)} chart(s) → {out_path}[/]"
            )
        except Exception as exc:
            console.print(f"  [yellow]⚠ Dashboard generation failed (non-fatal): {exc}[/]")

    def _design_charts(
        self, df: pd.DataFrame, profile: DatasetProfile, charts: list[ChartSpec]
    ) -> list[ChartSpec]:
        """One low-effort LLM call that designs a few finding-driven charts and
        drops generic ones they make redundant. Any failure keeps `charts`."""
        if (
            not self.use_llm
            or self._prompt_manager is None
            or os.getenv("CHART_DESIGN", "").strip().lower() == "false"
        ):
            return charts
        try:
            findings = [f.to_dict() for f in self.memory.ranked_findings()]
            if not findings or self._llm_budget_exhausted():
                return charts
            if self.llm_client.get_context_window() < _MIN_CHART_DESIGN_CONTEXT:
                console.print("  [dim]Chart design skipped: the model's context window is too small for it.[/]")
                return charts
            self.llm_client.stage = "chart_design"
            reply = self.llm_client.call(
                CHART_DESIGN_PROMPT,
                self._prompt_manager.get_chart_design_prompt(
                    findings, [c.to_dict() for c in charts], profile
                ),
                reasoning_effort="low",
                max_tokens=3000,  # room for the recipes of as many charts as it judges worthwhile
            )
            if self._rlm_engine is not None:
                self._rlm_engine.record_usage(reply.get("_rlm_usage"))
                self.memory.set_context("llm_usage", self._rlm_engine.usage_summary())
            design = design_from_reply(df, reply)
            designed = [
                c for i, raw in enumerate(design.charts)
                if (c := designed_chart(raw, i, findings)) is not None
            ]
            return merge_designed(charts, designed, set(design.drop_ids))
        except Exception as exc:
            if self._rlm_engine is not None:  # a failed call still spent tokens
                self._rlm_engine.record_usage(getattr(exc, "rlm_usage", None))
            console.print(f"  [dim]Chart design skipped (non-fatal): {exc}[/]")
            return charts

    def _write_run_summary(self) -> None:
        """reports/summary.json: a compact record of this run for run-to-run
        comparison. Non-fatal on any failure."""
        meta = self.memory.dataset_metadata
        if meta is None:
            return
        try:
            times: dict[str, float] = {}
            for r in self.memory.tool_results:
                times[r.tool_name] = times.get(r.tool_name, 0.0) + float(r.execution_time_ms or 0.0) / 1000.0
            summary = {
                "dataset": {"name": Path(meta.file_path).stem, "rows": meta.row_count, "cols": meta.column_count},
                "objective": self.objective,
                "llm": {"enabled": bool(self.use_llm), "model": self.llm_client.model if self.use_llm else None},
                "created": datetime.now(UTC).isoformat(timespec="seconds"),
                "findings": [
                    {"finding_id": f.finding_id, "kind": f.kind, "layer": f.layer,
                     "importance": f.importance, "headline": f.headline}
                    for f in self.memory.ranked_findings()
                ],
                "charts": [{"chart_id": c.get("chart_id"), "title": c.get("title")} for c in self._last_charts],
                "tool_seconds": {k: round(v, 3) for k, v in times.items()},
            }
            out = Path(self._output_dir) / "reports" / "summary.json"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
        except Exception as exc:
            console.print(f"  [yellow]Run summary not written (non-fatal): {exc}[/]")

    def _generate_html_report(self, llm_final: dict[str, Any]) -> None:
        """
        Write the self-contained HTML report (summary + interactive dashboard).
        Non-fatal on any failure.
        """
        meta = self.memory.dataset_metadata
        if meta is None:
            return
        try:
            from src.core.html_report import build_html_report

            html_doc = build_html_report(
                dataset_name=Path(meta.file_path).stem,
                llm_insights=llm_final,
                tool_results=[r.to_dict() for r in self.memory.tool_results],
                charts=self._last_charts,
                objective=self.objective,
                profile=self.memory.get_context("data_profile"),
                read_report=self.memory.get_context("read_report"),
                coercions=self.memory.get_context("coercions"),
                plan_rationales=self.memory.get_context("plan_rationales"),
                statistical_test_pvalues=self.memory.get_context("statistical_test_pvalues"),
                unverified_claims=self.memory.get_context("unverified_claims"),
                profile_status=self.memory.get_context("profile_status"),
                degradations=self.memory.get_context("degradations"),
                findings=[f.to_dict() for f in self.memory.ranked_findings()],
                analysis_decision=self.memory.get_context("analysis_decision"),
            )
            out_path = Path(self._output_dir) / "reports" / "report.html"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(html_doc, encoding="utf-8")
            self.memory.set_context("html_report_path", str(out_path))
            console.print(f"  [green]🌐 HTML report generated → {out_path}[/]")
        except Exception as exc:
            console.print(f"  [yellow]⚠ HTML report generation failed (non-fatal): {exc}[/]")

    def _deterministic_final(self) -> dict[str, Any]:
        """
        Synthesise a final report dict, projected from the finding bus (7.1)
        rather than re-deriving a narrative from raw tool output per tool.
        `best_model`/`key_metrics` stay derived directly from train_model's
        output since they're model-internal facts, not audience-facing
        findings; every analytical claim ("X differs by Y", "Z is trending")
        comes from `MemorySystem.ranked_findings()`, which is the same list
        every other surface (both reports, the dashboard) reads — a tool
        that gains a findings() implementation reaches all of them at once
        instead of needing a hardcoded case in each narrator.
        """
        recommendations: list[str] = []
        key_metrics: dict[str, Any] = {}
        best_model = self.memory.get_context("best_model_name")

        train = self.memory.get_last_result_for("train_model")
        if train and train.status == "success":
            models_trained = train.output.get("models_trained", {})
            best = train.output.get("best_model")
            if best and best in models_trained:
                best_model = best
                metrics = models_trained[best]
                key_metrics["cv_mean"] = metrics.get("cv_mean")
                key_metrics["cv_std"] = metrics.get("cv_std")
                key_metrics["train_test_gap"] = metrics.get("train_test_gap")
            warnings = train.output.get("overfit_warnings", [])
            if warnings:
                recommendations.append(
                    "Reduce model complexity (lower max_depth) or add regularisation "
                    "to close the train-test gap."
                )

        ranked = self.memory.ranked_findings()
        insights = [f.headline for f in ranked[:12]]
        if not insights:
            insights.append("Analysis produced no tool results to synthesise.")
        # Chosen deterministic mode and a mid-run LLM failure produce the same
        # findings but are not the same event, and saying so matters: telling
        # someone to "re-run with a reachable provider" when they deliberately
        # switched the LLM off reads as a malfunction rather than the mode
        # working as asked.
        if not self.use_llm:
            reasoning = (
                "Deterministic run: the LLM was switched off, so the plan came "
                "from the dataset profile and domain inference and these "
                "findings were compiled directly from tool output."
            )
            if not recommendations:
                recommendations.append(
                    "Turn the AI narrative on to get these same findings "
                    "interpreted and prioritised in plain language."
                )
        else:
            if not recommendations:
                recommendations.append(
                    "Re-run with a reachable LLM provider for narrative interpretation "
                    "of these deterministic findings."
                )
            reasoning = (
                "Deterministic synthesis: the LLM became unreachable mid-run, so "
                "findings were compiled directly from tool outputs."
            )
            llm_error = self.memory.get_context("llm_error")
            if llm_error:
                reasoning += f" (LLM error: {llm_error})"

        if not self.use_ml:
            recommendations.append(
                "Machine learning was switched off for this run — no model was "
                "fitted. Turn it on for predictive modelling and clustering."
            )
        if self.objective:
            reasoning = f"User objective: {self.objective}\n{reasoning}"

        return {
            "status": "complete",
            "llm_fallback": not self.use_llm or bool(self.memory.get_context("llm_error")),
            "deterministic_mode": not self.use_llm,
            "ml_enabled": self.use_ml,
            "reasoning": reasoning,
            "insights": insights,
            "recommendations": recommendations,
            "best_model": best_model,
            "key_metrics": key_metrics,
            "findings": [f.to_dict() for f in ranked],
        }

    def _verified_number_pool(self) -> set[str]:
        """Every numeric literal that actually appears in accumulated tool
        results, canonicalised for verbatim-citation checking."""
        pool, _per_tool = self._verified_number_pools()
        return pool

    def _verified_number_pools(self) -> tuple[set[str], dict[str, set[str]]]:
        """Global and per-tool verified-number pools — see
        `claim_verification.verified_number_pools`."""
        return verified_number_pools(self.memory)

    def _flag_unverified_claims(self, final_result: dict[str, Any]) -> list[str]:
        """Enforce the "cite only verbatim metrics" rule — see
        `claim_verification.flag_unverified_claims`."""
        return flag_unverified_claims(self.memory, final_result)

    def _audit_dependence_structure(self, df: pd.DataFrame | None) -> None:
        """Phase 4 (FutureScope §5.3) — dependence- and design-aware
        inference, wired as a post-hoc audit over already-collected findings
        and the profiled dataframe, the same pattern the Causal Claim Guard
        just below uses. Cost is ~0 when the triggering structure is absent:
        the clustering check only runs when `repeated_entity` finds a repeat-
        measurement column, and the Simpson's-paradox check only runs
        against `segment_lift` findings that already exist — a dataset with
        no lift claim to double-check pays nothing."""
        profile = self.last_profile
        if profile is None or df is None or df.empty:
            return

        def slug(text: str) -> str:
            return re.sub(r"[^a-z0-9]+", "_", str(text).lower()).strip("_") or "x"

        # ---- Cluster/hierarchy check: rows nested in an entity (students in
        # schools, orders per customer) are not independent; naive p-values
        # overstate significance unless cluster-robust. ----
        entity_col = repeated_entity(profile, df)
        if entity_col and entity_col in df.columns:
            measure_col = next(
                (c.name for c in profile.measures() if c.name in df.columns and c.name != entity_col),
                None,
            )
            if measure_col:
                try:
                    icc_res = intraclass_correlation(df, entity_col, measure_col)
                except Exception:
                    icc_res = {}
                if icc_res.get("needs_cluster_robust"):
                    icc_finding = Finding(
                        finding_id=f"dependence_icc_{slug(entity_col)}_{slug(measure_col)}",
                        kind="method_fit",
                        headline=(
                            f"'{measure_col}' is clustered by '{entity_col}' "
                            f"(ICC={icc_res['icc']:.2f}, design effect={icc_res['deff']:.2f}) — "
                            f"treat this as {icc_res['n_eff']:.0f} effective observations, not {icc_res['n']}."
                        ),
                        detail=icc_res.get("recommendation", ""),
                        evidence=icc_res,
                        source_tool="dependence_audit",
                        measure=measure_col,
                        dimension=entity_col,
                        confidence=0.7,
                        layer="analyst",
                        caveats=["Rows are repeated measurements within an entity; standard-error estimates assuming independence are optimistic."],
                    )
                    self.memory.add_findings([icc_finding])
                    console.print(
                        f"[yellow]⚠ Clustering detected: '{measure_col}' rows are not independent "
                        f"within '{entity_col}' (ICC={icc_res['icc']:.2f}); significance tests overstate "
                        "confidence unless cluster-robust.[/]"
                    )

        # ---- Simpson's paradox: does a real segment_lift finding reverse
        # sign once stratified by another dimension? ----
        segment_findings = [f for f in self.memory.findings if f.kind == "segment_lift"]
        if not segment_findings:
            return
        candidate_dims = [
            c.name for c in profile.dimensions()
            if c.kind == "categorical" and 2 <= c.nunique <= 12 and c.name in df.columns
        ]
        if not candidate_dims:
            return

        checked_pairs: set[tuple[str, str, str]] = set()
        top_findings = sorted(segment_findings, key=lambda f: -abs(f.effect or 0.0))[:5]
        for sfinding in top_findings:
            if not sfinding.measure or not sfinding.dimension or sfinding.level is None:
                continue
            if sfinding.measure not in df.columns or sfinding.dimension not in df.columns:
                continue
            confounders = [c for c in candidate_dims if c != sfinding.dimension][:2]
            for confounder in confounders:
                key = (sfinding.measure, sfinding.dimension, confounder)
                if key in checked_pairs:
                    continue
                checked_pairs.add(key)
                level_str = str(sfinding.level)
                try:
                    sub = df[[sfinding.measure, sfinding.dimension, confounder]].copy()
                    dim_as_str = sub[sfinding.dimension].astype(str)
                    sub["_dep_group"] = dim_as_str.where(dim_as_str == level_str, other="rest")
                    res = check_simpsons_paradox(sub, sfinding.measure, "_dep_group", confounder)
                except Exception:
                    continue
                if not res.get("paradox_detected"):
                    continue
                n_level = int((df[sfinding.dimension].astype(str) == level_str).sum())
                opposing = res.get("opposing_strata_count", 0)
                valid_strata = res.get("valid_strata_count", 0)
                para_finding = Finding(
                    finding_id=f"dependence_simpsons_{slug(sfinding.measure)}_{slug(sfinding.dimension)}_{slug(confounder)}",
                    kind="method_fit",
                    headline=(
                        f"The '{sfinding.level}' vs rest difference in '{sfinding.measure}' reverses direction "
                        f"in {opposing} of {valid_strata} strata of '{confounder}' — possible confounding; "
                        "treat the unstratified effect with caution."
                    ),
                    detail=res.get("explanation", ""),
                    evidence={**res, "n_level": n_level},
                    source_tool="dependence_audit",
                    measure=sfinding.measure,
                    dimension=sfinding.dimension,
                    level=sfinding.level,
                    confidence=0.6,
                    layer="analyst",
                    caveats=["Simpson's paradox: the aggregate and stratified effects disagree in sign."],
                )
                sfinding.caveats.append(
                    f"Possible Simpson's paradox when stratified by '{confounder}' — see the dependence audit finding."
                )
                self.memory.add_findings([para_finding])
                console.print(
                    f"[yellow]⚠ Simpson's paradox: '{sfinding.level}' vs rest on '{sfinding.measure}' "
                    f"reverses when stratified by '{confounder}'.[/]"
                )
                break

    def _generate_final_report(self, llm_final: dict[str, Any]) -> None:
        """
        Stage 7: Invoke GenerateReportTool to produce the Markdown/JSON report.

        The report tool is called with the serialised tool results and the
        LLM's final interpretation so it can produce a complete document.
        """
        meta = self.memory.dataset_metadata
        dataset_name = Path(meta.file_path).stem if meta else "dataset"

        tool_results_json = json.dumps(
            [r.to_dict() for r in self.memory.tool_results], default=str
        )

        report_tool = self.tool_registry.get("generate_report")
        result = report_tool.run(
            dataset_name=dataset_name,
            tool_results_json=tool_results_json,
            llm_insights=llm_final,
            output_dir=str(Path(self._output_dir) / "reports"),
            data_profile=self.memory.get_context("data_profile"),
            read_report=self.memory.get_context("read_report"),
            coercions=self.memory.get_context("coercions"),
            plan_rationales=self.memory.get_context("plan_rationales"),
            statistical_test_pvalues=self.memory.get_context("statistical_test_pvalues"),
            unverified_claims=self.memory.get_context("unverified_claims"),
            profile_status=self.memory.get_context("profile_status"),
            degradations=self.memory.get_context("degradations"),
            findings=[f.to_dict() for f in self.memory.ranked_findings()],
            analysis_decision=self.memory.get_context("analysis_decision"),
        )

        self.memory.append_tool_result(result)

        if result.status == "success":
            console.print(
                Panel(
                    f"[bold green]Stage 7 — Report Generated[/]\n"
                    f"Markdown: {result.output.get('markdown_path')}\n"
                    f"JSON:     {result.output.get('json_path')}",
                    border_style="green",
                )
            )
        else:
            console.print(
                f"[yellow]⚠ Report generation failed: {result.error_message}[/]"
            )
