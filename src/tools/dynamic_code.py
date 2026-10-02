"""
Dynamic Code Execution Tool — Execution Layer.

Wraps src.core.sandbox.run_sandboxed as a BaseTool so the existing
profile-driven planner can select it like any other tool. Additive:
use only for questions no other tool in the registry answers — or to adapt
one (filter/derive/reshape, then `dsa.run` the tool on that frame).

The output/finding helpers below are shared with GeneratedTool, which runs
the same sandbox contract under a registered name.
"""
from __future__ import annotations

import dataclasses
import json
import logging
import math
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.core.chart_spec import describe_chart
from src.core.sandbox import ALLOWED_MODULES_TEXT, SandboxResult, run_sandboxed
from src.tools.base import BaseTool, ToolExecutionError, collect_prior_results

if TYPE_CHECKING:
    from src.core.findings import Finding
    from src.core.memory import DatasetMetadata, MemorySystem
    from src.core.profiler import DatasetProfile

logger = logging.getLogger(__name__)

#: Deterministic finding_id slugs — never hash(), which is process-randomized.
_SLUG_RE = re.compile(r"[^a-z0-9]+")

_RESULT_PREVIEW_CHARS = 300
_TRACEBACK_TAIL_LINES = 6


def slug(text: str) -> str:
    return _SLUG_RE.sub("_", text.lower()).strip("_") or "na"


def sandbox_failure_message(result: SandboxResult) -> str:
    """Hint plus the traceback tail, raised as ToolExecutionError so the
    controller records a failure and the LLM sees what to fix."""
    message = f"Sandboxed code failed ({result.error_type}): {result.hint or 'see traceback.'}"
    if result.traceback:
        tail = result.traceback.strip().splitlines()[-_TRACEBACK_TAIL_LINES:]
        message += "\nTraceback (last lines):\n" + "\n".join(tail)
    return message


def sandbox_output(result: SandboxResult, label: str, derived_name: str | None) -> dict[str, Any]:
    """Success output shared by execute_dynamic_code and GeneratedTool."""
    preview = json.dumps(result.result, default=str)
    if len(preview) > _RESULT_PREVIEW_CHARS:
        preview = preview[:_RESULT_PREVIEW_CHARS] + "…"
    summary = f"{label} ran in {result.duration_ms:.0f} ms. RESULT: {preview}"

    output: dict[str, Any] = {
        "status": "ok",
        "result": result.result,
        "stdout": result.stdout,
        "duration_ms": result.duration_ms,
        "backend": result.backend,
    }
    # FINDING evidence numbers must also land in `output` itself, verbatim,
    # so controller._flag_unverified_claims can verify any narrative built
    # from this finding against the tool's own output.
    if isinstance(result.finding, dict):
        output["finding_payload"] = result.finding
        evidence = result.finding.get("evidence")
        if isinstance(evidence, dict):
            output["evidence"] = dict(evidence)
    if result.chart:
        output["chart"] = result.chart
        if critique := describe_chart(result.chart):
            summary += " " + critique
    if result.result not in (None, "", [], {}):
        # Tell the LLM what will and won't reach the report, so it can act.
        if not isinstance(result.finding, dict):
            summary += " No FINDING declared — this result stays out of the report; add FINDING = {...} if it is a real discovery."
        if not result.chart and not result.chart_error and isinstance(result.result, list) and len(result.result) > 1:
            summary += " No CHART declared for this table — add CHART = dsa.chart... to show it."
    if result.chart_error:
        output["chart_error"] = result.chart_error
        summary += f" CHART was rejected: {result.chart_error}"
    if result.derived_path:
        output["derived_dataset"] = {
            "name": derived_name,
            "path": result.derived_path,
            "rows": result.derived_rows,
            "columns": result.derived_columns,
        }
        summary += (
            f" Saved derived dataset '{derived_name}' ({result.derived_rows} rows) — "
            f"pass file_path='{result.derived_path}' to analyse it with any tool."
        )
    if result.derived_error:
        output["derived_error"] = result.derived_error
        summary += f" DF_OUT not saved: {result.derived_error}"
    if result.tool_findings:
        output["tool_findings"] = result.tool_findings
    output["summary"] = summary
    return output


def _optional_fields(payload: dict[str, Any]) -> dict[str, Any]:
    """Ranking fields from a FINDING payload; the LLM often puts them inside
    `evidence` instead of at the top level, so both places are read."""
    evidence = payload.get("evidence")
    sources = (payload, evidence) if isinstance(evidence, dict) else (payload,)

    def pick(key: str) -> Any:
        return next((s[key] for s in sources if s.get(key) is not None), None)

    fields: dict[str, Any] = {}
    for key in ("measure", "dimension", "effect_kind"):
        if isinstance(pick(key), str):
            fields[key] = pick(key)
    level = pick("level")
    if isinstance(level, (str, int, float)) and not isinstance(level, bool):
        fields["level"] = str(level)
    for key in ("effect", "p_value", "confidence"):
        value = pick(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
            fields[key] = float(value)
    if "confidence" in fields:
        fields["confidence"] = min(1.0, max(0.0, fields["confidence"]))
    return fields


def sandbox_findings(output: dict[str, Any], source_tool: str, kind: str, caveat: str) -> list[Finding]:
    """FINDING payload -> one Finding; each dsa.run finding dict -> a Finding.
    Never raises: malformed LLM output means "no finding", not a crash."""
    from src.core.findings import Finding

    found: list[Finding] = []
    chart = output.get("chart") if isinstance(output.get("chart"), dict) else None
    try:
        payload = output.get("finding_payload")
        payload = payload if isinstance(payload, dict) else {}
        headline = payload.get("headline") or payload.get("summary")
        if isinstance(headline, str) and headline.strip():
            evidence = payload.get("evidence")
            found.append(Finding(
                finding_id=f"{source_tool}_{slug(headline)[:60]}",
                kind=kind,
                headline=headline.strip(),
                detail=str(payload.get("detail") or ""),
                evidence=evidence if isinstance(evidence, dict) else {},
                source_tool=source_tool,
                chart_hint=chart,
                caveats=[caveat],
                **_optional_fields(payload),
            ))
    except Exception:
        logger.debug("building a finding from the code payload failed", exc_info=True)

    field_names = {f.name for f in dataclasses.fields(Finding)}
    for item in output.get("tool_findings") or []:
        try:
            kwargs = {k: v for k, v in item.items() if k in field_names}
            base_id = kwargs.get("finding_id") or slug(str(kwargs.get("headline", "")))[:60]
            kwargs["finding_id"] = f"dsa_{base_id}"
            kwargs["caveats"] = [*(kwargs.get("caveats") or []), "Computed via dsa.run on a sandbox-derived frame"]
            found.append(Finding(**kwargs))
        except Exception:
            continue
    return found


class DynamicCodeExecutionTool(BaseTool):
    """Execute custom Python code against the dataset in an isolated subprocess."""

    requires_llm = True
    executes_code = True
    uses_cleaned_file = True
    output_subdir = "derived"

    name = "execute_dynamic_code"
    description = (
        "Workbench: run your own Python against the dataset when no tool answers "
        "the question, or to adapt a tool to this data. Pre-loaded: `df` (DataFrame), "
        "`SCHEMA` ({column: kind}), `PRIOR_RESULTS` ({tool_name: output}), and `dsa`: "
        "dsa.run(tool_name, df=frame, **params) runs a built-in analysis tool on any "
        "frame; dsa.tools(), dsa.profile(frame), dsa.compare_groups(frame, measure, by), "
        "dsa.summarize(series), dsa.effect_size(a, b); dsa.chart.bar/line/area/scatter/"
        "histogram/heatmap(data, x, y, ...) build a chart spec (heatmap: long y=col, color=value; or "
        "wide y=[cols], scale='zscore'; dsa.chart.corr_heatmap(df)). Assign RESULT (required); "
        "optionally FINDING = {'headline', 'detail', 'evidence': {...}}, CHART = dsa.chart...(...), "
        "and DF_OUT = frame with parameter save_as to keep a derived dataset. No file or "
        f"network access; imports only from {ALLOWED_MODULES_TEXT}."
    )

    def prepare_params(
        self, params: dict[str, Any], memory: MemorySystem, output_root: str
    ) -> dict[str, Any]:
        params = super().prepare_params(params, memory, output_root)
        params["prior_results"] = collect_prior_results(memory)
        return params

    def execute(  # type: ignore[override]
        self,
        file_path: str,
        code: str,
        prior_results: dict[str, Any] | None = None,
        save_as: str | None = None,
        output_dir: str | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        if not code or not code.strip():
            raise ToolExecutionError("No code was provided to execute.")
        if not Path(file_path).exists():
            raise ToolExecutionError(f"Dataset file not found: {file_path}")

        derived_name = slug(save_as) if isinstance(save_as, str) and save_as.strip() else None
        derived_dest = None
        if derived_name:
            out_dir = Path(output_dir) if output_dir else Path(file_path).parent / "derived"
            derived_dest = str(out_dir / f"{derived_name}.parquet")

        sandbox_result = run_sandboxed(
            code=code,
            dataset_ref=file_path,
            prior_results=prior_results,
            derived_dest=derived_dest,
        )
        if sandbox_result.status != "ok":
            raise ToolExecutionError(sandbox_failure_message(sandbox_result))
        return sandbox_output(sandbox_result, "Custom code", derived_name)

    def findings(
        self,
        output: dict[str, Any],
        profile: DatasetProfile | None,
        metadata: DatasetMetadata | None,
    ) -> list[Finding]:
        return sandbox_findings(output, self.name, "custom_analysis", "Computed by LLM-authored code")

    def get_schema(self) -> dict[str, Any]:
        return {
            "file_path": {
                "type": "string",
                "description": "Path to the (cleaned) dataset.",
                "required": True,
            },
            "code": {
                "type": "string",
                "description": (
                    "Python code. `df`, `SCHEMA`, `PRIOR_RESULTS` and `dsa` are pre-loaded — "
                    "never read files. Must assign RESULT at the top level; may assign "
                    "FINDING (put every number the headline states in its 'evidence'), "
                    "CHART (a dsa.chart.* spec, aggregated, <= 500 rows) and DF_OUT "
                    "(a DataFrame, saved only when save_as is given)."
                ),
                "required": True,
            },
            "save_as": {
                "type": "string",
                "description": (
                    "Name for the DF_OUT dataset. Its path is returned so later steps "
                    "can pass it as file_path to any tool."
                ),
                "required": False,
            },
        }
