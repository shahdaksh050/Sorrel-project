"""
Tool Registry — discovers and maps all execution-layer tools.

Split out of ``src/core/controller.py``; re-exported from there, so
``from src.core.controller import ToolRegistry`` keeps working.
"""
from __future__ import annotations

import re
from typing import Any

from src.core.governance import code_execution_available

#: Parameters the controller/tool fills in itself (BaseTool.prepare_params) —
#: never required from the planner and never "unknown".
_INJECTED_PARAMS = frozenset({"file_path", "output_dir", "prior_results"})

#: Compact tool block (8k-context models): at most this many tools, one line of
#: at most this many characters (~70 tokens) each; the workbench tools are never cut.
_COMPACT_MAX_TOOLS = 10
_COMPACT_LINE_CHARS = 240
_COMPACT_MAX_OPTIONAL = 8
_ALWAYS_LISTED = frozenset({"clean_data", "execute_dynamic_code"})


class ToolRegistry:
    """
    Registry of all available execution-layer tools.

    The LLM references tools by name; this class resolves them to
    callable BaseTool instances. Tools register themselves generically
    (register()) rather than the registry hardcoding an exhaustive import
    list — new tools (built-in or, in future, generated) plug in the same
    way the built-ins do.
    """

    def __init__(self) -> None:
        self._registry: dict[str, Any] = {}
        self._register_builtin_tools()

    def _register_builtin_tools(self) -> None:
        from src.tools.anomaly import AnomalyAnalysisTool
        from src.tools.basket import BasketAnalysisTool
        from src.tools.change_analysis import ChangeAnalysisTool
        from src.tools.clustering import ClusterDataTool
        from src.tools.cohort_analysis import CohortAnalysisTool
        from src.tools.concentration_analysis import ConcentrationAnalysisTool
        from src.tools.curve_fit import CurveFitAnalysisTool
        from src.tools.data_processing import (
            CleanDataTool,
            CorrelationAnalysisTool,
            DetectOutliersTool,
            IngestDatasetTool,
        )
        from src.tools.define_analysis_tool import DefineAnalysisToolTool
        from src.tools.dimensionality import DimensionalityAnalysisTool
        from src.tools.dynamic_code import DynamicCodeExecutionTool
        from src.tools.elasticity import PriceElasticityTool
        from src.tools.equity import EquityAnalysisTool
        from src.tools.experiment_analysis import ExperimentAnalysisTool
        from src.tools.financial_analysis import FinancialAnalysisTool
        from src.tools.forecast import ForecastAnalysisTool
        from src.tools.geospatial import GeospatialAnalysisTool
        from src.tools.graph_analysis import GraphAnalysisTool
        from src.tools.mixed_model import MixedModelAnalysisTool
        from src.tools.ml_pipeline import EvaluateModelTool, TrainModelTool
        from src.tools.regression import RegressionAnalysisTool
        from src.tools.report_generator import GenerateReportTool
        from src.tools.segment_comparison import SegmentComparisonTool
        from src.tools.statistical_analysis import SelectStatisticalTestTool
        from src.tools.survival import SurvivalAnalysisTool
        from src.tools.text_analysis import TextAnalysisTool
        from src.tools.time_series import TimeSeriesAnalysisTool
        from src.tools.variable_methods import VariableScaleAnalysisTool
        from src.tools.visualization import GenerateVisualizationsTool
        from src.tools.workforce_analysis import WorkforceAnalysisTool

        for tool in (
            IngestDatasetTool(),
            CleanDataTool(),
            DetectOutliersTool(),
            CorrelationAnalysisTool(),
            SelectStatisticalTestTool(),
            TrainModelTool(),
            EvaluateModelTool(),
            ClusterDataTool(),
            GenerateVisualizationsTool(),
            GenerateReportTool(),
            TimeSeriesAnalysisTool(),
            TextAnalysisTool(),
            DimensionalityAnalysisTool(),
            GeospatialAnalysisTool(),
            DynamicCodeExecutionTool(),
            # Round 8 — LLM Sandbox mode: define_analysis_tool lets the agent
            # register a new, named, reusable tool at runtime (registration
            # itself happens in _maybe_register_generated_tool, called from
            # _execute_steps after this tool's own pure validate/smoke-test
            # step succeeds — see AGENTS.md's layer rule).
            DefineAnalysisToolTool(),
            FinancialAnalysisTool(),
            CohortAnalysisTool(),
            WorkforceAnalysisTool(),
            # 7.2 — insight library (segment comparison, concentration,
            # period-over-period change): the "why"/"what happened"
            # questions no prior tool answered directly.
            SegmentComparisonTool(),
            ConcentrationAnalysisTool(),
            ChangeAnalysisTool(),
            # Drivers (adjusted regression), experiments (A/B lift, power,
            # SRM) and anomalies (spikes, level shifts, unusual segments).
            RegressionAnalysisTool(),
            ExperimentAnalysisTool(),
            AnomalyAnalysisTool(),
            # Domain tools — each gates itself in applies_to (score 0 = never
            # offered), so they cost the planner prompt nothing on data they
            # don't fit: time-to-event, forecasting, nested/replicated data,
            # retail baskets and price response, pay/outcome equity.
            SurvivalAnalysisTool(),
            CurveFitAnalysisTool(),
            ForecastAnalysisTool(),
            MixedModelAnalysisTool(),
            BasketAnalysisTool(),
            PriceElasticityTool(),
            EquityAnalysisTool(),
            # FutureScope Phase 3: Variable-scale methods
            VariableScaleAnalysisTool(),
            # FutureScope Phase 5: Graph analysis
            GraphAnalysisTool(),
        ):
            self.register(tool)

    def register(self, tool: Any) -> None:
        """Add (or replace) a tool in the registry, keyed by its `name`."""
        self._registry[tool.name] = tool

    def get(self, name: str) -> Any:
        if name not in self._registry:
            raise KeyError(
                f"Unknown tool '{name}'. Available: {list(self._registry.keys())}"
            )
        return self._registry[name]

    def has(self, name: str) -> bool:
        return name in self._registry

    def names(self) -> list[str]:
        return list(self._registry.keys())

    def get_all_descriptions(self) -> str:
        """Descriptions for every registered tool, gating aside. Used by
        offline scripts (validate/dry_run) that have no DatasetProfile."""
        return "\n\n".join(t.to_prompt_description() for t in self._registry.values())

    def candidate_tools(
        self,
        profile: Any | None,
        metadata: Any | None,
        use_ml: bool = True,
        use_llm: bool = True,
    ) -> list[Any]:
        """
        Tools relevant to this dataset, ranked by applies_to() score
        (highest first). A tool scoring 0.0 is excluded entirely — this
        IS the dynamic-selection mechanism: what the planner sees is
        already filtered to what fits the data's nature.

        `use_ml`/`use_llm` drop the tools that declare they need those
        capabilities (BaseTool.requires_ml / requires_llm). Filtering here
        rather than at plan time means a disabled capability is invisible
        everywhere at once: the planner never sees the tool, the
        deterministic plan never schedules it, and the prompt never
        describes it.
        """
        scored = [(t, t.applies_to(profile, metadata)) for t in self._registry.values()]
        relevant = [
            (t, s)
            for t, s in scored
            if s > 0.0
            and not (getattr(t, "requires_ml", False) and not use_ml)
            and not (getattr(t, "requires_llm", False) and not use_llm)
            and not (getattr(t, "executes_code", False) and not code_execution_available())
        ]
        relevant.sort(key=lambda ts: ts[1], reverse=True)
        return [t for t, _ in relevant]

    def get_candidate_descriptions(
        self,
        profile: Any | None,
        metadata: Any | None,
        use_ml: bool = True,
        use_llm: bool = True,
        compact: bool = False,
    ) -> str:
        """`compact` (8k-context models): one `name(params) — sentence` line for
        each of the top-scoring candidates, the workbench tools always kept."""
        tools = self.candidate_tools(profile, metadata, use_ml=use_ml, use_llm=use_llm)
        if compact:
            tools = tools or list(self._registry.values())
            pinned = [t.name for t in tools if t.name in _ALWAYS_LISTED]
            others = [t.name for t in tools if t.name not in _ALWAYS_LISTED]
            keep = {*pinned, *others[: _COMPACT_MAX_TOOLS - len(pinned)]}
            lines = [_compact_tool_description(t) for t in tools if t.name in keep]
            return "\n".join([*lines, "(other tools available via dsa.tools())"])
        if not tools:
            return self.get_all_descriptions()
        return "\n\n".join(t.to_prompt_description() for t in tools)

    def get_candidate_short_descriptions(
        self,
        profile: Any | None,
        metadata: Any | None,
        use_ml: bool = True,
        use_llm: bool = True,
    ) -> str:
        """One line per candidate tool — name, first sentence, required
        params — for context-limited models (PromptManager picks it)."""
        tools = self.candidate_tools(profile, metadata, use_ml=use_ml, use_llm=use_llm) or list(
            self._registry.values()
        )
        return "\n".join(_short_tool_description(t) for t in tools)


def _first_sentence(tool: Any, limit: int) -> str:
    description = " ".join(str(getattr(tool, "description", "")).split())
    return re.split(r"(?<=[.!?])\s", description, maxsplit=1)[0][:limit]


def _tool_params(tool: Any) -> tuple[list[str], list[str]]:
    """(required, optional) planner-facing parameter names."""
    try:
        schema = tool.get_schema() or {}
    except Exception:
        schema = {}
    required = [
        k for k, v in schema.items()
        if isinstance(v, dict) and v.get("required") and k not in _INJECTED_PARAMS
    ]
    optional = [k for k in schema if k not in required and k not in _INJECTED_PARAMS]
    return required, optional


def _compact_tool_description(tool: Any) -> str:
    """`name(required; optional: a,b) — first sentence`, about 70 tokens at most."""
    required, optional = _tool_params(tool)
    args = ", ".join(required)
    if optional:
        args += ("; " if args else "") + "optional: " + ",".join(optional[:_COMPACT_MAX_OPTIONAL])
    head = f"{tool.name}({args}) — "
    return head + _first_sentence(tool, max(40, _COMPACT_LINE_CHARS - len(head)))


def _short_tool_description(tool: Any) -> str:
    first = _first_sentence(tool, 200)
    required, optional = _tool_params(tool)
    line = f"- {tool.name}: {first}"
    if required:
        line += f" Required: {', '.join(required)}."
    if optional:
        line += f" Optional: {', '.join(optional)}."
    return line
