"""
AgentController mixin: Stage 2 — parsing the planner's reply and building the deterministic fallback plan.

Split out of controller.py; `AgentController` inherits it, so the methods keep
their `self.*` state. Shared constants live in controller_common.
"""
from __future__ import annotations

import difflib
import json
import logging
import re
from typing import Any

from src.core.controller_common import (
    _MAX_STEPS_PER_CYCLE,
    ControllerState,
    _step_signature,
    console,
)
from src.core.governance import (
    code_execution_available,
)
from src.core.memory import AnalysisStep, ToolResult
from src.core.profiler import DatasetProfile
from src.core.prompt_manager import ARCHETYPE_CANONICAL_TOOLS
from src.core.question_router import comparison_objective

logger = logging.getLogger(__name__)


class PlanMixin(ControllerState):
    """Stage 2 — parsing the planner's reply and building the deterministic fallback plan."""

    _PLAN_KEYS = ("steps", "plan", "next_steps", "actions", "tasks")

    _TOOL_KEYS = ("tool_name", "tool", "name", "function", "action", "tool_call")

    _PARAM_KEYS = ("parameters", "params", "arguments", "args", "input", "inputs")

    _STEP_NO_KEYS = ("step_number", "step", "id", "index")

    _TOOL_PREFIX_RE = re.compile(r"^(?:(?:functions?|tools?)\s*[.:/]\s*)+", re.IGNORECASE)

    _DONE_STATUSES = frozenset({"complete", "completed", "done", "finished", "final", "finish"})

    _GOING_STATUSES = frozenset({"in_progress", "inprogress", "continue", "continuing", "running", "ongoing", "working"})

    _NUMERIC_ONLY_TOOLS = frozenset({
        "correlation_analysis", "statistical_analysis", "select_statistical_test",
        "cluster_data", "regression_analysis", "train_model", "evaluate_model",
        "scenario_simulation", "detect_outliers",
    })
    _OBJECTIVE_REQUIRED_TOOLS = frozenset({"scenario_simulation", "cluster_data"})
    _MAX_STEPS_PER_TOOL = 1

    def _numeric_column_count(self) -> int:
        """Count actual numeric features, excluding a target when possible."""
        profile = self.last_profile
        if profile is not None:
            count = len(profile.columns_of_kind("numeric"))
            if count > 0:
                return count
        meta = self.memory.dataset_metadata
        return len(meta.numerical_cols) if meta is not None else 0

    def _objective_allows_tool(self, tool_name: str) -> bool:
        """Whether an optional exploratory tool has a decision-relevant ask.

        Scenario simulation and clustering are not generic cleanup actions:
        applicability only tells us that they *can* run.  They consume budget
        only when the objective actually requests a decision/sensitivity or a
        segmentation question.
        """
        if tool_name not in self._OBJECTIVE_REQUIRED_TOOLS:
            return True
        objective = str(getattr(self, "objective", "") or "")
        auto_obj = str(getattr(self, "autonomous_objective", "") or "")
        combined = f"{objective} {auto_obj}".strip()
        target = getattr(self.memory.dataset_metadata, "target_column", None)
        if tool_name == "cluster_data" and target is None:
            return True
        if tool_name == "scenario_simulation":
            has_driver = any(f.kind == "driver" and f.dimension and f.measure for f in self.memory.findings)
            if has_driver:
                return True
            pattern = r"\b(?:what[ -]?if|scenario|sensitivity|interven|lever|optim|decision|reduce|increase|improve)\b"
            return bool(re.search(pattern, combined, re.IGNORECASE))
        return bool(re.search(r"\b(?:segment|cluster|cohort|persona|archetype)\b", combined, re.IGNORECASE))

    def _tool_hard_prerequisite_error(self, tool_name: str) -> str | None:
        """Return a deterministic reason a proposed step must not be planned."""
        if self.last_profile is None and self.memory.dataset_metadata is None:
            return None
        numeric = self._numeric_column_count()
        if tool_name in self._NUMERIC_ONLY_TOOLS and numeric == 0:
            return f"{tool_name} requires numeric columns; this dataset has none."
        if tool_name == "cluster_data":
            if numeric < 2:
                return "cluster_data requires at least two numeric columns."
            rows = self.last_profile.row_count if self.last_profile else (
                self.memory.dataset_metadata.row_count if self.memory.dataset_metadata else 0
            )
            if rows < 30:
                return "cluster_data requires at least 30 rows."
        if tool_name == "segment_comparison":
            cat_count = len(self.last_profile.columns_of_kind("categorical")) if self.last_profile else (
                len(self.memory.dataset_metadata.categorical_cols) if self.memory.dataset_metadata else 0
            )
            if cat_count == 0:
                meta = self.memory.dataset_metadata
                has_cat_derived = bool(
                    meta and (
                        any(c in meta.columns for c in ("day_part", "day_of_week", "hour_of_day"))
                        or any(str(c).endswith("_quartile") for c in meta.columns)
                    )
                )
                has_time_or_numeric = bool(
                    (self.last_profile and (self.last_profile.datetime_cols or len(self.last_profile.columns_of_kind("numeric")) >= 2))
                    or (meta and len(meta.numerical_cols) >= 2)
                )
                if not has_cat_derived and not has_time_or_numeric:
                    return "segment_comparison requires at least one categorical column or numeric/temporal features to derive segments."
        if tool_name in self._OBJECTIVE_REQUIRED_TOOLS and not self._objective_allows_tool(tool_name):
            if tool_name == "scenario_simulation":
                has_driver = any(f.kind == "driver" and f.dimension and f.measure for f in self.memory.findings)
                if not has_driver:
                    return f"{tool_name} requires an objective or at least one quantified driver finding."
            else:
                return f"{tool_name} requires an objective that requests its decision purpose."

        profile = self.last_profile
        if profile is not None and profile.archetype:
            from src.core.profiler import (
                ARCHETYPE_CONTINUOUS_SERIES,
                ARCHETYPE_SENSOR,
            )

            # Safe Quantitative Routing: Archetype Anti-Pattern Guards (§4.4)
            if tool_name in {"train_model", "evaluate_model", "regression_analysis"}:
                meta = self.memory.dataset_metadata
                if meta and (reason := profile.censoring_rationale(meta.target_column)):
                    return reason

            if profile.archetype in (ARCHETYPE_CONTINUOUS_SERIES, ARCHETYPE_SENSOR) and tool_name == "regression_analysis":
                numeric_cols = [c.name for c in profile.columns if c.kind == "numeric"]
                has_stationary = any("return" in c.lower() or "diff" in c.lower() or "delta" in c.lower() for c in numeric_cols)
                if not has_stationary:
                    return "regression_analysis on raw non-stationary time series produces spurious correlation; difference to stationary log returns or use time_series_analysis / forecast."

        return None

    @classmethod
    def _plan_items(cls, reply: Any) -> list[Any]:
        """The plan's step list wherever a drifting model put it: a top-level
        list, under steps/plan/next_steps/actions/tasks, or a dict keyed by
        step number."""
        if isinstance(reply, list):
            return reply
        if not isinstance(reply, dict):
            return []
        for key in cls._PLAN_KEYS:
            value = reply.get(key)
            if isinstance(value, list) and value:
                return value
            if isinstance(value, dict) and value:
                if any(k in value for k in cls._TOOL_KEYS):
                    return [value]
                if nested := cls._plan_items(value):
                    return nested
                keyed = [(k, v) for k, v in value.items() if isinstance(v, dict)]
                if keyed:
                    return [
                        {**v, "step_number": int(m.group())} if (m := re.search(r"\d+", str(k))) else v
                        for k, v in keyed
                    ]
        return []

    @classmethod
    def _normalise_reply(cls, reply: Any) -> Any:
        """Repair reply-form drift: a bare step list, a missing or oddly
        spelled `status`. Error replies pass through untouched."""
        if isinstance(reply, list):
            return {"status": "in_progress", "steps": reply}
        if not isinstance(reply, dict) or reply.get("status") == "error":
            return reply
        reply = dict(reply)
        status = re.sub(r"[\s-]+", "_", str(reply.get("status") or "").strip().lower())
        if status in cls._DONE_STATUSES:
            reply["status"] = "complete"
        elif status in cls._GOING_STATUSES or cls._plan_items(reply):
            reply["status"] = "in_progress"
        elif reply.get("insights") or reply.get("recommendations"):
            reply["status"] = "complete"
        return reply

    def _resolve_tool_name(self, raw: str) -> tuple[str | None, str]:
        """(registered tool name or None, the name cleaned of decoration).
        Small models write "functions.clean_data", "Clean_Data", "clean_dat":
        match case-/underscore-insensitively, then by a unique close match."""
        name = self._TOOL_PREFIX_RE.sub("", raw.strip()).strip("`'\" ").removesuffix("()").split(".")[-1].strip()
        if not name or self.tool_registry.has(name):
            return (name or None), name
        names = self.tool_registry.names()
        key = re.sub(r"[\W_]+", "", name.lower())
        exact = [n for n in names if re.sub(r"[\W_]+", "", n.lower()) == key]
        if len(exact) == 1:
            return exact[0], name
        close = difflib.get_close_matches(name.lower(), names, n=2, cutoff=0.8)
        return (close[0] if len(close) == 1 else None), name

    def _parse_steps(self, llm_response: Any, *, cap: bool = True) -> list[AnalysisStep]:
        """
        Parse LLM plan steps, skipping malformed entries instead of crashing.
        Field names, tool names and the step container are matched tolerantly;
        exact duplicates are dropped and the plan is capped at
        _MAX_STEPS_PER_CYCLE (`cap=False` for a plan the profile built).

        Anti-hallucination guard: steps naming tools that do not exist in the
        ToolRegistry are rejected here (never executed), and a planner note is
        recorded so the next reasoning cycle sees the correction.
        """
        steps: list[AnalysisStep] = []
        rejected: list[str] = []
        notes: list[str] = []
        seen: set[str] = set()
        tool_counts: dict[str, int] = {}
        duplicates = 0
        for idx, s in enumerate(self._plan_items(llm_response), 1):
            if isinstance(s, str):
                s = {"tool_name": s}
            if not isinstance(s, dict):
                continue
            for key in ("function", "tool_call"):  # OpenAI style: {"function": {"name", "arguments"}}
                if isinstance(s.get(key), dict):
                    s = {**s, **s[key]}
            candidates = [v for k in self._TOOL_KEYS if isinstance(v := s.get(k), str) and v.strip()]
            if not candidates:
                continue
            resolved = [self._resolve_tool_name(c) for c in candidates]
            hit = next(((n, c) for n, c in resolved if n), None)
            if hit is None:
                rejected.append(resolved[0][1] or candidates[0])
                continue
            tool_name, cleaned = hit
            prerequisite_error = self._tool_hard_prerequisite_error(tool_name)
            if prerequisite_error:
                rejected.append(f"{tool_name} ({prerequisite_error})")
                continue
            if tool_counts.get(tool_name, 0) >= self._MAX_STEPS_PER_TOOL:
                notes.append(f"dropped redundant {tool_name!r} step (per-tool budget {self._MAX_STEPS_PER_TOOL})")
                continue
            if (
                getattr(self.tool_registry.get(tool_name), "executes_code", False)
                and not code_execution_available()
            ):
                rejected.append(tool_name)
                continue
            if tool_name != cleaned:
                notes.append(f"corrected {cleaned!r} -> {tool_name!r}")
            parameters: Any = next((s[k] for k in self._PARAM_KEYS if s.get(k)), {})
            if isinstance(parameters, str):
                # ...and sometimes send the parameter object as a JSON string.
                try:
                    parameters = json.loads(parameters)
                except json.JSONDecodeError:
                    parameters = {}
            if not isinstance(parameters, dict):
                parameters = {}
            signature = f"{tool_name}|{json.dumps(parameters, sort_keys=True, default=str)}"
            if signature in seen:
                duplicates += 1
                continue
            seen.add(signature)
            tool_counts[tool_name] = tool_counts.get(tool_name, 0) + 1
            number = next((s[k] for k in self._STEP_NO_KEYS if s.get(k) is not None), None)
            if isinstance(number, str) and number.strip().isdigit():
                number = int(number)
            steps.append(
                AnalysisStep(
                    step_number=number if isinstance(number, int) and not isinstance(number, bool) and number > 0 else idx,
                    tool_name=tool_name,
                    parameters=parameters,
                    rationale=str(s.get("rationale", "")),
                )
            )
        if duplicates:
            notes.append(f"dropped {duplicates} duplicate step(s)")
        if cap and len(steps) > _MAX_STEPS_PER_CYCLE:
            notes.append(
                f"only the first {_MAX_STEPS_PER_CYCLE} of {len(steps)} steps were run, the rest "
                "were dropped; plan them in a later cycle"
            )
            steps = steps[:_MAX_STEPS_PER_CYCLE]
        if rejected:
            console.print(
                f"  [yellow]⚠ Rejected {len(rejected)} hallucinated tool name(s): "
                f"{', '.join(rejected)}[/]"
            )
        if rejected or notes:
            summary = [
                *(
                    [
                        f"Rejected unknown tool name(s): {', '.join(sorted(set(rejected)))}. "
                        f"Only these tools exist: {', '.join(self.tool_registry.names())}."
                    ]
                    if rejected else []
                ),
                *(f"Plan note: {n}." for n in notes),
            ]
            self.memory.append_tool_result(
                ToolResult(
                    tool_name="planner",
                    status="skipped",
                    output={"summary": " ".join(summary)},
                    iteration=self.memory.iteration_count,
                )
            )
        return steps

    def _bind_objective_comparison(self, steps: list[AnalysisStep]) -> list[AnalysisStep]:
        request = comparison_objective(self.objective, self.last_profile)
        if request is None or not (params := request.tool_parameters()):
            return steps
        name = "time_series_analysis" if request.hourly else "segment_comparison"
        if not self.tool_registry.has(name) or self._tool_hard_prerequisite_error(name):
            return steps
        for result in self.memory.tool_results:
            if result.tool_name != name or result.status != "success":
                continue
            if request.hourly and result.output.get("value_column") == request.measure:
                return steps
            if any(m == request.measure and (not request.dimension or d == request.dimension)
                   for m, d in result.output.get("pairs_tested", [])):
                return steps
        for step in steps:
            if step.tool_name == name:
                step.parameters.update(params)
                if not request.hourly and not request.dimension:
                    step.parameters.pop("dimension_column", None)
                return steps
        meta = self.memory.dataset_metadata
        if meta and self.tool_registry.get(name).applies_to(self.last_profile, meta) > 0:
            steps.append(AnalysisStep(step_number=len(steps) + 1, tool_name=name,
                                      parameters={"file_path": meta.file_path, **params},
                                      rationale="Compute the comparison requested in the objective."))
        return steps

    def _ensure_preprocessing_steps(self, steps: list[AnalysisStep]) -> list[AnalysisStep]:
        """Make deterministic data preparation non-optional in an LLM plan.

        The model may edit the suggested plan, but it cannot safely decide to
        analyse raw composite strings or skip the cleaned-file contract. Tool
        ``applies_to`` remains the generic relevance gate, so this does not
        force feature extraction onto already tidy numeric data.
        """
        meta = self.memory.dataset_metadata
        if meta is None:
            return steps
        completed = {
            result.tool_name for result in self.memory.tool_results if result.status == "success"
        }
        planned = {step.tool_name for step in steps}
        prefix: list[AnalysisStep] = []
        if "clean_data" not in completed and "clean_data" not in planned:
            prefix.append(AnalysisStep(
                step_number=0,
                tool_name="clean_data",
                parameters={"file_path": meta.file_path, "target_column": meta.target_column},
                rationale="Required preprocessing: establish the cleaned dataset used by every downstream tool.",
            ))
        if (
            self.tool_registry.has("extract_structured_features")
            and "extract_structured_features" not in completed
            and "extract_structured_features" not in planned
            and self.tool_registry.get("extract_structured_features").applies_to(self.last_profile, meta) > 0.0
        ):
            prefix.append(AnalysisStep(
                step_number=0,
                tool_name="extract_structured_features",
                parameters={"file_path": meta.file_path},
                rationale=(
                    "Required preprocessing: derive typed features from composite/unit-bearing "
                    "fields and quarantine semantic mismatches."
                ),
            ))
        ordered = [*prefix, *steps]
        for number, step in enumerate(ordered, 1):
            step.step_number = number
        return ordered

    def _ensure_agenda_anchor_steps(self, steps: list[AnalysisStep]) -> list[AnalysisStep]:
        """Guarantee that top-priority Question Agenda discoveries and the 4-Pillar
        Autonomous Cadence are executed reproducibly across reasoning cycles.

        Cycle 1 anchors lead Question Agenda segment comparisons.
        Cycle 2 anchors Archetype Discovery (cluster_data) and What-If Simulations (scenario_simulation)
        when they fit the data and have not run yet.
        """
        meta = self.memory.dataset_metadata
        if meta is None:
            return steps

        fp = str(self.memory.get_context("cleaned_file_path") or meta.file_path)
        completed = {
            result.tool_name for result in self.memory.tool_results if result.status == "success"
        }
        planned = {s.tool_name for s in steps}

        if self.memory.iteration_count == 1:
            profile = self.last_profile
            archetype = getattr(profile, "archetype", None)
            if archetype and archetype in ARCHETYPE_CANONICAL_TOOLS:
                canonical_tools = ARCHETYPE_CANONICAL_TOOLS[archetype]
                anchored_count = 0
                for tool_name in canonical_tools:
                    if tool_name not in completed and tool_name not in planned:
                        if (
                            self.tool_registry.has(tool_name)
                            and self._objective_allows_tool(tool_name)
                            and self._tool_hard_prerequisite_error(tool_name) is None
                        ):
                            tool = self.tool_registry.get(tool_name)
                            if tool.applies_to(profile, meta) <= 0.0 and getattr(profile, "archetype", None) != archetype:
                                continue
                            tool_params: dict[str, Any] = {"file_path": fp}
                            try:
                                tool_params.update(tool.default_params(profile, meta) or {})
                            except Exception as exc:
                                logger.debug("Failed resolving default params for %s: %s", tool_name, exc)
                            steps.append(
                                AnalysisStep(
                                    step_number=len(steps) + 1,
                                    tool_name=tool_name,
                                    parameters=tool_params,
                                    rationale=f"Archetype anchor pass: canonical blueprint analysis for {archetype} archetype.",
                                )
                            )
                            planned.add(tool_name)
                            anchored_count += 1
                            if anchored_count >= 2:
                                break

            if (
                self.tool_registry.has("segment_comparison")
                and "segment_comparison" not in completed
                and "segment_comparison" not in planned
            ):
                agenda = self.memory.get_context("question_agenda") or []
                for q in agenda:
                    if isinstance(q, dict) and q.get("suggested_tool") == "segment_comparison":
                        cols = q.get("columns") or []
                        if len(cols) >= 2:
                            steps.append(
                                AnalysisStep(
                                    step_number=len(steps) + 1,
                                    tool_name="segment_comparison",
                                    parameters={
                                        "file_path": fp,
                                        "measure_column": cols[0],
                                        "dimension_column": cols[1],
                                    },
                                    rationale=f"Anchor pass: test lead agenda question — {str(q.get('text', ''))[:80]}",
                                )
                            )
                            planned.add("segment_comparison")
                            break

        elif self.memory.iteration_count >= 2:
            profile = self.last_profile
            archetype = getattr(profile, "archetype", None)
            # Pillar 3: Unsupervised Archetype Discovery
            has_supervised = any(
                s.tool_name in ("regression_analysis", "train_model", "evaluate_model") or "target_column" in s.parameters
                for s in steps
            ) or bool(getattr(meta, "target_column", None) or self.memory.get_context("target_column"))
            objective = str(getattr(self, "objective", "") or "")
            auto_obj = str(getattr(self, "autonomous_objective", "") or "")
            combined = f"{objective} {auto_obj}".strip()
            explicit_clustering_ask = bool(re.search(r"\b(?:segment|cluster|cohort|persona|archetype)\b", combined, re.IGNORECASE))
            if (
                self.tool_registry.has("cluster_data")
                and "cluster_data" not in completed
                and "cluster_data" not in planned
                and self.use_ml
                and (explicit_clustering_ask or not has_supervised)
                and self._objective_allows_tool("cluster_data")
            ):
                tool = self.tool_registry.get("cluster_data")
                if tool.applies_to(profile, meta) >= 0.4:
                    num_cols = len(profile.columns_of_kind("numeric")) if profile else 0
                    if num_cols >= 2 and (profile.row_count if profile else 0) >= 30:
                        steps.append(
                            AnalysisStep(
                                step_number=len(steps) + 1,
                                tool_name="cluster_data",
                                parameters={"file_path": fp},
                                rationale="Autonomous cadence: discover natural behavioral archetypes and segment clusters.",
                            )
                        )
                        planned.add("cluster_data")

            # Pillar 4: Parametric What-If Simulation (Autonomous Lead-Driver Intervention)
            drivers = [f for f in self.memory.findings if f.kind == "driver" and f.dimension]
            if (
                self.tool_registry.has("scenario_simulation")
                and "scenario_simulation" not in completed
                and "scenario_simulation" not in planned
                and (self._objective_allows_tool("scenario_simulation") or bool(drivers))
                and self._tool_hard_prerequisite_error("scenario_simulation") is None
            ):
                sim_tool = self.tool_registry.get("scenario_simulation")
                if sim_tool.applies_to(profile, meta) > 0.0:
                    sim_params: dict[str, Any] = {"file_path": fp}
                    try:
                        sim_params.update(sim_tool.default_params(profile, meta) or {})
                    except Exception as exc:
                        logger.debug('Failed resolving default params for scenario_simulation: %s', exc)
                    if drivers and "driver_column" not in sim_params:
                        sim_params["driver_column"] = drivers[0].dimension
                        if drivers[0].measure:
                            sim_params["target_column"] = drivers[0].measure
                    steps.append(
                        AnalysisStep(
                            step_number=len(steps) + 1,
                            tool_name="scenario_simulation",
                            parameters=sim_params,
                            rationale="Autonomous cadence: quantify What-If driver interventions and gap closing.",
                        )
                    )
                    planned.add("scenario_simulation")

            # Anchor next tier of canonical blueprint tools
            if archetype and archetype in ARCHETYPE_CANONICAL_TOOLS:
                for tool_name in ARCHETYPE_CANONICAL_TOOLS[archetype]:
                    if tool_name not in completed and tool_name not in planned:
                        if (
                            self.tool_registry.has(tool_name)
                            and self._objective_allows_tool(tool_name)
                            and self._tool_hard_prerequisite_error(tool_name) is None
                        ):
                            tool = self.tool_registry.get(tool_name)
                            if tool.applies_to(profile, meta) > 0.0 or getattr(profile, "archetype", None) == archetype:
                                tool_params = {"file_path": fp}
                                try:
                                    tool_params.update(tool.default_params(profile, meta) or {})
                                except Exception:
                                    pass
                                prepared, _reason = self._validate_and_prepare_fallback_params(
                                    tool, tool_params, profile, meta
                                )
                                if not prepared:
                                    continue
                                steps.append(
                                    AnalysisStep(
                                        step_number=len(steps) + 1,
                                        tool_name=tool_name,
                                        parameters=prepared,
                                        rationale=f"Autonomous depth: tier analytical tool {tool_name} for {archetype}.",
                                    )
                                )
                                planned.add(tool_name)
                                break

        # Fallback to unanswered question agenda if fewer than 2 steps are planned in this cycle
        if len(steps) < 2:
            agenda = self.memory.get_context("question_agenda") or []
            for q in agenda:
                if isinstance(q, dict):
                    tool_cand = q.get("suggested_tool")
                    if (
                        tool_cand
                        and tool_cand not in completed
                        and tool_cand not in planned
                        and self.tool_registry.has(tool_cand)
                        and self._objective_allows_tool(tool_cand)
                        and self._tool_hard_prerequisite_error(tool_cand) is None
                    ):
                        tool_params = {"file_path": fp}
                        cols = q.get("columns") or []
                        if tool_cand == "segment_comparison" and len(cols) >= 2:
                            tool_params["measure_column"] = cols[0]
                            tool_params["dimension_column"] = cols[1]
                        else:
                            try:
                                t_obj = self.tool_registry.get(tool_cand)
                                tool_params.update(t_obj.default_params(profile, meta) or {})
                            except Exception:
                                pass
                        t_obj = self.tool_registry.get(tool_cand)
                        prepared, reason = self._validate_and_prepare_fallback_params(
                            t_obj, tool_params, profile, meta, question=q
                        )
                        if not prepared:
                            if isinstance(q, dict):
                                q["blocked"] = True
                                q["blocked_reason"] = reason or "Required parameters missing"
                            continue
                        steps.append(
                            AnalysisStep(
                                step_number=len(steps) + 1,
                                tool_name=tool_cand,
                                parameters=prepared,
                                rationale=f"Agenda fallback: answer top unaddressed question — {str(q.get('text', ''))[:80]}",
                            )
                        )
                        planned.add(tool_cand)
                        if len(steps) >= 3:
                            break

        # Breadth safeguard: if fewer than 2 steps planned, pull from unrun candidate tools
        if len(steps) < 2:
            candidate_tools = list(self.tool_registry.candidate_tools(
                profile, meta, use_ml=self.use_ml, use_llm=self.use_llm
            ))
            for cand_tool in candidate_tools:
                cand_name = getattr(cand_tool, "name", "")
                if (
                    cand_name
                    and cand_name not in completed
                    and cand_name not in planned
                    and cand_name not in self._FALLBACK_EXCLUDED_TOOLS
                    and self._objective_allows_tool(cand_name)
                    and self._tool_hard_prerequisite_error(cand_name) is None
                    and cand_tool.applies_to(profile, meta) > 0.0
                ):
                    c_params = {"file_path": fp}
                    try:
                        c_params.update(cand_tool.default_params(profile, meta) or {})
                    except Exception:
                        pass
                    prepared, _reason = self._validate_and_prepare_fallback_params(
                        cand_tool, c_params, profile, meta
                    )
                    if not prepared:
                        continue
                    steps.append(
                        AnalysisStep(
                            step_number=len(steps) + 1,
                            tool_name=cand_name,
                            parameters=prepared,
                            rationale=f"Autonomous breadth: candidate analytical tool {cand_name}.",
                        )
                    )
                    planned.add(cand_name)
                    if len(steps) >= 3:
                        break

        for number, step in enumerate(steps, 1):
            step.step_number = number
        return steps

    #: Nature-driven tools included in the fallback plan when applies_to()
    #: scores them at full confidence (1.0) — same tools, same gating logic
    #: the LLM planner sees, so there's one source of truth for "what suits
    #: this data" (dimensionality analysis uses the same profile signal).
    #: Tools the deterministic plan sequences explicitly (or never runs from
    #: the profile sweep): pipeline control, the supervised branch decided
    #: below on the target, and code execution, which needs an LLM to write
    #: the code and is meaningless without one.
    _FALLBACK_EXCLUDED_TOOLS = frozenset({
        "ingest_dataset", "clean_data", "generate_report",
        "extract_structured_features",
        "train_model", "evaluate_model", "cluster_data",
        "execute_dynamic_code",
        # Now returns chart specs the dashboard renders, but everything it
        # draws unprompted (distributions, correlation heatmap, importances)
        # the deterministic dashboard already builds from the profile and
        # tool outputs — scheduling it here would only duplicate panels. The
        # LLM planner can still call it for a specific chart_type.
        "generate_visualizations",
    })

    #: applies_to score a tool must reach to earn a slot in the deterministic
    #: plan. Below 1.0 so a domain matched on partial evidence (0.65 for a
    #: ticker+price file with no OHLC) still contributes its analysis.
    _FALLBACK_MIN_SCORE = 0.6

    _ARM_NAME_RE = re.compile(r"arm|variant|treat|group|condition|cohort|bucket|version", re.IGNORECASE)

    @classmethod
    def _experiment_arm_column(cls, profile: DatasetProfile | None, target: str | None) -> str | None:
        """The arm column of an experiment-archetype profile, else None:
        a low-cardinality dimension named in the archetype evidence (whatever
        its shape), else one with an arm-like name, else the first one."""
        if profile is None or getattr(profile, "archetype", None) != "experiment":
            return None
        dims = [c.name for c in profile.dimensions() if 2 <= c.nunique <= 6 and c.name != target]
        raw_evidence = getattr(profile, "archetype_evidence", None)
        if isinstance(raw_evidence, dict):
            evidence = [f"{k} {v}" for k, v in raw_evidence.items()]
        elif isinstance(raw_evidence, (list, tuple)):
            evidence = [str(e) for e in raw_evidence]
        else:
            evidence = [str(raw_evidence or "")]
        # A name counts only as a whole word/quoted token of an evidence item.
        named = [d for d in dims if any(re.search(rf"(?<!\w){re.escape(d)}(?!\w)", e) for e in evidence)]
        for group in (
            named,
            [d for d in dims if cls._ARM_NAME_RE.search(d)],
            dims,
        ):
            if group:
                return group[0]
        return None

    def _resolve_target_from_question(
        self, question: Any, profile: DatasetProfile | None, meta: Any
    ) -> str | None:
        """Resolve a target column from a question agenda item or profile/metadata.
        Returns the resolved column name if unambiguous, else None."""
        if meta and getattr(meta, "target_column", None):
            return str(meta.target_column)

        if isinstance(question, dict):
            cols = question.get("columns") or []
            text = str(question.get("text", "")).lower()
        else:
            cols = getattr(question, "columns", []) or []
            text = str(getattr(question, "text", "")).lower()

        # Check explicit columns in question
        numeric_cols = []
        if profile:
            col_map = {c.name.lower(): c.name for c in profile.columns}
            for col in cols:
                real_col = col_map.get(str(col).lower())
                if real_col:
                    cp = next((c for c in profile.columns if c.name == real_col), None)
                    if cp and cp.kind == "numeric" and cp.nunique > 1:
                        numeric_cols.append(real_col)

        if len(numeric_cols) == 1:
            return numeric_cols[0]

        # Check measures in profile that match question text
        if profile:
            matched_measures = []
            for m in profile.measures():
                if m.name.lower() in text and m.nunique > 1:
                    matched_measures.append(m.name)
            if len(matched_measures) == 1:
                return matched_measures[0]

        return None

    def _validate_and_prepare_fallback_params(
        self,
        tool: Any,
        raw_params: dict[str, Any],
        profile: DatasetProfile | None,
        meta: Any,
        question: Any = None,
    ) -> tuple[dict[str, Any] | None, str | None]:
        """Validate parameters for a tool in the fallback plan.
        Resolves missing target_column from question or profile if unambiguous.
        Returns (prepared_params, blocked_reason) — (None, reason) if parameter resolution fails."""
        tool_params = dict(raw_params)
        schema = tool.get_schema() if hasattr(tool, "get_schema") else {}
        required = [
            k for k, spec in schema.items()
            if isinstance(spec, dict) and spec.get("required") and k not in ("file_path", "df", "dataset")
        ]

        # Resolve target_column if required and missing
        if "target_column" in required and not tool_params.get("target_column"):
            target = self._resolve_target_from_question(question, profile, meta)
            if target:
                tool_params["target_column"] = target
            else:
                return None, f"Missing required parameter 'target_column' for {getattr(tool, 'name', '')}"

        # Resolve feature_column / measure_column if required and missing
        for col_param in ("feature_column", "measure_column"):
            if col_param in required and not tool_params.get(col_param):
                target = self._resolve_target_from_question(question, profile, meta)
                if target:
                    tool_params[col_param] = target
                else:
                    return None, f"Missing required parameter '{col_param}' for {getattr(tool, 'name', '')}"

        # Check all required parameters
        missing = [k for k in required if not tool_params.get(k)]
        if missing:
            return None, f"Missing required parameter(s): {', '.join(missing)}"

        # Check signature against known invalid steps
        tool_name = getattr(tool, "name", "")
        if _step_signature(tool_name, tool_params) in self._invalid_steps:
            return None, f"Step signature '{tool_name}' previously failed validation"

        return tool_params, None

    def _build_fallback_plan(self) -> dict[str, Any]:
        """
        Deterministic analysis plan used when the LLM is unreachable on the
        first reasoning cycle. Always: clean → outliers → correlation, plus
        whichever nature-specific tools the profile-driven gating says
        apply at full confidence, plus (train + evaluate) when a target
        exists or (cluster + visualise) otherwise.
        """
        meta = self.memory.dataset_metadata
        if meta is None:
            raise RuntimeError("No dataset loaded — cannot build a fallback plan.")
        fp = meta.file_path
        profile = self.last_profile
        steps: list[dict[str, Any]] = [
            {
                "step_number": 1,
                "tool_name": "clean_data",
                "parameters": {
                    "file_path": fp,
                    "target_column": meta.target_column,
                },
                # No imputation: inferential tools use complete cases and
                # models impute inside cross-validation.
                "rationale": "Fallback plan: clean the dataset (types, duplicates) before analysis.",
            },
        ]

        # Semantic/unit extraction is a preprocessing stage, not an optional
        # analytical hypothesis. Its own applies_to gate keeps clean numeric
        # datasets untouched; when relevant it must run before correlations,
        # segment comparisons or models see composite strings.
        if self.tool_registry.has("extract_structured_features"):
            feature_tool = self.tool_registry.get("extract_structured_features")
            if feature_tool.applies_to(profile, meta) > 0.0:
                steps.append({
                    "step_number": len(steps) + 1,
                    "tool_name": "extract_structured_features",
                    "parameters": {"file_path": fp},
                    "rationale": (
                        "Fallback plan: convert unit-bearing/composite fields into "
                        "typed features and quarantine semantic mismatches before analysis."
                    ),
                })

        # detect_outliers/correlation_analysis are near-universal but not
        # unconditional — e.g. correlation_analysis needs 2+ numeric columns
        # — so they go through the same applies_to gate as everything else
        # rather than being hardcoded past it (a single-numeric-column
        # dataset would otherwise error out here every time).
        for name, params, rationale in (
            ("detect_outliers", {"file_path": fp, "method": "iqr"}, "Fallback plan: flag anomalous rows."),
            ("correlation_analysis", {"file_path": fp, "target_column": meta.target_column}, "Fallback plan: quantify feature relationships."),
        ):
            if (
                self.tool_registry.has(name)
                and self.tool_registry.get(name).applies_to(profile, meta) > 0.0
                and self._tool_hard_prerequisite_error(name) is None
            ):
                steps.append({
                    "step_number": len(steps) + 1,
                    "tool_name": name,
                    "parameters": params,
                    "rationale": rationale,
                })

        # Every registered tool the profile says fits, ranked by its own
        # applies_to score — not a hardcoded name list. This is what makes the
        # no-LLM path a real analyst rather than a stub: a tool registered
        # after this function was written (the domain tools, anything added
        # later) is planned automatically, and a dataset recognised as
        # transactional gets its cohort analysis without an LLM ever being
        # reachable. The previous hardcoded tuple silently excluded every
        # tool it predated.
        already = {s["tool_name"] for s in steps} | self._FALLBACK_EXCLUDED_TOOLS
        candidate_tools = list(self.tool_registry.candidate_tools(
            profile, meta, use_ml=self.use_ml, use_llm=self.use_llm
        ))
        recommended_order: dict[str, int] = {}
        routing_family: str = ""
        if hasattr(self, "question_routing") and self.question_routing:
            routing_family = getattr(self.question_routing, "primary_family", "")
            rec_tools = getattr(self.question_routing, "recommended_tools", [])
            recommended_order = {n: i for i, n in enumerate(rec_tools)}
        elif self.memory:
            stored_routing = self.memory.get_context("question_routing")
            if isinstance(stored_routing, dict):
                routing_family = str(stored_routing.get("primary_family") or "")
                rec_tools = list(stored_routing.get("recommended_tools") or [])
                recommended_order = {n: i for i, n in enumerate(rec_tools)}

        if recommended_order:
            candidate_tools.sort(
                key=lambda t: (
                    0 if getattr(t, "name", "") in recommended_order else 1,
                    recommended_order.get(getattr(t, "name", ""), 999),
                    -t.applies_to(profile, meta),
                )
            )

        for tool in candidate_tools:
            name = getattr(tool, "name", "")
            if name in already:
                continue
            prerequisite_error = self._tool_hard_prerequisite_error(name)
            if prerequisite_error is not None:
                logger.debug("Fallback planner skipped %s: %s", name, prerequisite_error)
                continue
            score = tool.applies_to(profile, meta)
            if score < self._FALLBACK_MIN_SCORE:
                continue

            tool_params: dict[str, Any] = {"file_path": fp}
            try:
                tool_params.update(tool.default_params(profile, meta) or {})
            except Exception:
                logger.debug("default_params failed for a scheduled tool", exc_info=True)

            # Never schedule a step that cannot run. file_path and output_dir
            # are injected by BaseTool.prepare_params, and requires_context
            # entries are filled from memory, so only genuinely unfilled
            # required parameters disqualify a tool.
            try:
                raw_schema = tool.get_schema()
            except Exception:
                raw_schema = {}
            schema: dict[str, Any] = {}
            if isinstance(raw_schema, dict):
                if "properties" in raw_schema and isinstance(raw_schema["properties"], dict):
                    req_list = raw_schema.get("required")
                    req_set = set(req_list) if isinstance(req_list, (list, tuple)) else set()
                    for k, v in raw_schema["properties"].items():
                        if isinstance(v, dict):
                            schema[k] = {**v, "required": bool(k in req_set or v.get("required", False))}
                        else:
                            schema[k] = {"type": "string", "required": bool(k in req_set)}
                else:
                    for k, v in raw_schema.items():
                        if isinstance(v, dict):
                            schema[k] = v
                        else:
                            schema[k] = {"type": str(v), "required": False}

            injected = {"file_path", "output_dir"} | set(
                getattr(tool, "requires_context", {}).values()
            )
            unfilled = [
                key
                for key, spec in schema.items()
                if isinstance(spec, dict)
                and spec.get("required")
                and key not in tool_params
                and key not in injected
            ]
            if unfilled:
                continue

            steps.append({
                "step_number": len(steps) + 1,
                "tool_name": name,
                "parameters": tool_params,
                "rationale": (
                    f"Fallback plan: matched objective intent '{routing_family}' recommending '{name}' (applies_to {score:.2f})." if name in recommended_order and routing_family else f"Fallback plan: profile-driven selection scored '{name}' at {score:.2f} for this dataset."

                ),
            })
            already.add(name)

        # An experiment's primary question is the arm comparison — make sure
        # the draft tests it rather than leaving the grouping to chance.
        arm = self._experiment_arm_column(profile, meta.target_column)
        if arm and self.tool_registry.has("select_statistical_test"):
            test_tool = self.tool_registry.get("select_statistical_test")
            if (
                test_tool.applies_to(profile, meta) > 0.0
                and self._tool_hard_prerequisite_error("select_statistical_test") is None
            ):
                test_step = next((s for s in steps if s["tool_name"] == "select_statistical_test"), None)
                if test_step is None:
                    test_params: dict[str, Any] = {"file_path": fp}
                    try:
                        test_params.update(test_tool.default_params(profile, meta) or {})
                    except Exception:
                        logger.debug("default_params failed for a test tool", exc_info=True)
                    test_step = {
                        "step_number": len(steps) + 1,
                        "tool_name": "select_statistical_test",
                        "parameters": test_params,
                        "rationale": f"Fallback plan: experiment data — compare outcomes across arms of '{arm}'.",
                    }
                    steps.append(test_step)
                test_step["parameters"]["group_column"] = arm

        # High-value question agenda pre-seeding: if the agenda recommends
        # segment_comparison (e.g. key measure broken out by prominent dimension),
        # ensure it is present in the draft plan rather than relying on chance.
        if self.tool_registry.has("segment_comparison") and not any(s["tool_name"] == "segment_comparison" for s in steps):
            agenda = self.memory.get_context("question_agenda") or []
            for q in agenda:
                if isinstance(q, dict) and q.get("suggested_tool") == "segment_comparison":
                    cols = q.get("columns") or []
                    if len(cols) >= 2:
                        steps.append({
                            "step_number": len(steps) + 1,
                            "tool_name": "segment_comparison",
                            "parameters": {
                                "file_path": fp,
                                "measure_column": cols[0],
                                "dimension_column": cols[1],
                            },
                            "rationale": f"Fallback plan: test primary agenda question — {str(q.get('text', ''))[:80]}",
                        })
                        already.add("segment_comparison")
                        break

        # Decision Intelligence: pre-seed scenario simulation for continuous drivers
        if (
            self.tool_registry.has("scenario_simulation")
            and self._objective_allows_tool("scenario_simulation")
            and not any(s["tool_name"] == "scenario_simulation" for s in steps)
        ):
            sim_tool = self.tool_registry.get("scenario_simulation")
            if sim_tool.applies_to(profile, meta) > 0.0:
                sim_params: dict[str, Any] = {"file_path": fp}
                try:
                    sim_params.update(sim_tool.default_params(profile, meta) or {})
                except Exception as exc:
                    logger.debug('Failed resolving default params for scenario_simulation fallback: %s', exc)
                steps.append({
                    "step_number": len(steps) + 1,
                    "tool_name": "scenario_simulation",
                    "parameters": sim_params,
                    "rationale": "Fallback plan: quantify What-If driver shifts and segment gap opportunities.",
                })
                already.add("scenario_simulation")

        if not self.use_ml:
            # No model-fitting branch at all: no supervised training and no
            # clustering fallback. The profile-driven analyses above already
            # ran, so the plan is complete and genuinely ML-free.
            pass
        elif meta.task_type == "survival":
            if self.tool_registry.has("survival_analysis") and "survival_analysis" not in already:
                tool = self.tool_registry.get("survival_analysis")
                steps.append({
                    "step_number": len(steps) + 1,
                    "tool_name": "survival_analysis",
                    "parameters": {"file_path": fp, **tool.default_params(profile, meta), "event_column": meta.target_column},
                    "rationale": "Account for censoring and differing observation times.",
                })
        elif meta.target_column and meta.task_type in ("classification", "regression"):
            steps += [
                {
                    "step_number": len(steps) + 1,
                    "tool_name": "train_model",
                    "parameters": {
                        "file_path": fp,
                        "target_column": meta.target_column,
                        "task_type": meta.task_type,
                    },
                    "rationale": "Fallback plan: train baseline models with CV.",
                },
                {
                    "step_number": len(steps) + 2,
                    "tool_name": "evaluate_model",
                    "parameters": {
                        "file_path": fp,
                        "target_column": meta.target_column,
                        "task_type": meta.task_type,
                    },
                    "rationale": "Fallback plan: evaluate the best model on held-out data.",
                },
            ]
        elif (
            self._objective_allows_tool("cluster_data")
            and self._numeric_column_count() >= 2
            and self._tool_hard_prerequisite_error("cluster_data") is None
        ):
            c_tool = self.tool_registry.get("cluster_data") if self.tool_registry.has("cluster_data") else None
            if c_tool is not None and c_tool.applies_to(profile, meta) >= 0.4:
                steps += [
                    {
                        "step_number": len(steps) + 1,
                        "tool_name": "cluster_data",
                        "parameters": {"file_path": fp},
                        "rationale": "Fallback plan: no target — discover natural segments.",
                    },
                ]
        return {
            "status": "in_progress",
            "reasoning": "LLM unavailable — executing deterministic fallback plan.",
            "steps": steps,
            "deterministic": True,   # _parse_steps does not cap a plan the profile built
        }

    def _build_phase2_drilldown_plan(self) -> dict[str, Any]:
        """Phase 2 Drill-Down plan: deepens exploration on anomalies and top drivers
        discovered during Phase 1 (Broad Screening) to enforce the Two-Phase Adaptive Cadence.
        """
        meta = self.memory.dataset_metadata
        if meta is None:
            return self._build_fallback_plan()
        fp = str(self.memory.get_context("cleaned_file_path") or meta.file_path)
        profile = self.last_profile
        completed = {r.tool_name for r in self.memory.tool_results if r.status == "success"}

        steps: list[dict[str, Any]] = []

        # 1. Drill-down on top drivers or segment differences discovered in Cycle 1
        drivers = [f for f in self.memory.findings if f.kind in ("driver", "segment_lift") and f.measure and f.dimension]
        if drivers and self.tool_registry.has("segment_comparison"):
            for d in drivers[:2]:
                steps.append({
                    "step_number": len(steps) + 1,
                    "tool_name": "segment_comparison",
                    "parameters": {
                        "file_path": fp,
                        "measure_column": d.measure,
                        "dimension_column": d.dimension,
                    },
                    "rationale": f"Phase 2 Drill-Down: deep-dive comparison on key driver '{d.dimension}' for '{d.measure}'.",
                })

        # 2. Anomaly investigation if outliers or spikes detected
        outlier_findings = [f for f in self.memory.findings if f.kind == "outlier_scan"]
        if (
            outlier_findings
            and self.tool_registry.has("anomaly_analysis")
            and "anomaly_analysis" not in completed
            and self._objective_allows_tool("anomaly_analysis")
        ):
            anomaly_tool = self.tool_registry.get("anomaly_analysis")
            if anomaly_tool.applies_to(profile, meta) > 0.0:
                steps.append({
                    "step_number": len(steps) + 1,
                    "tool_name": "anomaly_analysis",
                    "parameters": {"file_path": fp},
                    "rationale": "Phase 2 Drill-Down: investigate temporal and segment anomaly patterns.",
                })

        # 3. What-If Simulation on key drivers if applicable
        if (
            self.tool_registry.has("scenario_simulation")
            and "scenario_simulation" not in completed
            and self._objective_allows_tool("scenario_simulation")
        ):
            sim_tool = self.tool_registry.get("scenario_simulation")
            if sim_tool.applies_to(profile, meta) > 0.0:
                sim_params: dict[str, Any] = {"file_path": fp}
                try:
                    sim_params.update(sim_tool.default_params(profile, meta) or {})
                except Exception:
                    pass
                if drivers and "driver_column" not in sim_params:
                    sim_params["driver_column"] = drivers[0].dimension
                    if drivers[0].measure:
                        sim_params["target_column"] = drivers[0].measure
                steps.append({
                    "step_number": len(steps) + 1,
                    "tool_name": "scenario_simulation",
                    "parameters": sim_params,
                    "rationale": "Phase 2 Drill-Down: quantify What-If driver interventions and gap closing.",
                })

        # 4. Cluster data if not yet run
        if (
            not steps
            and self.tool_registry.has("cluster_data")
            and "cluster_data" not in completed
            and self.use_ml
            and self._objective_allows_tool("cluster_data")
        ):
            c_tool = self.tool_registry.get("cluster_data")
            if c_tool.applies_to(profile, meta) >= 0.4:
                steps.append({
                    "step_number": len(steps) + 1,
                    "tool_name": "cluster_data",
                    "parameters": {"file_path": fp},
                    "rationale": "Phase 2 Drill-Down: discover behavioral archetype clusters.",
                })

        # 5. If still empty, fall back to remaining tools from candidate tools
        if not steps:
            fb = self._build_fallback_plan()
            unrun = [s for s in fb.get("steps", []) if s.get("tool_name") not in completed]
            steps = unrun[:3] if unrun else fb.get("steps", [])[:2]

        for i, s in enumerate(steps, 1):
            s["step_number"] = i

        return {
            "status": "in_progress",
            "reasoning": "Two-Phase Adaptive Cadence: Phase 2 Drill-Down on anomalies, key drivers, and deeper segments.",
            "steps": steps,
        }
