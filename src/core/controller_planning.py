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

from src.core.controller_common import _MAX_STEPS_PER_CYCLE, ControllerState, console
from src.core.governance import (
    code_execution_available,
)
from src.core.memory import AnalysisStep, ToolResult
from src.core.profiler import DatasetProfile

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

    #: Nature-driven tools included in the fallback plan when applies_to()
    #: scores them at full confidence (1.0) — same tools, same gating logic
    #: the LLM planner sees, so there's one source of truth for "what suits
    #: this data" (controller._should_decompose folds in the same way).
    #: Tools the deterministic plan sequences explicitly (or never runs from
    #: the profile sweep): pipeline control, the supervised branch decided
    #: below on the target, and code execution, which needs an LLM to write
    #: the code and is meaningless without one.
    _FALLBACK_EXCLUDED_TOOLS = frozenset({
        "ingest_dataset", "clean_data", "generate_report",
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

        # detect_outliers/correlation_analysis are near-universal but not
        # unconditional — e.g. correlation_analysis needs 2+ numeric columns
        # — so they go through the same applies_to gate as everything else
        # rather than being hardcoded past it (a single-numeric-column
        # dataset would otherwise error out here every time).
        for name, params, rationale in (
            ("detect_outliers", {"file_path": fp, "method": "iqr"}, "Fallback plan: flag anomalous rows."),
            ("correlation_analysis", {"file_path": fp, "target_column": meta.target_column}, "Fallback plan: quantify feature relationships."),
        ):
            if self.tool_registry.has(name) and self.tool_registry.get(name).applies_to(profile, meta) > 0.0:
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
        for tool in self.tool_registry.candidate_tools(
            profile, meta, use_ml=self.use_ml, use_llm=self.use_llm
        ):
            name = getattr(tool, "name", "")
            if name in already:
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
                schema = tool.get_schema()
            except Exception:
                schema = {}
            injected = {"file_path", "output_dir"} | set(
                getattr(tool, "requires_context", {}).values()
            )
            unfilled = [
                key
                for key, spec in schema.items()
                if spec.get("required")
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
                    f"Fallback plan: profile-driven selection scored '{name}' "
                    f"at {score:.2f} for this dataset."
                ),
            })
            already.add(name)

        # An experiment's primary question is the arm comparison — make sure
        # the draft tests it rather than leaving the grouping to chance.
        arm = self._experiment_arm_column(profile, meta.target_column)
        if arm and self.tool_registry.has("select_statistical_test"):
            test_tool = self.tool_registry.get("select_statistical_test")
            if test_tool.applies_to(profile, meta) > 0.0:
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

        if not self.use_ml:
            # No model-fitting branch at all: no supervised training and no
            # clustering fallback. The profile-driven analyses above already
            # ran, so the plan is complete and genuinely ML-free.
            pass
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
        else:
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
