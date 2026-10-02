"""
Shared module-level pieces of the controller package: the console, the
prepared-step record, the dataset reader used for dashboards, and the tunable
constants. Kept in their own module so the controller and its mixins
(controller_planning / controller_steps / controller_report) can all import
them without a circular dependency. `src.core.controller` re-exports each name.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, NamedTuple

import pandas as pd
from rich.console import Console

from src.core.governance import CodeGovernor
from src.core.hypothesis import HypothesisTree
from src.core.io import read_any
from src.core.llm_client import LLMClient
from src.core.memory import AnalysisStep, MemorySystem, ToolResult
from src.core.profiler import DatasetProfile
from src.core.prompt_manager import PromptManager
from src.core.tool_registry import ToolRegistry
from src.rlm.engine import RLMEngine

if TYPE_CHECKING:
    from src.core.findings import Finding

console = Console()


class _PreparedStep(NamedTuple):
    """A plan step resolved against run state and cleared to execute."""

    idx: int
    step: AnalysisStep
    tool: Any
    params: dict[str, Any]
    dropped_note: str
    cache_key: str
    cached: ToolResult | None


def _read_dataframe(file_path: str) -> pd.DataFrame:
    """Load a CSV/TSV/Excel dataset for dashboard generation."""
    df, _report = read_any(file_path)
    return df

# Max retries before abandoning a failed step
MAX_STEP_RETRIES = 2
#: Code-running tools fail as part of normal self-correction (a wrong column
#: name, a pandas idiom) — each failure returns a hint the next attempt uses,
#: so they get a larger budget than a deterministic tool that is simply broken.
MAX_CODE_STEP_RETRIES = 5

#: Appended to a reasoning prompt when the first reply was unusable
#: (non-JSON, truncated at max_tokens, or empty) — see analyze().
_COMPACT_RETRY_NOTE = (
    "\n\n## Retry Note\nYour previous reply could not be used (it was not "
    "complete, valid JSON — possibly cut off by the output length limit). "
    "Reply again with ONLY the JSON object: at most 8 steps, one short "
    "sentence per rationale, no markdown fences, no text outside the JSON.\n"
)

#: Ceiling on the prompt budget derived from the model's context window. A
#: 1M-token window is not a reason to send 750k-token planner prompts: cost
#: and latency grow with prompt size while the ranked findings already keep
#: what matters compact. LLM_CONTEXT_TOKENS, when set, overrides all of this.
_MAX_PLANNER_CONTEXT_TOKENS = 32_000

#: Steps run per reasoning cycle — a model that lists twenty is padding, and
#: each extra step costs a tool run plus digest tokens on every later prompt.
_MAX_STEPS_PER_CYCLE = 8

#: Below this context window the chart-design call is skipped: a small model
#: would spend its scarce budget on it and the deterministic dashboard stands.
_MIN_CHART_DESIGN_CONTEXT = 16_000

# Target auto-detection confidence thresholds
_AUTODETECT_HIGH = 0.75   # proceed autonomously above this
_AUTODETECT_LOW  = 0.40   # prompt user (CLI) or best-guess (UI) above this


class ControllerState:
    """State the controller's mixins read and write.

    Annotations only — `AgentController.__init__` assigns every attribute, so
    this adds nothing at runtime; it exists so the mixins type-check against the
    same shape. The three methods below live on `AgentController` itself and are
    declared for the type checker only.
    """

    memory: MemorySystem
    llm_client: LLMClient
    tool_registry: ToolRegistry
    use_llm: bool
    use_ml: bool
    objective: str
    hypothesis_tree: HypothesisTree
    last_profile: DatasetProfile | None
    on_step_callback: Any
    _output_dir: str
    _governor: CodeGovernor
    _rlm_engine: RLMEngine | None
    _prompt_manager: PromptManager | None
    _step_cache: dict[str, ToolResult]
    _tool_failure_counts: dict[str, int]
    _hypothesis_parent_by_key: dict[tuple[str, str], str]
    _hypothesis_parent_by_tool: dict[str, str]

    if TYPE_CHECKING:

        def _stop_requested(self) -> bool: ...
        def _notify_findings(self, findings: list[Finding]) -> None: ...
        def _llm_budget_exhausted(self) -> bool: ...
