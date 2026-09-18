"""
Concentration Analysis Tool — Execution Layer (IMPROVEMENTS.md item 7.2 #2).

Pareto/Gini concentration of a measure across an entity dimension — "the top
10% of customers drive 18.9% of revenue". `cohort_analysis.py` already
computes exactly this one good sentence for transactional data (revenue by
customer); this tool generalises it to *any* file with a measure and an
entity-like dimension, independent of the transactional domain match.

Triviality suppression (T5): a uniform distribution across N entities gives
each roughly 1/N of the total, so the top 10% would hold ~10% of it by
construction. A top-10% share only slightly above 10-15% says nothing —
`findings()` skips it.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from src.core.findings import Finding
from src.core.profiler import _ENTITY_REPEAT_THRESHOLD, profile_dataframe
from src.tools.base import BaseTool, ToolExecutionError
from src.tools.data_processing import _read_df

if TYPE_CHECKING:
    from src.core.memory import DatasetMetadata
    from src.core.profiler import ColumnProfile, DatasetProfile

#: Entity-candidate cardinality window when no explicit column is given and
#: `DatasetProfile.entity_col` is unset — too few entities isn't a
#: distribution worth measuring, too many (near-identifier) is impractical.
_MIN_ENTITY_CARD = 5
_MAX_ENTITY_CARD = 10_000

#: A top-10% share at or below this is what a near-uniform distribution
#: would already produce — not a real concentration story.
_UNIFORM_SHARE_CEILING = 0.15

_ENTITY_NOUNS = {
    "customer": "customers", "client": "clients", "user": "users",
    "account": "accounts", "patient": "patients", "member": "members",
    "employee": "employees", "subscriber": "subscribers", "product": "products",
}

#: Name fragments that mark a column as an entity id even when the profiler
#: classified it as a plain numeric measure — a `customer_id` with too few
#: repeats to trip the identifier/entity heuristics in profiler.py (e.g. 400
#: distinct customers over 2,000 rows: not >=98% unique, so it lands as
#: kind="numeric" -> semantic_role="measure") is still a legitimate grouping
#: key for concentration purposes.
_ENTITY_NAME_HINTS = (
    "customer", "client", "user", "account", "patient", "member",
    "employee", "subscriber", "product",
)


def _entity_noun(column_name: str) -> str:
    name_l = column_name.lower()
    for hint, noun in _ENTITY_NOUNS.items():
        if hint in name_l:
            return noun
    return "entities"


def _pick_measure_columns(profile: DatasetProfile) -> list[ColumnProfile]:
    measures = list(profile.measures())
    return sorted(measures, key=lambda c: (c.unit_hint != "currency", c.name))


def _is_repeating_entity(profile: DatasetProfile, c: ColumnProfile) -> bool:
    """A column is a genuine *entity* to concentrate over only if rows
    actually repeat per value — one row per value (nunique == row_count,
    e.g. a per-row Customer_ID / order line id) is a row key, not an
    entity, and "top 10% of X" over it is just "top 10% of rows" wearing an
    entity noun. Mirrors `profiler.py`'s own entity detection: same
    rows-per-value formula and the same `_ENTITY_REPEAT_THRESHOLD` constant
    (imported, since it's the stable part of that module's contract) so
    this tool doesn't accept a column the profiler's own grain detection
    would reject."""
    if c.nunique <= 0 or c.nunique >= profile.row_count:
        return False
    return (profile.row_count / c.nunique) >= _ENTITY_REPEAT_THRESHOLD


def _pick_entity_column(profile: DatasetProfile) -> str | None:
    if profile.entity_col:
        return profile.entity_col
    candidates = [
        c for c in profile.columns
        if c.semantic_role in ("dimension", "identifier")
        and _MIN_ENTITY_CARD <= c.nunique <= _MAX_ENTITY_CARD
        and _is_repeating_entity(profile, c)
    ]
    if candidates:
        return max(candidates, key=lambda c: c.nunique).name
    name_hint_candidates = [
        c for c in profile.columns
        if any(h in c.name.lower() for h in _ENTITY_NAME_HINTS)
        and _MIN_ENTITY_CARD <= c.nunique <= _MAX_ENTITY_CARD
        and _is_repeating_entity(profile, c)
    ]
    if name_hint_candidates:
        return max(name_hint_candidates, key=lambda c: c.nunique).name
    return None


def _gini(values: np.ndarray) -> float:
    """Population Gini coefficient via the rank-weighted formula. 0 = perfect
    equality, approaching 1 = maximal concentration. Assumes non-negative
    values, which every measure this tool is pointed at (revenue, spend,
    counts) is expected to be."""
    sorted_vals = np.sort(values.astype(float))
    n = len(sorted_vals)
    if n == 0:
        return 0.0
    total = float(sorted_vals.sum())
    if total == 0:
        return 0.0
    index = np.arange(1, n + 1)
    return float((2.0 * np.sum(index * sorted_vals) / (n * total)) - (n + 1.0) / n)


class ConcentrationAnalysisTool(BaseTool):
    """Pareto/Gini concentration of a measure across an entity dimension."""

    name = "concentration_analysis"
    description = (
        "Measure how concentrated a numeric measure is across an entity "
        "dimension (customers, accounts, products...): the share of the "
        "total held by the top 10%/20%/50% of entities, and the Gini "
        "coefficient. Auto-selects the measure and entity column when omitted."
    )

    def applies_to(self, profile: DatasetProfile | None, metadata: DatasetMetadata | None) -> float:
        if profile is None:
            return 0.0
        if not _pick_measure_columns(profile):
            return 0.0
        if profile.entity_col:
            return 0.9
        return 0.7 if _pick_entity_column(profile) else 0.0

    def default_params(
        self, profile: DatasetProfile | None, metadata: DatasetMetadata | None
    ) -> dict[str, Any]:
        if profile is None:
            return {}
        measures = _pick_measure_columns(profile)
        entity = _pick_entity_column(profile)
        if not measures or not entity:
            return {}
        return {"measure_column": measures[0].name, "entity_column": entity}

    def execute(  # type: ignore[override]
        self,
        file_path: str,
        measure_column: str | None = None,
        entity_column: str | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        df = _read_df(file_path)
        if df.empty:
            raise ToolExecutionError("Dataset has no rows.")

        profile = profile_dataframe(df)

        if measure_column and measure_column not in df.columns:
            raise ToolExecutionError(f"Column '{measure_column}' not found in dataset.")
        if entity_column and entity_column not in df.columns:
            raise ToolExecutionError(f"Column '{entity_column}' not found in dataset.")

        if not measure_column:
            candidates = _pick_measure_columns(profile)
            if not candidates:
                raise ToolExecutionError(
                    "No numeric measure column found. Pass measure_column explicitly."
                )
            measure_column = candidates[0].name

        if not entity_column:
            entity_column = _pick_entity_column(profile)
            if not entity_column:
                raise ToolExecutionError(
                    "No entity-like column found. Pass entity_column explicitly."
                )

        numeric = pd.to_numeric(df[measure_column], errors="coerce")
        if numeric.notna().sum() == 0:
            raise ToolExecutionError(
                f"'{measure_column}' is not numeric and cannot be used as a measure."
            )

        work = pd.DataFrame({
            entity_column: df[entity_column],
            measure_column: numeric,
        }).dropna()
        if work.empty:
            raise ToolExecutionError(
                "No rows have both a valid entity and a numeric measure value."
            )

        grouped = work.groupby(entity_column)[measure_column].sum().sort_values(ascending=False)
        n_entities = int(grouped.shape[0])
        if n_entities < 2:
            raise ToolExecutionError(
                f"'{entity_column}' has fewer than 2 distinct entities — nothing to concentrate."
            )

        # Guard against a per-row identifier being used as the entity — not
        # just for the auto-picked column (`_pick_entity_column` already
        # filters these out) but also for an `entity_column` the caller (the
        # LLM) passed explicitly, which skips that filter entirely. Without
        # this, a Customer_ID that is unique per row silently produces
        # n_entities == n_rows and a "top 10% of customers" claim that is
        # really just "top 10% of rows" — same failure profiler.py's own
        # entity detection avoids via `_ENTITY_REPEAT_THRESHOLD`.
        rows_per_entity = len(work) / n_entities
        if rows_per_entity < _ENTITY_REPEAT_THRESHOLD:
            raise ToolExecutionError(
                f"'{entity_column}' has {n_entities:,} distinct values across {len(work):,} "
                f"rows ({rows_per_entity:.2f} rows/value) — that is essentially one row per "
                f"value, not a repeating entity. Concentration over a per-row identifier is "
                "meaningless (it would just report 'top 10% of rows' relabelled as "
                f"'{_entity_noun(entity_column)}'). Pass a column that genuinely repeats per "
                "entity (e.g. a customer/account id with multiple transactions/orders), or "
                "aggregate the data to one row per entity first."
            )

        total_measure = float(grouped.sum())
        if total_measure == 0:
            raise ToolExecutionError(
                f"Total of '{measure_column}' is zero — cannot compute concentration."
            )

        shares: dict[int, float] = {}
        for pct in (10, 20, 50):
            cutoff = max(1, int(np.ceil(n_entities * pct / 100.0)))
            shares[pct] = float(grouped.head(cutoff).sum()) / total_measure

        gini = _gini(grouped.to_numpy())
        noun = _entity_noun(entity_column)

        summary = (
            f"The top 10% of {noun} account for {shares[10] * 100:.1f}% of "
            f"{measure_column} (Gini={gini:.2f}, n={n_entities:,})."
        )

        return {
            "summary": summary,
            "measure_column": measure_column,
            "entity_column": entity_column,
            "n_entities": n_entities,
            "total_measure": round(total_measure, 2),
            "top_10_pct_share": round(shares[10], 4),
            "top_20_pct_share": round(shares[20], 4),
            "top_50_pct_share": round(shares[50], 4),
            "gini_coefficient": round(gini, 4),
        }

    def findings(
        self,
        output: dict[str, Any],
        profile: DatasetProfile | None,
        metadata: DatasetMetadata | None,
    ) -> list[Finding]:
        share10 = output.get("top_10_pct_share")
        measure = output.get("measure_column")
        entity = output.get("entity_column")
        if share10 is None or measure is None or entity is None:
            return []
        if share10 <= _UNIFORM_SHARE_CEILING:
            return []  # T5: no more concentrated than a uniform split would be

        noun = _entity_noun(entity)
        headline = f"The top 10% of {noun} account for {share10 * 100:.1f}% of {measure}"
        return [Finding(
            finding_id=f"concentration_{measure}_{entity}",
            kind="concentration",
            headline=headline,
            detail=(
                f"Gini coefficient {output.get('gini_coefficient')}; top 20% share "
                f"{output.get('top_20_pct_share', 0) * 100:.1f}%; top 50% share "
                f"{output.get('top_50_pct_share', 0) * 100:.1f}%; n={output.get('n_entities')} "
                f"{noun}."
            ),
            evidence=output,
            source_tool=self.name,
            measure=measure,
            dimension=entity,
            effect=share10,
            effect_kind="share",
            confidence=min(1.0, (output.get("n_entities") or 0) / 200.0),
            surprise=min(1.0, max(0.0, share10 - 0.10)),
        )]

    def get_schema(self) -> dict[str, Any]:
        return {
            "file_path": {
                "type": "string",
                "description": "Path to the dataset.",
                "required": True,
            },
            "measure_column": {
                "type": "string",
                "description": (
                    "Numeric measure to sum per entity (revenue, spend...). "
                    "Auto-selected (preferring a currency measure) when omitted."
                ),
                "required": False,
            },
            "entity_column": {
                "type": "string",
                "description": (
                    "Column identifying the entity to group by (customer, "
                    "account, product...). Auto-selected via the dataset's "
                    "entity/grain column, or a reasonable-cardinality dimension, "
                    "when omitted."
                ),
                "required": False,
            },
        }
