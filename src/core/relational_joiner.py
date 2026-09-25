"""
Automated Multi-Entity Relational Discovery and Star-Schema Assembly.

Infers foreign key relationships across arbitrary collections of DataFrames
(e.g., multi-tab Excel workbooks, relational database exports) and assembles
a unified denormalized star-schema analysis frame.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass
class ForeignKeyRelation:
    """A discovered relational link between two tables."""

    parent_table: str
    parent_key: str
    child_table: str
    child_key: str
    confidence: float
    cardinality: str  # "one_to_many" | "one_to_one"

    def to_dict(self) -> dict[str, Any]:
        return {
            "parent_table": self.parent_table,
            "parent_key": self.parent_key,
            "child_table": self.child_table,
            "child_key": self.child_key,
            "confidence": round(self.confidence, 3),
            "cardinality": self.cardinality,
        }


def _clean_token(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def discover_foreign_keys(tables: dict[str, pd.DataFrame]) -> list[ForeignKeyRelation]:
    """
    Scan all pairs of tables to detect foreign-key relationships.
    Uses column naming heuristics, uniqueness constraints, and value overlap.
    """
    relations: list[ForeignKeyRelation] = []
    table_names = list(tables.keys())

    for i, t1 in enumerate(table_names):
        df1 = tables[t1]
        if df1.empty:
            continue
        for j, t2 in enumerate(table_names):
            if i == j:
                continue
            df2 = tables[t2]
            if df2.empty:
                continue

            # Candidate columns in t1 that could match columns in t2
            for c1 in df1.columns:
                c1_str = str(c1)
                t1_tok = _clean_token(t1)
                c1_tok = _clean_token(c1_str)

                # Skip non-key-like columns (floats, timestamps)
                if pd.api.types.is_float_dtype(df1[c1]) or pd.api.types.is_datetime64_any_dtype(df1[c1]):
                    continue

                for c2 in df2.columns:
                    c2_str = str(c2)
                    t2_tok = _clean_token(t2)
                    c2_tok = _clean_token(c2_str)

                    if pd.api.types.is_float_dtype(df2[c2]) or pd.api.types.is_datetime64_any_dtype(df2[c2]):
                        continue

                    # Naming heuristic:
                    # 1. Exact match (e.g. customer_id == customer_id)
                    # 2. Parent is 'id' / 'code' / 'key' and Child is '<parent>_id'
                    # 3. Child is '<parent>_id'
                    is_name_match = (
                        c1_tok == c2_tok
                        or (c1_tok in ("id", "code", "key") and c2_tok == f"{t1_tok}{c1_tok}")
                        or (c2_tok in ("id", "code", "key") and c1_tok == f"{t2_tok}{c2_tok}")
                        or (c1_tok.endswith("id") and c2_tok.endswith("id") and (c1_tok in c2_tok or c2_tok in c1_tok))
                    )
                    if not is_name_match:
                        continue

                    # Value overlap check
                    s1 = df1[c1].dropna().astype(str).str.strip()
                    s2 = df2[c2].dropna().astype(str).str.strip()
                    if s1.empty or s2.empty:
                        continue

                    u1 = set(s1.unique())
                    u2 = set(s2.unique())
                    overlap = len(u1 & u2)
                    if not overlap:
                        continue

                    jaccard = overlap / len(u1 | u2)
                    containment_1_in_2 = overlap / len(u1)
                    containment_2_in_1 = overlap / len(u2)

                    # Determine parent vs child: parent has unique keys
                    is_u1_unique = len(u1) / len(s1) >= 0.95
                    is_u2_unique = len(u2) / len(s2) >= 0.95

                    if is_u1_unique and containment_2_in_1 >= 0.6:
                        # t1 is parent (dimension), t2 is child (fact)
                        card = "one_to_one" if is_u2_unique else "one_to_many"
                        relations.append(
                            ForeignKeyRelation(
                                parent_table=t1,
                                parent_key=c1_str,
                                child_table=t2,
                                child_key=c2_str,
                                confidence=max(jaccard, containment_2_in_1),
                                cardinality=card,
                            )
                        )
                    elif is_u2_unique and containment_1_in_2 >= 0.6:
                        # t2 is parent (dimension), t1 is child (fact)
                        card = "one_to_one" if is_u1_unique else "one_to_many"
                        relations.append(
                            ForeignKeyRelation(
                                parent_table=t2,
                                parent_key=c2_str,
                                child_table=t1,
                                child_key=c1_str,
                                confidence=max(jaccard, containment_1_in_2),
                                cardinality=card,
                            )
                        )

    # Deduplicate relations
    unique_rels: dict[tuple[str, str, str, str], ForeignKeyRelation] = {}
    for r in relations:
        key = (r.parent_table, r.parent_key, r.child_table, r.child_key)
        if key not in unique_rels or r.confidence > unique_rels[key].confidence:
            unique_rels[key] = r

    return sorted(unique_rels.values(), key=lambda r: r.confidence, reverse=True)


def assemble_star_schema(
    tables: dict[str, pd.DataFrame],
    relations: list[ForeignKeyRelation] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Assemble a star schema DataFrame by identifying the central fact table
    and joining surrounding dimension tables.
    """
    if not tables:
        return pd.DataFrame(), ["No tables provided."]

    if len(tables) == 1:
        name = next(iter(tables))
        return tables[name].copy(), [f"Single table '{name}' loaded."]

    if relations is None:
        relations = discover_foreign_keys(tables)

    notes: list[str] = []
    if not relations:
        # Fall back to largest table
        largest_name = max(tables.keys(), key=lambda k: len(tables[k]))
        notes.append(f"No foreign keys discovered. Preserved largest table '{largest_name}'.")
        return tables[largest_name].copy(), notes

    # Determine fact table: the table that appears most as child_table
    child_counts: dict[str, int] = {}
    for r in relations:
        child_counts[r.child_table] = child_counts.get(r.child_table, 0) + 1

    fact_table_name = max(child_counts.keys(), key=lambda k: (child_counts[k], len(tables[k])))
    joined_df = tables[fact_table_name].copy()
    joined_parents: set[str] = set()

    notes.append(f"Selected '{fact_table_name}' as central fact table ({len(joined_df)} rows).")

    # Join parent tables
    for r in relations:
        if r.child_table != fact_table_name:
            continue
        if r.parent_table in joined_parents:
            continue

        parent_df = tables[r.parent_table]

        # Guard against merge fan-out: `discover_foreign_keys` scores a
        # parent key as "unique" at a >=0.95 threshold, not exact uniqueness,
        # so a left-join on it can still silently multiply fact rows for any
        # duplicated key value. Collapse duplicates deterministically (keep
        # the first row per key) before joining, and note it so the fan-out
        # is visible rather than a silent row-count change.
        dup_count = int(parent_df[r.parent_key].duplicated().sum())
        if dup_count:
            parent_df = parent_df.drop_duplicates(subset=[r.parent_key], keep="first")
            notes.append(
                f"'{r.parent_table}.{r.parent_key}' had {dup_count} duplicate key value(s); "
                f"kept the first row per key before joining '{fact_table_name}' to prevent fan-out."
            )

        # Explicit, meaningful suffixes for any surviving overlapping columns
        # (other than the join keys themselves) instead of pandas' default
        # _x/_y, which gives no clue which table a colliding column came from.
        suffix_fact = f"_{fact_table_name}"
        suffix_parent = f"_{r.parent_table}"
        overlapping = (set(parent_df.columns) & set(joined_df.columns)) - {r.child_key, r.parent_key}

        try:
            joined_df = joined_df.merge(
                parent_df,
                left_on=r.child_key,
                right_on=r.parent_key,
                how="left",
                suffixes=(suffix_fact, suffix_parent),
                validate="many_to_one",
            )
            joined_parents.add(r.parent_table)
            note = (
                f"Joined dimension '{r.parent_table}' on {r.child_key}={r.parent_key} "
                f"({r.cardinality}, confidence {r.confidence:.2f})."
            )
            if overlapping:
                note += (
                    f" Overlapping column(s) suffixed {suffix_fact}/{suffix_parent}: "
                    f"{', '.join(sorted(overlapping))}."
                )
            notes.append(note)
        except pd.errors.MergeError as exc:
            notes.append(
                f"Refused to join '{r.parent_table}': cardinality check failed after "
                f"de-duplication ({exc})."
            )
        except Exception as exc:
            notes.append(f"Failed to join '{r.parent_table}': {exc}")

    return joined_df, notes
