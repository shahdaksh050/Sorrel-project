"""
Hypothesis-Refutation Tree and Counterfactual Reasoning Engine.

Structures agent reasoning as an explicit tree of testable hypotheses
(H0 -> H1 ...) rather than an unstructured linear list of steps.
Supports automated counterfactual probe generation to verify if discovered
associations hold under confounding controls.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class HypothesisNode:
    id: str
    statement: str
    status: str = "untested"  # "untested" | "supported" | "refuted" | "inconclusive"
    parent_id: str | None = None
    confidence: float = 0.5
    evidence_finding_ids: list[str] = field(default_factory=list)
    counterfactual_queries: list[str] = field(default_factory=list)
    rationale: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "statement": self.statement,
            "status": self.status,
            "parent_id": self.parent_id,
            "confidence": round(self.confidence, 3),
            "evidence_finding_ids": list(self.evidence_finding_ids),
            "counterfactual_queries": list(self.counterfactual_queries),
            "rationale": self.rationale,
        }


class HypothesisTree:
    """
    Hierarchical graph of active, verified, or refuted hypotheses during an analysis run.
    """

    def __init__(self, primary_objective: str = "") -> None:
        self.primary_objective = primary_objective
        self.nodes: dict[str, HypothesisNode] = {}
        self._counter: int = 0

        if primary_objective:
            self.add_hypothesis(
                statement=f"Primary Goal: {primary_objective}",
                parent_id=None,
                status="untested",
                rationale="Root hypothesis inferred from user objective.",
            )

    def add_hypothesis(
        self,
        statement: str,
        parent_id: str | None = None,
        status: str = "untested",
        rationale: str = "",
        counterfactuals: list[str] | None = None,
    ) -> HypothesisNode:
        self._counter += 1
        node_id = f"H{self._counter}"
        node = HypothesisNode(
            id=node_id,
            statement=statement.strip(),
            status=status,
            parent_id=parent_id,
            rationale=rationale,
            counterfactual_queries=counterfactuals or [],
        )
        self.nodes[node_id] = node
        return node

    def update_status(
        self,
        node_id: str,
        status: str,
        confidence: float | None = None,
        finding_id: str | None = None,
        rationale: str | None = None,
    ) -> None:
        if node_id not in self.nodes:
            return
        node = self.nodes[node_id]
        node.status = status
        if confidence is not None:
            node.confidence = max(0.0, min(1.0, confidence))
        if finding_id and finding_id not in node.evidence_finding_ids:
            node.evidence_finding_ids.append(finding_id)
        if rationale:
            node.rationale = rationale

    def get_active_hypotheses(self) -> list[HypothesisNode]:
        """Return hypotheses that are either untested or in need of deeper testing."""
        return [n for n in self.nodes.values() if n.status in ("untested", "inconclusive")]

    def to_dict(self) -> dict[str, Any]:
        return {
            "primary_objective": self.primary_objective,
            "total_hypotheses": len(self.nodes),
            "nodes": {k: v.to_dict() for k, v in self.nodes.items()},
        }

    def to_prompt_summary(self) -> str:
        """Compact summary of the hypothesis tree for injection into the reasoning prompt."""
        if not self.nodes:
            return "No active hypotheses recorded."

        lines = ["Active Hypothesis Tree:"]
        for node in self.nodes.values():
            icon = {
                "supported": "✓ Supported",
                "refuted": "✗ Refuted",
                "inconclusive": "? Inconclusive",
                "untested": "○ Untested",
            }.get(node.status, "○ Untested")
            cf_note = f" (Counterfactuals: {len(node.counterfactual_queries)})" if node.counterfactual_queries else ""
            parent_note = f" [Child of {node.parent_id}]" if node.parent_id else ""
            lines.append(f"- [{node.id}] {icon}: {node.statement}{parent_note}{cf_note}")
        return "\n".join(lines)


def generate_counterfactual_probes(
    col1: str,
    col2: str,
    candidate_confounders: list[str],
) -> list[str]:
    """
    Generate candidate counterfactual questions for a discovered correlation between col1 and col2.

    Example:
        col1='price', col2='churn', confounders=['tenure', 'plan']
        -> "Does association between price and churn persist after stratifying by tenure?"
    """
    probes: list[str] = []
    for conf in candidate_confounders:
        if conf not in (col1, col2):
            probes.append(
                f"Does association between '{col1}' and '{col2}' persist after controlling for '{conf}'?"
            )
    return probes[:3]
