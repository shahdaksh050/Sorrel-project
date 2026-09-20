"""Small-cell suppression in finding text (headline/detail), not just tables."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from src.core.findings import Finding
from src.core.privacy import redact_level_in_text, redact_small_level
from src.tools.workforce_analysis import WorkforceAnalysisTool


def _frame(small_dept: str, n_small: int) -> pd.DataFrame:
    """Ops (60, ~10% leave), Sales (40, ~10% leave) and a `small_dept` where everyone left."""
    rows = []
    for dept, n, leavers in (("Ops", 60, 6), ("Sales", 40, 4), (small_dept, n_small, n_small)):
        for i in range(n):
            rows.append({
                "employee_id": f"{dept}{i}", "department": dept, "salary": 50000 + 100 * i,
                "status": "terminated" if i < leavers else "active",
            })
    return pd.DataFrame(rows)


def _findings(tmp_path: Path, df: pd.DataFrame) -> list[Finding]:
    path = tmp_path / "hr.csv"
    df.to_csv(path, index=False)
    tool = WorkforceAnalysisTool()
    res = tool.run(file_path=str(path), department_column="department", salary_column="salary",
                   status_column="status")
    assert res.status == "success", res.error_message
    return tool.findings(res.output, None, None)


def _text(findings: list[Finding]) -> str:
    return " ".join(f"{f.headline} {f.detail}" for f in findings)


def test_two_person_department_is_not_named(tmp_path: Path) -> None:
    text = _text(_findings(tmp_path, _frame("Legal", 2)))
    assert "Legal" not in text


def test_twelve_person_department_is_named(tmp_path: Path) -> None:
    findings = _findings(tmp_path, _frame("Legal", 12))
    assert any("Legal" in f.headline and "highest attrition" in f.headline for f in findings)


def test_env_override_changes_threshold(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DSA_MIN_CELL_SIZE", "2")
    assert "Legal" in _text(_findings(tmp_path, _frame("Legal", 2)))
    monkeypatch.setenv("DSA_MIN_CELL_SIZE", "15")
    assert "Legal" not in _text(_findings(tmp_path, _frame("Legal", 12)))


def test_redact_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DSA_MIN_CELL_SIZE", raising=False)
    assert redact_small_level("Legal", 2) == "a small group"
    assert redact_small_level("Legal", 5) == "Legal"
    assert redact_small_level("Legal", None) == "Legal"
    assert redact_level_in_text("Legal leads", "Legal", 3) == "a small group leads"
    assert redact_level_in_text("Legalese", "Legal", 3) == "Legalese"


def _finding(n: Any) -> Finding:
    return Finding(
        finding_id="f", kind="segment_lift", headline="Legal dept attrition is 100%",
        detail="Legal: n=2", evidence={"n": n}, level="Legal",
    )


def test_finding_guard_redacts_small_level_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DSA_MIN_CELL_SIZE", raising=False)
    f = _finding(2)
    assert "Legal" not in f.headline + f.detail
    assert f.level == "Legal"  # dedupe key untouched
    assert _finding(12).headline == "Legal dept attrition is 100%"
    assert Finding(finding_id="g", kind="x", headline="Legal", level="Legal").headline == "Legal"
