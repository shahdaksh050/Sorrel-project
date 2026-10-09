"""Question-agenda tests: question-led coverage must not depend on column order."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.core.agenda import build_agenda
from src.core.profiler import profile_dataframe


def test_agenda_always_asks_distribution_and_quality_questions() -> None:
    frame = pd.DataFrame(
        {"amount": np.linspace(1.25, 40.75, 40), "group": ["a", "b"] * 20}
    )
    questions = build_agenda(profile_dataframe(frame), {"mode": "describe"})
    assert any(question.kind == "distribution" and question.columns == ["amount"] for question in questions)
    assert any(question.kind == "quality" for question in questions)


def test_agenda_excludes_semantically_quarantined_columns() -> None:
    frame = pd.DataFrame(
        {
            "value": np.linspace(0.25, 15.0, 60),
            "bad_label": ["x", "y", "z"] * 20,
            "valid_segment": ["a", "b"] * 30,
        }
    )
    questions = build_agenda(
        profile_dataframe(frame),
        {"mode": "describe"},
        excluded_columns={"bad_label"},
    )
    assert not any("bad_label" in question.columns for question in questions)
    assert any("valid_segment" in question.columns for question in questions)
