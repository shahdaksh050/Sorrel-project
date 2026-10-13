from __future__ import annotations

import pandas as pd
import pytest

from src.core.agenda import build_agenda
from src.core.profiler import profile_dataframe
from src.tools.segment_comparison import SegmentComparisonTool, _find_entity_name_map


def test_agenda_penalizes_surrogate_ids() -> None:
    # Dataset with both a surrogate ID and a semantic category column
    df = pd.DataFrame({
        "leagueID": [1, 1, 2, 2, 3, 3] * 10,
        "league_name": ["Premier League", "Premier League", "La Liga", "La Liga", "Serie A", "Serie A"] * 10,
        "goals": [2.5, 3.1, 1.8, 2.0, 1.2, 1.5] * 10,
    })
    profile = profile_dataframe(df)
    agenda = build_agenda(profile, decision=None)

    # First segment question should target 'league_name', not 'leagueID'
    segment_questions = [q for q in agenda if q.kind == "segment"]
    assert len(segment_questions) > 0
    top_dim = segment_questions[0].columns[1]
    assert top_dim == "league_name"


def test_find_entity_name_map() -> None:
    df = pd.DataFrame({
        "leagueID": [1, 2, 3, 1, 2, 3],
        "league_name": ["EPL", "La Liga", "Serie A", "EPL", "La Liga", "Serie A"],
        "goals": [2, 3, 1, 2, 1, 0],
    })
    mapping = _find_entity_name_map(df, "leagueID")
    assert mapping is not None
    assert mapping.get(1) == "EPL" or mapping.get("1") == "EPL"
    assert mapping.get(2) == "La Liga" or mapping.get("2") == "La Liga"
    assert mapping.get(3) == "Serie A" or mapping.get("3") == "Serie A"


def test_segment_comparison_enriches_level_label(tmp_path: pytest.TempPathFactory) -> None:
    csv_path = tmp_path / "entity_test.csv"  # type: ignore[operator]
    df = pd.DataFrame({
        "leagueID": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2] * 5,
        "league_name": ["Premier League"] * 50 + ["Serie A"] * 50,
        "goals": [3.5] * 50 + [1.2] * 50,
    })
    df.to_csv(csv_path, index=False)

    tool = SegmentComparisonTool()
    output = tool.execute(
        file_path=str(csv_path),
        measure_column="goals",
        dimension_column="leagueID",
    )
    assert "comparisons" in output
    comps = output["comparisons"]
    assert len(comps) > 0
    # Check that level_label is populated
    assert any(c.get("level_label") in ("Premier League", "Serie A") for c in comps)

    findings = tool.findings(output, profile=profile_dataframe(df), metadata=None)
    assert len(findings) > 0
    # Headline should mention the semantic name
    assert any("Premier League" in f.headline or "Serie A" in f.headline for f in findings)
