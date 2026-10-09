"""Tests for document-level token prevalence in TextAnalysisTool."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.core.profiler import profile_dataframe
from src.tools.text_analysis import TextAnalysisTool


def test_token_stats_distinguish_document_prevalence_from_occurrences(tmp_path: Path) -> None:
    path = tmp_path / "notes.csv"
    path.write_text("notes\ncamera camera camera\ncamera battery\nbattery\n", encoding="utf-8")

    result = TextAnalysisTool().run(file_path=str(path), text_column="notes")

    assert result.status == "success"
    output = result.output
    camera = output["top_token_stats"]["camera"]
    assert output["top_tokens"]["camera"] == 4
    assert camera["occurrence_count"] == 4
    assert camera["document_count"] == 2
    assert camera["document_prevalence"] == pytest.approx(2 / 3, abs=0.0001)
    assert camera["occurrence_frequency"] == pytest.approx(4 / 6, abs=0.0001)


def test_text_finding_uses_document_prevalence_not_occurrences(tmp_path: Path) -> None:
    path = tmp_path / "notes.csv"
    path.write_text("notes\ncamera camera camera\ncamera battery\nbattery\n", encoding="utf-8")
    tool = TextAnalysisTool()
    result = tool.run(file_path=str(path), text_column="notes")

    findings = tool.findings(result.output, None, None)

    assert len(findings) == 1
    finding = findings[0]
    assert finding.effect == pytest.approx(2 / 3, abs=0.0001)
    assert "2 of 3 rows" in finding.headline
    assert "4 total occurrences" in finding.headline


def test_unit_heavy_compact_specs_are_not_treated_as_prose(tmp_path: Path) -> None:
    path = tmp_path / "specs.csv"
    path.write_text(
        'specs\n"50 MP + 12 MP camera"\n"5000 mAh battery, 33 W charging"\n"6.5 inch display, 120 Hz"\n',
        encoding="utf-8",
    )

    result = TextAnalysisTool().run(file_path=str(path), text_column="specs")

    assert result.status == "error"
    assert "unit-bearing specifications" in (result.error_message or "")


def test_unit_heavy_profile_does_not_offer_text_analysis() -> None:
    frame = pd.DataFrame(
        {
            "camera": [
                f"50 MP + 12 MP camera with 4K video and 8 MP front camera variant {index}"
                for index in range(30)
            ]
        }
    )
    assert TextAnalysisTool().applies_to(profile_dataframe(frame), None) == 0.0


def test_named_short_reviews_still_offer_text_analysis() -> None:
    frame = pd.DataFrame(
        {"review_text": [f"Good value and reliable service item {index}" for index in range(30)]}
    )
    assert TextAnalysisTool().applies_to(profile_dataframe(frame), None) == 1.0
