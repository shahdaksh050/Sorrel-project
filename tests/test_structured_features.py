"""Tests for deterministic, additive structured feature extraction."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.tools.structured_features import StructuredFeatureExtractionTool


def test_extracts_typed_features_and_preserves_source_columns(tmp_path: Path) -> None:
    """Composite unit strings become additive typed columns with provenance."""
    source = tmp_path / "catalog.csv"
    pd.DataFrame(
        {
            "price": ["₹14,999", "INR 20,000"],
            "ram": ["6 GB RAM, 128 GB ROM", "8GB RAM, 256GB Storage"],
            "battery": ["5000mAh, 33W Fast Charging", "4500 mAh, 20 W"],
            "operating_system": ["Android v14", "No FM Radio"],
            "network": ["5G, NFC, Bluetooth", "No 5G, Wi-Fi"],
        }
    ).to_csv(source, index=False)

    result = StructuredFeatureExtractionTool().execute(file_path=str(source), output_dir=str(tmp_path / "derived"))
    derived = result["derived_dataset"]
    destination = Path(str(derived["path"]))
    output = pd.read_csv(destination)

    assert destination.exists()
    assert all(column in output.columns for column in ["price", "ram", "battery", "operating_system", "network"])
    assert output["currency_amount"].tolist() == [14999.0, 20000.0]
    assert output["currency_code"].tolist() == ["INR", "INR"]
    assert output["memory_gb"].tolist() == [6.0, 8.0]
    assert output["storage_gb"].tolist() == [128.0, 256.0]
    assert output["battery_mah"].tolist() == [5000.0, 4500.0]
    assert output["platform"].iloc[0] == "Android"
    assert bool(output["has_5g"].iloc[1]) is False
    assert derived["added_columns"] > 0
    assert derived["name"] == "analysis_ready_structured"
    assert "platform" in derived["columns"]
    assert result["semantic_readiness"]["columns_assessed"] > 0
    assert all("source_column" in item and "coverage_pct" in item for item in result["provenance"])
    assert all(item["source_column"] == "multiple_sources" for item in result["provenance"])
    assert not any(column.startswith("derived__battery__") for column in output.columns)
    row_platform = next(item for item in result["provenance"] if item["derived_column"] == "platform")
    assert row_platform["source_columns"] == ["operating_system"]


def test_extraction_is_deterministic_for_the_same_input(tmp_path: Path) -> None:
    """The same input yields stable metadata and derived column names."""
    source = tmp_path / "input.csv"
    pd.DataFrame({"specification": ["2.5 GHz, 6.5 inch display, 50 MP"]}).to_csv(source, index=False)
    tool = StructuredFeatureExtractionTool()

    first = tool.execute(file_path=str(source), output_dir=str(tmp_path / "one"))
    second = tool.execute(file_path=str(source), output_dir=str(tmp_path / "two"))

    assert first["derived_dataset"]["added_columns"] == second["derived_dataset"]["added_columns"]
    assert [item["derived_column"] for item in first["provenance"]] == [
        item["derived_column"] for item in second["provenance"]
    ]


def test_quarantines_materially_misaligned_raw_column(tmp_path: Path) -> None:
    """A shifted categorical source is removed while trustworthy fields remain."""
    source = tmp_path / "misaligned.csv"
    pd.DataFrame(
        {
            "operating_system": ["Android", "No FM Radio", "No NFC", "Android"],
            "price": ["₹10,000", "₹11,000", "₹12,000", "₹13,000"],
        }
    ).to_csv(source, index=False)

    result = StructuredFeatureExtractionTool().execute(file_path=str(source), output_dir=str(tmp_path / "derived"))
    output = pd.read_csv(Path(str(result["derived_dataset"]["path"])))

    assert "operating_system" not in output.columns
    assert "price" in output.columns
    assert result["quarantined_columns"] == [
        {
            "source_column": "operating_system",
            "action": "excluded_from_analysis_ready_dataset",
            "reason": "low_name_value_semantic_conformance",
            "conformance_pct": 50.0,
            "classified_value_count": 4,
        }
    ]


def test_does_not_duplicate_numeric_source_measure(tmp_path: Path) -> None:
    """Already numeric measures are retained, not copied as parser-derived features."""
    source = tmp_path / "numeric.csv"
    pd.DataFrame({"price": [100.0, 120.0], "description": ["5G", "4G"]}).to_csv(source, index=False)

    result = StructuredFeatureExtractionTool().execute(file_path=str(source), output_dir=str(tmp_path / "derived"))
    output = pd.read_csv(Path(str(result["derived_dataset"]["path"])))

    assert "price" in output.columns
    assert "currency_amount" not in output.columns


def test_canonical_name_collision_never_overwrites_source_column(tmp_path: Path) -> None:
    """A pre-existing canonical field wins; recovered values get a derived fallback name."""
    source = tmp_path / "collision.csv"
    pd.DataFrame(
        {
            "memory_gb": [4.0, 8.0],
            "specification": ["6 GB RAM", "12 GB RAM"],
        }
    ).to_csv(source, index=False)

    result = StructuredFeatureExtractionTool().execute(
        file_path=str(source), output_dir=str(tmp_path / "derived")
    )
    output = pd.read_csv(Path(str(result["derived_dataset"]["path"])))

    assert output["memory_gb"].tolist() == [4.0, 8.0]
    assert output["derived__row__memory_gb"].tolist() == [6.0, 12.0]
