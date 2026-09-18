"""
Geospatial Analysis Tool — Execution Layer.

Stage 3: Bounding box, centroid, and grid-density hotspots for datasets
with latitude/longitude columns (DatasetProfile.has_geo()).

Pure pandas/numpy binning — no mapping dependency needed for the summary
statistics an autonomous agent can reason over textually.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from src.core.findings import Finding
from src.tools.base import BaseTool, ToolExecutionError
from src.tools.data_processing import _read_df

if TYPE_CHECKING:
    from src.core.memory import DatasetMetadata
    from src.core.profiler import DatasetProfile

_LAT_NAME_HINTS = ("lat", "latitude")
_LON_NAME_HINTS = ("lon", "lng", "longitude")

#: How many densest grid cells to report.
TOP_DENSE_CELLS = 5


def _autodetect_geo_columns(df: pd.DataFrame) -> tuple[str | None, str | None]:
    lat_col: str | None = None
    lon_col: str | None = None
    for col in df.select_dtypes(include="number").columns:
        name_l = str(col).lower()
        clean = df[col].dropna()
        if clean.empty:
            continue
        lo, hi = float(clean.min()), float(clean.max())
        if lat_col is None and any(h in name_l for h in _LAT_NAME_HINTS) and -90.0 <= lo and hi <= 90.0:
            lat_col = str(col)
        elif lon_col is None and any(h in name_l for h in _LON_NAME_HINTS) and -180.0 <= lo and hi <= 180.0:
            lon_col = str(col)
    return lat_col, lon_col


class GeospatialAnalysisTool(BaseTool):
    """Bounding box, centroid, and density hotspots for lat/lon coordinates."""

    name = "geospatial_analysis"
    description = (
        "Analyse latitude/longitude coordinates: bounding box, centroid, point count, "
        "and the densest grid cells (spatial hotspots). Use when the data profile shows "
        "geo_lat_col/geo_lon_col. Auto-detects the columns when omitted."
    )

    def applies_to(self, profile: DatasetProfile | None, metadata: DatasetMetadata | None) -> float:
        return 1.0 if profile is not None and profile.has_geo() else 0.0

    def execute(  # type: ignore[override]
        self,
        file_path: str,
        lat_column: str | None = None,
        lon_column: str | None = None,
        grid_size: int = 10,
        **_: Any,
    ) -> dict[str, Any]:
        df = _read_df(file_path)

        if lat_column is None or lon_column is None:
            auto_lat, auto_lon = _autodetect_geo_columns(df)
            lat_column = lat_column or auto_lat
            lon_column = lon_column or auto_lon
        if not lat_column or not lon_column or lat_column not in df.columns or lon_column not in df.columns:
            raise ToolExecutionError(
                "No usable lat/lon column pair found. Pass lat_column and lon_column explicitly."
            )

        coords = df[[lat_column, lon_column]].dropna()
        coords = coords[
            coords[lat_column].between(-90, 90) & coords[lon_column].between(-180, 180)
        ]
        if len(coords) < 5:
            raise ToolExecutionError(
                f"Only {len(coords)} valid coordinate rows after filtering — need at least 5."
            )

        lat = coords[lat_column].to_numpy(dtype=float)
        lon = coords[lon_column].to_numpy(dtype=float)

        bounding_box = {
            "min_lat": round(float(lat.min()), 5),
            "max_lat": round(float(lat.max()), 5),
            "min_lon": round(float(lon.min()), 5),
            "max_lon": round(float(lon.max()), 5),
        }
        centroid = {"lat": round(float(lat.mean()), 5), "lon": round(float(lon.mean()), 5)}

        grid_size = max(2, int(grid_size))
        lat_bins = np.linspace(bounding_box["min_lat"], bounding_box["max_lat"], grid_size + 1)
        lon_bins = np.linspace(bounding_box["min_lon"], bounding_box["max_lon"], grid_size + 1)
        lat_idx = np.clip(np.digitize(lat, lat_bins) - 1, 0, grid_size - 1)
        lon_idx = np.clip(np.digitize(lon, lon_bins) - 1, 0, grid_size - 1)

        cell_counts: dict[tuple[int, int], int] = {}
        for li, lo_i in zip(lat_idx, lon_idx, strict=True):
            key = (int(li), int(lo_i))
            cell_counts[key] = cell_counts.get(key, 0) + 1

        densest = sorted(cell_counts.items(), key=lambda kv: kv[1], reverse=True)[:TOP_DENSE_CELLS]
        densest_cells = [
            {
                "lat_range": [round(float(lat_bins[li]), 5), round(float(lat_bins[li + 1]), 5)],
                "lon_range": [round(float(lon_bins[lo_i]), 5), round(float(lon_bins[lo_i + 1]), 5)],
                "count": count,
            }
            for (li, lo_i), count in densest
        ]

        return {
            "summary": (
                f"{len(coords):,} coordinate(s) spanning lat [{bounding_box['min_lat']}, "
                f"{bounding_box['max_lat']}], lon [{bounding_box['min_lon']}, "
                f"{bounding_box['max_lon']}]. Densest cell has {densest_cells[0]['count']} point(s)."
                if densest_cells
                else f"{len(coords):,} coordinate(s) analysed."
            ),
            "lat_column": lat_column,
            "lon_column": lon_column,
            "n_points": len(coords),
            "bounding_box": bounding_box,
            "centroid": centroid,
            "grid_size": grid_size,
            "densest_cells": densest_cells,
        }

    def findings(  # type: ignore[override]
        self,
        output: dict[str, Any],
        profile: DatasetProfile | None,
        metadata: DatasetMetadata | None,
    ) -> list[Finding]:
        densest_cells = output.get("densest_cells")
        n_points = output.get("n_points")
        grid_size = output.get("grid_size")
        if not densest_cells or not n_points or not grid_size:
            return []

        top_cell = densest_cells[0]
        count = top_cell.get("count", 0)
        share = count / n_points if n_points else 0.0
        expected_share = 1.0 / (grid_size ** 2)
        # Suppress when the densest cell is no denser than a uniform scatter
        # would produce by chance — no real geographic concentration (T5).
        if share < expected_share * 2.0:
            return []

        return [
            Finding(
                finding_id=f"{self.name}_hotspot",
                kind="geospatial",
                headline=(
                    f"A single grid cell holds {count:,} of {n_points:,} points "
                    f"({share * 100:.1f}% of all coordinates)"
                ),
                detail=(
                    f"Densest cell spans lat {top_cell.get('lat_range')}, "
                    f"lon {top_cell.get('lon_range')}, on a {grid_size}x{grid_size} grid."
                ),
                evidence={
                    "densest_cell": top_cell,
                    "n_points": n_points,
                    "grid_size": grid_size,
                    "centroid": output.get("centroid"),
                },
                measure="point_density",
                dimension="location",
                effect=round(share - expected_share, 4),
                effect_kind="share",
                confidence=0.6,
                surprise=0.5,
            )
        ]

    def get_schema(self) -> dict[str, Any]:
        return {
            "file_path": {"type": "string", "description": "Path to the (cleaned) dataset.", "required": True},
            "lat_column": {"type": "string", "description": "Latitude column. Auto-detected if omitted.", "required": False},
            "lon_column": {"type": "string", "description": "Longitude column. Auto-detected if omitted.", "required": False},
            "grid_size": {
                "type": "int",
                "description": "N×N grid for density hotspots. Default: 10.",
                "required": False,
            },
        }
