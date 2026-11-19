from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.core.geo_stats import haversine_distance, spherical_center
from src.tools.geospatial import GeospatialAnalysisTool


def test_dateline_center_and_valid_population(tmp_path: Path) -> None:
    path = tmp_path / "dateline.csv"
    pd.DataFrame({"latitude": [0] * 6 + [91], "longitude": [179, -179] * 3 + [0]}).to_csv(path, index=False)
    result = GeospatialAnalysisTool().run(file_path=str(path))
    assert result.status == "success", result.error_message
    assert abs(result.output["centroid"]["lon"]) == pytest.approx(180)
    assert (result.output["n_points"], result.output["n_excluded"]) == (6, 1)


def test_undefined_or_invalid_centers() -> None:
    assert spherical_center(np.array([0, 0]), np.array([0, 180])) is None
    assert spherical_center(np.array([91, np.nan]), np.array([5, 6])) is None
    with pytest.raises(ValueError, match="same length"):
        spherical_center(np.array([0]), np.array([]))


def test_haversine_dateline_wrapping_and_bounds() -> None:
    # Points at (0, 179) and (0, -179) around equator center at (0, 180)
    lats = np.array([0.0, 0.0, 95.0, 0.0])  # row 2 has invalid latitude > 90
    lons = np.array([179.0, -179.0, 0.0, 200.0])  # row 3 has invalid longitude > 180
    dists = haversine_distance(lats, lons, center_lat=0.0, center_lon=180.0)
    # Valid points: 1 degree on equator is ~111.19 km
    assert dists[0] == pytest.approx(111.195, rel=1e-3)
    assert dists[1] == pytest.approx(111.195, rel=1e-3)
    # Invalid coordinates return NaN
    assert np.isnan(dists[2])
    assert np.isnan(dists[3])

