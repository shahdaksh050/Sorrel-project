"""Shared spherical calculations for geographic coordinates in degrees."""
from __future__ import annotations

import numpy as np

EARTH_RADIUS_KM: float = 6371.0088


def spherical_center(latitude: np.ndarray, longitude: np.ndarray) -> tuple[float, float] | None:
    """Mean direction of valid paired coordinates, undefined for a cancelling resultant."""
    if len(latitude) != len(longitude):
        raise ValueError("Latitude and longitude must have the same length.")
    valid = (np.abs(latitude) <= 90) & (np.abs(longitude) <= 180) & np.isfinite(latitude) & np.isfinite(longitude)
    if not valid.any():
        return None
    lat, lon = np.radians(latitude[valid]), np.radians(longitude[valid])
    x = float(np.mean(np.cos(lat) * np.cos(lon)))
    y = float(np.mean(np.cos(lat) * np.sin(lon)))
    z = float(np.mean(np.sin(lat)))
    horizontal = np.hypot(x, y)
    if np.hypot(horizontal, z) < 1e-12:
        return None
    return float(np.degrees(np.arctan2(z, horizontal))), float(np.degrees(np.arctan2(y, x)))


def haversine_distance(
    latitude: np.ndarray | float,
    longitude: np.ndarray | float,
    center_lat: float,
    center_lon: float,
) -> np.ndarray:
    """Compute great-circle distance (km) on a sphere with antimeridian longitudinal wrapping.

    Coordinates outside valid bounds ([-90, 90] lat, [-180, 180] lon) or non-finite return NaN.
    """
    lat_arr = np.asarray(latitude, dtype=float)
    lon_arr = np.asarray(longitude, dtype=float)

    valid = (
        (np.abs(lat_arr) <= 90.0)
        & (np.abs(lon_arr) <= 180.0)
        & np.isfinite(lat_arr)
        & np.isfinite(lon_arr)
        & (-90.0 <= center_lat <= 90.0)
        & (-180.0 <= center_lon <= 180.0)
    )

    out = np.full(lat_arr.shape, np.nan, dtype=float)
    if not np.any(valid):
        return out

    lat_v = lat_arr[valid]
    lon_v = lon_arr[valid]

    dlat = np.radians(lat_v - center_lat)
    # Longitude difference wrapped to [-180, 180] degrees across antimeridian
    dlon_deg = ((lon_v - center_lon + 180.0) % 360.0) - 180.0
    dlon = np.radians(dlon_deg)

    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(np.radians(center_lat)) * np.cos(np.radians(lat_v)) * np.sin(dlon / 2.0) ** 2
    )
    c = 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
    out[valid] = EARTH_RADIUS_KM * c
    return out

