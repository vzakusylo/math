"""Геометрия: локальная плоская система координат, расстояния, скорость звука."""
from __future__ import annotations

import numpy as np

R_EARTH = 6_371_008.8  # м


class LocalFrame:
    """Синусоидальная проекция вокруг (lat0, lon0).

    x — на восток, y — на север, метры. Масштаб по x берётся по широте самой точки,
    поэтому локальные расстояния (самолёт ↔ наблюдатель, единицы км) остаются точными
    в областях радиусом в сотни километров.
    """

    def __init__(self, lat0: float, lon0: float):
        self.lat0 = float(lat0)
        self.lon0 = float(lon0)

    def to_xy(self, lat, lon):
        lat = np.asarray(lat, dtype=float)
        lon = np.asarray(lon, dtype=float)
        x = R_EARTH * np.deg2rad(lon - self.lon0) * np.cos(np.deg2rad(lat))
        y = R_EARTH * np.deg2rad(lat - self.lat0)
        return x, y

    def to_latlon(self, x, y):
        lat = self.lat0 + np.rad2deg(np.asarray(y, dtype=float) / R_EARTH)
        lon = self.lon0 + np.rad2deg(
            np.asarray(x, dtype=float) / (R_EARTH * np.cos(np.deg2rad(lat)))
        )
        return lat, lon


def speed_of_sound(temp_c: float = 15.0) -> float:
    """Скорость звука в сухом воздухе, м/с."""
    return float(331.3 * np.sqrt(1.0 + temp_c / 273.15))


def haversine_m(lat1, lon1, lat2, lon2):
    """Расстояние по большому кругу, м."""
    p1, p2 = np.deg2rad(lat1), np.deg2rad(lat2)
    dp = p2 - p1
    dl = np.deg2rad(np.asarray(lon2) - np.asarray(lon1))
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * R_EARTH * np.arcsin(np.sqrt(a))
