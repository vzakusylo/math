"""Загрузка ADS-B треков и подготовка их к поиску.

Поддерживается CSV в стиле OpenSky state vectors (time, icao24, lat, lon, geoaltitude, ...)
и похожие форматы. Нужные колонки ищутся по нескольким типичным именам.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
from scipy.ndimage import uniform_filter1d

from .geo import LocalFrame

FT = 0.3048

# Префиксы ICAO-кодов типов реактивных самолётов (эвристика, можно расширять).
# Винтовые (C172, PA28, DHC6, AT72, BE20, PC12, ...) сюда не попадают.
JET_TYPE_PREFIXES = (
    "A19", "A20", "A21", "A22", "A30", "A31", "A32", "A33", "A34", "A35", "A38", "A3ST",
    "B37", "B38", "B39", "B71", "B72", "B73", "B74", "B75", "B76", "B77", "B78", "BCS",
    "CRJ", "E13", "E14", "E17", "E19", "E29", "E75", "E35L", "E50P", "E55P", "E545", "E550",
    "MD8", "MD9", "DC9", "DC10", "F70", "F100", "RJ", "B46", "A148", "SU95", "T204",
    "C25", "C50", "C51", "C52", "C55", "C56", "C65", "C68", "C70", "C75",
    "GLF", "GL5", "GL6", "GL7", "GLEX", "CL30", "CL35", "CL60", "LJ", "F2TH", "FA",
    "F900", "PC24", "HDJT", "SF50", "H25", "PRM1", "BE40", "G150", "G280", "ASTR",
)

_TIME = ("time", "timestamp", "ts", "unix_time", "datetime")
_ICAO = ("icao24", "icao", "hex", "icao_address")
_CALL = ("callsign", "flight", "call_sign")
_TYPE = ("typecode", "type", "aircraft_type", "icaotype", "type_code")
_LAT = ("lat", "latitude")
_LON = ("lon", "lng", "long", "longitude")
_ALT = ("geoaltitude", "geo_altitude", "alt_geom", "geometric_altitude",
        "altitude", "baroaltitude", "alt_baro", "alt")
_GROUND = ("onground", "on_ground", "ground")


def is_jet(typecode: str) -> bool:
    tc = (typecode or "").strip().upper()
    return bool(tc) and tc.startswith(JET_TYPE_PREFIXES)


def parse_time(value: str | float) -> float:
    """unix-секунды или ISO-8601 → unix-секунды (UTC)."""
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip()
    try:
        return float(s)
    except ValueError:
        pass
    dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def format_time(unix: float) -> str:
    return datetime.fromtimestamp(unix, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")[:-5] + "Z"


@dataclass
class Flight:
    icao24: str
    callsign: str
    typecode: str
    t: np.ndarray    # unix-время, по возрастанию
    lat: np.ndarray
    lon: np.ndarray
    alt: np.ndarray  # м над уровнем моря

    @property
    def label(self) -> str:
        return f"{self.callsign or '?'} [{self.icao24}, {self.typecode or '?'}]"


@dataclass
class Track:
    """Участок траектории, равномерно передискретизированный в локальных координатах."""
    flight_idx: int
    t: np.ndarray     # (M,) unix-время излучения звука
    pos: np.ndarray   # (M, 3) м: x восток, y север, z высота
    vel: np.ndarray   # (M, 3) м/с
    open_start: bool  # начало участка — искусственный разрез (а не начало данных)
    open_end: bool

    @property
    def dt(self) -> float:
        return float(self.t[1] - self.t[0])


def _pick(header: list[str], names: tuple[str, ...]) -> str | None:
    low = {h.lower().strip(): h for h in header}
    for n in names:
        if n in low:
            return low[n]
    return None


def load_aircraft_db(path: str) -> dict[str, str]:
    """CSV с колонками icao24 и typecode (например, база OpenSky aircraftDatabase.csv)."""
    out = {}
    with open(path, newline="", encoding="utf-8", errors="replace") as fh:
        rd = csv.DictReader(fh)
        ci, ct = _pick(rd.fieldnames, _ICAO), _pick(rd.fieldnames, _TYPE)
        if not ci or not ct:
            raise ValueError(f"{path}: нужны колонки icao24 и typecode")
        for row in rd:
            if row[ct]:
                out[row[ci].strip().lower()] = row[ct].strip().upper()
    return out


def load_adsb(path: str, alt_unit: str = "m", aircraft_db: dict[str, str] | None = None,
              max_gap_s: float = 120.0) -> list[Flight]:
    """Читает CSV, группирует по icao24 и режет треки на разрывах > max_gap_s."""
    alt_k = FT if alt_unit == "ft" else 1.0
    groups: dict[str, list] = {}
    meta: dict[str, tuple[str, str]] = {}
    with open(path, newline="", encoding="utf-8", errors="replace") as fh:
        rd = csv.DictReader(fh)
        h = rd.fieldnames or []
        ct, ci, cc, cy = _pick(h, _TIME), _pick(h, _ICAO), _pick(h, _CALL), _pick(h, _TYPE)
        cla, clo, cal, cg = _pick(h, _LAT), _pick(h, _LON), _pick(h, _ALT), _pick(h, _GROUND)
        missing = [n for n, c in (("time", ct), ("icao24", ci), ("lat", cla), ("lon", clo),
                                  ("altitude", cal)) if c is None]
        if missing:
            raise ValueError(f"{path}: не найдены колонки {missing}; есть {h}")
        for row in rd:
            try:
                lat, lon, alt = float(row[cla]), float(row[clo]), float(row[cal])
                t = parse_time(row[ct])
            except (TypeError, ValueError):
                continue
            if cg and str(row[cg]).strip().lower() in ("true", "1", "yes"):
                continue
            icao = row[ci].strip().lower()
            groups.setdefault(icao, []).append((t, lat, lon, alt * alt_k))
            call = row[cc].strip() if cc and row[cc] else ""
            typ = row[cy].strip().upper() if cy and row[cy] else ""
            old = meta.get(icao, ("", ""))
            meta[icao] = (call or old[0], typ or old[1])

    flights = []
    for icao, rows in groups.items():
        a = np.array(sorted(rows))
        _, uniq = np.unique(a[:, 0], return_index=True)
        a = a[uniq]
        call, typ = meta[icao]
        if not typ and aircraft_db:
            typ = aircraft_db.get(icao, "")
        cuts = np.where(np.diff(a[:, 0]) > max_gap_s)[0] + 1
        for seg in np.split(a, cuts):
            if len(seg) >= 3:
                flights.append(Flight(icao, call, typ, seg[:, 0], seg[:, 1], seg[:, 2], seg[:, 3]))
    return flights


def make_tracks(flights: list[Flight], frame: LocalFrame, t_min: float | None = None,
                t_max: float | None = None, dt: float = 1.0, piece_len: float = 600.0,
                piece_step: float = 300.0, smooth_s: float = 8.0) -> list[Track]:
    """Передискретизация в 1 с, сглаживание шума ADS-B, нарезка на перекрывающиеся участки.

    Каждый участок даёт свою точку максимального сближения (CPA), поэтому самолёт,
    который кружит и пролетает рядом дважды, тоже обрабатывается правильно.
    """
    tracks = []
    for fi, f in enumerate(flights):
        t0 = f.t[0] if t_min is None else max(f.t[0], t_min)
        t1 = f.t[-1] if t_max is None else min(f.t[-1], t_max)
        if t1 - t0 < 3 * dt:
            continue
        x, y = frame.to_xy(f.lat, f.lon)
        tg = np.arange(t0, t1 + 1e-9, dt)
        pos = np.column_stack([np.interp(tg, f.t, x), np.interp(tg, f.t, y), np.interp(tg, f.t, f.alt)])
        size = max(1, int(round(smooth_s / dt)))
        if size > 1 and len(tg) > size:
            pos = uniform_filter1d(pos, size=size, axis=0, mode="nearest")
        vel = np.gradient(pos, dt, axis=0)
        n, L, S = len(tg), int(piece_len / dt), int(piece_step / dt)
        start = 0
        while True:
            end = min(start + L, n)
            tracks.append(Track(fi, tg[start:end], pos[start:end], vel[start:end],
                                open_start=start > 0 or t0 > f.t[0],
                                open_end=end < n or t1 < f.t[-1]))
            if end >= n:
                break
            start += S
    return tracks
