"""Физика: распространение звука, эффект Доплера, точка максимального сближения (CPA).

Обозначения: источник (самолёт) движется, наблюдатель неподвижен.
  t_e — время излучения, t_a = t_e + r(t_e)/c — время прихода к наблюдателю,
  v_r — проекция скорости самолёта на направление «самолёт → наблюдатель»,
  f_rx = f0 · c / (c − v_r)  (приближение — f_rx > f0, удаление — f_rx < f0).
«Задержанное время CPA» T = t_cpa + d/c — момент, когда на записи f_rx = f0.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .adsb import Track


def received_curve(track: Track, obs: np.ndarray, c: float):
    """Для каждой точки трека: время прихода, отношение f_rx/f0 и дистанция."""
    d = np.asarray(obs, dtype=float)[None, :] - track.pos
    r = np.sqrt((d * d).sum(axis=1))
    v_r = (track.vel * d).sum(axis=1) / r
    return track.t + r / c, 1.0 / (1.0 - v_r / c), r


def straight_pass_ratio(t_a, v: float, d: float, t_c: float, c: float):
    """f_rx/f0 для прямолинейного равномерного пролёта (замкнутая формула).

    v — скорость, d — наклонная дистанция CPA, t_c — время излучения в CPA.
    Из (t_a − t_e)·c = sqrt(d² + v²(t_e − t_c)²) получаем квадратное уравнение
    относительно τ = t_e − t_c и берём меньший корень (причинный).
    """
    A = np.asarray(t_a, dtype=float) - t_c
    c2, v2 = c * c, v * v
    tau = (c2 * A - np.sqrt(v2 * c2 * A * A + (c2 - v2) * d * d)) / (c2 - v2)
    s = v * tau
    r = np.hypot(d, s)
    return 1.0 / (1.0 + v * s / (r * c))


@dataclass
class CPAResult:
    point: np.ndarray     # индексы точек сетки
    T: np.ndarray         # задержанное время CPA (unix), когда f_rx = f0
    D: np.ndarray         # наклонная дистанция CPA, м
    V: np.ndarray         # скорость самолёта в CPA, м/с
    interior: np.ndarray  # минимум внутри участка (а не на его краю)
    artifact: np.ndarray  # минимум на искусственном разрезе участка — не настоящий CPA


def cpa_near(points: np.ndarray, track: Track, c: float, max_dist: float,
             chunk_elems: int = 4_000_000) -> CPAResult:
    """CPA трека для всех точек сетки ближе max_dist (с параболическим уточнением)."""
    P = track.pos
    M = len(P)
    lo = P.min(axis=0) - max_dist
    hi = P.max(axis=0) + max_dist
    sel = np.where((points[:, 0] >= lo[0]) & (points[:, 0] <= hi[0]) &
                   (points[:, 1] >= lo[1]) & (points[:, 1] <= hi[1]))[0]
    parts = []
    if len(sel) and M >= 2:
        p2 = (P * P).sum(axis=1)
        step = max(1, chunk_elems // M)
        for s in range(0, len(sel), step):
            ii = sel[s:s + step]
            O = points[ii]
            d2 = (O * O).sum(axis=1)[:, None] + p2[None, :] - 2.0 * (O @ P.T)
            j = d2.argmin(axis=1)
            jm, jp = np.clip(j - 1, 0, M - 1), np.clip(j + 1, 0, M - 1)
            e0 = ((O - P[j]) ** 2).sum(axis=1)
            em = ((O - P[jm]) ** 2).sum(axis=1)
            ep = ((O - P[jp]) ** 2).sum(axis=1)
            a = 0.5 * (em + ep - 2.0 * e0)
            b = 0.5 * (ep - em)
            inner = (j > 0) & (j < M - 1) & (a > 0)
            x = np.where(inner, -b / np.where(a > 0, 2.0 * a, 1.0), 0.0).clip(-1.0, 1.0)
            dmin2 = np.maximum(e0 + b * x + a * x * x, 0.0)
            D = np.sqrt(dmin2)
            keep = D <= max_dist
            t_e = track.t[j] + x * track.dt
            V = np.linalg.norm(track.vel[j], axis=1)
            artifact = ((j == 0) & track.open_start) | ((j == M - 1) & track.open_end)
            parts.append((ii[keep], (t_e + D / c)[keep], D[keep], V[keep],
                          inner[keep], artifact[keep]))
    if not parts:
        e = np.empty(0)
        return CPAResult(np.empty(0, int), e, e, e, np.empty(0, bool), np.empty(0, bool))
    cols = [np.concatenate(z) for z in zip(*parts)]
    return CPAResult(*cols)
