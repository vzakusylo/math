"""Синтез звука прямолинейного пролёта источника мимо неподвижного наблюдателя.

Учебная модель. Источник (винтовой самолёт) летит по прямой на постоянной высоте
с постоянной скоростью. Наблюдатель стоит на земле со смещением по горизонтали,
так что наклонная дистанция в точке наибольшего сближения (CPA) равна `dist`.

Эффекты, которые воспроизводим:
  * доплеровский сдвиг частоты для движущегося источника;
  * запаздывание прихода звука (то, что слышно «сейчас», излучено раньше);
  * спад громкости ~1/r и поглощение высоких частот с расстоянием;
  * несколько гармоник гула винта + широкополосный шум фона.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geo import speed_of_sound


@dataclass
class FlyoverTruth:
    """Истинные параметры пролёта — чтобы потом сверить с восстановленными."""
    f0: float        # излучаемая основная частота, Гц
    speed: float     # м/с
    dist: float      # наклонная дистанция в CPA, м
    t_cpa_emit: float  # момент излучения в CPA, с
    t_cpa_recv: float  # момент прихода звука из CPA (когда слышимая частота = f0), с
    c: float         # скорость звука, м/с


def synth_flyover(duration: float = 90.0, sr: int = 8000, f0: float = 95.0,
                  speed: float = 70.0, dist: float = 600.0, t_cpa_emit: float | None = None,
                  harmonics=(1.0, 0.55, 0.3, 0.16, 0.08), temp_c: float = 15.0,
                  noise_db: float = -26.0, seed: int = 0):
    """Возвращает (audio, sr, truth).

    Звук строим по времени ПРИХОДА. Для каждого отсчёта приёмника t_a решаем, когда
    он был излучён (t_e), из условия t_a = t_e + r(t_e)/c, где r — расстояние до источника.
    Для равномерного прямолинейного движения это квадратное уравнение относительно t_e.
    Мгновенная фаза берётся как интеграл излучаемой частоты до t_e, поэтому доплеровский
    сдвиг получается «бесплатно», физически корректно, без приближений.
    """
    rng = np.random.default_rng(seed)
    c = speed_of_sound(temp_c)
    if t_cpa_emit is None:
        t_cpa_emit = duration / 2.0 - dist / c  # чтобы CPA по приёму был около середины

    t_a = np.arange(int(round(duration * sr))) / sr

    # t_e(t_a): источник на прямой x(t) = v*(t − t_cpa_emit), y = dist (CPA при t=t_cpa_emit).
    # r(t_e)^2 = dist^2 + v^2 (t_e − t_cpa_emit)^2 ;  (t_a − t_e) c = r(t_e).
    # Подстановка τ = t_e − t_cpa_emit, A = t_a − t_cpa_emit даёт:
    #   (c^2 − v^2) τ^2 − 2 c^2 A τ + (c^2 A^2 − dist^2) = 0.
    v = speed
    A = t_a - t_cpa_emit
    a = c * c - v * v
    b = -2.0 * c * c * A
    cc = c * c * A * A - dist * dist
    disc = np.maximum(b * b - 4 * a * cc, 0.0)
    tau = (-b - np.sqrt(disc)) / (2 * a)   # причинный корень: t_e < t_a
    t_e = tau + t_cpa_emit
    s = v * tau                             # координата источника вдоль трассы
    r = np.hypot(dist, s)

    # фаза = 2π f0 ∫ dt_e ; шаг t_e между соседними отсчётами приёмника неравномерен
    dte = np.gradient(t_e, t_a)
    phase = 2 * np.pi * f0 * np.cumsum(dte) / sr

    # затухание: геометрическое 1/r (нормировано на CPA) и «мягкий» ВЧ-спад с расстоянием
    amp = dist / np.maximum(r, 1.0)
    sig = np.zeros_like(t_a)
    for k, g in enumerate(harmonics, start=1):
        hf_roll = np.exp(-(k - 1) * (r - dist) / 4000.0)  # дальше — глуше обертоны
        sig += g * hf_roll * np.sin(k * phase)
    sig *= amp

    rms = np.sqrt(np.mean(sig ** 2)) + 1e-12
    sig = sig / rms
    noise = rng.standard_normal(sig.shape) * (10 ** (noise_db / 20.0))
    out = sig + noise
    out = 0.9 * out / np.max(np.abs(out))

    truth = FlyoverTruth(f0=f0, speed=speed, dist=dist, t_cpa_emit=t_cpa_emit,
                         t_cpa_recv=t_cpa_emit + dist / c, c=c)
    return out.astype(np.float64), sr, truth
