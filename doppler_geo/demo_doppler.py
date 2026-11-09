"""Учебная демонстрация доплеровской части.

Синтезируем звук прямолинейного пролёта самолёта (с шумом, затуханием, гармониками),
а затем ТОЛЬКО по этому звуку восстанавливаем:
    * излучаемую частоту гула винта f0,
    * скорость самолёта,
    * наклонную дистанцию в точке наибольшего сближения,
    * момент пролёта.
И сравниваем с истинными значениями, с которыми синтезировали.

Запуск:  py demo_doppler.py
Результат: таблица «истина / оценка / ошибка» в консоли и график doppler_demo.png.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from aerodoppler.geo import speed_of_sound
from aerodoppler.spectro import compute_spectrogram, fit_pass, refine_centroid, track_ridge
from aerodoppler.synth import synth_flyover


def recover(audio, sr, c, window):
    """Аудио → спектрограмма → кривая → параметры пролёта."""
    # Окно 1 с даёт разрешение ~1 Гц по частоте; быстрый сдвиг у точки пролёта затем
    # аккуратно уточняется центроидом (refine_centroid), поэтому короткое окно не нужно.
    spec = compute_spectrogram(audio, sr, win_s=1.0, hop_s=0.25, fmin=20.0, fmax=400.0,
                               zero_pad=2)
    t0, t1, fmin, fmax = window
    rt, rf, rw = track_ridge(spec, t0, t1, fmin, fmax,
                             max_rise_hz_s=1.0, max_fall_hz_s=40.0)
    rf = refine_centroid(spec, rt, rf, half_hz=5.0)
    fit = fit_pass(rt, rf, rw, c)
    return spec, (rt, rf, rw), fit


def report(truth, fit):
    rows = [
        ("осн. частота f0, Гц", truth.f0, fit.f0, fit.sig_f0),
        ("скорость, м/с",       truth.speed, fit.speed, fit.sig_speed),
        ("скорость, км/ч",      truth.speed * 3.6, fit.speed * 3.6, fit.sig_speed * 3.6),
        ("дистанция CPA, м",    truth.dist, fit.dist, fit.sig_dist),
        ("момент пролёта, с",   truth.t_cpa_recv, fit.t_cpa, fit.sig_t),
    ]
    w = 22
    print(f"\n{'параметр':<{w}}{'истина':>12}{'оценка':>12}{'±σ':>10}{'ошибка':>12}")
    print("-" * (w + 46))
    for name, tru, est, sig in rows:
        err = est - tru
        rel = f"{100 * err / tru:+.1f}%" if abs(tru) > 1e-6 else ""
        print(f"{name:<{w}}{tru:>12.2f}{est:>12.2f}{sig:>10.2f}{err:>12.2f}  {rel}")
    print(f"\nневязка подгонки кривой: RMS = {fit.rms_hz:.2f} Гц по {fit.n_points} точкам")


def make_plot(spec, ridge, truth, fit, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rt, rf, rw = ridge
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 9), height_ratios=[2, 1])

    extent = [spec.t[0], spec.t[-1], spec.f[0], spec.f[-1]]
    ax1.imshow(spec.B, origin="lower", aspect="auto", extent=extent, cmap="magma",
               vmin=0, vmax=spec.clip_db)
    ax1.plot(rt, rf, ".", ms=3, color="cyan", alpha=0.5, label="найденная кривая")
    tt = np.linspace(rt[0], rt[-1], 400)
    ax1.plot(tt, fit.model(tt, truth.c), "-", color="lime", lw=1.6, label="подгонка (прямой пролёт)")
    ax1.axhline(fit.f0, color="white", ls="--", lw=0.8, alpha=0.7)
    ax1.axvline(fit.t_cpa, color="white", ls=":", lw=0.8, alpha=0.7)
    ax1.set(xlabel="время, с", ylabel="частота, Гц",
            title="Спектрограмма пролёта и доплеровская кривая", ylim=(spec.f[0], 300))
    ax1.legend(loc="upper right", framealpha=0.3, labelcolor="white")

    f_model = fit.model(rt, truth.c)
    ax2.plot(rt, rf - fit.f0, ".", ms=3, color="tab:blue", label="данные − f0")
    ax2.plot(rt, f_model - fit.f0, "-", color="tab:red", lw=1.4, label="модель − f0")
    ax2.axhline(0, color="k", lw=0.6)
    ax2.axvline(truth.t_cpa_recv, color="green", ls="--", lw=1, label="истинный CPA")
    ax2.set(xlabel="время, с", ylabel="сдвиг частоты, Гц",
            title="Доплеровский сдвиг: подъём на подлёте, спад на отлёте")
    ax2.legend(loc="upper right", framealpha=0.5)
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    print(f"\nграфик сохранён: {path}")


def main():
    c = speed_of_sound(15.0)
    # Истинные параметры (демо): лёгкий винтовой самолёт.
    audio, sr, truth = synth_flyover(duration=90.0, sr=8000, f0=95.0, speed=72.0,
                                     dist=550.0, noise_db=-24.0, seed=7)
    print(f"скорость звука c = {c:.1f} м/с")
    print(f"синтез: f0={truth.f0} Гц, v={truth.speed} м/с ({truth.speed*3.6:.0f} км/ч), "
          f"CPA={truth.dist} м, пролёт на {truth.t_cpa_recv:.1f} с")

    # Окно поиска кривой: весь файл по времени, широкий диапазон частот основного тона.
    spec, ridge, fit = recover(audio, sr, c, window=(2.0, 88.0, 60.0, 160.0))
    report(truth, fit)

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "doppler_demo.png")
    make_plot(spec, ridge, truth, fit, out)


if __name__ == "__main__":
    main()
