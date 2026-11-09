"""Аудио → спектрограмма → доплеровская кривая → параметры пролёта."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from math import ceil, gcd

import numpy as np
from scipy import signal
from scipy.io import wavfile
from scipy.optimize import least_squares

from .physics import straight_pass_ratio


def load_audio(path: str, sr_out: int = 2000):
    """WAV → моно float, передискретизация до sr_out (гул винтов — это десятки–сотни Гц).

    Для видео сначала извлеките звук:  ffmpeg -i video.mp4 -ac 1 -ar 8000 audio.wav
    """
    sr, x = wavfile.read(path)
    if np.issubdtype(x.dtype, np.integer):
        x = x.astype(np.float64) / float(np.iinfo(x.dtype).max)
    else:
        x = x.astype(np.float64)
    if x.ndim == 2:
        x = x.mean(axis=1)
    if sr != sr_out:
        g = gcd(int(sr), int(sr_out))
        x = signal.resample_poly(x, sr_out // g, int(sr) // g)
    return x, sr_out


@dataclass
class Spectrogram:
    t: np.ndarray    # центры кадров, с от начала записи
    f: np.ndarray    # Гц
    db: np.ndarray   # (F, T) «отбеленная» мощность, дБ над медианой
    clip_db: float = 20.0

    def __post_init__(self):
        self.hop = float(self.t[1] - self.t[0])
        self.df = float(self.f[1] - self.f[0])
        self.B = np.clip(self.db, 0.0, self.clip_db)  # «яркость» для поиска и скоринга

    def frame_index(self, t):
        return np.rint((np.asarray(t, dtype=float) - self.t[0]) / self.hop).astype(int)

    def freq_index(self, f):
        return (np.asarray(f, dtype=float) - self.f[0]) / self.df

    def sample(self, fi, ti):
        """Яркость в (дробный индекс частоты, целый индекс кадра); вне диапазона → 0."""
        fi, ti = np.broadcast_arrays(fi, ti)
        F, T = self.B.shape
        ok = np.isfinite(fi) & (fi >= 0) & (fi <= F - 1) & (ti >= 0) & (ti < T)
        fc = np.where(ok, fi, 0.0)
        tc = np.where(ok, ti, 0)
        f0 = np.minimum(np.floor(fc).astype(int), F - 2)
        w = fc - f0
        v = self.B[f0, tc] * (1.0 - w) + self.B[f0 + 1, tc] * w
        return np.where(ok, v, 0.0)


def compute_spectrogram(x: np.ndarray, sr: int, win_s: float = 1.0, hop_s: float = 0.25,
                        fmin: float = 20.0, fmax: float = 500.0, zero_pad: int = 2) -> Spectrogram:
    """STFT с длинным окном (разрешение ~1 Гц) и двойным «отбеливанием».

    - вычитание медианы по времени в каждой частоте убирает постоянные тона (сеть 50/60 Гц,
      вентиляторы) и цвет шума;
    - вычитание медианы по частоте в каждом кадре убирает широкополосные всплески (ветер, удары).
    Узкая медленно плывущая линия самолёта при этом остаётся.
    """
    nper = int(round(win_s * sr))
    hop = int(round(hop_s * sr))
    f, t, Z = signal.stft(x, fs=sr, window="hann", nperseg=nper, noverlap=nper - hop,
                          nfft=nper * zero_pad, boundary=None, padded=False)
    sel = (f >= fmin) & (f <= fmax)
    P = 10.0 * np.log10(np.abs(Z[sel]) ** 2 + 1e-20)
    P -= np.median(P, axis=1, keepdims=True)
    P -= np.median(P, axis=0, keepdims=True)
    return Spectrogram(t, f[sel], P)


def track_ridge(spec: Spectrogram, t0: float, t1: float, fmin: float, fmax: float,
                max_rise_hz_s: float = 0.5, max_fall_hz_s: float = 8.0, jump_penalty: float = 0.4):
    """Самая яркая непрерывная кривая в окне (алгоритм Витерби).

    Доплеровская кривая пролёта монотонно убывает, поэтому подъём частоты почти запрещён,
    а спуск ограничен разумной скоростью.
    """
    ti = np.where((spec.t >= t0) & (spec.t <= t1))[0]
    fi = np.where((spec.f >= fmin) & (spec.f <= fmax))[0]
    if len(ti) < 5 or len(fi) < 3:
        raise ValueError("слишком маленькое окно для поиска кривой")
    B = spec.B[np.ix_(fi, ti)]
    F, T = B.shape
    up = max(1, ceil(max_rise_hz_s * spec.hop / spec.df))
    dn = max(1, ceil(max_fall_hz_s * spec.hop / spec.df))
    shifts = np.arange(-dn, up + 1)  # j_new − j_prev
    acc = B[:, 0].copy()
    back = np.zeros((F, T), dtype=np.int32)
    jj = np.arange(F)
    for i in range(1, T):
        best = np.full(F, -np.inf)
        arg = np.zeros(F, dtype=np.int32)
        for s in shifts:
            prev = jj - s
            ok = (prev >= 0) & (prev < F)
            cand = np.full(F, -np.inf)
            cand[ok] = acc[prev[ok]] - jump_penalty * abs(s)
            better = cand > best
            best[better] = cand[better]
            arg[better] = prev[better]
        acc = best + B[:, i]
        back[:, i] = arg
    path = np.empty(T, dtype=int)
    path[-1] = int(np.argmax(acc))
    for i in range(T - 1, 0, -1):
        path[i - 1] = back[path[i], i]
    # субдискретное уточнение частоты параболой по соседним бинам
    jm, jp = np.clip(path - 1, 0, F - 1), np.clip(path + 1, 0, F - 1)
    cols = np.arange(T)
    ym, y0, yp = B[jm, cols], B[path, cols], B[jp, cols]
    den = ym - 2 * y0 + yp
    off = np.where(den < 0, 0.5 * (ym - yp) / np.where(den < 0, den, -1.0), 0.0).clip(-0.5, 0.5)
    freqs = spec.f[fi][path] + off * spec.df
    return spec.t[ti], freqs, y0


def refine_centroid(spec: Spectrogram, t: np.ndarray, f: np.ndarray, half_hz: float = 5.0):
    """Уточнение частоты кривой интенсивностно-взвешенным центроидом в полосе ±half_hz.

    Пик спектрограммы (argmax) у быстро «плывущего» тона смещён из-за асимметрии спектрального
    «растекания» при оконном преобразовании. Центр тяжести яркости в узкой полосе вокруг
    найденной кривой оценивает мгновенную частоту гораздо точнее, что критично:
    ошибка даже в 1 Гц на краях кривой заметно искажает восстановленные скорость и дистанцию.
    """
    ti = spec.frame_index(t)
    hb = int(round(half_hz / spec.df))
    F = spec.B.shape[0]
    out = f.copy()
    for k in range(len(f)):
        j0 = int(round(spec.freq_index(f[k])))
        lo, hi = max(0, j0 - hb), min(F - 1, j0 + hb)
        w = np.clip(spec.B[lo:hi + 1, ti[k]], 0.0, None)
        s = w.sum()
        if s > 0:
            out[k] = float(np.dot(w, spec.f[lo:hi + 1]) / s)
    return out


@dataclass
class PassFit:
    f0: float          # излучаемая частота, Гц
    speed: float       # м/с
    dist: float        # наклонная дистанция CPA, м
    t_cpa: float       # задержанное время CPA на записи, с (когда f = f0)
    sig_f0: float
    sig_speed: float
    sig_dist: float
    sig_t: float
    rms_hz: float
    n_points: int
    ridge_t: np.ndarray = field(repr=False)
    ridge_f: np.ndarray = field(repr=False)
    used: np.ndarray = field(repr=False)

    def model(self, t, c):
        return self.f0 * straight_pass_ratio(t, self.speed, self.dist, self.t_cpa - self.dist / c, c)


def fit_pass(t: np.ndarray, f: np.ndarray, w: np.ndarray, c: float) -> PassFit:
    """Подгонка f(t) = f0 · c/(c − v_r(t)) для прямолинейного пролёта.

    Асимптоты кривой дают f0 и скорость (f_hi/f_lo = (c+v)/(c−v)),
    крутизна в центре — дистанцию CPA, положение центра — время CPA.
    """
    keep = w > 0.15 * np.percentile(w, 90)
    t, f, w = t[keep], f[keep], w[keep]
    if len(t) < 8:
        raise ValueError("кривая слишком короткая или слабая")
    q = max(3, len(t) // 10)
    f_hi, f_lo = np.median(f[:q]), np.median(f[-q:])
    f0_i = 2 * f_hi * f_lo / (f_hi + f_lo)
    v_i = float(np.clip(c * (f_hi - f_lo) / (f_hi + f_lo), 25, 0.9 * c))
    T_i = float(t[np.argmin(np.abs(f - f0_i))])
    sw = np.sqrt(w / w.max())

    def resid(p, tt, ff, ww):
        f0, v, d, T = p
        return ww * (f0 * straight_pass_ratio(tt, v, d, T - d / c, c) - ff)

    lb = [1.0, 5.0, 30.0, t.min() - 600]
    ub = [5000.0, 0.95 * c, 50_000.0, t.max() + 600]
    best = None
    for d0 in (300.0, 800.0, 2000.0, 5000.0):
        for v0 in (v_i, 0.7 * v_i, 1.3 * v_i):
            p0 = np.clip([f0_i, v0, d0, T_i], np.add(lb, 1e-6), np.subtract(ub, 1e-6))
            r = least_squares(resid, p0, args=(t, f, sw), bounds=(lb, ub), loss="soft_l1", f_scale=0.5)
            if best is None or r.cost < best.cost:
                best = r
    # отбрасываем выбросы (захват речи/помех) и уточняем обычными наименьшими квадратами
    res = resid(best.x, t, f, np.ones_like(t))
    rms = np.sqrt(np.mean(res ** 2))
    used = np.abs(res) < max(3 * rms, 1.0)
    r = least_squares(resid, best.x, args=(t[used], f[used], sw[used]), bounds=(lb, ub))
    J = r.jac
    dof = max(1, used.sum() - 4)
    s2 = 2 * r.cost / dof
    cov = np.linalg.pinv(J.T @ J) * s2
    sig = np.sqrt(np.clip(np.diag(cov), 0, None))
    res_u = resid(r.x, t[used], f[used], np.ones(used.sum()))
    used_full = np.zeros(keep.shape, bool)
    used_full[np.where(keep)[0][used]] = True
    return PassFit(*map(float, r.x), *map(float, sig),
                   rms_hz=float(np.sqrt(np.mean(res_u ** 2))), n_points=int(used.sum()),
                   ridge_t=t, ridge_f=f, used=used)


def extract_event(spec: Spectrogram, window, c: float, rel_speed: float = 0.25,
                  dist_factor: float = 1.8, **ridge_kw):
    """Окно (t0, t1, fmin, fmax) → событие с консервативными диапазонами скорости и дистанции.

    Допуски широкие: модель прямолинейного пролёта, ветер и шум ADS-B вносят ошибки,
    а ADS-B даёт путевую скорость, а не воздушную.
    """
    from .locate import Event

    t0, t1, fmin, fmax = window
    rt, rf, rw = track_ridge(spec, t0, t1, fmin, fmax, **ridge_kw)
    fit = fit_pass(rt, rf, rw, c)
    v_lo = min(fit.speed * (1 - rel_speed), fit.speed - 3 * fit.sig_speed)
    v_hi = max(fit.speed * (1 + rel_speed), fit.speed + 3 * fit.sig_speed)
    d_lo = min(fit.dist / dist_factor, fit.dist - 3 * fit.sig_dist)
    d_hi = max(fit.dist * dist_factor, fit.dist + 3 * fit.sig_dist)
    ev = Event(t_cpa=fit.t_cpa, f0=fit.f0, speed=fit.speed, dist=fit.dist,
               speed_range=(max(5.0, v_lo), v_hi), dist_range=(max(30.0, d_lo), d_hi),
               window=tuple(float(v) for v in window),
               fit={k: v for k, v in asdict(fit).items() if not isinstance(v, np.ndarray)})
    return ev, fit
