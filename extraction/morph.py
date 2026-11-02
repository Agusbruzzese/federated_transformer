"""The 19 morphology scalars of workstream D (features.morph_feats) on the TRUE time base.

Port of research/extraction/D_features/features.py::morph_feats (sha256 68e7db3b...), with these changes:
  E1. fs is the event's effective output rate (fs_local*25/1208, ~50 Hz) instead of D's nominal 100 Hz
      (which itself assumed 2400 Hz, -0.69 %). All sample counts -> seconds use it.
  E2. The 5 Hz smoothing Butterworth is designed at that fs. D's module-level _SOS_LP5 is fixed for 100 Hz;
      calling D's function with fs=50 would silently smooth at 2.5 Hz.
  E3. log_energy = log10(sum(x^2)/fs + 1)  [counts^2 * s, a time integral]. D used log10(sum(x^2) + 1) at
      100 Hz, i.e. D's value ~= this + log10(100.67) = this + 2.003. log_area_abs was already a time
      integral in D (sum/fs) and is unchanged.
  E4. Guard: if 3*sigma_edge exceeds the peak (no sample above thr10), i0 = i1 = peak index (D: IndexError).
Everything else (thresholds, lobe rule, edge noise estimate) is identical to D.
"""
from __future__ import annotations

import numpy as np
from scipy import signal
from scipy.stats import kurtosis

MORPH_NAMES = ["log_amp", "polarity", "pos_frac", "neg_frac", "dur10", "dur50_main", "rise_main",
               "fall_main", "asym", "n_lobes", "first_lobe_sign", "t_pp", "area_pos_n", "area_neg_n",
               "log_area_abs", "log_energy", "t_spread", "zc_rate", "kurt_env"]
TIME_FEATS = ["dur10", "dur50_main", "rise_main", "fall_main", "t_pp", "t_spread"]   # seconds
RATE_FEATS = ["zc_rate"]                                                             # 1/s

_SOS = {}


def _sos_lp5(fs):
    k = float(fs)                     # exact key: a rounded key made results depend on cache order (~1e-11)
    if k not in _SOS:
        _SOS[k] = signal.butter(4, 5.0, fs=fs, output="sos")
    return _SOS[k]


def _cross(y, i0, thr, step):
    i = i0
    while 0 <= i + step < len(y) and y[i + step] >= thr:
        i += step
    return i


def morph_feats(x: np.ndarray, fs: float, edge_s: float = 1.5) -> np.ndarray:
    x = signal.sosfiltfilt(_sos_lp5(fs), np.asarray(x, float))
    ne = int(edge_s * fs)
    edge = np.r_[x[:ne], x[-ne:]]
    sig = 1.4826 * np.median(np.abs(np.diff(edge))) / np.sqrt(2) + 1e-9
    ax = np.abs(x)
    ip = int(np.argmax(ax))
    A = ax[ip] + 1e-9
    pol = np.sign(x[ip])
    thr10 = max(0.1 * A, 3 * sig)
    above = np.where(ax > thr10)[0]
    i0, i1 = (above[0], above[-1]) if len(above) else (ip, ip)          # E4
    dur10 = (i1 - i0 + 1) / fs
    y = x * pol
    l50, r50 = _cross(y, ip, 0.5 * A, -1), _cross(y, ip, 0.5 * A, +1)
    l10, r10 = _cross(y, ip, thr10, -1), _cross(y, ip, thr10, +1)
    l90, r90 = _cross(y, ip, 0.9 * A, -1), _cross(y, ip, 0.9 * A, +1)
    rise = (l90 - l10) / fs
    fall = (r10 - r90) / fs
    asym = (ip - i0) / max(i1 - i0, 1)
    thr20 = max(0.2 * A, 5 * sig)
    s = np.where(x > thr20, 1, np.where(x < -thr20, -1, 0))
    lab = []
    run_s, run_n = 0, 0
    for v in np.r_[s, 0]:
        if v == run_s:
            run_n += 1
            continue
        if run_s != 0 and run_n >= int(0.1 * fs) and (not lab or lab[-1] != run_s):
            lab.append(run_s)
        run_s, run_n = v, 1
    lobes = len(lab)
    first = lab[0] if lab else 0
    seg = x[i0:i1 + 1]
    t_pp = (np.argmax(x) - np.argmin(x)) / fs
    ap = seg[seg > 0].sum() / fs
    an = -seg[seg < 0].sum() / fs
    norm = A * dur10 + 1e-9
    e = x ** 2
    tc = np.sum(np.arange(len(x)) * e) / e.sum()
    tsp = np.sqrt(np.sum((np.arange(len(x)) - tc) ** 2 * e) / e.sum()) / fs
    zc = np.sum(np.diff(np.sign(seg - 0.0)) != 0) / dur10
    return np.array([np.log10(A), pol, x.max() / A, -x.min() / A, dur10, (r50 - l50) / fs, rise, fall,
                     asym, lobes, first, t_pp, ap / norm, an / norm, np.log10(ap + an + 1),
                     np.log10(e.sum() / fs + 1), tsp, zc, kurtosis(ax)], float)                 # E3
