"""All detectors compared in workstream C. Each takes the preprocessed stream (t, y, z, sigma) and params and
returns an int array of [start, stop) sample segments at 50 Hz."""
import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import fftconvolve
import detector as D

FS = D.FS

PARAMS = {
    "FIXED": dict(win_s=4.0, k=5.0),
    "HYST": dict(D.DEFAULTS),
    "STALTA": dict(sta_s=0.5, lta_s=30.0, r_on=8.0, r_off=2.0, merge_gap_s=1.0, min_dur_s=0.3),
    "CUSUM": dict(nu=4.0, h=100.0, merge_gap_s=1.0, min_dur_s=0.3),
    "WAVELET": dict(scales=(0.15, 0.3, 0.6, 1.2), k_on=5.0, k_off=2.0, merge_gap_s=1.0, min_dur_s=0.3, env_s=0.2),
    "HYSTBP": dict(D.DEFAULTS, hp_s=4.0),
    # same high-passed detection channel for the other rules (fair comparison of the *rule*, not the front end)
    "STALTABP": dict(sta_s=0.5, lta_s=30.0, r_on=8.0, r_off=2.0, merge_gap_s=1.0, min_dur_s=0.3, hp_s=4.0),
    "CUSUMBP": dict(nu=4.0, h=100.0, merge_gap_s=1.0, min_dur_s=0.3, hp_s=4.0),
    "FIXEDLOC": dict(win_s=4.0, k=5.0, hp_s=4.0),   # fixed windows, but local baseline + robust sigma + HP channel
    # workstream B's detector (B_signal/scripts/det2.py + preproc.py + run_pipeline.py), re-implemented on C's 50 Hz stream
    "BDET": dict(lp_hz=5.0, base_win_s=20.0, k_hi=8.0, k_lo=2.0, merge_gap_s=1.0, min_dur_s=0.4, pass1_k=5.0, bridge_s=2.0),
}
# name of the parameter swept as "threshold" for ROC / operating point
THRESH = {"FIXED": "k", "HYST": "k_on", "STALTA": "r_on", "CUSUM": "h", "WAVELET": "k_on", "HYSTBP": "k_on",
          "STALTABP": "r_on", "CUSUMBP": "h", "FIXEDLOC": "k", "BDET": "k_hi"}


def det_fixed(t, y, z, sigma, p, dc=None):
    """legacy: DC = long-context mean (proxy for a 120-min DC baseline); fixed non-overlapping windows."""
    if dc is None:
        dc = np.mean(y)
    r = y - dc
    sg = 1.4826 * np.median(np.abs(r - np.median(r)))          # global robust sigma of the DC-relative signal
    n = int(p["win_s"] * FS)
    nw = len(y) // n
    m = np.abs(r[: nw * n].reshape(nw, n)).max(1) > p["k"] * sg
    idx = np.where(m)[0]
    return np.stack([idx * n, idx * n + n], 1) if len(idx) else np.zeros((0, 2), int)


def _post(segs, p):
    segs = D.merge_segments(segs, int(p["merge_gap_s"] * FS))
    if len(segs):
        segs = segs[(segs[:, 1] - segs[:, 0]) >= p["min_dur_s"] * FS]
    return segs


def det_hyst(t, y, z, sigma, p):
    return D.detect_hyst(t, z, sigma, p)


def det_stalta(t, y, z, sigma, p):
    e = z ** 2
    ns, nl = int(p["sta_s"] * FS), int(p["lta_s"] * FS)
    c = np.concatenate([[0.0], np.cumsum(e)])
    i = np.arange(len(e))
    sta = (c[np.minimum(i + 1, len(e))] - c[np.maximum(i + 1 - ns, 0)]) / ns       # trailing STA ending at i
    j = np.maximum(i + 1 - ns, 0)                                                   # LTA ends where STA begins
    lta = (c[j] - c[np.maximum(j - nl, 0)]) / np.maximum(j - np.maximum(j - nl, 0), 1)
    floor = np.median(sigma) ** 2 * 0.25
    ratio = sta / np.maximum(lta, floor)
    # onset = trigger - STA length ; offset = ratio < r_off
    segs = []
    k = 0
    n = len(e)
    on = np.where(ratio > p["r_on"])[0]
    while k < len(on):
        a = on[k]
        b = a
        while b < n and ratio[b] > p["r_off"]:
            b += 1
        segs.append([max(a - ns, 0), b])
        k = np.searchsorted(on, b + 1)
    segs = np.array(segs, int).reshape(-1, 2)
    return _post(segs, p)


def _cusum_zero_before_alarm(v, nu, h, cap):
    """forward Page CUSUM on v, capped at `cap` (bounded post-alarm decay);
    returns boolean 'active' marking [last zero before alarm .. return to zero] regions."""
    n = len(v)
    s = 0.0
    act = np.zeros(n, bool)
    last0 = 0
    alarm = False
    for i in range(n):
        s = min(max(0.0, s + v[i] - nu), cap)
        if s == 0.0:
            if alarm:
                act[last0:i] = True
                alarm = False
            last0 = i
        elif s > h:
            alarm = True
    if alarm:
        act[last0:] = True
    return act


try:  # numba not guaranteed; pure python loop is fast enough at 50 Hz (180k samples/h)
    import numba
    _cusum_zero_before_alarm = numba.njit(cache=True)(_cusum_zero_before_alarm)
except Exception:
    pass


def det_cusum(t, y, z, sigma, p):
    """Two-sided Page CUSUM on the normalised energy. Forward pass: onset = last zero of S before the alarm.
    Backward pass (time-reversed): its 'onset' is the event's true end. Event = forward-active AND backward-active."""
    u2 = np.minimum((z / sigma) ** 2, p.get("clip", 100.0))
    cap = p.get("cap_mult", 1.5) * p["h"]
    f = _cusum_zero_before_alarm(u2, p["nu"], p["h"], cap)
    b = _cusum_zero_before_alarm(u2[::-1].copy(), p["nu"], p["h"], cap)[::-1]
    segs = D.runs_of(f & b)
    return _post(segs, p)


def _gauss_deriv_kernels(scales):
    ks = []
    for s in scales:
        sd = s * FS
        x = np.arange(-int(4 * sd), int(4 * sd) + 1)
        g = np.exp(-0.5 * (x / sd) ** 2)
        for order in (1, 2):
            k = -x * g if order == 1 else (1 - (x / sd) ** 2) * g
            k = k - k.mean()
            k = k / np.sqrt(np.sum(k ** 2))
            ks.append(k)
    return ks


def det_wavelet(t, y, z, sigma, p):
    best = np.zeros(len(z))
    for k in _gauss_deriv_kernels(p["scales"]):
        r = fftconvolve(z, k, mode="same")
        # robust normalisation of each response (rolling via block MAD over the whole stream, masked by D baseline)
        blk = int(FS)
        nb = len(r) // blk
        mad = np.median(np.abs(r[: nb * blk].reshape(nb, blk)), axis=1)
        from scipy.ndimage import median_filter
        s = 1.4826 * D._to_samples(median_filter(mad, size=301, mode="nearest"), len(r), blk)
        best = np.maximum(best, np.abs(r) / np.maximum(s, 1e-9))
    env = uniform_filter1d(best, max(1, int(p["env_s"] * FS)))
    segs = D.hysteresis(env, p["k_on"], p["k_off"])
    # refine boundaries with the plain normalised envelope (as HYST)
    u_env = D.envelope(z / sigma, 0.2)
    ref = []
    for a, b in segs:
        while a > 0 and u_env[a - 1] > 2.0:
            a -= 1
        while b < len(z) and u_env[b] > 2.0:
            b += 1
        ref.append([a, b])
    segs = np.array(ref, int).reshape(-1, 2)
    return _post(segs, p)


def det_hystbp(t, y, z, sigma, p):
    """HYST on a high-passed detection channel (removes 0.01-0.1 Hz wander that is not vehicle-core energy)."""
    zh, sh = D.highpass_channel(z, p["hp_s"])
    return D.detect_hyst(t, zh, sh, p)


def _on_hp(fn):
    def g(t, y, z, sigma, p):
        zh, sh = D.highpass_channel(z, p["hp_s"])
        return fn(t, y, zh, sh, p)
    return g


def det_fixedloc(t, y, z, sigma, p):
    """fixed non-overlapping windows flagged if max |u| > k on the locally normalised (HP) detection channel."""
    zh, sh = D.highpass_channel(z, p["hp_s"])
    u = np.abs(zh / sh)
    n = int(p["win_s"] * FS)
    nw = len(u) // n
    idx = np.where(u[: nw * n].reshape(nw, n).max(1) > p["k"])[0]
    return np.stack([idx * n, idx * n + n], 1) if len(idx) else np.zeros((0, 2), int)


def det_bdet(t, y, z, sigma, p):
    """B's chain from the (gap-bridged) 50 Hz waveform y: 5 Hz 4th-order zero-phase LP (even padding), two-pass 20 s
    running median (pass 2 bridges pass-1 events +-2 s), per-60 s-block MAD sigma, hysteresis k_hi core / k_lo extent,
    merge < 1 s, drop < 0.4 s."""
    from scipy import signal, ndimage
    sos = signal.butter(4, p["lp_hz"], fs=FS, output="sos")
    yl = signal.sosfiltfilt(sos, y - np.median(y), padtype="even", padlen=min(len(y) - 1, int(3 * FS)))
    w = int(p["base_win_s"] * FS) | 1

    def rsig(r):
        B = int(60 * FS); s = np.empty(len(r))
        for c in range(0, len(r), B):
            seg = r[c:c + B]; s[c:c + B] = 1.4826 * np.median(np.abs(seg - np.median(seg)))
        return np.maximum(s, 1e-9)

    def hyst(r, sg, khi):
        lo = np.abs(r) > p["k_lo"] * sg
        segs = [s for s in D.runs_of(lo) if np.any(np.abs(r[s[0]:s[1]]) > khi * sg[s[0]:s[1]])]
        return D.merge_segments(np.array(segs, int).reshape(-1, 2), int(p["merge_gap_s"] * FS))

    r1 = yl - ndimage.median_filter(yl, w, mode="nearest")
    em = np.zeros(len(yl), bool)
    for a_, e_ in hyst(r1, rsig(r1), p["pass1_k"]):
        em[max(0, a_ - int(p["bridge_s"] * FS)):e_ + int(p["bridge_s"] * FS)] = True
    zz = yl.copy(); idx = np.arange(len(zz))
    if (~em).sum() > 2:
        zz[em] = np.interp(idx[em], idx[~em], zz[~em])
    r = yl - ndimage.median_filter(zz, w, mode="nearest")
    segs = hyst(r, rsig(r), p["k_hi"])
    if len(segs):
        segs = segs[(segs[:, 1] - segs[:, 0]) >= p["min_dur_s"] * FS]
    return segs


FUNCS = {"FIXED": det_fixed, "BDET": det_bdet, "HYST": det_hyst, "STALTA": det_stalta, "CUSUM": det_cusum, "WAVELET": det_wavelet,
         "HYSTBP": det_hystbp, "STALTABP": _on_hp(det_stalta), "CUSUMBP": _on_hp(det_cusum), "FIXEDLOC": det_fixedloc}


def run(name, t, y, z, sigma, params=None, **kw):
    p = dict(PARAMS[name], **(params or {}))
    return FUNCS[name](t, y, z, sigma, p, **kw)
