"""Reference preprocessing chain (workstream B, v2), the one recommended in REPORT.md.

Input: raw 24-bit codes `v` and device ticks `t` of one chunk or of consecutive chunks (a span).
  1. Slots: dt > 1.5*median(dt) marks lost samples; n_lost = round(dt/median) - 1 (A's estimator).
     Losses of 1-4 samples are filled by linear interpolation at 2416 Hz; losses >= 5 split the span
     into segments. Segments shorter than MIN_SEG_S are dropped (masked).
  2. Mains: per segment, 50 Hz + NH harmonics removed by least squares in 10 s blocks with 50 %
     overlap, cross-faded with Hann weights (no block-edge steps); f0 re-estimated per block.
  3. Low-pass: 4th-order Butterworth, zero-phase (sosfiltfilt), even padding, per segment.
     Detection path: 5 Hz.  Feature path: FEATURE_LP_HZ (see REPORT §4).
  4. Resample each segment to exactly 50 Hz with resample_poly(y, 25, 1208) (2416*25/1208 = 50),
     placed on a global 50 Hz grid (segment start rounded to the nearest 20 ms); masked samples NaN.
  5. Baseline: running median of 20 s on the 50 Hz grid (NaNs linearly bridged for the median only),
     subtracted (never divided). Optional 2nd pass re-estimates the median with events masked.
Time: slot index / FS_ADC, with FS_ADC = 2416 Hz (A: fs_local 2407-2421, REPORT A §5.4)."""
import numpy as np
from scipy import signal, ndimage

FS_ADC = 2416.0
FS_OUT = 50.0
UP, DOWN = 25, 1208
NH = 20
BLOCK_S = 10.0
MIN_SEG_S = 0.5
FILL_MAX = 4
DETECT_LP_HZ = 5.0
FEATURE_LP_HZ = 8.0
BASE_WIN_S = 20.0


# ---------------------------------------------------------------- slots and segments
def slots(t):
    """Return (x_index_of_each_sample_in_slots, list of (slot_a, slot_b) segments, n_slots)."""
    d = np.diff(t).astype(float)
    dm = np.median(d)
    lost = np.zeros(len(t), np.int64)
    lost[1:] = np.where(d > 1.5 * dm, np.round(d / dm).astype(np.int64) - 1, 0)
    s = np.arange(len(t)) + np.cumsum(lost)
    brk = np.where(lost >= FILL_MAX + 1)[0]
    edges = np.r_[0, brk, len(t)]
    segs = [(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]  # sample-index ranges
    return s, segs, lost


def fill_segment(v, s, a, b):
    """Samples a..b-1 (no big gap inside) on their slot grid; 1-4-sample holes linearly filled."""
    ss = s[a:b]
    grid = np.arange(ss[0], ss[-1] + 1)
    if len(grid) == b - a:
        return grid, v[a:b].astype(float)
    return grid, np.interp(grid, ss, v[a:b].astype(float))


# ---------------------------------------------------------------- mains removal
def _f0(x, fs=FS_ADC):
    n = len(x)
    X = np.fft.rfft((x - x.mean()) * np.hanning(n), 8 * n)
    f = np.fft.rfftfreq(8 * n, 1 / fs)
    m = (f > 49.0) & (f < 51.0)
    return f[m][np.argmax(np.abs(X[m]))]


def _fit_block(x, f0, fs=FS_ADC, nh=NH):
    n = np.arange(len(x)) / fs
    H = np.empty((len(x), 2 * nh))
    for k in range(1, nh + 1):
        H[:, 2 * k - 2] = np.cos(2 * np.pi * k * f0 * n)
        H[:, 2 * k - 1] = np.sin(2 * np.pi * k * f0 * n)
    nn = (n - n.mean()) / max(np.ptp(n), 1e-9)
    M = np.c_[np.ones(len(n)), nn, nn ** 2, nn ** 3, H]
    co = np.linalg.solve(M.T @ M + 1e-9 * np.eye(M.shape[1]), M.T @ x)
    return H @ co[4:]


def remove_mains(x, fs=FS_ADC, block_s=BLOCK_S, nh=NH, overlap=True):
    """Harmonic regression; overlap=True: 50 %-overlapping blocks cross-faded with Hann weights."""
    B = int(block_s * fs)
    N = len(x)
    if N <= B:
        return x - _fit_block(x, _f0(x), fs, nh)
    if not overlap:
        y = x.copy()
        for c in range(0, N, B):
            blk = x[c:c + B]
            if len(blk) < 100:
                continue
            y[c:c + B] = blk - _fit_block(blk, _f0(blk), fs, nh)
        return y
    hop = B // 2
    starts = list(range(0, N - B + 1, hop))
    if starts[-1] != N - B:
        starts.append(N - B)
    model = np.zeros(N)
    wsum = np.zeros(N)
    w = np.hanning(B) + 1e-3
    for c in starts:
        blk = x[c:c + B]
        model[c:c + B] += w * _fit_block(blk, _f0(blk), fs, nh)
        wsum[c:c + B] += w
    return x - model / wsum


# ---------------------------------------------------------------- filtering and resampling
def lowpass(x, fc, fs=FS_ADC):
    sos = signal.butter(4, fc, fs=fs, output="sos")
    return signal.sosfiltfilt(sos, x, padtype="even", padlen=min(len(x) - 1, int(3 * fs)))


def to50(y):
    return signal.resample_poly(y, UP, DOWN)


# ---------------------------------------------------------------- whole-span driver
def clean_span(v, t, lp_list=(DETECT_LP_HZ, FEATURE_LP_HZ), keep_raw=False):
    """Returns dict with:
       'g50': {fc: 50 Hz series on global grid (NaN where masked)}, 'mask50': bool array,
       't50': seconds since first slot, 'segments': list of (slot_a, slot_b),
       'raw': list of (slot_a, mains-free raw at 2416 Hz) if keep_raw."""
    v = v.astype(float) - np.median(v)
    s, segs, lost = slots(t)
    n_slots = s[-1] + 1
    n50 = int(np.ceil(n_slots / FS_ADC * FS_OUT)) + 1
    out = {fc: np.full(n50, np.nan) for fc in lp_list}
    raw = []
    seglist = []
    for a, b in segs:
        grid, x = fill_segment(v, s, a, b)
        if len(x) < MIN_SEG_S * FS_ADC:
            continue
        xm = remove_mains(x)
        seglist.append((grid[0], grid[-1] + 1))
        if keep_raw:
            raw.append((grid[0], xm.astype(np.float32)))
        j0 = int(round(grid[0] / FS_ADC * FS_OUT))
        for fc in lp_list:
            y = to50(lowpass(xm, fc))
            j1 = min(n50, j0 + len(y))
            # trim 1 sample at each end (resample_poly edge) to be safe
            out[fc][j0 + 1:j1 - 1] = y[1:j1 - j0 - 1]
    mask = np.isnan(out[lp_list[0]])
    return dict(g50=out, mask50=mask, t50=np.arange(n50) / FS_OUT, segments=seglist, raw=raw,
                n_lost=int(lost.sum()))


def baseline(y, mask, win_s=BASE_WIN_S, fs=FS_OUT, event_mask=None):
    """Running median; NaN/masked (and optional event) samples are bridged by linear interpolation
    before the median, so they do not pull it."""
    z = y.copy()
    bad = mask.copy() if event_mask is None else (mask | event_mask)
    idx = np.arange(len(z))
    if (~bad).sum() < 2:
        return np.zeros_like(z)
    z[bad] = np.interp(idx[bad], idx[~bad], z[~bad])
    w = int(win_s * fs) | 1
    return ndimage.median_filter(z, w, mode="nearest")


def robust_sigma(r, mask, block_s=60.0, fs=FS_OUT):
    """Per-block MAD sigma, piecewise constant."""
    B = int(block_s * fs)
    sig = np.full(len(r), np.nan)
    for c in range(0, len(r), B):
        seg = r[c:c + B][~mask[c:c + B]]
        if len(seg) > 0.2 * B:
            sig[c:c + B] = 1.4826 * np.median(np.abs(seg - np.median(seg)))
    if np.isnan(sig).all():
        return np.full(len(r), np.nan)
    ok = ~np.isnan(sig)
    sig[~ok] = np.interp(np.where(~ok)[0], np.where(ok)[0], sig[ok])
    return sig
