"""SUPERSEDED exploratory front end (Time-clock 20 ms bins) used only by scripts/look.py, overview.py, zoom.py.
The Time counter is not a valid sample clock (workstream A F3); see detector.front_end."""
"""Front end: gap-aware time-binned mean at 50 Hz (20 ms bins). The 20 ms boxcar has exact nulls at
50 Hz and all harmonics (mains interference), acts as anti-alias LPF (-3 dB ~22 Hz). Empty bins (gaps) are NaN."""
import numpy as np
import cio

FS = 50.0
BIN = 1.0 / FS

def bin_series(x, t, t0=None, t1=None):
    """x counts, t seconds. Returns (tb bin-centre times, y mean per bin (NaN if empty), n samples per bin)."""
    if t0 is None: t0 = np.floor(t[0] / BIN) * BIN
    if t1 is None: t1 = t[-1]
    nb = int(np.ceil((t1 - t0) / BIN))
    k = ((t - t0) / BIN).astype(np.int64)
    ok = (k >= 0) & (k < nb)
    s = np.bincount(k[ok], weights=x[ok], minlength=nb)
    n = np.bincount(k[ok], minlength=nb)
    with np.errstate(invalid="ignore", divide="ignore"):
        y = s / n
    y[n == 0] = np.nan
    return t0 + (np.arange(nb) + 0.5) * BIN, y, n

def fill_gaps(y):
    """linear interpolation over NaN bins; returns filled y and mask of filled bins."""
    m = np.isnan(y)
    if m.any() and (~m).any():
        idx = np.arange(len(y))
        y = y.copy(); y[m] = np.interp(idx[m], idx[~m], y[~m])
    return y, m

def load_binned(run, i0, n):
    x, t, c = cio.load_chunks(run, i0, n)
    tb, y, cnt = bin_series(x, t)
    y, gap = fill_gaps(y)
    # chunk id per bin (for boundary analysis)
    cb = c[np.clip(np.searchsorted(t, tb), 0, len(c) - 1)]
    return tb, y, gap, cb
