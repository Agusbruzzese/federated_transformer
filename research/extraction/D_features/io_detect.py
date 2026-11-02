"""Provisional loader + event detector (workstream D). NOT the official detector (workstream C)."""
import numpy as np, glob, os, re, datetime as dt
from scipy import signal

ROOT = "/home/augustin/Documents/federated_transformer/Spintronic Sensor Dataset for Vehicle Detection and Car Model Recognition (14 Days, Unlabeled with Timestamps)"
FS = 2400.0          # nominal; median timestamp dt ~416-435 us (2300-2400 Hz) -> see REPORT caveat
DEC = 24             # decimation factor -> ~100 Hz
FSD = FS / DEC
_rx = re.compile(r"BIN Values - (\d+) Start@ (\d{8}) - (\d\d)h(\d\d)m(\d\d)s")

def list_chunks():
    out = []
    for ri, r in enumerate(sorted(glob.glob(ROOT + "/Run*"))):
        for f in sorted(glob.glob(r + "/S0C0 BIN Values*")):
            m = _rx.search(f)
            idx, d, H, M, S = m.groups()
            t0 = dt.datetime.strptime(d + H + M + S, "%Y%m%d%H%M%S")
            out.append((ri, int(idx), t0, f))
    return out

def load_chunk(f):
    v = np.fromfile(f, dtype="<u4").astype(np.float64)
    t = np.fromfile(f.replace("BIN Values", "BIN Time"), dtype="<u8")
    return v, t

_sos = signal.butter(4, 20, fs=FS, output="sos")

def load_block(files):
    """Concatenate consecutive chunks. Returns raw (float64), timestamps(us)."""
    V, T = zip(*[load_chunk(f) for f in files])
    return np.concatenate(V), np.concatenate(T)

def lowpass_dec(v):
    x = signal.sosfiltfilt(_sos, v - np.median(v))
    return x[::DEC]

def robust_baseline(x, fsd=FSD, blk_s=2.0, win_blocks=31):
    """Median of 2-s blocks -> running median over ~60 s -> interpolate."""
    b = int(blk_s * fsd); n = len(x) // b
    if n < 3:
        return np.full_like(x, np.median(x))
    bm = np.median(x[: n * b].reshape(n, b), axis=1)
    from scipy.ndimage import median_filter
    bm = median_filter(bm, size=min(win_blocks, n | 1), mode="nearest")
    c = (np.arange(n) + 0.5) * b
    return np.interp(np.arange(len(x)), c, bm)

def detect(x, fsd=FSD, k_hi=8.0, k_lo=3.0, smooth_s=0.25, merge_s=1.5, min_s=0.3):
    base = robust_baseline(x, fsd)
    r = x - base
    sig = 1.4826 * np.median(np.abs(r - np.median(r)))
    env = np.convolve(np.abs(r), np.ones(int(smooth_s * fsd)) / int(smooth_s * fsd), mode="same")
    hi = env > k_hi * sig; lo = env > k_lo * sig
    # hysteresis: lo-regions containing a hi sample
    lab = np.diff(np.r_[0, lo.astype(int), 0]); st = np.where(lab == 1)[0]; en = np.where(lab == -1)[0]
    segs = [(s, e) for s, e in zip(st, en) if hi[s:e].any()]
    merged = []
    for s, e in segs:
        if merged and s - merged[-1][1] < merge_s * fsd:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))
    merged = [(s, e) for s, e in merged if e - s >= min_s * fsd]
    return r, sig, merged
