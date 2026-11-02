"""Vehicle event detection for the single-channel TMR sensor (workstream C).

Pipeline (all operating on a 50 Hz series; see REPORT.md for the evidence behind every choice):
  1. front end   : dropout slots re-inserted from the Time ticks; 481-tap FIR low-pass (10 Hz) + decimation by 48
                   on the true ~2416 Hz sample grid -> 50.33 Hz; mains + harmonics suppressed >50 dB; 50 Hz samples
                   whose FIR support touches a dropout are masked and bridged (flagged: has_dropout).
                   Detection channel additionally median-filtered over 0.5 s (removes a periodic 0.1-0.15 s pulser).
  2. baseline    : two-pass rolling median (60 s) of 1 s block medians; pass 2 masks samples that pass 1 flags
                   (|z|>4 sigma, dilated +-8 s) so clustered/long events do not bias the baseline.
  3. noise sigma : rolling (5 min) robust scale of the residual, masked samples excluded.
  4. detection   : recommended = HYST-BP: the detection channel is high-passed (4 s moving-average subtraction),
                   re-normalised by a rolling robust sigma, and a hysteresis rule on the 0.2 s moving-RMS envelope
                   (on 5 sigma, off 2 sigma, merge gaps < 1 s, split at deep valleys, drop < 0.3 s) gives segments.
                   Chosen on Runs 1/2/4 by the pre-registered rule; see REPORT.md.
Other detectors (FIXED, STALTA, CUSUM, WAVELET) are provided for the comparison in scripts/.

Usage
-----
    import detector
    ev, aux = detector.detect_stream(x_counts, t_seconds)      # raw samples + timestamps (s)
    # or, on an already binned 50 Hz series:
    ev, aux = detector.detect_binned(t50, y50)
Each event is a dict: onset, offset, peak_t, centroid_t (seconds on the 2416 Hz sample clock, i.e. slot/2416;
NOT the device Time ticks), peak_z (counts, signed), snr (peak |z|/sigma), dur (s), e_on, e_off (1 %/99 %
cumulative-energy extent inside the segment), i0/i1 (50 Hz indices), clean (snr >= 30), n_neighbours_in_window,
and from detect_stream also raw_on/raw_off/raw_peak (indices into the raw input x) and has_dropout.
Recommended use: keep all events (with snr) but train on the `clean` tier; cut a 21 s window (pre-registered rule; 12 s core-only alternative) centred on
centroid_t with `segment_window(...)` and keep [onset, offset] as a mask; repair (do not drop) has_dropout events.
"""
import numpy as np
import pandas as pd
from scipy.ndimage import median_filter, binary_dilation, uniform_filter1d


DEFAULTS = dict(
    base_win_s=60.0,     # rolling-median baseline window
    sigma_win_s=300.0,   # rolling robust-sigma window
    mask_k=4.0,          # pass-1 mask threshold (sigma)
    mask_dil_s=8.0,      # pass-1 mask dilation
    env_s=0.2,           # envelope moving-RMS length
    k_on=5.0,            # hysteresis on-threshold (sigma)
    k_off=2.0,           # hysteresis off-threshold (sigma)
    merge_gap_s=1.0,     # merge segments separated by less than this
    min_dur_s=0.3,       # drop shorter segments
    split=True,          # split merged segments at deep envelope valleys (consecutive vehicles)
    split_ratio=0.2,     # valley must be < split_ratio * min(neighbouring peaks)
    split_min_sep_s=1.0, # minimum separation between the two peaks
    pulse_med_s=0.5,     # detection-channel median width: removes <=0.25 s impulsive interference (3 s periodic pulser)
    hp_s=4.0,            # detection-channel high-pass (moving-average subtraction, ~0.1 Hz): removes slow wander
    snr_min_clean=30.0,  # "clean" tier: peak |z| / sigma >= 30 (hand-checked holdout precision 0.96 [0.87, 0.99])
    window_s=21.0,       # fixed segment length (pre-registered rule: p95 of 1-99 % energy extent incl. ramps, clean tier;
                         # REPORT 6.6). 12 s = core-only alternative (1.8 % segment truncation, but 42 % ramp truncation)
)


# ----------------------------------------------------------------------------- front end
FS_RAW = 2416.0      # true sample rate estimated from the 50 Hz mains line (Time counter runs 1-5 % off; see REPORT)
DEC = 48             # decimation factor -> FS = 50.33 Hz
FS = FS_RAW / DEC
BIN = 1.0 / FS
_H = None


def _fir():
    global _H
    if _H is None:
        from scipy.signal import firwin
        _H = firwin(481, 10.0, fs=FS_RAW)   # Hamming, -6 dB @10 Hz, >50 dB above ~27 Hz (kills mains + harmonics)
    return _H


def insert_dropouts(x, t):
    """Re-insert samples lost in dropouts (workstream A: ~105-sample gaps, detectable as Time jumps).
    t: Time values (device ticks or seconds; only ratios are used). Returns (x on a uniform slot grid with
    dropouts linearly interpolated, slot index of every original sample, boolean mask of inserted slots)."""
    x = np.asarray(x, float)
    if t is None:
        return x, np.arange(len(x)), np.zeros(len(x), bool)
    d = np.diff(np.asarray(t, float))
    med = np.median(d)
    step = np.ones(len(d), np.int64)
    big = d > 1.5 * med
    step[big] = np.maximum(1, np.round(d[big] / med)).astype(np.int64)
    slot = np.concatenate([[0], np.cumsum(step)])
    n = int(slot[-1]) + 1
    xs = np.interp(np.arange(n), slot, x)
    ins = np.ones(n, bool)
    ins[slot] = False
    return xs, slot, ins


def front_end(x, t=None):
    """Dropout re-insertion (if t given) + index-domain anti-alias FIR + decimation by 48 on the true
    ~2416 Hz sample grid (not the Time counter, whose tick rate is 1-5 % off; workstream A F3/F4).
    Returns (t50, y50, slot50, drop50): output times in s on the sample clock (slot/FS_RAW), filtered counts,
    the slot index each output sample is aligned to, and a flag for output samples within 0.1 s of a dropout.
    Map slots back to raw sample indices with np.searchsorted(slot_of_raw, slot50). Edge samples (FIR
    transient, 0.1 s at each end) are dropped."""
    from scipy.signal import upfirdn
    xs, slot_of_raw, ins = insert_dropouts(x, t)
    m = xs.mean()
    h = _fir()
    y = upfirdn(h, xs - m, down=DEC) + m
    half = (len(h) - 1) // 2
    idx = np.arange(len(y)) * DEC - half
    keep = (idx >= half) & (idx + half < len(xs))
    y, idx = y[keep], idx[keep]
    if ins.any():
        near = np.convolve(ins.astype(float), np.ones(2 * half + 1), mode="same") > 0
        drop = near[idx]
        # mask (not interpolate) at raw rate: output samples whose FIR support touches a dropout are
        # discarded and bridged linearly at 50 Hz (scripts/dropout_test.py: raw interpolation / concatenation
        # leave step artefacts of p99 5-10 sigma; this leaves < 1-2.5 sigma).
        y = y.copy(); y[drop] = np.nan
        y, _ = fill_gaps(y)
    else:
        drop = np.zeros(len(idx), bool)
    return idx / FS_RAW, y, idx, drop


def fill_gaps(y):
    m = np.isnan(y)
    if m.any() and (~m).any():
        i = np.arange(len(y))
        y = y.copy()
        y[m] = np.interp(i[m], i[~m], y[~m])
    return y, m


# ----------------------------------------------------------------------------- baseline / sigma
def _block_stat(v, blk, fn):
    nb = len(v) // blk
    b = v[: nb * blk].reshape(nb, blk)
    out = fn(b, axis=1)
    if len(v) > nb * blk:
        with np.errstate(all="ignore"):
            out = np.append(out, fn(v[nb * blk:][None, :], axis=1))
    return out


def _to_samples(bvals, n, blk):
    centres = (np.arange(len(bvals)) * blk + blk / 2.0)
    ok = np.isfinite(bvals)
    if not ok.any():
        return np.zeros(n)
    return np.interp(np.arange(n), centres[ok], bvals[ok])


def _rolling_nanmedian(v, w):
    return pd.Series(v).rolling(int(w), center=True, min_periods=max(3, int(w) // 6)).median().to_numpy()


def baseline_sigma(y, p=DEFAULTS, return_all=False):
    """Two-pass masked baseline and rolling robust sigma. Returns z (residual), sigma (per sample), mask, base."""
    blk = int(FS)  # 1 s blocks
    wb = int(p["base_win_s"])
    ws = int(p["sigma_win_s"])
    n = len(y)
    # pass 1
    bm = _block_stat(y, blk, np.median)
    b1 = _to_samples(median_filter(bm, size=wb | 1, mode="nearest"), n, blk)
    z1 = y - b1
    s_blk = _block_stat(np.abs(z1 - np.median(z1)), blk, np.median)
    s1 = 1.4826 * _to_samples(median_filter(s_blk, size=ws | 1, mode="nearest"), n, blk)
    s1 = np.maximum(s1, 1e-9)
    mask = np.abs(z1) > p["mask_k"] * s1
    mask = binary_dilation(mask, iterations=int(p["mask_dil_s"] * FS)) if mask.any() else mask
    # pass 2 (masked)
    ym = np.where(mask, np.nan, y)
    with np.errstate(all="ignore"):
        bm2 = _block_stat(ym, blk, lambda a, axis: np.where(np.mean(np.isnan(a), axis=axis) > 0.5, np.nan,
                                                            np.nanmedian(a, axis=axis)))
    bm2 = _rolling_nanmedian(bm2, wb | 1)
    base = _to_samples(bm2, n, blk)
    z = y - base
    zm = np.where(mask, np.nan, z)
    with np.errstate(all="ignore"):
        s_blk2 = _block_stat(np.abs(zm), blk, lambda a, axis: np.where(np.mean(np.isnan(a), axis=axis) > 0.5, np.nan,
                                                                       np.nanmedian(a, axis=axis)))
    s_blk2 = _rolling_nanmedian(s_blk2, ws | 1)
    sigma = 1.4826 * _to_samples(s_blk2, n, blk)
    sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, np.nanmedian(s1))
    if return_all:
        return z, sigma, mask, base
    return z, sigma


def preprocess(y50, p=DEFAULTS):
    """Returns y (waveform channel, gap-filled), z (waveform residual), zd (detection-channel residual:
    0.5 s median removes short impulsive interference), sigma (robust noise of zd), base (baseline)."""
    y, _ = fill_gaps(y50)
    w = int(round(p.get("pulse_med_s", 0.5) * FS)) | 1
    yd = median_filter(y, size=w, mode="nearest") if w > 1 else y
    zd, sigma, mask, base = baseline_sigma(yd, p, return_all=True)
    return dict(y=y, z=y - base, zd=zd, sigma=sigma, base=base, mask=mask)


# ----------------------------------------------------------------------------- segment utilities
def runs_of(b):
    """start/stop (exclusive) indices of True runs."""
    if not b.any():
        return np.zeros((0, 2), int)
    d = np.diff(np.concatenate([[0], b.astype(np.int8), [0]]))
    return np.stack([np.where(d == 1)[0], np.where(d == -1)[0]], 1)


def hysteresis(env, on, off):
    """segments where env>off that contain at least one sample > on."""
    segs = runs_of(env > off)
    keep = [s for s in segs if env[s[0]:s[1]].max() > on]
    return np.array(keep, int).reshape(-1, 2)


def merge_segments(segs, gap):
    if len(segs) == 0:
        return segs
    out = [list(segs[0])]
    for a, b in segs[1:]:
        if a - out[-1][1] < gap:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return np.array(out, int)


def split_segments(segs, env, ratio, min_sep):
    """split a segment at envelope valleys that are deep relative to both neighbouring peaks."""
    out = []
    w = int(0.5 * FS)
    for a, b in segs:
        e = uniform_filter1d(env[a:b], 25) if b - a > 25 else env[a:b]
        cuts = []
        if b - a > 2 * min_sep:
            # candidate valleys: local minima of the smoothed envelope
            lo = np.where((e[1:-1] <= e[:-2]) & (e[1:-1] <= e[2:]))[0] + 1
            last = 0
            for v in lo:
                left = e[last:v].max() if v > last else 0
                right = e[v:].max()
                if v - last >= min_sep and (len(e) - v) >= min_sep and e[v] < ratio * min(left, right):
                    cuts.append(v)
                    last = v
        edges = [0] + cuts + [b - a]
        for i in range(len(edges) - 1):
            out.append([a + edges[i], a + edges[i + 1]])
    return np.array(out, int).reshape(-1, 2)


def describe(segs, t, z, sigma):
    ev = []
    for a, b in segs:
        zz = z[a:b]
        j = int(np.argmax(np.abs(zz)))
        e = zz ** 2
        c = np.cumsum(e)
        tot = c[-1] if c[-1] > 0 else 1.0
        i1 = int(np.searchsorted(c, 0.01 * tot))
        i99 = min(int(np.searchsorted(c, 0.99 * tot)), b - a - 1)
        ev.append(dict(onset=t[a], offset=t[b - 1] + BIN, peak_t=t[a + j], peak_z=float(zz[j]),
                       snr=float(abs(zz[j]) / sigma[a + j]), dur=(b - a) * BIN,
                       centroid_t=float(np.sum(t[a:b] * e) / tot), e_on=t[a + i1], e_off=t[a + i99] + BIN,
                       i0=int(a), i1=int(b)))
    return ev


def rolling_sigma(v, win_s=300.0):
    """rolling robust sigma: 1.4826 * rolling median (win_s) of 1 s block medians of |v|."""
    blk = int(FS)
    b = _block_stat(np.abs(v), blk, np.median)
    return np.maximum(1.4826 * _to_samples(median_filter(b, size=int(win_s) | 1, mode="nearest"), len(v), blk), 1e-9)


def highpass_channel(zd, hp_s):
    """subtract an hp_s moving average (first null at 1/hp_s Hz; -3 dB ~0.44/hp_s Hz) and re-estimate sigma."""
    if not hp_s:
        return zd, rolling_sigma(zd)
    zh = zd - uniform_filter1d(zd, max(1, int(round(hp_s * FS))), mode="nearest")
    return zh, rolling_sigma(zh)


# ----------------------------------------------------------------------------- recommended detector
def envelope(u, env_s):
    return np.sqrt(uniform_filter1d(u ** 2, max(1, int(env_s * FS))))


def detect_hyst(t, z, sigma, p=DEFAULTS):
    u = z / sigma
    env = envelope(u, p["env_s"])
    segs = hysteresis(env, p["k_on"], p["k_off"])
    segs = merge_segments(segs, int(p["merge_gap_s"] * FS))
    if p.get("split", False):
        segs = split_segments(segs, env, p["split_ratio"], int(p["split_min_sep_s"] * FS))
    segs = segs[(segs[:, 1] - segs[:, 0]) >= p["min_dur_s"] * FS] if len(segs) else segs
    return segs


def detect_binned(t50, y50, params=None):
    p = dict(DEFAULTS, **(params or {}))
    P = preprocess(y50, p)
    zh, sh = highpass_channel(P["zd"], p["hp_s"])
    segs = detect_hyst(t50, zh, sh, p)
    P.update(t=t50, zh=zh, sigma_h=sh)
    ev = describe(segs, t50, P["z"], P["sigma"])
    cents = np.array([e["centroid_t"] for e in ev])
    for k, e in enumerate(ev):
        e["clean"] = bool(e["snr"] >= p["snr_min_clean"])
        # another detected event whose segment overlaps this event's fixed window (contamination flag)
        w0, w1 = e["centroid_t"] - p["window_s"] / 2, e["centroid_t"] + p["window_s"] / 2
        e["n_neighbours_in_window"] = int(sum(1 for j, o in enumerate(ev) if j != k and o["onset"] < w1 and o["offset"] > w0))
    return ev, P


def detect_stream(x, t=None, params=None):
    """x: raw counts of a contiguous (stitched) stream, t: optional Time stamps (s) per raw sample.
    Pass >=60 s of context either side of the region of interest (baseline edge effects) and stitch consecutive
    chunks (events straddle chunk boundaries). Event dicts additionally get raw_on/raw_off/raw_peak (raw sample
    indices into x) so segments can be cut at the native 2.4 kHz rate."""
    t50, y50, slot50, drop50 = front_end(x, t)
    ev, aux = detect_binned(t50, y50, params)
    _, slot_of_raw, _ = insert_dropouts(x, t)
    for e in ev:
        a, b = e["i0"], min(e["i1"], len(slot50) - 1)
        e["raw_on"], e["raw_off"] = (int(v) for v in np.searchsorted(slot_of_raw, [slot50[a], slot50[b]]))
        e["raw_peak"] = int(np.searchsorted(slot_of_raw, round(e["peak_t"] * FS_RAW)))
        e["has_dropout"] = bool(drop50[a:b + 1].any())
    aux.update(slot=slot50, dropout=drop50)
    return ev, aux


def segment_window(t50, z50, ev, length_s=DEFAULTS["window_s"], anchor="centroid"):
    """Fixed-length window (recommended representation) centred on the event's energy centroid.
    Returns (window samples of z at 50 Hz, index of window start). Pads with zeros at stream edges."""
    c = ev["centroid_t"] if anchor == "centroid" else ev["peak_t"]
    n = int(round(length_s * FS))
    i0 = int(round((c - t50[0]) / BIN)) - n // 2
    w = np.zeros(n)
    a, b = max(i0, 0), min(i0 + n, len(z50))
    if b > a:
        w[a - i0:b - i0] = z50[a:b]
    return w, i0
