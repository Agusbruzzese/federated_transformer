"""Feature path: workstream B's chain (preproc.py v2) on A's run-global slot grid.

  1. split at dropouts of >= 5 samples, linearly fill 1-4-sample dropouts          (B step 1, A's slots)
  2. per segment: 50 Hz + 20 harmonics regression, 10 s Hann-crossfaded blocks     (B step 2, preproc.remove_mains)
  3. per segment: LP 8 Hz, 4th-order Butterworth, zero-phase, even padding          (B step 3, preproc.lowpass)
     -- designed at the segment's true fs_local (slot-weighted mean over its chunks) instead of 2416 Hz
  4. per segment: polyphase resampling by 25/1208 onto a RUN-GLOBAL output grid:
     output sample j sits exactly at slot j*1208/25 (phase-exact; B rounded each segment start to the
     nearest 20 ms). Effective rate = fs_local*25/1208 (49.8-50.1 Hz), recorded per event.
  5. repair: samples not covered by a segment (dropout holes, segments < 0.5 s) are bridged linearly.
     flag mask: holes plus +-0.15 s around every split (B's mask definition, used for mask_frac and the
     per-sample repair mask). The guard zone is NOT bridged: research/extraction/E_pipeline/
     dropout_repair_test.py shows bridging it is 3-5x worse than keeping the (line-padded) segment edges.
  6. baseline: 20 s running median, two passes (pass 2 bridges pass-1 5-sigma events and C's detections,
     each +-2 s), subtracted (B step 5).
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage, signal

from . import params as PR
from ._vendor import preproc as BP

_H = None


def _resample_filter():
    """The exact filter scipy.signal.resample_poly designs for (25, 1208) with its default Kaiser(5) window."""
    global _H
    if _H is None:
        half_len = 10 * max(PR.UP, PR.DOWN)
        _H = (signal.firwin(2 * half_len + 1, 1.0 / max(PR.UP, PR.DOWN), window=("kaiser", 5.0)) * PR.UP, half_len)
    return _H


def grid_first(slot_a: int) -> int:
    """First run-global output index j with j*DOWN/UP >= slot_a."""
    return -(-(PR.UP * int(slot_a)) // PR.DOWN)


def resample_to_grid(x: np.ndarray, slot_a: int, padtype: str = PR.RESAMPLE_PADTYPE):
    """Polyphase resampling of x (samples at slots slot_a, slot_a+1, ...) by UP/DOWN, evaluated exactly at
    the run-global grid positions slot = j*DOWN/UP. Same filter as resample_poly; for slot_a a multiple of
    DOWN and padtype='constant' it reproduces resample_poly(x, UP, DOWN) (self-test).
    Returns (j_first, y)."""
    h, half_len = _resample_filter()
    up, down = PR.UP, PR.DOWN
    j_first = grid_first(slot_a)
    n0 = down * j_first - up * int(slot_a)          # integer offset in the upsampled domain, 0 <= n0 < down
    n_out = (up * (len(x) - 1) - n0) // down + 1
    if n_out <= 0:
        return j_first, np.zeros(0)
    p = (-(n0 + half_len)) % down
    q = (n0 + half_len + p) // down
    hp = np.concatenate([np.zeros(p), h])
    kw = dict(mode=padtype) if padtype != "constant" else dict(mode="constant", cval=0)
    y = signal.upfirdn(hp, x, up, down, **kw)
    if len(y) < q + n_out:
        y = np.concatenate([y, np.zeros(q + n_out - len(y))])
    return j_first, y[q:q + n_out]


def _detect_b(r, sig, mask, k_hi, k_lo=2.0, merge_s=1.0, fs=50.0):
    """B's det2.detect (B_signal/scripts/det2.py, copied verbatim in logic): hysteresis on |r|/sigma."""
    rr = np.where(mask, 0.0, r)
    lab, _ = ndimage.label(np.abs(rr) > k_lo * sig)
    evs = []
    for sl in ndimage.find_objects(lab):
        a, e = sl[0].start, sl[0].stop
        if np.any(np.abs(rr[a:e]) > k_hi * sig[a:e]):
            evs.append([a, e])
    out = []
    for a, e in evs:
        if out and a - out[-1][1] < merge_s * fs:
            out[-1][1] = e
        else:
            out.append([a, e])
    return out


def bridge(y, mask):
    y = y.copy()
    if mask.any() and (~mask).sum() >= 2:
        i = np.arange(len(y))
        y[mask] = np.interp(i[mask], i[~mask], y[~mask])
    elif mask.all():
        y[:] = 0.0
    return y


def feature_path(v: np.ndarray, slot: np.ndarray, clock, extra_event_slots=(), lp_hz=PR.FEATURE_LP_HZ,
                 gap_guard_s=PR.GAP_GUARD_S, padtype=PR.RESAMPLE_PADTYPE, bridge_guard=False, return_debug=False):
    """v: raw counts, slot: run-global slots (strictly increasing), clock: blockio.SlotClock of the run.
    extra_event_slots: iterable of (slot_on, slot_off) event extents (C's detections) bridged in pass 2.
    Returns dict(J0: run-global index of r[0], r: baseline-subtracted 50 Hz series (bridged, no NaN),
    mask: bool flag (hole or within gap_guard_s of a split), hole: bool (bridged samples), n_split, n_filled)."""
    step = PR.DOWN / PR.UP
    vc = v.astype(float) - np.median(v)
    lost = np.diff(slot) - 1
    brk = np.flatnonzero(lost >= PR.FILL_MAX + 1) + 1
    edges = np.r_[0, brk, len(v)]
    J0 = grid_first(slot[0])
    J1 = int(np.floor(slot[-1] * PR.UP / PR.DOWN))
    n50 = J1 - J0 + 1
    y50 = np.full(n50, np.nan)
    for a, b in zip(edges[:-1], edges[1:]):
        sa, sb = int(slot[a]), int(slot[b - 1])
        fs_seg = clock.fs_eff_mean(sa, sb + 1)[0] * PR.DOWN / PR.UP      # slot-weighted mean fs_local
        if sb - sa + 1 < PR.MIN_SEG_S * fs_seg:
            continue
        grid = np.arange(sa, sb + 1)
        x = vc[a:b] if len(grid) == b - a else np.interp(grid, slot[a:b], vc[a:b])
        xm = BP.remove_mains(x)                          # slot domain: f0 fitted per block, units consistent
        yl = BP.lowpass(xm, lp_hz, fs=fs_seg)
        j, y = resample_to_grid(yl, sa, padtype)
        k0, k1 = j - J0, min(n50, j - J0 + len(y))
        y50[k0:k1] = y[:k1 - k0]
    hole = np.isnan(y50)
    mask = hole.copy()
    # guard +-gap_guard_s around every split (the hole between slot[b-1] and slot[b])
    for b in brk:
        g = gap_guard_s * clock.fs_local(int(slot[b])) * PR.UP / PR.DOWN
        lo = int(np.floor(slot[b - 1] / step - g)) - J0
        hi = int(np.ceil(slot[b] / step + g)) - J0
        mask[max(lo, 0):max(min(hi + 1, n50), 0)] = True
    y = bridge(y50, mask if bridge_guard else hole)
    # two-pass 20 s running median (B)
    fs50 = 50.0
    b1 = BP.baseline(y, mask, win_s=PR.BASE_WIN_S, fs=fs50)
    r1 = y - b1
    sig1 = BP.robust_sigma(r1, mask)
    sig1 = np.where(np.isfinite(sig1), sig1, np.nanmedian(sig1) if np.isfinite(sig1).any() else 1.0)
    em = np.zeros(n50, bool)
    br = int(PR.PASS1_BRIDGE_S * fs50)
    for a, e in _detect_b(r1, sig1, mask, PR.PASS1_K):
        em[max(0, a - br):e + br] = True
    for so, sf in extra_event_slots:
        a, e = int(np.floor(so / step)) - J0, int(np.ceil(sf / step)) - J0
        em[max(0, a - br):max(0, e + br)] = True
    base = BP.baseline(y, mask, win_s=PR.BASE_WIN_S, fs=fs50, event_mask=em)
    r = y - base
    out = dict(J0=J0, r=r, mask=mask, hole=hole, n_split=len(brk), n_filled=int(((lost >= 1) & (lost <= PR.FILL_MAX)).sum()))
    if return_debug:
        out.update(y50_raw=y50, base=base, base1=b1, event_mask=em)
    return out
