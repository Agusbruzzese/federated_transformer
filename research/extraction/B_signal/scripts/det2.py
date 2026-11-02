"""Transient detector and morphology descriptors on the 50 Hz baseline-subtracted series (v2).
Hysteresis: core |r| > k_hi*sigma, extent while |r| > k_lo*sigma, merge if gap < merge_s."""
import numpy as np
from scipy import ndimage

FS = 50.0


def detect(r, sig, mask, k_hi=8.0, k_lo=2.0, merge_s=1.0, fs=FS):
    rr = np.where(mask, 0.0, r)
    lo = np.abs(rr) > k_lo * sig
    lab, n = ndimage.label(lo)
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


def lobes(s, frac=0.2):
    amax = np.abs(s).max()
    sg = np.sign(s)
    pat = []
    i = 0
    n = len(s)
    while i < n:
        j = i
        while j < n and sg[j] == sg[i]:
            j += 1
        if sg[i] != 0 and np.abs(s[i:j]).max() > frac * amax:
            c = "+" if sg[i] > 0 else "-"
            if not pat or pat[-1] != c:
                pat.append(c)
        i = j
    return "".join(pat)


def shape(s, fs=FS):
    """rise (10-90 % on the leading edge of the dominant lobe), FWHM of dominant lobe,
    90 %-energy duration, lobe pattern. s: baseline-removed event samples (with flanks)."""
    s = np.asarray(s, float)
    if np.isnan(s).any():  # masked samples inside an event: bridge linearly (flagged via mask_frac)
        ok = ~np.isnan(s)
        s = np.interp(np.arange(len(s)), np.where(ok)[0], s[ok]) if ok.sum() > 1 else np.zeros(len(s))
    ip = int(np.argmax(np.abs(s)))
    pol = np.sign(s[ip]) or 1.0
    x = pol * s
    pk = x[ip]
    j = ip
    while j > 0 and x[j] > 0.1 * pk:
        j -= 1
    seg = x[j:ip + 1]
    # sub-sample interpolation of the 10 % and 90 % crossings
    def cross(level):
        k = int(np.argmax(seg >= level * pk))
        if k == 0:
            return 0.0
        y0, y1 = seg[k - 1], seg[k]
        return k - 1 + (level * pk - y0) / max(y1 - y0, 1e-12)
    rise = (cross(0.9) - cross(0.1)) / fs
    above = x > 0.5 * pk
    lab, _ = ndimage.label(above)
    fwhm = (lab == lab[ip]).sum() / fs
    c = np.cumsum(s ** 2)
    c /= c[-1]
    d90 = (np.searchsorted(c, .95) - np.searchsorted(c, .05)) / fs
    return dict(rise=rise, fwhm=fwhm, d90=d90, pattern=lobes(s), nlobes=len(lobes(s)),
                dom_pol=int(pol), amax=float(np.abs(s).max()))


def describe(r, a, e, sig, mask, fs=FS):
    s = r[a:e]
    d = shape(s, fs)
    d.update(dur=(e - a) / fs, pos=float(np.nanmax(s)), neg=float(np.nanmin(s)), p2p=float(np.nanmax(s) - np.nanmin(s)),
             snr=float(np.nanmax(np.abs(s)) / np.median(sig[a:e])), sig=float(np.median(sig[a:e])),
             mask_frac=float(mask[a:e].mean()))
    # distance (s) to the nearest masked sample or span edge
    bad = np.where(mask)[0]
    left = a - (bad[bad < a].max() if (bad < a).any() else -1) - 1
    right = ((bad[bad >= e].min() if (bad >= e).any() else len(r)) - e)
    d["edge_dist"] = min(left, right) / fs
    return d
