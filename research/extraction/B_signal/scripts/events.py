"""Q3: simple, transparent transient detector on the 100 Hz series, and morphology descriptors.
Detector (for characterisation only, not the production detector):
  y   = 4th-order Butterworth LP 5 Hz (zero-phase) of the 20-Hz-LP 100 Hz series
  b   = running median, 15 s window (baseline)            r = y - b
  σ   = 1.4826·MAD(r) per chunk (robust noise)
  event core: |r| > K_HI·σ ; extent: expand while |r| > K_LO·σ ; merge if gap < MERGE s
"""
import numpy as np, pandas as pd
from scipy import signal, ndimage
FS = 100.0
SOS5 = signal.butter(4, 5, fs=FS, output="sos")
K_HI, K_LO, MERGE = 8.0, 2.0, 1.0


def detrend(z, win_s=15):
    y = signal.sosfiltfilt(SOS5, z)
    w = int(win_s * FS) | 1
    b = ndimage.median_filter(y, w, mode="nearest")
    return y, b, y - b


def detect(z, k_hi=K_HI, k_lo=K_LO, merge=MERGE, win_s=15):
    y, b, r = detrend(z, win_s)
    sig = 1.4826 * np.median(np.abs(r - np.median(r)))
    lo = np.abs(r) > k_lo * sig
    lab, n = ndimage.label(lo)
    evs = []
    for sl in ndimage.find_objects(lab):
        a, e = sl[0].start, sl[0].stop
        if np.abs(r[a:e]).max() > k_hi * sig:
            evs.append([a, e])
    # merge close events
    out = []
    for a, e in evs:
        if out and a - out[-1][1] < merge * FS:
            out[-1][1] = e
        else:
            out.append([a, e])
    return out, r, sig, y, b


def describe(r, a, e, sig):
    s = r[a:e]; n = len(s)
    pk = s.max(); tr = s.min(); amax = max(pk, -tr)
    # lobes: contiguous same-sign regions whose |peak| > 20 % of amax
    sg = np.sign(s); lab, nl = ndimage.label(sg != 0)
    lobes = []
    cur = None
    for i in range(n):
        if cur is None or sg[i] != cur[0]:
            if cur is not None: lobes.append(cur)
            cur = [sg[i], i, i, abs(s[i])]
        else:
            cur[2] = i; cur[3] = max(cur[3], abs(s[i]))
    lobes.append(cur)
    big = [l for l in lobes if l[3] > 0.2 * amax]
    # merge adjacent same-sign big lobes (small opposite wiggles in between)
    pat = []
    for l in big:
        if pat and pat[-1] == ("+" if l[0] > 0 else "-"): continue
        pat.append("+" if l[0] > 0 else "-")
    # rise time 10-90 % on the dominant lobe, measured on the leading edge
    ip = np.argmax(np.abs(s)); pol = np.sign(s[ip]); x = pol * s
    j = ip
    while j > 0 and x[j] > 0.1 * x[ip]: j -= 1
    seg = x[j:ip + 1]
    t10 = np.argmax(seg >= 0.1 * x[ip]); t90 = np.argmax(seg >= 0.9 * x[ip])
    # FWHM of dominant lobe
    above = x > 0.5 * x[ip]; labh, _ = ndimage.label(above); fw = (labh == labh[ip]).sum() / FS
    # energy-based duration: time containing central 90 % of energy
    c = np.cumsum(s ** 2); c /= c[-1]
    d90 = (np.searchsorted(c, .95) - np.searchsorted(c, .05)) / FS
    # spectral centroid / 95 % bandwidth of the event (zero-padded)
    S = np.abs(np.fft.rfft(s - s.mean(), 2048)) ** 2; ff = np.fft.rfftfreq(2048, 1 / FS)
    cs = np.cumsum(S) / S.sum()
    return dict(dur=n / FS, d90=d90, pos=pk, neg=tr, amax=amax, p2p=pk - tr, snr=amax / sig,
                pattern="".join(pat), nlobes=len(pat), dom_pol=int(pol), rise=(t90 - t10) / FS, fwhm=fw,
                bw95=ff[np.searchsorted(cs, .95)], bw99=ff[np.searchsorted(cs, .99)],
                fcent=(ff * S).sum() / S.sum())


def run_lp100():
    L = np.load("../data/lp100.npz")
    Z, start, run, idx, BAD = L["z"], L["start"], L["run"], L["idx"], L["bad"]
    rows = []; sigs = []
    for i in range(len(Z)):
        z = Z[i][~np.isnan(Z[i])].astype(float)
        evs, r, sig, y, b = detect(z)
        sigs.append(dict(ci=i, sig=sig, nev=len(evs), start=pd.Timestamp(int(start[i]), unit="us"), run=run[i], idx=idx[i],
                         base_rng=b.max() - b.min()))
        for a, e in evs:
            dsc = describe(r, a, e, sig)
            dsc.update(dropout=bool(BAD[i, a:e].any()), ci=i, a=a, e=e, t_s=a / FS, edge=(a == 0) or (e >= len(z) - 1),
                       start=pd.Timestamp(int(start[i]), unit="us") + pd.Timedelta(seconds=a / FS), run=run[i], sig=sig)
            rows.append(dsc)
    ev = pd.DataFrame(rows); ch = pd.DataFrame(sigs)
    ev.to_parquet("../data/events_lp100.parquet"); ch.to_parquet("../data/chunk_detect.parquet")
    return ev, ch


if __name__ == "__main__":
    ev, ch = run_lp100()
    BADFRAC = np.load("../data/lp100.npz")["bad"].mean()
    print(len(ev), "events in", len(ch), "chunks; hours covered", len(ch) * 41.4 / 3600)
    print("noise sigma (0-5Hz, 15s-median detrended) pct:", np.percentile(ch.sig, [1, 10, 50, 90, 99]).round(1))
    print("events overlapping a dropout mask:", ev.dropout.sum(), "clean:", (~ev.dropout).sum())
    print("frac of 100Hz grid masked:", BADFRAC)
    e = ev[~ev.edge & ~ev.dropout]
    print("non-truncated", len(e))
    print(e[["dur", "d90", "amax", "p2p", "snr", "nlobes", "rise", "fwhm", "bw95", "bw99", "fcent"]].describe(percentiles=[.05, .25, .5, .75, .95]).T.round(2).to_string())
    print(e.pattern.value_counts().head(12))
