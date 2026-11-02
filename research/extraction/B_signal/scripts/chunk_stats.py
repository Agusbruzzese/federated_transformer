"""Per-chunk statistics over a stratified subsample (every STEP-th chunk of every run).

Outputs (in ../data):
  chunk_stats.parquet   one row per sampled chunk (timing, raw stats, mains, LP stats)
  psd.npz               Welch PSDs (mean- and median-averaged) per chunk, float32
  lp100.npz             20-Hz low-passed signal resampled onto a uniform 100 Hz grid from the
                        timestamps (relative to chunk median, float32) -> used for event analysis
Usage: python chunk_stats.py [STEP] [NPROC]
"""
import sys, os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "1")
import numpy as np, pandas as pd
from multiprocessing import Pool
from scipy import signal, ndimage
import io_utils as io

STEP = int(sys.argv[1]) if len(sys.argv) > 1 else 8
NPROC = int(sys.argv[2]) if len(sys.argv) > 2 else 4
FS_NOM = 2416.0          # ADC rate implied by the 50 Hz mains line (host µs timestamps are 1-5% off; see REPORT)
NPER = 4096
FS_OUT = 100.0
OUT = os.path.join(os.path.dirname(__file__), "..", "data")
SOS_LP = signal.butter(4, 20, fs=FS_NOM, output="sos")
SOS_HP = signal.butter(2, 0.5, fs=FS_NOM, output="sos")  # only for high-band stats


def mains_peak(x, fs):
    """Precise frequency/amplitude of the ~50 Hz line from a zero-padded FFT of a contiguous run."""
    n = len(x)
    w = np.hanning(n)
    X = np.fft.rfft((x - x.mean()) * w, n=8 * n)
    f = np.fft.rfftfreq(8 * n, 1 / fs)
    m = (f > 45) & (f < 55)
    k = np.argmax(np.abs(X[m]))
    amp = 2 * np.abs(X[m][k]) / w.sum()
    return f[m][k], amp


def one(ch):
    v, t = io.load(ch)
    r = dict(run=ch["run"], idx=ch["idx"], start=ch["start"])
    d = np.diff(t)
    gaps = d > io.GAP_US
    r.update(t_first=t[0], t_last=t[-1], dur_s=(t[-1] - t[0]) / 1e6, n_gaps=int(gaps.sum()),
             gap_ms=float((d[gaps] - np.median(d)).sum() / 1e3), dt_med=float(np.median(d)),
             dt_mean_ok=float(d[~gaps].mean()), dt_min=int(d.min()), n_nonmono=int((d <= 0).sum()))
    fs = 1e6 / r["dt_mean_ok"]
    r["fs_eff"] = fs
    q = np.percentile(v, [0.1, 1, 5, 25, 50, 75, 95, 99, 99.9])
    for name, val in zip(["p001", "p01", "p05", "p25", "p50", "p75", "p95", "p99", "p999"], q):
        r[name] = val
    r.update(mean=v.mean(), std=v.std(), vmin=v.min(), vmax=v.max(),
             nsat=int(((v <= 0) | (v >= 2**24 - 1)).sum()), above24=int((v >= 2**24).sum()),
             diffsig=1.4826 * np.median(np.abs(np.diff(v))) / np.sqrt(2))
    # mains line from the longest contiguous segment
    segs = io.segments(t)
    a, b = max(segs, key=lambda s: s[1] - s[0])
    r["seg_max"] = b - a
    r["mains_f"], r["mains_amp"] = mains_peak(v[a:b], fs)
    r["mains_f_adc"] = mains_peak(v[a:b], FS_NOM)[0]
    # Welch PSD (counts^2/Hz) mean and median averaged
    x = v - np.median(v)
    f, Pm = signal.welch(x, fs=FS_NOM, nperseg=NPER, average="mean")
    _, Pd = signal.welch(x, fs=FS_NOM, nperseg=NPER, average="median")
    # low-pass + uniform resampling at 100 Hz on the timestamp axis
    # Per contiguous (dropout-free) segment: (1) subtract mains by harmonic regression (f0 from this
    # chunk, 20 harmonics, <=10 s blocks, cubic trend kept), (2) zero-phase LP with even padding.
    # Filtering across a dropout (phase-jumped 50 Hz) or odd-padding a segment that ends on a mains
    # peak both create spurious ~0.2 s / ~1000-count blips (see REPORT §4).
    f0 = r["mains_f_adc"]
    y = np.empty_like(x)
    for a0, b0 in segs:
        xs = x[a0:b0].copy()
        for c0 in range(0, len(xs), 24160):
            blk = xs[c0:c0 + 24160]
            if len(blk) < 2 * 49:
                continue
            n = np.arange(len(blk)) / FS_NOM
            H = np.array([fn(2 * np.pi * k * f0 * n) for k in range(1, 21) for fn in (np.cos, np.sin)]).T
            nn = (n - n.mean()) / max(np.ptp(n), 1e-9)
            M = np.c_[np.ones(len(n)), nn, nn**2, nn**3, H]
            co = np.linalg.solve(M.T @ M, M.T @ blk)
            xs[c0:c0 + 24160] = blk - H @ co[4:]
        y[a0:b0] = signal.sosfiltfilt(SOS_LP, xs, padtype="even", padlen=min(b0 - a0 - 1, 3 * 2416)) if b0 - a0 > 30 else np.median(xs)
    # time axis: sample index at the ADC rate, plus the samples lost in each timestamp gap
    miss = np.zeros(len(t))
    miss[1:] = np.where(gaps, np.round(d / r["dt_med"]) - 1, 0)
    tt = (np.arange(len(t)) + np.cumsum(miss)) / FS_NOM
    grid = np.arange(0, tt[-1], 1 / FS_OUT)
    z = np.interp(grid, tt, y)
    # dropout mask on the 100 Hz grid: within 0.15 s of any missing interval, plus short segments (<0.5 s)
    bad = np.zeros(len(grid), bool)
    for a0, b0 in segs:
        if b0 - a0 < 0.5 * FS_NOM:
            bad |= (grid >= tt[a0] - 0.15) & (grid <= tt[b0 - 1] + 0.15)
    for g in np.where(gaps)[0]:
        bad |= (grid >= tt[g] - 0.15) & (grid <= tt[g + 1] + 0.15)
    r["frac_bad100"] = float(bad.mean())
    # LP-domain stats: slow baseline vs fast residual
    base = ndimage.median_filter(z, 1001, mode="nearest")  # 10 s running median
    res = z - base
    sig_lp = 1.4826 * np.median(np.abs(res - np.median(res)))
    r.update(lp_sig=sig_lp, lp_maxdev=float(np.abs(res).max()), lp_pos=float(res.max()), lp_neg=float(res.min()),
             lp_frac5=float((np.abs(res) > 5 * sig_lp).mean()),
             base_range=float(base.max() - base.min()),
             first5=float(np.median(z[:500])), last5=float(np.median(z[-500:])))
    # high band (>0.5Hz, raw) std, and std of LP in quiet (1-s window minimum)
    wstd = z[: len(z) // 100 * 100].reshape(-1, 100).std(1)
    r.update(lp_wstd_min=float(wstd.min()), lp_wstd_med=float(np.median(wstd)), lp_wstd_max=float(wstd.max()))
    return r, Pm.astype(np.float32), Pd.astype(np.float32), z.astype(np.float32), bad


if __name__ == "__main__":
    C = io.list_chunks()
    sel = []
    for run in sorted(set(c["run"] for c in C)):
        rc = [c for c in C if c["run"] == run]
        sel += rc[::STEP]
    print(len(sel), "chunks", flush=True)
    with Pool(NPROC) as p:
        res = p.map(one, sel, chunksize=16)
    df = pd.DataFrame([r[0] for r in res])
    df.to_parquet(os.path.join(OUT, "chunk_stats.parquet"))
    f = np.fft.rfftfreq(NPER, 1 / FS_NOM)
    np.savez_compressed(os.path.join(OUT, "psd.npz"), f=f, Pmean=np.stack([r[1] for r in res]),
                        Pmed=np.stack([r[2] for r in res]))
    L = max(len(r[3]) for r in res)
    Z = np.full((len(res), L), np.nan, np.float32)
    B = np.ones((len(res), L), bool)
    for i, r in enumerate(res):
        Z[i, : len(r[3])] = r[3]
        B[i, : len(r[4])] = r[4]
    np.savez_compressed(os.path.join(OUT, "lp100.npz"), z=Z, bad=B, p50=df.p50.values, start=df.start.values.astype("datetime64[us]").astype("int64"),
                        run=df.run.values, idx=df.idx.values)
    print(df.describe().T.to_string())
