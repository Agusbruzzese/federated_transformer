"""Look at short (<0.4 s) detections at raw resolution after harmonic (mains) regression."""
import sys, numpy as np, pandas as pd, io_utils as io
from scipy import signal
from style import plt, RUNC, INK2
FS = 2416.0
ev = pd.read_parquet("../data/events_lp100.parquet"); raw = pd.read_parquet("../data/chunk_stats.parquet")
C = io.list_chunks(); cmap = {(c["run"], c["idx"]): c for c in C}
sel = ev.query(sys.argv[1]).sample(6, random_state=int(sys.argv[2]))
def mains_clean(x, n):
    # estimate f0 from spectrum then regress 1..20 harmonics + cubic trend
    X = np.fft.rfft((x - x.mean()) * np.hanning(len(x)), 16 * len(x)); f = np.fft.rfftfreq(16 * len(x), 1 / FS)
    m = (f > 45) & (f < 55); f0 = f[m][np.argmax(abs(X[m]))]
    cols = [np.ones(len(n)), n, n**2, n**3] + [fn(2*np.pi*k*f0*n/FS) for k in range(1, 21) for fn in (np.cos, np.sin)]
    M = np.array(cols).T; co = np.linalg.lstsq(M, x, rcond=None)[0]
    return x - M[:, 4:] @ co[4:], f0
fig, ax = plt.subplots(6, 1, figsize=(14, 14))
for k, (_, e) in enumerate(sel.iterrows()):
    ch = cmap[(int(raw.run[e.ci]), int(raw.idx[e.ci]))]; v, t = io.load(ch)
    d = np.diff(t); gaps = d > io.GAP_US
    miss = np.zeros(len(t)); miss[1:] = np.where(gaps, np.round(d / np.median(d)) - 1, 0); tt = (np.arange(len(t)) + np.cumsum(miss)) / FS
    c0 = np.searchsorted(tt, e.a / 100); c1 = np.searchsorted(tt, e.e / 100)
    a, b = max(0, c0 - int(1.5 * FS)), min(len(v), c1 + int(1.5 * FS))
    n = np.arange(a, b).astype(float); x = v[a:b] - np.median(v[a:b])
    xc, f0 = mains_clean(x, n - n[0])
    ax[k].plot(tt[a:b], x, lw=.3, color=RUNC[0], alpha=.5, label="raw")
    ax[k].plot(tt[a:b], xc, lw=.6, color=RUNC[1], label="mains(20 harmonics) removed")
    ax[k].plot(tt[a:b], signal.sosfiltfilt(signal.butter(4, 20, fs=FS, output="sos"), xc), lw=1.2, color="#0b0b0b", label="+LP 20 Hz")
    for gg in np.where(gaps[a:b - 1])[0]: ax[k].axvline(tt[a + gg], color=RUNC[3], lw=2)
    ax[k].axvspan(e.a / 100, e.e / 100, color=RUNC[2], alpha=.15)
    ax[k].set_title(f"{e.start} run{e.run} dur {e.dur:.2f}s amax {e.amax:.0f} f0={f0:.3f}  (yellow = dropout)", fontsize=8)
ax[0].legend(fontsize=7, ncol=3)
fig.tight_layout(); fig.savefig(sys.argv[3], dpi=80)
