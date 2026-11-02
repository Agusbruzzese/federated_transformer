import sys, numpy as np, pandas as pd, io_utils as io
from scipy import signal
from style import plt, RUNC
ev = pd.read_parquet("../data/events_lp100.parquet"); raw = pd.read_parquet("../data/chunk_stats.parquet")
C = io.list_chunks(); cmap = {(c["run"], c["idx"]): c for c in C}
sel = ev.query(sys.argv[1]).sample(8, random_state=int(sys.argv[3]) if len(sys.argv) > 3 else 1)
fig, ax = plt.subplots(8, 2, figsize=(15, 16))
FS = 2416.0
for k, (_, e) in enumerate(sel.iterrows()):
    ch = cmap[(int(raw.run[e.ci]), int(raw.idx[e.ci]))]; v, t = io.load(ch)
    c0 = int(e.a / 100 * FS); c1 = int(e.e / 100 * FS)
    a, b = max(0, c0 - int(3 * FS)), min(len(v), c1 + int(3 * FS))
    tt = np.arange(a, b) / FS
    x = v[a:b] - np.median(v[a:b])
    y = signal.sosfiltfilt(signal.butter(4, 20, fs=FS, output="sos"), x)
    ax[k, 0].plot(tt, x, lw=.3, color=RUNC[0]); ax[k, 0].plot(tt, y, lw=1, color=RUNC[1])
    ax[k, 0].axvspan(c0 / FS, c1 / FS, color=RUNC[3], alpha=.2)
    ax[k, 0].set_title(f"{e.start} dur {e.dur:.2f}s amax {e.amax:.0f} pat {e.pattern} snr {e.snr:.1f}", fontsize=8)
    m = slice(max(0, c0 - int(.3 * FS)), min(len(v), c1 + int(.3 * FS)))
    ax[k, 1].plot(np.arange(len(v))[m] / FS, v[m] - np.median(v[m]), lw=.5, color=RUNC[0])
    ax[k, 1].plot(np.arange(len(v))[m] / FS, signal.sosfiltfilt(signal.butter(4, 20, fs=FS, output="sos"), v - np.median(v))[m] - np.median(v[m]) + np.median(v), lw=1, color=RUNC[1])
fig.tight_layout(); fig.savefig(sys.argv[2], dpi=80)
