"""OBSOLETE (v1 quick-look only): uses a tick-derived 2381 Hz and treats ticks as µs. Superseded by preproc.py / chunk_stats.py. Kept for provenance; do not use.
"""
import sys, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from datetime import datetime
import io_utils as io
from scipy import signal
C = io.list_chunks()
def find(ts):
    t = datetime.strptime(ts, "%Y%m%d %H%M"); return min(C, key=lambda c: abs((c["start"]-t).total_seconds()))
targets = sys.argv[1:]
fig, ax = plt.subplots(len(targets), 2, figsize=(16, 3*len(targets)), squeeze=False)
for k, ts in enumerate(targets):
    ch = find(ts); v, t = io.load(ch)
    tt = (t - t[0])/1e6
    ax[k,0].plot(tt, v, lw=0.4); ax[k,0].set_title(f"{ch['start']} run{ch['run']} idx{ch['idx']}")
    f, P = signal.welch(v - np.median(v), fs=2381, nperseg=8192); ax[k,1].loglog(f[1:], P[1:], lw=0.6)
    print(ts, ch['start'], 'median', np.median(v), 'std', v.std(), 'p1/p99', np.percentile(v,[1,99]), 'min/max', v.min(), v.max(),
          'diffMADsig', 1.4826*np.median(abs(np.diff(v)))/np.sqrt(2))
fig.tight_layout(); fig.savefig("../figures/_quicklook.png", dpi=80)
