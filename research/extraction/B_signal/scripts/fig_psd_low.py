"""Q2b: low-frequency PSD (0.025-45 Hz) from the 100 Hz low-passed series; 9.5 Hz line family."""
import numpy as np, pandas as pd
from scipy import signal, ndimage
from style import plt, RUNC, INK2
L = np.load("../data/lp100.npz"); Z = L["z"]
raw = pd.read_parquet("../data/chunk_stats.parquet")
n = 4096
ok = ~np.isnan(Z[:, :n]).any(1)
X = Z[ok, :n].astype(float)
st = raw[ok].reset_index(drop=True)
f, Pw = signal.welch(X, fs=100, nperseg=1024, noverlap=512, detrend="linear", axis=1)  # 10.24 s segments
_, Pl = signal.welch(X, fs=100, nperseg=4096, detrend="linear", axis=1)          # 41 s, 1 segment (periodogram)
fl = np.fft.rfftfreq(4096, 0.01)
hr = st.start.dt.hour.values
quiet = (st.lp_maxdev < np.percentile(st.lp_maxdev, 40)).values
ev = (st.lp_maxdev > 1e4).values
night = (hr >= 1) & (hr < 5)
fig, ax = plt.subplots(1, 2, figsize=(15, 5))
ax[0].loglog(fl[1:], np.median(Pl[quiet & night], 0)[1:], color=RUNC[0], lw=.8, label="quiet night (41 s periodogram, median)")
ax[0].loglog(fl[1:], np.median(Pl[quiet & ~night], 0)[1:], color=RUNC[1], lw=.8, label="quiet other hours")
ax[0].loglog(fl[1:], np.median(Pl[ev], 0)[1:], color=RUNC[2], lw=.8, label="chunks containing an event >1e4 counts (median)")
ax[0].loglog(fl[1:], np.percentile(Pl[ev], 90, 0)[1:], color=RUNC[3], lw=.8, label="  … 90th percentile")
ax[0].set_xlabel("frequency (Hz)"); ax[0].set_ylabel("PSD (counts²/Hz)"); ax[0].legend(fontsize=8)
ax[0].set_title("Low-frequency PSD of the 20 Hz-LP / 100 Hz series")
# SNR-type curve: event median PSD / quiet PSD
r = np.median(Pl[ev], 0) / np.median(Pl[quiet], 0)
rs = ndimage.uniform_filter1d(r, 9)
ax[1].loglog(fl[1:], rs[1:], color=RUNC[0], lw=1)
ax[1].axhline(1, color=INK2, lw=.6)
ax[1].set_xlabel("frequency (Hz)"); ax[1].set_ylabel("PSD ratio  event chunks / quiet chunks"); ax[1].set_title("Where vehicle energy lies (ratio of medians, smoothed)")
fig.tight_layout(); fig.savefig("../figures/fig4_psd_lowfreq.png", dpi=110)
for fr in [0.03, 0.05, 0.1, 0.2, 0.5, 1, 2, 3, 5, 8, 10, 15, 20]:
    k = np.argmin(abs(fl - fr)); print(f"{fr} Hz: ratio {rs[k]:.1f}  quiet {np.median(Pl[quiet],0)[k]:.0f}  event-median {np.median(Pl[ev],0)[k]:.0f}")
# cumulative event energy above quiet: fraction of excess energy below f
exc = np.clip(np.median(Pl[ev], 0) - np.median(Pl[quiet], 0), 0, None)
c = np.cumsum(exc) / exc.sum()
for q in [.5, .9, .95, .99, .999]:
    print(f"{q*100:.1f}% of median excess (event) power below {fl[np.searchsorted(c, q)]:.3f} Hz")
# 9.5-Hz family exact frequencies from 41-s periodograms
qm = np.median(Pl[quiet], 0)
for lo, hi in [(8.5, 10.5), (18, 20.5), (27.5, 30), (37, 40)]:
    m = (fl > lo) & (fl < hi); k = np.argmax(qm[m]); print("line", fl[m][k], "Hz, prominence x", qm[m][k] / np.median(qm[m]))
# presence of 9.5 Hz line vs hour
m = (fl > 9.0) & (fl < 10.2); mm = (fl > 7) & (fl < 12)
prom = Pl[:, m].max(1) / np.median(Pl[:, mm], 1)
print("9.5Hz-line prominence median by hour:", pd.Series(prom).groupby(hr).median().round(1).to_dict())
print("peak freq spread (quiet):", np.percentile(fl[m][Pl[quiet][:, m].argmax(1)], [10, 50, 90]))
