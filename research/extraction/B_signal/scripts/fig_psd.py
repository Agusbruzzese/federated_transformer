"""Q2: noise PSD of quiet chunks, tonal lines, 1/f, day vs night, and event-band vs noise."""
import numpy as np, pandas as pd
from scipy import signal, ndimage
from style import plt, RUNC, INK2
d = pd.read_parquet("../data/chunk_stats_enriched.parquet")
raw = pd.read_parquet("../data/chunk_stats.parquet")
P = np.load("../data/psd.npz"); f = P["f"]; Pmed = P["Pmed"]; Pmean = P["Pmean"]
# align enriched (sorted) with raw order
key = raw.run.astype(str) + "_" + raw.idx.astype(str)
order = {k: i for i, k in enumerate(key)}
ii = np.array([order[f"{r}_{x}"] for r, x in zip(d.run, d.idx)])
Pmed, Pmean = Pmed[ii], Pmean[ii]
hr = d.start.dt.hour.values
quiet = (d.lp_maxdev < np.percentile(d.lp_maxdev, 40)).values
night = (hr >= 1) & (hr < 5); day = (hr >= 9) & (hr < 17)
eventful = (d.lp_maxdev > 1e4).values
q_all = np.median(Pmed[quiet], 0)
q_n = np.median(Pmed[quiet & night], 0); q_d = np.median(Pmed[quiet & day], 0)
ev = np.median(Pmean[eventful], 0)
df = f[1] - f[0]
fig, ax = plt.subplots(1, 2, figsize=(15, 5))
ax[0].loglog(f[1:], q_n[1:], color=RUNC[0], lw=.8, label=f"quiet night 01–05h (n={int((quiet&night).sum())})")
ax[0].loglog(f[1:], q_d[1:], color=RUNC[1], lw=.8, label=f"quiet day 09–17h (n={int((quiet&day).sum())})")
ax[0].loglog(f[1:], ev[1:], color=RUNC[2], lw=.8, label=f"chunks with event >1e4 counts (n={int(eventful.sum())}), mean-avg")
ax[0].set_xlabel("frequency (Hz)"); ax[0].set_ylabel("PSD (counts²/Hz)"); ax[0].legend(fontsize=8)
ax[0].set_title("Welch PSD (nperseg 4096 = 1.7 s, Δf = 0.59 Hz), median over chunks")
# tonal lines: ratio to running median floor
floor = ndimage.median_filter(q_all, 31)
ratio = 10 * np.log10(q_all / floor)
pk, _ = signal.find_peaks(ratio, height=10)
lines = [(round(f[k], 2), round(ratio[k], 1), round(float(np.sqrt(2 * q_all[k] * df * 1.5)), 1)) for k in pk]  # approx amplitude (hann ENBW 1.5 bins)
ax[1].semilogx(f[1:], ratio[1:], color=RUNC[0], lw=.7)
for k in pk:
    if ratio[k] > 20: ax[1].annotate(f"{f[k]:.1f}", (f[k], ratio[k]), fontsize=6, color=INK2)
ax[1].set_xlabel("frequency (Hz)"); ax[1].set_ylabel("dB above running-median floor"); ax[1].set_title("Tonal lines in quiet chunks")
fig.tight_layout(); fig.savefig("../figures/fig3_psd_noise.png", dpi=110)
print("tonal lines (f, dB above floor, approx amplitude counts):"); print(lines)
def bandpow(Pv, lo, hi):
    m = (f >= lo) & (f < hi); return Pv[m].sum() * df
for lo, hi in [(0.59, 1), (1, 2), (2, 5), (5, 10), (10, 20), (20, 45), (45, 55), (55, 140), (140, 160), (160, 400), (400, 1208)]:
    print(f"band {lo}-{hi} Hz: quiet night rms {np.sqrt(bandpow(q_n,lo,hi)):.1f}, quiet day {np.sqrt(bandpow(q_d,lo,hi)):.1f}, event chunks {np.sqrt(bandpow(ev,lo,hi)):.1f}")
# floor level (excluding lines) at a few frequencies
for fr in [1, 2, 5, 10, 20, 30, 40, 100, 300, 600, 1000]:
    k = np.argmin(abs(f - fr)); print(f"floor @{fr} Hz: night {floor[k]:.0f}  day {np.median(ndimage.median_filter(q_d,31)[k]):.0f} counts²/Hz -> {np.sqrt(floor[k]):.1f} counts/√Hz")
# 1/f slope fit 0.6-10 Hz on floor
m = (f > 0.6) & (f < 10)
print("log-log slope floor 0.6-10 Hz:", np.polyfit(np.log10(f[m]), np.log10(ndimage.median_filter(q_n,31)[m]), 1)[0])
m = (f > 10) & (f < 45)
print("log-log slope floor 10-45 Hz:", np.polyfit(np.log10(f[m]), np.log10(ndimage.median_filter(q_n,31)[m]), 1)[0])
np.savez("../data/psd_summary.npz", f=f, q_n=q_n, q_d=q_d, ev=ev, q_all=q_all)
