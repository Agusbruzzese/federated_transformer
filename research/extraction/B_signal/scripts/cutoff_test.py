"""SUPERSEDED by cutoff2.py (v1 compared against the 20 Hz series, not raw). Q5: how low can the LP cutoff go? Peak / waveform preservation of real events vs the 20 Hz reference,
and noise σ after each cutoff; plus amplitude of the 9.45 Hz line family and of mains after the LP."""
import numpy as np, pandas as pd
from scipy import signal
ev = pd.read_parquet("../data/events_classified.parquet"); Z = np.load("../data/lp100.npz")["z"]
FS = 100.0
veh = ev[(ev.cls == "vehicle-like (≥0.4 s)")]
out = []
for fc in [1, 2, 3, 5, 8, 10]:
    sos = signal.butter(4, fc, fs=FS, output="sos")
    for _, e in veh.iterrows():
        z = Z[e.ci]; z = z[~np.isnan(z)].astype(float)
        a, b = int(e.a - 300), int(e.e + 300)
        if a < 0 or b > len(z): continue
        s = z[a:b] - np.median(z[a:b]); s20 = s
        sf = signal.sosfiltfilt(sos, s, padtype="even")
        k = np.argmax(np.abs(s20[300:-300])) + 300
        # reference: 20 Hz series but lightly denoised by 10-sample (0.1 s) median? use raw 20 Hz peak
        out.append(dict(fc=fc, amax=e.amax, peak_ratio=np.abs(sf).max() / np.abs(s20).max(),
                        corr=np.corrcoef(sf[250:-250], s20[250:-250])[0, 1]))
o = pd.DataFrame(out); o["grp"] = pd.cut(o.amax, [0, 1500, 5000, 1e7], labels=["small<1.5k", "medium 1.5-5k", "large>5k"])
print(o.groupby(["grp", "fc"]).agg(peak_med=("peak_ratio", "median"), peak_p5=("peak_ratio", lambda x: np.percentile(x, 5)),
                                    corr_med=("corr", "median"), corr_p5=("corr", lambda x: np.percentile(x, 5)), n=("corr", "size")).round(3).to_string())
# noise sigma after each cutoff on quiet chunks (15 s median detrend)
st = pd.read_parquet("../data/chunk_stats.parquet"); q = np.where(st.lp_maxdev < np.percentile(st.lp_maxdev, 40))[0][:400]
from scipy import ndimage
for fc in [1, 2, 3, 5, 8, 10, 20]:
    sg = []
    for i in q:
        z = Z[i]; z = z[~np.isnan(z)].astype(float)
        y = signal.sosfiltfilt(signal.butter(4, fc, fs=FS, output="sos"), z) if fc < 20 else z
        r = y - ndimage.median_filter(y, 2001, mode="nearest")
        sg.append(1.4826 * np.median(np.abs(r - np.median(r))))
    print(f"quiet-chunk noise σ after LP {fc} Hz + 20 s median detrend: median {np.median(sg):.1f}  p90 {np.percentile(sg, 90):.1f} counts")
# 9.45 Hz family amplitude from 41-s periodograms of quiet chunks
f = np.fft.rfftfreq(4096, 0.01); A = []
for i in q:
    z = Z[i][:4096].astype(float)
    if np.isnan(z).any(): continue
    X = np.fft.rfft((z - z.mean()) * np.hanning(4096)); amp = 2 * np.abs(X) / np.hanning(4096).sum()
    A.append([amp[(f > lo) & (f < hi)].max() for lo, hi in [(9.2, 9.7), (18.6, 19.2), (28, 28.6), (37.4, 38.1)]])
print("9.45/18.9/28.3/37.7 Hz line amplitude (counts, median over quiet chunks):", np.median(A, 0).round(1), " p90", np.percentile(A, 90, 0).round(1))
