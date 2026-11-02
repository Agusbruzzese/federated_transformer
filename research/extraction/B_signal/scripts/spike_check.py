"""Impulsive-spike census (was an inline check in v1; saved for reproducibility; seed 1).
150 random chunks; longest dropout-free segment; mains removed (preproc.remove_mains);
residual vs a 49-sample (20 ms) running median; spikes = clusters with |res| > 10 MAD-sigma.
Output: data/out_spikes.txt"""
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "1")
import numpy as np, io_utils as io, preproc as P
from scipy import ndimage, signal
C = io.list_chunks(); rng = np.random.default_rng(1); sel = rng.choice(len(C), 150, replace=False)
rates, amps, widths = [], [], []
for i in sel:
    v, t = io.load(C[i]); x = v - np.median(v)
    s_, segs, _ = P.slots(t); a, b = max(segs, key=lambda s: s[1] - s[0])
    xs = P.remove_mains(x[a:b])
    r = xs - ndimage.median_filter(xs, 49)
    s = 1.4826 * np.median(np.abs(r)); m = np.abs(r) > 10 * s
    lab, n = ndimage.label(ndimage.binary_dilation(m, iterations=2))
    rates.append(n / ((b - a) / P.FS_ADC) * 3600)
    for sl in ndimage.find_objects(lab):
        amps.append(np.abs(r[sl]).max()); widths.append((sl[0].stop - sl[0].start - 4) / P.FS_ADC * 1e3)
imp = np.zeros(24160); imp[12000:12005] = 2000
L = [f"spikes per hour: median {np.median(rates):.0f}, p10/p90 {np.percentile(rates, [10, 90]).round(0)}",
     f"amplitude p10/p50/p90/p99 {np.percentile(amps, [10, 50, 90, 99]).round(0)} counts; width p10/p50/p90 {np.percentile(widths, [10, 50, 90]).round(1)} ms",
     f"a 2000-count, 2 ms impulse after LP 5 Hz: {np.abs(P.lowpass(imp, 5)).max():.1f} counts; after LP 8 Hz: {np.abs(P.lowpass(imp, 8)).max():.1f}"]
print("\n".join(L)); open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "out_spikes.txt"), "w").write("\n".join(L))
