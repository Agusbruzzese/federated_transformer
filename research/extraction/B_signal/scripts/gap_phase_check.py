"""Check that timestamp gaps are real sample dropouts, using mains (50 Hz) phase continuity.
For each gap: fit A*cos(2*pi*f*n/fs + phi) to 0.5 s before and after (fs=FS_ADC, f=50),
predict the phase advance for the timestamp-reported missing time, compare."""
import numpy as np, io_utils as io
from scipy import signal
FS = 2416.0
C = io.list_chunks()
rng = np.random.default_rng(0)
rows = []
n_seen = n_skip = 0
for c in rng.choice(C, 60, replace=False):
    v, t = io.load(c)
    x = signal.sosfiltfilt(signal.butter(2, [45, 55], btype="band", fs=FS, output="sos"), v - v.mean())
    d = np.diff(t)
    for g in np.where(d > io.GAP_US)[0]:
        n_seen += 1
        if g < 1500 or g > len(v) - 1500:
            n_skip += 1
            continue
        def ph(seg, n0):
            n = np.arange(len(seg)) + n0
            M = np.c_[np.cos(2*np.pi*50*n/FS), np.sin(2*np.pi*50*n/FS)]
            a, b = np.linalg.lstsq(M, seg, rcond=None)[0]
            return np.arctan2(-b, a)
        # fit same-index model before, then after (indices continue as if no loss)
        p1 = ph(x[g-1200:g-100], g-1200)
        p2 = ph(x[g+101:g+1201], g+101)
        dphi = (p2 - p1) / (2*np.pi)  # cycles of phase slip w.r.t. no-loss hypothesis (mod 1)
        miss_ts = (d[g] - np.median(d)) * 1e-6
        rows.append((dphi % 1, (miss_ts*50) % 1, miss_ts*1e3))
r = np.array(rows)
err = ((r[:, 0] - r[:, 1] + 0.5) % 1) - 0.5
print("gaps seen", n_seen, "skipped (<1500 samples from chunk edge)", n_skip, "used", len(r), "missing ms median", np.median(r[:, 2]))
print("phase slip predicted-vs-observed error (cycles): median |err|", np.median(np.abs(err)), " frac |err|<0.1:", np.mean(np.abs(err) < 0.1))
print("null (no-loss) |slip| median", np.median(np.abs(((r[:, 0] + .5) % 1) - .5)))
print("observed slip hist (cycles, 0..1, 10 bins):", np.histogram(r[:, 0], bins=10, range=(0, 1))[0])
print("predicted slip hist:", np.histogram(r[:, 1], bins=10, range=(0, 1))[0])
# implied missing samples = slip cycles * FS/50 (+k*48.3)
