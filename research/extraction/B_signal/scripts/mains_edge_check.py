"""Issue 8: do mains-regression block edges create artefacts? 60 random dropout-free chunks (runs 1-3).
Compare hard 10 s blocks (v1, chunk_stats.py) with 50 %-overlap cross-faded blocks (v2, preproc.py):
 (a) residual 50 Hz amplitude (45-55 Hz band envelope) within ±0.25 s of a hard-block edge vs elsewhere
 (b) |LP 5 Hz| step: median |y(edge+0.1 s) - y(edge-0.1 s)| vs the same statistic at random positions
 (c) residual-mains spectrum (Welch) of v1 vs v2 around 50/150/250 Hz
Output: data/out_mains_edge.txt"""
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "1")
import numpy as np
from scipy import signal
import io_utils as io, preproc as P
FS = P.FS_ADC; B = int(10 * FS)
C = [c for c in io.list_chunks() if c["run"] > 0]
rng = np.random.default_rng(5)
bp = signal.butter(4, [45, 55], btype="band", fs=FS, output="sos")
res = {"hard": dict(env_edge=[], env_mid=[], step_edge=[], step_rand=[], P=[]), "overlap": dict(env_edge=[], env_mid=[], step_edge=[], step_rand=[], P=[])}
n = 0
for c in rng.permutation(len(C)):
    v, t = io.load(C[c])
    if (np.diff(t) > 1.5 * np.median(np.diff(t))).any():
        continue
    x = v - np.median(v)
    for mode, ov in (("hard", False), ("overlap", True)):
        xm = P.remove_mains(x, overlap=ov)
        env = np.abs(signal.hilbert(signal.sosfiltfilt(bp, xm)))
        y = P.lowpass(xm, 5.0)
        edges = np.arange(B, len(x) - B // 2, B)
        near = np.zeros(len(x), bool)
        for e in edges:
            near[e - int(.25 * FS):e + int(.25 * FS)] = True
        res[mode]["env_edge"].append(np.median(env[near])); res[mode]["env_mid"].append(np.median(env[~near]))
        d = int(0.1 * FS)
        res[mode]["step_edge"] += [abs(y[e + d] - y[e - d]) for e in edges]
        rp = rng.integers(d, len(x) - d, 20)
        res[mode]["step_rand"] += [abs(y[p + d] - y[p - d]) for p in rp]
        f, Pw = signal.welch(xm, fs=FS, nperseg=8192); res[mode]["P"].append(Pw)
    n += 1
    if n == 60:
        break
L = []
pr = lambda s: L.append(s) or print(s)
pr(f"{n} dropout-free chunks from runs 0212-0224")
for mode in ("hard", "overlap"):
    r = res[mode]
    Pm = np.median(r["P"], 0)
    lines = {fr: np.sqrt(Pm[np.argmin(abs(f - fr))] * (f[1] - f[0]) * 1.5) for fr in (50, 150, 250)}
    pr(f"{mode:8s}: residual 50 Hz envelope median near block edges {np.median(r['env_edge']):.0f} vs elsewhere {np.median(r['env_mid']):.0f} counts; "
       f"LP5 |Δ| over ±0.1 s at edges median {np.median(r['step_edge']):.1f} (p95 {np.percentile(r['step_edge'], 95):.1f}) vs random {np.median(r['step_rand']):.1f} (p95 {np.percentile(r['step_rand'], 95):.1f}); "
       f"residual line amplitude ~ " + ", ".join(f"{k} Hz {v_:.0f}" for k, v_ in lines.items()))
open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "out_mains_edge.txt"), "w").write("\n".join(L))
