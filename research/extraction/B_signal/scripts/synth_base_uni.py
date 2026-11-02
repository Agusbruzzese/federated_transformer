"""Issue 9 worst case: UNIPOLAR injected events (half-sine of the given support), which a running
median cannot cancel by symmetry, through LP 5 Hz -> 50 Hz -> running median (1- and 2-pass).
Also: real events, 2-pass vs 1-pass (20 s) peak ratio vs duration (v2_events, 8 sigma).
Output: data/out_synth_uni.txt"""
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "1")
import pickle, numpy as np, pandas as pd
from multiprocessing import Pool
import synth as S
OUT = S.OUT
if __name__ == "__main__":
    Q = [q["x"].astype(float) for q in pickle.load(open(os.path.join(OUT, "v2_quiet.pkl"), "rb"))]
    rng = np.random.default_rng(3)
    uni = np.sin(np.linspace(0, np.pi, 5000))
    jobs = [(i, uni, sup, amp, Q[rng.integers(len(Q))]) for i in range(12) for sup in (2, 4, 8, 12, 16, 24) for amp in (2000, 20000)]
    with Pool(5) as p:
        B = pd.DataFrame([r for rr in p.map(S.base_job, jobs) for r in rr])
    lines = []
    pr = lambda s: lines.append(s) or print(s)
    pr("UNIPOLAR half-sine, amp 20000: median peak ratio / area ratio / corr by (passes, window, support)")
    t = B[B.amp == 20000].groupby(["npass", "win", "support"]).agg(peak=("peak", "median"), area=("area", "median"), rho=("rho", "median")).round(3)
    pr(t.unstack("support").to_string())
    pr("\nUNIPOLAR, amp 2000:")
    pr(B[B.amp == 2000].groupby(["npass", "win", "support"]).peak.median().round(3).unstack("support").to_string())
    ev = pd.read_parquet(os.path.join(OUT, "v2_events.parquet"))
    ev = ev[(ev.thr == 8) & (ev.dur >= 0.4) & (ev.edge_dist >= 5)]
    ev["ratio"] = ev.amax_1pass / ev.amax
    pr("\nREAL events (8σ, ≥0.4 s, ≥5 s from edge/mask): amax(1-pass 20 s median)/amax(2-pass) by duration bin")
    pr(ev.groupby(pd.cut(ev.dur, [0.4, 2, 4, 6, 8, 12, 60])).ratio.describe(percentiles=[.05, .5, .95]).round(3).to_string())
    open(os.path.join(OUT, "out_synth_uni.txt"), "w").write("\n".join(lines))
