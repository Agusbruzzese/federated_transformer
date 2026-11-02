"""Sampling-rate check from the mains-interference line, with an uncertainty budget.

Sample A (stratified): every run x 3-hour block of day, 3 random full chunks  (~96 chunks).
Sample B (dense, long-term): every 25th chunk of run 0204 from 2025-02-04 16:00 to 2025-02-08 16:00.
Per chunk: dropouts filled by linear interpolation on the exact slot grid (bsd_io v2), detrended,
Hann window, zero-padded FFT (2^22) + parabolic peak interpolation in 45-55 Hz -> f_cyc
(cycles/slot); fs_mains = 50/f_cyc. Repeat on each half of the chunk -> half-to-half
difference = estimator noise + grid wander within ~20 s.
Compared with fs_local (PC-clock fit) and 1e6/base_dt (device ticks)."""
import os, numpy as np, pandas as pd
from multiprocessing import Pool
import bsd_io as b
HERE = os.path.dirname(os.path.abspath(__file__))

def peak(x):
    x = x - np.polyval(np.polyfit(np.arange(len(x)), x, 3), np.arange(len(x)))
    x = x * np.hanning(len(x)); N = 1 << 22
    F = np.abs(np.fft.rfft(x, N)); f = np.fft.rfftfreq(N)
    band = np.flatnonzero((f > 45 / 2415) & (f < 55 / 2415))
    k = band[np.argmax(F[band])]
    a, bb, c = np.log(F[k - 1:k + 2]); p = 0.5 * (a - c) / (a - 2 * bb + c)
    return (k + p) / N, F[k] / np.median(F[band])

def one(a):
    sample, run, index, vp, tp, fs_local, base_dt, wall = a
    _, v, info = b.load_chunk(vp, tp, return_info=True)
    s = info["slots"]; grid = np.arange(s[-1] + 1); x = np.interp(grid, s, v)
    fc, snr = peak(x); h = len(x) // 2
    f1, _ = peak(x[:h]); f2, _ = peak(x[h:])
    return dict(sample=sample, run=run[4:12], index=index, wall=wall, hour=wall.hour, fs_mains=50 / fc, snr=snr,
                fs_mains_h1=50 / f1, fs_mains_h2=50 / f2, fs_wall=fs_local, fs_device=1e6 / base_dt,
                line_hz_if_2400=fc * 2400, line_hz_if_2415=fc * 2415)

if __name__ == "__main__":
    idx = b.list_chunks(); idx = idx[~idx.truncated].copy()
    idx["blk"] = idx.wall_start.dt.hour // 3
    A = pd.concat([g.sample(min(3, len(g)), random_state=7) for _, g in idx.groupby(["run", "blk"])])
    r0 = idx[idx.run.str.contains("20250204") & (idx.wall_start >= "2025-02-04 16:00") & (idx.wall_start < "2025-02-08 16:00")]
    B = r0.iloc[::25]
    jobs = [("A", r.run, r.index, r.values_path, r.time_path, r.fs_local, r.base_dt, r.wall_start) for r in A.itertuples()] + \
           [("B", r.run, r.index, r.values_path, r.time_path, r.fs_local, r.base_dt, r.wall_start) for r in B.itertuples()]
    with Pool(5) as p:
        o = pd.DataFrame(p.map(one, jobs))
    o.to_csv(os.path.join(HERE, "mains_check_v2.csv"), index=False)
    o["d_wall"] = o.fs_mains - o.fs_wall; o["d_half"] = o.fs_mains_h1 - o.fs_mains_h2
    pd.set_option("display.width", 200)
    for smp, g in o.groupby("sample"):
        print(f"\n=== sample {smp}: n={len(g)}")
        print(g[["fs_mains", "fs_wall", "fs_device", "d_wall", "d_half", "snr"]].describe().round(3).to_string())
        sl, ic = np.polyfit(g.fs_wall, g.fs_mains, 1)
        # bootstrap CI of slope
        rng = np.random.default_rng(0); bs = []
        for _ in range(2000):
            i = rng.integers(0, len(g), len(g)); bs.append(np.polyfit(g.fs_wall.values[i], g.fs_mains.values[i], 1)[0])
        print("regression fs_mains = a*fs_wall + c : a=%.3f  95%%CI [%.3f, %.3f]  corr=%.3f  resid std=%.2f Hz"
              % (sl, *np.percentile(bs, [2.5, 97.5]), np.corrcoef(g.fs_wall, g.fs_mains)[0, 1],
                 np.std(g.fs_mains - np.polyval([sl, ic], g.fs_wall))))
        print("by run:"); print(g.groupby("run")[["fs_mains", "fs_wall", "d_wall"]].agg(["mean", "std"]).round(2).to_string())
        day = g.hour.between(10, 17)
        print("day(10-17h) vs night: fs_mains %.2f vs %.2f | fs_wall %.2f vs %.2f | fs_device %.1f vs %.1f"
              % (g.fs_mains[day].mean(), g.fs_mains[~day].mean(), g.fs_wall[day].mean(), g.fs_wall[~day].mean(),
                 g.fs_device[day].mean(), g.fs_device[~day].mean()))
    # long-term PC-clock average over the dense window: total slots / total wall seconds
    r0s = r0.sort_values("slot_start")
    fs_long_wall = (r0s.slot_start.iloc[-1] - r0s.slot_start.iloc[0]) / (r0s.fn_start.iloc[-1] - r0s.fn_start.iloc[0]).total_seconds()
    gB = o[o["sample"] == "B"]
    print("\nFeb 4-8 long-term: PC-clock mean fs = %.3f Hz (+-%.3f from 1 s quantisation) ; mains mean fs = %.3f +- %.3f (SE, n=%d)"
          % (fs_long_wall, 2415 * 1.0 / (r0s.fn_start.iloc[-1] - r0s.fn_start.iloc[0]).total_seconds(),
             gB.fs_mains.mean(), gB.fs_mains.std() / np.sqrt(len(gB)), len(gB)))
