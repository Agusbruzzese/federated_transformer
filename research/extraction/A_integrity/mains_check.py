"""Independent sample-rate check: the mains interference line (50 Hz grid, stable to ~±0.1 %)
is located in cycles/sample on the reconstructed slot grid; fs_mains = 50 / f_cyc.
Compared with fs_local fitted against the PC clock (filename timestamps)."""
import os, numpy as np, pandas as pd
import bsd_io as b
HERE = os.path.dirname(os.path.abspath(__file__))
idx = b.list_chunks(); idx = idx[~idx.truncated]
rng = np.random.default_rng(3)
pick = idx.iloc[rng.choice(len(idx), 60, replace=False)]
rows = []
for r in pick.itertuples():
    _, v, info = b.load_chunk(r.values_path, r.time_path, return_info=True)
    s = info["slots"]
    # fill dropouts by linear interpolation on the slot grid (only for this spectral test)
    grid = np.arange(s[-1] + 1); x = np.interp(grid, s, v); x = x - np.polyval(np.polyfit(grid, x, 3), grid)
    x *= np.hanning(len(x))
    N = 1 << 22
    F = np.abs(np.fft.rfft(x, N)); f = np.fft.rfftfreq(N)     # cycles/sample, zero-padded
    band = (f > 45 / 2415) & (f < 55 / 2415)
    k = np.flatnonzero(band)[np.argmax(F[band])]
    fc = f[k]
    snr = F[k] / np.median(F[band])
    rows.append(dict(run=r.run[4:12], index=r.index, f_cyc=fc, fs_mains=50 / fc, fs_wall=r.fs_local,
                     fs_device=1e6 / r.base_dt, peak_snr=snr, line_hz_if_2400=fc * 2400))
o = pd.DataFrame(rows); o.to_csv(os.path.join(HERE, "mains_check.csv"), index=False)
o["diff_wall"] = o.fs_mains - o.fs_wall; o["diff_dev"] = o.fs_mains - o.fs_device
print(o.round(3).to_string())
print(o[["fs_mains", "fs_wall", "fs_device", "diff_wall", "diff_dev", "peak_snr"]].describe().round(3))
print("corr(fs_mains, fs_wall)=%.3f" % np.corrcoef(o.fs_mains, o.fs_wall)[0, 1])
