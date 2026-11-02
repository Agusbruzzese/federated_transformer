"""Are timestamp deltas with dt/base in (1.3, 1.5) (not flagged as gaps) hidden 1-sample losses?
For each such delta at k, fit a line to t[k-20..k] and measure the position of t[k+2..k+20]
(clean samples after): offset ~0 -> no loss (late stamp only); ~1 -> hidden loss."""
import numpy as np, pandas as pd
C = pd.read_parquet("full_scan_chunks.parquet"); import bsd_io as b
idx = b.list_chunks()
top = C.nlargest(40, "n_ambiguous_ratio").merge(idx[["run", "index", "time_path"]], on=["run", "index"])
offs = []; ratios = []
for r in top.itertuples():
    t = np.fromfile(r.time_path, "<u8").astype(np.int64); d = np.diff(t)
    base = d[d < 1.5 * np.median(d)].mean(); rr = d / base
    ks = np.flatnonzero((rr > 1.3) & (rr < 1.7))
    gp = np.flatnonzero(rr >= 1.5)
    for k in ks:
        if k < 25 or k > len(t) - 25 or np.any((gp != k) & (np.abs(gp - k) < 22)): continue
        x = np.arange(-20, 1); p = np.polyfit(x, t[k-20:k+1], 1)
        pos = (t[k+2:k+21] - p[1]) / p[0] - np.arange(2, 21)
        offs.append(np.median(pos)); ratios.append(rr[k])
offs = np.array(offs); ratios = np.array(ratios)
print("n tested", len(offs))
for lo, hi in [(1.3, 1.4), (1.4, 1.5), (1.5, 1.6), (1.6, 1.7)]:
    m = (ratios > lo) & (ratios <= hi)
    print(f"dt/base in ({lo},{hi}]: n={m.sum()}  post-offset slots: median {np.median(offs[m]):.3f}  frac |off-1|<0.3: {np.mean(np.abs(offs[m]-1)<0.3):.4f}  frac |off|<0.3: {np.mean(np.abs(offs[m])<0.3):.4f}")
