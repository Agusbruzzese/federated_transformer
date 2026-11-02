"""Empirical model of timestamps around dropouts, used to parameterise the synthetic test.
For isolated gaps (no other gap within +-20 samples) the local grid is fitted on 16 clean
samples before and 16 after (after = t[g+2..g+17]); we record:
 - lateness of the first post-gap stamp t[g+1] relative to its integer slot (slot(g+2)-1)
 - the fractional part of the post-gap segment position (should be ~0 if gap = integer slots)
 - gap size in slots.  Also white-jitter sigma of stamps in clean stretches."""
import numpy as np, pandas as pd, bsd_io as b
idx = b.list_chunks(); idx = idx[~idx.truncated]
rng = np.random.default_rng(11)
pick = pd.concat([g.sample(8, random_state=11) for _, g in idx.groupby("run")])
rows = []; jit = []
for r in pick.itertuples():
    t = np.fromfile(r.time_path, "<u8").astype(np.int64); d = np.diff(t); med = np.median(d)
    gp = np.flatnonzero(d > 1.5 * med)
    base = d[d < 1.5 * med].mean()
    # jitter: residual of 200-sample clean windows about a line
    for k in range(1000, len(t) - 1000, 20000):
        if not np.any((gp >= k) & (gp < k + 200)):
            x = np.arange(200); y = t[k:k+200]; jit.append((y - np.polyval(np.polyfit(x, y, 1), x)).std())
    for g in gp:
        if g < 20 or g > len(t) - 40: continue
        if np.any((gp != g) & (np.abs(gp - g) < 20)): continue
        xb = np.arange(-15, 1); yb = t[g-15:g+1]; pb = np.polyfit(xb, yb, 1)        # pre-gap line (slot 0 = t[g])
        ya = t[g+2:g+18]; pos_a = (ya - pb[1]) / pb[0]                               # positions of clean post-gap samples
        J = np.rint(np.median(pos_a - np.arange(16)))                                # slot of t[g+2]
        frac = np.median(pos_a - np.arange(16)) - J
        late = (t[g+1] - pb[1]) / pb[0] - (J - 1)
        rows.append(dict(run=r.run[4:12], size=int(J - 2), dt_ratio=d[g] / base, late=late, frac=frac))
o = pd.DataFrame(rows)
print("white stamp jitter (ticks rms): median %.2f p90 %.2f" % (np.median(jit), np.percentile(jit, 90)))
print(o.groupby(pd.cut(o["size"], [0, 1, 2, 5, 90, 200])).agg(n=("late", "size"), late_mean=("late", "mean"), late_std=("late", "std"),
      late_min=("late", "min"), late_max=("late", "max"), frac_absmax=("frac", lambda x: np.abs(x).max())))
print("size histogram", o["size"].value_counts().sort_index().to_dict())
o.to_csv("postgap_model.csv", index=False)
