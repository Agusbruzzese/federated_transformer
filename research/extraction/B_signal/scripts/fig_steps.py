"""Q4: what are the large baseline changes seen in the 14-day timeline? Zoom on the largest jumps between
sampled chunks using every chunk in between (1-s medians of raw codes; no filtering needed at this scale)."""
import numpy as np, pandas as pd, io_utils as io
from style import plt, RUNC, INK2
d = pd.read_parquet("../data/chunk_stats_enriched.parquet")
C = io.list_chunks(); byrun = {}
for c in C: byrun.setdefault(c["run"], {})[c["idx"]] = c
d = d.assign(adp=d.dp.abs()).dropna(subset=["dp"])
d = d[~((d.run == 1) & (d.start < pd.Timestamp("2025-02-12 16:00")))]  # exclude run 0212 settling hour (A: F7)
top = d.sort_values("adp", ascending=False)
sel = []
for _, r in top.iterrows():
    if all(abs((r.start - s.start).total_seconds()) > 3600 for s in sel): sel.append(r)
    if len(sel) == 6: break
fig, ax = plt.subplots(3, 2, figsize=(15, 10)); ax = ax.ravel()
FS = 2416.0
for k, r in enumerate(sel):
    ms = []; t0 = None
    for j in range(int(r.idx) - 12, int(r.idx) + 5):
        c = byrun[int(r.run)].get(j)
        if c is None: continue
        v = np.fromfile(c["vpath"], "<u4").astype(float)
        n = len(v) // 2416
        m = np.median(v[: n * 2416].reshape(n, 2416), 1)
        if t0 is None: t0 = c["start"]
        tt = (c["start"] - t0).total_seconds() + np.arange(n) * 1.0
        ms.append((tt, m))
    T = np.concatenate([a for a, b in ms]); M = np.concatenate([b for a, b in ms])
    ax[k].plot(T / 60, (M - M[0]) / 1e3, lw=.7, color=RUNC[0])
    dM = np.diff(M); jmax = np.argmax(np.abs(dM))
    ax[k].set_title(f"{t0:%m-%d %H:%M} run{int(r.run)}: Δ over window {(M[-1]-M[0])/1e3:+.1f}k; largest 1-s jump {dM[jmax]/1e3:+.1f}k", fontsize=8)
    ax[k].set_xlabel("minutes"); ax[k].set_ylabel("1-s median − first (k counts)")
    print(t0, "range k", np.ptp(M) / 1e3, "largest 1-s jump", dM[jmax], "p99 |1-s diff|", np.percentile(np.abs(dM), 99))
fig.tight_layout(); fig.savefig("../figures/fig9_baseline_steps.png", dpi=100)
