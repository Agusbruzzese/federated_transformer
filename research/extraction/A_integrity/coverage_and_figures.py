"""Coverage table per calendar day + figures (uses chunk_index.parquet, sanity scans)."""
import os, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
import bsd_io as b
HERE = os.path.dirname(os.path.abspath(__file__))
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": .25,
                     "font.size": 9, "figure.dpi": 130})
idx = b.list_chunks()
idx["wall_end"] = idx.wall_start + pd.to_timedelta((idx.n_samples - 1 + idx.n_missing) / idx.fs_local, unit="s")
idx["run_short"] = idx.run.str[4:12]
runs = sorted(idx.run_short.unique())

# ---------------- coverage per calendar day
intervals = idx[["wall_start", "wall_end", "n_samples", "n_missing", "run_short"]].sort_values("wall_start")
# data-free intervals: between consecutive chunks (> 50 ms), and outside runs
first, last = intervals.wall_start.min().normalize(), intervals.wall_end.max().normalize() + pd.Timedelta(days=1)
hole_rows = []
prev_end = first
for r in intervals.itertuples():
    if (r.wall_start - prev_end).total_seconds() > 0.05:
        hole_rows.append((prev_end, r.wall_start))
    prev_end = max(prev_end, r.wall_end)
hole_rows.append((prev_end, last))
holes = pd.DataFrame(hole_rows, columns=["start", "end"])
rows = []
for day in pd.date_range(first, last - pd.Timedelta(days=1), freq="D"):
    d0, d1 = day, day + pd.Timedelta(days=1)
    ch = intervals[(intervals.wall_end > d0) & (intervals.wall_start < d1)]
    cov = (ch.wall_end.clip(upper=d1) - ch.wall_start.clip(lower=d0)).dt.total_seconds().clip(lower=0).sum()
    hs = holes[(holes.end > d0) & (holes.start < d1)]
    hdur = (hs.end.clip(upper=d1) - hs.start.clip(lower=d0)).dt.total_seconds()
    frac_full = np.clip(((ch.wall_end.clip(upper=d1) - ch.wall_start.clip(lower=d0)).dt.total_seconds()
                         / (ch.wall_end - ch.wall_start).dt.total_seconds()), 0, 1)
    miss = (ch.n_missing * frac_full).sum(); got = (ch.n_samples * frac_full).sum()
    rows.append(dict(date=day.date(), runs=",".join(sorted(ch.run_short.unique())), n_chunks=len(ch),
                     hours_covered=round(cov / 3600, 2), first_sample=max(ch.wall_start.min(), d0).round("s") if len(ch) else pd.NaT,
                     last_sample=min(ch.wall_end.max(), d1).round("s") if len(ch) else pd.NaT,
                     n_gaps_gt50ms=len(hs), largest_gap_h=round(hdur.max() / 3600, 3) if len(hs) else 0.0,
                     samples_M=round(got / 1e6, 2), dropout_frac_pct=round(100 * miss / max(got + miss, 1), 3),
                     equiv_105_sample_dropouts=int(round(miss / 105))))
cov = pd.DataFrame(rows)
cov.to_csv(os.path.join(HERE, "coverage_per_day.csv"), index=False)
print(cov.to_string())
print("total hours", cov.hours_covered.sum(), "full days (>=23.9h)", (cov.hours_covered >= 23.9).sum())

# ---------------- Fig 1: three clocks
fig, ax = plt.subplots(3, 1, figsize=(10, 8), sharex=False)
W = 40
for i, (run, g) in enumerate(idx[~idx.truncated].groupby("run_short")):
    ws = (g.fn_start - g.fn_start.iloc[0]).dt.total_seconds().values
    k = np.arange(0, len(g) - W, W)
    raw = W * 1e5 / (ws[k + W] - ws[k])
    corr = (g.slot_start.values[k + W] - g.slot_start.values[k]) / (ws[k + W] - ws[k])
    dev = (g.t_first.values[k + W] - g.t_first.values[k]) / 1e6 / (ws[k + W] - ws[k])
    tt = g.fn_start.values[k]
    ax[0].plot(tt, raw, lw=1, color=C[i], label=run)
    ax[1].plot(tt, corr, lw=1, color=C[i], label=run)
    ax[2].plot(tt, dev, lw=1, color=C[i], label=run)
ax[0].set_title("A. Samples written per wall-clock second (raw count; 40-chunk windows)")
ax[1].set_title("B. ADC slots per wall-clock second (samples + reconstructed dropouts) = true ADC rate")
ax[2].set_title("C. Device 'Time' ticks per wall-clock second (would be 1.000 if ticks were µs)")
for a in ax[:2]:
    a.axhline(2400, color="k", lw=.8, ls="--"); a.set_ylabel("Hz"); a.set_ylim(2350, 2430)
ax[2].set_ylabel("ticks / µs"); ax[2].axhline(1, color="k", lw=.8, ls="--")
ax[0].legend(ncol=4, fontsize=8, loc="lower left")
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig1_clocks.png")); plt.close(fig)

# ---------------- Fig 2: device tick per slot vs hour of day
fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
for i, (run, g) in enumerate(idx[~idx.truncated].groupby("run_short")):
    h = g.wall_start.dt.hour + g.wall_start.dt.minute / 60
    ax[0].scatter(h, g.base_dt, s=1, color=C[i], alpha=.3, label=run)
    ax[1].scatter(h, g.fs_local, s=1, color=C[i], alpha=.3, label=run)
ax[0].set_title("Device ticks per ADC sample vs hour of day"); ax[0].set_xlabel("hour"); ax[0].set_ylabel("ticks/slot")
ax[1].set_title("Fitted ADC rate (vs PC clock) vs hour of day"); ax[1].set_xlabel("hour"); ax[1].set_ylabel("Hz")
ax[0].legend(markerscale=8, fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig2_diurnal_clock.png")); plt.close(fig)

# ---------------- Fig 3: dropout map (missing fraction per chunk over time) + coverage
fig, ax = plt.subplots(2, 1, figsize=(10, 5.5))
for i, (run, g) in enumerate(idx.groupby("run_short")):
    ax[0].plot(g.wall_start, 100 * g.n_missing / (g.n_samples + g.n_missing), ".", ms=1, color=C[i], label=run)
ax[0].set_ylabel("% samples dropped"); ax[0].set_title("Dropped samples per chunk (exact, full scan)")
ax[0].legend(markerscale=8, fontsize=8)
x = np.arange(len(cov))
ax[1].bar(x, cov.hours_covered, color=C[0], width=.8)
ax[1].set_xticks(x); ax[1].set_xticklabels([str(d)[5:] for d in cov.date], rotation=90)
ax[1].set_ylabel("hours with data"); ax[1].set_title("Coverage per calendar day")
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig3_dropouts_coverage.png")); plt.close(fig)

# ---------------- Fig 4: a dropout up close + timestamp delta histogram
r = idx[(idx.run_short == "20250204") & (idx["index"] == 5005)].iloc[0]
t, v = b.read_raw(r.values_path, r.time_path); t = t.astype(np.int64); v = v.astype(float)
d = np.diff(t); g = np.flatnonzero(d > 20 * np.median(d))
_, _, info = b.load_chunk(r.values_path, r.time_path, return_info=True)
k = g[(g > 500) & (g < len(t) - 500)][0]; sl = info["slots"]; w = slice(k - 400, k + 400)
fig, ax = plt.subplots(1, 3, figsize=(11, 3.2))
ax[0].plot(np.arange(len(v))[w] - k, v[w], lw=.8, color=C[0]); ax[0].set_title("Values by sample index (dropout hidden)")
ax[0].set_xlabel("sample index - gap"); ax[0].set_ylabel("ADC counts")
ax[1].plot(sl[w] - sl[k], v[w], lw=.8, color=C[0]); ax[1].axvspan(0, sl[k + 1] - sl[k], color=C[1], alpha=.2)
ax[1].set_title(f"Values on reconstructed slot grid ({sl[k+1]-sl[k]-1} missing)"); ax[1].set_xlabel("slot - gap")
dd = d[d < 600]
ax[2].hist(dd, bins=np.arange(250, 600, 2), color=C[0]); ax[2].set_yscale("log")
ax[2].set_title(f"Tick deltas (<600); {np.sum(d>=600)} deltas ≥600\n(44.5 ms class)"); ax[2].set_xlabel("ticks")
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig4_dropout_example.png")); plt.close(fig)

# ---------------- Fig 5: baseline over 14 days
fig, ax = plt.subplots(figsize=(10, 3))
for i, (run, g) in enumerate(idx.groupby("run_short")):
    ax.plot(g.wall_start, g.value_first, ".", ms=1, color=C[i], label=run)
ax.set_title("First ADC code of each chunk (baseline) — 24-bit code, rails at 0 and 16 777 215")
ax.set_ylabel("ADC counts"); ax.legend(markerscale=8, fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig5_baseline.png")); plt.close(fig)

# ---------------- Fig 6: largest event in the random sanity sample
s = pd.read_parquet(os.path.join(HERE, "sanity_scan.parquet"))
top = s.nlargest(1, "ptp").iloc[0]
r = idx[(idx.run == top.run) & (idx["index"] == top["index"])].iloc[0]
ts, v = b.load_chunk(r.values_path, r.time_path)
i0 = np.argmax(np.abs(v - np.median(v)))
w = slice(max(0, i0 - 12000), i0 + 12000)
fig, ax = plt.subplots(figsize=(10, 3))
ax.plot(ts[w], v[w], lw=.6, color=C[0])
ax.set_title(f"Largest excursion in sampled chunks: {top.run[4:12]} #{top['index']} (ptp {top.ptp:.0f} counts)")
ax.set_xlabel("s from chunk start"); ax.set_ylabel("ADC counts")
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig6_largest_event.png")); plt.close(fig)
