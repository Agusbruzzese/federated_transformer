"""Q1: 14-day baseline timeline, diurnal profile, run steps, within-chunk drift."""
import numpy as np, pandas as pd
from style import plt, RUNC, INK2
d = pd.read_parquet("../data/chunk_stats.parquet").sort_values("start").reset_index(drop=True)
d["hour"] = d.start.dt.hour + d.start.dt.minute / 60
d["day"] = d.start.dt.normalize()
fig, ax = plt.subplots(4, 1, figsize=(13, 11), sharex=True)
for r in range(4):
    s = d[d.run == r]
    ax[0].plot(s.start, (s.p50 - 2**23) / 1e3, ".", ms=2, color=RUNC[r], label=f"run {r}")
    ax[0].fill_between(s.start, (s.p01 - 2**23) / 1e3, (s.p99 - 2**23) / 1e3, color=RUNC[r], alpha=.25, lw=0)
    ax[1].plot(s.start, s.lp_sig, ".", ms=2, color=RUNC[r])
    ax[2].plot(s.start, s.mains_amp, ".", ms=2, color=RUNC[r])
    ax[3].semilogy(s.start, s.lp_maxdev, ".", ms=2, color=RUNC[r])
ax[0].set_ylabel("chunk median − 2^23\n(k counts; band = p1–p99)")
ax[0].legend(ncol=4, markerscale=5, loc="upper right")
ax[1].set_ylabel("LP(20 Hz) noise σ\n(robust, counts)")
ax[2].set_ylabel("50 Hz line amplitude\n(counts)")
ax[3].set_ylabel("max |LP − 10 s median|\n(counts, log)")
for a in ax:
    for day in pd.date_range("2025-02-04", "2025-02-28"):
        a.axvline(day, color=INK2, lw=.3, alpha=.4)
ax[0].set_title("Per-chunk statistics, every 8th chunk (3679 chunks × 41.4 s), 4 runs")
fig.tight_layout(); fig.savefig("../figures/fig1_baseline_timeline.png", dpi=110)

# diurnal profile: baseline anomaly relative to each day's mean, per hour
d["anom"] = d.p50 - d.groupby(["run", "day"]).p50.transform("median")
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
for r in range(4):
    s = d[d.run == r]
    ax[0].plot(s.hour, s.anom / 1e3, ".", ms=2, color=RUNC[r], label=f"run {r}")
h = d.groupby(d.start.dt.hour)
ax[0].plot(h.hour.mean(), h.anom.median() / 1e3, "-", color="#0b0b0b", lw=2, label="hourly median")
ax[0].set_xlabel("hour of day"); ax[0].set_ylabel("median − same-day median (k counts)"); ax[0].legend(markerscale=5, fontsize=7)
ax[0].set_title("Diurnal baseline cycle")
# rate of change of baseline between consecutive sampled chunks (5.6 min apart)
d["dp"] = d.groupby("run").p50.diff()
d["dtm"] = d.groupby("run").start.diff().dt.total_seconds() / 60
rate = (d.dp / d.dtm).dropna()
ax[1].hist(rate.clip(-400, 400), bins=200, color=RUNC[0]); ax[1].set_yscale("log")
ax[1].set_xlabel("baseline slope between sampled chunks (counts/min)"); ax[1].set_title("Baseline drift rate")
# within-chunk: last5 - first5 of LP signal vs chunk
wc = (d.last5 - d.first5)
ax[2].hist(wc.clip(-2000, 2000), bins=200, color=RUNC[1]); ax[2].set_yscale("log")
ax[2].set_xlabel("within-chunk drift: median(last 5 s) − median(first 5 s) (counts)"); ax[2].set_title("Within-chunk drift (41 s)")
fig.tight_layout(); fig.savefig("../figures/fig2_baseline_diurnal_drift.png", dpi=110)

# numbers
print("global median-2^23 range (counts):", (d.p50.min() - 2**23), (d.p50.max() - 2**23))
print("per-run median of p50:", d.groupby("run").p50.median().round().to_dict())
print("per-run p50 min/max:", d.groupby("run").p50.agg(["min", "max"]).to_dict())
print("hourly anomaly median:", h.anom.median().round().to_dict())
print("peak-to-peak of hourly diurnal profile:", h.anom.median().max() - h.anom.median().min())
daily = d.groupby(["run", "day"]).p50.agg(["min", "max", "median", "count"]); daily["pp"] = daily["max"] - daily["min"]
print(daily.to_string())
print("drift rate percentiles |counts/min|:", np.percentile(np.abs(rate), [50, 90, 99, 99.9]).round(1))
print("within-chunk |drift| pct:", np.percentile(np.abs(wc), [50, 90, 99, 99.9]).round(1))
print("base_range (10 s median range within chunk) pct:", np.percentile(d.base_range, [50, 90, 99]).round(1))
print("lp_sig pct:", np.percentile(d.lp_sig, [1, 50, 99]).round(1), " diffsig pct", np.percentile(d.diffsig, [1, 50, 99]).round(1))
print("lp_sig by hour:", d.groupby(d.start.dt.hour).lp_sig.median().round(0).to_dict())
print("mains amp by run:", d.groupby("run").mains_amp.median().round().to_dict(), "by hour", d.groupby(d.start.dt.hour).mains_amp.median().round().to_dict())
# correlation of p50 anomaly with lp_sig? big jumps between consecutive sampled chunks
big = d[np.abs(d.dp) > 3000][["start", "run", "p50", "dp", "dtm"]]
print("jumps >3000 counts between sampled chunks:\n", big.to_string())
d.to_parquet("../data/chunk_stats_enriched.parquet")
