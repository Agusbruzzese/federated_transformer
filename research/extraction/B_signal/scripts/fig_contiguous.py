"""Q3/Q4 on contiguous 2-h windows: rates, inter-arrival, overlaps, baseline steps around events,
and baseline wander vs time-scale (structure function of 1-s medians) to choose the detrending window."""
import numpy as np, pandas as pd
from scipy import ndimage
import events as E
from style import plt, RUNC, INK2
D = np.load("../data/contiguous_lp100.npz")
keys = [k for k in D.files if not k.endswith("__bad")]
rows = []; sf = {}
fig, ax = plt.subplots(3, 1, figsize=(15, 10))
for key in keys:
    z = D[key].astype(float); bad = D[key + "__bad"]
    evs, r, sig, y, b = E.detect(z)
    for a, e in evs:
        d = E.describe(r, a, e, sig)
        pre = np.median(y[max(0, a - 1000):a]) if a > 200 else np.nan
        post = np.median(y[e:e + 1000]) if e + 200 < len(y) else np.nan
        d.update(win=key, a=a, e=e, dropout=bool(bad[a:e].any()), step=post - pre, sig=sig)
        rows.append(d)
    # structure function of 1-s medians (quiet-ish: exclude seconds with |r|>5 sig)
    m1 = np.array([np.median(z[i:i + 100]) for i in range(0, len(z) - 100, 100)])
    act = np.array([np.abs(r[i:i + 100]).max() > 5 * sig for i in range(0, len(z) - 100, 100)])
    lags = [1, 2, 5, 10, 20, 30, 60, 120, 300, 600, 1200, 3600]
    sf[key] = [np.median(np.abs(m1[l:] - m1[:-l])[~(act[l:] | act[:-l])]) for l in lags]
ev = pd.DataFrame(rows); ev["label"] = ev.win.str.split("__").str[1]
ok = ev[~ev.dropout]; veh = ok[ok.dur >= 0.4]; big = veh[veh.amax > 5000]
print("per window: hours=2; vehicle-like, large(>5000), rate/h")
for w, s in veh.groupby("win"):
    bb = s[s.amax > 5000]
    st = np.sort(bb.a.values) / 100; ia = np.diff(st)
    # overlaps: large event with another vehicle-like detection starting within 5 s of its end / before its start
    sa = np.sort(s.a.values) / 100; se = s.sort_values("a").e.values / 100
    close = np.sum(np.diff(sa) < 5)
    print(f"{w:35s} veh {len(s):4d}  large {len(bb):4d}  median inter-arrival large {np.median(ia) if len(ia) else np.nan:7.1f}s  "
          f"min {ia.min() if len(ia) else np.nan:6.1f}s  pairs<5s apart {close:3d}  merged(dur>8s) {(s.dur > 8).sum()}  multi-lobe(>=4) {(s.nlobes >= 4).sum()}")
print("large-event baseline step |post-pre| (10 s medians) pct:", np.percentile(np.abs(big.step.dropna()), [25, 50, 75, 90, 95]).round(0))
print("  as fraction of amax:", np.percentile((np.abs(big.step) / big.amax).dropna(), [25, 50, 75, 90, 95]).round(3))
print("  frac of large events with |step| > 1000 counts:", (np.abs(big.step) > 1000).mean().round(3), " >3000:", (np.abs(big.step) > 3000).mean().round(3))
print("rise 10-90% large:", np.percentile(big.rise, [5, 50, 95]), "fwhm", np.percentile(big.fwhm, [5, 50, 95]))
lags = np.array([1, 2, 5, 10, 20, 30, 60, 120, 300, 600, 1200, 3600])
for k, (key, v) in enumerate(sf.items()):
    lab = key.split("__")[1]; c = RUNC[0] if "night" in lab else (RUNC[1] if "AM" in lab else (RUNC[2] if "PM" in lab else RUNC[3]))
    ax[0].loglog(lags, v, "-o", ms=3, color=c, lw=1, label=key.replace("_", " "))
    print("SF", key, np.round(v, 0))
ax[0].set_xlabel("lag τ (s)"); ax[0].set_ylabel("median |m(t+τ) − m(t)| of 1-s medians (counts)")
ax[0].set_title("Baseline wander vs time scale (quiet seconds only)"); ax[0].legend(fontsize=6, ncol=3)
# 30-min strip of busiest AM window
key = max(keys, key=lambda k: (big.win == k).sum())
z = D[key].astype(float); y, b, r = E.detrend(z)
for j, (lo, hi) in enumerate([(0, 2 * 3600), (int(np.sort(big[big.win == key].a.values)[0] / 100) - 60, int(np.sort(big[big.win == key].a.values)[0] / 100) + 540)]):
    tt = np.arange(len(z)) / 100
    m = (tt >= lo) & (tt < hi)
    ax[1 + j].plot(tt[m] / 60, y[m] - np.median(y), lw=.5, color=RUNC[0], label="5 Hz LP")
    ax[1 + j].plot(tt[m] / 60, b[m] - np.median(y), lw=1.2, color=RUNC[1], label="15 s running median")
    ax[1 + j].set_xlabel("minutes from window start"); ax[1 + j].set_ylabel("counts")
ax[1].set_title(f"{key.replace('_', ' ')}: full 2 h"); ax[2].set_title("zoom 10 min"); ax[1].legend(fontsize=7)
fig.tight_layout(); fig.savefig("../figures/fig8_contiguous_rushhour.png", dpi=110)
ev.to_parquet("../data/events_contiguous.parquet")
