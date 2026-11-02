"""Q3: morphology of clean transients; separate logger blips from vehicle-like events;
event bandwidth measured on the 20 Hz series with padding; day/night statistics."""
import numpy as np, pandas as pd
from scipy import ndimage
from style import plt, RUNC, INK2
ev = pd.read_parquet("../data/events_lp100.parquet"); ch = pd.read_parquet("../data/chunk_detect.parquet")
L = np.load("../data/lp100.npz"); Z = L["z"]
ev = ev[~ev.edge].copy()
ev["hour"] = ev.start.dt.hour
ev["cls"] = np.where(ev.dropout, "dropout-overlap", np.where(ev.dur < 0.4, "short blip (<0.4 s)", "vehicle-like (≥0.4 s)"))
# --- bandwidth of each clean event from the 20 Hz series: window = event ± 1 s, baseline = linear fit to
# 1-s flanks, Tukey taper, energy spectrum; compare with same-length quiet noise spectrum
FS = 100.0
bw = []
for i, e in ev.iterrows():
    z = Z[e.ci]; z = z[~np.isnan(z)]
    a, b = int(e.a - FS), int(e.e + FS)
    if a < 0 or b > len(z) or e.dropout: bw.append((np.nan,) * 4); continue
    s = z[a:b].astype(float)
    fl = np.r_[np.arange(0, 50), np.arange(len(s) - 50, len(s))]
    s = s - np.polyval(np.polyfit(fl, s[fl], 1), np.arange(len(s)))
    w = np.ones(len(s)); tp = np.hanning(100); w[:50] = tp[:50]; w[-50:] = tp[50:]
    S = np.abs(np.fft.rfft(s * w, 4096)) ** 2; f = np.fft.rfftfreq(4096, 1 / FS)
    # noise energy spectrum expectation for this window: sigma_20Hz^2 * len * (flat) -> subtract median level at 30-45 Hz
    nl = np.median(S[(f > 30) & (f < 45)])
    Sx = np.clip(S - nl, 0, None); c = np.cumsum(Sx) / max(Sx.sum(), 1e-9)
    bw.append((f[np.searchsorted(c, .9)], f[np.searchsorted(c, .95)], f[np.searchsorted(c, .99)], (f * Sx).sum() / Sx.sum()))
ev[["bw90", "bw95z", "bw99z", "fcentz"]] = np.array(bw)
ev.to_parquet("../data/events_classified.parquet")
clean = ev[~ev.dropout]
veh = clean[clean.dur >= 0.4]
blip = clean[clean.dur < 0.4]

fig, ax = plt.subplots(2, 3, figsize=(16, 9))
for k, (lab, s) in enumerate([("dropout-overlap", ev[ev.dropout]), ("short blip (<0.4 s)", blip), ("vehicle-like (≥0.4 s)", veh)]):
    ax[0, 0].loglog(s.dur, s.amax, ".", ms=2, color=RUNC[k], alpha=.5, label=f"{lab} (n={len(s)})")
ax[0, 0].set_xlabel("duration above 2σ (s)"); ax[0, 0].set_ylabel("peak |deviation| (counts)"); ax[0, 0].legend(markerscale=5, fontsize=7)
ax[0, 0].set_title("All detections (8σ trigger, 0–5 Hz)")
for r in range(4):
    s = clean[clean.run == r]
    ax[0, 1].loglog(s.dur, s.amax, ".", ms=2, color=RUNC[r], alpha=.5, label=f"run {r}")
ax[0, 1].legend(markerscale=5, fontsize=7); ax[0, 1].set_xlabel("duration (s)"); ax[0, 1].set_title("Clean detections by run (blips concentrate in run 0)")
bins = np.logspace(np.log10(200), np.log10(3e5), 50)
ax[0, 2].hist(veh.amax, bins=bins, color=RUNC[2], label="vehicle-like"); ax[0, 2].hist(blip.amax, bins=bins, color=RUNC[1], alpha=.6, label="short blip")
ax[0, 2].set_xscale("log"); ax[0, 2].set_xlabel("peak |deviation| (counts)"); ax[0, 2].legend(fontsize=7); ax[0, 2].set_title("Amplitude distribution")
ax[1, 0].hist(veh.dur, bins=np.linspace(0.4, 10, 49), color=RUNC[2], label="duration above 2σ"); ax[1, 0].hist(veh.d90, bins=np.linspace(0, 10, 51), histtype="step", color="#0b0b0b", label="90 % energy duration")
ax[1, 0].set_xlabel("s"); ax[1, 0].legend(fontsize=7); ax[1, 0].set_title("Vehicle-like: duration")
pc = veh.pattern.value_counts(); ax[1, 1].bar(pc.index[:8], pc.values[:8], color=RUNC[0]); ax[1, 1].set_title("Vehicle-like: lobe pattern (sign sequence)")
ax[1, 2].hist(veh.bw95z.dropna(), bins=np.linspace(0, 20, 81), color=RUNC[2], label="95 % energy"); ax[1, 2].hist(veh.bw99z.dropna(), bins=np.linspace(0, 20, 81), histtype="step", color="#0b0b0b", label="99 % energy")
ax[1, 2].set_xlabel("bandwidth (Hz)"); ax[1, 2].legend(fontsize=7); ax[1, 2].set_title("Vehicle-like: per-event bandwidth (20 Hz series)")
fig.tight_layout(); fig.savefig("../figures/fig5_event_morphology.png", dpi=110)

# day/night: rates & amplitude by hour (runs 1-3 + run 0 separately; hours sampled per hour-of-day)
ch["hour"] = ch.start.dt.hour
hh = ch.groupby("hour").size() * 41.4 / 3600
rate = veh.groupby("hour").size().reindex(range(24), fill_value=0) / hh
big = veh[veh.amax > 5000]; rate_big = big.groupby("hour").size().reindex(range(24), fill_value=0) / hh
fig, ax = plt.subplots(1, 3, figsize=(16, 4))
ax[0].bar(range(24), rate.values, color=RUNC[0], label="all vehicle-like"); ax[0].bar(range(24), rate_big.values, color=RUNC[1], label="amax > 5000")
ax[0].set_xlabel("hour"); ax[0].set_ylabel("events per hour of recording"); ax[0].legend(fontsize=7); ax[0].set_title("Diurnal event rate (every-8th-chunk sample)")
dn = veh.assign(p=np.where(veh.hour.between(7, 19), "day 07–19h", "night 20–06h"))
for k, (lab, s) in enumerate(dn.groupby("p")):
    ax[1].hist(np.log10(s.amax), bins=40, density=True, histtype="step", lw=1.5, color=RUNC[k], label=f"{lab} (n={len(s)})")
    ax[2].hist(s.dur, bins=np.linspace(0.4, 8, 39), density=True, histtype="step", lw=1.5, color=RUNC[k], label=lab)
ax[1].set_xlabel("log10 peak |deviation| (counts)"); ax[1].legend(fontsize=7); ax[1].set_title("Amplitude, day vs night")
ax[2].set_xlabel("duration (s)"); ax[2].legend(fontsize=7); ax[2].set_title("Duration, day vs night")
fig.tight_layout(); fig.savefig("../figures/fig6_event_diurnal.png", dpi=110)

def pct(x): return np.percentile(x.dropna(), [5, 25, 50, 75, 95]).round(2)
print("counts: all", len(ev), "dropout", ev.dropout.sum(), "blip", len(blip), "vehicle-like", len(veh))
for col in ["dur", "d90", "amax", "p2p", "snr", "rise", "fwhm", "bw90", "bw95z", "bw99z", "fcentz"]:
    print(f"veh {col}: 5/25/50/75/95 = {pct(veh[col])}")
print("veh big(>5000) dur:", pct(big.dur), "rise", pct(big.rise), "bw95", pct(big.bw95z), "bw99", pct(big.bw99z), "n", len(big))
print("blip amax", pct(blip.amax), "dur", pct(blip.dur))
print("veh pattern:", (veh.pattern.value_counts(normalize=True).round(3)).head(8).to_dict())
print("big pattern:", (big.pattern.value_counts(normalize=True).round(3)).head(8).to_dict())
print("dominant polarity veh:", veh.dom_pol.value_counts(normalize=True).round(3).to_dict(), " big:", big.dom_pol.value_counts(normalize=True).round(3).to_dict())
print("rate by hour:", rate.round(1).to_dict()); print("rate big by hour:", rate_big.round(1).to_dict())
for lab, s in dn.groupby("p"): print(lab, "amax", pct(s.amax), "dur", pct(s.dur), "frac>5000", (s.amax > 5000).mean().round(3))
print("veh amax log-hist:", np.histogram(np.log10(veh.amax), bins=np.arange(2.4, 5.6, .2))[0])
print("veh per run per hour:", (veh.groupby("run").size() / (ch.groupby("run").size() * 41.4 / 3600)).round(1).to_dict())
print("noise sigma (0-5 Hz) of chunks: median", ch.sig.median().round(1), " day", ch[ch.hour.between(9,17)].sig.median().round(1), " night", ch[ch.hour.between(1,5)].sig.median().round(1))
