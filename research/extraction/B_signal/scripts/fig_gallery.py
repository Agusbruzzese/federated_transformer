"""Q3: gallery of ~20 diverse detections: raw (mains removed, 2416 Hz), 20 Hz LP and 2 Hz LP."""
import numpy as np, pandas as pd, io_utils as io, preproc as P
from scipy import signal
from style import plt, RUNC, INK2
ev = pd.read_parquet("../data/events_classified.parquet"); raw = pd.read_parquet("../data/chunk_stats.parquet")
C = io.list_chunks(); cmap = {(c["run"], c["idx"]): c for c in C}
veh = ev[ev.cls == "vehicle-like (≥0.4 s)"]; rng = 7
pick = []
def take(q, n, label):
    s = q.sample(min(n, len(q)), random_state=rng); pick.extend([(i, label) for i in s.index])
take(veh[(veh.amax > 2e4) & (veh.pattern == "+-")], 4, "large bipolar +-")
take(veh[(veh.amax > 5e3) & (veh.pattern == "-+")], 2, "large bipolar -+")
take(veh[(veh.amax > 5e3) & veh.pattern.isin(["-+-", "+-+-", "+-+"])], 2, "large multi-lobe")
take(veh[(veh.amax > 5e3) & veh.pattern.isin(["-", "+"])], 2, "large unipolar")
take(veh[(veh.amax > 1e5)], 1, "largest")
take(veh[(veh.amax.between(1500, 5000))], 3, "medium")
take(veh[(veh.amax < 1200) & (veh.hour.between(9, 17))], 2, "small, daytime")
take(veh[(veh.amax < 1200) & ~(veh.hour.between(6, 20))], 2, "small, night")
take(ev[ev.cls == "short blip (<0.4 s)"], 1, "logger blip (artefact)")
take(ev[(ev.cls == "dropout-overlap") & (ev.dur < .4)], 1, "dropout artefact")
FS = P.FS_ADC
fig, ax = plt.subplots(10, 2, figsize=(16, 26)); ax = ax.T.ravel()
sos2 = signal.butter(4, 2, fs=FS, output="sos"); sos20 = signal.butter(4, 20, fs=FS, output="sos")
for k, (i, label) in enumerate(pick[:20]):
    e = ev.loc[i]; ch = cmap[(int(raw.run[e.ci]), int(raw.idx[e.ci]))]; v, t = io.load(ch)
    x = v - np.median(v); tt = P.time_axis(t); segs = P.segments(t)
    f0 = P.mains_f0(x[max(segs, key=lambda s: s[1]-s[0])[0]:max(segs, key=lambda s: s[1]-s[0])[1]])
    t0, t1 = e.a / 100 - 4, e.e / 100 + 4
    for a, b in segs:
        if b - a < 200 or tt[b-1] < t0 or tt[a] > t1: continue
        xs = P.remove_mains(x[a:b], f0)
        y20 = signal.sosfiltfilt(sos20, xs, padtype="even", padlen=min(b-a-1, 7000))
        y2 = signal.sosfiltfilt(sos2, xs, padtype="even", padlen=min(b-a-1, 7000))
        m = (tt[a:b] >= t0) & (tt[a:b] <= t1)
        base = np.median(y2[m]) if m.any() else 0
        ax[k].plot(tt[a:b][m], xs[m] - base, lw=.25, color=RUNC[0], alpha=.45)
        ax[k].plot(tt[a:b][m], y20[m] - base, lw=.8, color=RUNC[1])
        ax[k].plot(tt[a:b][m], y2[m] - base, lw=1.4, color="#0b0b0b")
    for g in np.where(np.diff(t) > 1000)[0]:
        if t0 < tt[g] < t1: ax[k].axvline(tt[g], color=RUNC[3], lw=1.5)
    ax[k].axvspan(e.a / 100, e.e / 100, color=RUNC[2], alpha=.12)
    ax[k].set_xlim(t0, t1)
    ax[k].set_title(f"[{label}] {e.start:%m-%d %H:%M:%S} run{e.run}  |dev|max {e.amax:,.0f}  dur {e.dur:.1f}s  pat {e.pattern}  SNR {e.snr:.0f}", fontsize=8)
ax[0].plot([], [], color=RUNC[0], label="raw, mains removed (2416 Hz)"); ax[0].plot([], [], color=RUNC[1], label="LP 20 Hz"); ax[0].plot([], [], color="#0b0b0b", label="LP 2 Hz")
ax[0].legend(fontsize=7, ncol=3, loc="lower left")
for a in ax[9::10]: a.set_xlabel("s within chunk")
fig.supylabel("counts relative to local median")
fig.tight_layout(); fig.savefig("../figures/fig7_event_gallery.png", dpi=90)
