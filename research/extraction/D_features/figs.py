"""Figures: event vs background PSD; legacy front end vs correct front end on example events."""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy import signal
from features import legacy_read_int16, FS

SCR = "/tmp/claude-1000/-home-augustin-Documents-federated-transformer/8ccb016b-de96-4655-ab49-d3318afeb915/scratchpad/"
RAW = np.load(SCR + "RAW.npy", mmap_mode="r"); LP = np.load(SCR + "LP.npy", mmap_mode="r")
meta = pd.read_parquet(SCR + "meta.parquet")
rng = np.random.default_rng(0)
dev = meta.run.values != 2
ev = rng.choice(np.where((meta.kind == "event").values & dev)[0], 300, replace=False)
bg = rng.choice(np.where((meta.kind == "bg").values & dev)[0], 300, replace=False)

def psd(ix):
    P = []
    for i in ix:
        f, p = signal.welch(np.asarray(RAW[i], float), fs=FS, nperseg=4096)
        P.append(p)
    return f, np.array(P)

f, Pe = psd(ev); _, Pb = psd(bg)
fig, ax = plt.subplots(1, 2, figsize=(13, 4.2))
for P, lab, c in [(Pe, "events (n=300, dev)", "#1f5fa8"), (Pb, "background (n=300, dev)", "#999999")]:
    m = np.median(P, 0); lo, hi = np.percentile(P, [25, 75], 0)
    ax[0].loglog(f[1:], m[1:], color=c, label=lab); ax[0].fill_between(f[1:], lo[1:], hi[1:], color=c, alpha=.2)
ax[0].axvline(4, ls="--", c="k", lw=.8); ax[0].text(4.3, ax[0].get_ylim()[1] / 30, "4 Hz", fontsize=8)
ax[0].set_xlabel("Hz"); ax[0].set_ylabel("PSD (counts²/Hz)"); ax[0].legend(fontsize=8)
ax[0].set_title("Median Welch PSD, 10-s windows at 2400 Hz")
ratio = np.median(Pe, 0) / np.median(Pb, 0)
ax[1].semilogx(f[1:], 10 * np.log10(ratio[1:]), c="#1f5fa8"); ax[1].axhline(0, c="k", lw=.6)
ax[1].axhline(1, c="grey", lw=.6, ls=":"); ax[1].set_xlabel("Hz"); ax[1].set_ylabel("event / background (dB)")
ax[1].set_title("Where do vehicle events add energy?")
plt.tight_layout(); plt.savefig("fig_psd_event_vs_bg.png", dpi=90)
band = {f"{a}-{b} Hz": float(10 * np.log10(np.median(ratio[(f >= a) & (f < b)]))) for a, b in
        [(0, 1), (1, 2), (2, 4), (4, 8), (8, 20), (20, 45), (45, 55), (55, 200), (200, 1200)]}
print("event/bg median PSD ratio (dB) per band:", {k: round(v, 1) for k, v in band.items()})
pd.Series(band).to_csv("psd_ratio_bands_dB.csv")

# legacy vs correct front end
big = np.where((meta.kind == "event").values & dev & (meta.peak.values > 3e4))[0]
pick = [big[0], big[len(big) // 2], np.where((meta.kind == "event").values & dev)[0][7]]
fig, ax = plt.subplots(len(pick), 2, figsize=(13, 2.6 * len(pick)))
for r, i in enumerate(pick):
    codes = np.asarray(RAW[i], float) + meta.dc_baseline[i]
    t = np.arange(len(codes)) / FS - 5
    ax[r, 0].plot(t[::24], legacy_read_int16(codes)[::24], lw=.7, c="#b8412c")
    ax[r, 0].set_title(f"LEGACY input: int16 low word, naive ::24  (event {i})", fontsize=9)
    ax[r, 1].plot(np.arange(1200) / 100 - 6, LP[i], lw=.8, c="#1f5fa8")
    ax[r, 1].set_title("correct: 24-bit code, 20 Hz LP, 100 Hz, baseline removed", fontsize=9)
    ax[r, 1].set_xlim(-5, 5)
plt.tight_layout(); plt.savefig("fig_legacy_vs_correct_input.png", dpi=85)
