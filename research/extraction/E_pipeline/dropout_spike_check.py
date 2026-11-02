"""Gallery finding: ~1000-count spikes at dropouts in run 0204 windows. Repair artefact or in the raw data?
For every >=5-sample dropout within +-10 s of two gallery events, compare (a) the raw 24-bit codes around the gap,
smoothed by a 1-mains-period (20 ms) moving average that needs no filtering across the gap, with (b) the E feature series."""
import os, sys, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.normpath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO)
from extraction.blockio import chunk_index, load_block, SlotClock
from extraction.featpath import feature_path
from extraction.pipeline import STEP
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ix = chunk_index(); runs = sorted(ix.run.unique())
m = pd.read_parquet(os.path.join(REPO, "data", "events", "run0_20250204_meta.parquet"))
ixr = ix[ix.run == runs[0]].sort_values("index").reset_index(drop=True); clock = SlotClock(ixr)
fig, axs = plt.subplots(2, 1, figsize=(12, 6))
rows = []
for ax, eid in zip(axs, ["20250204_01246744037", "20250204_01296041702"]):
    e = m[m.event_id == eid].iloc[0]
    p = int(np.flatnonzero(ixr["index"] == e.chunk)[0])
    B = load_block(ixr, p - 3, p + 4); v, slot = B["v"], B["slot"]
    F = feature_path(v, slot, clock)
    sc = e.slot_centroid
    lost = np.diff(slot) - 1; gaps = np.flatnonzero(lost >= 5) + 1
    gaps = gaps[np.abs(slot[gaps] - sc) < 10 * 2416]
    k = 48  # ~20 ms moving average = one mains period, computed only from one side of each gap
    for b in gaps:
        pre = v[b - 600:b]; post = v[b:b + 600]
        if len(pre) < 600 or len(post) < 600 or (np.diff(slot[b - 600:b]) > 1).any() or (np.diff(slot[b:b + 600]) > 1).any():
            continue
        ma_pre = np.convolve(pre, np.ones(k) / k, "valid"); ma_post = np.convolve(post, np.ones(k) / k, "valid")
        base = np.median(np.r_[ma_pre[:200], ma_post[-200:]])
        # raw excursion right after the gap vs 0.2 s away
        rows.append(dict(event=eid, gap_slot=int(slot[b]), n_missing=int(lost[b - 1]),
                         raw_step_after_gap=float(ma_post[:50].mean() - ma_post[-100:].mean()),
                         raw_step_before_gap=float(ma_pre[-50:].mean() - ma_pre[:100].mean())))
    tt = (np.arange(len(F["r"])) + F["J0"]) * STEP
    sel = np.abs(tt - sc) < 10 * 2416
    ax.plot((tt[sel] - sc) / 2416, F["r"][sel], lw=0.9, label="E feature series")
    # raw 20 ms moving average, mains-free enough to see blips, median-removed
    ma = np.convolve(v - np.median(v), np.ones(k) / k, "same")
    rs = np.abs(slot - sc) < 10 * 2416
    ax.plot((slot[rs] - sc) / 2416, ma[rs] - np.median(ma[rs]), lw=0.6, alpha=0.7, label="raw, 20 ms moving average")
    for b in gaps:
        ax.axvline((slot[b] - sc) / 2416, color="r", lw=0.5, alpha=0.5)
    ax.set_title(eid + " (red lines: dropouts >= 5 samples)", fontsize=9); ax.legend(fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_dropout_spikes.png"), dpi=110)
R = pd.DataFrame(rows); print(R.round(0).to_string())
print("median raw step just AFTER a gap (vs 0.2 s later):", R.raw_step_after_gap.median().round(0),
      " just BEFORE (vs 0.2 s earlier):", R.raw_step_before_gap.median().round(0))
