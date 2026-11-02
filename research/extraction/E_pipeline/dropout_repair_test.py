"""E design check: how should the feature path repair a >=5-sample dropout?

Real high-SNR events (run 0218 = Run 3 and run 0224 = Run 4, few natural dropouts) are processed intact (reference)
and with an artificial 106-sample dropout (the dominant dropout size, A) deleted at offsets -4..+4 s from the
centroid. The baseline-subtracted feature series around the event is compared with the reference.

Configurations: resample padding {constant (B's preproc default: zero padding), line} x
gap guard {0 s (only the hole is bridged), 0.15 s (B's +-0.15 s mask, bridged)}.
(Run with bridge_guard=True, the E pipeline's default being bridge_guard=False = the 'guard 0' repair.)
Metric: max |error| within +-6 s of the centroid, in units of the local noise sigma and of the event peak.
Output: dropout_repair_test.csv, stdout summary.
"""
import os, sys
import numpy as np, pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
from extraction.blockio import chunk_index, load_block, SlotClock
from extraction.featpath import feature_path
from extraction.pipeline import detect_block, STEP

HERE = os.path.dirname(os.path.abspath(__file__))
ix = chunk_index()


def job(arg):
    run_id, st = arg
    rng = np.random.default_rng(run_id * 100000 + st)
    rows = []
    run = sorted(ix.run.unique())[run_id]
    ixr = ix[ix.run == run].sort_values("index").reset_index(drop=True)
    clock = SlotClock(ixr)
    if True:
        B = load_block(ixr, st, st + 8)
        v, slot = B["v"], B["slot"]
        ev = detect_block(v, slot)
        evs = [(e["slot_onset"], e["slot_offset"]) for e in ev]
        cand = [e for e in ev if e["snr"] >= 30 and not e["has_dropout"]
                and 90 * 2416 < e["slot_centroid_t"] - slot[0] < len(v) - 90 * 2416]
        if not cand:
            return rows
        ref = {}
        for pad in ("constant", "line"):
            for g in (0.0, 0.15):
                ref[(pad, g)] = feature_path(v, slot, clock, evs, padtype=pad, gap_guard_s=g, bridge_guard=True)
        for e in cand[:2]:
            sc = e["slot_centroid_t"]
            ic = int(np.searchsorted(slot, sc))
            for off_s in np.arange(-4, 4.01, 1.0):
                p = ic + int(off_s * 2416) + int(rng.integers(0, 48))
                keep = np.ones(len(v), bool); keep[p:p + 106] = False
                for pad in ("constant", "line"):
                    for g in (0.0, 0.15):
                        R = ref[(pad, g)]
                        F = feature_path(v[keep], slot[keep], clock, evs, padtype=pad, gap_guard_s=g, bridge_guard=True)
                        jc = int(round(sc / STEP))
                        a, b = jc - 300 - R["J0"], jc + 300 - R["J0"]
                        a2, b2 = jc - 300 - F["J0"], jc + 300 - F["J0"]
                        err = F["r"][a2:b2] - R["r"][a:b]
                        sig = 1.4826 * np.median(np.abs(np.diff(R["r"][max(0, a - 3000):a]))) / np.sqrt(2)
                        peak = np.abs(R["r"][a:b]).max()
                        rows.append(dict(run=run_id, snr=e["snr"], off_s=off_s, pad=pad, guard=g,
                                         max_err=np.abs(err).max(), rms_err=np.sqrt(np.mean(err ** 2)),
                                         sigma=sig, peak=peak, max_err_sig=np.abs(err).max() / sig,
                                         max_err_peak=np.abs(err).max() / peak))
    return rows


if __name__ == "__main__":
  from multiprocessing import Pool
  jobs = [(2, s) for s in range(150, 4400, 150)] + [(3, s) for s in range(150, 6000, 200)]
  with Pool(14) as p:
    rows = [r for rr in p.map(job, jobs) for r in rr]
  df = pd.DataFrame(rows)
  df.to_csv(os.path.join(HERE, "dropout_repair_test.csv"), index=False)
  print(f"{df.groupby(['pad','guard']).size().iloc[0]} dropout placements on "
        f"{df[['run','snr']].drop_duplicates().shape[0]} events (SNR {df.snr.min():.0f}-{df.snr.max():.0f})")
  g = df.groupby(["pad", "guard"])
  print(pd.DataFrame({"median max|err|/sigma": g.max_err_sig.median(), "p90 max|err|/sigma": g.max_err_sig.quantile(0.9),
                      "max max|err|/sigma": g.max_err_sig.max(), "median max|err|/peak": g.max_err_peak.median(),
                      "p90 max|err|/peak": g.max_err_peak.quantile(0.9)}).round(3).to_string())
