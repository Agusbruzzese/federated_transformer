"""Real background-noise pool (implementation choice I5).

Stretches of B's feature path (the exact chain E used: extraction.featpath.feature_path) from TRAINING days only
(runs 0204, 0212, 0224 minus Feb 26; never run 0218). A 12.8 s stretch is accepted if:
  - no C detection of ANY SNR (including events E later excluded) has [onset - 3 s, offset + 3 s] overlapping it;
  - no dropout flag (hole or +-0.15 s guard) inside it;
  - it lies inside the block core;
  - max|x - median| <= 6 * robust sigma of the stretch (rejects sub-threshold transients).
Stored as unit-robust-sigma float32 [M, 640] plus provenance.

    python -m ssl_central.bgpool --blocks-per-run 8 --procs 12
"""
from __future__ import annotations

import argparse
import os
from multiprocessing import Pool

import numpy as np
import pandas as pd

from . import config as C

STEP = 1208 / 25


def _job(args):
    run, core, seed = args
    from extraction.blockio import chunk_index, SlotClock, load_block
    from extraction.featpath import feature_path
    from extraction.pipeline import detect_block
    ix = chunk_index()
    ixr = ix[ix.run == run].sort_values("index").reset_index(drop=True)
    c0, c1 = core; n = len(ixr)
    i0, i1 = max(0, c0 - 3), min(n, c1 + 3)
    B = load_block(ixr, i0, i1)
    clock = SlotClock(ixr)
    ev = detect_block(B["v"], B["slot"])
    F = feature_path(B["v"], B["slot"], clock, extra_event_slots=[(e["slot_onset"], e["slot_offset"]) for e in ev])
    r, mask, J0 = F["r"], F["mask"], F["J0"]
    busy = mask.copy()
    g = int(3.0 * 50)
    for e in ev:
        a = int(np.floor(e["slot_onset"] / STEP)) - J0 - g
        b = int(np.ceil(e["slot_offset"] / STEP)) - J0 + g
        busy[max(a, 0):max(min(b, len(r)), 0)] = True
    core_a = int(np.ceil(ixr.slot_start.iloc[c0] / STEP)) - J0
    core_b = (int(np.floor(ixr.slot_start.iloc[c1] / STEP)) - J0) if c1 < n else len(r)
    L = C.BG_LEN
    segs, prov = [], []
    cs = np.r_[0, np.cumsum(busy)]
    j = max(core_a, 0)
    while j + L <= min(core_b, len(r)):
        if cs[j + L] - cs[j] == 0:
            x = r[j:j + L].astype(np.float64)
            x = x - np.median(x)
            s = 1.4826 * np.median(np.abs(x))
            if s > 0 and np.abs(x).max() <= 6 * s:
                segs.append((x / s).astype(np.float32))
                wall = clock.wall_time(np.array([int((j + J0) * STEP)]))[0]
                prov.append(dict(run=run, j=j + J0, wall_time=wall, sigma=s))
                j += L
                continue
        j += L // 4
    rng = np.random.default_rng(seed)
    if len(segs) > 120:
        keep = np.sort(rng.choice(len(segs), 120, replace=False))
        segs = [segs[k] for k in keep]; prov = [prov[k] for k in keep]
    return segs, prov


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blocks-per-run", type=int, default=8)
    ap.add_argument("--procs", type=int, default=12)
    a = ap.parse_args()
    from extraction.blockio import chunk_index
    from extraction.pipeline import make_blocks
    ix = chunk_index()
    runs = sorted(ix.run.unique())
    rng = np.random.default_rng(0)
    jobs = []
    for rid in C.TRAIN_RUN_IDS:
        run = runs[rid]
        ixr = ix[ix.run == run].sort_values("index").reset_index(drop=True)
        dates = pd.to_datetime(ixr.wall_start).dt.strftime("%Y-%m-%d").to_numpy()
        ok = [(c0, c1) for c0, c1 in make_blocks(ixr) if not (dates[max(0, c0 - 3):min(len(ixr), c1 + 3)] == C.VAL_DATE).any()]
        pick = rng.choice(len(ok), min(a.blocks_per_run, len(ok)), replace=False)
        jobs += [(run, ok[k], int(rng.integers(1 << 30))) for k in sorted(pick)]
    print(f"{len(jobs)} blocks", flush=True)
    with Pool(a.procs) as p:
        res = p.map(_job, jobs)
    segs = [s for S, _ in res for s in S]
    prov = pd.DataFrame([q for _, P in res for q in P])
    assert not (prov.wall_time.dt.strftime("%Y-%m-%d") == C.VAL_DATE).any()
    assert not prov.run.str.contains("20250218").any()
    os.makedirs(C.OUT, exist_ok=True)
    np.save(os.path.join(C.OUT, "bgpool.npy"), np.stack(segs))
    prov.to_parquet(os.path.join(C.OUT, "bgpool_meta.parquet"))
    print(len(segs), "segments;", prov.groupby("run").size().to_dict(), flush=True)
    print(prov.wall_time.dt.hour.value_counts().sort_index().to_dict())
    print("sigma (counts):", prov.sigma.describe().round(1).to_dict())


if __name__ == "__main__":
    main()
