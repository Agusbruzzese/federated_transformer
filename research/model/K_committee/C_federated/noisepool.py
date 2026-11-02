"""Committee C: PER-DAY real background-noise pools (so each FL client uses only noise from its own days).

Reuses ssl_central.bgpool._job unchanged (same acceptance rules I5: no detection of any SNR within 3 s,
no dropout, block core, max|x| <= 6 robust sigma, unit-robust-sigma 640-sample stretches, <= 120 per block).
Block choice per TRAINING day (13 days incl. the run-0218 days that P11 gives to the server): all 60-chunk
blocks whose +-3-chunk read range lies entirely inside that calendar day; BLOCKS_PER_DAY of them, evenly spaced
in time (deterministic, no RNG for the choice). Test/val days are never touched.

    .venv/bin/python research/model/K_committee/C_federated/noisepool.py [blocks_per_day=6] [procs=12]
    .venv/bin/python research/model/K_committee/C_federated/noisepool.py relaxed [blocks_per_day=6] [procs=12]
Strict result (6 blocks/day, 19 min on 14 procs): 02-06 44, 02-08 1, 02-09 0, 02-11 0 segments (dropout-dense run 0204),
hence the `relaxed` fallback for those four days only.
Output: data/k_committee/C_federated/bgpool_day.npy [M,640] + bgpool_day_meta.parquet (day column).
"""
import os, sys, time
from multiprocessing import Pool
import numpy as np, pandas as pd
ROOT = os.getcwd(); sys.path.insert(0, ROOT)
from ssl_central.bgpool import _job, STEP
from ssl_central import config as C

RELAX_DAYS = ["2025-02-06", "2025-02-08", "2025-02-09", "2025-02-11"]   # strict I5 yields 44 / 1 / 0 / 0 segments
MAX_MASK = 0.10


def _job_relaxed(args):
    """Fallback for dropout-dense days (run 0204 from Feb 6: 61-93 % of SNR>=20 windows contain dropouts).
    Same as ssl_central.bgpool._job EXCEPT the dropout flag no longer vetoes a stretch: a stretch is accepted if
    <= 10 % of its samples are flagged; flagged samples are linearly bridged from their neighbours and the robust
    sigma / 6-sigma test use unflagged samples only. Detections of any SNR (+-3 s) still veto."""
    run, core, seed = args
    from extraction.blockio import chunk_index, SlotClock, load_block
    from extraction.featpath import feature_path, bridge
    from extraction.pipeline import detect_block
    ix = chunk_index(); ixr = ix[ix.run == run].sort_values("index").reset_index(drop=True)
    c0, c1 = core; n = len(ixr); i0, i1 = max(0, c0 - 3), min(n, c1 + 3)
    B = load_block(ixr, i0, i1); clock = SlotClock(ixr); ev = detect_block(B["v"], B["slot"])
    F = feature_path(B["v"], B["slot"], clock, extra_event_slots=[(e["slot_onset"], e["slot_offset"]) for e in ev])
    r, mask, J0 = F["r"], F["mask"], F["J0"]
    busy = np.zeros(len(r), bool); g = int(3.0 * 50)
    for e in ev:
        a = int(np.floor(e["slot_onset"] / STEP)) - J0 - g; b = int(np.ceil(e["slot_offset"] / STEP)) - J0 + g
        busy[max(a, 0):max(min(b, len(r)), 0)] = True
    core_a = int(np.ceil(ixr.slot_start.iloc[c0] / STEP)) - J0
    core_b = (int(np.floor(ixr.slot_start.iloc[c1] / STEP)) - J0) if c1 < n else len(r)
    L = C.BG_LEN; segs, prov = [], []; cs = np.r_[0, np.cumsum(busy)]; cm = np.r_[0, np.cumsum(mask)]; j = max(core_a, 0)
    while j + L <= min(core_b, len(r)):
        mf = (cm[j + L] - cm[j]) / L
        if cs[j + L] - cs[j] == 0 and mf <= MAX_MASK:
            mk = mask[j:j + L]; x = bridge(r[j:j + L].astype(np.float64).copy(), mk) if mk.any() else r[j:j + L].astype(np.float64)
            x = x - np.median(x[~mk]); sg = 1.4826 * np.median(np.abs(x[~mk]))
            if sg > 0 and np.abs(x[~mk]).max() <= 6 * sg:
                segs.append((x / sg).astype(np.float32))
                prov.append(dict(run=run, j=j + J0, wall_time=clock.wall_time(np.array([int((j + J0) * STEP)]))[0], sigma=sg, mask_frac=mf))
                j += L; continue
        j += L // 4
    rng = np.random.default_rng(seed)
    if len(segs) > 120:
        keep = np.sort(rng.choice(len(segs), 120, replace=False)); segs = [segs[k] for k in keep]; prov = [prov[k] for k in keep]
    return segs, prov

OUT = os.path.join(ROOT, "data", "k_committee", "C_federated")
TRAIN_DAYS = ["2025-02-04", "2025-02-05", "2025-02-06", "2025-02-08", "2025-02-09", "2025-02-11", "2025-02-12",
              "2025-02-14", "2025-02-18", "2025-02-20", "2025-02-24", "2025-02-26", "2025-02-27"]
FORBIDDEN = ["2025-02-07", "2025-02-13", "2025-02-19", "2025-02-25", "2025-02-10"]


def plan(bpd):
    from extraction.blockio import chunk_index
    from extraction.pipeline import make_blocks
    ix = chunk_index(); jobs = []
    for run in sorted(ix.run.unique()):
        ixr = ix[ix.run == run].sort_values("index").reset_index(drop=True); n = len(ixr)
        dates = pd.to_datetime(ixr.wall_start).dt.strftime("%Y-%m-%d").to_numpy()
        for day in TRAIN_DAYS:
            ok = [(c0, c1) for c0, c1 in make_blocks(ixr) if set(dates[max(0, c0 - 3):min(n, c1 + 3)]) == {day}]
            if not ok: continue
            pick = np.unique(np.round(np.linspace(0, len(ok) - 1, min(bpd, len(ok)))).astype(int))
            jobs += [(run, ok[k], 1000 * TRAIN_DAYS.index(day) + int(k)) for k in pick]
    return jobs


def relaxed():
    """Adds relaxed-rule segments for RELAX_DAYS to the strict pool (rows tagged rule='relaxed')."""
    procs = int(sys.argv[3]) if len(sys.argv) > 3 else 12
    jobs = [j for j in plan(int(sys.argv[2]) if len(sys.argv) > 2 else 6)
]
    from extraction.blockio import chunk_index
    ix = chunk_index(); ix["day"] = pd.to_datetime(ix.wall_start).dt.strftime("%Y-%m-%d")
    def day_of(job):
        run, (c0, c1), _ = job; ixr = ix[ix.run == run].sort_values("index").reset_index(drop=True); return ixr.day.iloc[c0]
    jobs = [j for j in jobs if day_of(j) in RELAX_DAYS]; print(len(jobs), "relaxed blocks", flush=True); t0 = time.time()
    with Pool(procs) as p: res = p.map(_job_relaxed, jobs)
    segs = [s for S, _ in res for s in S]; prov = pd.DataFrame([q for _, P in res for q in P])
    prov["day"] = pd.to_datetime(prov.wall_time).dt.strftime("%Y-%m-%d"); prov["rule"] = "relaxed"
    assert not prov.day.isin(FORBIDDEN).any()
    bg = np.load(os.path.join(OUT, "bgpool_day.npy")); meta = pd.read_parquet(os.path.join(OUT, "bgpool_day_meta.parquet"))
    if "rule" not in meta: meta["rule"] = "strict"
    keep = (meta.rule == "strict").to_numpy()
    np.save(os.path.join(OUT, "bgpool_day.npy"), np.concatenate([bg[keep], np.stack(segs)]))
    pd.concat([meta[keep], prov], ignore_index=True).to_parquet(os.path.join(OUT, "bgpool_day_meta.parquet"))
    print(f"{len(segs)} relaxed segments in {time.time() - t0:.0f}s")
    print(prov.groupby("day").agg(n=("sigma", "size"), sigma_med=("sigma", "median"), mask_frac_mean=("mask_frac", "mean")).to_string())


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "relaxed": return relaxed()
    bpd = int(sys.argv[1]) if len(sys.argv) > 1 else 6; procs = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    jobs = plan(bpd); print(len(jobs), "blocks", flush=True); t0 = time.time()
    with Pool(procs) as p: res = p.map(_job, jobs)
    segs = [s for S, _ in res for s in S]; prov = pd.DataFrame([q for _, P in res for q in P])
    prov["day"] = pd.to_datetime(prov.wall_time).dt.strftime("%Y-%m-%d")
    assert not prov.day.isin(FORBIDDEN).any()
    os.makedirs(OUT, exist_ok=True)
    np.save(os.path.join(OUT, "bgpool_day.npy"), np.stack(segs)); prov.to_parquet(os.path.join(OUT, "bgpool_day_meta.parquet"))
    print(f"{len(segs)} segments in {time.time() - t0:.0f}s"); print(prov.groupby("day").agg(n=("sigma", "size"), sigma_med=("sigma", "median")).to_string())


if __name__ == "__main__":
    main()
