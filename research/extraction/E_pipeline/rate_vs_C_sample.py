"""Why are E's full-data rates below C's large-sample estimates (53 / 16.3 / 12.1 per h)?
Count E's events inside exactly C's sampled windows (C_detection/scripts/diurnal.py: 6-chunk blocks every 24 chunks,
30 s trimmed at each end; data/diurnal_blocks.parquet) and compare with C's HYST-BP counts there."""
import os, sys, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.normpath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO)
from extraction.blockio import chunk_index
C = os.path.join(REPO, "research", "extraction", "C_detection", "data")
b = pd.read_parquet(os.path.join(C, "diurnal_blocks.parquet")); d = pd.read_parquet(os.path.join(C, "diurnal_events.parquet"))
d = d[d.det == "HYSTBP"]
ix = chunk_index(); runs = sorted(ix.run.unique())
ev = pd.concat([pd.read_parquet(os.path.join(REPO, "data", "events", f)) for f in sorted(os.listdir(os.path.join(REPO, "data", "events"))) if f.endswith("_meta.parquet")])
ex = pd.read_parquet(os.path.join(REPO, "data", "events", "excluded.parquet"))
allev = pd.concat([ev.assign(excl=False), ex.assign(excl=True)])
rows = []
for r in b.itertuples():
    ixr = ix[ix.run == runs[r.run]].set_index("index")
    if r.block + 6 not in ixr.index:
        continue
    s0 = ixr.slot_start[r.block] + 30 * 2416; s1 = ixr.slot_start[r.block + 6] - 30 * 2416
    e = allev[(allev.run_id == r.run) & (allev.slot_centroid >= s0) & (allev.slot_centroid < s1)]
    c = d[(d.run == r.run) & (d.block == r.block)]
    rows.append(dict(run=r.run, block=r.block, hours=r.hours, E_all=len(e), E_20=(e.snr >= 20).sum(), E_30=(e.snr >= 30).sum(),
                     C_all=len(c), C_20=(c.snr >= 20).sum(), C_30=(c.snr >= 30).sum()))
T = pd.DataFrame(rows); H = T.hours.sum()
print(f"{len(T)} of C's sampled blocks, {H:.1f} h (C used {b.hours.sum():.1f} h)")
for k in ("all", "20", "30"):
    print(f"  tier {k:>3}: C {T['C_' + k].sum() / H:6.2f}/h   E on the same windows {T['E_' + k].sum() / H:6.2f}/h")
print("per run (SNR>=20 per h, C vs E):")
for rr, g in T.groupby("run"):
    print(f"  run {rr}: C {g.C_20.sum() / g.hours.sum():.2f}  E {g.E_20.sum() / g.hours.sum():.2f}  (all: C {g.C_all.sum()/g.hours.sum():.1f} E {g.E_all.sum()/g.hours.sum():.1f})")
print(f"E full-data rate incl. excluded events: all {len(allev)/340.10:.2f}/h, SNR>=20 {(allev.snr>=20).sum()/340.10:.2f}/h, SNR>=30 {(allev.snr>=30).sum()/340.10:.2f}/h")
print(f"C sample hours by hour of day vs E coverage share: C 07-09 share {b[b.hour.isin([7,8])].hours.sum()/b.hours.sum():.3f}")
T.to_csv(os.path.join(HERE, "rate_vs_C_sample.csv"), index=False)

# per-event: match C's sampled detections (centroid_t = seconds on its 2416 Hz slot clock from block start) to E
m_rows = []
for (rr, blk), c in d.groupby(["run", "block"]):
    ixr = ix[ix.run == runs[rr]].set_index("index")
    e = allev[(allev.run_id == rr) & (allev.slot_centroid >= ixr.slot_start[blk] - 60 * 2416)
              & (allev.slot_centroid < ixr.slot_start[blk] + 7 * 100000)]
    for x in c.itertuples():
        s = ixr.slot_start[blk] + x.centroid_t * 2416
        dd = np.abs(e.slot_centroid.to_numpy() - s) / 2416 if len(e) else np.array([np.inf])
        k = int(np.argmin(dd))
        m_rows.append(dict(C_snr=x.snr, dist_s=dd[k], E_snr=e.snr.iloc[k] if len(e) else np.nan,
                           C_off=x.centroid_t, C_len=None))
M = pd.DataFrame(m_rows)
mt = M.dist_s < 0.5
print(f"\nC sampled HYST-BP events: {len(M)}; matched to an E event within 0.5 s: {mt.mean():.3f}")
for lo in (20, 30):
    s = M[M.C_snr >= lo]
    print(f"  C SNR>={lo}: n={len(s)}, matched {(s.dist_s < 0.5).mean():.3f}, E SNR >= {lo} among matched "
          f"{(s[s.dist_s < 0.5].E_snr >= lo).mean():.3f}, median E/C SNR ratio {np.median(s[s.dist_s < 0.5].E_snr / s[s.dist_s < 0.5].C_snr):.3f}")
print(f"  all matched: median E/C SNR ratio {np.median(M[mt].E_snr / M[mt].C_snr):.3f}")
