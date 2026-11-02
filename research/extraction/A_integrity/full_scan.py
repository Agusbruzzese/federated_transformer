"""Full scan of ALL chunk files (Time + Values), one chunk at a time, 5 worker processes.
Per chunk: exact dropout list via bsd_io.slots_from_timestamps (v2), timestamp checks,
value sanity (range, rails, zeros, stuck runs, isolated spikes).
Outputs: full_scan_chunks.parquet (1 row / chunk), full_scan_gaps.parquet (1 row / dropout
or late-stamp event)."""
import os, sys, time, numpy as np, pandas as pd
from multiprocessing import Pool
import bsd_io as b
HERE = os.path.dirname(os.path.abspath(__file__))

def one(a):
    run, idx, tp, vp = a
    t = np.fromfile(tp, "<u8").astype(np.int64)
    v = np.fromfile(vp, "<u4").astype(np.int64)
    d = np.diff(t); med = np.median(d)
    slots, base = b.slots_from_timestamps(t)
    inc = np.diff(slots)
    gflag = d >= b.GAP_FACTOR * med
    ev = np.flatnonzero(gflag | (inc > 1))
    gaps = pd.DataFrame(dict(run=run, index=idx, pos=ev + 1, n_missing=inc[ev] - 1, dt_ticks=d[ev],
                             dt_slots=d[ev] / base))
    ratio = d / base
    dv = np.diff(v)
    same = np.concatenate([[0], (dv == 0).astype(np.int8), [0]]); e = np.flatnonzero(np.diff(same))
    runs = e[1::2] - e[::2]
    mad = 1.4826 * np.median(np.abs(dv - np.median(dv))) + 1e-9
    aa = v[1:-1] - v[:-2]; bb = v[1:-1] - v[2:]
    spikes = int(np.sum((np.sign(aa) == np.sign(bb)) & (np.minimum(np.abs(aa), np.abs(bb)) > 20 * mad)))
    row = dict(run=run, index=idx, n=len(t), n_v=len(v), base_dt=base, dt_min=int(d.min()) if len(d) else 0,
               n_nonpos=int((d <= 0).sum()), n_missing=int(slots[-1] - (len(t) - 1)),
               n_drop_events=int((inc > 1).sum()), n_late_only=int((gflag & (inc == 1)).sum()),
               n_ambiguous_ratio=int(((ratio > 1.3) & (ratio < 1.7)).sum()),   # would-be hidden single losses
               max_dt_ticks=int(d.max()) if len(d) else 0, max_gap_slots=int(inc.max() - 1) if len(inc) else 0,
               vmin=int(v.min()), vmax=int(v.max()), top_byte=int((v >> 24).max()), n_zero=int((v == 0).sum()),
               max_same_run=int(runs.max()) if len(runs) else 0, n_spikes=spikes,
               t_first=int(t[0]), t_last=int(t[-1]))
    return row, gaps

if __name__ == "__main__":
    idx = b.list_chunks()
    jobs = list(zip(idx.run, idx["index"], idx.time_path, idx.values_path))
    t0 = time.time(); rows = []; gl = []
    with Pool(5) as p:
        for i, (r, g) in enumerate(p.imap(one, jobs, chunksize=16)):
            rows.append(r); gl.append(g)
            if i % 2000 == 0:
                print(f"{i}/{len(jobs)}  {time.time()-t0:.0f}s", flush=True)
    C = pd.DataFrame(rows); G = pd.concat(gl, ignore_index=True)
    C.to_parquet(os.path.join(HERE, "full_scan_chunks.parquet")); G.to_parquet(os.path.join(HERE, "full_scan_gaps.parquet"))
    print(f"done {len(C)} chunks, {len(G)} events, {time.time()-t0:.0f}s")
