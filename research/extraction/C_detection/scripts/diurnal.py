"""Large-sample pass: diurnal event-rate plausibility, chunk-boundary straddling, durations, dropout flags, cost.
Blocks of 6 consecutive chunks (~4.1 min) every 24 chunks (~1/4 of the data) in every run; events are counted
only if their segment midpoint lies in the block's central part (30 s context trimmed at both ends).
usage: diurnal.py  -> data/diurnal_events.parquet, data/diurnal_blocks.parquet
All five detectors at the frozen operating points (data/frozen_params.json); legacy FIXED k=5 as well.
"""
import sys, os, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..')); sys.path.insert(0, os.path.dirname(__file__))
from common import *
import cio, pandas as pd, datetime as dt
from build_spans import load as load_raw
from inject_eval import run_setting
from multiprocessing import Pool

FS = D.FS
BLOCK, EVERY, TRIM = 6, 24, 30.0
FZ = json.load(open(os.path.join(os.path.dirname(__file__), '..', 'data', 'frozen_params.json')))
SETS = [(d, FZ[d]['thr']) for d in NAMES] + [(f'{d}@legacy', FZ[d]['legacy_thr']) for d in NAMES if 'legacy_thr' in FZ[d]]


def work(args):
    ri, i0 = args
    r = cio.runs()[ri]
    T0 = time.process_time()
    x, t, c, wall = load_raw(r, i0, BLOCK)
    t_io = time.process_time() - T0
    if len(x) < 5 * 100000:
        return [], []
    T1 = time.process_time()
    t50, y50, idx, drop = D.front_end(x, t)
    _, slot_of_raw, _ = D.insert_dropouts(x, t)
    ridx = np.clip(np.searchsorted(slot_of_raw, idx), 0, len(x) - 1)
    P = prep(y50)
    t_prep = time.process_time() - T1
    ch, wl = c[ridx], wall[ridx]
    a0, a1 = int(TRIM * FS), len(t50) - int(TRIM * FS)
    ev, costs = [], {}
    for (d, thr) in SETS:
        T2 = time.process_time()
        segs = run_setting(d, thr, t50, P)
        costs[d] = time.process_time() - T2
        for k, (a, b) in enumerate(segs):
            # context energy extent: +-10 s around the segment, clipped halfway to neighbouring segments
            lo = max(a - int(10 * FS), (segs[k - 1][1] + a) // 2 if k > 0 else 0)
            hi = min(b + int(10 * FS), (b + segs[k + 1][0]) // 2 if k + 1 < len(segs) else len(t50))
            cz = np.cumsum(P['z'][lo:hi] ** 2); cz /= max(cz[-1], 1e-12)
            x_on, x_off = t50[lo + int(np.searchsorted(cz, 0.01))], t50[lo + min(int(np.searchsorted(cz, 0.99)), hi - lo - 1)]
            mid = (a + b) // 2
            if not (a0 <= mid < a1):
                continue
            e = D.describe(np.array([[a, b]]), t50, P['z'], P['sigma'])[0]
            ia = int(np.clip(round((e['e_on'] - t50[0]) * FS), 0, len(t50) - 1))
            ib = int(np.clip(round((e['e_off'] - t50[0]) * FS), 0, len(t50) - 1))
            ic = int(np.clip(round((e['centroid_t'] - t50[0]) * FS), 0, len(t50) - 1))
            w = dt.datetime.fromtimestamp(wl[ic])
            ev.append(dict(run=ri, block=i0, det=d, snr=e['snr'], dur=e['dur'], e_dur=e['e_off'] - e['e_on'],
                           peak_z=e['peak_z'], wall=wl[ic], hour=w.hour, dow=w.weekday(), date=w.strftime('%m-%d'),
                           straddle_seg=bool(ch[a] != ch[min(b, len(ch) - 1)]), straddle_e=bool(ch[ia] != ch[ib]),
                           has_dropout=bool(drop[a:b + 1].any()), onset_t=t50[a], offset_t=t50[min(b, len(t50) - 1)],
                           centroid_t=e['centroid_t'], e_on=e['e_on'], e_off=e['e_off'], peak_t=e['peak_t'],
                           t_block0=t50[0], t_block1=t50[-1], x_on=x_on, x_off=x_off, x_dur=x_off - x_on))
    w0 = dt.datetime.fromtimestamp(wl[a0])
    blk = dict(run=ri, block=i0, hours=(a1 - a0) / FS / 3600, wall=wl[a0], hour=w0.hour, dow=w0.weekday(),
               date=w0.strftime('%m-%d'), t_io=t_io, t_prep=t_prep, data_hours=len(t50) / FS / 3600,
               drop_frac=drop.mean(), **{f'cpu_{k}': v for k, v in costs.items()})
    return ev, [blk]


if __name__ == '__main__':
    jobs = []
    for ri, r in enumerate(cio.runs()):
        n = len(cio.index(r))
        jobs += [(ri, i) for i in range(0, n - BLOCK, EVERY)]
    with Pool(4) as pool:
        res = pool.map(work, jobs, chunksize=8)
    out = os.path.join(os.path.dirname(__file__), '..', 'data')
    pd.DataFrame([e for r in res for e in r[0]]).to_parquet(f'{out}/diurnal_events.parquet')
    pd.DataFrame([b for r in res for b in r[1]]).to_parquet(f'{out}/diurnal_blocks.parquet')
    print('blocks', sum(len(r[1]) for r in res))
