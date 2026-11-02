"""Scan whole dataset with provisional detector; save event windows (LP 100 Hz 12 s, raw 2400 Hz 8 s) + background windows."""
import numpy as np, datetime as dt, pandas as pd, time, os
from multiprocessing import Pool
from io_detect import *
SCR = "/tmp/claude-1000/-home-augustin-Documents-federated-transformer/8ccb016b-de96-4655-ab49-d3318afeb915/scratchpad/"
W_LP = int(12 * FSD)      # 1200 samples @100 Hz
W_RAW = int(10 * FS)      # 24000 samples @2400 Hz
B = 30

def make_blocks():
    ch = list_chunks(); blocks = []; cur = []
    for c in ch:
        if cur and (c[0] != cur[-1][0] or c[1] != cur[-1][1] + 1 or len(cur) >= B):
            blocks.append(cur); cur = []
        cur.append(c)
    blocks.append(cur); return blocks

def proc(blk):
    rng = np.random.default_rng(blk[0][0] * 100000 + blk[0][1])
    v, t = load_block([c[3] for c in blk])
    dcraw = float(np.median(v))
    med_dt = float(np.median(np.diff(t[:20000].astype(np.int64))))
    x = lowpass_dec(v); r, sig, segs = detect(x)
    base = x - r
    out = []; taken = np.zeros(len(x), bool)
    for s, e in segs:
        env = np.convolve(np.abs(r[s:e]), np.ones(25) / 25, mode="same")
        c = int(s + np.argmax(env))   # centre on envelope peak (energy centroid mis-centres merged multi-vehicle events)
        taken[max(0, s - int(5*FSD)):e + int(5*FSD)] = True
        a, b = c - W_LP // 2, c + W_LP // 2
        ra, rb = c * DEC - W_RAW // 2, c * DEC + W_RAW // 2
        if a < 0 or b > len(x) or ra < 0 or rb > len(v):
            continue
        tus = t[c * DEC]
        # wall clock: chunk start name + (t - t_chunkstart)
        ci = min(c * DEC // 100000, len(blk) - 1)
        wall = blk[ci][2] + dt.timedelta(microseconds=int(t[c*DEC] - t[ci*100000]))
        rawwin = (v[ra:rb] - dcraw - base[c]).astype(np.float32)
        dd = np.diff(t[ra:rb].astype(np.int64)); gp = dd > 1.5 * med_dt
        out.append(dict(kind="event", run=blk[0][0], chunk=blk[ci][1], wall=wall, dur=(e - s) / FSD,
                        n_gaps=int(gp.sum()), n_missing=int(np.sum(np.round(dd[gp] / med_dt) - 1)),
                        on=(s - a) / FSD, off=(e - a) / FSD, peak=float(np.abs(r[s:e]).max()),
                        pmax=float(r[s:e].max()), pmin=float(r[s:e].min()), sigma=sig,
                        dc_baseline=dcraw + base[c], lp=r[a:b].astype(np.float32), raw=rawwin))
    # background windows: up to 2 per block
    free = np.where(~taken)[0]
    for _ in range(2):
        c = int(rng.integers(W_LP, len(x) - W_LP))
        if taken[c - W_LP // 2:c + W_LP // 2].any():
            continue
        ci = min(c * DEC // 100000, len(blk) - 1)
        wall = blk[ci][2] + dt.timedelta(microseconds=int(t[c*DEC] - t[ci*100000]))
        a, b = c - W_LP // 2, c + W_LP // 2; ra, rb = c * DEC - W_RAW // 2, c * DEC + W_RAW // 2
        dd = np.diff(t[ra:rb].astype(np.int64)); gp = dd > 1.5 * med_dt
        out.append(dict(kind="bg", run=blk[0][0], chunk=blk[ci][1], wall=wall, dur=np.nan, on=np.nan, off=np.nan,
                        n_gaps=int(gp.sum()), n_missing=int(np.sum(np.round(dd[gp] / med_dt) - 1)),
                        peak=float(np.abs(r[a:b]).max()), pmax=float(r[a:b].max()), pmin=float(r[a:b].min()), sigma=sig,
                        dc_baseline=dcraw + base[c], lp=r[a:b].astype(np.float32), raw=(v[ra:rb]-dcraw-base[c]).astype(np.float32)))
    return out

if __name__ == "__main__":
    import sys
    FLAGS_ONLY = "--flags-only" in sys.argv   # recompute metadata (dropout flags) without touching LP/RAW arrays
    blocks = make_blocks(); print("blocks", len(blocks)); t0 = time.time()
    with Pool(int(os.environ.get('NPROC', 4))) as p:
        res = p.map(proc, blocks, chunksize=4)
    rows = [o for r_ in res for o in r_]
    print("n", len(rows), "time", time.time() - t0)
    LP = np.stack([o.pop("lp") for o in rows]); RAW = np.stack([o.pop("raw") for o in rows])
    meta = pd.DataFrame(rows)
    if FLAGS_ONLY:
        meta.to_parquet(SCR + "meta_flags.parquet"); raise SystemExit
    np.save(SCR + "LP.npy", LP); np.save(SCR + "RAW.npy", RAW); meta.to_parquet(SCR + "meta.parquet")
    meta.to_csv("events_meta.csv", index=False)
    print(meta.groupby("kind").size()); print(meta[meta.kind=="event"].groupby(meta.wall.dt.date).size())
