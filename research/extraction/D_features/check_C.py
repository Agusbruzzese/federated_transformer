"""PROTOCOL check (b): recompute D's representation comparison on events from workstream C's detector.

Only the SEGMENTATION changes (which events, where they are centred); D's preprocessing (20 Hz LP ->
100 Hz, 60-s running-median baseline) and feature code are unchanged, so any ranking change is
attributable to the event definition.

Usage
  python check_C.py extract            # run ../C_detection/detector.py on every 30-chunk block -> C_ windows
  python check_C.py extract --events F # instead cut windows at events given in F (parquet/csv with columns
                                       #   run (0-3), chunk (file index), sample (raw sample offset in chunk))
  PREFIX=C_ N_EVAL_DEV=700 N_EVAL_HOLD=500 python compute_all.py
  python check_C.py evaluate           # evaluate + stability vs D's own events (eval_main_summary.csv)
"""
import sys, os, json, hashlib, datetime as dt
import numpy as np, pandas as pd
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "C_detection"))
from io_detect import load_block, lowpass_dec, robust_baseline, FS, FSD, DEC
from scan_all import make_blocks, W_LP, W_RAW, SCR

EVENTS_FILE = None
CONTEXT_S = 30.0      # C asks for >=60 s context for its baseline; events whose centre is <30 s from a block edge are dropped


def _cut(blk, v, t, x, r, base, c_raw, rng=None, kind="event", extra=None):
    c = int(round(c_raw / DEC))
    a, b = c - W_LP // 2, c + W_LP // 2
    ra, rb = c * DEC - W_RAW // 2, c * DEC + W_RAW // 2
    if a < 0 or b > len(x) or ra < 0 or rb > len(v):
        return None
    ci = min(c * DEC // 100000, len(blk) - 1)
    wall = blk[ci][2] + dt.timedelta(microseconds=int(t[c * DEC] - t[ci * 100000]))
    dd = np.diff(t[ra:rb].astype(np.int64)); med = np.median(np.diff(t[:20000].astype(np.int64))); gp = dd > 1.5 * med
    w = r[a:b]
    d = dict(kind=kind, run=blk[0][0], chunk=blk[ci][1], wall=wall, n_gaps=int(gp.sum()),
             peak=float(np.abs(w[W_LP // 2 - 200:W_LP // 2 + 200]).max()), sigma=np.nan,
             dc_baseline=float(np.median(v)) + base[c], lp=w.astype(np.float32),
             raw=(v[ra:rb] - np.median(v) - base[c]).astype(np.float32))
    d.update(extra or {})
    return d


def proc(blk):
    import detector as C
    rng = np.random.default_rng(blk[0][0] * 100000 + blk[0][1])
    v, t = load_block([c[3] for c in blk])
    x = lowpass_dec(v); base = robust_baseline(x); r = x - base
    out = []; taken = np.zeros(len(x), bool)
    if EVENTS_FILE is None:
        ev, aux = C.detect_stream(v, t.astype(float))
        _, slot_of_raw, _ = C.insert_dropouts(v, t.astype(float))
        for e in ev:
            c_raw = int(np.searchsorted(slot_of_raw, round(e["centroid_t"] * C.FS_RAW)))
            on = int(np.searchsorted(slot_of_raw, round(e["onset"] * C.FS_RAW)))
            off = int(np.searchsorted(slot_of_raw, round(e["offset"] * C.FS_RAW)))
            taken[max(0, on // DEC - 500):off // DEC + 500] = True
            if c_raw < CONTEXT_S * FS or c_raw > len(v) - CONTEXT_S * FS:
                continue
            seg = r[on // DEC:off // DEC + 1]
            o = _cut(blk, v, t, x, r, base, c_raw, extra=dict(dur=e["dur"], pmax=float(seg.max()), pmin=float(seg.min()),
                                                                snr=e["snr"], c_has_dropout=e.get("has_dropout", np.nan)))
            if o: out.append(o)
    else:
        E = EVENTS_FILE
        for c in blk:
            for s in E[(E.run == c[0]) & (E.chunk == c[1])]["sample"].values:
                c_raw = (c[1] - blk[0][1]) * 100000 + int(s)
                o = _cut(blk, v, t, x, r, base, c_raw)
                if o:
                    cc = int(c_raw // DEC); seg = r[max(0, cc - 150):cc + 150]
                    o.update(dur=np.nan, pmax=float(seg.max()), pmin=float(seg.min()))
                    out.append(o); taken[max(0, cc - 800):cc + 800] = True
    for _ in range(2):
        c = int(rng.integers(W_LP, len(x) - W_LP))
        if taken[c - W_LP // 2:c + W_LP // 2].any():
            continue
        o = _cut(blk, v, t, x, r, base, c * DEC, kind="bg", extra=dict(dur=np.nan, pmax=np.nan, pmin=np.nan))
        if o: out.append(o)
    return out


def extract():
    global EVENTS_FILE
    if "--events" in sys.argv:
        f = sys.argv[sys.argv.index("--events") + 1]
        EVENTS_FILE = pd.read_parquet(f) if f.endswith(".parquet") else pd.read_csv(f)
    blocks = make_blocks()
    with Pool(4) as p:
        res = p.map(proc, blocks, chunksize=4)
    rows = [o for rr in res for o in rr]
    LP = np.stack([o.pop("lp") for o in rows]); RAW = np.stack([o.pop("raw") for o in rows])
    meta = pd.DataFrame(rows)
    if EVENTS_FILE is None:   # dur from C (s); for external events without dur, fall back to morph dur10 later
        pass
    np.save(SCR + "C_LP.npy", LP); np.save(SCR + "C_RAW.npy", RAW); meta.to_parquet(SCR + "C_meta.parquet")
    md5 = hashlib.md5(open(os.path.join(HERE, "..", "C_detection", "detector.py"), "rb").read()).hexdigest()
    info = dict(detector_md5=md5, detector_mtime=os.path.getmtime(os.path.join(HERE, "..", "C_detection", "detector.py")),
                n={f"{a}_{b}": int(v) for (a, b), v in meta.groupby(["run", "kind"]).size().items()})
    json.dump(info, open("C_events_info.json", "w"), indent=1, default=str)
    print(info)


def evaluate_C():
    import evaluate as E
    m = pd.read_parquet(SCR + "C_meta2.parquet")
    if m.dur.isna().all():      # external events without durations: use morph dur10 proxy
        F = np.load(SCR + "C_feats.npz"); m["dur"] = F["orig_morph"][:, 4]
        m.to_parquet(SCR + "C_meta2.parquet")
    results, R, st_hold = E.main(SCR + "C_feats.npz", SCR + "C_meta2.parquet", "C")
    D = pd.read_csv("eval_main_summary.csv")
    out = {"C_dev_vs_C_holdout": st_hold}
    for split in ["dev", "holdout"]:
        A = D[D.split == split].assign(split="D"); Bc = R[R.split == split].assign(split="C")
        out[f"D_{split}_vs_C_{split}"] = E.stability(pd.concat([A, Bc]), "D", "C")
    json.dump(out, open("eval_C_vs_D_stability.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    {"extract": extract, "evaluate": evaluate_C}[sys.argv[1]]()
