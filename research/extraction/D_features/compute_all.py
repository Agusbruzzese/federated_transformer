"""Compute every representation on all windows (events + background) and on augmented
copies of an evaluation subset. Output -> scratchpad/feats.npz (+ small copy of meta)."""
import numpy as np, pandas as pd, time, sys
from multiprocessing import Pool
from scipy import signal
from features import (legacy66, hht_lp, REPS)

SCR = "/tmp/claude-1000/-home-augustin-Documents-federated-transformer/8ccb016b-de96-4655-ab49-d3318afeb915/scratchpad/"
import os
PREFIX = os.environ.get("PREFIX", "")            # "" = D's provisional events, "C_" = workstream C's detector events
N_EVAL_DEV, N_EVAL_HOLD = int(os.environ.get("N_EVAL_DEV", 700)), int(os.environ.get("N_EVAL_HOLD", 500))
LP = np.load(SCR + PREFIX + "LP.npy", mmap_mode="r")      # 12 s @100 Hz, 1200
RAW = np.load(SCR + PREFIX + "RAW.npy", mmap_mode="r")    # 10 s @2400 Hz, 24000
meta = pd.read_parquet(SCR + PREFIX + "meta.parquet")
LP_CUT = float(os.environ.get("LP_CUT", 0))       # extra zero-phase LP on the 100 Hz path (hyperparameter test)
OUT = PREFIX + (f"lp{int(LP_CUT)}_" if LP_CUT else "")
LP_ONLY = LP_CUT > 0                               # legacy (raw path) unaffected by the cut-off -> skip
_SOS_CUT = signal.butter(4, LP_CUT, fs=100.0, output="sos") if LP_CUT else None
meta["t_hours"] = meta.wall.dt.hour + meta.wall.dt.minute / 60 + meta.wall.dt.second / 3600
meta["t_hours_cum"] = (meta.wall - pd.Timestamp("2025-02-04")).dt.total_seconds() / 3600   # legacy t_hours (cumulative)
BG = np.where(meta.kind == "bg")[0]
AUGS = ["orig", "shift+0.1", "shift-0.25", "shift+0.5", "noise", "scale0.8", "stretch1.15"]
LP_C, RAW_C, L_LP, L_RAW = 600, 12000, 800, 19200


def crop(i, shift_s=0.0, stretch=1.0):
    if stretch == 1.0:
        a = LP_C - L_LP // 2 + int(round(shift_s * 100)); ra = RAW_C - L_RAW // 2 + int(round(shift_s * 2400))
        return np.array(LP[i, a:a + L_LP], float), np.array(RAW[i, ra:ra + L_RAW], float)
    n, nr = int(round(L_LP / stretch)), int(round(L_RAW / stretch))
    x = np.array(LP[i, LP_C - n // 2: LP_C - n // 2 + n], float)
    r = np.array(RAW[i, RAW_C - nr // 2: RAW_C - nr // 2 + nr], float)
    return signal.resample(x, L_LP), signal.resample(r, L_RAW)


def get_aug(i, aug, rng):
    x, r = _get_aug(i, aug, rng)
    if _SOS_CUT is not None:
        x = signal.sosfiltfilt(_SOS_CUT, x)
    return x, r


def _get_aug(i, aug, rng):
    if aug == "orig":
        return crop(i)
    if aug.startswith("shift"):
        return crop(i, float(aug[5:]))
    if aug == "noise":
        x, r = crop(i); j = int(rng.choice(BG)); xb, rb = crop(j)
        return x + xb, r + (rb - np.median(rb))
    if aug == "scale0.8":
        x, r = crop(i); return 0.8 * x, 0.8 * r
    if aug == "stretch1.15":
        return crop(i, stretch=1.15)


def all_reps(x, r, dcb, th, do_legacy=True, do_ceemdan=False):
    out = {}
    tim = {}
    for k, (f, _) in REPS.items():
        t = time.perf_counter(); out[k] = f(x); tim[k] = time.perf_counter() - t
    if do_legacy:
        t = time.perf_counter(); out["legacy66"] = legacy66(r + dcb, th); tim["legacy66"] = time.perf_counter() - t
    if do_ceemdan:
        t = time.perf_counter(); out["hht_lp_ceemdan"] = hht_lp(x, method="ceemdan"); tim["hht_lp_ceemdan"] = time.perf_counter() - t
        t = time.perf_counter(); out["hht_lp_eemd"] = hht_lp(x, method="eemd"); tim["hht_lp_eemd"] = time.perf_counter() - t
    return out, tim


def job(args):
    i, augs, ceem = args
    rng = np.random.default_rng(i)
    res = []
    for a in augs:
        x, r = get_aug(i, a, rng)
        o, t = all_reps(x, r, meta.dc_baseline[i], meta.t_hours_cum[i], do_legacy=not LP_ONLY, do_ceemdan=ceem)
        res.append((a, o, t))
    return i, res


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    ev = np.where(meta.kind == "event")[0]
    # PROTOCOL.md: run index 2 (folder 'Run 20250218 ...') is the HOLDOUT
    dev = ev[meta.run.values[ev] != 2]; hold = ev[meta.run.values[ev] == 2]
    EVAL = np.sort(np.r_[rng.choice(dev, min(N_EVAL_DEV, len(dev)), replace=False), rng.choice(hold, min(N_EVAL_HOLD, len(hold)), replace=False)])
    CEEM = set(np.sort(rng.choice(EVAL[meta.run.values[EVAL] != 2], min(300, N_EVAL_DEV), replace=False)).tolist())   # EMD-variant audit: dev only
    ES = set(EVAL.tolist()); tasks = []
    for i in range(len(meta)):
        if i in ES:
            tasks.append((i, AUGS, False))   # EEMD/CEEMDAN audited in emd_audit.py (serial cost ~3-8 s/event)
        else:
            tasks.append((i, ["orig"], False))
    t0 = time.time()
    with Pool(4) as p:
        res = p.map(job, tasks, chunksize=8)
    print("time", time.time() - t0)
    reps = list(res[0][1][0][1].keys())
    store = {}
    for k in reps:
        store["orig_" + k] = np.stack([rr[0][1][k] for _, rr in res])
    tim = {k: np.median([rr[0][2][k] for _, rr in res]) for k in reps}
    eval_pos = {i: n for n, i in enumerate(EVAL)}
    for k in reps:
        arr = []
        idx = EVAL if not k.startswith("hht_lp_") else np.array(sorted(CEEM))
        for i in idx:
            rr = res[i][1]
            arr.append(np.stack([o[k] for _, o, _ in rr]))
        store["aug_" + k] = np.stack(arr)
    np.savez(SCR + OUT + "feats.npz", EVAL=EVAL, CEEM=np.array(sorted(CEEM)), **store)
    meta.to_parquet(SCR + OUT + "meta2.parquet")
    pd.Series(tim).to_csv(OUT + "cost_per_event_s.csv")
    print(pd.Series(tim))
