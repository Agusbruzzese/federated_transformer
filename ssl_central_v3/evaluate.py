"""Phase G evaluation on the OOF embeddings: P1 (kNN, J5-J8), P2 (LR, J9-J11), CNN vs TF (J13), secondary.

    python -m ssl_central_v3.evaluate sanity p1 p2 p2ws tiers p3p4 hgb verdict
Tables -> research/model/G_loro/tab_*.csv ; caches -> data/ssl_central_v3/cache/ (incremental).
"""
from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

import json
import sys
import time

import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

from ssl_central import augment as A
from ssl_central import embed as FE
from ssl_central.probes import hgb_oof
from . import config as C
from . import data as D
from . import probes as P
from .embed import EMB, sel
from .train import ckpt, tag

CACHE = os.path.join(C.OUT, "cache")
N_JOBS = 16
ARCHS = C.ARCHS
SEEDS = C.SEEDS


# ------------------------------------------------------------------ helpers
def fast_metric(y, p, kind):
    if kind == "reg":
        return 1.0 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)
    cls, yi = np.unique(y, return_inverse=True)
    K = len(cls)
    hit = (y == p).astype(float)
    rec = np.bincount(yi, weights=hit, minlength=K) / np.bincount(yi, minlength=K)
    return (rec.mean() - 1 / K) / (1 - 1 / K)


def boot(y, p, kind, idxs):
    return np.array([fast_metric(y[i], p[i], kind) for i in idxs])


def load_rep(name, m):
    if name == "morph8":
        return m[[c for c in m.columns if c.startswith("m8_")]].to_numpy(float)
    if name == "snr":
        return m.log_snr.to_numpy(float)[:, None]
    return np.load(os.path.join(EMB, f"{name}.npy"))


def model_reps(prefixes=("cnn_ssl", "cnn_rand", "tf_ssl", "tf_rand")):
    return [f"{p}_s{s}" for p in prefixes for s in SEEDS]


def cached(name, fn):
    f = os.path.join(CACHE, name + ".pkl")
    if os.path.exists(f):
        return pd.read_pickle(f)
    r = fn(); os.makedirs(CACHE, exist_ok=True); pd.to_pickle(r, f)
    return r


def pct(a):
    return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))


def save(df, name):
    df.to_csv(os.path.join(C.RES, name), index=False)


# ------------------------------------------------------------------ sanity: amp_resid
def sanity():
    m, _ = D.load_all()
    idx = np.flatnonzero(D.tier_mask(m)); s = m.iloc[idx]; g = s.blk30.to_numpy()
    sp = P.splits(g)
    la = s.m8_log_amp.to_numpy(); ls = s.log_snr.to_numpy()
    yt = np.empty(len(s)); pl = np.empty(len(s))
    for tr, te in sp:
        rtr, rte = D.amp_resid(la, ls, tr, te)
        yt[te] = rte; pl[te] = 0.0                       # a linear SNR model predicts residual 0 by construction
        b = np.polyfit(ls[tr], rtr, 1); assert abs(b[0]) < 1e-8   # residual uncorrelated with SNR in-fold
    lin = fast_metric(yt, pl, "reg")
    yk, pk = P.knn_oof(ls, s, "amp_resid", g)
    knn = fast_metric(yk, pk, "reg")
    yraw, praw = P.knn_oof(ls, s.assign(y_log_dur=s.m8_log_amp), "log_dur", g)  # SNR-alone kNN on raw log_amp
    out = dict(n=len(s), r2_linear_snr_on_amp_resid=lin, r2_knn_snr_on_amp_resid=knn,
               r2_knn_snr_on_raw_log_amp=fast_metric(yraw, praw, "reg"),
               corr_amp_snr=float(np.corrcoef(la, ls)[0, 1]))
    json.dump(out, open(os.path.join(C.RES, "sanity_amp_resid.json"), "w"), indent=1)
    print(out)
    # training-loss check
    rows = []
    for d in sorted(os.listdir(os.path.join(C.OUT, "ckpt"))):
        lg = json.load(open(os.path.join(C.OUT, "ckpt", d, "log.json")))
        rows.append(dict(tag=d, fold=lg["fold"], arch=lg["arch"], lr=lg["lr"], temp=lg["temp"], seed=lg["seed"],
                         nostretch=lg["nostretch"], n_train=lg["n_train"], n_pool=lg["n_pool"],
                         loss_first=lg["loss"][0], loss_last=lg["loss"][-1],
                         loss_min_last10=min(lg["loss"][-10:]), falls=lg["loss"][-1] < lg["loss"][0],
                         guard=np.mean(lg["guard_frac"]), total_s=lg["total_s"]))
    T = pd.DataFrame(rows); save(T, "tab_training_runs.csv")
    print(f"{len(T)} runs; loss falls on all: {T.falls.all()}; total train {T.total_s.sum() / 60:.1f} min")


# ------------------------------------------------------------------ P1
def p1_rep(rep, tier):
    m, _ = D.load_all()
    idx = np.flatnonzero(D.tier_mask(m, tier)); s = m.iloc[idx]; g = s.blk30.to_numpy()
    X = load_rep(rep, m)[idx]
    sp = P.splits(g); boots = P.boot_indices(g, C.N_BOOT, 0)
    run = s.run_id.to_numpy()
    out = {}
    for t in C.TARGETS:
        k = C.KIND[t]
        yt, pr = P.knn_oof(X, s, t, None, sp=sp)
        out[t] = dict(value=fast_metric(yt, pr, k), boot=boot(yt, pr, k, boots),
                      per_run={r: fast_metric(yt[run == r], pr[run == r], k) for r in C.RUNS})
    return out


def p1_tier(tier, tname, reps):
    todo = [r for r in reps if not os.path.exists(os.path.join(CACHE, f"p1_{tname}_{r}.pkl"))]
    res = Parallel(n_jobs=N_JOBS)(delayed(p1_rep)(r, tier) for r in todo)
    os.makedirs(CACHE, exist_ok=True)
    for r, x in zip(todo, res):
        pd.to_pickle(x, os.path.join(CACHE, f"p1_{tname}_{r}.pkl"))
    return {r: pd.read_pickle(os.path.join(CACHE, f"p1_{tname}_{r}.pkl")) for r in reps}


def paired(R, a_prefix, b_prefix, t, per_run=False):
    """mean_s (a_s - b_s); b may be a single rep name (broadcast)."""
    A_ = [R[f"{a_prefix}_s{s}"][t] for s in SEEDS]
    B_ = [R[f"{b_prefix}_s{s}"][t] for s in SEEDS] if f"{b_prefix}_s0" in R else [R[b_prefix][t]] * len(SEEDS)
    d_s = np.array([a["value"] - b["value"] for a, b in zip(A_, B_)])
    d_b = np.mean([a["boot"] - b["boot"] for a, b in zip(A_, B_)], 0)
    r = dict(delta=float(d_s.mean()), seed_sd=float(d_s.std(ddof=1)), seed_min=float(d_s.min()),
             seed_max=float(d_s.max()), lo=pct(d_b)[0], hi=pct(d_b)[1],
             p_one_sided=float((1 + np.sum(d_b <= 0)) / (1 + len(d_b))))
    for k in C.RUNS:
        r[f"d_run{C.RUN_NAME[k]}"] = float(np.mean([a["per_run"][k] - b["per_run"][k] for a, b in zip(A_, B_)]))
    r["n_runs_pos"] = int(sum(r[f"d_run{C.RUN_NAME[k]}"] > 0 for k in C.RUNS))
    return r


def p1_tables(R, tname):
    vals = []
    for rep, x in R.items():
        for t in C.TARGETS:
            vals.append(dict(tier=tname, rep=rep, target=t, value=x[t]["value"], lo=pct(x[t]["boot"])[0],
                             hi=pct(x[t]["boot"])[1], **{f"run{C.RUN_NAME[k]}": x[t]["per_run"][k] for k in C.RUNS}))
    prefixes = sorted({r.rsplit("_s", 1)[0] for r in R if r[-3:-1] == "_s" and r[-1].isdigit()})
    for p in prefixes:
        for t in C.TARGETS:
            v = [R[f"{p}_s{s}"][t]["value"] for s in SEEDS]
            bb = np.mean([R[f"{p}_s{s}"][t]["boot"] for s in SEEDS], 0)
            vals.append(dict(tier=tname, rep=f"{p}_mean5", target=t, value=float(np.mean(v)),
                             seed_sd=float(np.std(v, ddof=1)), lo=pct(bb)[0], hi=pct(bb)[1],
                             **{f"run{C.RUN_NAME[k]}": float(np.mean([R[f"{p}_s{s}"][t]["per_run"][k] for s in SEEDS]))
                                for k in C.RUNS}))
    V = pd.DataFrame(vals)
    tests = []
    comps = [(a, f"{a}_ssl", f"{a}_rand") for a in ARCHS]
    comps += [(a, f"{a}_ssl", b) for a in ARCHS for b in ("snr", "morph8", "pca64", f"{a}_foldinit") if b in R or f"{b}_s0" in R]
    if "cnnns_ssl_s0" in R:
        comps += [("cnn_nostretch", "cnnns_ssl", "cnn_rand"), ("cnn_nostretch", "cnnns_ssl", "cnn_ssl")]
    if "cnn_foldinit_s0" in R:
        comps += [(a, f"{a}_foldinit", f"{a}_rand") for a in ARCHS]
    for arch, a, b in comps:
        for t in C.TARGETS:
            tests.append(dict(tier=tname, arch=arch, a=a, b=b, target=t, **paired(R, a, b, t)))
    T = pd.DataFrame(tests)
    T["p_holm"] = np.nan
    for _, ii in T.groupby(["arch", "a", "b"]).groups.items():
        T.loc[ii, "p_holm"] = P.holm(T.loc[ii, "p_one_sided"].to_numpy())
    T["pass"] = (T.delta > 0) & (T.p_holm < 0.05) & (T.n_runs_pos >= 3)
    # CNN vs TF on the P1 mean (J13)
    def p1mean(p, s):
        return (np.mean([R[f"{p}_s{s}"][t]["value"] for t in C.TARGETS]),
                np.mean([R[f"{p}_s{s}"][t]["boot"] for t in C.TARGETS], 0))
    cmp = []
    for kind in ("ssl", "rand"):
        d = [p1mean(f"cnn_{kind}", s)[0] - p1mean(f"tf_{kind}", s)[0] for s in SEEDS]
        db = np.mean([p1mean(f"cnn_{kind}", s)[1] - p1mean(f"tf_{kind}", s)[1] for s in SEEDS], 0)
        cmp.append(dict(tier=tname, kind=kind, cnn=np.mean([p1mean(f"cnn_{kind}", s)[0] for s in SEEDS]),
                        tf=np.mean([p1mean(f"tf_{kind}", s)[0] for s in SEEDS]), d=float(np.mean(d)),
                        seed_sd=float(np.std(d, ddof=1)), lo=pct(db)[0], hi=pct(db)[1]))
    return V, T, pd.DataFrame(cmp)


def p1(tier=C.TIER, tname="snr20", extra=True):
    reps = model_reps()
    if extra:
        reps += model_reps(("cnnns_ssl", "cnn_foldinit", "tf_foldinit")) + ["pca64", "snr", "morph8"]
        reps = [r for r in reps if os.path.exists(os.path.join(EMB, f"{r}.npy")) or r in ("snr", "morph8")]
    else:
        reps += ["snr", "morph8"]
    t0 = time.time()
    R = p1_tier(tier, tname, reps)
    V, T, Cm = p1_tables(R, tname)
    save(V, f"tab_P1_values_{tname}.csv"); save(T, f"tab_P1_tests_{tname}.csv"); save(Cm, f"tab_P1_cnn_vs_tf_{tname}.csv")
    print(f"P1 {tname} {time.time() - t0:.0f}s")
    print(T[T.b.str.endswith("rand") & (T.a.str.endswith("ssl"))].round(3).to_string(index=False))
    print(Cm.round(3).to_string(index=False))


# ------------------------------------------------------------------ P2
def p2_subsets(m, tier):
    tm = D.tier_mask(m, tier); run = m.run_id.to_numpy()
    return {"date": (tm, "date", False), "run_x0204": (tm & (run != 0), "run_id", False),
            "is0204": (tm, "is0204", False), "drop0204": (tm & (run == 0), "has_dropout", True)}


def p2_task(rep, tier, target, perms):
    m, _ = D.load_all()
    m = m.assign(is0204=(m.run_id == 0).astype(int))
    mask, col, within = p2_subsets(m, tier)[target]
    idx = np.flatnonzero(mask); s = m.iloc[idx]
    X = load_rep(rep, m)[idx]
    y = s[col].astype(str).to_numpy(); g = s.blk30.to_numpy()
    sp = P.splits(g)
    out = {}
    for p in perms:
        yy = y if p < 0 else P.null_labels(y, g, p, within)
        out[p] = (yy, P.lr_oof(X, yy, sp))
    return out


def p2_run(tier, tname, reps, n_perm, targets=("date", "run_x0204", "is0204", "drop0204"), probe="lr"):
    """Incremental: one cache file per (rep, target)."""
    jobs = []
    for r in reps:
        for t in targets:
            f = os.path.join(CACHE, f"p2{probe}_{tname}_{r}_{t}_{n_perm}.pkl")
            if not os.path.exists(f):
                # split permutations into chunks so 16 workers stay busy
                chunks = [list(range(-1, n_perm))[i::6] for i in range(6)] if n_perm > 10 else [list(range(-1, n_perm))]
                for ch in chunks:
                    jobs.append((r, t, ch, f))
    print(f"P2 {tname}: {len(jobs)} jobs", flush=True)
    os.makedirs(CACHE, exist_ok=True)
    fn = p2_task if probe == "lr" else hgb_task
    # process per rep group so caches land incrementally
    byf = {}
    for j in jobs:
        byf.setdefault(j[3], []).append(j)
    files = list(byf)
    for i in range(0, len(files), 8):
        grp = [j for f in files[i:i + 8] for j in byf[f]]
        res = Parallel(n_jobs=N_JOBS, batch_size=1)(delayed(fn)(r, tier, t, ch) for r, t, ch, _ in grp)
        acc = {}
        for (r, t, ch, f), x in zip(grp, res):
            acc.setdefault(f, {}).update(x)
        for f, x in acc.items():
            pd.to_pickle(x, f)
        print(f"  cached {len(acc)} files ({i + len(acc)}/{len(files)})", flush=True)
    out = {}
    for r in reps:
        for t in targets:
            out[r, t] = pd.read_pickle(os.path.join(CACHE, f"p2{probe}_{tname}_{r}_{t}_{n_perm}.pkl"))
    return out


def hgb_task(rep, tier, target, perms):
    m, _ = D.load_all()
    m = m.assign(is0204=(m.run_id == 0).astype(int))
    mask, col, within = p2_subsets(m, tier)[target]
    idx = np.flatnonzero(mask); s = m.iloc[idx]
    X = load_rep(rep, m)[idx]
    y = s[col].astype(str).to_numpy(); g = s.blk30.to_numpy()
    out = {}
    for p in perms:
        yy = y if p < 0 else P.null_labels(y, g, p, within)
        out[p] = (yy, hgb_oof(X, yy, g))
    return out


def p2_tables(OUT, tier, tname, probe="lr"):
    m, _ = D.load_all()
    m = m.assign(is0204=(m.run_id == 0).astype(int))
    S = p2_subsets(m, tier)
    rows, B = [], {}
    for (r, t), x in OUT.items():
        mask = S[t][0]; s = m[mask]
        boots = P.boot_indices(s.blk30.to_numpy(), C.N_BOOT, 0)
        y, pr = x[-1]
        nulls = [fast_metric(*x[p], "cls") for p in x if p >= 0]
        B[r, t] = boot(y, pr, "cls", boots)
        row = dict(tier=tname, probe=probe, rep=r, target=t, n=len(y), K=len(np.unique(y)), skill=fast_metric(y, pr, "cls"),
                   lo=pct(B[r, t])[0], hi=pct(B[r, t])[1], n_null=len(nulls),
                   null_mean=np.mean(nulls) if nulls else np.nan, null_sd=np.std(nulls, ddof=1) if len(nulls) > 1 else np.nan,
                   null_max=np.max(nulls) if nulls else np.nan,
                   p_perm=(1 + np.sum(np.array(nulls) >= fast_metric(y, pr, "cls"))) / (1 + len(nulls)) if nulls else np.nan)
        if t == "date":
            run = s.run_id.to_numpy()
            for k in C.RUNS:
                row[f"run{C.RUN_NAME[k]}"] = fast_metric(y[run == k], pr[run == k], "cls")
        rows.append(row)
    Df = pd.DataFrame(rows)
    prefixes = sorted({r.rsplit("_s", 1)[0] for r, _ in OUT if r[-3:-1] == "_s" and r[-1].isdigit()})
    agg = []
    for p in prefixes:
        for t in Df.target.unique():
            sub = Df[(Df.target == t) & Df.rep.isin([f"{p}_s{s}" for s in SEEDS])]
            if len(sub) < len(SEEDS):
                continue
            bb = np.mean([B[f"{p}_s{s}", t] for s in SEEDS], 0); B[f"{p}_mean5", t] = bb
            sd = dict(tier=tname, probe=probe, rep=f"{p}_mean5", target=t, n=sub.n.iloc[0], K=sub.K.iloc[0],
                      skill=sub.skill.mean(), seed_sd=sub.skill.std(ddof=1), seed_min=sub.skill.min(),
                      seed_max=sub.skill.max(), lo=pct(bb)[0], hi=pct(bb)[1], n_null=sub.n_null.iloc[0],
                      null_mean=sub.null_mean.mean(), null_sd=sub.null_sd.mean(), null_max=sub.null_max.max())
            for k in C.RUNS:
                if f"run{C.RUN_NAME[k]}" in sub:
                    sd[f"run{C.RUN_NAME[k]}"] = sub[f"run{C.RUN_NAME[k]}"].mean()
            agg.append(sd)
    return pd.concat([Df, pd.DataFrame(agg)], ignore_index=True), B


def p2_criteria(T, B, tname="snr20"):
    def sk(rep, t):
        return float(T[(T.rep == rep) & (T.target == t)].skill.iloc[0])
    rows = []
    for a in ARCHS:
        ssl, rnd = f"{a}_ssl_mean5", f"{a}_rand_mean5"
        dc = B[ssl, "is0204"] - B[rnd, "is0204"]; dd = B[ssl, "drop0204"] - B[rnd, "drop0204"]
        crit = [
            ("P2a", "date skill <= 0.03", sk(ssl, "date"), C.P2A_THR, sk(ssl, "date") <= C.P2A_THR,
             (1 + np.sum(B[ssl, "date"] <= C.P2A_THR)) / (1 + C.N_BOOT), pct(B[ssl, "date"])),
            ("P2b", "run skill excl. 0204 <= 0.03", sk(ssl, "run_x0204"), C.P2B_THR, sk(ssl, "run_x0204") <= C.P2B_THR,
             (1 + np.sum(B[ssl, "run_x0204"] <= C.P2B_THR)) / (1 + C.N_BOOT), pct(B[ssl, "run_x0204"])),
            ("P2c", "is-0204 <= rand + 0.05 and <= morph8", sk(ssl, "is0204"),
             min(sk(rnd, "is0204") + C.P2C_MARGIN, sk("morph8", "is0204")),
             (sk(ssl, "is0204") <= sk(rnd, "is0204") + C.P2C_MARGIN) and (sk(ssl, "is0204") <= sk("morph8", "is0204")),
             (1 + np.sum(dc <= C.P2C_MARGIN)) / (1 + C.N_BOOT), pct(B[ssl, "is0204"])),
            ("P2d", "has_dropout in 0204 <= rand + 0.05", sk(ssl, "drop0204"), sk(rnd, "drop0204") + C.P2D_MARGIN,
             sk(ssl, "drop0204") <= sk(rnd, "drop0204") + C.P2D_MARGIN,
             (1 + np.sum(dd <= C.P2D_MARGIN)) / (1 + C.N_BOOT), pct(B[ssl, "drop0204"])),
        ]
        for name, desc, val, thr, ok, p, ci in crit:
            r = dict(tier=tname, arch=a, criterion=name, rule=desc, value=val, threshold=thr, ci_lo=ci[0], ci_hi=ci[1],
                     passes=bool(ok), p_exceed=float(p))
            if name == "P2c":
                r.update(rand=sk(rnd, "is0204"), morph8=sk("morph8", "is0204"), d_vs_rand=sk(ssl, "is0204") - sk(rnd, "is0204"),
                         d_lo=pct(dc)[0], d_hi=pct(dc)[1])
            if name == "P2d":
                r.update(rand=sk(rnd, "drop0204"), d_vs_rand=sk(ssl, "drop0204") - sk(rnd, "drop0204"),
                         d_lo=pct(dd)[0], d_hi=pct(dd)[1])
            rows.append(r)
    Cr = pd.DataFrame(rows)
    Cr["p_exceed_holm"] = np.nan
    for a, ii in Cr.groupby("arch").groups.items():
        Cr.loc[ii, "p_exceed_holm"] = P.holm(Cr.loc[ii, "p_exceed"].to_numpy())
    return Cr


def p2(tier=C.TIER, tname="snr20"):
    t0 = time.time()
    prim = model_reps() + ["morph8"]
    O = p2_run(tier, tname, prim, C.N_PERM)
    sec = [r for r in model_reps(("cnnns_ssl", "cnn_foldinit", "tf_foldinit")) + ["pca64", "snr"]
           if r == "snr" or os.path.exists(os.path.join(EMB, f"{r}.npy"))]
    O.update(p2_run(tier, tname, sec, C.N_PERM_SEC))
    T, B = p2_tables(O, tier, tname)
    save(T, f"tab_P2_{tname}.csv")
    Cr = p2_criteria(T, B, tname); save(Cr, f"tab_P2_criteria_{tname}.csv")
    print(f"P2 {tname} {time.time() - t0:.0f}s")
    print(T[T.rep.str.endswith("mean5") | (T.rep == "morph8")][["rep", "target", "skill", "lo", "hi", "null_mean", "null_max"]]
          .round(3).to_string(index=False))
    print(Cr.round(3).to_string(index=False))


def p2ws():
    """D1 within-space: each fold model embeds ALL events; date and run_x0204 in that single space (no nulls)."""
    reps = [f"ws_f{k}_{a}_s{s}" for k in C.RUNS for a in ARCHS for s in SEEDS]
    O = p2_run(C.TIER, "snr20", reps, 0, targets=("date", "run_x0204"))
    T, _ = p2_tables(O, C.TIER, "snr20_ws")
    T = T[~T.rep.str.endswith("mean5")]
    T["fold"] = T.rep.str.slice(4, 5).astype(int); T["arch"] = T.rep.str.split("_").str[2]
    save(T, "tab_P2_within_space.csv")
    print(T.groupby(["arch", "fold", "target"]).skill.mean().unstack().round(3).to_string())


# ------------------------------------------------------------------ secondary tiers
def tiers():
    for tier, tname in ((30.0, "snr30"), (None, "all")):
        p1(tier, tname, extra=False)
        O = p2_run(tier, tname, model_reps() + ["morph8"], C.N_PERM_SEC)
        T, B = p2_tables(O, tier, tname); save(T, f"tab_P2_{tname}.csv")
        Cr = p2_criteria(T, B, tname); save(Cr, f"tab_P2_criteria_{tname}.csv")
        print(Cr.round(3).to_string(index=False))


def hgb(n_perm=C.N_PERM_SEC, targets=("date", "run_x0204", "is0204", "drop0204")):
    reps = model_reps() + ["morph8"]
    O = p2_run(C.TIER, "snr20", reps, n_perm, targets=targets, probe="hgb")
    T, B = p2_tables(O, C.TIER, "snr20", probe="hgb"); save(T, "tab_P2_hgb_snr20.csv")
    if set(targets) == {"date", "run_x0204", "is0204", "drop0204"}:
        Cr = p2_criteria(T, B, "snr20_hgb"); save(Cr, "tab_P2_hgb_criteria.csv"); print(Cr.round(3).to_string(index=False))


# ------------------------------------------------------------------ P3 / P4
@torch.no_grad()
def p3p4():
    m, W = D.load_all(); S = sel(); dev = "cuda"
    rows3, rows4 = [], []

    def r1(Q, G):
        Qn = Q / np.linalg.norm(Q, axis=1, keepdims=True); Gn = G / np.linalg.norm(G, axis=1, keepdims=True)
        return float(np.mean((Qn @ Gn.T).argmax(1) == np.arange(len(Q))))

    def nn(X):
        X = (X - X.mean(0)) / np.where(X.std(0) > 0, X.std(0), 1)
        return NearestNeighbors(n_neighbors=11).fit(X).kneighbors(X, return_distance=False)[:, 1:]
    for k in C.RUNS:
        idx = D.test_index(m, k); s = m.iloc[idx]
        Wt = torch.tensor(W[idx], device=dev); fs = torch.tensor(s.fs_eff.to_numpy(np.float32), device=dev)
        bg = torch.tensor(D.fold_pool(k)[0], device=dev)
        sign = torch.tensor(np.random.default_rng(0).choice([-1.0, 1.0], len(idx)), device=dev, dtype=torch.float32)
        g = torch.Generator(device=dev); g.manual_seed(0)
        views = {"clean": A.eval_view(Wt, fs), "heldout_aug": A.heldout_view(Wt, fs, sign),
                 "train_aug": A.train_view(Wt, fs, torch.tensor(s.ev_i0.to_numpy(), device=dev),
                                           torch.tensor(s.ev_i1.to_numpy(), device=dev),
                                           torch.tensor(s.sigma_ev.to_numpy(np.float32), device=dev), bg, g)}
        for a in ARCHS:
            G = {"ssl": [], "rand": []}
            for seed in SEEDS:
                t_ = tag(k, a, S[f"{k}_{a}"]["lr"], S[f"{k}_{a}"]["temp"], seed)
                for which, lab in (("final", "ssl"), ("init", "rand")):
                    enc = FE.load_encoder(ckpt(t_, which))
                    Z = {q: FE.embed_views(enc, v) for q, v in views.items()}
                    for q in ("heldout_aug", "train_aug"):
                        rows3.append(dict(fold=k, rep=f"{a}_{lab}", seed=seed, query=q, R_at_1=r1(Z[q], Z["clean"])))
                    G[lab].append(nn(Z["clean"]))
            for lab, L in G.items():
                for i in range(len(SEEDS)):
                    for j in range(i + 1, len(SEEDS)):
                        jac = np.mean([len(set(L[i][n]) & set(L[j][n])) / len(set(L[i][n]) | set(L[j][n]))
                                       for n in range(len(idx))])
                        rows4.append(dict(fold=k, rep=f"{a}_{lab}", pair=f"{i}-{j}", jaccard=jac))
        tr = D.train_index(m, k)
        pca = PCA(64, random_state=0).fit(FE.eval_crops(W[tr], m.fs_eff.to_numpy()[tr]).cpu().numpy())
        Zp = {q: pca.transform(v.cpu().numpy()) for q, v in views.items()}
        for q in ("heldout_aug", "train_aug"):
            rows3.append(dict(fold=k, rep="pca64", seed=0, query=q, R_at_1=r1(Zp[q], Zp["clean"])))
        N = len(idx)
        rows4.append(dict(fold=k, rep="random_graph_expectation", pair="-", jaccard=(100 / (N - 1)) / (20 - 100 / (N - 1))))
    T3 = pd.DataFrame(rows3); T4 = pd.DataFrame(rows4)
    save(T3, "tab_P3.csv"); save(T4, "tab_P4.csv")
    print(T3.groupby(["rep", "query"]).R_at_1.agg(["mean", "std"]).round(3).to_string())
    print(T4.groupby(["rep"]).jaccard.agg(["mean", "std"]).round(3).to_string())


# ------------------------------------------------------------------ verdict
def verdict():
    T = pd.read_csv(os.path.join(C.RES, "tab_P1_tests_snr20.csv"))
    Cr = pd.read_csv(os.path.join(C.RES, "tab_P2_criteria_snr20.csv"))
    Cm = pd.read_csv(os.path.join(C.RES, "tab_P1_cnn_vs_tf_snr20.csv"))
    res = {}
    for a in ARCHS:
        t = T[(T.a == f"{a}_ssl") & (T.b == f"{a}_rand")]
        n_pass = int(t["pass"].sum()); P1 = n_pass >= 3
        c = Cr[Cr.arch == a]; P2 = bool(c.passes.all())
        v = "Positive" if (P1 and P2) else ("Learns but leaks" if P1 else "No evidence of learning")
        res[a] = dict(P1_targets_passed=n_pass, P1_pass=P1,
                      P1_target_pass={r.target: bool(r["pass"]) for _, r in t.iterrows()},
                      P2={r.criterion: bool(r.passes) for _, r in c.iterrows()}, P2_pass=P2, verdict=v)
    r = Cm[Cm.kind == "ssl"].iloc[0]
    res["cnn_vs_tf"] = dict(d=float(r.d), lo=float(r.lo), hi=float(r.hi),
                            outcome=("CNN" if r.lo > 0 else "transformer" if r.hi < 0 else "tie"))
    json.dump(res, open(os.path.join(C.RES, "results_verdict.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    t0 = time.time()
    for part in sys.argv[1:]:
        {"sanity": sanity, "p1": p1, "p2": p2, "p2ws": p2ws, "tiers": tiers, "p3p4": p3p4, "hgb": hgb,
         "verdict": verdict}[part]()
        print(f"[{part}] cumulative {time.time() - t0:.0f}s", flush=True)
