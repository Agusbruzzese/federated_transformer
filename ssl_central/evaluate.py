"""Stage 3: final evaluation P1-P5 on frozen embeddings (run only after select.py has frozen the choice).

    python -m ssl_central.evaluate embed        # export embeddings of every event for every model/baseline
    python -m ssl_central.evaluate p1 p2 p3 p4 p5 verdict
Outputs: <OUT>/emb/*.npy, research/model/F_central/tab_*.csv, results.json.
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
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.neighbors import NearestNeighbors

from . import augment as A
from . import config as C
from . import data as D
from . import embed as E
from . import probes as P
from .train import cell_tag

EMB = os.path.join(C.OUT, "emb")
N_JOBS = 16
ARCHS = ("cnn", "tf")


def sel():
    return json.load(open(os.path.join(C.OUT, "selection.json")))


def rep_names():
    R = [f"{a}_ssl_s{s}" for a in ARCHS for s in C.SEEDS] + [f"{a}_rand_s{s}" for a in ARCHS for s in C.SEEDS]
    return R + ["morph8", "morph21", "pca64", "snr"]


def load_rep(name, m):
    if name == "morph8":
        return m[[c for c in m.columns if c.startswith("m8_")]].to_numpy(float)
    if name == "morph21":
        return m[[c for c in m.columns if c.startswith("m_")]].to_numpy(float)
    if name == "snr":
        return np.log10(m.snr.to_numpy(float))[:, None]
    return np.load(os.path.join(EMB, f"{name}.npy"))


# ------------------------------------------------------------------ embeddings
def do_embed(m, W):
    os.makedirs(EMB, exist_ok=True)
    S = sel()
    crops = E.eval_crops(W, m.fs_eff.to_numpy())
    for a in ARCHS:
        for s in C.SEEDS:
            tag = cell_tag(a, S[a]["lr"], S[a]["temp"], s)
            for which, lab in (("final", "ssl"), ("init", "rand")):
                X = E.embed_views(E.load_encoder(E.ckpt_path(tag, which)), crops)
                np.save(os.path.join(EMB, f"{a}_{lab}_s{s}.npy"), X.astype(np.float32))
    tr = D.split_index(m, "train")
    Xc = crops.cpu().numpy()
    pca = PCA(64, random_state=0).fit(Xc[tr])
    np.save(os.path.join(EMB, "pca64.npy"), pca.transform(Xc).astype(np.float32))
    json.dump(dict(pca_explained=float(pca.explained_variance_ratio_.sum())), open(os.path.join(EMB, "pca_info.json"), "w"))
    m[["event_id", "gi", "split", "snr"]].to_parquet(os.path.join(EMB, "index.parquet"))
    print("embeddings written; PCA-64 explained variance", pca.explained_variance_ratio_.sum())


# ------------------------------------------------------------------ P1
def p1(m, tier=C.TIER, prefix="y", tag="snr20"):
    idx = D.split_index(m, "holdout", tier)
    s = m.iloc[idx]; g = s.blk30.to_numpy()
    boots = P.boot_indices(g)
    reps = rep_names()
    OOF, PT, BT = {}, {}, {}
    for r in reps:
        X = load_rep(r, m)[idx]
        for t in C.TARGETS:
            k = C.TARGET_KIND[t]; y = s[f"{prefix}_{t}"].to_numpy()
            pr = P.knn_oof(X, y, g, k)
            PT[r, t] = P.metric(y, pr, k)
            BT[r, t] = P.boot_metric(y, pr, k, boots)
    rows = []
    for r in reps:
        for t in C.TARGETS:
            rows.append(dict(tier=tag, targets=prefix, rep=r, target=t, value=PT[r, t],
                             lo=np.percentile(BT[r, t], 2.5), hi=np.percentile(BT[r, t], 97.5)))
    per = pd.DataFrame(rows)
    # seed-aggregated model rows and tests
    tests = []
    for a in ARCHS:
        for kind in ("ssl", "rand"):
            for t in C.TARGETS:
                v = [PT[f"{a}_{kind}_s{s_}", t] for s_ in C.SEEDS]
                bb = np.mean([BT[f"{a}_{kind}_s{s_}", t] for s_ in C.SEEDS], 0)
                rows.append(dict(tier=tag, targets=prefix, rep=f"{a}_{kind}_mean3", target=t, value=np.mean(v),
                                 seed_sd=np.std(v, ddof=1), lo=np.percentile(bb, 2.5), hi=np.percentile(bb, 97.5)))
        for comp in ("rand", "morph8", "snr", "pca64"):
            for t in C.TARGETS:
                ssl_pt = np.array([PT[f"{a}_ssl_s{s_}", t] for s_ in C.SEEDS])
                ssl_b = np.stack([BT[f"{a}_ssl_s{s_}", t] for s_ in C.SEEDS])
                if comp == "rand":   # paired by seed index
                    c_pt = np.array([PT[f"{a}_rand_s{s_}", t] for s_ in C.SEEDS])
                    c_b = np.stack([BT[f"{a}_rand_s{s_}", t] for s_ in C.SEEDS])
                else:
                    c_pt = np.full(3, PT[comp, t]); c_b = np.stack([BT[comp, t]] * 3)
                d_pt = float(np.mean(ssl_pt - c_pt)); d_b = np.mean(ssl_b - c_b, 0)
                p = (1 + np.sum(d_b <= 0)) / (1 + len(d_b))
                tests.append(dict(tier=tag, targets=prefix, arch=a, vs=comp, target=t, delta=d_pt,
                                  delta_seed_sd=float(np.std(ssl_pt - c_pt, ddof=1)),
                                  lo=np.percentile(d_b, 2.5), hi=np.percentile(d_b, 97.5), p_one_sided=p))
    T = pd.DataFrame(tests)
    T["p_holm"] = np.nan
    for (a, comp), ii in T.groupby(["arch", "vs"]).groups.items():
        T.loc[ii, "p_holm"] = P.holm(T.loc[ii, "p_one_sided"].to_numpy())
    info = dict(tier=tag, n_events=len(idx), n_blocks=int(len(np.unique(g))),
                events_per_block=pd.Series(g).value_counts().describe().round(2).to_dict())
    return pd.DataFrame(rows), T, info


def p1_all():
    m, W, *_ = D.load_all()
    R, T, I = [], [], []
    for tier, tag, prefix in ((20.0, "snr20", "y"), (30.0, "snr30", "y"), (None, "all", "y"), (20.0, "snr20", "y21")):
        t0 = time.time()
        r, t, i = p1(m, tier, prefix, tag); R.append(r); T.append(t); I.append(i | dict(targets=prefix))
        print(f"P1 {tag}/{prefix} done in {time.time() - t0:.0f}s", flush=True)
    pd.concat(R).to_csv(os.path.join(C.RES, "tab_P1_values.csv"), index=False)
    pd.concat(T).to_csv(os.path.join(C.RES, "tab_P1_tests.csv"), index=False)
    json.dump(I, open(os.path.join(C.RES, "P1_info.json"), "w"), indent=1, default=str)


# ------------------------------------------------------------------ P2
P2_TARGETS = (("date", "blk30", False), ("run_id", "blk30", False), ("hbin", "blk30", False),
              ("has_dropout", "date", True))


def _p2_task(X, y, g, within, perm_seed, probe):
    if perm_seed >= 0:
        y = P.permute_labels(y, g, np.random.default_rng(1000 + perm_seed), within=within)
    pr = P.hgb_oof(X, y, g) if probe == "hgb" else P.knn_oof(X, y, g, "cls")
    return pr


def p2(m, tier=C.TIER, tag="snr20", subset="all_runs", n_null=C.N_NULL, reps=None, probe="hgb"):
    mask = D.tier_mask(m, tier)
    if subset == "no0204":
        mask = mask & (m.run_id != 0).to_numpy()
    idx = np.flatnonzero(mask)
    s = m.iloc[idx]
    targets = [(t, g, w) for t, g, w in P2_TARGETS]
    if subset == "all_runs":
        s = s.assign(is0204=(s.run_id == 0).astype(int))
        targets.append(("is0204", "blk30", False))
    else:
        targets = [x for x in targets if x[0] in ("date", "run_id", "hbin")]
    reps = reps or rep_names()
    jobs, keys = [], []
    for r in reps:
        X = load_rep(r, m)[idx]
        for t, gcol, within in targets:
            y = s[t].astype(str).to_numpy(); g = s[gcol].to_numpy()
            for ps in range(-1, n_null):
                jobs.append(delayed(_p2_task)(X, y, g, within, ps, probe)); keys.append((r, t, ps))
    cache = os.path.join(C.OUT, "p2cache", f"{tag}_{subset}_{probe}_{n_null}_{len(reps)}.pkl")
    if os.path.exists(cache):
        out = pd.read_pickle(cache)
    else:
        t0 = time.time()
        out = Parallel(n_jobs=N_JOBS, batch_size=1)(jobs)
        print(f"P2 {tag}/{subset}/{probe}: {len(jobs)} CV fits in {time.time() - t0:.0f}s", flush=True)
        os.makedirs(os.path.dirname(cache), exist_ok=True); pd.to_pickle(out, cache)
    boots = P.boot_indices(s.blk30.to_numpy())
    PRED = dict(zip(keys, out))
    rows = []
    for r in reps:
        for t, gcol, within in targets:
            y = s[t].astype(str).to_numpy()
            pr = PRED[r, t, -1]
            nulls = [P.skill(P.permute_labels(y, s[gcol].to_numpy(), np.random.default_rng(1000 + ps), within), PRED[r, t, ps])
                     for ps in range(n_null)]
            row = dict(tier=tag, subset=subset, probe=probe, rep=r, target=t, n=len(idx), K=len(np.unique(y)),
                       skill=P.skill(y, pr), null_mean=np.mean(nulls) if nulls else np.nan,
                       null_sd=np.std(nulls, ddof=1) if len(nulls) > 1 else np.nan,
                       null_max=np.max(nulls) if nulls else np.nan)
            if t == "run_id":
                rec = {k: float(np.mean(pr[y == k] == k)) for k in np.unique(y)}
                row["recall_0204"] = rec.get("0", np.nan)
            row["_boot"] = P.boot_metric(y, pr, "cls", boots)
            rows.append(row)
    Df = pd.DataFrame(rows)
    agg = []
    for a in ARCHS:
        for kind in ("ssl", "rand"):
            for t in Df.target.unique():
                sub = Df[(Df.target == t) & Df.rep.isin([f"{a}_{kind}_s{x}" for x in C.SEEDS])]
                if len(sub) < 3:
                    continue
                bb = np.mean(np.stack(sub._boot.to_list()), 0)
                agg.append(dict(tier=tag, subset=subset, probe=probe, rep=f"{a}_{kind}_mean3", target=t,
                                n=sub.n.iloc[0], K=sub.K.iloc[0], skill=sub.skill.mean(), seed_sd=sub.skill.std(ddof=1),
                                null_mean=sub.null_mean.mean(), null_sd=sub.null_sd.mean(), null_max=sub.null_max.max(),
                                recall_0204=sub.recall_0204.mean() if "recall_0204" in sub else np.nan,
                                lo=np.percentile(bb, 2.5), hi=np.percentile(bb, 97.5)))
    Df["lo"] = [np.percentile(b, 2.5) for b in Df._boot]; Df["hi"] = [np.percentile(b, 97.5) for b in Df._boot]
    return pd.concat([Df.drop(columns="_boot"), pd.DataFrame(agg)], ignore_index=True)


def p2_all(which):
    m, *_ = D.load_all()
    parts = []
    if "primary" in which:
        parts += [(20.0, "snr20", "all_runs", C.N_NULL, "hgb"), (20.0, "snr20", "no0204", C.N_NULL, "hgb"),
                  (20.0, "snr20", "all_runs", 0, "knn")]
    if "tiers" in which:
        # compute budget (orchestrator, 2026-09-30): secondary tiers use only the representations that enter the
        # P2 criterion (the 6 SSL models + morph8) and fewer permutations (5 at SNR>=30, 2 at all-SNR)
        tr = [f"{a}_ssl_s{s_}" for a in ARCHS for s_ in C.SEEDS] + ["morph8"]
        parts += [(30.0, "snr30", "all_runs", 5, "hgb", tr), (None, "all", "all_runs", 2, "hgb", tr)]
    for part in parts:
        tier, tag, sub, nn_, probe = part[:5]
        _save_p2(p2(m, tier, tag, sub, n_null=nn_, probe=probe, reps=part[5] if len(part) > 5 else None))


def _save_p2(D_):
    f = os.path.join(C.RES, "tab_P2.csv")
    if os.path.exists(f):
        old = pd.read_csv(f)
        key = ["tier", "subset", "probe"]
        done = set(map(tuple, D_[key].drop_duplicates().to_numpy()))
        old = old[[tuple(x) not in done for x in old[key].to_numpy()]]
        D_ = pd.concat([old, D_], ignore_index=True)
    D_.to_csv(f, index=False)


# ------------------------------------------------------------------ P3-P5
@torch.no_grad()
def p3():
    m, W, *_ = D.load_all()
    S = sel(); dev = "cuda"
    idx = D.split_index(m, "holdout"); s = m.iloc[idx]
    Wt = torch.tensor(W[idx], device=dev); fs = torch.tensor(s.fs_eff.to_numpy(np.float32), device=dev)
    bg = torch.tensor(np.load(os.path.join(C.OUT, "bgpool.npy")), device=dev)
    sign = torch.tensor(np.random.default_rng(0).choice([-1.0, 1.0], len(idx)), device=dev, dtype=torch.float32)
    g = torch.Generator(device=dev); g.manual_seed(0)
    views = {"clean": A.eval_view(Wt, fs), "heldout_aug": A.heldout_view(Wt, fs, sign),
             "train_aug": A.train_view(Wt, fs, torch.tensor(s.ev_i0.to_numpy(), device=dev),
                                       torch.tensor(s.ev_i1.to_numpy(), device=dev),
                                       torch.tensor(s.sigma_ev.to_numpy(np.float32), device=dev), bg, g)}
    tr = D.split_index(m, "train")
    pca = PCA(64, random_state=0).fit(E.eval_crops(W[tr], m.fs_eff.to_numpy()[tr]).cpu().numpy())

    def r1(Q, G):
        Qn = Q / np.linalg.norm(Q, axis=1, keepdims=True); Gn = G / np.linalg.norm(G, axis=1, keepdims=True)
        return float(np.mean((Qn @ Gn.T).argmax(1) == np.arange(len(Q))))
    rows = []
    for a in ARCHS:
        for s_ in C.SEEDS:
            tag = cell_tag(a, S[a]["lr"], S[a]["temp"], s_)
            for which, lab in (("final", "ssl"), ("init", "rand")):
                enc = E.load_encoder(E.ckpt_path(tag, which))
                Z = {k: E.embed_views(enc, v) for k, v in views.items()}
                for q in ("heldout_aug", "train_aug"):
                    rows.append(dict(rep=f"{a}_{lab}", seed=s_, query=q, R_at_1=r1(Z[q], Z["clean"])))
    Z = {k: pca.transform(v.cpu().numpy()) for k, v in views.items()}
    for q in ("heldout_aug", "train_aug"):
        rows.append(dict(rep="pca64", seed=0, query=q, R_at_1=r1(Z[q], Z["clean"])))
        Zr = {k: v.cpu().numpy() for k, v in views.items()}
        rows.append(dict(rep="raw_crop", seed=0, query=q, R_at_1=r1(Zr[q], Zr["clean"])))
    T = pd.DataFrame(rows); T.to_csv(os.path.join(C.RES, "tab_P3.csv"), index=False)
    print(T.groupby(["rep", "query"]).R_at_1.agg(["mean", "std"]).round(3).to_string())


def _z(X):
    return (X - X.mean(0)) / np.where(X.std(0) > 0, X.std(0), 1)


def p4():
    m, *_ = D.load_all()
    idx = D.split_index(m, "holdout")
    rows = []

    def nn(X):
        return NearestNeighbors(n_neighbors=11).fit(X).kneighbors(X, return_distance=False)[:, 1:]
    for a in ARCHS:
        for kind in ("ssl", "rand"):
            G = [nn(_z(load_rep(f"{a}_{kind}_s{s_}", m)[idx])) for s_ in C.SEEDS]
            for i in range(3):
                for j in range(i + 1, 3):
                    jac = np.mean([len(set(G[i][n]) & set(G[j][n])) / len(set(G[i][n]) | set(G[j][n])) for n in range(len(idx))])
                    rows.append(dict(rep=f"{a}_{kind}", pair=f"{i}-{j}", jaccard=jac))
    k = 10; N = len(idx)
    rows.append(dict(rep="random_graph_expectation", pair="-", jaccard=(k * k / (N - 1)) / (2 * k - k * k / (N - 1))))
    T = pd.DataFrame(rows); T.to_csv(os.path.join(C.RES, "tab_P4.csv"), index=False)
    print(T.groupby("rep").jaccard.agg(["mean", "std"]).round(3).to_string())


def _p5_one(X, k, g, n_boot=20):
    km = KMeans(k, n_init=10, random_state=0).fit(X)
    sil = silhouette_score(X, km.labels_, random_state=0, sample_size=min(len(X), 5000))
    boots = P.boot_indices(g, n_boot, seed=1)
    aris = []
    for b in boots:
        u = np.unique(b)
        kb = KMeans(k, n_init=10, random_state=0).fit(X[b])
        aris.append(adjusted_rand_score(km.labels_[u], kb.predict(X[u])))
    return sil, float(np.mean(aris))


def p5():
    m, *_ = D.load_all()
    idx = D.split_index(m, "holdout"); g = m.blk30.to_numpy()[idx]
    jobs, keys = [], []
    for a in ARCHS:
        for kind in ("ssl", "rand"):
            for s_ in C.SEEDS:
                X = _z(load_rep(f"{a}_{kind}_s{s_}", m)[idx])
                for k in range(3, 11):
                    jobs.append(delayed(_p5_one)(X, k, g)); keys.append((f"{a}_{kind}", s_, k))
    for r in ("pca64", "morph8"):
        X = _z(load_rep(r, m)[idx])
        for k in range(3, 11):
            jobs.append(delayed(_p5_one)(X, k, g)); keys.append((r, 0, k))
    out = Parallel(n_jobs=N_JOBS)(jobs)
    T = pd.DataFrame([dict(rep=r, seed=s_, k=k, silhouette=a_, boot_ari=b_) for (r, s_, k), (a_, b_) in zip(keys, out)])
    T.to_csv(os.path.join(C.RES, "tab_P5.csv"), index=False)
    print(T.groupby(["rep", "k"])[["silhouette", "boot_ari"]].mean().unstack(0).round(3).to_string())


# ------------------------------------------------------------------ verdict
def verdict():
    T = pd.read_csv(os.path.join(C.RES, "tab_P1_tests.csv"))
    V = pd.read_csv(os.path.join(C.RES, "tab_P1_values.csv"))
    L = pd.read_csv(os.path.join(C.RES, "tab_P2.csv"))
    res = {}
    for tier in ("snr20", "snr30", "all"):
        for a in ARCHS:
            t = T[(T.tier == tier) & (T.targets == "y") & (T.arch == a) & (T.vs == "rand")]
            n_pass = int(((t.delta > 0) & (t.p_holm < 0.05)).sum())
            P1 = n_pass >= 3
            l = L[(L.tier == tier) & (L.subset == "all_runs") & (L.probe == "hgb")]
            if len(l) == 0:
                res[f"{tier}_{a}"] = dict(P1_targets_passed=n_pass, P1=P1); continue
            sk = lambda rep, tg: float(l[(l.rep == rep) & (l.target == tg)].skill.iloc[0])
            date_ok = sk(f"{a}_ssl_mean3", "date") <= sk("morph8", "date") + 0.05
            run_ok = sk(f"{a}_ssl_mean3", "run_id") <= sk("morph8", "run_id") + 0.05
            drop_ok = sk(f"{a}_ssl_mean3", "has_dropout") <= 0.2
            P2 = date_ok and run_ok and drop_ok
            v = "Positive" if (P1 and P2) else ("Learns structure but leaks" if P1 else "No evidence of learning")
            vv = V[(V.tier == tier) & (V.targets == "y") & (V.rep == f"{a}_ssl_mean3")]
            res[f"{tier}_{a}"] = dict(P1_targets_passed=n_pass, P1=P1, P2_date=date_ok, P2_run=run_ok,
                                      P2_dropout=drop_ok, P2=P2, verdict=v, P1_mean=float(vv.value.mean()),
                                      P2_date_skill=sk(f"{a}_ssl_mean3", "date"),
                                      P2_run_skill=sk(f"{a}_ssl_mean3", "run_id"),
                                      P2_dropout_skill=sk(f"{a}_ssl_mean3", "has_dropout"),
                                      morph_date=sk("morph8", "date"), morph_run=sk("morph8", "run_id"))
    json.dump(res, open(os.path.join(C.RES, "results_verdict.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    t0 = time.time()
    for part in sys.argv[1:]:
        if part == "embed":
            m, W, *_ = D.load_all(); do_embed(m, W)
        elif part == "p1":
            p1_all()
        elif part in ("p2", "p2primary"):
            p2_all(("primary",))
        elif part == "p2tiers":
            p2_all(("tiers",))
        elif part == "p3":
            p3()
        elif part == "p4":
            p4()
        elif part == "p5":
            p5()
        elif part == "verdict":
            verdict()
        print(f"[{part}] cumulative {time.time() - t0:.0f}s", flush=True)
