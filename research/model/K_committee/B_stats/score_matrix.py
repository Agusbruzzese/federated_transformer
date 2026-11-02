"""Committee B: per-(model, seed, day) score matrix on the 5 held-out days (4 M4/M5 test days + val day).

Purpose: measure variance components of the CENTRAL model itself (seed, day, seed x day) and of
other arms, to set equivalence margins and drive the power simulation. Nothing here is a new claim
about the models: all 5 days have been seen before (exploratory use only).

    python research/model/K_committee/B_stats/score_matrix.py

Fixed probe protocol (no validation-chosen scaling): StandardScaler + {kNN k=15, logistic C=1 / RidgeCV},
fitted on the M5 training days, scored PER DAY on each held-out day.
Outputs (this folder): scores_day.csv (probe + label-free metrics, per model x day),
pairs_day.csv (CKA and kNN-overlap for every model pair x day), leak.csv (day-identity skill per model).
"""
import os, sys, json, time
import numpy as np, pandas as pd, torch
from joblib import Parallel, delayed

ROOT = os.getcwd(); sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "research/model/M5_long")); sys.path.insert(0, os.path.join(ROOT, "research/model/M4_methods"))
import m4, m5
from ssl_central import augment as A
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor, NearestNeighbors
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold, cross_val_predict

HERE = os.path.dirname(os.path.abspath(__file__))
KB = os.path.join(ROOT, "data", "k_committee", "B_stats")
EVAL_DAYS = m4.TEST_DAYS + [m4.VAL_DAY]
DEV = "cuda" if torch.cuda.is_available() else "cpu"
PROBES = [(k, nm) for k in ("knn", "lin") for nm in ("polarity", "lobes", "duration")]


def ckpt(tag):
    meth, s, which, src = tag
    if src == "m5": return os.path.join(ROOT, "data", "m5", "ckpt", f"{meth}_s{s}", f"{which}.pt")
    return os.path.join(KB, src, "ckpt", f"{meth}_s{s}", f"{which}.pt")


@torch.no_grad()
def embed(path, meth, X):
    enc, _ = m5.build(meth); enc.load_state_dict(torch.load(path, map_location=DEV)); enc = enc.to(DEV).eval()
    out = [enc(X[i:i + 2048]).float().cpu().numpy() for i in range(0, len(X), 2048)]
    return np.concatenate(out)


def models():
    """name -> (arm, seed, method, which, src)."""
    M = {}
    for s in range(5): M[f"central_s{s}"] = ("central", s, "simclr_tf", "final", "m5")
    for s in (5, 6, 7): M[f"central_s{s}"] = ("central", s, "simclr_tf", "final", "full")
    for s in range(5):
        M[f"init_s{s}"] = ("tf_init", s, "simclr_tf", "init", "m5")
        M[f"short_s{s}"] = ("tf_short1k", s, "simclr_tf", "final", "short")
        M[f"cnn_s{s}"] = ("simclr_cnn", s, "simclr_cnn", "final", "m5")
        M[f"vicreg_s{s}"] = ("vicreg_tf", s, "vicreg_tf", "final", "m5")
    return M


def skill(y, p): return m4.skill(y, p)


def day_scores(X, m, tg, tr, ev_day):
    """Probe scores per day for one representation."""
    out = []
    for k, nm in PROBES:
        if nm == "duration": est = KNeighborsRegressor(15) if k == "knn" else RidgeCV(alphas=np.logspace(-3, 3, 13))
        else: est = KNeighborsClassifier(15) if k == "knn" else LogisticRegression(C=1.0, max_iter=20000, tol=1e-8)
        X = X.astype(np.float64)                                       # converged, always-scaled linear probe (cf. A_model finding 1)
        e = make_pipeline(StandardScaler(), est).fit(X[tr], tg[nm][tr])
        for d, ix in ev_day.items():
            out.append(dict(metric=f"{k}_{nm}", day=d, value=float(m4.metric(nm, tg[nm][ix], e.predict(X[ix])))))
    return out


def label_free(X, Z1, Z2, ev_day, ev_pos):
    out = []
    for d, ix in ev_day.items():
        x = X[ix]; s = np.linalg.svd(x, compute_uv=False); p = s / s.sum() + 1e-12
        out.append(dict(metric="rankme", day=d, value=float(np.exp(-(p * np.log(p)).sum()))))
        z = x / np.linalg.norm(x, axis=1, keepdims=True); D2 = ((z[:, None] - z[None]) ** 2).sum(-1)
        iu = np.triu_indices(len(z), 1)
        out.append(dict(metric="uniformity", day=d, value=float(np.log(np.mean(np.exp(-2 * D2[iu]))))))
        j = ev_pos[d]; a = Z1[j] / np.linalg.norm(Z1[j], axis=1, keepdims=True); b = Z2[j] / np.linalg.norm(Z2[j], axis=1, keepdims=True)
        out.append(dict(metric="alignment", day=d, value=float(((a - b) ** 2).sum(1).mean())))
    return out


def leak(X, m, ev):
    """Day identity among held-out days: balanced-accuracy skill, logistic, GroupKFold(5) by 30-min block."""
    y = m.date.astype(str).to_numpy()[ev]; g = m.blk.to_numpy()[ev]
    p = cross_val_predict(make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=3000)), X[ev], y, groups=g, cv=GroupKFold(5))
    return float(skill(y, p))


def lin_cka(a, b):
    a = a - a.mean(0); b = b - b.mean(0)
    return float(np.linalg.norm(a.T @ b) ** 2 / (np.linalg.norm(a.T @ a) * np.linalg.norm(b.T @ b)))


def main():
    t0 = time.time()
    m, W = m4.load(); tg = m4.targets(m); tr = (m.role == "train").to_numpy()
    dstr = m.date.astype(str).to_numpy()
    ev = np.isin(dstr, EVAL_DAYS); ev_idx = np.flatnonzero(ev)
    ev_day = {d: np.flatnonzero(dstr == d) for d in EVAL_DAYS}
    ev_pos = {d: np.flatnonzero(dstr[ev_idx] == d) for d in EVAL_DAYS}
    crops = m5.crops()
    # two fixed augmented views of the held-out events (alignment)
    bg, _ = m4.bgpool(); bg = torch.tensor(bg, device=DEV); s = m.iloc[ev_idx]
    args = (torch.tensor(W[ev_idx], device=DEV), torch.tensor(s.fs_eff.to_numpy(np.float32), device=DEV),
            torch.tensor(s.ev_i0.to_numpy(), device=DEV), torch.tensor(s.ev_i1.to_numpy(), device=DEV),
            torch.tensor(s.sigma_ev.to_numpy(np.float32), device=DEV), bg)
    g = torch.Generator(device=DEV); g.manual_seed(12345)
    V1 = A.train_view(*args, g); V2 = A.train_view(*args, g)

    E, AUG, META = {}, {}, {}
    for name, (arm, sd, meth, which, src) in models().items():
        p = ckpt((meth, sd, which, src))
        if not os.path.exists(p): print("MISSING", name, p); continue
        E[name] = embed(p, meth, crops); AUG[name] = (embed(p, meth, V1), embed(p, meth, V2)); META[name] = (arm, sd)
    # planted variants built from the 8 central seeds (fixed RNG)
    rng = np.random.default_rng(2026)
    cen = [n for n in E if n.startswith("central_")]
    days_all = np.unique(dstr)
    for n in cen:
        X = E[n]; sdv = X[tr].std(0)
        sd = META[n][1]
        for tau in (0.1, 0.25, 0.5, 1.0):                                    # isotropic noise, relative to per-dim SD
            nz = rng.standard_normal(X.shape) * sdv * tau
            E[f"noise{tau}_{n}"] = X + nz; META[f"noise{tau}_{n}"] = (f"noise{tau}", sd)
            AUG[f"noise{tau}_{n}"] = (AUG[n][0] + rng.standard_normal(AUG[n][0].shape) * sdv * tau,
                                      AUG[n][1] + rng.standard_normal(AUG[n][1].shape) * sdv * tau)
        for kap in (0.005, 0.01, 0.02, 0.05):                                    # planted day-identity leak: per-day offset
            off = {d: rng.standard_normal(X.shape[1]) for d in days_all}
            off = {d: v / np.linalg.norm(v) * kap * np.linalg.norm(sdv) for d, v in off.items()}
            O = np.stack([off[d] for d in dstr])
            E[f"leak{kap}_{n}"] = X + O; META[f"leak{kap}_{n}"] = (f"leak{kap}", sd)
            Oe = O[ev_idx]; AUG[f"leak{kap}_{n}"] = (AUG[n][0] + Oe, AUG[n][1] + Oe)
    np.savez_compressed(os.path.join(KB, "emb.npz"), **E)
    print(f"{len(E)} representations embedded, {time.time() - t0:.0f}s", flush=True)

    def one(name):
        X = E[name]; rows = day_scores(X, m, tg, tr, ev_day) + label_free(X, AUG[name][0], AUG[name][1], ev_day, ev_pos)
        arm, sd = META[name]
        return [dict(model=name, arm=arm, seed=sd, **r) for r in rows], dict(model=name, arm=arm, seed=sd, day_id_skill=leak(X, m, ev))
    res = Parallel(n_jobs=12)(delayed(one)(n) for n in E)
    pd.DataFrame([r for a, _ in res for r in a]).to_csv(os.path.join(HERE, "scores_day.csv"), index=False)
    pd.DataFrame([b for _, b in res]).to_csv(os.path.join(HERE, "leak.csv"), index=False)
    print(f"probes done {time.time() - t0:.0f}s", flush=True)

    # pairwise CKA / kNN-overlap (k=15, within day) between every representation and every central seed
    names = list(E); rows = []
    nnb = {(n, d): NearestNeighbors(n_neighbors=16).fit(E[n][ix]).kneighbors(E[n][ix], return_distance=False)[:, 1:]
           for n in names for d, ix in ev_day.items()}
    for a in names:
        for b in cen:
            if a == b: continue
            for d, ix in ev_day.items():
                A_, B_ = nnb[(a, d)], nnb[(b, d)]
                ov = np.mean([len(np.intersect1d(A_[i], B_[i])) / 15 for i in range(len(ix))])
                rows.append(dict(a=a, b=b, day=d, cka=lin_cka(E[a][ix], E[b][ix]), knn_overlap=float(ov)))
    pd.DataFrame(rows).to_csv(os.path.join(HERE, "pairs_day.csv"), index=False)
    print(f"done {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
