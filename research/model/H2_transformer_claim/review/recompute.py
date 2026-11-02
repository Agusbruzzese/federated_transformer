"""Independent recomputation for the H2 review. Own kNN, own balanced-accuracy skill, own R^2, own block folds.
Does not import h2.py. Writes review/recompute_*.csv."""
import os, sys, numpy as np, pandas as pd, torch
sys.path.insert(0, os.getcwd())
from ssl_central import data as D, embed as E
from ssl_central_v3 import train as T3, config as C3
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = {0: "0204", 1: "0212", 2: "0218", 3: "0224"}

m, W, _, _ = D.load_all(); m = m.reset_index(drop=True)
keep = (m.snr >= 20).to_numpy(); m = m[keep].reset_index(drop=True); W = W[keep]
blk = pd.to_datetime(m.wall_time).dt.floor("30min").astype("int64").to_numpy()
amp = 10 ** m.m8_log_amp.to_numpy()
Y = dict(polarity=(m.m8_polarity.to_numpy() > 0).astype(int),
         lobes=np.clip(m.m8_n_lobes.to_numpy(), None, 4).astype(int),
         duration=np.log10(np.maximum(m.m8_dur10.to_numpy(), 0.05)),
         magnitude=(amp >= 5000).astype(int))
crops = E.eval_crops(W, m.fs_eff.to_numpy())
raw = crops.float().cpu().numpy()
run = m.run_id.to_numpy()

def folds_sorted(g, n=5):
    """Contiguous-in-time block folds (different from sklearn GroupKFold's size balancing)."""
    ub = np.unique(g); a = {b: i * n // len(ub) for i, b in enumerate(ub)}
    return np.array([a[b] for b in g])

def folds_random(g, n=5, seed=11):
    ub = np.unique(g); r = np.random.default_rng(seed).permutation(len(ub))
    a = {b: r[i] % n for i, b in enumerate(ub)}
    return np.array([a[b] for b in g])

def zs(tr, te):
    mu, sd = tr.mean(0), tr.std(0) + 1e-8
    return (tr - mu) / sd, (te - mu) / sd

def knn_pred(Xtr, ytr, Xte, k=15, reg=False, z=True):
    if z: Xtr, Xte = zs(Xtr, Xte)
    Xtr_t = torch.tensor(Xtr, dtype=torch.float64, device="cuda"); Xte_t = torch.tensor(Xte, dtype=torch.float64, device="cuda")
    d = torch.cdist(Xte_t, Xtr_t); idx = d.topk(k, largest=False).indices.cpu().numpy()
    nb = ytr[idx]
    if reg: return nb.mean(1)
    cls = np.unique(ytr); cnt = np.stack([(nb == c).sum(1) for c in cls], 1)
    return cls[cnt.argmax(1)]

def skill(y, p):
    cls = np.unique(y); ba = np.mean([(p[y == c] == c).mean() for c in cls]); K = len(cls)
    return (ba - 1 / K) / (1 - 1 / K)

def r2(y, p): return 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()

def probe(X, y, f, kind="knn", reg=False, z=True):
    p = np.zeros(len(y), float if reg else int)
    for i in np.unique(f):
        tr, te = f != i, f == i
        if kind == "knn": p[te] = knn_pred(X[tr], y[tr], X[te], reg=reg, z=z)
        else:
            a, b = zs(X[tr], X[te])
            mdl = Ridge(1.0) if reg else LogisticRegression(max_iter=3000, C=1.0)
            p[te] = mdl.fit(a, y[tr]).predict(b)
    return r2(y, p) if reg else skill(y, p)

def emb(k, arch, t, s, which):
    path = os.path.join(C3.OUT, "ckpt", T3.tag(k, arch, 3e-4, t, s, False), f"{which}.pt")
    return E.embed_views(E.load_encoder(path), crops)

rows = []
for k in [0, 3, 2, 1]:
    sel = run == k; g = blk[sel]
    F = {"sorted": folds_sorted(g), "random": folds_random(g)}
    reps = {}
    for s in range(5):
        reps[("tf", s)] = emb(k, "tf", 0.1, s, "final")[sel]
        reps[("tfinit", s)] = emb(k, "tf", 0.1, s, "init")[sel]
        reps[("cnn01", s)] = emb(k, "cnn", 0.1, s, "final")[sel]
        reps[("cnn02", s)] = emb(k, "cnn", 0.2, s, "final")[sel]
        reps[("cnninit", s)] = emb(k, "cnn", 0.1, s, "init")[sel]
    tr = run != k
    pca = PCA(64, random_state=0).fit(raw[tr])
    reps[("pca64", 0)] = pca.transform(raw)[sel]
    reps[("raw", 0)] = raw[sel]
    for (rep, s), X in reps.items():
        for name in ["polarity", "lobes", "duration"]:
            y = Y[name][sel]; reg = name == "duration"
            for fn, f in F.items():
                rows.append(dict(run=RUNS[k], rep=rep, seed=s, score=name, probe="knn15_z", folds=fn, value=probe(X, y, f, reg=reg)))
            if rep.startswith("pca") or rep == "raw":
                rows.append(dict(run=RUNS[k], rep=rep, seed=s, score=name, probe="knn15_noz", folds="sorted", value=probe(X, y, F["sorted"], reg=reg, z=False)))
            rows.append(dict(run=RUNS[k], rep=rep, seed=s, score=name, probe="linear", folds="sorted", value=probe(X, y, F["sorted"], kind="lin", reg=reg)))
    print("done", RUNS[k], flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(HERE, "recompute_scores.csv"), index=False)
