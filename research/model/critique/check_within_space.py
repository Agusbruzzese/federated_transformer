"""Independent critic check: SSL vs its OWN random-init, within a single model's embedding space, per held-out run.

For fold k (held-out run k), seed s:
  SSL  = ws_f{k}_{arch}_s{s}  (fold-k SSL model; run k never seen in training)
  RAND = {arch}_rand_s{s}      (the exact initial weights of that same model: init identical across folds, J3)
Probe A ("within-run"):   kNN k=15, z-scored, GroupKFold(5) by 30-min block, on run k's snr>=20 events only.
Probe B ("transfer"):     kNN fit on the 3 training runs' events (embedded in the same single space), predict run k.
Also: is-0204 and date leakage within a single space (LR, GroupKFold(5) by block), SSL vs its own init.
    .venv/bin/python research/model/critique/check_within_space.py
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
import sys, json
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import warnings; warnings.filterwarnings("ignore")

REPO = "/home/augustin/Documents/federated_transformer"
sys.path.insert(0, REPO)
from ssl_central_v3 import data as D
EMB = os.path.join(REPO, "data/ssl_central_v3/emb")
OUT = os.path.join(REPO, "research/model/critique")
TARGETS = {"amp_resid": "reg", "log_dur": "reg", "polarity": "cls", "n_lobes": "cls"}
RUNS = {0: "0204", 1: "0212", 2: "0218", 3: "0224"}
SEEDS = range(5); B = 1000

m, _ = D.load_all()
idx = pd.read_parquet(os.path.join(EMB, "index.parquet"))
assert (idx.event_id.values == m.event_id.values).all()
tier = (m.snr >= 20).to_numpy()
run = m.run_id.to_numpy(); blk = m.blk30.to_numpy()
la = m.m8_log_amp.to_numpy(float); ls = m.log_snr.to_numpy(float)
Y = {"log_dur": m.y_log_dur.to_numpy(float), "polarity": m.y_polarity.to_numpy(), "n_lobes": m.y_n_lobes.to_numpy()}


def metric(y, p, kind):
    if kind == "reg":
        return 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)
    c, yi = np.unique(y, return_inverse=True); K = len(c)
    rec = np.bincount(yi, weights=(y == p).astype(float), minlength=K) / np.bincount(yi, minlength=K)
    return (rec.mean() - 1 / K) / (1 - 1 / K)


def z(a, b):
    mu = a.mean(0); sd = a.std(0); sd[sd == 0] = 1
    return (a - mu) / sd, (b - mu) / sd


def knn_pred(Xtr, Xte, t, itr, ite):
    kind = TARGETS[t]
    if t == "amp_resid":
        c = np.polyfit(ls[itr], la[itr], 1)
        ytr = la[itr] - np.polyval(c, ls[itr]); yte = la[ite] - np.polyval(c, ls[ite])
    else:
        ytr, yte = Y[t][itr], Y[t][ite]
    a, b = z(Xtr, Xte)
    mdl = (KNeighborsRegressor if kind == "reg" else KNeighborsClassifier)(n_neighbors=15)
    return yte, mdl.fit(a, ytr).predict(b)


def boot_idx(g, seed=0):
    rng = np.random.default_rng(seed); ug, inv = np.unique(g, return_inverse=True)
    mem = [np.flatnonzero(inv == q) for q in range(len(ug))]
    return [np.concatenate([mem[q] for q in rng.integers(0, len(ug), len(ug))]) for _ in range(B)]


def eval_rep(X, k):
    """Returns {(probe,t): (value, boot array)} for held-out run k."""
    te_all = np.flatnonzero(tier & (run == k)); tr_all = np.flatnonzero(tier & (run != k))
    bi = boot_idx(blk[te_all])
    out = {}
    for t, kind in TARGETS.items():
        # Probe A: within run k, GroupKFold by block
        yt = np.empty(len(te_all)); pr = np.empty(len(te_all), float if kind == "reg" else Y.get(t, la).dtype)
        for tr, te in GroupKFold(5).split(te_all, groups=blk[te_all]):
            yt[te], pr[te] = knn_pred(X[te_all[tr]], X[te_all[te]], t, te_all[tr], te_all[te])
        out[("within_run", t)] = (metric(yt, pr, kind), np.array([metric(yt[i], pr[i], kind) for i in bi]))
        # Probe B: fit on training runs (same space), predict run k
        yt2, pr2 = knn_pred(X[tr_all], X[te_all], t, tr_all, te_all)
        out[("transfer", t)] = (metric(yt2, pr2, kind), np.array([metric(yt2[i], pr2[i], kind) for i in bi]))
    return out


def job_p1(arch, k, s):
    S = np.load(os.path.join(EMB, f"ws_f{k}_{arch}_s{s}.npy"))
    R = np.load(os.path.join(EMB, f"{arch}_rand_s{s}.npy"))   # identical across folds (info.json), = this model's init
    return (arch, k, s), eval_rep(S, k), eval_rep(R, k)


def lr_skill(X, y, g):
    pred = np.empty_like(y)
    for tr, te in GroupKFold(5).split(X, groups=g):
        pred[te] = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000)).fit(X[tr], y[tr]).predict(X[te])
    return metric(y, pred, "cls")


def job_p2(arch, k, s):
    S = np.load(os.path.join(EMB, f"ws_f{k}_{arch}_s{s}.npy"))
    R = np.load(os.path.join(EMB, f"{arch}_rand_s{s}.npy"))
    ii = np.flatnonzero(tier)
    y04 = (run[ii] == 0).astype(int)
    res = {}
    for nm, X in (("ssl", S), ("rand", R)):
        res[(nm, "is0204")] = lr_skill(X[ii], y04, blk[ii])
        jj = np.flatnonzero(tier & (run != 0))
        res[(nm, "run_x0204")] = lr_skill(X[jj], run[jj], blk[jj])
        # is-0204 with a kNN probe too (nonlinear)
        pred = np.empty_like(y04)
        for tr, te in GroupKFold(5).split(ii, groups=blk[ii]):
            a, b = z(X[ii][tr], X[ii][te]); pred[te] = KNeighborsClassifier(15).fit(a, y04[tr]).predict(b)
        res[(nm, "is0204_knn")] = metric(y04, pred, "cls")
    return (arch, k, s), res


if __name__ == "__main__":
    jobs = [(a, k, s) for a in ("cnn", "tf") for k in RUNS for s in SEEDS]
    R1 = Parallel(n_jobs=8)(delayed(job_p1)(*j) for j in jobs)
    rows = []; agg = {}
    for (a, k, s), So, Ro in R1:
        for key in So:
            probe, t = key
            rows.append(dict(arch=a, run=RUNS[k], seed=s, probe=probe, target=t, ssl=So[key][0], rand=Ro[key][0],
                             delta=So[key][0] - Ro[key][0]))
            agg.setdefault((a, probe, t, k), []).append(So[key][1] - Ro[key][1])
    P = pd.DataFrame(rows); P.to_csv(os.path.join(OUT, "ws_p1_per_seed.csv"), index=False)
    summ = []
    for (a, probe, t), g in P.groupby(["arch", "probe", "target"]):
        r = dict(arch=a, probe=probe, target=t, ssl=g.ssl.mean(), rand=g.rand.mean(), delta=g.delta.mean(),
                 n_pos_of_20=int((g.delta > 0).sum()))
        bb_runs = []
        for k, nm in RUNS.items():
            gk = g[g.run == nm]; bk = np.mean(agg[(a, probe, t, k)], 0); bb_runs.append(bk)
            r[f"d_{nm}"] = gk.delta.mean(); r[f"lo_{nm}"] = np.percentile(bk, 2.5); r[f"hi_{nm}"] = np.percentile(bk, 97.5)
        bb = np.mean(bb_runs, 0)    # independent block resampling within each run, equal run weight
        r["lo"], r["hi"] = np.percentile(bb, [2.5, 97.5]); r["p_one_sided"] = (1 + np.sum(bb <= 0)) / (1 + B)
        r["n_runs_pos"] = int(sum(r[f"d_{nm}"] > 0 for nm in RUNS.values()))
        summ.append(r)
    Sm = pd.DataFrame(summ); Sm.to_csv(os.path.join(OUT, "ws_p1_summary.csv"), index=False)
    pd.set_option("display.width", 250)
    print(Sm.round(3).to_string())
    R2 = Parallel(n_jobs=8)(delayed(job_p2)(*j) for j in jobs)
    rows = []
    for (a, k, s), res in R2:
        for t in ("is0204", "run_x0204", "is0204_knn"):
            rows.append(dict(arch=a, fold_heldout=RUNS[k], seed=s, target=t, ssl=res[("ssl", t)], rand=res[("rand", t)],
                             delta=res[("ssl", t)] - res[("rand", t)]))
    Q = pd.DataFrame(rows); Q.to_csv(os.path.join(OUT, "ws_p2_per_seed.csv"), index=False)
    Qs = Q.groupby(["arch", "target", "fold_heldout"])[["ssl", "rand", "delta"]].agg(["mean", "min", "max"]).round(3)
    Qs.to_csv(os.path.join(OUT, "ws_p2_summary.csv")); print(Qs.to_string())
