# Reviewer's Q3 check: own conditional-leakage Δ (own PCA/z-score/skill/CV loop), plus calibration nulls.
import sys, json, numpy as np
sys.path.insert(0, "research/model/M4_methods"); sys.path.insert(0, ".")
import m4
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from joblib import Parallel, delayed
m, _ = m4.load(); E = dict(np.load(sys.argv[1])); OUT = sys.argv[2]
day = m.date.astype(str).to_numpy(); y0 = (m.run_id.to_numpy() == 0).astype(int)
morph = m[[c for c in m.columns if c.startswith("m8_")]].fillna(0).to_numpy(float)
def zs(X): return (X - X.mean(0)) / (X.std(0) + 1e-12)
def pca(X, k): Xc = zs(X); U, s, Vt = np.linalg.svd(Xc - Xc.mean(0), full_matrices=False); return U[:, :k] * s[:k]
def skill(y, p): return np.mean([(p[y == c] == c).mean() for c in (0, 1)]) * 2 - 1
def cvskill(Z, y, cv="gkf5"):
    p = np.empty(len(y), int); sp = GroupKFold(5) if cv == "gkf5" else LeaveOneGroupOut()
    for a, b in sp.split(Z, y, day):
        mu, sd = Z[a].mean(0), Z[a].std(0) + 1e-12
        p[b] = LogisticRegression(max_iter=3000).fit((Z[a] - mu) / sd, y[a]).predict((Z[b] - mu) / sd)
    return skill(y, p)
def delta(X, y, cv="gkf5"):
    base = cvskill(morph, y, cv); both = cvskill(np.c_[zs(pca(X, 19)), zs(morph)], y, cv)
    return dict(morph=base, both=both, delta=both - base, alone=cvskill(X, y, cv))
res = {}
for s in range(5): res[f"simclr_cnn_s{s}"] = delta(E[f"simclr_cnn_s{s}"], y0)
res["simclr_tf_s0"] = delta(E["simclr_tf_s0"], y0)
res["simclr_cnn_s0_LODO"] = delta(E["simclr_cnn_s0"], y0, "lodo")
res["simclr_tf_s0_LODO"] = delta(E["simclr_tf_s0"], y0, "lodo")
res["B1_s0_LODO"] = delta(E["B1_s0"], y0, "lodo")
rng = np.random.default_rng(1)
res["noise19"] = [delta(rng.standard_normal((len(y0), 64)), y0)["delta"] for _ in range(5)]
# day-level label permutation null: 'is-0204' assigned to 8 random days of 18
ud = np.unique(day)
def perm(i):
    r = np.random.default_rng(100 + i); pos = r.choice(ud, 8, replace=False); yp = np.isin(day, pos).astype(int)
    return delta(E["simclr_cnn_s0"], yp)["delta"], delta(E["B1_s0"], yp)["delta"]
P = Parallel(n_jobs=6, backend="threading")(delayed(perm)(i) for i in range(60))
res["perm_null_cnn_s0"] = [p[0] for p in P]; res["perm_null_B1_s0"] = [p[1] for p in P]
for k in ("perm_null_cnn_s0", "perm_null_B1_s0", "noise19"):
    v = np.array(res[k]); print(k, "mean %.3f sd %.3f q95 %.3f max %.3f" % (v.mean(), v.std(), np.quantile(v, .95), v.max()))
for k, v in res.items():
    if isinstance(v, dict): print(k, {a: round(b, 4) for a, b in v.items()})
json.dump(res, open(OUT, "w"), indent=1, default=float)
