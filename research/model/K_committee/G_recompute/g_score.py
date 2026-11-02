"""G stage 2: probes per (run, eval day) with own code, from stage-1 embeddings. Writes g_scores.parquet."""
import os, sys, json, warnings
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning

sys.path.insert(0, os.getcwd()); sys.path.insert(0, "research/model/M4_methods")
import m4

OUT = "research/model/K_committee/G_recompute"
ARMS = ["C256", "F-nat", "S+F", "F-X+P", "S+C"]
m, _ = m4.load()
day = m.date.astype(str).to_numpy()
TGT = dict(POLARITY=(m.m8_polarity.to_numpy() > 0).astype(int),
           LOBES=np.minimum(m.m8_n_lobes.to_numpy(), 4).astype(int))


def skill(y, p):
    cls = np.unique(y); K = len(cls)
    ba = np.mean([np.mean(p[y == c] == c) for c in cls])
    return (ba - 1 / K) / (1 - 1 / K)


def run(arm, f, s):
    z = np.load(f"{OUT}/emb/v21_{arm}_f{f}_s{s}.npz")
    sc = StandardScaler().fit(z["client"]); Xc = sc.transform(z["client"]); Xe = sc.transform(z["eval"])
    de = day[z["ei"]]; rows = []
    for met, y in TGT.items():
        yc, ye = y[z["ci"]], y[z["ei"]]
        pk = KNeighborsClassifier(15).fit(Xc, yc).predict(Xe)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always", ConvergenceWarning)
            lr = LogisticRegression(C=1.0, tol=1e-8, max_iter=100000).fit(Xc, yc)
        nconv = sum(issubclass(x.category, ConvergenceWarning) for x in w)
        pl = lr.predict(Xe)
        for d in sorted(set(de)):
            k = de == d
            sk, sl = skill(ye[k], pk[k]), skill(ye[k], pl[k])
            rows.append(dict(arm=arm, seed=s, fold=f, day=d, metric=met, value=(sk + sl) / 2,
                             knn=sk, lin=sl, n_iter=int(lr.n_iter_.max()), nonconv=nconv, n_classes=len(np.unique(ye[k]))))
    return rows


jobs = [(a, f, s) for a in ARMS for f in range(3) for s in range(10)]
res = Parallel(n_jobs=16)(delayed(run)(*j) for j in jobs)
df = pd.DataFrame([r for rr in res for r in rr])
fn = f"{OUT}/g_scores.parquet"; assert not os.path.exists(fn)
df.to_parquet(fn)
print(df.groupby(["arm", "metric"]).value.mean().unstack(), "\nnonconv total:", df.nonconv.sum(), " max iter:", df.n_iter.max())
