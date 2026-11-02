"""G stage 4: gate spot check (Q only, no permutation null), F-nat and C256 seeds 0-2, fold 0.
Variants are reported because report 09 section 6.2 does not fix where PCA is fitted."""
import os, sys, json, warnings
import numpy as np, pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold

sys.path.insert(0, os.getcwd()); sys.path.insert(0, "research/model/M4_methods")
import m4

OUT = "research/model/K_committee/G_recompute"
m, _ = m4.load()
M8 = [c for c in m.columns if c.startswith("m8_")]


def skill(y, p):
    cls = np.unique(y); K = len(cls)
    return (np.mean([np.mean(p[y == c] == c) for c in cls]) - 1 / K) / (1 - 1 / K)


def cv_skill(X, y, grp, pca_inside=None):
    pred = np.empty_like(y)
    for tr, te in GroupKFold(5).split(X, y, grp):
        Xtr, Xte = X[tr], X[te]
        if pca_inside is not None:
            E = pca_inside
            pc = PCA(19).fit(E[tr]); Xtr = np.hstack([pc.transform(E[tr]), Xtr]); Xte = np.hstack([pc.transform(E[te]), Xte])
        sc = StandardScaler().fit(Xtr)
        lr = LogisticRegression(C=1.0, max_iter=2000).fit(sc.transform(Xtr), y[tr])
        pred[te] = lr.predict(sc.transform(Xte))
    return skill(y, pred)


prot = pd.read_parquet("research/model/K_committee/protocol/scoring/models_combined_20261012_211200.parquet")
rows = []
for arm in ["C256", "F-nat"]:
    for s in range(3):
        z = np.load(f"{OUT}/emb/v21_{arm}_f0_s{s}.npz"); ei = z["ei"]; E = z["eval"]; me = m.iloc[ei]
        y = pd.factorize(me.date.astype(str))[0]; grp = me.blk.to_numpy()
        H = np.eye(4)[me.hbin.to_numpy().astype(int)]
        N = np.hstack([me[M8].to_numpy(float), H, me.has_dropout.to_numpy(float)[:, None]])
        ref = cv_skill(N, y, grp)
        q_out = cv_skill(np.hstack([PCA(19).fit_transform(E), N]), y, grp) - ref
        q_in = cv_skill(N, y, grp, pca_inside=E) - ref
        q_out_std = cv_skill(np.hstack([PCA(19).fit_transform(StandardScaler().fit_transform(E)), N]), y, grp) - ref
        pq = prot[(prot.arm == arm) & (prot.seed == s) & (prot.fold == 0)].Q.item()
        rows.append(dict(arm=arm, seed=s, ref_skill=ref, Q_G_pca_global=q_out, Q_G_pca_in_cv=q_in, Q_G_pca_global_std=q_out_std, Q_protocol=pq))
        print(rows[-1], flush=True)
df = pd.DataFrame(rows)
df.to_csv("research/model/K_committee/G_recompute/g_gate_defaulttol.csv", index=False)
print(df.to_string()); print("n m8 cols:", len(M8))
