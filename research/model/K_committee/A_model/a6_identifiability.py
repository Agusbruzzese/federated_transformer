"""A6: client (calendar-day) identifiability baseline on TRAINING-day events only.

y = calendar date (9 training days = the natural federated clients). Probe: StandardScaler + LogisticRegression(C=1,
max_iter=5000), out-of-fold predictions with GroupKFold(5) over 30-min blocks (m.blk), so every day appears in both
train and test folds but no block is split. Score: balanced-accuracy skill (m4.skill; 0 = chance, 1 = perfect).
Representations: SimCLR-TF final/init s0-4, SimCLR-CNN final/init s0-4, morph8 (m8_* columns, NaN->0), raw-crop PCA-64.
Null: 50 block-level permutations (each 30-min block gets the day label of another block, block sizes kept) for
morph8, TF final s0, TF init s0 -> 95th percentile of null skill.
Output: a6_identifiability.csv, a6_null.csv
"""
from common import *   # noqa
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_predict
from joblib import Parallel, delayed

m, W = m4.load(); tr = (m.role == "train").to_numpy()
y = m.date.astype(str).to_numpy()[tr]; blk = m.blk.to_numpy()[tr]
print("training days", dict(zip(*np.unique(y, return_counts=True))), "blocks", len(np.unique(blk)), flush=True)
reps = {}
for meth, tag in (("simclr_tf", "tf"), ("simclr_cnn", "cnn")):
    for which in ("final", "init"):
        for s in SEEDS: reps[f"{tag}|{which}|{s}"] = emb(meth, s, which)[tr]
reps["morph8|baseline|0"] = m[[c for c in m.columns if c.startswith("m8_")]].fillna(0).to_numpy(float)[tr]
reps["raw_pca64|baseline|0"] = PCA(64, random_state=0).fit(m5.crops().cpu().numpy()[tr]).transform(m5.crops().cpu().numpy()[tr])

def sk(Z, yy):
    p = cross_val_predict(make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000)), Z, yy, groups=blk, cv=GroupKFold(5))
    return m4.skill(yy, p)

res = Parallel(n_jobs=8)(delayed(sk)(Z, y) for Z in reps.values())
D = pd.DataFrame([dict(zip(("rep", "which", "seed"), k.split("|")), skill=v) for k, v in zip(reps, res)])
D.to_csv(os.path.join(HERE, "a6_identifiability.csv"), index=False)
pd.set_option("display.width", 200); print(D.round(4).to_string())
print(D.groupby(["rep", "which"]).skill.agg(["mean", "std", "min", "max"]).round(4).to_string(), flush=True)

# block-level permutation null
ub, inv = np.unique(blk, return_inverse=True); bday = pd.Series(y).groupby(inv).first().to_numpy()
def perm(Z, i):
    r = np.random.default_rng(i); return sk(Z, r.permutation(bday)[inv])
nl = []
for k in ("morph8|baseline|0", "tf|final|0", "tf|init|0"):
    v = Parallel(n_jobs=8)(delayed(perm)(reps[k], i) for i in range(50))
    nl.append(dict(rep=k, null_mean=np.mean(v), null_sd=np.std(v, ddof=1), null_q95=np.quantile(v, 0.95), null_max=np.max(v),
                   observed=float(D[(D.rep + "|" + D.which + "|" + D.seed.astype(str)) == k].skill.iloc[0])))
N = pd.DataFrame(nl); N.to_csv(os.path.join(HERE, "a6_null.csv"), index=False); print(N.round(4).to_string())
