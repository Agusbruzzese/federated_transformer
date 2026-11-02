"""A6b: where does the day signal of A6 come from? Same probe/CV as a6_identifiability.py.
(1) per-day recall (TF final mean over 5 seeds, TF init mean, CNN final mean, morph8) minus 1/K;
(2) day skill recomputed on training days EXCLUDING run 0204 (run_id == 0), the run with dropout blips.
Output: a6b_perday_recall.csv, a6b_no0204.csv
"""
from common import *   # noqa
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_predict
from joblib import Parallel, delayed

m, W = m4.load(); tr = (m.role == "train").to_numpy()
print(m[tr].groupby(m.date.astype(str)[tr]).run_id.first().to_dict())
reps = {}
for meth, tag in (("simclr_tf", "tf"), ("simclr_cnn", "cnn")):
    for which in ("final", "init"):
        for s in SEEDS: reps[f"{tag}|{which}|{s}"] = emb(meth, s, which)
reps["morph8|baseline|0"] = m[[c for c in m.columns if c.startswith("m8_")]].fillna(0).to_numpy(float)

def oof(Z, sel):
    y = m.date.astype(str).to_numpy()[sel]
    p = cross_val_predict(make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000)), Z[sel], y, groups=m.blk.to_numpy()[sel], cv=GroupKFold(5))
    return y, p

res = Parallel(n_jobs=8)(delayed(oof)(Z, tr) for Z in reps.values())
rec = []
for k, (y, p) in zip(reps, res):
    days = np.unique(y)
    for d in days: rec.append(dict(rep=k.rsplit("|", 1)[0], seed=k.rsplit("|", 1)[1], day=d, n=(y == d).sum(), recall_minus_chance=np.mean(p[y == d] == d) - 1 / len(days)))
R = pd.DataFrame(rec).groupby(["rep", "day"]).agg(n=("n", "first"), recall_minus_chance=("recall_minus_chance", "mean")).reset_index()
R = R.pivot(index=["day", "n"], columns="rep", values="recall_minus_chance").reset_index()
R.to_csv(os.path.join(HERE, "a6b_perday_recall.csv"), index=False)
pd.set_option("display.width", 200); print(R.round(3).to_string())

sel = tr & (m.run_id.to_numpy() != 0)
res2 = Parallel(n_jobs=8)(delayed(oof)(Z, sel) for Z in reps.values())
D = pd.DataFrame([dict(rep=k.rsplit("|", 1)[0], seed=k.rsplit("|", 1)[1], skill_no0204=m4.skill(y, p)) for k, (y, p) in zip(reps, res2)])
D.to_csv(os.path.join(HERE, "a6b_no0204.csv"), index=False)
print("days kept", np.unique(m.date.astype(str)[sel]).tolist())
print(D.groupby("rep").skill_no0204.agg(["mean", "std", "min", "max"]).round(4).to_string())
