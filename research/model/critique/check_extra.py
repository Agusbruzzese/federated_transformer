"""Critic extras: (1) CNN vs TF absolute, fair (within-space) per seed/run; (2) morph8 & SNR reference under the same
transfer/within-run probes; (3) how much of each embedding is linearly explained by morph8 + log SNR (held-out run).
    .venv/bin/python research/model/critique/check_extra.py
"""
import os; os.environ["OMP_NUM_THREADS"] = "1"
import sys, numpy as np, pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import Ridge
sys.path.insert(0, "/home/augustin/Documents/federated_transformer/research/model/critique")
import check_within_space as W

OUT = W.OUT
P = pd.read_csv(os.path.join(OUT, "ws_p1_per_seed.csv"))
pv = P.pivot_table(index=["probe", "target", "run", "seed"], columns="arch", values="ssl").reset_index()
pv["d"] = pv.cnn - pv.tf
c = pv.groupby(["probe", "target"]).agg(cnn=("cnn", "mean"), tf=("tf", "mean"), d_cnn_minus_tf=("d", "mean"),
                                        n_cnn_better_of_20=("d", lambda x: int((x > 0).sum())), d_min=("d", "min"), d_max=("d", "max"))
mean_row = pv.groupby(["probe", "run", "seed"])[["cnn", "tf", "d"]].mean().groupby("probe").agg(["mean", "min", "max"])
print(c.round(3).to_string()); print("P1 mean over targets:\n", mean_row.round(3).to_string())
c.to_csv(os.path.join(OUT, "ws_cnn_vs_tf.csv"))

m = W.m; tier = W.tier; run = W.run
M8 = m[[q for q in m.columns if q.startswith("m8_")]].to_numpy(float)
SNR = W.ls[:, None]
ref = []
for nm, X in (("morph8", M8), ("snr", SNR), ("pca64", np.load(os.path.join(W.EMB, "pca64.npy")))):
    for k, rn in W.RUNS.items():
        o = W.eval_rep(X, k)
        for (probe, t), (v, _) in o.items():
            ref.append(dict(rep=nm, run=rn, probe=probe, target=t, value=v))
R = pd.DataFrame(ref).groupby(["rep", "probe", "target"]).value.mean().unstack("target").round(3)
print(R.to_string()); R.to_csv(os.path.join(OUT, "ws_reference_reps.csv"))

# linear explainability of each embedding by morph8 + log SNR, fit on training runs, R2 on held-out run (single space)
Z = np.hstack([M8, SNR]); Z = (Z - Z[tier].mean(0)) / Z[tier].std(0)
def lin_r2(arch, k, s, kind):
    X = np.load(os.path.join(W.EMB, f"ws_f{k}_{arch}_s{s}.npy" if kind == "ssl" else f"{arch}_rand_s{s}.npy"))
    tr = tier & (run != k); te = tier & (run == k)
    X = (X - X[tr].mean(0)) / (X[tr].std(0) + 1e-9)
    p = Ridge(1.0).fit(Z[tr], X[tr]).predict(Z[te])
    r2 = 1 - ((X[te] - p) ** 2).sum(0) / ((X[te] - X[te].mean(0)) ** 2).sum(0)
    return dict(arch=arch, run=W.RUNS[k], seed=s, kind=kind, r2_mean_dim=float(r2.mean()),
                r2_total=float(1 - ((X[te] - p) ** 2).sum() / ((X[te] - X[te].mean(0)) ** 2).sum()))
L = pd.DataFrame(Parallel(n_jobs=8)(delayed(lin_r2)(a, k, s, kd) for a in ("cnn", "tf") for k in W.RUNS
                                    for s in W.SEEDS for kd in ("ssl", "rand")))
Ls = L.groupby(["arch", "kind"])[["r2_mean_dim", "r2_total"]].agg(["mean", "min", "max"]).round(3)
print(Ls.to_string()); Ls.to_csv(os.path.join(OUT, "ws_morph8_explains_embedding.csv"))
