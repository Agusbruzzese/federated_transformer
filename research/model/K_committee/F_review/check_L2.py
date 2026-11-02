"""Reviewer F: does the per-(seed, fold) permutation null DETECT planted per-day offsets that the gate still PASSES?
Report 09 8.2 L2 requires gate FAIL at "the smallest strength the permutation null detects"; report 12's test only
requires FAIL for some kappa. Same planting as test_evaluator.py (M5 seeds 0-4, rng(500 + fold) unit vectors).
Output: F_review/check_L2.json (exclusive create).
"""
import json, os, sys, time
import numpy as np
from joblib import Parallel, delayed
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(HERE), "protocol"))
import evaluator as E
FOLD = [0, 1, 2]; EMB = {}
for i in range(5):
    enc = E.load_encoder(f"data/m5/ckpt/simclr_tf_s{i}/final.pt")
    for f in FOLD:
        tr, ev, _ = E.fold_sets(f); EMB[i, f] = (E.embed(enc, tr), E.embed(enc, ev))
E.data()
def job(i, f, kappa):
    tr, ev, days = E.fold_sets(f); Etr, Eev = EMB[i, f]
    if kappa > 0:
        y = E.data()["day"][ev]; r = np.random.default_rng(500 + f); sd = Etr.std(0)
        U = {d: (lambda v: v / np.linalg.norm(v))(r.standard_normal(Eev.shape[1])) for d in days}
        Eev = Eev + kappa * np.linalg.norm(sd) * np.stack([U[d] for d in y])
    g = E.gate_core(Eev, *E.gate_arrays(ev))
    return (i, f, kappa), dict(Q=g["Q"], Q0=g["Q0"], Z=g["Z"], q95=g["Q_null_q95"], detected=bool(g["Q"] > g["Q_null_q95"]))
t0 = time.time()
jobs = [(i, f, k) for k in (0.0, 0.01, 0.02) for i in range(5) for f in FOLD]
G = dict(Parallel(n_jobs=12, backend="threading")(delayed(job)(*j) for j in jobs))
out = {}
for k in (0.0, 0.01, 0.02):
    det = [G[i, f, k]["detected"] for i in range(5) for f in FOLD]
    out[str(k)] = dict(detected_frac=float(np.mean(det)), Zbar=[float(np.mean([G[i, f, k]["Z"] for f in FOLD])) for i in range(5)],
                       per_run=[dict(seed=i, fold=f, **G[i, f, k]) for i in range(5) for f in FOLD])
out["secs"] = round(time.time() - t0)
json.dump(out, open(os.path.join(HERE, "check_L2.json"), "x"), indent=1)
print(json.dumps({k: (v["detected_frac"], v["Zbar"]) if isinstance(v, dict) else v for k, v in out.items()}, indent=1))
