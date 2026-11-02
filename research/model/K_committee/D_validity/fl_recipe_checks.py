"""Committee member D: checks of what breaks / changes when the M5 central recipe is run per client (day).

    .venv/bin/python research/model/K_committee/D_validity/fl_recipe_checks.py

1. m5.train loop control (copied verbatim: `spe = N // bs`, `for b in range(spe)`, `step += 1` inside) with
   N < 256 never increments `step` -> infinite loop. Simulated with a step cap.
2. False-negative guard (|dt| < 30 s masked as negatives): fraction of masked negative pairs in a central
   batch (256 drawn from all training days) vs a client batch (min(256, N_day) drawn from one day).
3. Number of negatives per anchor per client batch (2B - 2) vs central (510).
Writes fl_recipe_checks.json next to this file.
"""
import os, sys, json
import numpy as np, pandas as pd

ROOT = os.getcwd(); sys.path.insert(0, ROOT)
from ssl_central import data as D

HERE = os.path.dirname(os.path.abspath(__file__))
TEST = ["2025-02-07", "2025-02-13", "2025-02-19", "2025-02-25"]; VAL = "2025-02-10"


def m5_loop_terminates(N, BATCH=256, STEPS=50, cap=10_000):
    bs = BATCH; spe = N // bs; step = 0; it = 0
    while step < STEPS:
        it += 1
        if it > cap:
            return False
        for b in range(spe):
            if step >= STEPS: break
            step += 1
    return True


def guard_frac(t, rng, B, reps=200):
    out = []
    for _ in range(reps):
        tt = np.sort(rng.choice(t, min(B, len(t)), replace=False)); tt = np.r_[tt, tt]; n = len(tt) // 2
        near = np.abs(tt[:, None] - tt[None, :]) < 30.0
        idx = np.arange(2 * n); same = np.zeros_like(near); same[idx, idx] = True; same[idx, np.r_[idx[n:], idx[:n]]] = True
        out.append((near & ~same).sum() / (2 * n * (2 * n - 2)))
    return float(np.mean(out))


m, _, _, _ = D.load_all()
m = m[m.snr >= 20].copy(); m["day"] = m.date.astype(str)
tr = m[~m.day.isin(TEST + [VAL])]
rng = np.random.default_rng(0)
res = dict(loop_terminates={str(N): m5_loop_terminates(N) for N in (35, 255, 256, 2636)})
res["guard_masked_frac_central_B256"] = round(guard_frac(tr.t_s.to_numpy(), rng, 256), 5)
per = {}
for d, g in tr.groupby("day"):
    B = min(256, len(g))
    per[d] = dict(n=len(g), batch=B, negatives_per_anchor=2 * B - 2, guard_masked_frac=round(guard_frac(g.t_s.to_numpy(), rng, B), 5))
res["per_client_day"] = per
json.dump(res, open(os.path.join(HERE, "fl_recipe_checks.json"), "w"), indent=1)
print(json.dumps(res, indent=1))
