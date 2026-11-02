"""Committee B: evaluation-data options, event/day accounting (SNR>=20 events, m4.load()).

    python research/model/K_committee/B_stats/eval_options.py   -> eval_options.csv, day_counts.csv
"""
import os, sys
import numpy as np, pandas as pd
ROOT = os.getcwd(); sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "research/model/M4_methods"))
import m4

HERE = os.path.dirname(os.path.abspath(__file__))
m, _ = m4.load()
d = m.groupby(m.date.astype(str)).agg(n=("snr", "size"), blocks=("blk", "nunique"), run=("run_id", "first")).reset_index()
d["seen_as_test"] = d.date.isin(m4.TEST_DAYS + [m4.VAL_DAY])
d["large"] = d.n >= 140
d.to_csv(os.path.join(HERE, "day_counts.csv"), index=False)
N = d.n.sum(); large = d[d.large].date.tolist(); small = d[~d.large].date.tolist()

rows = []
def opt(name, eval_days, train_days, folds=1, note=""):
    e = d[d.date.isin(eval_days)]; t = d[d.date.isin(train_days)]
    rows.append(dict(option=name, folds=folds, eval_days_total=len(e) if folds == 1 else len(eval_days), eval_events_total=int(e.n.sum()),
                     eval_days_per_fold=len(e) // folds, train_days_per_fold=len(t) if folds == 1 else len(train_days),
                     train_events_per_fold=int(t.n.sum()) if folds == 1 else None, note=note))

seen = m4.TEST_DAYS; val = [m4.VAL_DAY]
opt("a reuse 4 exhausted test days", seen, [x for x in d.date if x not in seen + val], note="exploratory only; D=4")
fresh = [x for x in large if x not in seen + val]
opt("b1 4 fresh held-out days (largest unseen) + reuse none", sorted(fresh, key=lambda x: -d.set_index('date').n[x])[:4],
    [x for x in d.date if x not in sorted(fresh, key=lambda x: -d.set_index('date').n[x])[:4] + val])
opt("b1' 8 held-out days = 4 old test + 4 fresh", seen + sorted(fresh, key=lambda x: -d.set_index('date').n[x])[:4],
    [x for x in d.date if x not in seen + sorted(fresh, key=lambda x: -d.set_index('date').n[x])[:4] + val])
# b2: K-fold cross-fitting over the 13 large days (small days always train), each fold retrains every arm
rng = np.random.default_rng(0)
for K in (2, 3, 4):
    lg = sorted(large); perm = list(rng.permutation(lg)); fold = {x: i % K for i, x in enumerate(perm)}
    tr_ev = []; tr_days = []
    for k in range(K):
        ev = [x for x in lg if fold[x] == k]; tr = [x for x in d.date if x not in ev]
        tr_ev.append(int(d[d.date.isin(tr)].n.sum())); tr_days.append(len(tr))
    rows.append(dict(option=f"b2 {K}-fold cross-fitting over 13 large days", folds=K, eval_days_total=len(lg), eval_events_total=int(d[d.date.isin(lg)].n.sum()),
                     eval_days_per_fold=f"{len(lg)//K}-{-(-len(lg)//K)}", train_days_per_fold=f"{min(tr_days)}-{max(tr_days)}",
                     train_events_per_fold=f"{min(tr_ev)}-{max(tr_ev)}", note=f"training cost x{K}; every large day evaluated once"))
# b3: within-day temporal holdout (last 25 % of 30-min blocks of every day)
hold = 0; tot_tr = 0
for x, g in m.groupby(m.date.astype(str)):
    b = np.sort(g.blk.unique()); cut = b[int(np.floor(0.75 * len(b)))] if len(b) > 1 else b[-1] + 1
    hold += int((g.blk >= cut).sum()); tot_tr += int((g.blk < cut).sum())
rows.append(dict(option="b3 within-day temporal holdout (last 25 % of blocks per day)", folds=1, eval_days_total=len(d), eval_events_total=hold,
                 eval_days_per_fold=len(d), train_days_per_fold=len(d), train_events_per_fold=tot_tr,
                 note="every client keeps its own held-out tail; enables client-identity leakage; same-day environment shared"))
rows.append(dict(option="c synthetic held-out events (synth/)", folds=1, eval_days_total="any", eval_events_total="unlimited",
                 eval_days_per_fold="-", train_days_per_fold=len(d), train_events_per_fold=int(N), note="known labels; generator fitted on real data; C2ST 0.60"))
R = pd.DataFrame(rows); R.to_csv(os.path.join(HERE, "eval_options.csv"), index=False)
print(d.to_string()); print(R.to_string())
