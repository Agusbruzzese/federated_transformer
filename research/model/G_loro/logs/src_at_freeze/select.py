"""Per-fold selection (J4): seed-0 grid cells scored by leave-one-DAY-out kNN on TRAINING-run events only.

    python -m ssl_central_v3.select
Writes research/model/G_loro/tab_selection.csv and data/ssl_central_v3/selection.json.
"""
from __future__ import annotations

import itertools
import json
import os

import numpy as np
import pandas as pd

from ssl_central import embed as FE
from . import config as C
from . import data as D
from . import probes as P
from .train import ckpt, tag


def score(X, s, sp):
    r = {}
    for t in C.TARGETS:
        yt, pr = P.knn_oof(X, s, t, None, sp=sp)
        r[t] = P.metric(yt, pr, C.KIND[t])
    r["mean"] = float(np.mean([r[t] for t in C.TARGETS]))
    return r


def main():
    m, W = D.load_all()
    rows, sel = [], {}
    for k in C.RUNS:
        idx = D.train_index(m, k)
        D.firewall(m, k, idx)                                   # selection set has no test-run event
        s = m.iloc[idx]
        assert set(s.event_id) .isdisjoint(set(m.event_id[m.run_id == k]))
        sp = P.splits(s.date.to_numpy(), "logo")
        crops = FE.eval_crops(W[idx], s.fs_eff.to_numpy())
        for arch in C.ARCHS:
            best = None
            for lr, temp in itertools.product(C.GRID_LR, C.GRID_T):
                t_ = tag(k, arch, lr, temp, 0)
                X = FE.embed_views(FE.load_encoder(ckpt(t_, "final")), crops)
                r = score(X, s, sp)
                lg = json.load(open(os.path.join(C.OUT, "ckpt", t_, "log.json")))
                rows.append(dict(fold=k, test_run=C.RUN_NAME[k], arch=arch, lr=lr, temp=temp, n_sel=len(idx),
                                 n_days=s.date.nunique(), **r, loss0=lg["loss"][0], loss_final=lg["loss"][-1]))
                print(rows[-1], flush=True)
                key = (r["mean"], -lr * temp)
                if best is None or key > best[0]:
                    best = (key, lr, temp, r["mean"])
            X = FE.embed_views(FE.load_encoder(ckpt(tag(k, arch, C.GRID_LR[0], C.GRID_T[0], 0), "init")), crops)
            rows.append(dict(fold=k, test_run=C.RUN_NAME[k], arch=arch + "_randinit", n_sel=len(idx),
                             n_days=s.date.nunique(), **score(X, s, sp)))
            sel[f"{k}_{arch}"] = dict(fold=k, arch=arch, lr=best[1], temp=best[2], sel_mean=best[3])
    T = pd.DataFrame(rows)
    T.to_csv(os.path.join(C.RES, "tab_selection.csv"), index=False)
    json.dump(sel, open(os.path.join(C.OUT, "selection.json"), "w"), indent=1)
    print(T.round(4).to_string(index=False)); print("SELECTED", json.dumps(sel, indent=1))


if __name__ == "__main__":
    main()
