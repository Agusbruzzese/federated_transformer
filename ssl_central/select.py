"""Stage 1+2: grid training (seed 0), selection on the VALIDATION day only, then seeds 1 and 2 of the winner.

    python -m ssl_central.select
Writes research/model/F_central/tab_selection.csv and <OUT>/selection.json.
Only train (training) and val (selection) events are touched; holdout rows are never embedded here.
"""
from __future__ import annotations

import itertools
import json
import os

import numpy as np
import pandas as pd

from . import config as C
from . import data as D
from . import embed as E
from . import probes as P
from .train import cell_tag, train


def val_score(tag, m, W, va, which="final"):
    enc = E.load_encoder(E.ckpt_path(tag, which))
    X = E.embed_views(enc, E.eval_crops(W[va], m.fs_eff.to_numpy()[va]))
    s = m.iloc[va]; g = s.blk30.to_numpy()
    r = {}
    for t in C.TARGETS:
        y = s[f"y_{t}"].to_numpy()
        r[t] = P.metric(y, P.knn_oof(X, y, g, C.TARGET_KIND[t]), C.TARGET_KIND[t])
    r["mean"] = float(np.mean([r[t] for t in C.TARGETS]))
    return r


def main():
    m, W, EM, RM = D.load_all()
    tr = D.split_index(m, "train"); va = D.split_index(m, "val")
    D.firewall(m, tr, va)
    rows = []
    for arch, lr, temp in itertools.product(("cnn", "tf"), C.GRID_LR, C.GRID_T):
        tag = cell_tag(arch, lr, temp, 0)
        if not os.path.exists(E.ckpt_path(tag, "final")):
            train(arch, lr, temp, 0)
        lg = json.load(open(os.path.join(C.OUT, "ckpt", tag, "log.json")))
        r = val_score(tag, m, W, va)
        rows.append(dict(arch=arch, lr=lr, temp=temp, seed=0, **{f"val_{k}": v for k, v in r.items()},
                         final_loss=lg["loss"][-1], train_s=lg["total_s"]))
        print(rows[-1], flush=True)
    for arch in ("cnn", "tf"):        # random-init reference on validation (reported, not used for selection)
        r = val_score(cell_tag(arch, C.GRID_LR[0], C.GRID_T[0], 0), m, W, va, which="init")
        rows.append(dict(arch=arch + "_randinit", lr=np.nan, temp=np.nan, seed=0,
                         **{f"val_{k}": v for k, v in r.items()}))
    T = pd.DataFrame(rows)
    T.to_csv(os.path.join(C.RES, "tab_selection.csv"), index=False)
    sel = {}
    for arch in ("cnn", "tf"):
        t = T[T.arch == arch].copy()
        t["lt"] = t.lr * t.temp
        best = t.sort_values(["val_mean", "lt"], ascending=[False, True]).iloc[0]
        sel[arch] = dict(lr=float(best.lr), temp=float(best.temp), val_mean=float(best.val_mean))
    json.dump(sel, open(os.path.join(C.OUT, "selection.json"), "w"), indent=1)
    print("SELECTED", sel, flush=True)
    for arch, s in sel.items():
        for seed in C.SEEDS:
            tag = cell_tag(arch, s["lr"], s["temp"], seed)
            if not os.path.exists(E.ckpt_path(tag, "final")):
                train(arch, s["lr"], s["temp"], seed)
    print(T.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
