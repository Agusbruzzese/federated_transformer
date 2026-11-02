"""Supplement to confound_test.py: correctly-read (24-bit) baseline and hour-of-day variants.
(The legacy dc_baseline is int16-wrapped, which partly scrambles the absolute level.)"""
import numpy as np, pandas as pd
from evaluate import leakage, Scaler, SCR, HOLDOUT_RUN

meta = pd.read_parquet(SCR + "meta2.parquet")
fl = pd.read_parquet(SCR + "meta_flags.parquet"); meta["has_dropout"] = (fl.n_gaps.values > 0).astype(int)
rows = []
for kind in ["event", "bg"]:
    for split, rm in [("dev", meta.run.values != HOLDOUT_RUN), ("holdout", meta.run.values == HOLDOUT_RUN)]:
        m = (meta.kind.values == kind) & rm; mm = meta[m]
        feats = {"dc_baseline_24bit": mm[["dc_baseline"]].values,
                 "hour_of_day": mm[["t_hours"]].values,
                 "has_dropout": mm[["has_dropout"]].values.astype(float)}
        targets = {"hour_bin(6h)": mm.wall.dt.hour.values // 6, "date": mm.wall.dt.strftime("%m%d").values}
        if split == "dev":
            targets["run"] = mm.run.values
        for fn, X in feats.items():
            for tn, y in targets.items():
                r = leakage(Scaler().fit(X).transform(X), y)
                rows.append(dict(windows=kind, split=split, features=fn, target=tn, n=len(y), skill=r["skill"][0],
                                 skill_lo=r["skill"][1], skill_hi=r["skill"][2], null_shuffled=r["null_shuffled"][0]))
                print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv("confound_extra.csv", index=False)
