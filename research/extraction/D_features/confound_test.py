"""PROTOCOL item 4: do t_hours and dc_baseline (and the other absolute-level legacy columns) leak
time-of-day / day / run?  H0: a feature set predicts the target no better than shuffled labels.
Also measured on BACKGROUND windows (no vehicle) -> pure sensor/clock state."""
import json, numpy as np, pandas as pd
from features import LEGACY_NAMES, LEGACY_CONFOUNDS
from evaluate import leakage, Scaler, SCR, HOLDOUT_RUN, build_reps

F = dict(np.load(SCR + "feats.npz")); meta = pd.read_parquet(SCR + "meta2.parquet")
L = build_reps(F, meta)["legacy70"]
col = {n: i for i, n in enumerate(LEGACY_NAMES)}
sets = {
    "t_hours": ["t_hours"],
    "dc_baseline": ["dc_baseline"],
    "dc_level": ["dc_level"],
    "3_abs_cols(t_hours,dc_baseline,dc_level)": LEGACY_CONFOUNDS,
    "legacy70": LEGACY_NAMES,
    "legacy_minus_3": [n for n in LEGACY_NAMES if n not in LEGACY_CONFOUNDS],
    "morph": None,
}
rows = []
for kind in ["event", "bg"]:
    for split, rm in [("dev", meta.run.values != HOLDOUT_RUN), ("holdout", meta.run.values == HOLDOUT_RUN)]:
        m = (meta.kind.values == kind) & rm
        mm = meta[m]
        targets = {"hour_bin(6h)": mm.wall.dt.hour.values // 6, "date": mm.wall.dt.strftime("%m%d").values}
        if split == "dev":
            targets["run"] = mm.run.values
        for sname, cols in sets.items():
            X = F["orig_morph"][m] if cols is None else L[m][:, [col[c] for c in cols]]
            Z = Scaler().fit(X).transform(X)
            for tname, y in targets.items():
                r = leakage(Z, y)
                rows.append(dict(windows=kind, split=split, features=sname, target=tname, K=r["n_classes"], n=len(y),
                                 skill=r["skill"][0], skill_lo=r["skill"][1], skill_hi=r["skill"][2],
                                 null_shuffled=r["null_shuffled"][0], null_shuf_hi=r["null_shuffled"][2],
                                 null_majority=r["null_majority"]))
                print(rows[-1], flush=True)
df = pd.DataFrame(rows)
df.to_csv("confound_leakage.csv", index=False)
print(df.round(3).to_string())
