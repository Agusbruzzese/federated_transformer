"""Why do the morph scalars predict run/date on the all-SNR tier (skill 0.31/0.19) but not on SNR>=20 (date 0.03)?"""
import os, sys, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "..", "..")))
from validate import load_meta, skill_cv
from extraction.morph import MORPH_NAMES
m = load_meta(); m["blk30"] = m.wall_time.dt.floor("30min").astype(str); m["hbin"] = m.hour // 6
m["sigma"] = np.abs(m.peak_z) / m.snr
m["wkday"] = pd.to_datetime(m.date).dt.dayofweek < 5
cols = [f"m_{k}" for k in MORPH_NAMES]
def sk(s, target, X=None):
    X = s[cols].to_numpy(float) if X is None else X
    return skill_cv(X, s[target].astype(str).to_numpy(), s["blk30"].to_numpy())[0]
print("local noise sigma (counts, |peak_z|/snr) median by run:", m.groupby("run_id").sigma.median().round(1).to_dict())
print("share of events with SNR<10 by run:", m.groupby("run_id").snr.apply(lambda x: (x < 10).mean()).round(3).to_dict())
rows = []
def add(name, s, target, X=None):
    v = sk(s, target, X); rows.append(dict(test=name, target=target, n=len(s), skill=v)); print(rows[-1], flush=True)
A = m
add("all, runs 1-3 only", A[A.run_id > 0], "run_id"); add("all, runs 1-3 only", A[A.run_id > 0], "date")
add("all, run 0204 only", A[A.run_id == 0], "date"); add("all, run 0218 only", A[A.run_id == 2], "date")
mh = A[A.wkday & A.hour.between(7, 18)]
add("all, weekdays 07-19 h only", mh, "run_id"); add("all, weekdays 07-19 h only", mh, "date")
for lo, hi in ((0, 10), (10, 20), (20, 1e9)):
    s = A[(A.snr >= lo) & (A.snr < hi)]
    add(f"SNR {lo}-{hi}", s, "run_id"); add(f"SNR {lo}-{hi}", s, "date")
add("all, local sigma alone (not a morph feature)", A, "run_id", A[["sigma"]].to_numpy())
add("all, SNR alone", A, "run_id", A[["snr"]].to_numpy())
# single-feature run skill (all tier)
for c in cols:
    add(f"all, single feature {c}", A, "run_id", A[[c]].to_numpy())
pd.DataFrame(rows).to_csv(os.path.join(HERE, "tab_leakage_investigate.csv"), index=False)
m["band"] = pd.cut(m.snr, [0, 10, 20, 1e9], labels=["<10", "10-20", ">=20"])
T = m.groupby(["band", "run_id"]).agg(n=("snr", "size"), dur50_med=("m_dur50_main", "median"),
                                      short_frac=("m_dur50_main", lambda x: (x <= 0.2).mean()),
                                      dur10_med=("m_dur10", "median"))
print(T.round(3).to_string())
T.to_csv(os.path.join(HERE, "tab_leakage_lobe_by_run.csv"))
