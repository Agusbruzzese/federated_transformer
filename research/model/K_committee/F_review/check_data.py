"""Reviewer F: data-level checks of the protocol v2 code (no training). Output: F_review/check_data.json (exclusive create).
    .venv/bin/python research/model/K_committee/F_review/check_data.py
"""
import hashlib, json, os, sys, time
import numpy as np, pandas as pd
from scipy import stats
ROOT = os.getcwd(); sys.path[:0] = [ROOT, os.path.join(ROOT, "research/model/M4_methods")]
import m4
PROT = os.path.join(ROOT, "research/model/K_committee/protocol"); HERE = os.path.dirname(os.path.abspath(__file__))
R = {}
sha = lambda f: hashlib.sha256(open(f, "rb").read()).hexdigest()
R["sha"] = {f: sha(os.path.join(PROT, f)) for f in sorted(os.listdir(PROT)) if f.endswith((".py", ".json"))}
R["sha"]["bgpool_day_v2.npy"] = sha(os.path.join(ROOT, "data/k_committee/protocol/bgpool_day_v2.npy"))
m, W = m4.load(); F = json.load(open(os.path.join(PROT, "folds.json"))); day = m.date.astype(str).to_numpy()
tg = m4.targets(m)
# 1. class composition per evaluation day and the tuning day
ev_days = [d for f in "012" for d in F["folds"][f]["eval_days"]] + [F["tuning_day"]]
comp = {}
for d in ev_days:
    k = day == d
    comp[d] = dict(n=int(k.sum()), lobes={int(c): int((tg["lobes"][k] == c).sum()) for c in np.unique(tg["lobes"])},
                   polarity={int(c): int((tg["polarity"][k] == c).sum()) for c in (0, 1)})
R["class_counts"] = comp
R["lobe_classes_all"] = np.unique(tg["lobes"]).tolist()
# 2. blocks never span two calendar days (gate permutation assumes one label per block)
g = pd.DataFrame(dict(blk=m.blk.to_numpy(), day=day))
R["blocks_spanning_days"] = int((g.groupby("blk").day.nunique() > 1).sum())
# hour / date / wall_time consistency
wt = pd.to_datetime(m.wall_time)
R["date_vs_walltime_date_mismatch"] = int((wt.dt.date.astype(str).to_numpy() != day).sum())
R["hour_vs_walltime_hour_mismatch"] = int((wt.dt.hour.to_numpy() != m.hour.to_numpy()).sum())
R["hbin_values"] = sorted(np.unique(m.hbin).tolist())
# blocks per eval day (GroupKFold(5) needs >= 5 groups; permutation granularity)
R["blocks_per_eval_day"] = {d: int(g[g.day == d].blk.nunique()) for d in ev_days}
# 3. noise pool covers every client/server/tuning day
meta = pd.read_parquet(os.path.join(ROOT, "data/k_committee/protocol/bgpool_day_v2_meta.parquet"))
need = sorted(set(F["server"]) | {F["tuning_day"]} | {d for f in F["folds"].values() for c in f["clients"] for d in c["days"]})
R["pool_counts_needed_days"] = {d: int((meta.day == d).sum()) for d in need}
R["pool_day_dtype_matches"] = bool(set(meta.day.astype(str)) >= set(need))
# 4. P derivation (m2)
X = 10_000 * 256; ntr = [F["folds"][f]["train_events"] for f in "012"]; ep = [X / n for n in ntr]
R["exposure_central"] = [round(e, 1) for e in ep]; R["P_recomputed"] = int(round(np.mean(ep) * F["server_events"] / 256) * 256)
# 5. m_leak rule vs "L1 PASS by construction" (report 09 8.2): analytic counter-example and the T5 numbers
T = json.load(open(os.path.join(PROT, "test_evaluator_20261010_124310.json")))["T5"]
sys.path.insert(0, PROT); import evaluator as E
ZA, ZB = T["Z_A"], T["Z_B"]; r = E.gate_test(ZA, ZB, np.inf)
R["T5_gate_numbers"] = dict(delta=r["delta"], se=r["se"], df=r["df"], two_abs_delta=2 * abs(r["delta"]), two_se=2 * r["se"], U=r["U"], m_leak=T["m_leak"])
# counter-example without the 0.03 floor: Delta = SE -> U = SE (1 + t) > 2 SE = m_leak
rng = np.random.default_rng(0); fails = 0; nsim = 20000; fl = 0
for _ in range(nsim):
    a, b = rng.normal(0, 0.02, 5), rng.normal(0, 0.02, 5)
    mr = max(2 * abs(a.mean() - b.mean()), 2 * np.sqrt(a.var(ddof=1) / 5 + b.var(ddof=1) / 5))       # floor removed
    fails += E.gate_test(b, a, mr)["label"] == "FAIL"
    mr2 = max(0.03, mr); fl += E.gate_test(b, a, mr2)["label"] == "FAIL"
R["L1_fail_rate_rule_without_floor_sd0.02"] = fails / nsim
R["L1_fail_rate_rule_with_floor_sd0.02"] = fl / nsim
# 6. two_stratum: invariance to day order permutation applied to both arms, sensitivity to misalignment
YA = rng.normal(0, 1, (10, 9)) + np.linspace(0, 1, 9); YB = rng.normal(0, 1, (10, 9)) + np.linspace(0, 1, 9)
p = rng.permutation(9); s1 = E.two_stratum(YA, YB); s2 = E.two_stratum(YA[:, p], YB[:, p]); s3 = E.two_stratum(YA[:, p], YB)
R["two_stratum_perm_invariant"] = bool(np.isclose(s1["se"], s2["se"]) and np.isclose(s1["delta"], s2["delta"]))
R["two_stratum_misaligned_se_ratio"] = s3["se"] / s1["se"]
# 7. skill with one class present (degenerate day)
with np.errstate(all="ignore"):
    R["skill_single_class_day"] = str(m4.skill(np.array([1, 1, 1]), np.array([1, 0, 1])))
json.dump(R, open(os.path.join(HERE, "check_data.json"), "x"), indent=1, default=str)
print(json.dumps(R, indent=1, default=str))
