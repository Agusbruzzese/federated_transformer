"""M3c' (report 34, PRE-REGISTERED; this file's sha256 is recorded in report 34 before any M3c' run starts).
Question: is S+F trained with SecAgg+ EQUIVALENT to plain S+F (the study models, = plain fl_system2 bit for bit, report 31)?
PRIMARY:   seeds 3-9 (not seen before), fold 0, its 3 evaluation days, frozen evaluator; seed-paired two-stratum t (analyze.paired),
           90 % CI; margins LOBES 0.04, POLARITY 0.02 (study); label by evaluator.contrast_label (EQUIVALENT only if both EQ).
SECONDARY: same on seeds 0-9 (0-2 already seen in M3c); DURATION (margin 0.03); unpaired two_stratum as a sensitivity check.
VALIDITY:  every SecAgg+ run has 200 committed rounds and 0 clipped values; otherwise the analysis refuses (INVALID)."""
import json, os, sys, time
import numpy as np
R = os.getcwd(); sys.path[:0] = [R + "/research/model/K_committee/protocol", R, R + "/research/model/M4_methods"]
import evaluator as E, evaluator2 as E2, analyze as AN
MARG = dict(LOBES=0.04, POLARITY=0.02, DURATION=0.03); PRIM, ALL = range(3, 10), range(10)
SEC = lambda s: f"{R}/data/fl_system/fold0_v2/server/runs3/" + (f"m3c_SpF_s{s}" if s < 3 else f"m3cp_SpF_s{s}")
PLAIN = lambda s: f"{R}/data/k_committee/protocol/runs3/v21_S+F_f0_s{s}"
def per_day(path):
    rows, _ = E2.score(E.load_encoder(path + "/final.pt"), 0, do_gate=False); out = {}
    for r in rows: out.setdefault(r["metric"], {})[r["day"]] = r["value"]
    return out
valid = {}
for s in ALL:
    recs = [json.loads(l) for l in open(SEC(s) + "/rounds.jsonl")]
    valid[s] = dict(rounds=len({r["round"] for r in recs}), over_clip=sum(r["n_over_clip"] for r in recs), final=os.path.exists(SEC(s) + "/final.pt"))
assert all(v["rounds"] == 200 and v["over_clip"] == 0 and v["final"] for v in valid.values()), ("INVALID", valid)
A = {s: per_day(SEC(s)) for s in ALL}; B = {s: per_day(PLAIN(s)) for s in ALL}; days = sorted(B[0]["LOBES"])
def contrast(seeds):
    out = {}
    for k, m in MARG.items():
        YA = np.array([[A[s][k][d] for d in days] for s in seeds]); YB = np.array([[B[s][k][d] for d in days] for s in seeds])
        pr = AN.paired(YA, YB); ts = E.two_stratum(YA, YB)
        out[k] = dict(margin=m, delta=pr["delta"], CI90=[pr["L"], pr["U"]], outcome=E.outcome(pr["L"], pr["U"], m),
                      unpaired_CI90=[ts["L"], ts["U"]], unpaired_outcome=E.outcome(ts["L"], ts["U"], m),
                      per_seed_diff=(YA.mean(1) - YB.mean(1)).round(5).tolist())
    out["label"] = E.contrast_label({k: out[k]["outcome"] for k in ("LOBES", "POLARITY")}); return out
res = dict(prereg="report 34", days=days, validity=valid, primary_seeds_3_9=contrast(PRIM), secondary_seeds_0_9=contrast(ALL))
p = time.strftime(R + "/fl_system3/results_M3cp_%Y%m%d_%H%M%S.json"); json.dump(res, open(p, "x"), indent=1)
print("PRIMARY", res["primary_seeds_3_9"]["label"], "SECONDARY", res["secondary_seeds_0_9"]["label"], "->", p)
