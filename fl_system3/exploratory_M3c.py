"""EXPLORATORY (not pre-registered) follow-up to the M3c failure (report 33): is the SecAgg+ effect larger than ordinary
training variability, and is it within the study's equivalence margins (LOBES 0.04, POLARITY 0.02)?
Uses fold-0 evaluation days (3) x seeds 0-2, per-day scores from the frozen evaluator."""
import json, os, sys, time
import numpy as np, torch
R = os.getcwd(); sys.path[:0] = [R + "/research/model/K_committee/protocol", R, R + "/research/model/M4_methods"]
import evaluator as E, evaluator2 as E2, analyze as AN
def per_day(path):
    rows, _ = E2.score(E.load_encoder(path), 0, do_gate=False); out = {}
    for r in rows: out.setdefault(r["metric"], {})[r["day"]] = r["value"]
    return out
S = {s: per_day(f"{R}/data/fl_system/fold0_v2/server/runs3/m3c_SpF_s{s}/final.pt") for s in (0, 1, 2)}
P = {s: per_day(f"{R}/data/k_committee/protocol/runs3/v21_S+F_f0_s{s}/final.pt") for s in (0, 1, 2)}
days = sorted(P[0]["LOBES"]); res = dict(note="EXPLORATORY, not pre-registered", days=days, metrics={})
for k, m in (("LOBES", 0.04), ("POLARITY", 0.02), ("DURATION", 0.03)):
    YA = np.array([[S[s][k][d] for d in days] for s in (0, 1, 2)]); YB = np.array([[P[s][k][d] for d in days] for s in (0, 1, 2)])
    pr = AN.paired(YA, YB); ts = E.two_stratum(YA, YB)
    seed_means_B = YB.mean(1); sd_seed = float(seed_means_B.std(ddof=1))                     # natural seed variability (study S+F, fold 0)
    res["metrics"][k] = dict(margin=m, mean_diff=pr["delta"], paired_CI90=[pr["L"], pr["U"]], paired_outcome=E.outcome(pr["L"], pr["U"], m),
                             two_stratum_CI90=[ts["L"], ts["U"]], per_seed_diff=(YA.mean(1) - YB.mean(1)).round(4).tolist(),
                             study_seed_sd_fold0=sd_seed, max_abs_seed_diff_over_seed_sd=float(np.abs(YA.mean(1) - YB.mean(1)).max() / sd_seed))
out = time.strftime(R + "/fl_system3/exploratory_M3c_%Y%m%d_%H%M%S.json"); json.dump(res, open(out, "x"), indent=1); print(json.dumps(res, indent=1), "->", out)
