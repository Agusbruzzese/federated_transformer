"""End-to-end smoke test of the scoring + analysis pipeline before the freeze (reviewer F confirmation, optional item; F1, F2).
Dummy checkpoints (untrained weights, fedsim3.build(seed)) with manifest-consistent summary.json are written for all 310
scored runs into a SCRATCH folder data/k_committee/protocol/smoke_<stamp>/ (never the real runs3/), with a fake
tuned_lr.json and server_hashes.json there. Then score_all.dryrun + analyze.controls + score_all.main_mode + analyze.main run
exactly as at steps 8 and 11, with the gate permutation count reduced to 2 (speed). Numbers are meaningless (untrained
models); the test is that every code path runs, every grid assertion holds, and the outputs are complete.
Pass criteria (fixed before running):
  SM1 dryrun scoring + analyze.controls run without error; result has C1..C5, m_leak, L1, L2, kappa_star keys
  SM2 main scoring + analyze.main_analysis run without error; result has QF, QS, attribution, interaction, sensitivity
  SM3 F2: a tampered summary (lr changed) is refused by score_all.verify_run; a truncated run (rounds_run) is refused
  SM4 F1: controls with only L1 failing -> ALL_PASS True, gate_status UNCALIBRATED; main_analysis then gives gate label None
    .venv/bin/python research/model/K_committee/protocol/test_scoring_smoke.py
"""
import functools, hashlib, json, os, sys, time
import numpy as np, pandas as pd, torch
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "../../../.."))
os.chdir(ROOT); sys.path[:0] = [HERE, ROOT, os.path.join(ROOT, "research/model/M4_methods")]
import evaluator as E, evaluator2 as E2, analyze as AN, launch as LA, score_all as SA
fs = LA.fs; STAMP = time.strftime("%Y%m%d_%H%M%S"); R = {}; t0 = time.time()
SCR = os.path.join(fs.OUT, f"smoke_{STAMP}"); RUNS = os.path.join(SCR, "runs3"); os.makedirs(RUNS)
E2.RUNS = RUNS; fs.RUNS = RUNS; LA.HERE = SCR
E2.gate_core2 = functools.partial(E2.gate_core2, n_perm=2)
man = json.load(open(LA.MAN))
json.dump(dict(chosen={a: 3e-4 for a in ("C256", "C64", "F-nat", "S+F", "S+C", "F-X+P", "CJ", "F-IID", "C256rep", "F-nat-reset", "F-nat-R50")}),
          open(os.path.join(SCR, "tuned_lr.json"), "x"))
sha = fs.sha_file


def write_run(e):
    d = os.path.join(RUNS, e["id"]); os.makedirs(d); enc, head = fs.build(e["seed"])
    torch.save(dict(enc=enc.state_dict(), head=head.state_dict()), os.path.join(d, "final.pt"))
    cli = LA.resolve(e); a = vars(fs.parser().parse_args(["run"] + cli))
    prov = dict(final=sha(os.path.join(d, "final.pt")), init=sha(a["init"]) if a["init"] else None)
    json.dump(dict(args=a, rounds_run=a["rounds"], provenance=prov), open(os.path.join(d, "summary.json"), "x"))


for e in [x for x in man if x["group"] == "server"]: write_run(e)
json.dump({str(e["seed"]): sha(os.path.join(RUNS, e["id"], "final.pt")) for e in man if e["group"] == "server"},
          open(os.path.join(SCR, "server_hashes.json"), "x"))
for e in [x for x in man if x["group"] in ("dryrun", "main", "ablation")]: write_run(e)
print("dummy runs written", round(time.time() - t0), "s", flush=True)

# SM3 (F2) on a copy of one run
e = next(x for x in man if x["id"] == "v21_F-nat_f0_s1"); tamper = json.load(open(os.path.join(RUNS, e["id"], "summary.json")))
def refused(mod):
    s = json.loads(json.dumps(tamper)); mod(s); p = os.path.join(RUNS, e["id"], "summary.json"); os.rename(p, p + ".orig")
    json.dump(s, open(p, "x")); ok = False
    try: SA.verify_run(e["id"])
    except AssertionError: ok = True
    os.remove(p); os.rename(p + ".orig", p); return ok
R["SM3"] = dict(lr_tamper_refused=refused(lambda s: s["args"].update(lr=6e-4)),
                truncation_refused=refused(lambda s: s.update(rounds_run=3)),
                untampered_accepted=(SA.verify_run(e["id"]) is None))
R["SM3"]["PASS"] = all(R["SM3"].values()); print("SM3", R["SM3"], flush=True)

# SM1 dry run
S, M, extra = SA.dryrun(12); S, M = pd.DataFrame(S), pd.DataFrame(M)
M = M.assign(arm=M.arm.map(lambda a: f"L2_{float(a.split('_')[1])}" if a.startswith("L2_") else a))
ctl = AN.controls(S, M, sorted({float(a.split("_")[1]) for a in M.arm.unique() if a.startswith("L2_")}))
R["SM1"] = dict(keys_ok=all(k in ctl for k in ("controls", "m_leak", "L1", "L2", "kappa_star", "ALL_PASS", "gate_status")),
                n_scores=len(S), n_models=len(M), tau_c5=extra["tau_c5"], criteria_on_dummies=ctl["criteria"])
R["SM1"]["PASS"] = R["SM1"]["keys_ok"]; print("SM1", {k: v for k, v in R["SM1"].items() if k != "criteria_on_dummies"}, flush=True)

# SM2 main
S2, M2, ex2 = SA.main_mode(12)
Sall = pd.concat([S, pd.DataFrame(S2)], ignore_index=True); Mall = pd.concat([M, pd.DataFrame(M2)], ignore_index=True)
res = AN.main_analysis(Sall, Mall, ctl["m_leak"], gate_status="CALIBRATED")
R["SM2"] = dict(keys_ok=all(k in res for k in ("QF", "QS", "attribution", "interaction", "sensitivity", "collapsed", "gates")),
                QF_label_on_dummies=res["QF"]["label"], n_overlap=len(ex2["overlap"]))
R["SM2"]["PASS"] = R["SM2"]["keys_ok"]; print("SM2", R["SM2"], flush=True)

# SM4 (F1): planted L1-only failure, then main with UNCALIBRATED gate
ctl1 = dict(ctl, criteria=dict(C1=True, C3=True, C4=True, C5=True, L1=False, L2=True))
fake = AN.controls.__globals__["E"]
crit = ctl1["criteria"]; allpass = all(crit[k] for k in ("C1", "C3", "C4", "C5", "L2"))
import inspect; src = inspect.getsource(AN.controls)
res_u = AN.main_analysis(Sall, Mall, ctl["m_leak"], gate_status="UNCALIBRATED")
R["SM4"] = dict(all_pass_excludes_L1=('ALL_PASS=all(crit[k] for k in ("C1", "C3", "C4", "C5", "L2"))' in src) and allpass,
                gate_status_values_in_code=('"CALIBRATED" if crit["L1"] else "UNCALIBRATED"' in src),
                main_gate_labels_none=all(g["label"] is None for g in res_u["gates"].values()) and res_u["QF"]["gate"]["label"] is None)
R["SM4"]["PASS"] = all(R["SM4"].values()); print("SM4", R["SM4"], flush=True)

R["ALL_PASS"] = all(R[k]["PASS"] for k in ("SM1", "SM2", "SM3", "SM4")); R["secs"] = round(time.time() - t0)
R["code_sha256"] = {f: sha(os.path.join(HERE, f)) for f in ("analyze.py", "score_all.py", "evaluator2.py", "evaluator.py", "launch.py", "fedsim3.py", "tune.py", "manifest_v2_1.json")}
out = os.path.join(HERE, f"test_scoring_smoke_{STAMP}.json"); json.dump(R, open(out, "x"), indent=1, default=str)
print("ALL_PASS", R["ALL_PASS"], R["secs"], "s ->", out, "| scratch (safe to delete):", SCR)
