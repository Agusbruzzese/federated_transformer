"""T1 (report 25): the real federated system computes what the frozen simulator fedsim3 computed.
Runs fedsim3 (study env .venv, outputs redirected to data/fl_system/tests/) with the configuration of a real-system run, records
the global vector at the start of every round (client 0's load) and the final weights, and compares them with the real
system's saved globals (global_r001..r003.pt) and final.pt.  Criterion: max |diff| <= 1e-5 for every round.
    .venv/bin/python fl_system/tests/test_T1_equivalence.py <real_run_dir> <rounds> <examples> <lr>
"""
import json, os, sys, time
import numpy as np, torch
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); os.chdir(ROOT)
sys.path[:0] = [os.path.join(ROOT, "research/model/K_committee/protocol"), ROOT]
import fedsim3 as fs
run_dir, R, X, lr = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4]); stamp = time.strftime("%Y%m%d_%H%M%S")
fs.RUNS = os.path.join(ROOT, "data/fl_system/tests", f"fedsim_ref_{stamp}")
starts = []; orig = fs.load_vec
def hook(v, enc, head):
    orig(v, enc, head)
    if len(starts) % 10 == 0: pass
    starts.append(v.detach().clone())
fs.load_vec = hook
init = os.path.join(ROOT, "data/k_committee/protocol/runs3/v21_server_s0/final.pt")
a = fs.parser().parse_args(["run", "--arm", "fed", "--fold", "0", "--kind", "natural", "--seed", "0", "--rounds", str(R), "--batch", "64",
                            "--examples", str(X), "--lr", str(lr), "--init", init, "--eval-every", "0", "--tag", f"T1_ref_{stamp}"])
summ, enc, head = fs.run(a); K = summ["plan"]["K"]
glob_after = [starts[K * r] for r in range(1, R)] + [fs.vec(enc, head)]       # global after round r = start of round r+1; last = final
res = dict(rounds=R, examples=X, lr=lr, per_round_maxabs=[], fedsim_plan_s=summ["plan"]["s"])
for r in range(1, R + 1):
    real = torch.load(os.path.join(run_dir, f"global_r{r:03d}.pt")).to(glob_after[r - 1].device)
    res["per_round_maxabs"].append(float((real - glob_after[r - 1]).abs().max()))
sd = torch.load(os.path.join(run_dir, "final.pt"))
real_final = torch.cat([v.reshape(-1) for v in list(sd["enc"].values()) + list(sd["head"].values())]).to(glob_after[-1].device)
res["final_maxabs"] = float((real_final - glob_after[-1]).abs().max())
res["bit_identical"] = all(x == 0.0 for x in res["per_round_maxabs"]) and res["final_maxabs"] == 0.0
res["PASS"] = max(res["per_round_maxabs"] + [res["final_maxabs"]]) <= 1e-5
out = os.path.join(ROOT, "fl_system/tests", f"T1_{stamp}.json"); json.dump(res, open(out, "x"), indent=1); print(json.dumps(res), "->", out)
