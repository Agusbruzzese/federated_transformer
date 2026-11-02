"""Reviewer F: tiny GPU tests of fedsim2 paths that the step-3 tests do not cover. All runs go to F_review/_runs
(fs.OUT is redirected; nothing under data/ is written). Output: F_review/test_review_fedsim_<stamp>.json (exclusive).
    .venv/bin/python research/model/K_committee/F_review/test_review_fedsim.py
  I1  --init (server ckpt) is really loaded: central (S+C) and every fed client (S+F) start from the checkpoint vector
  I2  fresh optimiser after init: AdamW state is empty at the first step of S+C and of every S+F client
  I3  fresh schedule after init: the first LRs of the client phase equal peak * lam(i, steps) from i = 0 (warm-up restart)
  I4  monitoring does not perturb training (eval_every 0 vs 1 -> bit-identical final weights, central K = 1 and fed)
  I5  determinism: same fed run twice -> bit-identical
  I6  full-length LR schedules (offline, plan + lam): warm-up fraction and end LR for every arm/fold
  I7  evaluator: scores a fedsim2 checkpoint (S0 path) and does NOT check the fold the model was trained on
"""
import hashlib, json, os, sys, time, shutil
import numpy as np, torch
ROOT = os.getcwd(); HERE = os.path.dirname(os.path.abspath(__file__)); PROT = os.path.join(os.path.dirname(HERE), "protocol")
sys.path.insert(0, PROT)
import fedsim2 as fs
STAMP = time.strftime("%Y%m%d_%H%M%S"); fs.OUT = os.path.join(HERE, "_runs_" + STAMP)
R = {}
H = lambda t: hashlib.sha256(t.detach().double().cpu().numpy().tobytes()).hexdigest()[:16]

# instrumentation: record LR and optimiser-state size at every step, and every load_vec start vector
TRACE = []
_orig_step = torch.optim.AdamW.step
def _step(self, *a, **k):
    TRACE.append(dict(id=id(self), lr=self.param_groups[0]["lr"], state=len(self.state)))
    return _orig_step(self, *a, **k)
torch.optim.AdamW.step = _step
STARTS = []; _orig_load = fs.load_vec
def _load(v, enc, head):
    _orig_load(v, enc, head); STARTS.append(fs.vec(enc, head))
fs.load_vec = _load

def run(tag, extra):
    TRACE.clear(); STARTS.clear()
    a = fs.parser().parse_args(["run", "--fold", "0", "--tag", tag] + extra)
    summ, enc, head = fs.run(a)
    return a, summ, fs.vec(enc, head), list(TRACE), list(STARTS)

# tiny server pretraining: 4 rounds x 2 steps x 256
a, s, vsrv, _, _ = run("srv", ["--arm", "server", "--batch", "256", "--rounds", "4", "--examples", str(8 * 256), "--eval-every", "0"])
ck = os.path.join(fs.OUT, "runs", "srv", "final.pt")
sd = torch.load(ck); enc0, head0 = fs.build(0); enc0.load_state_dict(sd["enc"]); head0.load_state_dict(sd["head"]); vck = fs.vec(enc0, head0)

# S+C: central, init
a, s, v, tr, st = run("spc", ["--arm", "central", "--batch", "256", "--rounds", "2", "--examples", str(20 * 256), "--eval-every", "0", "--init", ck, "--seed", "0"])
steps = a.examples // 256
R["I1_SpC_start_equals_ckpt"] = bool(torch.equal(st[0], vck))
R["I2_SpC_fresh_opt_state_at_step0"] = tr[0]["state"] == 0
exp = [a.lr * fs.lam(i, steps) for i in range(len(tr))]
R["I3_SpC_lr_trace_matches_fresh_schedule"] = bool(np.allclose([t["lr"] for t in tr], exp, rtol=0, atol=1e-15))
R["I3_SpC_first_lrs"] = [t["lr"] for t in tr[:3]]

# S+F: fed, init
a, s, v, tr, st = run("spf", ["--arm", "fed", "--kind", "natural", "--batch", "64", "--rounds", "2", "--examples", str(1280), "--eval-every", "0", "--init", ck])
K = s["plan"]["K"]; sk = s["plan"]["s"]
R["I1_SpF_all_clients_start_equal_ckpt_round0"] = bool(all(torch.equal(x, vck) for x in st[:K]))
first = {}
for t in tr: first.setdefault(t["id"], t)
R["I2_SpF_fresh_opt_state_each_client"] = bool(len(first) == K and all(t["state"] == 0 for t in first.values()))
steps = a.examples // 256; ok = True; i = 0
for r in range(2):
    for k in range(K):
        for j in range(sk[k]):
            ok &= abs(tr[i]["lr"] - a.lr * fs.lam((r * sk[k] + j) * steps / (2 * sk[k]), steps)) < 1e-15; i += 1
R["I3_SpF_lr_trace_matches_fresh_schedule"] = bool(ok and i == len(tr))
R["I2_SpF_moments_persist_round1"] = bool(all(t["state"] > 0 for t in tr[sum(sk):]))

# I4: monitoring does not perturb training
for arm in (["--arm", "central", "--batch", "256", "--rounds", "3", "--examples", str(6 * 256)],
            ["--arm", "fed", "--kind", "natural", "--batch", "64", "--rounds", "2", "--examples", "1280"]):
    nm = arm[1]
    _, _, v0, _, _ = run(f"mon0_{nm}", arm + ["--eval-every", "0"])
    _, _, v1, _, _ = run(f"mon1_{nm}", arm + ["--eval-every", "1"])
    R[f"I4_monitoring_no_effect_{nm}"] = bool(torch.equal(v0, v1))
# I5 determinism (fed, iid)
arm = ["--arm", "fed", "--kind", "iid", "--batch", "64", "--rounds", "2", "--examples", "1280", "--eval-every", "0"]
_, _, va, _, _ = run("det_a", arm); _, _, vb, _, _ = run("det_b", arm)
R["I5_fed_iid_bit_identical_rerun"] = bool(torch.equal(va, vb))
# I5b different seeds -> different weights (seed actually used) ; IID partition identical across seeds
_, _, vc, _, _ = run("det_c", arm + ["--seed", "1"]); R["I5_seed_changes_weights"] = not bool(torch.equal(va, vc))

# I6 full-length schedules (offline)
X, P = 10_000 * 256, 1_656 * 256
ARMS = {"C256": ("central", "natural", 256, 200, X), "C64": ("central", "natural", 64, 200, X), "F-nat": ("fed", "natural", 64, 200, X),
        "F-X+P": ("fed", "natural", 64, 233, X + P), "CJ": ("cj", "natural", 256, 188, X + P), "server": ("server", "natural", 256, 184, P)}
sch = {}
for f in ("0", "1", "2"):
    for nm, (arm, kind, b, Rr, Xx) in ARMS.items():
        plan, _, _ = fs.run(fs.parser().parse_args(["run", "--fold", f, "--plan-only", "--arm", arm, "--kind", kind, "--batch", str(b),
                                                   "--rounds", str(Rr), "--examples", str(Xx)]))
        stp = Xx // 256; out = []
        for k, skk in enumerate(plan["s"]):
            ii = np.array([(r * skk + j) * stp / (Rr * skk) for r in range(Rr) for j in range(skk)])
            lr = np.array([fs.lam(i, stp) for i in ii]); wu = int(0.05 * stp)
            out.append(dict(warm_frac=float((ii < wu).mean()), lr_last=float(lr[-1]), lr_max=float(lr.max())))
        sch[f"{nm}_f{f}"] = dict(warm_frac=[round(o["warm_frac"], 4) for o in out][:3], lr_last_max=max(o["lr_last"] for o in out),
                                lr_max_max=max(o["lr_max"] for o in out))
R["I6_schedules"] = sch
# default-flag hazard: server with the CLI default rounds (200) misses P
plan, _, _ = fs.run(fs.parser().parse_args(["run", "--fold", "0", "--plan-only", "--arm", "server", "--batch", "256", "--examples", str(P)]))
R["I6_server_default_rounds_examples"] = plan["examples_total"]
plan, _, _ = fs.run(fs.parser().parse_args(["run", "--fold", "0", "--plan-only", "--arm", "central", "--examples", str(X)]))
R["I6_central_default_batch"] = plan["b"]

# I7 evaluator on the tiny server checkpoint (S0 path), fold 0 and fold 1 (no provenance check)
import evaluator as E
o0 = E.score(ck, 0, do_gate=False); o1 = E.score(ck, 1, do_gate=False)
R["I7_S0_scores_fold0"] = o0["comp"].round(4).to_dict()
R["I7_evaluator_accepts_any_fold_for_any_ckpt"] = True
torch.optim.AdamW.step = _orig_step; fs.load_vec = _orig_load
json.dump(R, open(os.path.join(HERE, f"test_review_fedsim_{STAMP}.json"), "x"), indent=1, default=str)
print(json.dumps(R, indent=1, default=str))
