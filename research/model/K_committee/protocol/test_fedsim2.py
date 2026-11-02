"""Step 3 tests for fedsim2 (protocol v2). Outputs: protocol/test_fedsim2_<stamp>.json (never overwritten).
  G   guard: the pooled ssl_central/bgpool.npy cannot be loaded (protocol path)
  P   partitions: per fold, natural and IID clients have identical sizes, every client event once, 10 clients
  W   firewall: every arm x fold passes; a planted forbidden day raises (negative test)
  A   accounting: examples, gradient steps, epochs per event for every arm x fold (plan-only, no training)
  C0  FedAvg proofs on fedsim2 (identical starts; reversal invariance; weighted mean)
  S   smoke: 1 round of every arm on fold 0 (server -> S+F, S+C with --init), monitoring on Feb 10 own pool
    .venv/bin/python research/model/K_committee/protocol/test_fedsim2.py
"""
import hashlib, json, os, sys, time
import numpy as np, torch
ROOT = os.getcwd(); HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import fedsim2 as fs
STAMP = time.strftime("%Y%m%d_%H%M%S"); X, P = 10_000 * 256, 1_656 * 256
ARMS = {  # protocol v2 section 3: name -> CLI
    "C256": ["--arm", "central", "--batch", "256", "--rounds", "200", "--examples", str(X)],
    "C64": ["--arm", "central", "--batch", "64", "--rounds", "200", "--examples", str(X)],
    "F-nat": ["--arm", "fed", "--kind", "natural", "--batch", "64", "--rounds", "200", "--examples", str(X)],
    "F-IID": ["--arm", "fed", "--kind", "iid", "--batch", "64", "--rounds", "200", "--examples", str(X)],
    "F-X+P": ["--arm", "fed", "--kind", "natural", "--batch", "64", "--rounds", "233", "--examples", str(X + P)],
    "CJ": ["--arm", "cj", "--batch", "256", "--rounds", "188", "--examples", str(X + P)],
    "server": ["--arm", "server", "--batch", "256", "--rounds", "184", "--examples", str(P)],
}
R = {}
# G
try:
    fs.GUARD["on"] = True; np.load(os.path.join(ROOT, "data", "ssl_central", "bgpool.npy")); R["G_guard_blocks_pooled_noise"] = False
except RuntimeError:
    R["G_guard_blocks_pooled_noise"] = True
# P + W(negative)
Pres = {}
for f in (0, 1, 2):
    m, nat, roles, day = fs.partition(f, "natural"); _, iid, _, _ = fs.partition(f, "iid")
    U = np.concatenate(nat); Ui = np.concatenate(iid)
    Pres[f] = dict(K=len(nat), sizes=[len(c) for c in nat], iid_sizes_equal=[len(c) for c in nat] == [len(c) for c in iid],
                   each_event_once=bool(len(U) == len(set(U)) and len(Ui) == len(set(Ui)) and set(U) == set(Ui)),
                   iid_client_days_mixed=float(np.mean([len(set(day[c])) > 1 for c in iid])))
R["P_partitions"] = Pres
try:
    m, nat, roles, day = fs.partition(0, "natural"); bad = np.flatnonzero(day == roles["eval"][0])[:3]
    fs.firewall(day, np.concatenate([nat[0], bad]), roles["eval"], "planted"); R["W_negative_test_raises"] = False
except AssertionError:
    R["W_negative_test_raises"] = True
# A (plan-only; also runs every firewall)
acc = []
for f in ("0", "1", "2"):
    for name, cli in ARMS.items():
        plan, _, _ = fs.run(fs.parser().parse_args(["run", "--fold", f, "--plan-only"] + cli))
        n = np.array(plan["n"]); ep = np.array(plan["s"]) * np.array(plan["b"]) * fs.parser().parse_args(["run"] + cli).rounds / n
        acc.append(dict(fold=f, arm=name, K=plan["K"], events=plan["n_events"], examples=plan["examples_total"],
                        ratio_to_budget=round(plan["examples_total"] / plan["budget"], 4), grad_steps=plan["grad_steps_total"],
                        epochs_min=round(float(ep.min()), 1), epochs_max=round(float(ep.max()), 1)))
R["A_accounting"] = acc; R["W_all_arm_fold_firewalls_pass"] = True
print(json.dumps({k: v for k, v in R.items() if k != "A_accounting"}, indent=1), flush=True)
for r in acc: print(r, flush=True)
# C0 on fedsim2
H = lambda t: hashlib.sha256(t.detach().double().cpu().numpy().tobytes()).hexdigest()[:16]
orig_load, orig_vec = fs.load_vec, fs.vec
def go(reverse):
    starts, ends, svecs = [], [], []
    def load(v, enc, head):
        orig_load(v, enc, head); x = orig_vec(enc, head); starts.append(H(x)); svecs.append(x)
    def vec(enc, head):
        out = orig_vec(enc, head); ends.append(out); return out
    fs.load_vec, fs.vec = load, vec
    args = ["run", "--fold", "0", "--arm", "fed", "--kind", "natural", "--seed", "0", "--rounds", "2", "--examples", str(640 * 2),
            "--batch", "64", "--eval-every", "0", "--tag", f"test_c0_rev{int(reverse)}_{STAMP}"] + (["--reverse-clients"] if reverse else [])
    summ, enc, head = fs.run(fs.parser().parse_args(args)); fs.load_vec, fs.vec = orig_load, orig_vec
    return starts, ends, svecs, orig_vec(enc, head), summ["plan"]
s0, e0, v0, g0, plan = go(False); s1, e1, _, g1, _ = go(True); K = plan["K"]
w = torch.tensor(plan["b"], dtype=torch.float64) * torch.tensor(plan["s"], dtype=torch.float64); w = w / w.sum()
cl0 = e0[1:1 + K]; cl1 = e1[1:1 + K][::-1]
R["C0a_identical_starts"] = [len(set(s0[r * K:(r + 1) * K])) == 1 for r in range(2)]
R["C0b_reversal_bit_identical"] = bool(all(torch.equal(a, b) for a, b in zip(cl0, cl1)) and torch.equal(g0, g1))
R["C0c_weighted_mean_maxabs"] = float((sum(w[k] * cl0[k].double() for k in range(K)) - v0[K].double()).abs().max())
R["C0_PASS"] = all(R["C0a_identical_starts"]) and R["C0b_reversal_bit_identical"] and R["C0c_weighted_mean_maxabs"] < 1e-5
print({k: R[k] for k in R if k.startswith("C0")}, flush=True)
# S smoke: 1 round each, fold 0
smoke = {}; srv_tag = f"test_smoke_server_{STAMP}"
for name, cli in [("server", ARMS["server"])] + [(k, v) for k, v in ARMS.items() if k != "server"] + \
        [("S+F", ARMS["F-nat"] + ["--init", None]), ("S+C", ARMS["C256"] + ["--init", None])]:
    cli = [c if c is not None else os.path.join(fs.OUT, "runs", srv_tag, "final.pt") for c in cli]
    tag = srv_tag if name == "server" else f"test_smoke_{name.replace('+', 'p')}_{STAMP}"
    t0 = time.time(); summ, _, _ = fs.run(fs.parser().parse_args(["run", "--fold", "0", "--max-rounds", "1", "--eval-every", "1", "--tag", tag] + cli))
    rec = json.loads(open(os.path.join(fs.OUT, "runs", tag, "rounds.jsonl")).readline())
    smoke[name] = dict(secs=round(time.time() - t0, 1), round_secs=round(rec["secs"], 2), loss=round(rec["loss"], 4),
                       val_loss=round(rec.get("val_loss", float("nan")), 4), erank=round(rec.get("erank", float("nan")), 2),
                       finite=bool(np.isfinite(rec["loss"])), init=summ["provenance"]["init"])
R["S_smoke"] = smoke; R["S_PASS"] = all(v["finite"] for v in smoke.values()) and smoke["S+F"]["init"] is not None
R["ALL_PASS"] = R["G_guard_blocks_pooled_noise"] and R["W_negative_test_raises"] and R["C0_PASS"] and R["S_PASS"] and \
    all(v["iid_sizes_equal"] and v["each_event_once"] and v["K"] == 10 for v in Pres.values())
out = os.path.join(HERE, f"test_fedsim2_{STAMP}.json"); json.dump(R, open(out, "x"), indent=1, default=str)
print(json.dumps(smoke, indent=1)); print("ALL_PASS", R["ALL_PASS"], "->", out)
