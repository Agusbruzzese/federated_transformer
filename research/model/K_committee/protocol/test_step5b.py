"""Step 5b tests (report 15). Pass criteria fixed here before the first run; output test_step5b_<stamp>.json (never overwritten)
with the sha256 of every protocol file tested.
  AN  analyze.py on toy tidy data with known answers: misalignment/incompleteness raises; row order irrelevant; Q-F, Q-S,
      attributions, interaction, seed-paired formula, collapse, controls() give the known labels
  TU  tune.select rule
  MA  manifest: 338 unique entries, group sizes, every placeholder resolvable; launcher refuses S arms before servers are
      pinned and tuned arms before tuned_lr.json exists
  FS  fedsim3: C0 (FedAvg), C0' (bit-identical to m5.train), lr trace = peak * lam, guard refuses a renamed copy of the
      pooled noise (content), pinned-hash assertion fires on mismatch, required CLI arguments
  EV  evaluator2: refuses to score a checkpoint on another fold; skill refuses a single-class day
    .venv/bin/python research/model/K_committee/protocol/test_step5b.py
"""
import glob, hashlib, json, os, shutil, sys, time
import numpy as np, pandas as pd, torch
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "../../../.."))
os.chdir(ROOT); sys.path[:0] = [HERE, ROOT, os.path.join(ROOT, "research/model/M4_methods")]
import evaluator as E, evaluator2 as E2, analyze as AN, tune as TU, manifest as MF, launch as LA, fedsim3 as fs
STAMP = time.strftime("%Y%m%d_%H%M%S"); R = {}
def raises(fn, exc=Exception):
    try: fn(); return False
    except exc: return True

# ------------------------------------------------------------------ AN
rng = np.random.default_rng(0); BASE = {d: dict(LOBES=0.6 + rng.normal(0, .05), POLARITY=0.9 + rng.normal(0, .01), DURATION=0.85) for d in AN.DAYS}
def toy(effects, seeds=AN.MAIN_SEEDS, noise=0.003):
    rows = []
    for arm, eff in effects.items():
        for s in (seeds if arm not in ("C256rep", "C4", "C5") else AN.REP_SEEDS):
            for d in AN.DAYS:
                for k in ("LOBES", "POLARITY", "DURATION"):
                    rows.append(dict(arm=arm, seed=s, fold=AN.FOLD_OF_DAY[d], day=d, metric=k, value=BASE[d][k] + eff.get(k, 0) + rng.normal(0, noise)))
    return pd.DataFrame(rows)
def toy_models(arms, z=0.0, erank=20.0, seeds=AN.MAIN_SEEDS, override=None):
    rows = []
    for a in arms:
        for s in (seeds if a != "C256rep" else AN.REP_SEEDS):
            for f in (0, 1, 2):
                rows.append(dict(arm=a, seed=s, fold=f, erank=erank, Z=z + rng.normal(0, .003), Q=0.0, Q_null_q95=0.05))
    m = pd.DataFrame(rows)
    if override: m.loc[(m.arm == override[0]) & (m.seed == override[1]), "erank"] = override[2]
    return m
ARMS0 = {a: {} for a in ("C256", "C64", "F-IID", "F-nat", "S+F", "S+C", "F-X+P", "CJ")}
S0 = toy(ARMS0); M0 = toy_models(list(ARMS0))
an = {}
an["QF_equivalent"] = AN.main_analysis(S0, M0, 0.03)["QF"]["label"] == "EQUIVALENT"
an["row_order_irrelevant"] = AN.contrast(S0.sample(frac=1, random_state=1), "F-nat", "C256", AN.MAIN_SEEDS, AN.MAIN_SEEDS)["metrics"]["LOBES"]["se"] == \
    AN.contrast(S0, "F-nat", "C256", AN.MAIN_SEEDS, AN.MAIN_SEEDS)["metrics"]["LOBES"]["se"]
bad = S0.copy(); i = bad.index[(bad.arm == "F-nat") & (bad.day == AN.DAYS[0])][0]; bad.loc[i, "fold"] = 2
an["wrong_fold_raises"] = raises(lambda: AN.grid(bad, "F-nat", "LOBES", AN.MAIN_SEEDS))
an["missing_day_raises"] = raises(lambda: AN.grid(S0[~((S0.arm == "F-nat") & (S0.day == AN.DAYS[4]))], "F-nat", "LOBES", AN.MAIN_SEEDS))
an["duplicate_raises"] = raises(lambda: AN.grid(pd.concat([S0, S0.iloc[:1]]), S0.iloc[0].arm, S0.iloc[0].metric, AN.MAIN_SEEDS))
an["missing_seed_raises"] = raises(lambda: AN.grid(S0[~((S0.arm == "F-nat") & (S0.seed == 3))], "F-nat", "LOBES", AN.MAIN_SEEDS))
an["gate_incomplete_raises"] = raises(lambda: AN.zbar(M0[~((M0.arm == "F-nat") & (M0.fold == 1) & (M0.seed == 2))], "F-nat", AN.MAIN_SEEDS))
S1 = toy(dict(ARMS0, **{"F-nat": dict(LOBES=-0.10)})); r1 = AN.main_analysis(S1, M0, 0.03)
an["QF_inferior"] = r1["QF"]["label"] == "INFERIOR"
an["attr_non_iid"] = r1["attribution"]["LOBES"] == dict(applies=True, fewer_negatives=False, averaging=False, non_iid=True)
S2 = toy(dict(ARMS0, **{"C64": dict(LOBES=-0.10), "F-IID": dict(LOBES=-0.10), "F-nat": dict(LOBES=-0.10)})); r2 = AN.main_analysis(S2, M0, 0.03)
an["attr_fewer_negatives"] = r2["attribution"]["LOBES"] == dict(applies=True, fewer_negatives=True, averaging=False, non_iid=False)
S3 = toy(dict(ARMS0, **{"F-IID": dict(LOBES=-0.10), "F-nat": dict(LOBES=-0.10)})); r3 = AN.main_analysis(S3, M0, 0.03)
an["attr_averaging"] = r3["attribution"]["LOBES"] == dict(applies=True, fewer_negatives=False, averaging=True, non_iid=False)
S4 = toy(dict(ARMS0, **{"S+F": dict(LOBES=0.05)})); r4 = AN.main_analysis(S4, M0, 0.03)
an["QS_helps"] = r4["QS"]["verdict"] == "HELPS"; an["QS_does_not_help"] = AN.main_analysis(S0, M0, 0.03)["QS"]["verdict"] == "DOES NOT HELP"
an["interaction_~0.05"] = abs(r4["interaction"]["LOBES"]["delta"] - 0.05) < 0.005
S5 = toy(dict(ARMS0, **{"S+F": dict(LOBES=0.05), "S+C": dict(LOBES=0.05)}))
an["interaction_~0"] = abs(AN.main_analysis(S5, M0, 0.03)["interaction"]["LOBES"]["delta"]) < 0.005
YA, YB = AN.grid(S1, "F-nat", "LOBES", AN.MAIN_SEEDS), AN.grid(S1, "C256", "LOBES", AN.MAIN_SEEDS); p = AN.paired(YA, YB)
dd = YA.mean(0) - YB.mean(0); se_direct = np.sqrt(np.var(YA.mean(1) - YB.mean(1), ddof=1) / 10 + np.var(dd - dd.mean(), ddof=1) / 9)
an["paired_formula"] = abs(p["se"] - se_direct) < 1e-12 and abs(p["delta"] - dd.mean()) < 1e-12
an["collapse"] = AN.collapsed(toy_models(list(ARMS0), override=("F-nat", 3, 5.0)), "F-nat", AN.MAIN_SEEDS) == [3]
an["leak_gate_fail"] = AN.gate(toy_models(["C256"]).assign(Z=0.0).pipe(lambda m: pd.concat([m, m.assign(arm="F-nat", Z=0.08)])),
                               "F-nat", "C256", AN.MAIN_SEEDS, AN.MAIN_SEEDS, 0.03)["label"] == "FAIL"
SC = toy(dict(C256={}, C256rep={}, INIT=dict(LOBES=-0.3, POLARITY=-0.15), C4=dict(LOBES=-0.11, POLARITY=-0.03), C5=dict(LOBES=-0.05)))
MC = toy_models(["C256", "C256rep"])
MC = pd.concat([MC] + [toy_models(["C256rep"], z=zk).assign(arm=f"L2_{k}", Q=qk, Q_null_q95=0.05)
                       for k, zk, qk in ((0.01, 0.0, 0.0), (0.03, 0.06, 0.08))])
c = AN.controls(SC, MC, [0.01, 0.03])
an["controls_known"] = c["criteria"] == dict(C1=True, C3=True, C4=True, C5=True, L1=True, L2=True) and c["kappa_star"] == 0.03
R["AN"] = dict(an, PASS=all(an.values())); print("AN", R["AN"], flush=True)

# ------------------------------------------------------------------ TU
tu = dict(keep_default=TU.select({3e-4: .70, 6e-4: .71, 1e-3: .69, 1.5e-4: .60}) == 3e-4,
          best_of_several=TU.select({3e-4: .70, 6e-4: .73, 1e-3: .75, 1.5e-4: .60}) == 1e-3,
          boundary_not_enough=TU.select({3e-4: .70, 6e-4: .72}) == 3e-4)
R["TU"] = dict(tu, PASS=all(tu.values())); print("TU", R["TU"], flush=True)

# ------------------------------------------------------------------ MA
man = json.load(open(os.path.join(HERE, "manifest_v2_1.json"))); from collections import Counter
g = Counter(e["group"] for e in man)
ph = {e["cli"][e["cli"].index("--lr") + 1].split(":")[1] for e in man if "TUNED:" in e["cli"][e["cli"].index("--lr") + 1]}
ma = dict(n=len(man) == 338, unique=len({e["id"] for e in man}) == 338, groups=dict(g) == dict(server=10, tuning=28, dryrun=60, main=210, ablation=30),
          regenerates_identically=MF.build() == man, placeholders_resolvable=ph <= set(MF.TUNED),
          server_inits_seed_matched=all(e["init"] in (None, f"SERVER:{e['seed']}", "SERVER:0") for e in man),
          refuses_S_before_pins=raises(lambda: LA.resolve(next(e for e in man if e["arm"] == "S+F" and e["group"] == "main"))),
          refuses_tuned_before_tuning=raises(lambda: LA.resolve(next(e for e in man if e["group"] == "main" and e["arm"] == "C64"))))
R["MA"] = dict(ma, PASS=all(ma.values())); print("MA", R["MA"], flush=True)

# ------------------------------------------------------------------ FS
fsr = {}
H = lambda t: hashlib.sha256(t.detach().double().cpu().numpy().tobytes()).hexdigest()[:16]
orig_load, orig_vec = fs.load_vec, fs.vec
def go(rev):
    starts, ends, svecs = [], [], []
    def load(v, enc, head): orig_load(v, enc, head); x = orig_vec(enc, head); starts.append(H(x)); svecs.append(x)
    def vec(enc, head): o = orig_vec(enc, head); ends.append(o); return o
    fs.load_vec, fs.vec = load, vec
    a = ["run", "--arm", "fed", "--fold", "0", "--kind", "natural", "--seed", "0", "--rounds", "2", "--examples", "1280", "--batch", "64",
         "--lr", "3e-4", "--eval-every", "0", "--tag", f"test5b_c0_rev{int(rev)}_{STAMP}"] + (["--reverse-clients"] if rev else [])
    summ, enc, head = fs.run(fs.parser().parse_args(a)); fs.load_vec, fs.vec = orig_load, orig_vec
    return starts, ends, svecs, orig_vec(enc, head), summ
s0, e0, v0, g0, sm = go(False); s1, e1, _, g1, _ = go(True); K = sm["plan"]["K"]
w = torch.tensor(sm["plan"]["b"], dtype=torch.float64) * torch.tensor(sm["plan"]["s"], dtype=torch.float64); w /= w.sum()
fsr["C0a"] = all(len(set(s0[r * K:(r + 1) * K])) == 1 for r in range(2))
fsr["C0b"] = bool(all(torch.equal(a, b) for a, b in zip(e0[1:1 + K], e1[1:1 + K][::-1])) and torch.equal(g0, g1))
fsr["C0c"] = float((sum(w[k] * e0[1 + k].double() for k in range(K)) - v0[K].double()).abs().max()) < 1e-5
fsr["final_hash_recorded"] = sm["provenance"]["final"] is not None and sm["provenance"]["fedsim3"] is not None
rec = [json.loads(l) for l in open(os.path.join(fs.RUNS, f"test5b_c0_rev0_{STAMP}", "rounds.jsonl"))]
fsr["lr_trace_matches_schedule"] = all(abs(c["lr_first"] - 3e-4 * fs.lam(c["i_eq_first"], 1280 // 256)) < 1e-15 for r in rec for c in r["clients"])
tmp = os.path.join(HERE, f"_tmp_{STAMP}"); os.makedirs(tmp); shutil.copy(fs.RAW_POOL, os.path.join(tmp, "renamed.npy"))
fs.GUARD["on"] = True; fsr["guard_by_content"] = raises(lambda: np.load(os.path.join(tmp, "renamed.npy")), RuntimeError); shutil.rmtree(tmp)
pin = fs.PIN_POOL; fs.PIN_POOL = "0" * 64; fsr["pool_pin_asserts"] = raises(lambda: fs.day_pool(["2025-02-05"]), AssertionError); fs.PIN_POOL = pin
fsr["required_args"] = raises(lambda: fs.parser().parse_args(["run", "--fold", "0"]), SystemExit)
st = fs.selftest(os.path.join(HERE, f"selftest_fedsim3_{STAMP}.json")); fsr["C0prime_bit_identical_m5"] = st["identical"]
R["FS"] = dict(fsr, PASS=all(fsr.values())); print("FS", R["FS"], flush=True)

# ------------------------------------------------------------------ EV
ev = dict(fold_mismatch_refused=raises(lambda: E2.score_run(f"test5b_c0_rev0_{STAMP}", 1), AssertionError),
          own_fold_accepted=not raises(lambda: E2.score_run(f"test5b_c0_rev0_{STAMP}", 0, do_gate=False)),
          single_class_refused=raises(lambda: E2.safe_metric("lobes", np.array([2, 2, 2]), np.array([2, 2, 1])), AssertionError))
R["EV"] = dict(ev, PASS=all(ev.values())); print("EV", R["EV"], flush=True)

R["ALL_PASS"] = all(R[k]["PASS"] for k in ("AN", "TU", "MA", "FS", "EV"))
R["code_sha256"] = {os.path.basename(f): hashlib.sha256(open(f, "rb").read()).hexdigest() for f in sorted(glob.glob(os.path.join(HERE, "*.py")) + [os.path.join(HERE, "folds.json"), os.path.join(HERE, "manifest_v2_1.json")])}
out = os.path.join(HERE, f"test_step5b_{STAMP}.json"); json.dump(R, open(out, "x"), indent=1, default=str); print("ALL_PASS", R["ALL_PASS"], "->", out)
