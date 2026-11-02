"""Reviewer F, confirmation check of step 5b (report 15). Independent recomputation; writes only
F_review/confirm_step5b_<stamp>.json (exclusive). GPU: model construction + a few embeddings only.
    .venv/bin/python research/model/K_committee/F_review/confirm_step5b.py
"""
import hashlib, json, os, sys, time
from collections import Counter
import numpy as np, pandas as pd, torch
from scipy import stats
ROOT = os.getcwd(); HERE = os.path.dirname(os.path.abspath(__file__)); PROT = os.path.join(os.path.dirname(HERE), "protocol")
sys.path[:0] = [PROT, ROOT, os.path.join(ROOT, "research/model/M4_methods")]
import evaluator as E, evaluator2 as E2, analyze as AN, manifest as MF, tune as TU
import fedsim3 as fs
STAMP = time.strftime("%Y%m%d_%H%M%S"); R = {}
sha = lambda f: hashlib.sha256(open(f, "rb").read()).hexdigest()
R["sha_now"] = {f: sha(os.path.join(PROT, f)) for f in ("fedsim3.py", "evaluator.py", "evaluator2.py", "analyze.py", "score_all.py",
                                                          "tune.py", "launch.py", "manifest.py", "manifest_v2_1.json", "folds.json")}
ref = json.load(open(os.path.join(PROT, "test_step5b_20261011_094902.json")))["code_sha256"]
R["sha_equal_to_tested"] = {f: R["sha_now"][f] == ref[f] for f in R["sha_now"]}

# ---- M: manifest vs report 14 A1
man = json.load(open(os.path.join(PROT, "manifest_v2_1.json")))
X, P = 10_000 * 256, 1_656 * 256
A1 = {"C256": ("central", "natural", 256, 200, X, None), "C64": ("central", "natural", 64, 200, X, None),
      "F-nat": ("fed", "natural", 64, 200, X, None), "F-IID": ("fed", "iid", 64, 200, X, None),
      "F-X+P": ("fed", "natural", 64, 233, X + P, None), "S+F": ("fed", "natural", 64, 200, X, "S"),
      "S+C": ("central", "natural", 256, 200, X, "S"), "CJ": ("cj", "natural", 256, 188, X + P, None),
      "server": ("server", "natural", 256, 184, P, None), "C256rep": ("central", "natural", 256, 200, X, None),
      "F-nat-reset": ("fed", "natural", 64, 200, X, None), "F-nat-R50": ("fed", "natural", 64, 50, X, None)}
LRSRC = {"C256": "C256", "C64": "C64", "F-nat": "F-nat", "F-IID": "F-nat", "F-X+P": "F-X+P", "S+F": "S+F", "S+C": "S+C", "CJ": "CJ",
         "C256rep": "C256", "F-nat-reset": "F-nat", "F-nat-R50": "F-nat"}
bad = []
for e in man:
    c = e["cli"]; g = lambda k: c[c.index(k) + 1]
    fa, kind, b, Rr, ex, srv = A1[e["arm"]]
    ok = (g("--arm"), g("--kind"), int(g("--batch")), int(g("--rounds")), int(g("--examples"))) == (fa, kind, b, Rr, ex)
    ok &= g("--seed") == str(e["seed"]) and g("--fold") == str(e["fold"]) and g("--tag") == e["id"]
    ok &= ("--opt" in c) == (e["arm"] == "F-nat-reset") and "--init" not in c and "--max-rounds" not in c
    if e["group"] == "tuning":
        ok &= e["fold"] == 0 and e["seed"] == 0 and float(g("--lr")) in MF.LR_GRID
        ok &= e["init"] == ("SERVER:0" if srv else None)
    elif e["group"] == "server":
        ok &= float(g("--lr")) == 3e-4 and e["init"] is None
    else:
        ok &= g("--lr") == f"TUNED:{LRSRC[e['arm']]}" and e["init"] == (f"SERVER:{e['seed']}" if srv else None)
    if not ok: bad.append(e["id"])
R["M_entries_not_matching_A1"] = bad
cells = Counter((e["group"], e["arm"]) for e in man)
R["M_cells"] = {f"{k[0]}|{k[1]}": v for k, v in sorted(cells.items())}
seeds = {}
for e in man: seeds.setdefault((e["group"], e["arm"]), set()).add(e["seed"])
R["M_seeds"] = {f"{k[0]}|{k[1]}": sorted(v) for k, v in seeds.items()}
# tags expected by score_all
need = [f"v21_{'C256' if s < 10 else 'C256rep'}_f{f}_s{s}" for f in range(3) for s in range(20)] + \
       [f"v21_{a}_f{f}_s{s}" for f in range(3) for s in range(10) for a in ("C64", "F-nat", "F-IID", "F-X+P", "S+F", "S+C", "CJ")] + \
       [f"v21_server_s{s}" for s in range(10)] + [f"v21_{a}_f{f}_s{s}" for f in range(3) for s in range(5) for a in ("F-nat-reset", "F-nat-R50")]
ids = {e["id"] for e in man}; R["M_score_all_tags_missing_from_manifest"] = [t for t in need if t not in ids]
R["M_tune_tags"] = sorted(e["id"] for e in man if e["group"] == "tuning")[:3]

# ---- A: independent recomputation of analyze formulas on random data
rng = np.random.default_rng(42)
def my_two(YA, YB):
    nA, D = YA.shape; nB = YB.shape[0]; dd = YA.mean(0) - YB.mean(0); de = dd.mean()
    MA, MB = YA.mean(1), YB.mean(1); s2 = (MA.var(ddof=1) * (nA - 1) + MB.var(ddof=1) * (nB - 1)) / (nA + nB - 2)
    a, b = s2 * (1 / nA + 1 / nB), dd.var(ddof=1) / D; df = (a + b) ** 2 / (a * a / (nA + nB - 2) + b * b / (D - 1))
    t = stats.t.ppf(.95, df); se = np.sqrt(a + b); return de, se, df, de - t * se, de + t * se
def my_pair_like(Iseed, cday):
    n, D = len(Iseed), len(cday); a, b = Iseed.var(ddof=1) / n, cday.var(ddof=1) / D
    df = (a + b) ** 2 / (a * a / (n - 1) + b * b / (D - 1)); t = stats.t.ppf(.95, df); se = np.sqrt(a + b); de = cday.mean()
    return de, se, df, de - t * se, de + t * se
mx = 0.0
for _ in range(200):
    Y = {a: rng.normal(0.6, 0.03, (10, 9)) + rng.normal(0, .05, 9) for a in "ABCD"}
    r = E.two_stratum(Y["A"], Y["B"]); m = my_two(Y["A"], Y["B"])
    mx = max(mx, max(abs(r[k] - v) for k, v in zip(("delta", "se", "df", "L", "U"), m)))
    p = AN.paired(Y["A"], Y["B"]); m = my_pair_like(Y["A"].mean(1) - Y["B"].mean(1), Y["A"].mean(0) - Y["B"].mean(0))
    mx = max(mx, max(abs(p[k] - v) for k, v in zip(("delta", "se", "df", "L", "U"), m)))
    i = AN.interaction(Y["A"], Y["B"], Y["C"], Y["D"])
    m = my_pair_like(Y["A"].mean(1) - Y["B"].mean(1) - Y["C"].mean(1) + Y["D"].mean(1), Y["A"].mean(0) - Y["B"].mean(0) - Y["C"].mean(0) + Y["D"].mean(0))
    mx = max(mx, max(abs(i[k] - v) for k, v in zip(("delta", "se", "df", "L", "U"), m)))
R["A_formulas_max_abs_diff"] = mx

# Q-S rule: own implementation on random scenarios vs analyze.q_s
def fake(dlt_p):   # cs dict with given (delta, p_sup) for the 4 tests; S+F vs F-nat label random
    cs = {}
    for c in ("S+F vs F-nat", "S+F vs F-X+P"):
        cs[c] = dict(metrics={k: dict(delta=dlt_p[c, k][0], p_sup=dlt_p[c, k][1]) for k in E.PRIMARY}, label=rng.choice(["EQUIVALENT", "NON-INFERIOR", "INCONCLUSIVE"]))
    return cs
mism = 0
for _ in range(5000):
    dp = {(c, k): (rng.uniform(-.03, .06), rng.uniform(0, .1)) for c in ("S+F vs F-nat", "S+F vs F-X+P") for k in E.PRIMARY}
    cs = fake(dp); tests = [(c, k) for c in ("S+F vs F-nat", "S+F vs F-X+P") for k in E.PRIMARY]
    p = np.array([dp[t][1] for t in tests]); o = np.argsort(p); rej = np.zeros(4, bool)
    for j, k in enumerate(o):
        if p[k] <= .05 / (4 - j): rej[k] = True
        else: break
    rj = dict(zip(tests, rej))
    helps = any(all(rj[c, k] and dp[c, k][0] >= E.MARGIN[k] / 2 for c in ("S+F vs F-nat", "S+F vs F-X+P")) for k in E.PRIMARY)
    mine = "HELPS" if helps else ("DOES NOT HELP" if cs["S+F vs F-nat"]["label"] == "EQUIVALENT" else "INCONCLUSIVE")
    mism += AN.q_s(cs)["verdict"] != mine
R["A_QS_mismatches_of_5000"] = mism

# attribution: exhaustive over outcome combos (signs of deltas negative) vs report 09 5.8 text
OUT = ("EQ", "NI", "SUP", "INC", "INF"); mism = 0; nn = 0
for oF in OUT:
    for oC64 in OUT:
        for oIC64 in OUT:
            for oIC256 in OUT:
                for U in (-0.01, 0.01):
                    mk = lambda o: dict(metrics={"LOBES": dict(outcome=o, delta=-0.02, U=U)})
                    cs = {"F-nat vs C256": mk(oF), "C64 vs C256": mk(oC64), "F-IID vs C64": mk(oIC64), "F-IID vs C256": mk(oIC256), "F-nat vs F-IID": mk("EQ")}
                    r = AN.attribution(cs, "LOBES"); sh = oF in ("INF", "INC"); nn += 1
                    exp = dict(applies=sh, fewer_negatives=sh and oC64 in ("INF", "INC") and oIC64 in ("EQ", "NI"),
                               averaging=sh and oC64 in ("EQ", "NI") and oIC64 in ("INF", "INC"), non_iid=sh and oIC256 in ("EQ", "NI") and U < 0)
                    mism += r != exp
R["A_attribution_mismatches"] = f"{mism} of {nn}"

# collapse per fold & L1-only failure handling
DAYS = AN.DAYS
def toy(arms, eff=None, sd=.01):
    rows = []
    for a, ss in arms.items():
        for s in ss:
            for d in DAYS:
                for k in ("LOBES", "POLARITY", "DURATION"):
                    rows.append(dict(arm=a, seed=s, fold=AN.FOLD_OF_DAY[d], day=d, metric=k, value=.6 + (eff or {}).get(a, 0) + rng.normal(0, sd)))
    return pd.DataFrame(rows)
def tmodels(arms, zsd):
    return pd.DataFrame([dict(arm=a, seed=s, fold=f, erank=20., Z=z, Q=0., Q_null_q95=.05) for a, (ss, zf) in arms.items() for s in ss for f in range(3)
                         for z in [zf(s, f)]])
SC = toy({"C256": range(10), "C256rep": range(10, 20), "INIT": range(10), "C4": range(10, 20), "C5": range(10, 20)}, dict(INIT=-.3, C4=-.12, C5=-.05))
# L1 deliberately failing: rep Z shifted by +0.03 with SD 0.004
MC = tmodels({"C256": (range(10), lambda s, f: rng.normal(0, .004)), "C256rep": (range(10, 20), lambda s, f: .02 + rng.normal(0, .004)),
              "L2_0.03": (range(10, 20), lambda s, f: .2)}, None)
MC.loc[MC.arm == "L2_0.03", "Q"] = .2
c = AN.controls(SC, MC, [0.03])
R["A_controls_L1_fail_case"] = dict(criteria=c["criteria"], ALL_PASS=c["ALL_PASS"], gate_status=c["gate_status"], m_leak=c["m_leak"], L1=c["L1"]["label"])
# does main_analysis consume gate_status? (signature inspection)
import inspect
R["A_main_analysis_signature"] = str(inspect.signature(AN.main_analysis))
R["A_main_reads_gate_status"] = "gate_status" in inspect.getsource(AN.main_analysis) or "gate_status" in open(os.path.join(PROT, "analyze.py")).read().split("def main_analysis")[1].split("def _without")[0]
R["A_main_cli_reads"] = [l.strip() for l in open(os.path.join(PROT, "analyze.py")) if "sys.argv[4]" in l]

# ---- T: tune.select edge cases
R["T_select"] = dict(exact_plus_0_02=TU.select({3e-4: .70, 6e-4: .72}), just_above=TU.select({3e-4: .70, 6e-4: .7201}),
                     lower_lr_wins=TU.select({3e-4: .70, 1.5e-4: .75, 6e-4: .73}))

# ---- G: INIT encoder == fedsim3.build(seed) encoder; gate_core2 == gate_core
same = []
for s in (0, 7):
    e1 = E2.init_encoder(s); e2, _ = fs.build(s)
    same.append(all(torch.equal(a, b) for a, b in zip(e1.state_dict().values(), e2.state_dict().values())))
R["G_init_encoder_equals_build"] = same
enc = E.load_encoder("data/m5/ckpt/simclr_tf_s0/final.pt"); tr, ev, days = E.fold_sets(0); Eev = E.embed(enc, ev)
g1 = E.gate_core(Eev, *E.gate_arrays(ev), n_perm=3); g2 = E2.gate_core2(Eev, *E.gate_arrays(ev), n_perm=3)
R["G_gate_core2_equals_gate_core"] = all(abs(g1[k] - g2[k]) == 0 for k in ("Q", "Q0", "Z", "Q_null_q95", "Z_unc"))
R["G_gate_fits_nonconverged"] = [g2["gate_fits_nonconverged"], g2["gate_fits"]]
# evaluator/analyze do not pin folds.json themselves
R["G_analyze_or_evaluator_pins_folds"] = any("PIN_FOLDS" in open(os.path.join(PROT, f)).read() for f in ("analyze.py", "evaluator2.py", "score_all.py"))
json.dump(R, open(os.path.join(HERE, f"confirm_step5b_{STAMP}.json"), "x"), indent=1, default=str)
print(json.dumps({k: v for k, v in R.items() if k not in ("M_seeds",)}, indent=1, default=str))
