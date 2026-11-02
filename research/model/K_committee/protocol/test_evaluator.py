"""Step 4 tests of the protocol v2 evaluator. Pass criteria fixed here BEFORE the first run (report 12):
  T1  decision table: 200 000 random (L, U) pairs -> exactly one outcome each; boundary cases as in report 09 5.6;
      the contrast label rule follows 5.7 on all 25 outcome pairs.
  T2  calibration of the two-stratum t on the Gaussian model (B's variance components, n = 5 vs 5, D = 9, 3 folds):
      P(EQ) at true Delta = -m and +m <= 0.07 for LOBES and POLARITY (type I); power at Delta = 0 is reported.
  T3  Holm on a known example.
  T4  determinism: scoring the same checkpoint twice gives identical numbers.
  T5  planted controls on REAL checkpoints over the 3 folds (9 days). PROCEDURE TEST ONLY: the M5 models trained on
      some of these evaluation days, so no number here is a result about central vs federated.
        A = M5 SimCLR-TF seeds 0-4 (10k steps); B = B's extra M5-recipe seeds 5-7 (independent central); I = init of 0-4
        C1  A vs B                         -> not INFERIOR (EQUIVALENT expected; INCONCLUSIVE tolerated: n_B = 3)
        C3  I vs B                         -> INFERIOR
        C4  A + noise tau = 0.5 SD vs B    -> INFERIOR
        C5  A + noise tau* vs B, tau* by bisection on fold 0 so that LOBES Delta(fold 0) = -1.25 m -> not EQUIVALENT
        L1  gate A vs B with m_leak from the margin rule -> PASS
        L2  A + planted per-evaluation-day offset kappa x SD x u_day, kappa in {0.005, 0.01, 0.02, 0.03, 0.05, 0.1}
            -> gate FAIL for at least one kappa in the grid; the smallest such kappa is reported
    .venv/bin/python research/model/K_committee/protocol/test_evaluator.py
"""
import itertools, json, os, sys, time
import numpy as np
from joblib import Parallel, delayed
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import evaluator as E
STAMP = time.strftime("%Y%m%d_%H%M%S"); R = {}; t_all = time.time()

# ---------------- T1
rng = np.random.default_rng(1); ok = True
for m in (0.02, 0.04):
    for _ in range(100_000):
        L, w = rng.uniform(-0.2, 0.2), rng.uniform(0, 0.2); o = E.outcome(L, L + w, m)
        ok &= o in ("EQ", "NI", "SUP", "INC", "INF")
b = dict(EQ=E.outcome(-0.039, 0.039, 0.04), NI=E.outcome(-0.03, 0.05, 0.04), SUP=E.outcome(0.04, 0.09, 0.04),
         INC=E.outcome(-0.04, 0.0, 0.04), INF=E.outcome(-0.09, -0.04, 0.04))
lab = {f"{a}/{c}": E.contrast_label(dict(LOBES=a, POLARITY=c)) for a, c in itertools.product(["EQ", "NI", "SUP", "INC", "INF"], repeat=2)}
exp_ok = all((lab[k] == "INFERIOR") == ("INF" in k) for k in lab) and lab["EQ/EQ"] == "EQUIVALENT" and lab["NI/SUP"] == "NON-INFERIOR" \
    and lab["INC/EQ"] == "INCONCLUSIVE" and lab["INF/INC"] == "INFERIOR"
R["T1"] = dict(single_outcome=bool(ok), boundaries=b, boundaries_ok=all(k == v for k, v in b.items()), labels=lab, PASS=bool(ok and exp_ok and all(k == v for k, v in b.items())))
print("T1", R["T1"]["PASS"], flush=True)

# ---------------- T2 (B's components: varcomp.csv; lobes sigma_seed .007, sigma_e .027, sigma_v .030; polarity .000/.008/.009)
def sim(delta, s_seed, s_e, s_v, n=5, D=9, nsim=4000, seed=7):
    r = np.random.default_rng(seed); eq = 0; m = None
    for _ in range(nsim):
        u = r.normal(0, 0.088, D); vA = r.normal(0, s_v, D)
        YA = 0.6 + delta + u + vA + r.normal(0, s_seed, (n, 1)) + r.normal(0, s_e, (n, D))
        YB = 0.6 + u + r.normal(0, s_seed, (n, 1)) + r.normal(0, s_e, (n, D))
        s = E.two_stratum(YA, YB); eq += E.outcome(s["L"], s["U"], mg) == "EQ"
    return eq / nsim
T2 = {}
for name, (s_seed, s_e, s_v) in dict(LOBES=(0.007, 0.027, 0.030), POLARITY=(0.001, 0.008, 0.009)).items():
    mg = E.MARGIN[name]
    T2[name] = dict(typeI_minus_m=sim(-mg, s_seed, s_e, s_v), typeI_plus_m=sim(mg, s_seed, s_e, s_v), power_at_0=sim(0.0, s_seed, s_e, s_v))
R["T2"] = dict(T2, PASS=all(v["typeI_minus_m"] <= 0.07 and v["typeI_plus_m"] <= 0.07 for v in T2.values()))
print("T2", R["T2"], flush=True)

# ---------------- T3
h = E.holm([0.01, 0.04, 0.03, 0.005]); R["T3"] = dict(holm=h, PASS=h == [True, False, False, True])
print("T3", R["T3"], flush=True)

# ---------------- T4
r1 = E.score("data/m5/ckpt/simclr_tf_s0/final.pt", 0, do_gate=False); r2 = E.score("data/m5/ckpt/simclr_tf_s0/final.pt", 0, do_gate=False)
R["T4"] = dict(PASS=bool(r1["comp"].equals(r2["comp"]) and r1["per_probe"].equals(r2["per_probe"])))
print("T4", R["T4"], flush=True)

# ---------------- T5: embed every model x fold once (GPU), then probes and gates (CPU, parallel)
CK = dict(A=[f"data/m5/ckpt/simclr_tf_s{s}/final.pt" for s in range(5)],
          B=[f"data/k_committee/B_stats/full/ckpt/simclr_tf_s{s}/final.pt" for s in (5, 6, 7)],
          I=[f"data/m5/ckpt/simclr_tf_s{s}/init.pt" for s in range(5)])
FOLD = [0, 1, 2]; EMB = {}
for g, paths in CK.items():
    for i, p in enumerate(paths):
        enc = E.load_encoder(p)
        for f in FOLD:
            tr, ev, days = E.fold_sets(f); EMB[g, i, f] = (E.embed(enc, tr), E.embed(enc, ev))
days_all = sum([E.fold_sets(f)[2] for f in FOLD], []); fold_of_day = sum([[f] * 3 for f in FOLD], [])
def noise(Etr, Eev, tau, seed):
    r = np.random.default_rng(seed); sd = Etr.std(0)
    return Etr + tau * sd * r.standard_normal(Etr.shape), Eev + tau * sd * r.standard_normal(Eev.shape)
def comp_matrix(group, transform=None, folds=FOLD):
    out = {k: [] for k in ("LOBES", "POLARITY", "DURATION")}; n = len(CK[group])
    for i in range(n):
        row = {k: [] for k in out}
        for f in folds:
            tr, ev, days = E.fold_sets(f); Etr, Eev = EMB[group, i, f]
            if transform is not None: Etr, Eev = transform(Etr, Eev, i, f)
            _, comp, _ = E.probe_scores(Etr, Eev, tr, ev, days)
            for k in out: row[k] += comp[k].tolist()
        for k in out: out[k].append(row[k])
    return {k: np.array(v) for k, v in out.items()}
Y = {g: comp_matrix(g) for g in ("A", "B", "I")}
C1 = E.contrast(Y["A"], Y["B"], fold_of_day); C3 = E.contrast(Y["I"], Y["B"], fold_of_day)
YC4 = comp_matrix("A", lambda a, b, i, f: noise(a, b, 0.5, 100 + 10 * i + f)); C4 = E.contrast(YC4, Y["B"], fold_of_day)
print("C1", C1["label"], "C3", C3["label"], "C4", C4["label"], flush=True)
# C5 bisection on fold 0 (LOBES delta of fold 0 days only)
YB0 = comp_matrix("B", folds=[0])["LOBES"]; target = -1.25 * E.MARGIN["LOBES"]; lo, hi = 0.0, 0.5; trace = []
for _ in range(8):
    tau = (lo + hi) / 2; YA0 = comp_matrix("A", lambda a, b, i, f: noise(a, b, tau, 100 + 10 * i + f), folds=[0])["LOBES"]
    d = float(YA0.mean() - YB0.mean()); trace.append((round(tau, 4), round(d, 4)))
    lo, hi = (tau, hi) if d > target else (lo, tau)
tau_star = (lo + hi) / 2
YC5 = comp_matrix("A", lambda a, b, i, f: noise(a, b, tau_star, 100 + 10 * i + f)); C5 = E.contrast(YC5, Y["B"], fold_of_day)
print("C5 tau*", tau_star, C5["label"], flush=True)
# gates (parallel)
def gate_job(g, i, f, kappa):
    tr, ev, days = E.fold_sets(f); Etr, Eev = EMB[g, i, f]
    if kappa > 0:
        y = E.data()["day"][ev]; r = np.random.default_rng(500 + f); sd = Etr.std(0)
        U = {d: (lambda v: v / np.linalg.norm(v))(r.standard_normal(Eev.shape[1])) for d in days}
        Eev = Eev + kappa * np.linalg.norm(sd) * np.stack([U[d] for d in y])
    return (g, i, f, kappa), E.gate_core(Eev, *E.gate_arrays(ev))
E.data()
jobs = [("A", i, f, 0.0) for i in range(5) for f in FOLD] + [("B", i, f, 0.0) for i in range(3) for f in FOLD]
KAPPA = [0.005, 0.01, 0.02, 0.03, 0.05, 0.1]
jobs += [("A", i, f, k) for k in KAPPA for i in range(5) for f in FOLD]
t0 = time.time(); GR = dict(Parallel(n_jobs=12, backend="threading")(delayed(gate_job)(*j) for j in jobs))
print(f"{len(jobs)} gates in {time.time() - t0:.0f}s", flush=True)
Zbar = lambda g, n, k: [float(np.mean([GR[g, i, f, k]["Z"] for f in FOLD])) for i in range(n)]
ZA, ZB = Zbar("A", 5, 0.0), Zbar("B", 3, 0.0); m_leak = E.m_leak_rule(ZA, ZB)
L1 = E.gate_test(ZA, ZB, m_leak)
L2 = {k: E.gate_test(Zbar("A", 5, k), ZB, m_leak) for k in KAPPA}
kmin = next((k for k in KAPPA if L2[k]["label"] == "FAIL"), None)
strip = lambda c: dict(label=c["label"], metrics={k: {kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in v.items()} for k, v in c["metrics"].items()})
R["T5"] = dict(note="procedure test only; M5 models trained on some evaluation days", days=days_all,
               C1=strip(C1), C3=strip(C3), C4=strip(C4), C5=dict(strip(C5), tau_star=tau_star, bisection=trace),
               m_leak=m_leak, Z_A=ZA, Z_B=ZB, L1=L1, L2={str(k): v for k, v in L2.items()}, L2_kappa_min_fail=kmin,
               Z_unc_A=Zbar("A", 5, 0.0) and [float(np.mean([GR["A", i, f, 0.0]["Z_unc"] for f in FOLD])) for i in range(5)])
crit = dict(C1=C1["label"] != "INFERIOR", C3=C3["label"] == "INFERIOR", C4=C4["label"] == "INFERIOR",
            C5=C5["label"] != "EQUIVALENT", L1=L1["label"] == "PASS", L2=kmin is not None)
R["T5"]["criteria"] = crit; R["T5"]["PASS"] = all(crit.values())
R["ALL_PASS"] = all(R[k]["PASS"] for k in ("T1", "T2", "T3", "T4", "T5")); R["secs"] = round(time.time() - t_all)
out = os.path.join(HERE, f"test_evaluator_{STAMP}.json"); json.dump(R, open(out, "x"), indent=1, default=str)
print(json.dumps(dict(criteria=crit, m_leak=m_leak, L1=L1["label"], kmin=kmin, ALL_PASS=R["ALL_PASS"]), indent=1), "->", out)
