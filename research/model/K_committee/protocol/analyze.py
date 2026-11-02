"""Protocol v2.1 analysis (report 14, sections A3-A6; report 09 sections 3.5, 5, 6, 8.2). Frozen before any training.

Inputs (tidy, never positional):
  scores  rows (arm, seed, fold, day, metric, value), metric in {LOBES, POLARITY, DURATION}
  models  rows (arm, seed, fold, erank, Z, Z_unc, Q, Q_null_q95, ...)
Every grid is built by grid(): it asserts the exact seed set and the canonical 9-day order from folds.json, no missing and
no duplicate cell, so arms can never be misaligned.

    python analyze.py main     scores.parquet models.parquet results_controls.json  -> results_main_<stamp>.json
    (scores/models = concatenation of the dryrun and main tables; C256 rows come from the dry run)
    python analyze.py controls scores.parquet models.parquet              -> results_controls_<stamp>.json (+ m_leak)
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy import stats

import evaluator as E

MAIN_SEEDS = list(range(10)); REP_SEEDS = list(range(10, 20)); ABL_SEEDS = list(range(5))
F = json.load(open(E.FOLDS))
DAYS = [d for f in ("0", "1", "2") for d in F["folds"][f]["eval_days"]]
FOLD_OF_DAY = {d: int(f) for f in ("0", "1", "2") for d in F["folds"][f]["eval_days"]}
EXCL_DAY = "2025-02-07"                                          # sensitivity (d)


# ------------------------------------------------------------------ grids
def grid(scores, arm, metric, seeds, days=DAYS):
    s = scores[(scores.arm == arm) & (scores.metric == metric)]
    assert not s.duplicated(["seed", "day"]).any(), f"{arm}/{metric}: duplicated (seed, day)"
    s = s[s.seed.isin(seeds) & s.day.isin(days)]
    piv = s.pivot(index="seed", columns="day", values="value")
    assert sorted(piv.index) == sorted(seeds), f"{arm}/{metric}: seeds {sorted(piv.index)} != {sorted(seeds)}"
    assert set(piv.columns) == set(days), f"{arm}/{metric}: days {sorted(piv.columns)} != {sorted(days)}"
    for d in days:
        bad = s[(s.day == d) & (s.fold != FOLD_OF_DAY[d])]
        assert bad.empty, f"{arm}/{metric}: day {d} scored by a model of another fold"
    Y = piv.loc[sorted(seeds), list(days)].to_numpy(float)
    assert np.isfinite(Y).all(), f"{arm}/{metric}: non-finite scores"
    return Y


def grids(scores, arm, seeds, days=DAYS, metrics=("LOBES", "POLARITY", "DURATION")):
    return {k: grid(scores, arm, k, seeds, days) for k in metrics}


# ------------------------------------------------------------------ inference
def satterthwaite(a, df1, b, df2):
    if a + b == 0: return np.inf
    return (a + b) ** 2 / ((a ** 2 / df1 if df1 > 0 else 0) + (b ** 2 / df2 if df2 > 0 else 0))


def _finish(delta, a, df1, b, df2):
    se = float(np.sqrt(a + b)); df = satterthwaite(a, df1, b, df2); t = stats.t.ppf(0.95, df)
    return dict(delta=float(delta), se=se, df=float(df), L=float(delta - t * se), U=float(delta + t * se),
                p_sup=float(stats.t.sf(delta / se, df)) if se > 0 else float(delta <= 0))


def paired(YA, YB):
    """Seed-paired two-stratum t (report 14 A4): seeds must be identical; seed stratum var_s(M_A - M_B)/n, df n-1."""
    n, D = YA.shape; assert YB.shape == (n, D)
    dd = YA.mean(0) - YB.mean(0); delta = dd.mean()
    a = np.var(YA.mean(1) - YB.mean(1), ddof=1) / n; b = np.var(dd - delta, ddof=1) / D
    return _finish(delta, a, n - 1, b, D - 1)


def interaction(YSF, YF, YSC, YC):
    """(S+F - F-nat) - (S+C - C256), seeds shared (report 14 A4)."""
    n, D = YSF.shape
    I = YSF.mean(1) - YF.mean(1) - YSC.mean(1) + YC.mean(1)
    c = YSF.mean(0) - YF.mean(0) - YSC.mean(0) + YC.mean(0); delta = c.mean()
    return _finish(delta, np.var(I, ddof=1) / n, n - 1, np.var(c - delta, ddof=1) / D, D - 1)


def contrast(scores, A, B, seedsA, seedsB, days=DAYS, method="two_stratum"):
    YA, YB = grids(scores, A, seedsA, days), grids(scores, B, seedsB, days); out = {}
    fod = [FOLD_OF_DAY[d] for d in days]
    for k in YA:
        r = E.two_stratum(YA[k], YB[k]) if method == "two_stratum" else paired(YA[k], YB[k])
        r["outcome"] = E.outcome(r["L"], r["U"], E.MARGIN[k])
        dd = YA[k].mean(0) - YB[k].mean(0)
        r["per_fold"] = {f: float(dd[np.asarray(fod) == f].mean()) for f in sorted(set(fod))}
        r["partition_dependent"] = E.heterogeneity(list(r["per_fold"].values()), E.MARGIN[k]) if len(r["per_fold"]) > 1 else False
        out[k] = r
    return dict(A=A, B=B, n=[len(seedsA), len(seedsB)], days=len(days), method=method, metrics=out,
                label=E.contrast_label({k: out[k]["outcome"] for k in E.PRIMARY}))


# ------------------------------------------------------------------ gate
def zbar(models, arm, seeds):
    m = models[(models.arm == arm) & models.seed.isin(seeds)]
    assert not m.duplicated(["seed", "fold"]).any(); piv = m.pivot(index="seed", columns="fold", values="Z")
    assert sorted(piv.index) == sorted(seeds) and sorted(piv.columns) == [0, 1, 2], f"{arm}: gate grid incomplete"
    assert np.isfinite(piv.to_numpy(float)).all(), f"{arm}: missing (seed, fold) gate value"   # found by test_step5b run 1
    return piv.loc[sorted(seeds)].mean(1).to_numpy()


def gate(models, A, B, seedsA, seedsB, m_leak):
    return E.gate_test(zbar(models, A, seedsA), zbar(models, B, seedsB), m_leak)


# ------------------------------------------------------------------ collapse
def collapsed(models, arm, seeds, ref="C256", ref_seeds=MAIN_SEEDS):
    """Seeds of `arm` with erank < 0.5 x median C256 erank of the same fold, in ANY fold (report 09 section 3.5)."""
    out = set()
    for f in (0, 1, 2):
        med = models[(models.arm == ref) & models.seed.isin(ref_seeds) & (models.fold == f)].erank.median()
        m = models[(models.arm == arm) & models.seed.isin(seeds) & (models.fold == f)]
        out |= set(m.seed[m.erank < 0.5 * med].tolist())
    return sorted(out)


# ------------------------------------------------------------------ decisions
def attribution(cs, k):
    """report 09 section 5.8 rules, on metric k, given contrast results cs[name]."""
    o = lambda c: cs[c]["metrics"][k]["outcome"]; d = lambda c: cs[c]["metrics"][k]["delta"]
    neg = lambda c: o(c) in ("INF", "INC") and d(c) < 0
    shortfall = neg("F-nat vs C256")
    return dict(applies=shortfall,
                fewer_negatives=shortfall and neg("C64 vs C256") and o("F-IID vs C64") in ("EQ", "NI"),
                averaging=shortfall and o("C64 vs C256") in ("EQ", "NI") and neg("F-IID vs C64"),
                non_iid=shortfall and o("F-IID vs C256") in ("EQ", "NI") and cs["F-nat vs F-IID"]["metrics"][k]["U"] < 0)


def q_s(cs):
    """Holm family {S+F > F-nat, S+F > F-X+P} x {LOBES, POLARITY}; helps iff for one metric both reject and both deltas >= m/2."""
    tests = [(c, k) for c in ("S+F vs F-nat", "S+F vs F-X+P") for k in E.PRIMARY]
    rej = E.holm([cs[c]["metrics"][k]["p_sup"] for c, k in tests]); R = dict(zip([f"{c}|{k}" for c, k in tests], rej))
    helps = any(all(R[f"{c}|{k}"] and cs[c]["metrics"][k]["delta"] >= E.MARGIN[k] / 2 for c in ("S+F vs F-nat", "S+F vs F-X+P"))
                for k in E.PRIMARY)
    verdict = "HELPS" if helps else ("DOES NOT HELP" if cs["S+F vs F-nat"]["label"] == "EQUIVALENT" else "INCONCLUSIVE")
    return dict(holm_reject=R, verdict=verdict)


PAIRS = [("F-nat", "C256"), ("C64", "C256"), ("F-IID", "C256"), ("F-IID", "C64"), ("F-nat", "F-IID"), ("S+F", "F-nat"),
         ("S+F", "F-X+P"), ("S+F", "C256"), ("S+C", "C256"), ("CJ", "C256"), ("F-X+P", "C256")]


def main_analysis(scores, models, m_leak, seeds=MAIN_SEEDS, days=DAYS, gate_status="CALIBRATED"):
    cs = {f"{a} vs {b}": contrast(scores, a, b, seeds, seeds, days) for a, b in PAIRS}
    gates = {f"{a} vs C256": gate(models, a, "C256", seeds, seeds, m_leak) for a in ("F-nat", "S+F", "F-IID", "C64", "S+C", "CJ", "F-X+P")}
    if gate_status != "CALIBRATED":                                   # report 14 A2: L1 failed -> gate descriptive only
        for g in gates.values(): g.update(label=None, note="gate UNCALIBRATED (L1 failed): descriptive only")
    Yi = {a: grids(scores, a, seeds, days) for a in ("S+F", "F-nat", "S+C", "C256")}
    inter = {k: interaction(Yi["S+F"][k], Yi["F-nat"][k], Yi["S+C"][k], Yi["C256"][k]) for k in ("LOBES", "POLARITY", "DURATION")}
    res = dict(
        QF=dict(label=cs["F-nat vs C256"]["label"], gate=gates["F-nat vs C256"], contrast=cs["F-nat vs C256"]),
        QS=q_s(cs), attribution={k: attribution(cs, k) for k in E.PRIMARY}, interaction=inter, contrasts=cs, gates=gates,
        collapsed={a: collapsed(models, a, seeds) for a in ("C256", "C64", "F-IID", "F-nat", "S+F", "S+C", "F-X+P", "CJ")})
    # sensitivity analyses (report 09 section 5.9)
    f0 = [d for d in days if FOLD_OF_DAY[d] == 0]; nof7 = [d for d in days if d != EXCL_DAY]
    res["sensitivity"] = dict(
        a_fold0_only=contrast(scores, "F-nat", "C256", seeds, seeds, f0)["label"],
        b_without_collapsed=_without_collapsed(scores, res["collapsed"], seeds, days),
        c_seed_paired=contrast(scores, "F-nat", "C256", seeds, seeds, days, method="paired")["label"],
        d_without_feb7=contrast(scores, "F-nat", "C256", seeds, seeds, nof7)["label"],
        QS_seed_paired=q_s({f"{a} vs {b}": contrast(scores, a, b, seeds, seeds, days, method="paired")
                            for a, b in (("S+F", "F-nat"), ("S+F", "F-X+P"))})["verdict"])
    return res


def _without_collapsed(scores, col, seeds, days):
    sA = [s for s in seeds if s not in col["F-nat"]]; sB = [s for s in seeds if s not in col["C256"]]
    if len(sA) < 2 or len(sB) < 2: return "NOT ESTIMABLE"
    return contrast(scores, "F-nat", "C256", sA, sB, days)["label"]


# ------------------------------------------------------------------ dry-run controls (report 14 A2)
def controls(scores, models, kappas):
    """C1, C3, C4, C5 on scores; m_leak, L1, L2 on models. Planted arms are named C4, C5, L2_<kappa> (seeds 10-19)."""
    c = dict(C1=contrast(scores, "C256rep", "C256", REP_SEEDS, MAIN_SEEDS), C3=contrast(scores, "INIT", "C256", MAIN_SEEDS, MAIN_SEEDS),
             C4=contrast(scores, "C4", "C256", REP_SEEDS, MAIN_SEEDS), C5=contrast(scores, "C5", "C256", REP_SEEDS, MAIN_SEEDS))
    zr, zm = zbar(models, "C256rep", REP_SEEDS), zbar(models, "C256", MAIN_SEEDS)
    m_leak = E.m_leak_rule(zr, zm); L1 = E.gate_test(zr, zm, m_leak)
    det = {}
    for k in kappas:
        m = models[(models.arm == f"L2_{k}") & models.seed.isin(REP_SEEDS)]
        assert len(m) == len(REP_SEEDS) * 3, f"L2_{k}: incomplete"
        det[k] = float((m.Q > m.Q_null_q95).mean())
    kstar = next((k for k in sorted(kappas) if det[k] >= 0.5), None)
    L2 = gate(models, f"L2_{kstar}", "C256", REP_SEEDS, MAIN_SEEDS, m_leak) if kstar is not None else None
    crit = dict(C1=c["C1"]["label"] == "EQUIVALENT", C3=c["C3"]["label"] == "INFERIOR", C4=c["C4"]["label"] == "INFERIOR",
                C5=c["C5"]["label"] != "EQUIVALENT", L1=L1["label"] == "PASS", L2=(L2 is not None and L2["label"] == "FAIL"))
    # report 14 A2: an L1 failure alone does not stop the study; it makes the gate UNCALIBRATED (descriptive only)
    return dict(controls=c, m_leak=m_leak, L1=L1, L2_detection_rate=det, kappa_star=kstar, L2=L2, criteria=crit,
                ALL_PASS=all(crit[k] for k in ("C1", "C3", "C4", "C5", "L2")),
                gate_status="CALIBRATED" if crit["L1"] else "UNCALIBRATED")


if __name__ == "__main__":
    cmd = sys.argv[1]; S = pd.read_parquet(sys.argv[2]); M = pd.read_parquet(sys.argv[3]); stamp = time.strftime("%Y%m%d_%H%M%S")
    here = os.path.dirname(os.path.abspath(__file__))
    if cmd == "controls":
        kap = sorted({float(a.split("_")[1]) for a in M.arm.unique() if a.startswith("L2_")})
        M = M.assign(arm=M.arm.map(lambda a: f"L2_{float(a.split('_')[1])}" if a.startswith("L2_") else a))
        out = controls(S, M, kap)
    else:
        ctl = json.load(open(sys.argv[4])); assert ctl["ALL_PASS"], "dry-run controls did not pass: main analysis refused"
        out = main_analysis(S, M, ctl["m_leak"], gate_status=ctl["gate_status"])
    p = os.path.join(here, f"results_{cmd}_{stamp}.json"); json.dump(out, open(p, "x"), indent=1, default=str); print(p)
