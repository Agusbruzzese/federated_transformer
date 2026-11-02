"""Protocol v2 evaluator (report 09 sections 5-6; step 4). Identical for every arm; outside the federated system.

Scoring (per model, per fold):
  embed the fold's client events and evaluation-day events (eval view: centred 12 s crop, max-normalised, 64-d);
  probes fitted on the client events, scored per evaluation day:
    kNN     k = 15, Euclidean, StandardScaler on
    linear  LogisticRegression(C = 1, lbfgs, float64, tol 1e-8, max_iter 10 000; non-convergence recorded) /
            RidgeCV(logspace(-3, 3, 13)) for duration; StandardScaler on
  score: m4.skill (chance-corrected balanced accuracy over the classes present that day) or R2 (duration)
  PRIMARY: LOBES = mean(kNN, linear) skill on min(n_lobes, 4); POLARITY = mean(kNN, linear) skill. DURATION secondary.
Nuisance-encoding gate (per model, per fold; section 6):
  Q = skill(day | [emb PCA-19, morph8, hour bin, has_dropout]) - skill(day | [morph8, hour bin, has_dropout]),
  logistic C = 1 + StandardScaler, GroupKFold(5) over 30-min blocks; null = 50 block permutations of the day
  labels; Z = Q - mean(Q_null). Unconditioned version (embedding alone) reported as secondary.
Statistics (section 5.3-5.8): two-stratum t, per-metric outcome, contrast label, gate test, margin rule, Holm,
heterogeneity flag.
"""
from __future__ import annotations

import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = os.getcwd(); HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [ROOT, os.path.join(ROOT, "research/model/M4_methods")]
import m4
from ssl_central import augment as A, models as Mo

FOLDS = os.path.join(HERE, "folds.json"); DEV = "cuda" if torch.cuda.is_available() else "cpu"
MARGIN = dict(LOBES=0.04, POLARITY=0.02, DURATION=0.03); PRIMARY = ("LOBES", "POLARITY")
N_PERM, GATE_SEED = 50, 20261010
_C = {}


# ------------------------------------------------------------------ data
def data():
    if "m" not in _C:
        m, W = m4.load()
        with torch.no_grad():
            x = A.eval_view(torch.tensor(W, device=DEV), torch.tensor(m.fs_eff.to_numpy(np.float32), device=DEV))
        tg = m4.targets(m)
        F = json.load(open(FOLDS)); day = m.date.astype(str).to_numpy()
        morph8 = m[[c for c in m.columns if c.startswith("m8_")]].to_numpy(np.float64)
        hb = np.eye(4)[m.hbin.to_numpy().astype(int)]
        _C.update(m=m, x=x, tg=tg, F=F, day=day, conf=np.column_stack([morph8, hb, m.has_dropout.to_numpy(np.float64)]),
                  blk=m.blk.to_numpy())
    return _C


def fold_sets(fold):
    C = data(); f = C["F"]["folds"][str(fold)]
    tr = np.flatnonzero(np.isin(C["day"], [d for c in f["clients"] for d in c["days"]]))
    ev = np.flatnonzero(np.isin(C["day"], f["eval_days"]))
    return tr, ev, list(f["eval_days"])


def load_encoder(path):
    sd = torch.load(path, map_location=DEV); sd = sd.get("enc", sd) if isinstance(sd, dict) and "enc" in sd else sd
    enc = Mo.make("tf").to(DEV); enc.load_state_dict(sd); return enc.eval()


@torch.no_grad()
def embed(enc, idx):
    x = data()["x"][torch.as_tensor(idx, device=DEV)]
    return torch.cat([enc(x[i:i + 2048]) for i in range(0, len(x), 2048)]).double().cpu().numpy()


# ------------------------------------------------------------------ probes
def _fit_predict(kind, target, Etr, ytr, Eev):
    if target == "duration":
        est = KNeighborsRegressor(15) if kind == "knn" else RidgeCV(alphas=np.logspace(-3, 3, 13))
    else:
        est = KNeighborsClassifier(15) if kind == "knn" else LogisticRegression(C=1.0, tol=1e-8, max_iter=10_000)
    pipe = make_pipeline(StandardScaler(), est)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", ConvergenceWarning); pipe.fit(Etr, ytr)
    conv = not any(issubclass(x.category, ConvergenceWarning) for x in w)
    return pipe.predict(Eev), conv


def probe_scores(Etr, Eev, tr, ev, days):
    """Per-day scores: rows (day, target, probe, value) + composites (LOBES, POLARITY, DURATION)."""
    C = data(); rows, conv = [], True; dev_day = C["day"][ev]
    for target in ("polarity", "lobes", "duration"):
        y = C["tg"][target]
        for kind in ("knn", "lin"):
            p, ok = _fit_predict(kind, target, Etr, y[tr], Eev); conv &= ok
            for d in days:
                k = dev_day == d
                rows.append(dict(day=d, target=target, probe=kind, value=float(m4.metric(target, y[ev][k], p[k]))))
    df = pd.DataFrame(rows)
    comp = df.groupby(["day", "target"]).value.mean().unstack()
    comp = comp.rename(columns=dict(lobes="LOBES", polarity="POLARITY", duration="DURATION"))
    return df, comp.reindex(days), conv


# ------------------------------------------------------------------ nuisance-encoding gate
def _cv_skill(X, y, groups):
    p = np.empty(len(y), dtype=y.dtype)
    for a, b in GroupKFold(5).split(X, y, groups):
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000)).fit(X[a], y[a]); p[b] = clf.predict(X[b])
    return m4.skill(y, p)


def gate_arrays(ev):
    C = data(); return C["day"][ev], C["blk"][ev], C["conf"][ev]


def gate_core(Eev, y, g, conf, n_perm=N_PERM, seed=GATE_SEED):
    """Pure function (parallel-safe): Z (primary, conditional) and Z_unc (secondary)."""
    pc = PCA(19, random_state=0).fit_transform(Eev)
    Xc, Xr, Xu = np.column_stack([pc, conf]), conf, pc
    def stat(lab):
        return _cv_skill(Xc, lab, g) - _cv_skill(Xr, lab, g), _cv_skill(Xu, lab, g)
    Q, Qu = stat(y)
    ub, inv = np.unique(g, return_inverse=True); blab = np.array([y[inv == i][0] for i in range(len(ub))])
    rng = np.random.default_rng(seed); null = np.array([stat(blab[rng.permutation(len(ub))][inv]) for _ in range(n_perm)])
    return dict(Q=Q, Q0=float(null[:, 0].mean()), Z=Q - float(null[:, 0].mean()), Q_null_q95=float(np.quantile(null[:, 0], .95)),
                Qu=Qu, Qu0=float(null[:, 1].mean()), Z_unc=Qu - float(null[:, 1].mean()))


def gate(Eev, ev, n_perm=N_PERM, seed=GATE_SEED):
    return gate_core(Eev, *gate_arrays(ev), n_perm=n_perm, seed=seed)


# ------------------------------------------------------------------ one model, one fold
def score(path=None, fold=0, enc=None, transform=None, do_gate=True):
    """transform(Etr, Eev, ev_days_array, rng) -> (Etr, Eev): planted controls only."""
    enc = enc if enc is not None else load_encoder(path)
    tr, ev, days = fold_sets(fold); Etr, Eev = embed(enc, tr), embed(enc, ev)
    if transform is not None:
        Etr, Eev = transform(Etr, Eev, data()["day"][ev])
    df, comp, conv = probe_scores(Etr, Eev, tr, ev, days)
    out = dict(fold=fold, days=days, per_probe=df, comp=comp, lin_converged=conv)
    if do_gate: out["gate"] = gate(Eev, ev)
    return out


# ------------------------------------------------------------------ statistics
def two_stratum(YA, YB):
    """YA [nA, D], YB [nB, D], same day order. Returns Delta, SE, df, 90 % CI (L, U)."""
    YA, YB = np.asarray(YA, float), np.asarray(YB, float); nA, D = YA.shape; nB = YB.shape[0]
    dd = YA.mean(0) - YB.mean(0); delta = float(dd.mean())
    MA, MB = YA.mean(1), YB.mean(1)
    s2seed = (((MA - MA.mean()) ** 2).sum() + ((MB - MB.mean()) ** 2).sum()) / (nA + nB - 2)
    s2day = float(np.var(dd - delta, ddof=1)) if D > 1 else 0.0
    a, b = s2seed * (1 / nA + 1 / nB), s2day / D; se = float(np.sqrt(a + b))
    df = (a + b) ** 2 / (a ** 2 / (nA + nB - 2) + (b ** 2 / (D - 1) if D > 1 else 0)) if a + b > 0 else np.inf
    t = stats.t.ppf(0.95, df)
    return dict(delta=delta, se=se, df=float(df), L=delta - t * se, U=delta + t * se,
                p_sup=float(stats.t.sf(delta / se, df)) if se > 0 else float(delta <= 0))


def outcome(L, U, m):
    """Section 5.6: exhaustive, mutually exclusive."""
    if L > -m:
        if U < m: return "EQ"
        return "SUP" if L >= m else "NI"
    return "INC" if U > -m else "INF"


def contrast_label(per_metric):
    """Section 5.7 primary label from {metric: outcome} over the primary metrics."""
    o = [per_metric[k] for k in PRIMARY]
    if "INF" in o: return "INFERIOR"
    if "INC" in o: return "INCONCLUSIVE"
    if all(x == "EQ" for x in o): return "EQUIVALENT"
    return "NON-INFERIOR"


def gate_test(ZA, ZB, m_leak):
    """Welch t over seeds on per-seed fold-averaged Z; one-sided 95 % bounds."""
    ZA, ZB = np.asarray(ZA, float), np.asarray(ZB, float); d = float(ZA.mean() - ZB.mean())
    va, vb = ZA.var(ddof=1) / len(ZA), ZB.var(ddof=1) / len(ZB); se = float(np.sqrt(va + vb))
    df = (va + vb) ** 2 / (va ** 2 / (len(ZA) - 1) + vb ** 2 / (len(ZB) - 1)); t = stats.t.ppf(0.95, df)
    U, L = d + t * se, d - t * se
    lab = "PASS" if U < m_leak else "FAIL"
    note = "shown" if (lab == "FAIL" and L >= m_leak) else ("within margin, positive" if (lab == "PASS" and L > 0) else "")
    return dict(delta=d, se=se, df=float(df), L=L, U=U, m_leak=m_leak, label=lab, note=note)


def m_leak_rule(ZC1, ZC2):
    """Section 6.3: m_leak = max(0.03, 2 |Delta(C256 0-4 vs 5-9)|, 2 SE)."""
    r = gate_test(ZC1, ZC2, np.inf)
    return float(max(0.03, 2 * abs(r["delta"]), 2 * r["se"]))


def holm(pvals, alpha=0.05):
    """Holm step-down; returns reject flags in input order."""
    p = np.asarray(pvals, float); o = np.argsort(p); rej = np.zeros(len(p), bool)
    for i, k in enumerate(o):
        if p[k] <= alpha / (len(p) - i): rej[k] = True
        else: break
    return rej.tolist()


def heterogeneity(per_fold_deltas, m):
    return bool(np.ptp(per_fold_deltas) > m)


def contrast(YA, YB, fold_of_day):
    """YA/YB: {metric: [n, D]}; returns per-metric stats, outcomes, primary label, per-fold deltas and flags."""
    res = {}
    for k in YA:
        r = two_stratum(YA[k], YB[k]); r["outcome"] = outcome(r["L"], r["U"], MARGIN[k])
        dd = np.asarray(YA[k]).mean(0) - np.asarray(YB[k]).mean(0)
        r["per_fold"] = {f: float(dd[np.asarray(fold_of_day) == f].mean()) for f in sorted(set(fold_of_day))}
        r["partition_dependent"] = heterogeneity(list(r["per_fold"].values()), MARGIN[k]); res[k] = r
    return dict(metrics=res, label=contrast_label({k: res[k]["outcome"] for k in PRIMARY}))
