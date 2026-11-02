"""Protocol v2.1 evaluator additions (report 14; reviewer F R2/R4). Extends evaluator.py (left as is) with:
  - score_run(): reads summary.json next to the checkpoint and ASSERTS that the checkpoint was trained for that fold
    (server checkpoints, which never see client or evaluation days, may be scored on any fold);
  - init_encoder(seed): the untrained weights exactly as fedsim3.build(seed) creates them (control C3);
  - per-day skill with an assertion of >= 2 classes per day;
  - effective rank of the evaluation embedding (collapse rule, report 09 section 3.5);
  - gate with a convergence count of its logistic fits;
  - score_tuning(): probes fitted on fold-0 client events, scored on the tuning day (Feb 10);
  - knn_overlap(): 15-NN overlap between two embeddings of the same events (descriptive noise floor);
  - tidy rows (arm, seed, fold, day, metric, value) so that analysis never relies on column order.
"""
from __future__ import annotations

import json
import os
import warnings

import numpy as np
import torch
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import evaluator as E
import m4
from ssl_central import models as Mo

RUNS = os.path.join(E.ROOT, "data", "k_committee", "protocol", "runs3")


def run_info(tag):
    return json.load(open(os.path.join(RUNS, tag, "summary.json")))


def score_run(tag, fold, **kw):
    s = run_info(tag); a = s["args"]
    assert a["arm"] == "server" or str(a["fold"]) == str(fold), f"{tag}: trained for fold {a['fold']}, refused on fold {fold}"
    assert s["provenance"]["final"] is not None
    return score(E.load_encoder(os.path.join(RUNS, tag, "final.pt")), fold, **kw)


def init_encoder(seed):
    torch.manual_seed(seed); np.random.seed(seed)                   # == fedsim3.build(seed)
    return Mo.make("tf").to(E.DEV).eval()


def safe_metric(target, y, p):
    if target != "duration":
        assert len(np.unique(y)) >= 2, "skill undefined: fewer than 2 classes on a day"
    return float(m4.metric(target, y, p))


def probe_rows(Etr, Eev, tr, ev, days):
    C = E.data(); rows, conv = [], True; dday = C["day"][ev]
    for target in ("polarity", "lobes", "duration"):
        y = C["tg"][target]
        for kind in ("knn", "lin"):
            p, ok = E._fit_predict(kind, target, Etr, y[tr], Eev); conv &= ok
            for d in days:
                k = dday == d
                rows.append(dict(day=d, target=target, probe=kind, value=safe_metric(target, y[ev][k], p[k])))
    comp = {}
    for d in days:
        for target, name in (("lobes", "LOBES"), ("polarity", "POLARITY"), ("duration", "DURATION")):
            comp[d, name] = float(np.mean([r["value"] for r in rows if r["day"] == d and r["target"] == target]))
    return rows, comp, conv


def erank(Z):
    sv = np.linalg.svd(Z - Z.mean(0), compute_uv=False); p = sv / sv.sum()
    return float(np.exp(-(p * np.log(p + 1e-12)).sum()))


def _cv_skill_count(X, y, groups, counter):
    p = np.empty(len(y), dtype=y.dtype)
    for a, b in GroupKFold(5).split(X, y, groups):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always", ConvergenceWarning)
            clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000)).fit(X[a], y[a])
        counter[0] += any(issubclass(x.category, ConvergenceWarning) for x in w); counter[1] += 1; p[b] = clf.predict(X[b])
    return m4.skill(y, p)


def gate_core2(Eev, y, g, conf, n_perm=E.N_PERM, seed=E.GATE_SEED):
    """evaluator.gate_core with a convergence count (same statistic, same permutations)."""
    from sklearn.decomposition import PCA
    cnt = [0, 0]; pc = PCA(19, random_state=0).fit_transform(Eev)
    Xc, Xr, Xu = np.column_stack([pc, conf]), conf, pc
    stat = lambda lab: (_cv_skill_count(Xc, lab, g, cnt) - _cv_skill_count(Xr, lab, g, cnt), _cv_skill_count(Xu, lab, g, cnt))
    Q, Qu = stat(y)
    ub, inv = np.unique(g, return_inverse=True); blab = np.array([y[inv == i][0] for i in range(len(ub))])
    rng = np.random.default_rng(seed); null = np.array([stat(blab[rng.permutation(len(ub))][inv]) for _ in range(n_perm)])
    return dict(Q=float(Q), Q0=float(null[:, 0].mean()), Z=float(Q - null[:, 0].mean()), Q_null_q95=float(np.quantile(null[:, 0], .95)),
                Qu=float(Qu), Z_unc=float(Qu - null[:, 1].mean()), gate_fits_nonconverged=cnt[0], gate_fits=cnt[1])


def embed_fold(enc, fold):
    tr, ev, days = E.fold_sets(fold); return tr, ev, days, E.embed(enc, tr), E.embed(enc, ev)


def score(enc, fold, transform=None, do_gate=True, emb=None):
    """transform(Etr, Eev, eval_day_labels) -> (Etr, Eev): planted controls only. Returns (rows, model_info)."""
    tr, ev, days, Etr, Eev = emb if emb is not None else embed_fold(enc, fold)
    if transform is not None: Etr, Eev = transform(Etr, Eev, E.data()["day"][ev])
    prow, comp, conv = probe_rows(Etr, Eev, tr, ev, days)
    rows = [dict(fold=int(fold), day=d, metric=k, value=v) for (d, k), v in comp.items()]
    info = dict(fold=int(fold), erank=erank(Eev), lin_converged=bool(conv), per_probe=prow)
    if do_gate: info.update(gate_core2(Eev, *E.gate_arrays(ev)))
    return rows, info


def score_tuning(enc):
    """Probes fitted on fold-0 client events; scored on the tuning day only. Returns mean(LOBES, POLARITY) and parts."""
    C = E.data(); F = C["F"]; tr, _, _ = E.fold_sets(0); ev = np.flatnonzero(C["day"] == F["tuning_day"])
    _, comp, conv = probe_rows(E.embed(enc, tr), E.embed(enc, ev), tr, ev, [F["tuning_day"]])
    d = F["tuning_day"]; return dict(score=float((comp[d, "LOBES"] + comp[d, "POLARITY"]) / 2), LOBES=comp[d, "LOBES"],
                                     POLARITY=comp[d, "POLARITY"], DURATION=comp[d, "DURATION"], lin_converged=conv)


def knn_overlap(Za, Zb, k=15):
    """Mean fraction of shared k nearest neighbours (z-scored embeddings, self excluded)."""
    def nn(Z):
        Z = (Z - Z.mean(0)) / (Z.std(0) + 1e-12); return NearestNeighbors(n_neighbors=k + 1).fit(Z).kneighbors(Z)[1][:, 1:]
    A, B = nn(Za), nn(Zb)
    return float(np.mean([len(set(a) & set(b)) / k for a, b in zip(A, B)]))
