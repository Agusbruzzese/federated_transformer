"""Probes (kNN for P1/selection, HGB for P2), metrics and the 30-min block bootstrap."""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import balanced_accuracy_score, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

from . import config as C


def skill(y, p):
    K = len(np.unique(y))
    return (balanced_accuracy_score(y, p) - 1 / K) / (1 - 1 / K)


def metric(y, p, kind):
    return r2_score(y, p) if kind == "reg" else skill(y, p)


def _z(Xtr, Xte):
    mu = Xtr.mean(0); sd = Xtr.std(0); sd[sd == 0] = 1
    return (Xtr - mu) / sd, (Xte - mu) / sd


def knn_oof(X, y, groups, kind, k=C.K_NN, n_splits=5):
    X = np.asarray(X, float)
    X = X[:, None] if X.ndim == 1 else X
    pred = np.empty(len(y), float if kind == "reg" else y.dtype)
    for tr, te in GroupKFold(n_splits=n_splits).split(X, y, groups):
        a, b = _z(X[tr], X[te])
        mdl = (KNeighborsRegressor if kind == "reg" else KNeighborsClassifier)(n_neighbors=k)
        pred[te] = mdl.fit(a, y[tr]).predict(b)
    return pred


def hgb_oof(X, y, groups, n_splits=5, seed=0):
    X = np.asarray(X, float)
    X = X[:, None] if X.ndim == 1 else X
    pred = np.empty_like(y)
    for tr, te in GroupKFold(n_splits=n_splits).split(X, y, groups):
        clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, random_state=seed)
        pred[te] = clf.fit(X[tr], y[tr]).predict(X[te])
    return pred


def boot_indices(groups, B=C.N_BOOT, seed=0):
    """Block bootstrap: resample the 30-min blocks with replacement; returns a list of event-index arrays."""
    rng = np.random.default_rng(seed)
    ug, inv = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inv == k) for k in range(len(ug))]
    out = []
    for _ in range(B):
        pick = rng.integers(0, len(ug), len(ug))
        out.append(np.concatenate([members[k] for k in pick]))
    return out


def boot_metric(y, pred, kind, idxs):
    return np.array([metric(y[i], pred[i], kind) for i in idxs])


def permute_labels(y, groups, rng, within=False):
    """E's null: permute labels at the level of the CV group (labels constant within a 30-min block stay so),
    or (within=True, has_dropout grouped by day) permute event labels within each group."""
    y = np.asarray(y); yn = y.copy()
    if within:
        for gval in np.unique(groups):
            ii = np.flatnonzero(groups == gval); yn[ii] = rng.permutation(yn[ii])
        return yn
    ug, first = np.unique(groups, return_index=True)
    lab = y[first]
    if not all((y[groups == u] == l).all() for u, l in zip(ug, lab)):
        # label not constant within group: permute whole-group label vectors by block order
        raise ValueError("label not constant within group")
    perm = dict(zip(ug, rng.permutation(lab)))
    return np.array([perm[g] for g in groups])


def holm(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p)
    adj = np.empty(n); run = 0.0
    for r, i in enumerate(o):
        run = max(run, (n - r) * p[i]); adj[i] = min(1.0, run)
    return adj
