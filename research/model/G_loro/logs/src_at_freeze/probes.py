"""v3 probes: kNN with in-fold amp_resid (J5), logistic-regression leakage probe (J9), nulls (J10)."""
from __future__ import annotations

import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ssl_central.probes import _z, boot_indices, holm, metric, permute_labels, skill  # noqa: F401  (re-export)
from . import config as C


def splits(groups, how="gkf5"):
    dummy = np.zeros(len(groups))
    if how == "logo":
        return list(LeaveOneGroupOut().split(dummy, dummy, groups))
    return list(GroupKFold(n_splits=5).split(dummy, dummy, groups))


def target_y(m_sub, t):
    """Raw target column for kNN; amp_resid is built inside folds from (log_amp, log_snr)."""
    if t == "amp_resid":
        return None
    col = {"log_dur": "y_log_dur", "polarity": "y_polarity", "n_lobes": "y_n_lobes"}[t]
    return m_sub[col].to_numpy()


def knn_oof(X, m_sub, t, groups, how="gkf5", k=C.K_NN, sp=None):
    """Returns (y_true_oof, y_pred_oof). For amp_resid, y_true_oof is the residual under each event's own CV
    training fold's SNR fit (J5)."""
    X = np.asarray(X, float)
    X = X[:, None] if X.ndim == 1 else X
    kind = C.KIND[t]
    sp = sp if sp is not None else splits(groups, how)
    if t == "amp_resid":
        la = m_sub.m8_log_amp.to_numpy(float); ls = m_sub.log_snr.to_numpy(float)
        ytrue = np.empty(len(X)); pred = np.empty(len(X))
    else:
        y = target_y(m_sub, t); ytrue = y
        pred = np.empty(len(X), float if kind == "reg" else y.dtype)
    for tr, te in sp:
        a, b = _z(X[tr], X[te])
        if t == "amp_resid":
            b_ = np.polyfit(ls[tr], la[tr], 1)
            ytr = la[tr] - np.polyval(b_, ls[tr]); ytrue[te] = la[te] - np.polyval(b_, ls[te])
        else:
            ytr = y[tr]
        mdl = (KNeighborsRegressor if kind == "reg" else KNeighborsClassifier)(n_neighbors=k)
        pred[te] = mdl.fit(a, ytr).predict(b)
    return ytrue, pred


def lr_oof(X, y, sp):
    X = np.asarray(X, float)
    X = X[:, None] if X.ndim == 1 else X
    pred = np.empty_like(y)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        for tr, te in sp:
            if len(np.unique(y[tr])) < 2:
                pred[te] = y[tr][0]; continue
            clf = make_pipeline(StandardScaler(), LogisticRegression(C=C.LR_C, max_iter=2000))
            pred[te] = clf.fit(X[tr], y[tr]).predict(X[te])
    return pred


def null_labels(y, groups, p, within):
    return permute_labels(y, groups, np.random.default_rng(1000 + p), within=within)
