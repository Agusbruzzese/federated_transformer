"""Event loading, the PREREG split, 30-min blocks, P1 targets and the holdout firewall."""
from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd

from . import config as C


def load_all():
    """All kept events of all runs (every SNR), concatenated in run order.
    Returns (meta DataFrame with a global index `gi`, windows float32 [N,1050], evmask, repmask)."""
    M, W, EM, RM = [], [], [], []
    for f in sorted(glob.glob(os.path.join(C.EVENTS, "run*_meta.parquet"))):
        stem = f[:-len("_meta.parquet")]
        m = pd.read_parquet(f).sort_values("row")
        w = np.load(stem + "_windows.npy"); em = np.load(stem + "_evmask.npy"); rm = np.load(stem + "_repmask.npy")
        assert len(w) == len(m) and (m.row.to_numpy() == np.arange(len(m))).all()
        M.append(m); W.append(w); EM.append(em); RM.append(rm)
    m = pd.concat(M, ignore_index=True)
    m["gi"] = np.arange(len(m))
    m["blk30"] = m.wall_time.dt.floor("30min").astype(str)
    m["hbin"] = m.hour // 6
    m["t_s"] = (m.wall_time - pd.Timestamp("2025-02-01")).dt.total_seconds()
    m["split"] = np.where(m.run_id == C.HOLDOUT_RUN_ID, "holdout",
                          np.where(m.date == C.VAL_DATE, "val",
                                   np.where(m.run_id.isin(C.TRAIN_RUN_IDS), "train", "none")))
    assert (m.split != "none").all()
    # P1 targets (addendum A1: central ~8 s crop) and the 21 s secondary versions
    m["y_log_amp"] = m.m8_log_amp
    m["y_log_dur"] = np.log10(m.m8_dur10)
    m["y_polarity"] = m.m8_polarity.astype(int)
    m["y_n_lobes"] = m.m8_n_lobes.clip(upper=4).astype(int)
    m["y21_log_amp"] = m.m_log_amp
    m["y21_log_dur"] = np.log10(m.m_dur10)
    m["y21_polarity"] = m.m_polarity.astype(int)
    m["y21_n_lobes"] = m.m_n_lobes.clip(upper=4).astype(int)
    W = np.concatenate(W); EM = np.concatenate(EM); RM = np.concatenate(RM)
    m["sigma_ev"] = noise_sigma(W, EM, RM)
    return m, W, EM, RM


def noise_sigma(W, EM, RM, pad=50):
    """Robust sigma of each window outside its [onset, offset] (+-1 s) and outside the repair mask (I5).
    Used only to scale the added background noise; never an encoder input."""
    out = np.empty(len(W), np.float32)
    k = np.ones(2 * pad + 1)
    for i in range(len(W)):
        ev = np.convolve(EM[i].astype(float), k, "same") > 0
        ok = ~ev & ~RM[i]
        x = W[i][ok] if ok.sum() >= 150 else W[i][~RM[i]]
        x = x - np.median(x)
        out[i] = 1.4826 * np.median(np.abs(x)) + 1e-6
    return out


def tier_mask(m, tier=C.TIER):
    return (m.snr >= tier).to_numpy() if tier is not None else np.ones(len(m), bool)


def split_index(m, split, tier=C.TIER):
    return np.flatnonzero((m.split == split).to_numpy() & tier_mask(m, tier))


def firewall(m, train_idx, sel_idx=None):
    """Assert (on event_id sets) that no holdout event enters training or selection, and no validation
    event enters training."""
    ids = m.event_id.to_numpy()
    hold = set(ids[(m.split == "holdout").to_numpy()])
    val = set(ids[(m.split == "val").to_numpy()])
    tr = set(ids[train_idx])
    assert not (tr & hold), "holdout event in training"
    assert not (tr & val), "validation event in training"
    assert all(ids[train_idx] != "") and (m.run_id.to_numpy()[train_idx] != C.HOLDOUT_RUN_ID).all()
    assert not (m.date.to_numpy()[train_idx] == C.VAL_DATE).any()
    if sel_idx is not None:
        se = set(ids[sel_idx])
        assert not (se & hold), "holdout event in selection"
        assert not (se & tr), "training event in selection set"
    return True
