"""LORO folds, firewall, per-fold background pool, amp_resid helper."""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from ssl_central import data as FD
from . import config as C

_CACHE = {}


def load_all():
    """F's loader (all events, all SNR) with per-fold bookkeeping. Cached per process."""
    if "m" not in _CACHE:
        m, W, EM, RM = FD.load_all()
        m = m.drop(columns=["split"])
        m["y_log_dur"] = np.log10(m.m8_dur10)
        m["log_snr"] = np.log10(m.snr)
        _CACHE.update(m=m, W=W)
    return _CACHE["m"], _CACHE["W"]


def tier_mask(m, tier=C.TIER):
    return (m.snr >= tier).to_numpy() if tier is not None else np.ones(len(m), bool)


def train_index(m, k, tier=C.TIER):
    """SSL-training (and selection) events of fold k: tier events of the 3 other runs."""
    return np.flatnonzero(tier_mask(m, tier) & (m.run_id != k).to_numpy())


def test_index(m, k, tier=C.TIER):
    return np.flatnonzero(tier_mask(m, tier) & (m.run_id == k).to_numpy())


def firewall(m, k, used_idx):
    """Assert on event_id sets that no event of test run k (any SNR) is in `used_idx`."""
    ids = m.event_id.to_numpy()
    test_all = set(ids[(m.run_id == k).to_numpy()])
    used = set(ids[used_idx])
    assert len(used) == len(used_idx), "duplicate event ids"
    assert not (used & test_all), f"fold {k}: test-run event in training/selection"
    assert (m.run_id.to_numpy()[used_idx] != k).all()
    return True


def fold_pool(k):
    """F's background pool minus every stretch from test run k (J2)."""
    bg = np.load(os.path.join(C.F_OUT, "bgpool.npy"))
    meta = pd.read_parquet(os.path.join(C.F_OUT, "bgpool_meta.parquet"))
    assert len(meta) == len(bg)
    tag = "Run 2025" + C.RUN_NAME[k]
    keep = ~meta.run.str.startswith(tag).to_numpy()
    assert not meta.run[keep].str.startswith(tag).any()
    return bg[keep], meta[keep].reset_index(drop=True)


def amp_resid(log_amp, log_snr, tr, te):
    """J5: OLS of log_amp on log_snr fitted on `tr` only; returns (resid_tr, resid_te)."""
    b, a = np.polyfit(log_snr[tr], log_amp[tr], 1)
    return log_amp[tr] - (a + b * log_snr[tr]), log_amp[te] - (a + b * log_snr[te])
