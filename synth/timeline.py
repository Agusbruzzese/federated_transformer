"""When events happen: an inhomogeneous Poisson process over the real recording schedule (report 07).

rate(hour, daytype)          = real family events / recorded hours, per hour of day, weekday vs weekend
P(stratum | 6-h bin, daytype) = real stratum mix, smoothed toward the overall mix (Dirichlet, ALPHA events)
"""
from __future__ import annotations

import os
import re

import numpy as np
import pandas as pd

from . import generator as G

DATASET = os.path.join(G.ROOT, "Spintronic Sensor Dataset for Vehicle Detection and Car Model Recognition "
                                "(14 Days, Unlabeled with Timestamps)")
ALPHA = 10.0


def coverage():
    """Recording intervals [(start, end)] from the run folder names (no files are missing inside a run)."""
    out = []
    for d in sorted(os.listdir(DATASET)):
        r = re.match(r"Run (\d{8}) (\d\d)h(\d\d)m(\d\d)s - (\d{8}) (\d\d)h(\d\d)m(\d\d)s", d)
        if r:
            g = r.groups()
            out.append((pd.Timestamp(f"{g[0]} {g[1]}:{g[2]}:{g[3]}"), pd.Timestamp(f"{g[4]} {g[5]}:{g[6]}:{g[7]}")))
    return out


def minutes(cov, days=None):
    """One row per recorded minute: start time, hour, daytype (0 weekday, 1 weekend); optionally only `days`."""
    t = pd.DatetimeIndex(np.concatenate([pd.date_range(a.ceil("min"), b, freq="min", inclusive="left") for a, b in cov]))
    if days is not None:
        t = t[np.isin(t.date.astype(str), list(days))]
    return pd.DataFrame(dict(t=t, hour=t.hour, wkend=(t.dayofweek >= 5).astype(int)))


class Timeline:
    def __init__(self, rate, mix, keys):
        self.rate, self.mix, self.keys = rate, mix, keys     # rate [24, 2] events/h; mix [4, 2, n_keys]

    @classmethod
    def fit(cls, times, keys, mins, allowed):
        """times: wall times of real family events; keys: their strata; mins: recorded minutes they come from;
        allowed: strata the generator can draw (others are dropped from the mix)."""
        t = pd.DatetimeIndex(times); allowed = sorted(allowed)
        ok = np.array([k in allowed for k in keys])
        t = t[ok]; keys = [k for k, o in zip(keys, ok) if o]
        expo = mins.groupby(["hour", "wkend"]).size().unstack(fill_value=0).reindex(index=range(24), columns=[0, 1], fill_value=0) / 60
        cnt = pd.DataFrame(dict(hour=t.hour, wkend=(t.dayofweek >= 5).astype(int))).groupby(["hour", "wkend"]).size() \
            .unstack(fill_value=0).reindex(index=range(24), columns=[0, 1], fill_value=0)
        rate = (cnt / expo.replace(0, np.nan)).fillna(0).to_numpy()
        ki = np.array([allowed.index(k) for k in keys]); hb = t.hour.to_numpy() // 6; we = (t.dayofweek >= 5).astype(int)
        overall = np.bincount(ki, minlength=len(allowed)) / len(ki)
        mix = np.zeros((4, 2, len(allowed)))
        for b in range(4):
            for w in range(2):
                c = np.bincount(ki[(hb == b) & (we == w)], minlength=len(allowed))
                mix[b, w] = (c + ALPHA * overall) / (c.sum() + ALPHA)
        return cls(rate, mix, allowed)

    def simulate(self, mins, rng):
        """Event times and strata over the recorded minutes `mins`."""
        rng = np.random.default_rng(rng)
        lam = self.rate[mins.hour.to_numpy(), mins.wkend.to_numpy()] / 60
        n = rng.poisson(lam)
        rep = np.repeat(np.arange(len(mins)), n)
        t = mins.t.to_numpy()[rep] + pd.to_timedelta(rng.uniform(0, 60, len(rep)), unit="s")
        hb = mins.hour.to_numpy()[rep] // 6; we = mins.wkend.to_numpy()[rep]
        u = rng.random(len(rep)); cdf = np.cumsum(self.mix[hb, we], 1)
        k = (u[:, None] > cdf).sum(1).clip(max=len(self.keys) - 1)
        order = np.argsort(t)
        return pd.DatetimeIndex(t[order]), [self.keys[i] for i in k[order]]
