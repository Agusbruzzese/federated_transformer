"""Fit / sample / save / load of the family-conditioned generator. CPU, numpy + scipy only."""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import scipy.signal as ss

N = 600                                     # samples per crop (12 s)
FS = 50.0
C0 = (N - 1) / 2.0                          # crop centre = detector energy centroid
LP_HZ = 5.0                                 # shape smoother (same as the morph features)
S_GRID = np.exp(np.linspace(np.log(0.6), np.log(1.7), 31))   # stretch factors searched (>1 = longer event)
MAX_LAG = 50                                # +-1 s shift searched
N_ITER = 4                                  # template refinement passes
VAR_KEEP = 0.95                             # PCA variance kept (capped to K_MAX)
K_MAX = 20                                  # chosen on dev days carved from training (report 06)
MIN_STRATUM = 30                            # strata with fewer training events are dropped
NOISE_TAPS = 31                             # noise-colouring FIR length (pool rows 640 -> 610 valid samples)
EDGE_N = 100                                # 2 s at each crop end, used to measure real background noise
SHORT_FAMILIES = (0, 1, 2, 4, 6)            # families whose events leave the crop edges quiet (dur ~2-3.5 s)
TEST_DAYS = ["2025-02-07", "2025-02-13", "2025-02-19", "2025-02-25"]   # M4/M5 split
VAL_DAY = "2025-02-10"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data", "synth")
FAMILIES = os.path.join(ROOT, "research", "model", "C2_lookalikes", "families.csv")
_SOS = ss.butter(4, LP_HZ, fs=FS, output="sos")
_SOS_FEAT = ss.butter(4, 8.0, fs=FS, output="sos")            # extraction feature-path low-pass (params.FEATURE_LP_HZ)


# ------------------------------------------------------------------ data
def load_events():
    """SNR >= 20 events as raw-count 600-sample crops on the exact 50 Hz grid (= ssl_central eval view before
    max-normalisation), with role (train/val/test as in M4/M5), C2 family and noise sigma."""
    import torch
    from ssl_central import data as D, augment as A, config as C
    m, W, _, _ = D.load_all()
    keep = (m.snr >= 20).to_numpy()
    m = m[keep].reset_index(drop=True); W = W[keep]
    with torch.no_grad():
        Wt = torch.tensor(W); fs = torch.tensor(m.fs_eff.to_numpy(np.float32))
        x = A.render(Wt, fs, torch.full((len(W),), C.CENTRE), torch.ones(len(W))).numpy().astype(np.float32)
    day = m.date.astype(str).to_numpy()
    m["role"] = np.where(np.isin(day, TEST_DAYS), "test", np.where(day == VAL_DAY, "val", "train"))
    fam = pd.read_csv(FAMILIES)
    m = m.merge(fam, on="event_id", how="left")
    assert m.family.notna().all()
    m["family"] = m.family.astype(int)
    return m, x


def bgpool(exclude_days=TEST_DAYS + [VAL_DAY]):
    """Real noise stretches [M, 640] at unit robust sigma, training days only."""
    d = os.path.join(ROOT, "data", "ssl_central")
    bg = np.load(os.path.join(d, "bgpool.npy")); meta = pd.read_parquet(os.path.join(d, "bgpool_meta.parquet"))
    ok = ~np.isin(pd.to_datetime(meta.wall_time).dt.date.astype(str).to_numpy(), exclude_days)
    return bg[ok]


# ------------------------------------------------------------------ warping
def _at(X, pos):
    """Linear interpolation of rows X [n, N] at positions pos [n, N]; zero (= baseline) outside the crop."""
    i0 = np.clip(np.floor(pos).astype(int), 0, N - 2); f = pos - i0
    r = np.arange(len(X))[:, None]
    v = X[r, i0] * (1 - f) + X[r, i0 + 1] * f
    return np.where((pos >= 0) & (pos <= N - 1), v, 0.0)


def warp(shape, s, lag):
    """Event from shape: x(t) = shape(C0 + (t - C0 - lag) / s). s > 1 stretches, lag > 0 delays."""
    t = np.arange(N)[None, :]
    return _at(np.atleast_2d(shape), C0 + (t - C0 - np.asarray(lag, float)[:, None]) / np.asarray(s, float)[:, None])


def register(x, s, lag):
    """Inverse of warp: shape(tau) = x(C0 + s (tau - C0) + lag)."""
    t = np.arange(N)[None, :]
    return _at(np.atleast_2d(x), C0 + np.asarray(s, float)[:, None] * (t - C0) + np.asarray(lag, float)[:, None])


def best_warp(U, T):
    """For each row of U, the (stretch, lag) maximising the correlation with warp(T, stretch, lag)."""
    n = len(U); L = 2 * N
    FU = np.fft.rfft(U, n=L)
    best = np.full(n, -np.inf); bs = np.ones(n); bl = np.zeros(n)
    lags = np.r_[np.arange(MAX_LAG + 1), np.arange(-MAX_LAG, 0)]
    for s in S_GRID:
        Ts = warp(T, [s], [0.0])[0]
        c = np.fft.irfft(np.conj(np.fft.rfft(Ts, n=L))[None, :] * FU, n=L)
        c = np.concatenate([c[:, :MAX_LAG + 1], c[:, -MAX_LAG:]], 1) / (np.linalg.norm(Ts) + 1e-12)
        j = c.argmax(1); v = c[np.arange(n), j]
        up = v > best
        best[up] = v[up]; bs[up] = s; bl[up] = lags[j[up]]
    return bs, bl


# ------------------------------------------------------------------ fit
def smooth_norm(x):
    """5 Hz zero-phase low-pass, then divide by the peak |.|. Returns (unit shapes, peak amplitude)."""
    y = ss.sosfiltfilt(_SOS, x.astype(np.float64), axis=1)
    A = np.abs(y).max(1)
    return y / A[:, None], A


def fit_stratum(x, sigma):
    U, A = smooth_norm(x)
    T = np.median(U, 0)
    for _ in range(N_ITER):
        s, lag = best_warp(U, T)
        R = register(U, s, lag)
        T = np.median(R, 0)
        # gauge: the template is the TYPICAL event (geometric-mean stretch, median lag), not an arbitrary one
        T = warp(T, [np.exp(np.mean(np.log(s)))], [np.median(lag)])[0]
        T /= np.abs(T).max()
    s, lag = best_warp(U, T)
    R = register(U, s, lag)
    mu = R.mean(0)
    _, sv, Vt = np.linalg.svd(R - mu, full_matrices=False)
    ev = sv ** 2 / (sv ** 2).sum()
    K = int(np.clip(np.searchsorted(np.cumsum(ev), VAR_KEEP) + 1, 2, K_MAX))
    V = Vt[:K]
    Z = (R - mu) @ V.T
    P = np.column_stack([Z, np.log(s), lag, np.log(A), np.log(sigma)])
    cov = np.cov(P, rowvar=False) + 1e-9 * np.eye(P.shape[1])
    return dict(T=T, mu=mu, V=V, P=P, chol=np.linalg.cholesky(cov), K=K, var_kept=float(np.cumsum(ev)[K - 1]),
                corr_median=float(np.median((U * warp(T, s, lag)).sum(1) /
                                            (np.linalg.norm(U, axis=1) * np.linalg.norm(warp(T, s, lag), axis=1)))))


def fit_noise_filter(x, sigma, bg):
    """Linear-phase FIR that maps pool noise (unit robust sigma, never resampled) onto the spectrum of the quiet
    crop edges of real events in units of their sigma_ev. It absorbs three real differences: the pool is
    whiter, sigma_ev over-states the local noise, and the 50 Hz rendering (linear interpolation) damps high
    frequencies of real crops but was never applied to the pool."""
    E = np.concatenate([x[:, :EDGE_N], x[:, -EDGE_N:]]) / np.concatenate([sigma, sigma])[:, None]
    f, pe = ss.welch(E - E.mean(1, keepdims=True), fs=FS, nperseg=64, axis=1)
    _, pb = ss.welch(bg[:, :EDGE_N] - bg[:, :EDGE_N].mean(1, keepdims=True), fs=FS, nperseg=64, axis=1)
    gain = np.sqrt(np.median(pe, 0) / np.median(pb, 0))
    return ss.firwin2(NOISE_TAPS, f, gain, fs=FS)


def colour(bg, taps):
    return ss.fftconvolve(bg, taps[None, :], mode="valid", axes=1).astype(np.float32)


def strata_of(m, x):
    """(family, polarity) key per event; family -1 (unassigned in C2) gets key None."""
    y = ss.sosfiltfilt(_SOS, x.astype(np.float64), axis=1)
    pol = np.sign(y[np.arange(len(y)), np.abs(y).argmax(1)]).astype(int)
    return [None if f < 0 else (int(f), int(p)) for f, p in zip(m.family.to_numpy(), pol)]


class Generator:
    def __init__(self, strata, bw, bg, taps):
        self.strata = strata                  # {key: dict(...)}
        self.bw = bw                          # smoothed-bootstrap bandwidth (fraction of the stratum covariance)
        self.taps = taps                      # noise-colouring FIR
        self.bg = colour(bg, taps)            # coloured noise rows [M, 610]
        keys = sorted(strata); n = np.array([len(strata[k]["P"]) for k in keys], float)
        self.keys, self.weights = keys, n / n.sum()

    @classmethod
    def fit(cls, m, x, bw=0.3, roles=("train",), sel=None, bg_exclude=None, min_stratum=None):
        """sel (bool mask) overrides roles; bg_exclude: days whose noise must not be used (default: val+test)."""
        keys = strata_of(m, x)
        sel = m.role.isin(roles).to_numpy() if sel is None else sel
        min_stratum = MIN_STRATUM if min_stratum is None else min_stratum
        strata = {}
        bg = bgpool() if bg_exclude is None else bgpool(bg_exclude)
        short = sel & m.family.isin(SHORT_FAMILIES).to_numpy()
        taps = fit_noise_filter(x[short], m.sigma_ev.to_numpy()[short], bg)
        for k in sorted({k for k in keys if k is not None}):
            i = sel & np.array([kk == k for kk in keys])
            if i.sum() >= min_stratum:
                strata[k] = fit_stratum(x[i], m.sigma_ev.to_numpy()[i])
                strata[k]["n"] = int(i.sum())
        return cls(strata, bw, bg, taps)

    def sample(self, n, key=None, rng=None, bw=None):
        """n synthetic crops [n, 600] (counts). key=None draws strata by their training proportions.
        Returns (x, info DataFrame with family, polarity, stretch, lag, amp, sigma, snr_peak)."""
        rng = np.random.default_rng(rng)
        ks = [key] * n if key is not None else [self.keys[j] for j in rng.choice(len(self.keys), n, p=self.weights)]
        return self.sample_keys(ks, rng, bw)

    def sample_keys(self, ks, rng=None, bw=None):
        """One synthetic event per stratum key in ks, returned in the same order. Only existing strata can be
        drawn: the generator cannot create a family that is not in the training data."""
        rng = np.random.default_rng(rng); bw = self.bw if bw is None else bw; n = len(ks)
        unknown = set(ks) - set(self.strata)
        if unknown:
            raise KeyError(f"strata not in the model: {sorted(unknown)}")
        X = np.empty((n, N), np.float32); rows = []
        ks_arr = np.array([f"{k[0]}_{k[1]}" for k in ks])
        for k in sorted(set(ks)):
            idx = np.flatnonzero(ks_arr == f"{k[0]}_{k[1]}"); S = self.strata[k]; K = S["K"]
            base = S["P"][rng.integers(0, len(S["P"]), len(idx))]
            p = base + bw * rng.standard_normal(base.shape) @ S["chol"].T
            z, ls, lag, la, lsig = p[:, :K], p[:, K], p[:, K + 1], p[:, K + 2], p[:, K + 3]
            s = np.exp(np.clip(ls, np.log(S_GRID[0]), np.log(S_GRID[-1]))); lag = np.clip(lag, -MAX_LAG, MAX_LAG)
            shape = S["mu"] + z @ S["V"]
            shape /= np.abs(shape).max(1, keepdims=True)
            # the feature path low-passes real events at 8 Hz; warping (linear interpolation) adds content above it
            clean = np.exp(la)[:, None] * ss.sosfiltfilt(_SOS_FEAT, warp(shape, s, lag), axis=1)
            b = self.bg[rng.integers(0, len(self.bg), len(idx))]
            o = rng.integers(0, self.bg.shape[1] - N + 1, len(idx))
            noise = b[np.arange(len(idx))[:, None], o[:, None] + np.arange(N)[None, :]]
            X[idx] = (clean + np.exp(lsig)[:, None] * noise).astype(np.float32)
            for j, i in enumerate(idx):
                rows.append(dict(i=i, family=k[0], polarity=k[1], stretch=s[j], lag_s=lag[j] / FS,
                                 amp=float(np.exp(la[j])), sigma=float(np.exp(lsig[j]))))
        info = pd.DataFrame(rows).sort_values("i").drop(columns="i").reset_index(drop=True)
        info["snr_peak"] = info.amp / info.sigma
        return X, info

    # ---------------------------------------------------------- persistence
    def save(self, path):
        d = dict(bw=self.bw, taps=self.taps, keys=np.array([f"{k[0]}_{k[1]}" for k in self.keys]))
        for k in self.keys:
            for f, v in self.strata[k].items():
                d[f"{k[0]}_{k[1]}__{f}"] = v
        np.savez(path, **d)

    @classmethod
    def load(cls, path):
        z = np.load(path); strata = {}
        for ks in z["keys"]:
            f, p = map(int, ks.split("_")); pre = f"{ks}__"
            strata[(f, p)] = {k[len(pre):]: (z[k].item() if z[k].ndim == 0 else z[k]) for k in z.files if k.startswith(pre)}
        return cls(strata, float(z["bw"]), bgpool(), z["taps"])

    def summary(self):
        return pd.DataFrame([dict(family=k[0], polarity=k[1], n_train=S["n"], K=S["K"], var_kept=round(S["var_kept"], 3),
                                  fit_corr_median=round(S["corr_median"], 3)) for k, S in self.strata.items()])
