"""Congruence tests: synthetic vs held-out real events (report 06).

Fit on training days. The bandwidth is chosen on the validation day (Feb 10); everything else is scored once on
the 4 test days. Every synthetic-vs-test number is reported next to the same number for real TRAINING days vs
test days: that is the natural day-to-day difference, and the bar synthetic data should meet.

Tests (stratum = C2 family x polarity; synthetic counts matched to the real set per stratum):
  V1  classifier two-sample test (C2ST), 5-fold AUC, in 3 spaces: morph8 features, SimCLR-TF M5 embedding, log PSD
  V2  per-feature KS distance of the 19 morph features
  V3  family fidelity: nearest-template family of synthetic events vs their intended family
  V4  data copying: nearest-neighbour distance of synthetic events to the training set vs that of test events
  V5  gallery figure

    python -m synth.validate            # ~2-3 min CPU
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
import scipy.signal as ss
from scipy.stats import ks_2samp
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from . import generator as G

RES = os.path.join(G.ROOT, "research", "model", "S_synth")
BW_GRID = [0.1, 0.2, 0.3, 0.5, 0.8]
CKPT = os.path.join(G.ROOT, "data", "m5", "ckpt", "simclr_tf_s0", "final.pt")


def morph(X):
    from extraction.morph import morph_feats
    return np.array([morph_feats(x[99:501].astype(np.float64), G.FS) for x in X])   # central ~8 s = m8_*


def embed(X):
    import torch
    from ssl_central import models as Mo
    enc = Mo.make("tf"); enc.load_state_dict(torch.load(CKPT, map_location="cpu")); enc.eval()
    with torch.no_grad():
        t = torch.tensor(X / np.abs(X).max(1, keepdims=True))
        return np.concatenate([enc(t[i:i + 512]).numpy() for i in range(0, len(t), 512)])


def logpsd(X):
    f, p = ss.welch(X / np.abs(X).max(1, keepdims=True), fs=G.FS, nperseg=128, axis=1)
    return np.log10(p[:, 1:34] + 1e-12)                    # 0.4-13 Hz


def c2st(A, B, seed=0):
    """AUC of a classifier separating A from B (balanced, 5-fold out-of-fold). 0.5 = indistinguishable."""
    n = min(len(A), len(B)); r = np.random.default_rng(seed)
    A = A[r.choice(len(A), n, replace=False)]; B = B[r.choice(len(B), n, replace=False)]
    X = np.vstack([A, B]); y = np.r_[np.zeros(n), np.ones(n)]; p = np.zeros(2 * n)
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
        p[te] = HistGradientBoostingClassifier(max_iter=200, random_state=seed).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    return float(roc_auc_score(y, p))


def matched_sample(g, keys, rng, bw=None):
    """One synthetic event per real event, in the same stratum (strata the generator lacks are skipped)."""
    keys = [k for k in keys if k in g.strata]
    Xs, Is = [], []
    for k in sorted(set(keys)):
        X, I = g.sample(sum(kk == k for kk in keys), key=k, rng=rng, bw=bw); Xs.append(X); Is.append(I)
    return np.vstack(Xs), pd.concat(Is, ignore_index=True)


def nearest_family(X, g):
    """Family of the stratum template with the highest shift-aligned correlation (+-1 s, no stretch)."""
    U, _ = G.smooth_norm(X); best = np.full(len(U), -np.inf); fam = np.zeros(len(U), int)
    for k, S in g.strata.items():
        s, lag = G.best_warp(U, S["T"])
        T = G.warp(S["T"], s, lag)
        c = (U * T).sum(1) / (np.linalg.norm(U, axis=1) * np.linalg.norm(T, axis=1))
        up = c > best; best[up] = c[up]; fam[up] = k[0]
    return fam


def main():
    os.makedirs(RES, exist_ok=True); os.makedirs(G.OUT, exist_ok=True)
    m, x = G.load_events()
    keys = G.strata_of(m, x)
    g = G.Generator.fit(m, x)
    ok = np.array([k in g.strata for k in keys])
    role = m.role.to_numpy()
    tr, va, te = (role == "train") & ok, (role == "val") & ok, (role == "test") & ok
    kte = [keys[i] for i in np.flatnonzero(te)]; kva = [keys[i] for i in np.flatnonzero(va)]
    out = dict(n=dict(train=int(tr.sum()), val=int(va.sum()), test=int(te.sum()),
                      unmodelled_test=int(((role == "test") & ~ok).sum())), strata=g.summary().to_dict("records"))

    # ---- bandwidth on the validation day (C2ST in embedding space; copying guard via V4 below)
    Eva = embed(x[va]); Etr = embed(x[tr])
    sel = {}
    for bw in BW_GRID:
        Xs, _ = matched_sample(g, kva, 1, bw=bw)
        sel[bw] = c2st(Eva, embed(Xs))
    bw = min(sel, key=sel.get); g.bw = bw
    out["bandwidth"] = dict(chosen=bw, val_auc_embedding={str(k): round(v, 3) for k, v in sel.items()})
    print("bandwidth", out["bandwidth"], flush=True)
    g.save(os.path.join(G.OUT, "model.npz"))

    # ---- test-day sets: real test, synthetic (matched strata), real train (matched strata, the baseline)
    Xte = x[te]; Xsy, Isy = matched_sample(g, kte, 2)
    r = np.random.default_rng(3); itr = np.flatnonzero(tr); ktr = np.array([str(keys[i]) for i in itr])
    pick = []
    for k in sorted(set(kte)):
        pool = itr[ktr == str(k)]; need = sum(kk == k for kk in kte)
        pick += list(r.choice(pool, need, replace=need > len(pool)))
    Xtr = x[np.array(pick)]; ftr = m.family.to_numpy()[np.array(pick)]

    spaces = dict(morph8=morph, embedding=embed, logpsd=logpsd)
    F = {name: (f(Xte), f(Xsy), f(Xtr)) for name, f in spaces.items()}
    # V1
    out["V1_c2st_auc"] = {name: dict(synthetic_vs_test=round(c2st(a, b), 3), train_vs_test=round(c2st(a, c), 3))
                          for name, (a, b, c) in F.items()}
    # V1 per family (embedding)
    fte = np.array([k[0] for k in kte]); fsy = Isy.family.to_numpy()
    out["V1_c2st_auc_embedding_by_family"] = {
        int(f): dict(n=int((fte == f).sum()), synthetic_vs_test=round(c2st(F["embedding"][0][fte == f], F["embedding"][1][fsy == f]), 3),
                     train_vs_test=round(c2st(F["embedding"][0][fte == f], F["embedding"][2][ftr == f]), 3))
        for f in sorted(set(fte)) if (fte == f).sum() >= 40}
    # V2
    from extraction.morph import MORPH_NAMES
    a, b, c = F["morph8"]
    ks = pd.DataFrame([dict(feature=n, synthetic_vs_test=ks_2samp(a[:, j], b[:, j]).statistic,
                            train_vs_test=ks_2samp(a[:, j], c[:, j]).statistic) for j, n in enumerate(MORPH_NAMES)])
    out["V2_ks_morph8"] = ks.round(3).to_dict("records")
    out["V2_ks_median"] = dict(synthetic_vs_test=round(float(ks.synthetic_vs_test.median()), 3),
                               train_vs_test=round(float(ks.train_vs_test.median()), 3))
    # V3
    out["V3_family_fidelity"] = dict(synthetic=round(float((nearest_family(Xsy, g) == fsy).mean()), 3),
                                     real_test=round(float((nearest_family(Xte, g) == fte).mean()), 3))
    # V4 (embedding space; distance to the nearest TRAINING event)
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=1).fit(Etr)
    d_sy = nn.kneighbors(F["embedding"][1])[0][:, 0]; d_te = nn.kneighbors(F["embedding"][0])[0][:, 0]
    out["V4_copying"] = dict(median_nn_dist_synthetic=round(float(np.median(d_sy)), 4),
                             median_nn_dist_test=round(float(np.median(d_te)), 4),
                             ratio=round(float(np.median(d_sy) / np.median(d_te)), 3),
                             frac_synthetic_below_test_p5=round(float((d_sy < np.quantile(d_te, 0.05)).mean()), 3))
    # SNR proxy (peak / noise sigma) distribution
    Ate = np.abs(ss.sosfiltfilt(G._SOS, Xte.astype(np.float64), axis=1)).max(1) / m.sigma_ev.to_numpy()[te]
    out["snr_peak"] = dict(test_p10_p50_p90=np.quantile(Ate, [.1, .5, .9]).round(1).tolist(),
                           synthetic_p10_p50_p90=np.quantile(Isy.snr_peak, [.1, .5, .9]).round(1).tolist(),
                           ks=round(float(ks_2samp(Ate, Isy.snr_peak).statistic), 3))
    json.dump(out, open(os.path.join(RES, "results.json"), "w"), indent=1)
    gallery(g, Xte, fte, Xsy, fsy)
    print(json.dumps({k: v for k, v in out.items() if k not in ("strata", "V2_ks_morph8")}, indent=1))


def gallery(g, Xte, fte, Xsy, fsy, n=6):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fams = sorted(set(fte)); t = np.arange(G.N) / G.FS; r = np.random.default_rng(0)
    fig, axs = plt.subplots(len(fams), 2, figsize=(12, 2.0 * len(fams)), sharex=True)
    for row, f in enumerate(fams):
        for col, (X, F, lab) in enumerate([(Xte, fte, "real (test days)"), (Xsy, fsy, "synthetic")]):
            ax = axs[row, col]; i = np.flatnonzero(F == f)
            for j in r.choice(i, min(n, len(i)), replace=False):
                ax.plot(t, X[j] / np.abs(X[j]).max(), lw=0.6)
            ax.set_title(f"family {f}: {lab}", fontsize=8); ax.set_ylim(-1.05, 1.05); ax.tick_params(labelsize=7)
    axs[-1, 0].set_xlabel("s"); axs[-1, 1].set_xlabel("s")
    fig.suptitle("Real vs synthetic events, peak-normalised (6 random per family)"); fig.tight_layout()
    fig.savefig(os.path.join(RES, "gallery_real_vs_synth.png"), dpi=90)


if __name__ == "__main__":
    sys.exit(main())
