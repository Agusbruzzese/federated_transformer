"""Shared helpers for committee member A (model analyst). Read-only use of M4/M5 code and checkpoints.

Run every script from the repo root, e.g.  .venv/bin/python research/model/K_committee/A_model/a1_probes.py
Embeddings are cached in research/model/K_committee/A_model/cache/ (delete it to recompute).
"""
import os, sys
import numpy as np, torch

ROOT = os.getcwd()
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "research/model/M4_methods"))
sys.path.insert(0, os.path.join(ROOT, "research/model/M5_long"))
import m4, m5                                    # noqa: E402  (m5.emb / m5.crops / m5.probe_all are reused verbatim)
from ssl_central import augment as A, models as Mo   # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache"); os.makedirs(CACHE, exist_ok=True)
DEV = "cuda"
SEEDS = list(range(5))


def emb(method, seed, which):
    """Encoder output (64-d, pre-projection-head) on the eval view of ALL SNR>=20 events, M5 order."""
    f = os.path.join(CACHE, f"{method}_s{seed}_{which}.npy")
    if not os.path.exists(f):
        np.save(f, m5.emb(method, seed, which))
    return np.load(f)


def m4_tf_emb(seed):
    """M4 SimCLR-TF (selected cell 0, 100 epochs ~ 1000 steps) for the training-length comparison."""
    f = os.path.join(CACHE, f"m4_simclr_tf_c0_s{seed}_final.npy")
    if not os.path.exists(f):
        enc = Mo.make("tf"); enc.load_state_dict(torch.load(os.path.join(ROOT, "data/m4/ckpt", f"simclr_tf_c0_s{seed}", "final.pt"), map_location=DEV))
        enc = enc.to(DEV).eval(); x = m5.crops()
        with torch.no_grad():
            np.save(f, np.concatenate([enc(x[i:i + 2048]).cpu().numpy() for i in range(0, len(x), 2048)]))
    return np.load(f)


def load_encoder(method, seed, which):
    enc, _ = m5.build(method)
    enc.load_state_dict(torch.load(m5.ck(method, seed, which), map_location=DEV))
    return enc.to(DEV).eval()


@torch.no_grad()
def embed_x(enc, x, bs=2048):
    """x: torch tensor [n,600] already max-normalised, on DEV."""
    return np.concatenate([enc(x[i:i + bs]).cpu().numpy() for i in range(0, len(x), bs)])


# ------------------------------------------------------------------ label-free metrics
def rankme(Z, eps=1e-7, center=False):
    Z = Z - Z.mean(0) if center else Z
    s = np.linalg.svd(Z.astype(np.float64), compute_uv=False)
    p = s / s.sum() + eps
    return float(np.exp(-(p * np.log(p)).sum()))


def l2n(Z): return Z / np.linalg.norm(Z, axis=1, keepdims=True).clip(1e-12)


def uniformity(Zn, t=2.0):
    d2 = np.maximum(2 - 2 * Zn @ Zn.T, 0)
    iu = np.triu_indices(len(Zn), 1)
    return float(np.log(np.mean(np.exp(-t * d2[iu]))))


def alignment(Z1n, Z2n): return float(np.mean(np.sum((Z1n - Z2n) ** 2, 1)))


# ------------------------------------------------------------------ representation similarity
def linear_cka(X, Y):
    X = X - X.mean(0); Y = Y - Y.mean(0)
    X = X.astype(np.float64); Y = Y.astype(np.float64)
    return float(np.linalg.norm(X.T @ Y) ** 2 / (np.linalg.norm(X.T @ X) * np.linalg.norm(Y.T @ Y)))


def knn_sets(X, k=15):
    """k nearest neighbours (Euclidean on z-scored features, self excluded)."""
    from sklearn.preprocessing import StandardScaler
    from sklearn.neighbors import NearestNeighbors
    Xs = StandardScaler().fit_transform(X)
    idx = NearestNeighbors(n_neighbors=k + 1).fit(Xs).kneighbors(Xs, return_distance=False)
    return idx[:, 1:]


def knn_overlap(I, J):
    k = I.shape[1]
    return float(np.mean([len(set(a) & set(b)) / k for a, b in zip(I, J)]))
