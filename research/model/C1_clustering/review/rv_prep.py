"""Independent reviewer: build crops, own aligned-corr, own AWD, R2s0 / R3 matrices; cache to data/c1/review."""
import os, sys, numpy as np, torch
ROOT = "/home/augustin/Documents/federated_transformer"; os.chdir(ROOT)
sys.path[:0] = [ROOT, f"{ROOT}/research/model/M4_methods", f"{ROOT}/research/model/M5_long", f"{ROOT}/research/model/C1_clustering"]
import m4, m5, c1
OUT = os.path.join(ROOT, "data", "c1", "review"); os.makedirs(OUT, exist_ok=True)
m, W = m4.load(); x = c1.raw_crops()                       # only use of c1: crop geometry
np.save(f"{OUT}/x.npy", x)
dev = "cuda"
def shiftmax(X, S=50, corr=True):
    """Own implementation: explicit loop over integer lags with zero padding (not FFT)."""
    X = torch.tensor(X, device=dev, dtype=torch.float64 if False else torch.float32)
    if corr: X = X - X.mean(1, keepdim=True); X = X / X.norm(dim=1, keepdim=True)
    n, L = X.shape; best = torch.full((n, n), -1e30, device=dev)
    for s in range(-S, S + 1):
        Y = torch.zeros_like(X)
        if s >= 0: Y[:, s:] = X[:, :L - s]
        else: Y[:, :L + s] = X[:, -s:]
        best = torch.maximum(best, X @ Y.T)                     # <x_i, shift_s(x_j)>
    best = torch.maximum(best, best.T)                          # lag sign symmetry
    if corr: return best.cpu().numpy()
    sq = (X ** 2).sum(1); D = torch.sqrt(torch.clamp(sq[:, None] + sq[None] - 2 * best, min=0)); D.fill_diagonal_(0); return D.cpu().numpy()
P = shiftmax(x); np.save(f"{OUT}/P.npy", P)
lc = np.sign(x) * np.log1p(np.abs(x) / 100.0); np.save(f"{OUT}/D100.npy", shiftmax(lc, corr=False))
m5.OUT = os.path.join(ROOT, "data", "c1")
for s in range(5): np.save(f"{OUT}/emb_s{s}.npy", m5.emb("simclr_tf", s, "final"))
Pc = np.load(f"{ROOT}/data/c1/aligned_corr.npy"); print("max |P_own - P_c1| =", np.abs(P - Pc).max())
