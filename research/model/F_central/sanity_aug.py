"""Sanity checks: (1) NT-Xent guard unit test; (2) figure of 8 TRAIN events x 2 augmented views;
(3) per-augmentation statistics; (4) holdout firewall asserts.  python research/model/F_central/sanity_aug.py"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
from ssl_central import augment as A, config as C, data as D, models as Mo  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
dev = "cuda"

# (1) guard: 4 events, 0 and 1 are 10 s apart -> their cross pairs must be excluded
z = torch.randn(4, 8, device=dev)
t = torch.tensor([0.0, 10.0, 500.0, 1000.0], dtype=torch.float64, device=dev)
l_g, f_g = Mo.nt_xent(z, z + 0.01, t, 0.1)
l_n, f_n = Mo.nt_xent(z, z + 0.01, t + torch.tensor([0, 1e4, 2e4, 3e4], dtype=torch.float64, device=dev), 0.1)
print(f"guard test: masked fraction {f_g:.4f} (expect 8 ordered view pairs / 48 negatives = {8/48:.4f}); no-near fraction {f_n}")
assert abs(f_g - 8 / 48) < 1e-9 and f_n == 0

# (4) firewall
m, W, EM, RM = D.load_all()
tr = D.split_index(m, "train"); va = D.split_index(m, "val"); ho = D.split_index(m, "holdout")
D.firewall(m, tr, va)
ids = m.event_id.to_numpy()
assert not set(ids[tr]) & set(ids[ho]) and not set(ids[va]) & set(ids[ho]) and not set(ids[tr]) & set(ids[va])
print("firewall ok:", len(tr), len(va), len(ho))

# (2) figure
rng = np.random.default_rng(1)
pick = np.sort(rng.choice(tr, 8, replace=False))
s = m.iloc[pick]
bg = torch.tensor(np.load(os.path.join(C.OUT, "bgpool.npy")), device=dev)
Wt = torch.tensor(W[pick], device=dev); fs = torch.tensor(s.fs_eff.to_numpy(np.float32), device=dev)
e0 = torch.tensor(s.ev_i0.to_numpy(), device=dev); e1 = torch.tensor(s.ev_i1.to_numpy(), device=dev)
sg = torch.tensor(s.sigma_ev.to_numpy(np.float32), device=dev)
g = torch.Generator(device=dev); g.manual_seed(0)
v1 = A.train_view(Wt, fs, e0, e1, sg, bg, g).cpu().numpy()
v2 = A.train_view(Wt, fs, e0, e1, sg, bg, g).cpu().numpy()
ev = A.eval_view(Wt, fs).cpu().numpy()
fig, ax = plt.subplots(8, 2, figsize=(14, 16), sharex="col")
tt = np.arange(C.WIN_N) / 50.0 - C.CENTRE / 50.0
tc = (np.arange(C.CROP_N) - (C.CROP_N - 1) / 2) / 50.0
for i in range(8):
    w = W[pick[i]]; a = np.abs(w).max()
    ax[i, 0].plot(tt, w / a, "k", lw=0.8)
    ax[i, 0].axvspan(tt[s.ev_i0.iloc[i]], tt[min(s.ev_i1.iloc[i], C.WIN_N - 1)], color="orange", alpha=0.2)
    ax[i, 0].axvspan(-6, 6, color="blue", alpha=0.06)
    ax[i, 0].set_ylabel(f"{s.event_id.iloc[i][-8:]}\nSNR {s.snr.iloc[i]:.0f}", fontsize=8)
    ax[i, 1].plot(tc, ev[i], color="0.6", lw=1.6, label="eval crop (centre)")
    ax[i, 1].plot(tc, v1[i], "C0", lw=0.8, label="view 1")
    ax[i, 1].plot(tc, v2[i], "C3", lw=0.8, label="view 2")
ax[0, 0].set_title("21 s window / max|x| (orange: C [onset, offset]; blue: central 12 s)")
ax[0, 1].set_title("encoder inputs (12 s, 50 Hz, max-normalised)")
ax[0, 1].legend(fontsize=7, loc="upper right")
ax[-1, 0].set_xlabel("s from centroid"); ax[-1, 1].set_xlabel("s (crop time)")
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_aug_views.png"), dpi=90)

# (3) stats over 2000 train views: sign preserved? correlation to the eval crop
ii = torch.tensor(np.random.default_rng(2).choice(tr, 2000, replace=False), device=dev)
iin = ii.cpu().numpy(); ss = m.iloc[iin]
args = (torch.tensor(W[iin], device=dev), torch.tensor(ss.fs_eff.to_numpy(np.float32), device=dev),
        torch.tensor(ss.ev_i0.to_numpy(), device=dev), torch.tensor(ss.ev_i1.to_numpy(), device=dev),
        torch.tensor(ss.sigma_ev.to_numpy(np.float32), device=dev), bg, g)
v = A.train_view(*args).cpu().numpy(); e = A.eval_view(args[0], args[1]).cpu().numpy()
pol_v = np.sign(v[np.arange(len(v)), np.abs(v).argmax(1)]); pol_e = np.sign(e[np.arange(len(e)), np.abs(e).argmax(1)])
noise_only = A.train_view(*args, parts=("noise",)).cpu().numpy()
print(f"polarity of view max == eval-crop polarity: {np.mean(pol_v == pol_e):.3f}")
print(f"noise-only view: median residual sigma / 1 (max-norm units): {np.median(np.std(noise_only - e, 1)):.4f}")
print(f"event mask fully inside crop (random crop rule): "
      f"{np.mean((ss.ev_i1 - ss.ev_i0).to_numpy() <= 600 * 0.85):.3f} of events can fit at every stretch")
