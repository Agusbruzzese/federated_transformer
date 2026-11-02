"""A2: label-free embedding quality on test-day events, per seed, trained vs init (SimCLR-TF; SimCLR-CNN for contrast).

- RankMe: exp(entropy of normalised singular values) of the raw 64-d encoder output (uncentred, Garrido et al. 2023);
  also a centred variant.
- Alignment: mean ||f(v1)-f(v2)||^2 over L2-normalised encoder outputs of two independent M5 training views
  (ssl_central.augment.train_view, all parts, real background pool from m4.bgpool() = training days only),
  averaged over R=5 view-pair draws (torch generator seed 1000+r).
- Uniformity: log mean exp(-2 ||zi-zj||^2) over all pairs i<j of L2-normalised eval-view embeddings.
- align_ratio: alignment / mean squared distance between different events (eval view) -> scale-free invariance.
Output: a2_labelfree.csv
"""
from common import *   # noqa
import pandas as pd

m, W = m4.load(); te = np.flatnonzero((m.role == "test").to_numpy()); s = m.iloc[te]
bg, _ = m4.bgpool()
T = dict(W=torch.tensor(W[te], device=DEV), fs=torch.tensor(s.fs_eff.to_numpy(np.float32), device=DEV),
         ev0=torch.tensor(s.ev_i0.to_numpy(), device=DEV), ev1=torch.tensor(s.ev_i1.to_numpy(), device=DEV),
         sig=torch.tensor(s.sigma_ev.to_numpy(np.float32), device=DEV), bg=torch.tensor(bg, device=DEV))
R = 5
views = []
for r in range(R):                                  # identical views for every model
    g = torch.Generator(device=DEV); g.manual_seed(1000 + r)
    a = (T["W"], T["fs"], T["ev0"], T["ev1"], T["sig"], T["bg"], g)
    views.append((A.train_view(*a), A.train_view(*a)))
print("n test events", len(te), flush=True)

rows = []
for meth in ("simclr_tf", "simclr_cnn"):
    for which in ("final", "init"):
        for sd in SEEDS:
            Z = emb(meth, sd, which)[te]; Zn = l2n(Z)
            enc = load_encoder(meth, sd, which)
            al = [alignment(l2n(embed_x(enc, v1)), l2n(embed_x(enc, v2))) for v1, v2 in views]
            d2 = np.maximum(2 - 2 * Zn @ Zn.T, 0); iu = np.triu_indices(len(Zn), 1)
            rows.append(dict(rep=meth, which=which, seed=sd, rankme=rankme(Z), rankme_centered=rankme(Z, center=True),
                             alignment=np.mean(al), alignment_sd_over_draws=np.std(al, ddof=1), uniformity=uniformity(Zn),
                             mean_pair_d2=d2[iu].mean(), align_ratio=np.mean(al) / d2[iu].mean()))
            del enc; torch.cuda.empty_cache()
D = pd.DataFrame(rows); D.to_csv(os.path.join(HERE, "a2_labelfree.csv"), index=False)
pd.set_option("display.width", 250)
print(D.round(4).to_string())
print(D.groupby(["rep", "which"]).agg(["mean", "std"]).drop(columns="seed").round(4).T.to_string())
