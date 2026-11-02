"""A5: probes on synthetic events with known generation labels (synth/ generator fitted on training days only).

4000 events, Generator.sample(4000, rng=20261009); crops max-normalised (= eval_view normalisation) then embedded.
5-fold StratifiedKFold (stratified by family x polarity, shuffle, random_state=0), StandardScaler inside the fold:
  family (7 classes)  : LogisticRegression(C=1, max_iter=5000), balanced-accuracy skill (m4.skill)
  polarity            : same, skill
  log stretch         : RidgeCV(alphas=logspace(-3,3,13)), out-of-fold R2;  also log stretch minus its family mean
                        (stretch is defined relative to the family template, so this is the within-family part)
  log snr_peak        : RidgeCV, R2
Representations: SimCLR-TF final / init s0-4, SimCLR-CNN final / init s0-4, raw-crop PCA-64 (PCA fit on real
training-day eval crops, identical to M5 B4), raw 600-d crop.
Output: a5_synth.csv, a5_synth_summary.csv
"""
from common import *   # noqa
import pandas as pd
from synth.generator import Generator
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.model_selection import StratifiedKFold, cross_val_predict

N, SEED = 4000, 20261009
g = Generator.load(os.path.join(ROOT, "data/synth/model.npz")); X, info = g.sample(N, rng=SEED)
x = torch.tensor(X, device=DEV); x = x / x.abs().amax(1, keepdim=True).clamp_min(1e-6)
fam = info.family.to_numpy(); pol = (info.polarity.to_numpy() > 0).astype(int)
lst = np.log(info.stretch.to_numpy()); lsnr = np.log(info.snr_peak.to_numpy())
lst_w = lst - pd.Series(lst).groupby(fam).transform("mean").to_numpy()
strat = fam * 2 + pol
cv = StratifiedKFold(5, shuffle=True, random_state=0)
print("family counts", np.bincount(fam), "polarity +", pol.mean().round(3), flush=True)

m, W = m4.load(); tr = (m.role == "train").to_numpy()
pca = PCA(64, random_state=0).fit(m5.crops().cpu().numpy()[tr])
reps = {"raw_pca64": pca.transform(x.cpu().numpy()), "raw600": x.cpu().numpy()}
for meth in ("simclr_tf", "simclr_cnn"):
    for which in ("final", "init"):
        for s in SEEDS:
            enc = load_encoder(meth, s, which); reps[f"{meth}|{which}|{s}"] = embed_x(enc, x); del enc
torch.cuda.empty_cache()

def clf(Z, y): return m4.skill(y, cross_val_predict(make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000)), Z, y, cv=cv.split(Z, strat)))
def reg(Z, y): return m4.r2(y, cross_val_predict(make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 13))), Z, y, cv=cv.split(Z, strat)))

rows = []
for k, Z in reps.items():
    rep, which, s = (k.split("|") + ["", ""])[:3] if "|" in k else (k, "baseline", 0)
    rows.append(dict(rep=rep, which=which, seed=int(s), family_skill=clf(Z, fam), polarity_skill=clf(Z, pol),
                     log_stretch_r2=reg(Z, lst), log_stretch_within_family_r2=reg(Z, lst_w), log_snr_r2=reg(Z, lsnr)))
    print(rows[-1], flush=True)
D = pd.DataFrame(rows); D.to_csv(os.path.join(HERE, "a5_synth.csv"), index=False)
cols = ["family_skill", "polarity_skill", "log_stretch_r2", "log_stretch_within_family_r2", "log_snr_r2"]
S = D.groupby(["rep", "which"])[cols].agg(["mean", "std", "min", "max"])
S.to_csv(os.path.join(HERE, "a5_synth_summary.csv"))
# paired trained - init
dl = []
for rep in ("simclr_tf", "simclr_cnn"):
    f = D[(D.rep == rep) & (D.which == "final")].set_index("seed")[cols]; i = D[(D.rep == rep) & (D.which == "init")].set_index("seed")[cols]
    d = f - i; dl.append(pd.DataFrame({"rep": rep, "stat": ["delta_mean", "delta_sd"], **{c: [d[c].mean(), d[c].std()] for c in cols}}))
pd.concat(dl).to_csv(os.path.join(HERE, "a5_synth_deltas.csv"), index=False)
pd.set_option("display.width", 250); print(S.round(4).to_string()); print(pd.concat(dl).round(4).to_string())
