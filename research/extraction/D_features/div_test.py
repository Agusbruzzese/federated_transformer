"""Baseline SUBTRACTED vs DIVIDED ('DC-relative') -- does it change cluster structure? (dev events)
divided: x_div = x / dc_baseline_code * median(dc_baseline_code)  (i.e. relative deviation, rescaled to counts)."""
import numpy as np, pandas as pd, json
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from features import morph_feats, wavelet_feats, stft_feats, waveform_feats
from evaluate import Scaler, pca_space, cluster_metrics, SCR, HOLDOUT_RUN

LP = np.load(SCR + "LP.npy", mmap_mode="r"); meta = pd.read_parquet(SCR + "meta.parquet")
idx = np.where(((meta.kind == "event") & (meta.run != HOLDOUT_RUN)).values)[0]
b = meta.dc_baseline.values[idx]; scale = np.median(b) / b
print("divisor spread: scale factor range %.4f-%.4f" % (scale.min(), scale.max()))
out = {"scale_factor_range": [float(scale.min()), float(scale.max())]}
for name, f in [("morph", morph_feats), ("wavelet", wavelet_feats), ("stft", stft_feats), ("waveform", waveform_feats)]:
    Xs = np.stack([f(np.asarray(LP[i, 200:1000], float)) for i in idx])
    Xd = np.stack([f(np.asarray(LP[i, 200:1000], float) * s) for i, s in zip(idx, scale)])
    Zs = Scaler().fit(Xs).transform(Xs); Zd = Scaler().fit(Xd).transform(Xd)
    ls = KMeans(5, n_init=4, random_state=0).fit_predict(pca_space(Zs))
    ld = KMeans(5, n_init=4, random_state=0).fit_predict(pca_space(Zd))
    cs, cd = cluster_metrics(Zs), cluster_metrics(Zd)
    out[name] = dict(ARI_sub_vs_div=float(adjusted_rand_score(ls, ld)),
                     boot_ARI_sub=cs["real"]["ari"], boot_ARI_div=cd["real"]["ari"],
                     sil_excess_sub=cs["sil_excess"], sil_excess_div=cd["sil_excess"])
    print(name, out[name], flush=True)
json.dump(out, open("div_vs_sub.json", "w"), indent=1)
