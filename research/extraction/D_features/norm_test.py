"""Per-client vs global normalisation under non-IID clients (dev runs only).
Simulated clients = (date, 4-h slot) groups with >=30 events -- mirrors the README split
(clients = days and within-day time slices).
H: per-client z-scoring removes genuine between-client differences in vehicle mix (label shift),
so the same physical event maps to different normalised values on different clients.
Measure, pooled over clients (5-fold kNN): R2 for physical log-amplitude and log-duration;
ARI between KMeans(k=5) partitions of global vs per-client normalised data; client-id leakage."""
import numpy as np, pandas as pd, json
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from evaluate import build_reps, Scaler, proxies, leakage, pca_space, SCR, HOLDOUT_RUN, boot_ci

F = dict(np.load(SCR + "feats.npz")); meta = pd.read_parquet(SCR + "meta2.parquet")
reps = build_reps(F, meta)
m = ((meta.kind == "event") & (meta.run != HOLDOUT_RUN)).values
mm = meta[m].reset_index(drop=True)
client = (mm.wall.dt.strftime("%m%d") + "_" + (mm.wall.dt.hour // 4).astype(str)).values
vc = pd.Series(client).value_counts(); ok = np.isin(client, vc[vc >= 30].index)
mm = mm[ok].reset_index(drop=True); client = client[ok]
rows = []
for name in ["morph", "wavelet", "stft", "morph+wavelet", "legacy70", "waveform"]:
    X = reps[name][m][ok]
    Zg = Scaler().fit(X).transform(X)
    Zc = np.zeros_like(Zg)
    for c in np.unique(client):
        s = client == c
        Zc[s] = Scaler().fit(X[s]).transform(X[s])
    res = {}
    for tag, Z in [("global", Zg), ("per_client", Zc)]:
        p = proxies(Z, mm)
        res[tag] = dict(logamp_R2=p["logamp_R2"], logdur_R2=p["logdur_R2"], polarity=p["polarity_skill"],
                        client_leak=leakage(Z, client)["skill"])
    lg = KMeans(5, n_init=4, random_state=0).fit_predict(pca_space(Zg))
    lc = KMeans(5, n_init=4, random_state=0).fit_predict(pca_space(Zc))
    rows.append(dict(rep=name, n=len(X), n_clients=len(np.unique(client)),
                     **{f"{t}_{k}": v[0] for t in res for k, v in res[t].items()},
                     **{f"{t}_{k}_ci": f"[{v[1]:.3f},{v[2]:.3f}]" for t in res for k, v in res[t].items()},
                     ARI_global_vs_perclient=adjusted_rand_score(lg, lc)))
    print(rows[-1], flush=True)
# how different are clients physically? (spread of per-client medians of physical proxies)
phys = mm.assign(client=client, logamp=np.log10(mm.peak), logdur=np.log10(mm.dur)).groupby("client")[["logamp", "logdur"]].median()
out = pd.DataFrame(rows); out.to_csv("norm_test.csv", index=False)
json.dump({"per_client_median_logamp_range": [float(phys.logamp.min()), float(phys.logamp.max())],
           "per_client_median_logdur_range": [float(phys.logdur.min()), float(phys.logdur.max())]},
          open("norm_test_phys.json", "w"), indent=1)
print(out.round(3).to_string()); print(phys.describe())
