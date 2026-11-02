import os, sys, json, numpy as np, pandas as pd
ROOT = "/home/augustin/Documents/federated_transformer"; os.chdir(ROOT)
sys.path[:0] = [ROOT, f"{ROOT}/research/model/M4_methods"]
import m4
from sklearn.cluster import HDBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler as SS
from sklearn.metrics import adjusted_rand_score as ARI
from sklearn.neighbors import NearestNeighbors
OUT = f"{ROOT}/data/c1/review"; RV = f"{ROOT}/research/model/C1_clustering/review"
m, _ = m4.load(); x = np.load(f"{OUT}/x.npy"); P = np.load(f"{OUT}/P.npy"); D = np.load(f"{OUT}/D100.npy"); n = len(m)
ref = pd.read_csv(f"{ROOT}/research/model/C1_clustering/cluster_labels.csv")
amp = 10 ** m.m8_log_amp.to_numpy(); dur = m.m8_dur10.to_numpy(); peak = np.abs(x).max(1)
day = pd.to_datetime(m.date.astype(str)).dt.day.to_numpy(); runid = m.run_id.to_numpy()
hdb = lambda Z, mcs=15: HDBSCAN(min_cluster_size=mcs).fit_predict(Z.astype(np.float64))
hdbp = lambda Dm, mcs=15: HDBSCAN(min_cluster_size=mcs, metric="precomputed").fit_predict(Dm.astype(np.float64))
m8 = [c for c in m.columns if c.startswith("m8_")]; DISC = ["m8_polarity", "m8_first_lobe_sign", "m8_n_lobes"]
def r3(cols, df=m): return SS().fit_transform(df[cols].fillna(0).to_numpy(float))
def r2(s):
    e = np.load(f"{OUT}/emb_s{s}.npy"); p = PCA(8, random_state=0).fit(SS().fit_transform(e))
    sc = np.c_[np.log10(amp), np.log10(np.clip(dur, .05, None))]
    return np.c_[SS().fit_transform(p.transform(SS().fit_transform(e))), SS().fit_transform(sc)]
R = {}
def summ(lab):
    c = lab[lab >= 0]; return dict(k=int(lab.max() + 1), cov=round(float((lab >= 0).mean()), 3),
        largest=round(float(np.bincount(c).max() / len(c)), 3) if len(c) else None, top3=round(float(np.sort(np.bincount(c))[-3:].sum() / len(c)), 3) if len(c) else None)
def tight(lab):
    """exact all-pairs within-cluster median aligned corr; amplitude/duration spread in user's terms (size-weighted)"""
    rows = []
    for c in np.unique(lab[lab >= 0]):
        i = np.flatnonzero(lab == c); iu = np.triu_indices(len(i), 1); la = np.log10(peak[i]); ld = np.log10(np.clip(dur[i], .05, None))
        rows.append(dict(n=len(i), corr=np.median(P[np.ix_(i, i)][iu]), maxmin=peak[i].max() / peak[i].min(),
                         p90p10=10 ** (np.quantile(la, .9) - np.quantile(la, .1)), iqr_la=np.subtract(*np.quantile(la, [.75, .25])),
                         iqr_ld=np.subtract(*np.quantile(ld, [.75, .25])), dur_p90p10=10 ** (np.quantile(ld, .9) - np.quantile(ld, .1)),
                         cv_amp=amp[i].std() / amp[i].mean(), frac_pairs_corr90=(P[np.ix_(i, i)][iu] >= .9).mean(),
                         frac_pairs_amp2=(np.abs(la[:, None] - la[None])[iu] <= np.log10(1.5)).mean()))
    d = pd.DataFrame(rows); w = d.n / d.n.sum()
    out = {k: round(float((d[k] * w).sum()), 3) for k in d.columns if k != "n"}
    out["median_cluster_p90p10"] = round(float(d.p90p10.median()), 2); return out, d
def repro(Zfun, groups, pre=False):
    res = []
    for g in np.unique(groups):
        A, B = np.flatnonzero(groups != g), np.flatnonzero(groups == g)
        if pre: la, lb = hdbp(D[np.ix_(A, A)]), hdbp(D[np.ix_(B, B)]); nn_ = np.argmin(D[np.ix_(B, A)], 1)
        else:
            Z = Zfun; la, lb = hdb(Z[A]), hdb(Z[B]); nn_ = NearestNeighbors(n_neighbors=1).fit(Z[A]).kneighbors(Z[B])[1][:, 0]
        tr = la[nn_]; j = (tr >= 0) & (lb >= 0)
        res.append((ARI(lb[j], tr[j]), ARI(lb, tr), j.mean()))
    return [round(float(v), 3) for v in np.mean(res, 0)]
def full(name, lab, Z=None, pre=False):
    t, d = tight(lab)
    R[name] = dict(**summ(lab), **t, oddeven=repro(Z, day % 2, pre), loro=repro(Z, runid, pre)); print(name, R[name], flush=True); return d
# ---- 1. reproduce labels
Z3 = r3(m8); L3 = hdb(Z3); Z2 = r2(0); L2 = hdb(Z2); L1 = hdbp(D)
for nm, L in (("R3_morph8", L3), ("R2_simclr_s0", L2), ("R1_awd_s100", L1)): print(nm, "ARI vs delivered labels", ARI(ref[nm], L), "identical", (ref[nm].to_numpy() == L).all())
d3 = full("R3", L3, Z3); d2 = full("R2s0", L2, Z2); d1 = full("R1", L1, pre=True)
# random-grouping reference for user metrics (same sizes as R3)
rng = np.random.default_rng(7); Lr = L3.copy(); cl = L3 >= 0; Lr[cl] = rng.permutation(L3[cl]); R["random_sizes_R3"] = tight(Lr)[0]
# ---- 2. discrete-feature structure of R3
key = m.m8_polarity.astype(int).astype(str) + "|" + m.m8_first_lobe_sign.astype(int).astype(str) + "|" + m.m8_n_lobes.clip(upper=4).astype(int).astype(str)
ct = pd.crosstab(L3[L3 >= 0], key[L3 >= 0]); purity = (ct.max(1) / ct.sum(1)); print("R3 cluster purity w.r.t. (pol,first,nlobes<=4):\n", purity.round(3).to_dict())
R["R3_discrete_purity_weighted"] = float((ct.max(1)).sum() / ct.values.sum())
key2 = key.copy(); ct2 = pd.crosstab(L2[L2 >= 0], key[L2 >= 0]); R["R2s0_discrete_purity_weighted"] = float(ct2.max(1).sum() / ct2.values.sum())
# within combination: amplitude separation of R3 clusters?
dfc = pd.DataFrame(dict(c=L3, key=key, la=np.log10(peak)))[L3 >= 0]; print(dfc.groupby(["key", "c"]).la.agg(["size", "median", "min", "max"]).round(2))
# ablations
cont = [c for c in m8 if c not in DISC]
Zc = r3(cont); Lc = hdb(Zc); full("R3_noDiscrete", Lc, Zc)
rngj = np.random.default_rng(1); mj = m.copy()
for c in DISC: mj[c] = mj[c] + rngj.normal(0, 0.5, n)          # jitter discrete columns (half a step)
Zj = r3(m8, mj); Lj = hdb(Zj); full("R3_jitter0.5", Lj, Zj); print("ARI(R3, R3_jitter)", ARI(L3, Lj))
# discrete-only: how many islands does HDBSCAN find from the 3 discrete columns alone?
Zd = r3(DISC); Ld = hdb(Zd + rngj.normal(0, 1e-3, Zd.shape)); R["discrete_only"] = summ(Ld); R["ARI_R3_vs_discreteonly_joint"] = float(ARI(L3[(L3 >= 0) & (Ld >= 0)], Ld[(L3 >= 0) & (Ld >= 0)]))
# null preserving discrete structure: Gaussian with real covariance on continuous cols + real discrete cols
Zn = np.c_[rngj.multivariate_normal(Zc.mean(0), np.cov(Zc.T), n), SS().fit_transform(m[DISC].to_numpy(float))]
R["null_gauss_cont_plus_real_discrete"] = summ(hdb(Zn))
# ---- 2b. R2 seed consensus among seeds
Ls = [hdb(r2(s)) for s in range(5)]
pa = {f"{a}{b}": round(float(ARI(Ls[a][(Ls[a] >= 0) & (Ls[b] >= 0)], Ls[b][(Ls[a] >= 0) & (Ls[b] >= 0)])), 3) for a in range(5) for b in range(a + 1, 5)}
R["R2_pairwise_seed_ari_joint"] = pa; R["R2_seed_k"] = [int(L.max() + 1) for L in Ls]
json.dump(R, open(f"{RV}/review_numbers.json", "w"), indent=1, default=float)
pd.concat([d1.assign(rep="R1"), d2.assign(rep="R2s0"), d3.assign(rep="R3")]).to_csv(f"{RV}/per_cluster_tightness.csv", index=False)
print(json.dumps({k: v for k, v in R.items() if k not in ("R1", "R2s0", "R3")}, indent=1, default=float))
