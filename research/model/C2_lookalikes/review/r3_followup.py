"""Follow-up: corrected group counts, tightness/seed-stability of Louvain shape families, RGG modularity, amplitude modality inside shape families."""
import sys, json, numpy as np, networkx as nx
sys.path[:0] = ['.', 'research/model/M4_methods']; import m4
from sklearn.metrics import adjusted_rand_score as ARI
from sklearn.mixture import GaussianMixture
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
m, W = m4.load(); n = len(m); P = np.load('data/c1/aligned_corr.npy').astype(np.float64)
la = m.m8_log_amp.to_numpy(); ld = np.log10(m.m8_dur10.to_numpy()); run = m.run_id.to_numpy(); rng = np.random.default_rng(7); out = {}
def G_of(A):
    G = nx.Graph(); G.add_nodes_from(range(len(A))); G.add_edges_from(zip(*np.nonzero(np.triu(A, 1)))); return G
def louv(A, seed):
    C = nx.community.louvain_communities(G_of(A), seed=seed); l = -np.ones(len(A), int)
    for k, s in enumerate(C): l[list(s)] = k
    return l
def small(l, ms=10):
    l = l.copy(); c = np.bincount(l[l >= 0]); l[(l >= 0) & (c[np.maximum(l, 0)] < ms)] = -1; return l
ng = lambda l: int(len(np.unique(l[l >= 0])))
S = P >= .9; np.fill_diagonal(S, False)
A = S & (np.abs(la[:, None] - la[None]) <= np.log10(1.5)) & (np.abs(ld[:, None] - ld[None]) <= np.log10(1.5))
# corrected null group count (shape permuted within amp x dur decile cells)
qa = np.digitize(la, np.quantile(la, np.linspace(0, 1, 11)[1:-1])); qd = np.digitize(ld, np.quantile(ld, np.linspace(0, 1, 11)[1:-1])); cell = qa * 10 + qd
res = []
for _ in range(3):
    pi = np.arange(n)
    for c in np.unique(cell):
        i = np.flatnonzero(cell == c); pi[i] = rng.permutation(i)
    Dn = np.maximum.reduce([(1 - P[np.ix_(pi, pi)]) / .1, np.abs(la[:, None] - la[None]) / np.log10(1.5), np.abs(ld[:, None] - ld[None]) / np.log10(1.5)]); np.fill_diagonal(Dn, 0)
    l = small(fcluster(linkage(squareform(Dn, checks=False), 'complete'), 1.0, 'distance') - 1); res.append((ng(l), (l >= 0).mean()))
out['C2a_under_shape_permuted_null'] = dict(n_groups=[r[0] for r in res], coverage=[float(r[1]) for r in res])
# Louvain shape-only families: tightness, seed stability
LS = [small(louv(S, s)) for s in range(4)]; L = LS[0]
out['shape_family_louvain'] = dict(n=ng(L), coverage=float((L >= 0).mean()), seed_ari=[float(ARI(L, l)) for l in LS[1:]])
rows = []
for g in np.unique(L[L >= 0]):
    i = np.flatnonzero(L == g); sub = P[np.ix_(i, i)][np.triu_indices(len(i), 1)]
    rows.append((len(i), np.median(sub), np.quantile(sub, .1), (sub >= .9).mean(), 10 ** (np.quantile(la[i], .9) - np.quantile(la[i], .1)), 10 ** (np.quantile(ld[i], .9) - np.quantile(ld[i], .1)), S[i][:, i].sum() / S[i].sum()))
r = np.array(rows); w = r[:, 0] / r[:, 0].sum()
out['shape_family_louvain']['sizeweighted'] = dict(median_pair_corr=float(w @ r[:, 1]), p10_pair_corr=float(w @ r[:, 2]), frac_pairs_corr_ge_090=float(w @ r[:, 3]),
    amp_p90_p10=float(w @ r[:, 4]), dur_p90_p10=float(w @ r[:, 5]), internal_edge_share=float(w @ r[:, 6]))
top = np.argsort(-r[:, 0])[:8]; out['shape_family_louvain']['top8'] = [dict(size=int(r[k, 0]), med_corr=round(r[k, 1], 3), amp_p90_p10=round(r[k, 4], 2), dur_p90_p10=round(r[k, 5], 2)) for k in top]
# amplitude modality in the 8 largest shape families: GMM BIC 1 vs 2 vs 3 components on log amp
gs = np.unique(L[L >= 0])[top]; bic = []
for g in gs:
    v = la[L == g][:, None]; b = [GaussianMixture(k, random_state=0).fit(v).bic(v) for k in (1, 2, 3)]; bic.append(int(np.argmin(b)) + 1)
out['shape_family_louvain']['best_gmm_k_logamp_top8'] = bic
# C2a groups nested in shape families?
lab = small(fcluster(linkage(squareform(np.maximum.reduce([(1 - P) / .1, np.abs(la[:, None] - la[None]) / np.log10(1.5), np.abs(ld[:, None] - ld[None]) / np.log10(1.5)]) * (1 - np.eye(n)), checks=False), 'complete'), 1.0, 'distance') - 1)
purity = []
for g in np.unique(lab[lab >= 0]):
    f = L[lab == g]; purity.append(np.bincount(f[f >= 0]).max() / len(f) if (f >= 0).any() else 0)
out['C2a_groups_purity_wrt_shape_families'] = dict(median=float(np.median(purity)), frac_ge_090=float(np.mean(np.array(purity) >= .9)),
                                                   c2a_groups_per_family_median=float(np.median(np.bincount(L[(lab >= 0) & (L >= 0)])[np.bincount(L[(lab >= 0) & (L >= 0)]) > 0])) if True else 0)
gpf = {}
for g in np.unique(lab[lab >= 0]):
    f = L[lab == g]; f = f[f >= 0]
    if len(f): k = np.bincount(f).argmax(); gpf[k] = gpf.get(k, 0) + 1
out['C2a_groups_per_shape_family'] = dict(n_families_hosting=len(gpf), max=max(gpf.values()), median=float(np.median(list(gpf.values()))))
# modularity: real vs random geometric (3D, same mean degree)
mod = lambda A, l: nx.community.modularity(G_of(A), [set(np.flatnonzero(l == k)) for k in np.unique(l)])
X = rng.random((n, 3)); Dm = np.abs(X[:, None] - X[None]).max(-1); deg = A.sum(1)
rr = np.quantile(Dm[np.triu_indices(n, 1)], deg.mean() / (n - 1)); Ag = (Dm <= rr) & ~np.eye(n, dtype=bool)
out['modularity'] = dict(real_main=mod(A, louv(A, 0)), rgg3d=mod(Ag, louv(Ag, 0)))
# RGG LORO-like reproducibility (random 4-way split) as continuum reference for Louvain vote-transfer ARI
split = rng.integers(0, 4, n); ar = []
for g in range(4):
    iA, iB = np.flatnonzero(split != g), np.flatnonzero(split == g); lA = small(louv(Ag[np.ix_(iA, iA)], 0)); lB = small(louv(Ag[np.ix_(iB, iB)], 0))
    sub = Ag[np.ix_(iB, iA)]; tr = np.array([np.bincount(lA[sub[r_]][lA[sub[r_]] >= 0]).argmax() if (lA[sub[r_]] >= 0).any() else -1 for r_ in range(len(iB))])
    j = (lB >= 0) & (tr >= 0); ar.append(float(ARI(lB[j], tr[j])))
out['rgg3d_split_louvain_vote_ari'] = ar
json.dump(out, open('research/model/C2_lookalikes/review/r3.json', 'w'), indent=1, default=float); print(json.dumps(out, indent=1, default=float))
