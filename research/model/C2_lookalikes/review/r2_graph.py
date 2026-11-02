"""Independent review, part 2: graph structure, continuum vs discrete, reproducibility of alternative partitions, recurrence null."""
import sys, json, numpy as np, pandas as pd, networkx as nx
sys.path[:0] = ['.', 'research/model/M4_methods']; import m4
from sklearn.metrics import adjusted_rand_score as ARI
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from scipy.sparse.csgraph import connected_components
m, W = m4.load(); n = len(m); P = np.load('data/c1/aligned_corr.npy').astype(np.float64)
la = m.m8_log_amp.to_numpy(); ld = np.log10(m.m8_dur10.to_numpy()); run = m.run_id.to_numpy()
rng = np.random.default_rng(5); out = {}
def adj(P, c=.9, a=1.5, d=1.5, ignore_ad=False):
    A = P >= c
    if not ignore_ad: A &= (np.abs(la[:, None] - la[None]) <= np.log10(a)) & (np.abs(ld[:, None] - ld[None]) <= np.log10(d))
    np.fill_diagonal(A, False); return A
def G_of(A): 
    G = nx.Graph(); G.add_nodes_from(range(len(A))); G.add_edges_from(zip(*np.nonzero(np.triu(A)))); return G
def comps(A):
    _, l = connected_components(A.astype(np.int8), directed=False); c = np.bincount(l); return l, c
def louv(A, seed, res=1.0):
    C = nx.community.louvain_communities(G_of(A), resolution=res, seed=seed); l = -np.ones(len(A), int)
    for k, s in enumerate(C): l[list(s)] = k
    return l, nx.community.modularity(G_of(A), C) if seed == 0 else None
def small(l, ms=10):
    l = l.copy(); c = np.bincount(l); l[c[l] < ms] = -1; return l
def vote(A, labA, iA, iB):
    """assign each B event the most common label among its look-alikes in A (grouped ones); -1 if none."""
    sub = A[np.ix_(iB, iA)]; out = -np.ones(len(iB), int); L = labA
    for r in range(len(iB)):
        nb = L[sub[r]]; nb = nb[nb >= 0]
        if len(nb): out[r] = np.bincount(nb).argmax()
    return out
def loro(A, part):
    res = []
    for g in range(4):
        iA, iB = np.flatnonzero(run != g), np.flatnonzero(run == g)
        la_, lb = part(A[np.ix_(iA, iA)]), part(A[np.ix_(iB, iB)]); tr = vote(A, la_, iA, iB)
        j = (lb >= 0) & (tr >= 0); res.append((float(ARI(lb[j], tr[j])) if j.sum() > 1 else 0., float(j.mean()), int(lb.max() + 1)))
    return dict(ari=[r[0] for r in res], mean_ari=float(np.mean([r[0] for r in res])), frac_joint=float(np.mean([r[1] for r in res])), nB=[r[2] for r in res])
def c2a_part(Asub):
    # recover D from adjacency impossible; use precomputed D via closure
    raise NotImplementedError
A = adj(P); np.fill_diagonal(A, False)
# --- components
for name, Ax in (('main', A), ('strict', adj(P, .95, 1.25, 1.25)), ('vstrict', adj(P, .97, 1.15, 1.15)), ('shape_only_0.90', adj(P, ignore_ad=True)), ('shape_only_0.95', adj(P, .95, ignore_ad=True))):
    l, c = comps(Ax); deg = Ax.sum(1)
    out.setdefault('components', {})[name] = dict(edges=int(Ax.sum() // 2), isolated=float((deg == 0).mean()), giant_frac=float(c.max() / n),
                                                   n_comp_ge10=int((c >= 10).sum()), cover_comp_ge10_excl_giant=float((c[l] >= 10).mean() - c.max() / n))
# --- clustering vs degree-matched (edge-swap) random and continuum (random geometric) graphs
G = G_of(A); Cobs = nx.average_clustering(G); Gr = G.copy()
nx.double_edge_swap(Gr, nswap=4 * G.number_of_edges(), max_tries=10 ** 8, seed=1); Cr = nx.average_clustering(Gr)
deg = A.sum(1); rgg = {}
for dim in (2, 3, 5):
    X = rng.random((n, dim)); Dm = np.abs(X[:, None] - X[None]).max(-1); r = np.quantile(Dm[np.triu_indices(n, 1)], deg.mean() / (n - 1))
    rgg[dim] = nx.average_clustering(G_of((Dm <= r) & ~np.eye(n, dtype=bool)))
out['clustering'] = dict(observed=Cobs, degree_matched_random=Cr, random_geometric_same_mean_degree=rgg, transitivity=nx.transitivity(G))
# --- Louvain on full graph: modularity, seed stability
L0, Q = louv(A, 0); Ls = [louv(A, s)[0] for s in (1, 2, 3)]
out['louvain_main'] = dict(modularity=Q, n_comm=int(L0.max() + 1), sizes_top=sorted(np.bincount(L0).tolist())[::-1][:10],
                            seed_ari=[float(ARI(L0, l)) for l in Ls])
Lr0, Qr = louv(nx.to_numpy_array(Gr).astype(bool), 0); out['louvain_main']['modularity_degree_matched_random'] = Qr
# --- LORO for Louvain (main, shape-only) and for C2a with the same neighbour-vote transfer
D = np.maximum.reduce([(1 - P) / .1, np.abs(la[:, None] - la[None]) / np.log10(1.5), np.abs(ld[:, None] - ld[None]) / np.log10(1.5)]); np.fill_diagonal(D, 0)
def c2a_idx(idx):
    l = fcluster(linkage(squareform(D[np.ix_(idx, idx)], checks=False), 'complete'), 1.0, 'distance') - 1; return small(l)
def loro_idx(Aref, part_idx):
    res = []
    for g in range(4):
        iA, iB = np.flatnonzero(run != g), np.flatnonzero(run == g)
        la_, lb = part_idx(iA), part_idx(iB); tr = vote(Aref, la_, iA, iB)
        j = (lb >= 0) & (tr >= 0); res.append((float(ARI(lb[j], tr[j])) if j.sum() > 1 else 0., float(j.mean())))
    return dict(ari=[round(r[0], 3) for r in res], mean_ari=float(np.mean([r[0] for r in res])), frac_joint=float(np.mean([r[1] for r in res])))
out['loro'] = {}
out['loro']['C2a_complete_vote_transfer'] = loro_idx(A, c2a_idx)
for name, Ax in (('louvain_main', A), ('louvain_shape_only_0.90', adj(P, ignore_ad=True)), ('louvain_shape_only_0.95', adj(P, .95, ignore_ad=True))):
    out['loro'][name] = loro_idx(Ax, lambda idx: small(louv(Ax[np.ix_(idx, idx)], 0)[0]))
    out['loro'][name]['n_comm_full'] = int(small(louv(Ax, 0)[0]).max() + 1)
for name, Ax in (('components_strict', adj(P, .95, 1.25, 1.25)), ('components_vstrict', adj(P, .97, 1.15, 1.15))):
    out['loro'][name] = loro_idx(Ax, lambda idx: small(comps(Ax[np.ix_(idx, idx)])[0]))
# --- recurrence null: permute shape (rows+cols of P) within amp x dur bins
qa = np.digitize(la, np.quantile(la, np.linspace(0, 1, 11)[1:-1])); qd = np.digitize(ld, np.quantile(ld, np.linspace(0, 1, 11)[1:-1])); cell = qa * 10 + qd
rec_obs = (A & (run[:, None] != run[None])).any(1).mean(); nulls = []
for _ in range(20):
    pi = np.arange(n)
    for c in np.unique(cell):
        i = np.flatnonzero(cell == c); pi[i] = rng.permutation(i)
    Pn = P[np.ix_(pi, pi)]; An = adj(Pn)
    nulls.append(((An & (run[:, None] != run[None])).any(1).mean(), (An.sum(1) >= 1).mean(), An.sum() / A.sum()))
nulls = np.array(nulls)
out['recurrence_null_shape_permuted_within_ampdur_bins'] = dict(observed=float(rec_obs), null_mean=float(nulls[:, 0].mean()), null_range=[float(nulls[:, 0].min()), float(nulls[:, 0].max())],
                                                               null_ge1=float(nulls[:, 1].mean()), null_edges_ratio=float(nulls[:, 2].mean()))
# --- C2a groups under that null (do tight groups depend on the shape<->amp/dur link?)
pi = np.arange(n)
for c in np.unique(cell):
    i = np.flatnonzero(cell == c); pi[i] = rng.permutation(i)
Dn = np.maximum.reduce([(1 - P[np.ix_(pi, pi)]) / .1, np.abs(la[:, None] - la[None]) / np.log10(1.5), np.abs(ld[:, None] - ld[None]) / np.log10(1.5)]); np.fill_diagonal(Dn, 0)
ln = small(fcluster(linkage(squareform(Dn, checks=False), 'complete'), 1.0, 'distance') - 1)
out['C2a_under_shape_permuted_null'] = dict(n_groups=int(ln.max() + 1), coverage=float((ln >= 0).mean()))
json.dump(out, open('research/model/C2_lookalikes/review/r2.json', 'w'), indent=1, default=float); print(json.dumps(out, indent=1, default=float))
