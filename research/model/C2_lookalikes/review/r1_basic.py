"""Independent review, part 1: tolerance matrix sanity, claims 1, 3, 4."""
import sys, json, numpy as np, pandas as pd, torch
sys.path[:0] = ['.', 'research/model/M4_methods', 'research/model/C1_clustering']
import m4
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
m, W = m4.load(); n = len(m)
P = np.load('data/c1/aligned_corr.npy').astype(np.float64)
la = m.m8_log_amp.to_numpy(); ld = np.log10(m.m8_dur10.to_numpy())
run = m.run_id.to_numpy(); day = m.date.astype(str).to_numpy(); t = m.t_s.to_numpy(float)
out = {}
out['P_symmetric_maxabs'] = float(np.abs(P - P.T).max()); out['P_diag_min'] = float(P.diagonal().min())
# --- own aligned correlation on raw crops (independent impl, time-domain, Pearson on overlap and full-norm variants)
import c1
x = c1.raw_crops().astype(np.float64)
rng = np.random.default_rng(11); pr = rng.integers(0, n, (300, 2))
def mycorr(a, b, S=50, overlap=False):
    best = -9
    for s in range(-S, S + 1):
        if overlap:
            if s >= 0: u, v = a[s:], b[:len(b) - s]
            else: u, v = a[:s], b[-s:]
            c = np.corrcoef(u, v)[0, 1]
        else:
            az = a - a.mean(); bz = b - b.mean(); az /= np.linalg.norm(az); bz /= np.linalg.norm(bz)
            c = (az[s:] * bz[:len(bz) - s]).sum() if s >= 0 else (az[:s] * bz[-s:]).sum()
        best = max(best, c)
    return best
mine = np.array([mycorr(x[i], x[j]) for i, j in pr]); mine_ov = np.array([mycorr(x[i], x[j], overlap=True) for i, j in pr])
out['P_vs_my_fullnorm_maxabsdiff'] = float(np.abs(mine - P[pr[:, 0], pr[:, 1]]).max())
out['P_vs_my_overlapPearson_maxabsdiff'] = float(np.abs(mine_ov - P[pr[:, 0], pr[:, 1]]).max())
# features
out['dur_min_max'] = [float(10**ld.min()), float(10**ld.max())]
# tolerance
def tol(P, c=.9, a=1.5, d=1.5):
    D = np.maximum.reduce([(1 - P) / (1 - c), np.abs(la[:, None] - la[None]) / np.log10(a), np.abs(ld[:, None] - ld[None]) / np.log10(d)])
    np.fill_diagonal(D, 0); return D
D = tol(P); A = D <= 1; np.fill_diagonal(A, False)
np.save('research/model/C2_lookalikes/review/A_main.npy', A)
Ss = (P >= .9); np.fill_diagonal(Ss, False)
A2 = np.maximum(np.abs(la[:, None] - la[None]), np.abs(ld[:, None] - ld[None])) <= np.log10(1.5); np.fill_diagonal(A2, False)
# claim 1
lab = fcluster(linkage(squareform(D, checks=False), 'complete'), 1.0, 'distance')
u, c = np.unique(lab, return_counts=True); big = u[c >= 10]; g = np.isin(lab, big)
worst = max(D[np.ix_(np.flatnonzero(lab == k), np.flatnonzero(lab == k))].max() for k in big)
out['claim1'] = dict(n_groups=int(len(big)), coverage=float(g.mean()), worst_within_d=float(worst), sizes_top=sorted(c[c >= 10].tolist())[::-1][:8])
# conductance: share of look-alike edges of grouped events that leave their group
L = np.where(g, lab, -1); ein = 0; eall = 0
for k in big:
    i = np.flatnonzero(lab == k); ein += A[np.ix_(i, i)].sum(); eall += A[i].sum()
out['claim1']['share_of_edges_of_grouped_events_inside_own_group'] = float(ein / eall)
cnt = pd.Series(lab[g]).value_counts(); 
# claims 3/4
k = A.sum(1)
out['degree'] = dict(ge1=float((k >= 1).mean()), ge10=float((k >= 10).mean()), median=float(np.median(k)), mean=float(k.mean()))
oth = A & (run[:, None] != run[None]); rec = oth.any(1)
B = np.array([rec[rng.integers(0, n, n)].mean() for _ in range(2000)])
out['claim3_recurrence_other_run'] = [float(rec.mean()), float(np.quantile(B, .025)), float(np.quantile(B, .975))]
# block bootstrap by day (events are not independent)
days = np.unique(day); Bd = []
for _ in range(2000):
    dd = rng.choice(days, len(days)); idx = np.concatenate([np.flatnonzero(day == q) for q in dd]); Bd.append(rec[idx].mean())
out['claim3_dayblock_CI'] = [float(np.quantile(Bd, .025)), float(np.quantile(Bd, .975))]
out['claim3_among_events_with_ge1_lookalike'] = float(rec[k >= 1].mean())
# upper bound: amp+dur-only match in other run
out['amp_dur_only_match_other_run'] = float((A2 & (run[:, None] != run[None])).any(1).mean())
out['claim4_share'] = float(A.sum() / A2.sum())
out['shape_match_rate_all_pairs'] = float(Ss.sum() / (n * (n - 1)))
out['shape_match_rate_given_ampdur'] = float((Ss & A2).sum() / A2.sum())
out['shape_match_rate_given_not_ampdur'] = float((Ss & ~A2).sum() / (~A2).sum() if True else 0)
# same-run enrichment of look-alikes vs run share expectation
same = (A & (run[:, None] == run[None])).sum(); tot = A.sum()
sizes = np.bincount(run); exp_same = ((sizes - 1) * sizes).sum() / (n * (n - 1))
same2 = (A2 & (run[:, None] == run[None])).sum() / A2.sum()
out['same_run_share_of_lookalike_edges'] = dict(observed=float(same / tot), expected_random=float(exp_same), ampdur_only=float(same2))
sameday = (A & (day[:, None] == day[None])).sum() / tot; sameday2 = (A2 & (day[:, None] == day[None])).sum() / A2.sum()
out['same_day_share'] = dict(lookalike=float(sameday), ampdur_only=float(sameday2))
within1h = (A & (np.abs(t[:, None] - t[None]) < 3600)).sum() / tot; w2 = (A2 & (np.abs(t[:, None] - t[None]) < 3600)).sum() / A2.sum()
out['within_1h_share'] = dict(lookalike=float(within1h), ampdur_only=float(w2))
json.dump(out, open('research/model/C2_lookalikes/review/r1.json', 'w'), indent=1); print(json.dumps(out, indent=1))
