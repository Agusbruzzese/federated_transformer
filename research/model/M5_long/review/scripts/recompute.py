# Reviewer's independent recomputation. Own ridge (LOO-GCV via SVD), own kNN, own skill, own bootstraps.
# Uses only: m4.load() for metadata, frozen embeddings dumped by dump_emb.py.
import sys, json, numpy as np
sys.path.insert(0, "research/model/M4_methods"); sys.path.insert(0, ".")
import m4
m, _ = m4.load(); E = np.load(sys.argv[1]); OUT = sys.argv[2]
role = m.role.to_numpy(); tr, va, te = role == "train", role == "val", role == "test"
day = m.date.astype(str).to_numpy(); blk = m.blk.to_numpy()
dur = np.log10(m.m8_dur10.clip(lower=0.05).to_numpy()); lob = np.minimum(m.m8_n_lobes.to_numpy(), 4).astype(int)

def my_skill(y, p):
    c = np.unique(y); return (np.mean([(p[y == k] == k).mean() for k in c]) - 1 / len(c)) / (1 - 1 / len(c))
def my_r2(y, p): return 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()

def ridge_loo(X, y, Xe, alphas=np.logspace(-3, 3, 13)):
    mu, ym = X.mean(0), y.mean(); Xc, yc = X - mu, y - ym
    U, s, Vt = np.linalg.svd(Xc, full_matrices=False); Uy = U.T @ yc; best = None
    for a in alphas:
        f = s ** 2 / (s ** 2 + a); h = (U ** 2 * f).sum(1); r = (yc - U @ (f * Uy)) / (1 - h - 1 / len(y))
        e = (r ** 2).mean()
        if best is None or e < best[0]: best = (e, a)
    a = best[1]; w = Vt.T @ (s / (s ** 2 + a) * Uy); return (Xe - mu) @ w + ym

def knn_reg(X, y, Xe, k=15):
    out = np.empty(len(Xe)); n2 = (X ** 2).sum(1)
    for i in range(0, len(Xe), 512):
        d = n2[None] - 2 * Xe[i:i+512] @ X.T; nn_ = np.argpartition(d, k, 1)[:, :k]; out[i:i+512] = y[nn_].mean(1)
    return out

def z(X): mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-12; return (X - mu) / sd

def pred(X, y, kind, mask_eval, scaled):
    Xs = z(X) if scaled else X
    f = ridge_loo if kind == "lin" else knn_reg
    return f(Xs[tr], y[tr], Xs[mask_eval])

def chosen(X, y, kind):   # own validation-day scaling choice
    return max((True, False), key=lambda sc: my_r2(y[va], pred(X, y, kind, va, sc)))

res = {}
def duration_cell(A, Bn, kind):
    PA, PB, chA, chB = [], [], [], []
    for s in range(5):
        for nm, P, ch in ((A, PA, chA), (Bn, PB, chB)):
            X = E[f"{nm}_s{s}"]; sc = chosen(X, dur, kind); ch.append(sc); P.append(pred(X, dur, kind, te, sc))
    return np.array(PA), np.array(PB), chA, chB

yt, dt, bt = dur[te], day[te], blk[te]
days = np.unique(dt); blocks = {d: np.unique(bt[dt == d]) for d in days}; ib = {b: np.flatnonzero(bt == b) for b in np.unique(bt)}
def boots(PA, PB, metric, B=4000, seed=7):
    rng = np.random.default_rng(seed); H, Bk = np.empty(B), np.empty(B); allb = np.unique(bt)
    for i in range(B):
        ii = np.concatenate([ib[b] for d in rng.choice(days, 4) for b in rng.choice(blocks[d], len(blocks[d]))])
        sa, sb = rng.integers(0, 5, 5), rng.integers(0, 5, 5)
        H[i] = np.mean([metric(yt[ii], PA[s][ii]) for s in sa]) - np.mean([metric(yt[ii], PB[s][ii]) for s in sb])
        jj = np.concatenate([ib[b] for b in rng.choice(allb, len(allb))])
        Bk[i] = np.mean([metric(yt[jj], PA[s][jj]) for s in sa]) - np.mean([metric(yt[jj], PB[s][jj]) for s in sb])
    pt = np.mean([metric(yt, p) for p in PA]) - np.mean([metric(yt, p) for p in PB])
    perday = {d: float(np.mean([metric(yt[dt == d], p[dt == d]) for p in PA]) - np.mean([metric(yt[dt == d], p[dt == d]) for p in PB])) for d in days}
    pd_ = np.array(list(perday.values())); t = pd_.mean() / (pd_.std(ddof=1) / 2)
    from scipy import stats
    return dict(delta=float(pt), hier_ci=[float(np.quantile(H, .025)), float(np.quantile(H, .975))], hier_sd=float(H.std()),
                hier_p=float((1 + 2 * min((H <= 0).sum(), (H >= 0).sum())) / (B + 1)),
                block_ci=[float(np.quantile(Bk, .025)), float(np.quantile(Bk, .975))], block_sd=float(Bk.std()),
                per_day=perday, day_t_p=float(2 * stats.t.sf(abs(t), 3)), day_sign_all_pos=bool((pd_ > 0).all()))

for A, Bn, kind in (("simclr_tf", "vicreg_tf", "lin"), ("simclr_tf", "simclr_cnn", "lin"), ("simclr_tf", "B1", "lin"), ("simclr_tf", "B1", "knn")):
    PA, PB, chA, chB = duration_cell(A, Bn, kind)
    r = boots(PA, PB, my_r2); r.update(scale_A=chA, scale_B=chB, abs_A=[float(my_r2(yt, p)) for p in PA], abs_B=[float(my_r2(yt, p)) for p in PB])
    res[f"duration_{kind}_{A}_vs_{Bn}"] = r; print(f"duration_{kind}_{A}_vs_{Bn}", json.dumps(r, default=float)[:600], flush=True)
json.dump(res, open(OUT, "w"), indent=1, default=float)
