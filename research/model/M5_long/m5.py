"""M5 (PREREG.md v2, frozen 1105c2bb83dc8d20): 10k-step SSL, gated MAE, random/raw baselines, calibrated leakage.

    python research/model/M5_long/m5.py dryrun
    python research/model/M5_long/m5.py train <method> <seed>        # simclr_tf vicreg_tf mae_tf simclr_cnn
    python research/model/M5_long/m5.py gates
    python research/model/M5_long/m5.py score
"""
import os, sys, json, math, time, hashlib
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from joblib import Parallel, delayed

ROOT = os.getcwd(); sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "research/model/M4_methods"))
import m4                                                        # load(), bgpool(), targets(), vicreg_loss, VICRegHead, skill, r2, metric
from ssl_central import augment as A, models as Mo
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupKFold, cross_val_predict

HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(ROOT, "data", "m5")
STEPS, BATCH, WD, WARM = 10_000, 256, 1e-4, 0.05
CFG = {"simclr_tf": dict(lr=3e-4, T=0.1), "vicreg_tf": dict(lr=3e-4), "mae_tf": dict(lr=3e-4, mask=0.75), "simclr_cnn": dict(lr=3e-4, T=0.1)}
SSL = list(CFG); SEEDS = range(5); PRIMARY = ["polarity", "lobes", "duration"]; DEV = "cuda"


# ---------------------------------------------------------------- MAE, standard recipe (S3)
class MAE5(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = Mo.PatchTransformer(); d = self.enc.embed.out_features; p = self.enc.patch
        self.mask_token = nn.Parameter(torch.zeros(1, 1, d)); nn.init.normal_(self.mask_token, std=0.02)
        self.dec = nn.Sequential(nn.Linear(d, 128), nn.GELU(), nn.Linear(128, p))

    def patches(self, x): return x.view(x.shape[0], -1, self.enc.patch)

    @staticmethod
    def norm_t(P): return (P - P.mean(-1, keepdim=True)) / (P.std(-1, keepdim=True) + 1e-6)

    def make_mask(self, B, n, ratio, g, dev):
        k = int(round(ratio * n)); idx = torch.rand(B, n, generator=g, device=dev).argsort(1)[:, :k]
        mk = torch.zeros(B, n, dtype=torch.bool, device=dev); mk.scatter_(1, idx, True); return mk

    def forward_masked(self, x, mk):
        P = self.patches(x); t = self.enc.embed(P); B, n, d = t.shape
        t = torch.where(mk[..., None], self.mask_token.expand(B, n, d), t) + self.enc.pos
        return self.dec(self.enc.norm(self.enc.enc(t))), P

    def loss(self, x, ratio, g):
        mk = self.make_mask(x.shape[0], x.shape[1] // self.enc.patch, ratio, g, x.device)
        pred, P = self.forward_masked(x, mk)
        return F.mse_loss(pred[mk], self.norm_t(P)[mk])

    def embed(self, x):
        return self.enc.norm(self.enc.enc(self.enc.embed(self.patches(x)) + self.enc.pos)).mean(1)


def build(method):
    if method == "mae_tf": return MAE5(), None
    enc = Mo.make("cnn" if method == "simclr_cnn" else "tf")
    return enc, (m4.VICRegHead() if method == "vicreg_tf" else Mo.Head())


def ck(method, seed, which): return os.path.join(OUT, "ckpt", f"{method}_s{seed}", f"{which}.pt")


# ---------------------------------------------------------------- training (10k steps, warm-up + cosine)
def train(method, seed):
    m, W = m4.load(); cfg = CFG[method]
    idx = np.flatnonzero((m.role == "train").to_numpy()); s = m.iloc[idx]; bg, _ = m4.bgpool()
    T = dict(W=torch.tensor(W[idx], device=DEV), fs=torch.tensor(s.fs_eff.to_numpy(np.float32), device=DEV),
             ev0=torch.tensor(s.ev_i0.to_numpy(), device=DEV), ev1=torch.tensor(s.ev_i1.to_numpy(), device=DEV),
             sig=torch.tensor(s.sigma_ev.to_numpy(np.float32), device=DEV), t=torch.tensor(s.t_s.to_numpy(np.float64), device=DEV),
             bg=torch.tensor(bg, device=DEV))
    N = len(idx); torch.manual_seed(seed); np.random.seed(seed)
    enc, head = build(method); enc = enc.to(DEV); head = head.to(DEV) if head is not None else None
    os.makedirs(os.path.dirname(ck(method, seed, "x")), exist_ok=True); torch.save(enc.state_dict(), ck(method, seed, "init"))
    params = list(enc.parameters()) + (list(head.parameters()) if head is not None else [])
    opt = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=WD)
    wu = int(WARM * STEPS)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: (i + 1) / wu if i < wu else 0.5 * (1 + math.cos(math.pi * (i - wu) / (STEPS - wu))))
    g = torch.Generator(device=DEV); g.manual_seed(seed); gp = torch.Generator(); gp.manual_seed(seed)
    bs = BATCH; spe = N // bs; curve = []; step = 0; t0 = time.time()
    while step < STEPS:
        perm = torch.randperm(N, generator=gp).to(DEV); L = []
        enc.train()
        for b in range(spe):
            if step >= STEPS: break
            ii = perm[b * bs:(b + 1) * bs]
            args = (T["W"][ii], T["fs"][ii], T["ev0"][ii], T["ev1"][ii], T["sig"][ii], T["bg"], g)
            if method == "mae_tf":
                loss = enc.loss(A.train_view(*args, parts=("crop",)), cfg["mask"], g)
            else:
                v1, v2 = A.train_view(*args), A.train_view(*args)
                loss = m4.vicreg_loss(head(enc(v1)), head(enc(v2))) if method == "vicreg_tf" else Mo.nt_xent(head(enc(v1)), head(enc(v2)), T["t"][ii], cfg["T"])[0]
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step(); L.append(loss.item()); step += 1
        curve.append(float(np.mean(L)))
    torch.save(enc.state_dict(), ck(method, seed, "final"))
    json.dump(dict(method=method, seed=seed, cfg=cfg, steps=step, n_train=N, curve=curve, secs=time.time() - t0),
              open(ck(method, seed, "log").replace(".pt", ".json"), "w"))
    print(f"[{method}_s{seed}] {step} steps loss {curve[0]:.4f} -> {curve[-1]:.4f} {time.time() - t0:.0f}s", flush=True)


# ---------------------------------------------------------------- embeddings
_CROPS = {}
def crops():
    if "c" not in _CROPS:
        m, W = m4.load(); _CROPS["c"] = A.eval_view(torch.tensor(W, device=DEV), torch.tensor(m.fs_eff.to_numpy(np.float32), device=DEV))
    return _CROPS["c"]


@torch.no_grad()
def emb(method, seed, which, tokens=False):
    enc, _ = build(method); enc.load_state_dict(torch.load(ck(method, seed, which), map_location=DEV)); enc = enc.to(DEV).eval()
    x = crops(); out = []
    for i in range(0, len(x), 2048):
        xb = x[i:i + 2048]
        if tokens:                                                     # B2: untrained transformer, unpooled tokens
            P = xb.view(xb.shape[0], -1, enc.patch); h = enc.norm(enc.enc(enc.embed(P) + enc.pos)); out.append(h.flatten(1).cpu().numpy())
        elif method == "mae_tf":
            out.append(enc.embed(xb).cpu().numpy())
        else:
            out.append(enc(xb).cpu().numpy())
    return np.concatenate(out)


def representations(seed):
    """All representations for one seed index (single-seed baselines are returned for seed 0 only)."""
    m, W = m4.load(); tr = (m.role == "train").to_numpy(); R = {}
    for meth in SSL: R[meth] = emb(meth, seed, "final")
    R["B1_cnn_init"] = emb("simclr_cnn", seed, "init")
    tok = emb("simclr_tf", seed, "init", tokens=True); R["B2_tf_tokens"] = tok; R["B2_tf_tokens_pca"] = PCA(64, random_state=0).fit(tok[tr]).transform(tok)
    if seed == 0:
        raw = crops().cpu().numpy(); R["B3_raw"] = raw; R["B3_raw_pca"] = R["B4_pca64"] = PCA(64, random_state=0).fit(raw[tr]).transform(raw)
    return R


# ---------------------------------------------------------------- probes with validation-chosen scaling (C4)
def est(kind, name):
    if name == "duration": return KNeighborsRegressor(15) if kind == "knn" else RidgeCV(alphas=np.logspace(-3, 3, 13))
    return KNeighborsClassifier(15) if kind == "knn" else LogisticRegression(C=1.0, max_iter=3000)


def fit_pred(X, y, tr, ev, kind, name, scale):
    e = make_pipeline(StandardScaler(), est(kind, name)) if scale else est(kind, name)
    return e.fit(X[tr], y[tr]).predict(X[ev])


def probe_all(X, m, tg, kind, name):
    """Choose scaling on the validation day, then predict test."""
    tr, va, te = [(m.role == r).to_numpy() for r in ("train", "val", "test")]
    best = max((True, False), key=lambda sc: m4.metric(name, tg[name][va], fit_pred(X, tg[name], tr, va, kind, name, sc)))
    return fit_pred(X, tg[name], tr, te, kind, name, best), best, \
        m4.metric(name, tg[name][va], fit_pred(X, tg[name], tr, va, kind, name, best))


# ---------------------------------------------------------------- hierarchical paired bootstrap (C2, C3)
def boot(name, y, day, blk, PA, PB, B=10_000, seed=0):
    rng = np.random.default_rng(seed); days = np.unique(day)
    blocks = {d: np.unique(blk[day == d]) for d in days}; ib = {b: np.flatnonzero(blk == b) for b in np.unique(blk)}
    pt = np.mean([m4.metric(name, y, p) for p in PA]) - np.mean([m4.metric(name, y, p) for p in PB])
    ds = np.empty(B)
    for i in range(B):
        ii = np.concatenate([ib[b] for d in rng.choice(days, len(days)) for b in rng.choice(blocks[d], len(blocks[d]))])
        sa = rng.integers(0, len(PA), len(PA)); sb = rng.integers(0, len(PB), len(PB))
        ds[i] = np.mean([m4.metric(name, y[ii], PA[s][ii]) for s in sa]) - np.mean([m4.metric(name, y[ii], PB[s][ii]) for s in sb])
    p = (1 + 2 * min((ds <= 0).sum(), (ds >= 0).sum())) / (B + 1)
    return dict(delta=pt, lo=float(np.quantile(ds, 0.025)), hi=float(np.quantile(ds, 0.975)), p=min(1.0, p))


# ---------------------------------------------------------------- Q3 conditional leakage (C6)
def cond_leak(X, m, morph):
    y = (m.run_id.to_numpy() == 0).astype(int); g = m.date.astype(str).to_numpy()
    e19 = PCA(19, random_state=0).fit_transform(StandardScaler().fit_transform(X))
    out = {}
    for kind, clf in (("lr", LogisticRegression(max_iter=3000)), ("knn", KNeighborsClassifier(15))):
        def sk(Z): return m4.skill(y, cross_val_predict(make_pipeline(StandardScaler(), clf), Z, y, groups=g, cv=GroupKFold(5)))
        base = sk(morph); both = sk(np.c_[StandardScaler().fit_transform(e19), StandardScaler().fit_transform(morph)])
        out[kind] = dict(delta=both - base, alone=sk(X if X.shape[1] <= 64 else e19), morph=base)
    return out


# ---------------------------------------------------------------- dry run
def dryrun():
    m, W = m4.load(); tg = m4.targets(m); rng = np.random.default_rng(0); n = len(m)
    morph = m[[c for c in m.columns if c.startswith("m8_")]].fillna(0).to_numpy(float)
    noise = rng.standard_normal((n, 64))
    planted = {"morph8": morph, "noisy_lobes": np.eye(5)[tg["lobes"]] + rng.standard_normal((n, 5)) * 0.5,
               "noisy_duration": np.c_[tg["duration"] / tg["duration"].std() + rng.standard_normal(n) * 0.5, rng.standard_normal((n, 8))],
               "noise": noise}
    res = {}
    te = (m.role == "test").to_numpy()
    for nm, X in planted.items():
        res[nm] = {f"{k}_{s}": m4.metric(s, tg[s][te], probe_all(X, m, tg, k, s)[0]) for k in ("knn", "lin") for s in PRIMARY}
    runcol = np.c_[noise, (m.run_id.to_numpy() == 0)[:, None] * 5.0]
    res["leak_planted"] = cond_leak(runcol, m, morph); res["leak_morphcopy"] = cond_leak(morph + rng.standard_normal(morph.shape) * 1e-3, m, morph)
    yshuf = m.copy(); dmap = dict(zip(m.date.unique(), rng.permutation(m.date.unique()))); yshuf["run_id"] = m.date.map(dict(zip(m.date, m.run_id))).values
    ok = (min(res["morph8"]["knn_polarity"], res["morph8"]["lin_polarity"]) >= 0.90
          and min(res["noisy_lobes"]["knn_lobes"], res["noisy_lobes"]["lin_lobes"]) >= 0.50
          and min(res["noisy_duration"]["knn_duration"], res["noisy_duration"]["lin_duration"]) >= 0.50
          and max(res["noise"].values()) <= 0.10
          and res["leak_planted"]["lr"]["delta"] >= 0.50 and res["leak_morphcopy"]["lr"]["delta"] <= 0.05)
    res["PASS"] = bool(ok); json.dump(res, open(os.path.join(HERE, "dryrun.json"), "w"), indent=1, default=float)
    print(json.dumps(res, indent=1, default=lambda v: round(float(v), 3)))


# ---------------------------------------------------------------- gates (C5)
@torch.no_grad()
def gates():
    m, W = m4.load(); tg = m4.targets(m); va = (m.role == "val").to_numpy(); out = {"G1": {}, "G2": {}}
    x = crops()[torch.tensor(va, device=DEV)]
    for s in SEEDS:                                                     # G1: MAE vs linear interpolation, same mask
        enc, _ = build("mae_tf"); enc.load_state_dict(torch.load(ck("mae_tf", s, "final"), map_location=DEV)); enc = enc.to(DEV).eval()
        g = torch.Generator(device=DEV); g.manual_seed(12345)
        n = x.shape[1] // enc.enc.patch; mk = enc.make_mask(x.shape[0], n, 0.75, g, DEV)
        pred, P = enc.forward_masked(x, mk); tn = enc.norm_t(P); mse_model = F.mse_loss(pred[mk], tn[mk]).item()
        xs = x.cpu().numpy(); smk = mk.repeat_interleave(enc.enc.patch, 1).cpu().numpy(); interp = xs.copy()
        for i in range(len(xs)):
            known = np.flatnonzero(~smk[i]); interp[i, smk[i]] = np.interp(np.flatnonzero(smk[i]), known, xs[i, known])
        Pi = torch.tensor(interp, device=DEV).view_as(P)                                   # C10: interpolation judged on shape,
        sdi = Pi.std(-1, keepdim=True); Pn = torch.where(sdi > 1e-8, (Pi - Pi.mean(-1, keepdim=True)) / (sdi + 1e-12), torch.zeros_like(Pi))
        ok = mk & (P.std(-1) >= 1e-3)                                                       # exclude flat true patches for both
        mse_model = F.mse_loss(pred[ok], tn[ok]).item(); mse_interp = F.mse_loss(Pn[ok], tn[ok]).item()
        out["G1"][s] = dict(mse_model=mse_model, mse_interp=mse_interp, ratio=mse_model / mse_interp, passed=mse_model <= 0.90 * mse_interp)
    out["G1"]["mae_excluded"] = sum(not v["passed"] for v in out["G1"].values()) >= 3
    for meth in SSL:                                                    # G2: validation kNN mean vs own init
        for s in SEEDS:
            sc = {}
            for which in ("final", "init"):
                X = emb(meth, s, which); tr = (m.role == "train").to_numpy()
                sc[which] = float(np.mean([m4.metric(nm, tg[nm][va], fit_pred(X, tg[nm], tr, va, "knn", nm, True)) for nm in PRIMARY]))
            out["G2"][f"{meth}_s{s}"] = dict(**sc, passed=sc["final"] >= sc["init"] + 0.02)
    json.dump(out, open(os.path.join(HERE, "gates.json"), "w"), indent=1, default=float); print(json.dumps(out, indent=1, default=float))


# ---------------------------------------------------------------- scoring
def holm(p):
    p = np.asarray(p); o = np.argsort(p); adj = np.empty_like(p); run = 0.0
    for r, i in enumerate(o): run = max(run, min(1.0, (len(p) - r) * p[i])); adj[i] = run
    return adj


def score():
    m, W = m4.load(); tg = m4.targets(m); te = (m.role == "test").to_numpy()
    day = m.date.astype(str).to_numpy()[te]; blk = m.blk.to_numpy()[te]
    gate = json.load(open(os.path.join(HERE, "gates.json"))); ssl = [s for s in SSL if not (s == "mae_tf" and gate["G1"]["mae_excluded"])]
    morph = m[[c for c in m.columns if c.startswith("m8_")]].fillna(0).to_numpy(float)
    P, absrows, leak, choice = {}, [], [], []
    for s in SEEDS:
        R = representations(s)
        for rep, X in R.items():
            for k in ("knn", "lin"):
                for nm in PRIMARY:
                    p, sc, vs = probe_all(X, m, tg, k, nm); P[(rep, s, k, nm)] = p; choice.append(dict(rep=rep, seed=s, probe=k, score=nm, scaled=sc, val=vs))
                    absrows.append(dict(rep=rep, seed=s, probe=k, score=nm, value=m4.metric(nm, tg[nm][te], p)))
            if not rep.endswith("_pca") or rep == "B4_pca64":
                cl = cond_leak(X, m, morph); leak.append(dict(rep=rep, seed=s, **{f"{a}_{b}": cl[a][b] for a in cl for b in cl[a]}))
        print("seed done", s, flush=True)
    pd.DataFrame(absrows).to_csv(os.path.join(HERE, "scores_abs.csv"), index=False)
    pd.DataFrame(choice).to_csv(os.path.join(HERE, "scaling_choice.csv"), index=False)
    pd.DataFrame(leak).to_csv(os.path.join(HERE, "leakage.csv"), index=False)
    C = pd.DataFrame(choice)

    def preds(rep, k, nm):
        seeds = [s for s in SEEDS if (rep, s, k, nm) in P]
        return [P[(rep, s, k, nm)] for s in seeds]

    def best_variant(base, k, nm):                                      # C4: full vs PCA-64 chosen on validation
        cand = [r for r in (base, base + "_pca") if not C[C.rep == r].empty]
        return max(cand, key=lambda r: C[(C.rep == r) & (C.probe == k) & (C.score == nm)].val.mean())

    jobs = []
    for k in ("knn", "lin"):
        for nm in PRIMARY:
            bases = {"B1": "B1_cnn_init", "B2": best_variant("B2_tf_tokens", k, nm), "B3": best_variant("B3_raw", k, nm), "B4": "B4_pca64"}
            for a in ssl:
                for bl, b in bases.items(): jobs.append((k, nm, a, b, "vs_baseline"))
            for i in range(len(ssl)):
                for j in range(i + 1, len(ssl)): jobs.append((k, nm, ssl[i], ssl[j], "ssl_pair"))
    res = Parallel(n_jobs=14)(delayed(boot)(nm, tg[nm][te], day, blk, preds(a, k, nm), preds(b, k, nm), 10_000, 0) for (k, nm, a, b, _) in jobs)
    rows = [dict(probe=k, score=nm, a=a, b=b, kind=kind, **r) for (k, nm, a, b, kind), r in zip(jobs, res)]
    Rdf = pd.DataFrame(rows)
    for k in ("knn", "lin"):
        msk = Rdf.probe == k; Rdf.loc[msk, "p_holm"] = holm(Rdf.loc[msk, "p"].to_numpy())
    Rdf.to_csv(os.path.join(HERE, "comparisons.csv"), index=False)
    verdict(Rdf, pd.DataFrame(leak), ssl)


def verdict(R, L, ssl):
    def outcome(a, b, nm):
        x = R[(R.a == a) & (R.b == b) & (R.score == nm)]
        if len(x) != 2: return 0
        win = all(r.delta >= 0.02 and r.p_holm < 0.05 and r.lo > 0 for r in x.itertuples())
        loss = all(r.delta <= -0.02 and r.p_holm < 0.05 and r.hi < 0 for r in x.itertuples())
        return 1 if win else (-1 if loss else 0)
    out = {"Q1": {}, "Q2": {}, "Q3": {}}
    for a in ssl:
        per = {}
        for bl in ("B1", "B2", "B3", "B4"):
            res = []
            for nm in PRIMARY:                                          # bugfix (review): baseline variant chosen PER PROBE
                x = R[(R.a == a) & (R.score == nm) & (R.kind == "vs_baseline") & (R.b.str.startswith(bl))]
                if len(x) != 2: res.append(0); continue
                win = all(r.delta >= 0.02 and r.p_holm < 0.05 and r.lo > 0 for r in x.itertuples())
                loss = all(r.delta <= -0.02 and r.p_holm < 0.05 and r.hi < 0 for r in x.itertuples())
                res.append(1 if win else (-1 if loss else 0))
            per[bl] = dict(wins=sum(r == 1 for r in res), losses=sum(r == -1 for r in res))
        out["Q1"][a] = dict(per_baseline=per, beats_all=all(v["wins"] >= 2 and v["losses"] == 0 for v in per.values()))
    sc = {a: 0 for a in ssl}
    for i in range(len(ssl)):
        for j in range(i + 1, len(ssl)):
            for nm in PRIMARY:
                o = outcome(ssl[i], ssl[j], nm); sc[ssl[i]] += o; sc[ssl[j]] -= o
    out["Q2"] = dict(sorted(sc.items(), key=lambda kv: -kv[1]))
    out["Q1_conclusion"] = ("at least one SSL method beats all baselines" if any(v["beats_all"] for v in out["Q1"].values())
                            else "no evidence that self-supervised training adds value over random or raw features at this data size")
    for rep, g in L.groupby("rep"):
        d = g.lr_delta.to_numpy(); rng = np.random.default_rng(0)
        bs = [rng.choice(d, len(d)).mean() for _ in range(2000)] if len(d) > 1 else [d.mean()]
        out["Q3"][rep] = dict(delta_lr=float(d.mean()), lo=float(np.quantile(bs, 0.025)), delta_knn=float(g.knn_delta.mean()),
                              is0204_alone=float(g.lr_alone.mean()), morph=float(g.lr_morph.mean()),
                              flag=bool(d.mean() > 0.05 and np.quantile(bs, 0.025) > 0))
    json.dump(out, open(os.path.join(HERE, "verdict.json"), "w"), indent=1, default=float); print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    c = sys.argv[1]
    if c == "verdict":
        g = json.load(open(os.path.join(HERE, "gates.json"))); ssl = [x for x in SSL if not (x == "mae_tf" and g["G1"]["mae_excluded"])]
        verdict(pd.read_csv(os.path.join(HERE, "comparisons.csv")), pd.read_csv(os.path.join(HERE, "leakage.csv")), ssl); sys.exit()
    if c == "train": train(sys.argv[2], int(sys.argv[3]))
    else: {"dryrun": dryrun, "gates": gates, "score": score}[c]()
