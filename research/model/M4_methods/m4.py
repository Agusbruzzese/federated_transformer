"""M4: four self-supervised methods on the extracted events. Implements PREREG.md (frozen, sha256 3c9291c26acde93e).

    python research/model/M4_methods/m4.py dryrun          # planted controls through the scorer
    python research/model/M4_methods/m4.py smoke           # each method overfits 64 events
    python research/model/M4_methods/m4.py train <method> <cell> <seed>
    python research/model/M4_methods/m4.py select          # validation-day selection (seeds 0,1)
    python research/model/M4_methods/m4.py score           # frozen scorer on the test days
methods: simclr_tf, vicreg_tf, mae_tf, simclr_cnn
"""
import os, sys, json, math, hashlib, time
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F

sys.path.insert(0, os.getcwd())
from ssl_central import data as D, augment as A, models as Mo, config as C

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.getcwd(), "data", "m4")
TEST_DAYS = ["2025-02-07", "2025-02-13", "2025-02-19", "2025-02-25"]
VAL_DAY = "2025-02-10"
CELLS = {"simclr_tf": [dict(T=0.1), dict(T=0.2)], "simclr_cnn": [dict(T=0.1), dict(T=0.2)],
         "vicreg_tf": [dict(lr=1e-3), dict(lr=3e-4)], "mae_tf": [dict(mask=0.5), dict(mask=0.75)]}
EPOCHS, BATCH, WD = 100, 256, 1e-4
DEV = "cuda"


# ------------------------------------------------------------------ data
_CACHE = {}
def load():
    if "m" in _CACHE:
        return _CACHE["m"], _CACHE["W"]
    m, W, _, _ = D.load_all()
    keep = (m.snr >= 20).to_numpy()
    m = m[keep].reset_index(drop=True); W = W[keep]
    day = m.date.astype(str).to_numpy()
    m["role"] = np.where(np.isin(day, TEST_DAYS), "test", np.where(day == VAL_DAY, "val", "train"))
    m["blk"] = pd.to_datetime(m.wall_time).dt.floor("30min").astype("int64")
    _CACHE["m"], _CACHE["W"] = m, W
    return m, W


def bgpool():
    bg = np.load(os.path.join(C.OUT, "bgpool.npy")); meta = pd.read_parquet(os.path.join(C.OUT, "bgpool_meta.parquet"))
    d = pd.to_datetime(meta.wall_time).dt.date.astype(str).to_numpy()
    ok = ~np.isin(d, TEST_DAYS + [VAL_DAY])                          # B1: exclude test and validation days
    return bg[ok], int((~ok).sum())


def targets(m):
    return dict(polarity=(m.m8_polarity.to_numpy() > 0).astype(int),
                lobes=np.minimum(m.m8_n_lobes.to_numpy(), 4).astype(int),
                duration=np.log10(m.m8_dur10.clip(lower=0.05).to_numpy()))


# ------------------------------------------------------------------ models
class VICRegHead(nn.Module):
    def __init__(self, d=C.EMB, h=256):
        super().__init__(); self.net = nn.Sequential(nn.Linear(d, h), nn.ReLU(), nn.Linear(h, h))
    def forward(self, x): return self.net(x)


def vicreg_loss(z1, z2, lam=25.0, mu=25.0, nu=1.0):
    inv = F.mse_loss(z1, z2)
    def std(z): return F.relu(1 - torch.sqrt(z.var(0) + 1e-4)).mean()
    def cov(z):
        z = z - z.mean(0); n, d = z.shape; c = (z.T @ z) / (n - 1)
        off = c - torch.diag(torch.diag(c)); return (off ** 2).sum() / d
    return lam * inv + mu * (std(z1) + std(z2)) + nu * (cov(z1) + cov(z2))


class MAE(nn.Module):
    """B5: same PatchTransformer encoder; masked tokens replaced by a learnable token; linear patch decoder."""
    def __init__(self):
        super().__init__()
        self.enc = Mo.PatchTransformer()
        d = self.enc.embed.out_features
        self.mask_token = nn.Parameter(torch.zeros(1, 1, d)); nn.init.normal_(self.mask_token, std=0.02)
        self.dec = nn.Linear(d, self.enc.patch)

    def tokens(self, x):
        B = x.shape[0]; P = x.view(B, -1, self.enc.patch)
        return P, self.enc.embed(P)

    def loss(self, x, ratio, g):
        P, t = self.tokens(x); B, n, d = t.shape
        k = int(round(ratio * n))
        r = torch.rand(B, n, generator=g, device=x.device)
        idx = r.argsort(1)[:, :k]
        mask = torch.zeros(B, n, dtype=torch.bool, device=x.device); mask.scatter_(1, idx, True)
        t = torch.where(mask[..., None], self.mask_token.expand(B, n, d), t) + self.enc.pos
        h = self.enc.norm(self.enc.enc(t))
        pred = self.dec(h)
        return F.mse_loss(pred[mask], P[mask])

    def embed(self, x):                                            # B5: pooled, normed tokens (64-d), no masking
        _, t = self.tokens(x)
        return self.enc.norm(self.enc.enc(t + self.enc.pos)).mean(1)


def build(method):
    if method == "mae_tf":
        return MAE(), None
    enc = Mo.make("cnn" if method == "simclr_cnn" else "tf")
    head = VICRegHead() if method == "vicreg_tf" else Mo.Head()
    return enc, head


def embed_fn(method, enc):
    return enc.embed if method == "mae_tf" else enc


# ------------------------------------------------------------------ training
def tagname(method, cell, seed): return f"{method}_c{cell}_s{seed}"


def train(method, cell, seed, subset=None, epochs=EPOCHS, save=True):
    m, W = load(); cfg = CELLS[method][cell]
    idx = np.flatnonzero((m.role == "train").to_numpy())
    assert not set(m.event_id.iloc[idx]) & set(m.event_id[m.role != "train"]), "firewall"
    if subset: idx = np.sort(np.random.default_rng(seed).choice(idx, subset, replace=False))
    s = m.iloc[idx]; bg, nbg_out = bgpool()
    T = dict(W=torch.tensor(W[idx], device=DEV), fs=torch.tensor(s.fs_eff.to_numpy(np.float32), device=DEV),
             ev0=torch.tensor(s.ev_i0.to_numpy(), device=DEV), ev1=torch.tensor(s.ev_i1.to_numpy(), device=DEV),
             sig=torch.tensor(s.sigma_ev.to_numpy(np.float32), device=DEV),
             t=torch.tensor(s.t_s.to_numpy(np.float64), device=DEV), bg=torch.tensor(bg, device=DEV))
    N = len(idx)
    torch.manual_seed(seed); np.random.seed(seed)
    enc, head = build(method); enc = enc.to(DEV); head = head.to(DEV) if head is not None else None
    tag = tagname(method, cell, seed); od = os.path.join(OUT, "ckpt", tag)
    if save:
        os.makedirs(od, exist_ok=True); torch.save(enc.state_dict(), os.path.join(od, "init.pt"))
    params = list(enc.parameters()) + (list(head.parameters()) if head is not None else [])
    lr = cfg.get("lr", 3e-4)
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=WD)
    bs = min(BATCH, N); spe = max(1, N // bs); total = spe * epochs
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: 0.5 * (1 + math.cos(math.pi * min(i, total) / total)))
    g = torch.Generator(device=DEV); g.manual_seed(seed); gp = torch.Generator(); gp.manual_seed(seed)
    losses = []; t0 = time.time()
    for ep in range(epochs):
        perm = torch.randperm(N, generator=gp).to(DEV); L = []
        enc.train()
        for b in range(spe):
            ii = perm[b * bs:(b + 1) * bs]
            args = (T["W"][ii], T["fs"][ii], T["ev0"][ii], T["ev1"][ii], T["sig"][ii], T["bg"], g)
            if method == "mae_tf":
                loss = enc.loss(A.train_view(*args, parts=("crop",)), cfg["mask"], g)
            else:
                v1, v2 = A.train_view(*args), A.train_view(*args)
                if method == "vicreg_tf":
                    loss = vicreg_loss(head(enc(v1)), head(enc(v2)))
                else:
                    loss, _ = Mo.nt_xent(head(enc(v1)), head(enc(v2)), T["t"][ii], cfg["T"])
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step(); L.append(loss.item())
        losses.append(float(np.mean(L)))
    log = dict(method=method, cell=cell, cfg=cfg, seed=seed, n_train=N, bg_excluded=nbg_out, epochs=epochs,
               loss=losses, secs=time.time() - t0,
               train_ids_sha256=hashlib.sha256("\n".join(sorted(s.event_id)).encode()).hexdigest())
    if save:
        torch.save(enc.state_dict(), os.path.join(od, "final.pt")); json.dump(log, open(os.path.join(od, "log.json"), "w"))
    print(f"[{tag}] loss {losses[0]:.4f} -> {losses[-1]:.4f}  {log['secs']:.0f}s", flush=True)
    return log


@torch.no_grad()
def embed(method, cell, seed, which):
    m, W = load()
    enc, _ = build(method); enc.load_state_dict(torch.load(os.path.join(OUT, "ckpt", tagname(method, cell, seed), f"{which}.pt"), map_location=DEV))
    enc = enc.to(DEV).eval(); f = embed_fn(method, enc)
    out = []
    for i in range(0, len(W), 2048):
        x = A.eval_view(torch.tensor(W[i:i + 2048], device=DEV), torch.tensor(m.fs_eff.to_numpy(np.float32)[i:i + 2048], device=DEV))
        out.append(f(x).float().cpu().numpy())
    return np.concatenate(out)


# ------------------------------------------------------------------ probes (B2)
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold, cross_val_predict


def skill(y, p):
    cls = np.unique(y); K = len(cls)
    return (np.mean([np.mean(p[y == c] == c) for c in cls]) - 1 / K) / (1 - 1 / K)


def r2(y, p): return 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)


def metric(name, y, p): return r2(y, p) if name == "duration" else skill(y, p)


def probe(kind, name, scale=True):
    if name == "duration":
        est = KNeighborsRegressor(15) if kind == "knn" else RidgeCV(alphas=np.logspace(-3, 3, 13))
    else:
        est = KNeighborsClassifier(15) if kind == "knn" else LogisticRegression(C=1.0, max_iter=3000)
    return make_pipeline(StandardScaler(), est) if scale else est


def predict_test(X, m, tg, kind, name, scale=True, eval_role="test"):
    tr = (m.role == "train").to_numpy(); te = (m.role == eval_role).to_numpy()
    return probe(kind, name, scale).fit(X[tr], tg[name][tr]).predict(X[te])


def leak_scores(X, m):
    """B7: is-0204 over all events and test-day identity over test events; max over LR and kNN."""
    out = {}
    y = (m.run_id.to_numpy() == 0).astype(int); g = m.blk.to_numpy()
    te = (m.role == "test").to_numpy(); yd = m.date.astype(str).to_numpy()[te]
    for lab, XX, yy, gg in [("is0204", X, y, g), ("testday", X[te], yd, g[te])]:
        vals = []
        for clf in (LogisticRegression(max_iter=3000), KNeighborsClassifier(15)):
            p = cross_val_predict(make_pipeline(StandardScaler(), clf), XX, yy, groups=gg, cv=GroupKFold(5))
            vals.append(skill(yy, p))
        out[lab] = max(vals)
    return out


PRIMARY = ["polarity", "lobes", "duration"]


# ------------------------------------------------------------------ dry run (B8)
def dryrun():
    m, W = load(); tg = targets(m); rng = np.random.default_rng(0); n = len(m)
    noise = rng.standard_normal((n, 64))
    lob = np.eye(5)[tg["lobes"]] * 1.0 + rng.standard_normal((n, 5)) * 0.5
    dur = np.c_[tg["duration"] / tg["duration"].std() + rng.standard_normal(n) * 0.5, rng.standard_normal((n, 8))]
    morph = m[[c for c in m.columns if c.startswith("m8_")]].fillna(0).to_numpy(float)
    runhot = np.c_[noise, np.eye(4)[m.run_id.to_numpy()] * 5]
    dayhot = np.c_[noise, pd.get_dummies(m.date.astype(str)).to_numpy(float) * 5]
    res = {}
    for nm, X in [("morph8", morph), ("noisy_lobes", lob), ("noisy_duration", dur), ("noise", noise), ("noise+run", runhot), ("noise+day", dayhot)]:
        r = {f"{k}_{s}": metric(s, tg[s][(m.role == 'test').to_numpy()], predict_test(X, m, tg, k, s)) for k in ("knn", "lin") for s in PRIMARY}
        if nm.startswith("noise"): r.update(leak_scores(X, m))
        res[nm] = r
    ok = (min(res["morph8"]["knn_polarity"], res["morph8"]["lin_polarity"]) >= 0.90
          and min(res["noisy_lobes"]["knn_lobes"], res["noisy_lobes"]["lin_lobes"]) >= 0.50
          and min(res["noisy_duration"]["knn_duration"], res["noisy_duration"]["lin_duration"]) >= 0.50
          and max(v for k, v in res["noise"].items()) <= 0.10
          and res["noise+run"]["is0204"] >= 0.90 and res["noise+day"]["testday"] >= 0.90)
    res["PASS"] = bool(ok)
    json.dump(res, open(os.path.join(HERE, "dryrun.json"), "w"), indent=1)
    for k, v in res.items(): print(k, v if isinstance(v, bool) else {a: round(b, 3) for a, b in v.items()})


def smoke():
    out = {}
    for method in CELLS:
        log = train(method, 0, 0, subset=64, epochs=200, save=False)
        out[method] = dict(first=log["loss"][0], last=log["loss"][-1], drop=1 - log["loss"][-1] / log["loss"][0])
    out["PASS"] = all(v["drop"] > 0.30 for v in out.values())
    json.dump(out, open(os.path.join(HERE, "smoke.json"), "w"), indent=1); print(out)


# ------------------------------------------------------------------ selection (B4)
def select():
    m, W = load(); tg = targets(m); rows = []
    for method in CELLS:
        for cell in range(2):
            for seed in (0, 1):
                X = embed(method, cell, seed, "final")
                sc = [metric(s, tg[s][(m.role == 'val').to_numpy()], predict_test(X, m, tg, "knn", s, eval_role="val")) for s in PRIMARY]
                rows.append(dict(method=method, cell=cell, seed=seed, val=float(np.mean(sc)), **dict(zip(PRIMARY, sc))))
    df = pd.DataFrame(rows); df.to_csv(os.path.join(HERE, "selection.csv"), index=False)
    sel = {}
    for method, g in df.groupby("method"):
        mu = g.groupby("cell").val.mean(); sd = g.groupby("cell").val.std().mean()
        best = int(mu.idxmax()); sel[method] = 0 if abs(mu[0] - mu[1]) < sd else best   # B4 tie-break
        print(method, mu.round(4).to_dict(), "seed sd", round(sd, 4), "-> cell", sel[method])
    json.dump(sel, open(os.path.join(HERE, "selection.json"), "w"), indent=1)


# ------------------------------------------------------------------ scoring
def holm(p):
    p = np.asarray(p); o = np.argsort(p); adj = np.empty_like(p); run = 0
    for r, i in enumerate(o):
        run = max(run, min(1, (len(p) - r) * p[i])); adj[i] = run
    return adj


def boot_pair(name, y, blk, PA, PB, rng, B=2000):
    """B3: same resampled blocks for A and B; seeds of A and B drawn independently. PA, PB: [n_seeds, n_test]."""
    ub = np.unique(blk); ib = {b: np.flatnonzero(blk == b) for b in ub}
    pt = np.mean([metric(name, y, p) for p in PA]) - np.mean([metric(name, y, p) for p in PB])
    ds = []
    for _ in range(B):
        ii = np.concatenate([ib[b] for b in rng.choice(ub, len(ub))])
        sa = rng.integers(0, len(PA), len(PA)); sb = rng.integers(0, len(PB), len(PB))
        ds.append(np.mean([metric(name, y[ii], PA[s][ii]) for s in sa]) - np.mean([metric(name, y[ii], PB[s][ii]) for s in sb]))
    ds = np.array(ds)
    p = min(1.0, 2 * min((ds <= 0).mean(), (ds >= 0).mean()) + 1 / B)
    return pt, np.quantile(ds, 0.025), np.quantile(ds, 0.975), p


def score():
    from sklearn.decomposition import PCA
    m, W = load(); tg = targets(m); te = (m.role == "test").to_numpy(); sel = json.load(open(os.path.join(HERE, "selection.json")))
    blk = m.blk.to_numpy()[te]; rng = np.random.default_rng(0)
    preds = {}; absrows = []; leak = []
    for method in CELLS:
        for s in range(5):
            for which in ("final", "init"):
                X = embed(method, sel[method], s, which)
                for k in ("knn", "lin"):
                    for nm in PRIMARY:
                        p = predict_test(X, m, tg, k, nm); preds[(method, which, s, k, nm)] = p
                        absrows.append(dict(rep=method, which=which, seed=s, probe=k, score=nm, value=metric(nm, tg[nm][te], p)))
                leak.append(dict(method=method, which=which, seed=s, **leak_scores(X, m)))
        print("embedded+probed", method, flush=True)
    # baselines, with and without z-scoring (B2/baselines)
    raw = A.eval_view(torch.tensor(W, device=DEV), torch.tensor(m.fs_eff.to_numpy(np.float32), device=DEV)).cpu().numpy()
    tr = (m.role == "train").to_numpy()
    base = {"raw": raw, "pca64": PCA(64, random_state=0).fit(raw[tr]).transform(raw),
            "morph8": m[[c for c in m.columns if c.startswith("m8_")]].fillna(0).to_numpy(float)}
    for nm_b, X in base.items():
        for k in ("knn", "lin"):
            for nm in PRIMARY:
                v = max(metric(nm, tg[nm][te], predict_test(X, m, tg, k, nm, scale=sc)) for sc in ((True, False) if nm_b != "morph8" else (True,)))
                absrows.append(dict(rep=nm_b, which="baseline", seed=0, probe=k, score=nm, value=v))
    pd.DataFrame(absrows).to_csv(os.path.join(HERE, "scores_abs.csv"), index=False)
    pd.DataFrame(leak).to_csv(os.path.join(HERE, "leakage.csv"), index=False)
    # pairwise and learning checks
    rows = []
    methods = list(CELLS)
    for k in ("knn", "lin"):
        fam = []
        for i in range(4):
            for j in range(i + 1, 4):
                for nm in PRIMARY:
                    PA = [preds[(methods[i], "final", s, k, nm)] for s in range(5)]; PB = [preds[(methods[j], "final", s, k, nm)] for s in range(5)]
                    d, lo, hi, p = boot_pair(nm, tg[nm][te], blk, PA, PB, rng)
                    fam.append(dict(kind="pair", probe=k, a=methods[i], b=methods[j], score=nm, delta=d, lo=lo, hi=hi, p=p))
        for r, pa in zip(fam, holm([r["p"] for r in fam])): r["p_holm"] = pa
        rows += fam
        famL = []
        for meth in methods:
            for nm in PRIMARY:
                PA = [preds[(meth, "final", s, k, nm)] for s in range(5)]; PB = [preds[(meth, "init", s, k, nm)] for s in range(5)]
                d, lo, hi, p = boot_pair(nm, tg[nm][te], blk, PA, PB, rng)
                famL.append(dict(kind="learn", probe=k, a=meth, b=meth + "_init", score=nm, delta=d, lo=lo, hi=hi, p=p))
        for r, pa in zip(famL, holm([r["p"] for r in famL])): r["p_holm"] = pa
        rows += famL
        print("bootstrapped probe", k, flush=True)
    R = pd.DataFrame(rows); R.to_csv(os.path.join(HERE, "comparisons.csv"), index=False)
    verdict(R, pd.DataFrame(leak))


def verdict(R, L):
    def wins(a, b, nm, kind="pair"):
        x = R[(R.kind == kind) & (R.a == a) & (R.b == b) & (R.score == nm)]
        both_pos = all((r.delta >= 0.02) and (r.p_holm < 0.05) and (r.lo > 0) for r in x.itertuples())
        both_neg = all((r.delta <= -0.02) and (r.p_holm < 0.05) and (r.hi < 0) for r in x.itertuples())
        return 1 if both_pos else (-1 if both_neg else 0)
    methods = list(CELLS); out = {"pairs": [], "learning": {}, "rank": {}, "leak": {}}
    score_ = {mm: 0 for mm in methods}
    for i in range(4):
        for j in range(i + 1, 4):
            for nm in PRIMARY:
                w = wins(methods[i], methods[j], nm)
                out["pairs"].append(dict(a=methods[i], b=methods[j], score=nm, result={1: f"{methods[i]} better", -1: f"{methods[j]} better", 0: "no clear difference"}[w]))
                score_[methods[i]] += w; score_[methods[j]] -= w
    for mm in methods:
        n = sum(wins(mm, mm + "_init", nm, "learn") == 1 for nm in PRIMARY)
        out["learning"][mm] = dict(scores_beating_init=n, learning=n >= 2)
    out["rank"] = dict(sorted(score_.items(), key=lambda kv: -kv[1]))
    for mm in methods:
        f = L[(L.method == mm) & (L.which == "final")][["is0204", "testday"]].mean(); i0 = L[(L.method == mm) & (L.which == "init")][["is0204", "testday"]].mean()
        out["leak"][mm] = dict(is0204=round(f.is0204, 3), is0204_init=round(i0.is0204, 3), testday=round(f.testday, 3), testday_init=round(i0.testday, 3),
                               leaks=bool(f.is0204 > i0.is0204 + 0.05 or f.testday > i0.testday + 0.05))
    json.dump(out, open(os.path.join(HERE, "verdict.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "train":
        train(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
    elif cmd == "verdict":
        verdict(pd.read_csv(os.path.join(HERE, "comparisons.csv")), pd.read_csv(os.path.join(HERE, "leakage.csv")))
    else:
        {"dryrun": dryrun, "smoke": smoke, "select": select, "score": score}[cmd]()
