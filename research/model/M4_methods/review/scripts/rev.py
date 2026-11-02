import sys, numpy as np, pandas as pd, json
from sklearn.metrics import balanced_accuracy_score
from sklearn.linear_model import LogisticRegression, RidgeCV
sys.path.insert(0,"research/model/M4_methods"); import m4
m,_ = m4.load(); E = np.load(sys.argv[1]); tg = m4.targets(m)
tr = (m.role=="train").to_numpy(); te = (m.role=="test").to_numpy()
blk = m.blk.to_numpy()[te]; day = m.date.astype(str).to_numpy()[te]
def z(X):
    mu, sd = X[tr].mean(0), X[tr].std(0)+1e-8; return (X-mu)/sd
def knn(X, y, k=15):  # own kNN: majority vote / mean, euclidean on train-z-scored features
    A, B = X[te], X[tr]
    d = (A**2).sum(1)[:,None] - 2*A@B.T + (B**2).sum(1)[None]
    nn = np.argpartition(d, k, 1)[:, :k]; yy = y[tr][nn]
    if y.dtype.kind=='f': return yy.mean(1)
    cls = np.unique(y[tr]); cnt = np.stack([(yy==c).sum(1) for c in cls],1)
    return cls[cnt.argmax(1)]
def lin(X, y):
    if y.dtype.kind=='f': return RidgeCV(alphas=np.logspace(-3,3,13)).fit(X[tr],y[tr]).predict(X[te])
    return LogisticRegression(C=1.0, max_iter=3000).fit(X[tr],y[tr]).predict(X[te])
def met(nm, y, p):
    if nm=="duration": return 1-((y-p)**2).sum()/((y-y.mean())**2).sum()
    return balanced_accuracy_score(y, p, adjusted=True)
PR = ["polarity","lobes","duration"]
cache = {}
def preds(key, probe, nm):
    k=(key,probe,nm)
    if k not in cache:
        X = z(E[key].astype(np.float64)); cache[k] = (knn if probe=="knn" else lin)(X, tg[nm])
    return cache[k]
def P(meth, which, probe, nm): return [preds(f"{meth}__{which}__{s}", probe, nm) for s in range(5)]
ub = np.unique(blk); ib = [np.flatnonzero(blk==b) for b in ub]
def boot(nm, PA, PB, B=2000, seed=123):
    y = tg[nm][te]; rng = np.random.default_rng(seed)
    pt = np.mean([met(nm,y,p) for p in PA]) - np.mean([met(nm,y,p) for p in PB])
    ds = np.empty(B)
    for b in range(B):
        ii = np.concatenate([ib[j] for j in rng.integers(0,len(ub),len(ub))])
        sa = rng.integers(0,5,5); sb = rng.integers(0,5,5)
        ds[b] = np.mean([met(nm,y[ii],PA[s][ii]) for s in sa]) - np.mean([met(nm,y[ii],PB[s][ii]) for s in sb])
    lo, hi = np.quantile(ds,[.025,.975]); p = min(1, 2*min((ds<=0).mean(), (ds>=0).mean()) + 1/B)
    # per-day gaps (consistency) and per-seed spread
    pdays = {d: round(np.mean([met(nm,y[day==d],p[day==d]) for p in PA]) - np.mean([met(nm,y[day==d],p[day==d]) for p in PB]),3) for d in np.unique(day)}
    return dict(delta=round(pt,4), lo=round(lo,4), hi=round(hi,4), p=round(p,4), days=pdays)
res = {}
jobs = [("pair","simclr_tf","vicreg_tf","lobes"), ("pair","simclr_cnn","vicreg_tf","duration"),
        ("learn","simclr_cnn",None,"polarity"),("learn","simclr_cnn",None,"duration"),("learn","simclr_cnn",None,"lobes"),
        ("learn","mae_tf",None,"polarity"),("learn","mae_tf",None,"lobes"),("learn","mae_tf",None,"duration"),
        ("learn","maeflat",None,"polarity"),("learn","maeflat",None,"lobes"),("learn","maeflat",None,"duration")]
for kind,a,b,nm in jobs:
    for probe in ("knn","lin"):
        PA = P(a,"final",probe,nm); PB = P(b,"final",probe,nm) if kind=="pair" else P(a,"init",probe,nm)
        r = boot(nm, PA, PB); r.update(A=float(np.mean([met(nm,tg[nm][te],p) for p in PA])), B=float(np.mean([met(nm,tg[nm][te],p) for p in PB])))
        key=f"{kind}:{a}:{b or 'init'}:{nm}:{probe}"; res[key]=r; print(key, r, flush=True)
json.dump(res, open(sys.argv[2],"w"), indent=1)
