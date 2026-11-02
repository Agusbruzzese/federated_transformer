"""H2 scoring, implementing PREREG.md v2 (frozen, sha256 prefix 289b4cbb3727b6ce).

    python research/model/H2_transformer_claim/h2.py dryrun
    python research/model/H2_transformer_claim/h2.py score
    python research/model/H2_transformer_claim/h2.py examples
"""
import os, sys, json
import numpy as np, pandas as pd, torch
from joblib import Parallel, delayed
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor, NearestNeighbors
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA

sys.path.insert(0, os.getcwd())
from ssl_central import data as D, embed as E
from ssl_central_v3 import config as C3, train as T3

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = {0: "0204", 1: "0212", 2: "0218", 3: "0224"}
SEEDS = range(5)
PRIMARY = ["polarity", "lobes", "duration"]
SECONDARY = ["magnitude", "group4"]
NBOOT, ALPHA = 2000, 0.05 / 3          # 98.3 % CI (Bonferroni over 3 primary scores)


# ---------------------------------------------------------------- data
def load():
    m, W, _, _ = D.load_all()
    m = m.reset_index(drop=True)
    keep = np.flatnonzero((m.snr >= 20).to_numpy())
    m = m.iloc[keep].reset_index(drop=True); W = W[keep]
    blk = pd.to_datetime(m.wall_time).dt.floor("30min").astype("int64").to_numpy()
    big = (10 ** m.m8_log_amp.to_numpy() >= 5000).astype(int)
    pol = (m.m8_polarity.to_numpy() > 0).astype(int)
    tg = dict(polarity=pol, lobes=np.minimum(m.m8_n_lobes.to_numpy(), 4).astype(int),
              duration=np.log10(m.m8_dur10.clip(lower=0.05).to_numpy()), magnitude=big, group4=big * 2 + pol)
    return m, W, blk, tg


def ckpt(k, arch, temp, s, which):
    return os.path.join(C3.OUT, "ckpt", T3.tag(k, arch, 3e-4, temp, s, False), f"{which}.pt")


def embed_all(crops, path):
    return E.embed_views(E.load_encoder(path), crops)


# ---------------------------------------------------------------- metrics (fast, used inside bootstrap)
def skill_fast(y, p):
    cls = np.unique(y); K = len(cls)
    rec = [np.mean(p[y == c] == c) for c in cls]
    return (np.mean(rec) - 1 / K) / (1 - 1 / K)


def r2_fast(y, p):
    return 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)


def metric(name, y, p):
    return r2_fast(y, p) if name == "duration" else skill_fast(y, p)


def oof(X, y, g, name):
    est = KNeighborsRegressor(15) if name == "duration" else KNeighborsClassifier(15)
    return cross_val_predict(make_pipeline(StandardScaler(), est), X, y, groups=g, cv=GroupKFold(5))


def lr_skill(X, y, g):
    p = cross_val_predict(make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)), X, y, groups=g, cv=GroupKFold(5))
    return skill_fast(y, p)


def knn_skill(X, y, g):
    p = cross_val_predict(make_pipeline(StandardScaler(), KNeighborsClassifier(15)), X, y, groups=g, cv=GroupKFold(5))
    return skill_fast(y, p)


# ---------------------------------------------------------------- representations per held-out run
def reps_for_fold(k, m, W, crops, blk):
    """Embeddings of ALL tier events in each fold-k model's own space."""
    out = {}
    for s in SEEDS:
        out[("tf", s)] = embed_all(crops, ckpt(k, "tf", 0.1, s, "final"))
        out[("tfinit", s)] = embed_all(crops, ckpt(k, "tf", 0.1, s, "init"))
        out[("tf02", s)] = embed_all(crops, ckpt(k, "tf", 0.2, s, "final"))
        out[("cnn01", s)] = embed_all(crops, ckpt(k, "cnn", 0.1, s, "final"))
        out[("cnn02", s)] = embed_all(crops, ckpt(k, "cnn", 0.2, s, "final"))
    raw = crops.float().cpu().numpy()
    tr = (m.run_id.to_numpy() != k)
    out[("pca", 0)] = PCA(64, random_state=0).fit(raw[tr]).transform(raw)
    out[("raw", 0)] = raw
    out[("morph8", 0)] = m[[c for c in m.columns if c.startswith("m8_")]].fillna(0).to_numpy(float)
    return out


def run_preds(k, reps, m, blk, tg):
    sel = (m.run_id.to_numpy() == k)
    g = blk[sel]
    preds = {}
    for key, X in reps.items():
        for name in PRIMARY + SECONDARY:
            preds[(key, name)] = oof(X[sel], tg[name][sel], g, name)
    return preds, sel


def boot_delta(name, y, g, pa_list, pb_list, rng):
    """Block bootstrap of mean_s metric(A_s) - mean_s metric(B_s) over the run's 30-min blocks."""
    ub = np.unique(g); idx_by = {b: np.flatnonzero(g == b) for b in ub}
    point = np.mean([metric(name, y, p) for p in pa_list]) - np.mean([metric(name, y, p) for p in pb_list])
    ds = []
    for _ in range(NBOOT):
        ii = np.concatenate([idx_by[b] for b in rng.choice(ub, len(ub))])
        yy = y[ii]
        if name != "duration" and len(np.unique(yy)) < len(np.unique(y)):
            continue
        ds.append(np.mean([metric(name, yy, p[ii]) for p in pa_list]) - np.mean([metric(name, yy, p[ii]) for p in pb_list]))
    lo, hi = np.quantile(ds, [ALPHA / 2, 1 - ALPHA / 2])
    return point, lo, hi


# ---------------------------------------------------------------- dry run
def dryrun():
    m, W, blk, tg = load()
    rng = np.random.default_rng(0)
    morph = m[[c for c in m.columns if c.startswith("m8_")]].fillna(0).to_numpy(float)
    noise = rng.standard_normal((len(m), 64))
    leak = np.c_[noise, np.eye(4)[m.run_id.to_numpy()] * 5]
    y0204 = (m.run_id.to_numpy() == 0).astype(int)
    res = {}
    for nm, X in [("morph8", morph), ("noise", noise), ("noise+runonehot", leak)]:
        r = {"is0204_lr": lr_skill(X, y0204, blk)}
        for k in RUNS:
            sel = m.run_id.to_numpy() == k
            for name in PRIMARY + SECONDARY:
                r[f"{RUNS[k]}_{name}"] = metric(name, tg[name][sel], oof(X[sel], tg[name][sel], blk[sel], name))
            r[f"{RUNS[k]}_date"] = lr_skill(X[sel], m.date.to_numpy()[sel], blk[sel])
        res[nm] = r
    ok = (min(res["morph8"][f"{RUNS[k]}_polarity"] for k in RUNS) >= 0.90
          and max(abs(v) for kk, v in res["noise"].items() if not (kk.endswith("date") or kk.endswith("duration"))) <= 0.10
          and max(v for kk, v in res["noise"].items() if kk.endswith("duration")) <= 0.10
          and res["noise+runonehot"]["is0204_lr"] >= 0.90)
    res["PASS"] = bool(ok)
    json.dump(res, open(os.path.join(HERE, "dryrun.json"), "w"), indent=1)
    for k, v in res.items():
        print(k, v if isinstance(v, bool) else {a: round(b, 3) for a, b in v.items()})
    # H4 date threshold per run = morph8 within-run date skill + 0.03 (A5), fixed here before any TF result
    thr = {RUNS[k]: max(0.03, res["morph8"][f"{RUNS[k]}_date"] + 0.03) for k in RUNS}   # A10
    json.dump(thr, open(os.path.join(HERE, "h4_date_thresholds.json"), "w"), indent=1)
    print("H4 date thresholds:", {a: round(b, 3) for a, b in thr.items()})


# ---------------------------------------------------------------- main scoring
def score():
    m, W, blk, tg = load()
    crops = E.eval_crops(W, m.fs_eff.to_numpy())
    thr = json.load(open(os.path.join(HERE, "h4_date_thresholds.json")))
    y0204 = (m.run_id.to_numpy() == 0).astype(int)
    rows, leak_rows = [], []
    for k in RUNS:
        reps = reps_for_fold(k, m, W, crops, blk)
        preds, sel = run_preds(k, reps, m, blk, tg)
        g = blk[sel]; rng = np.random.default_rng(0)
        # absolute scores
        for key in reps:
            for name in PRIMARY + SECONDARY:
                rows.append(dict(run=RUNS[k], rep=key[0], seed=key[1], score=name,
                                 value=metric(name, tg[name][sel], preds[(key, name)])))
        # CNN oracle temperature per run and score, after seed-averaging (A4)
        comps = {}
        for name in PRIMARY + SECONDARY:
            m01 = np.mean([metric(name, tg[name][sel], preds[(("cnn01", s), name)]) for s in SEEDS])
            m02 = np.mean([metric(name, tg[name][sel], preds[(("cnn02", s), name)]) for s in SEEDS])
            comps[("cnn_oracle", name)] = "cnn01" if m01 >= m02 else "cnn02"
        # paired deltas with block-bootstrap CIs
        for name in PRIMARY + SECONDARY:
            y = tg[name][sel]
            tfp = [preds[(("tf", s), name)] for s in SEEDS]
            targets = {"tfinit": [preds[(("tfinit", s), name)] for s in SEEDS],
                       "cnn_oracle": [preds[((comps[("cnn_oracle", name)], s), name)] for s in SEEDS],
                       "cnn01": [preds[(("cnn01", s), name)] for s in SEEDS],
                       "cnn02": [preds[(("cnn02", s), name)] for s in SEEDS],
                       "pca": [preds[(("pca", 0), name)]], "raw": [preds[(("raw", 0), name)]],
                       "tf02_vs_tf01": None}
            for comp, pb in targets.items():
                if comp == "tf02_vs_tf01":
                    pa = [preds[(("tf02", s), name)] for s in SEEDS]; pb = tfp
                else:
                    pa = tfp
                d, lo, hi = boot_delta(name, y, g, pa, pb, rng)
                rows.append(dict(run=RUNS[k], rep=f"DELTA_tf_minus_{comp}", seed=-1, score=name, value=d, lo=lo, hi=hi,
                                 note=comps.get(("cnn_oracle", name), "") if comp == "cnn_oracle" else ""))
        # H4 leakage in each fold model's own space (A5)
        for s in SEEDS:
            for which in ("tf", "tfinit"):
                X = reps[(which, s)]
                r = dict(run=RUNS[k], seed=s, rep=which, is0204_lr=lr_skill(X, y0204, blk), is0204_knn=knn_skill(X, y0204, blk),
                         date_lr=lr_skill(X[sel], m.date.to_numpy()[sel], g))
                leak_rows.append(r)
            # 20 shuffled-label nulls at block level for the trained TF
            X = reps[("tf", s)]; ub = np.unique(blk); lab = {b: y0204[blk == b][0] for b in ub}
            rr = np.random.default_rng(100 + s); nulls = []
            for _ in range(20):
                perm = dict(zip(ub, rr.permutation([lab[b] for b in ub])))
                nulls.append(lr_skill(X, np.array([perm[b] for b in blk]), blk))
            leak_rows[-2]["null_mean"] = float(np.mean(nulls)); leak_rows[-2]["null_max"] = float(np.max(nulls))
        print(f"fold {RUNS[k]} done", flush=True)
        pd.DataFrame(rows).to_csv(os.path.join(HERE, "scores_long.csv"), index=False)
        pd.DataFrame(leak_rows).to_csv(os.path.join(HERE, "leakage.csv"), index=False)
    verdict(thr)


def decide(D, comp, name, runs):
    d = D[(D.rep == f"DELTA_tf_minus_{comp}") & (D.score == name) & (D.run.isin(runs))]
    mean = d.value.mean()
    tf_better = mean >= 0.02 and (d.value > 0).all() and (d.lo > 0).sum() >= len(runs) - 1
    x_better = mean <= -0.02 and (d.value < 0).all() and (d.hi < 0).sum() >= len(runs) - 1
    return dict(comp=comp, score=name, mean_delta=mean, runs_positive=int((d.value > 0).sum()),
                runs_ci_above0=int((d.lo > 0).sum()), runs_ci_below0=int((d.hi < 0).sum()),
                outcome="TF better" if tf_better else ("competitor better" if x_better else "no clear difference"),
                sign_test_p=0.5 ** len(runs) if (d.value > 0).all() else np.nan)


def verdict(thr):
    D = pd.read_csv(os.path.join(HERE, "scores_long.csv"), dtype={"run": str})   # bugfix: keep "0204" as text
    L = pd.read_csv(os.path.join(HERE, "leakage.csv"), dtype={"run": str})
    out = {}
    for label, runs in [("all4", list(RUNS.values())), ("excl0218", ["0204", "0212", "0224"])]:
        dec = pd.DataFrame([decide(D, c, n, runs) for c in ["tfinit", "cnn_oracle", "cnn01", "cnn02", "pca", "raw"]
                            for n in PRIMARY + SECONDARY])
        dec.to_csv(os.path.join(HERE, f"decisions_{label}.csv"), index=False)
        P = dec[dec.score.isin(PRIMARY)]
        def n(c, o): return int(((P.comp == c) & (P.outcome == o)).sum())
        H1 = n("tfinit", "TF better") >= 2 and n("tfinit", "competitor better") == 0
        H2 = n("cnn_oracle", "TF better") >= 2 and n("cnn_oracle", "competitor better") == 0
        H3 = n("pca", "TF better") >= 2 and n("raw", "TF better") >= 2
        Ls = L[L.run.isin(runs)]
        gap = (Ls[Ls.rep == "tf"].set_index(["run", "seed"]).is0204_lr - Ls[Ls.rep == "tfinit"].set_index(["run", "seed"]).is0204_lr).mean()
        date_ok = all(Ls[(Ls.rep == "tf") & (Ls.run == r)].date_lr.mean() <= thr[r] for r in runs)
        H4 = gap <= 0.05 and date_ok
        v = ("Supported on this data" if H1 and H2 and H3 and H4 else
             "Partly supported" if H1 and H3 else "Not supported")
        out[label] = dict(H1=H1, H2=H2, H3=H3, H4=H4, is0204_gap=float(gap), date_ok=date_ok, verdict=v)
        print(label, out[label]); print(dec.round(3).to_string(index=False))
    json.dump(out, open(os.path.join(HERE, "verdict.json"), "w"), indent=1)


# ---------------------------------------------------------------- examples (run 0212, fold-1 models, seed 0)
def examples():
    from sklearn.manifold import TSNE
    m, W, blk, tg = load()
    crops = E.eval_crops(W, m.fs_eff.to_numpy())
    k = 1; sel = np.flatnonzero(m.run_id.to_numpy() == k)
    D = pd.read_csv(os.path.join(HERE, "scores_long.csv"), dtype={"run": str})
    cm = D[(D.run == "0212") & D.rep.isin(["cnn01", "cnn02"]) & D.score.isin(PRIMARY)].groupby("rep").value.mean()
    cnn_rep = cm.idxmax(); cnn_t = 0.1 if cnn_rep == "cnn01" else 0.2
    reps = {"Transformer (SimCLR)": embed_all(crops, ckpt(k, "tf", 0.1, 0, "final"))[sel],
            f"CNN (SimCLR, T {cnn_t})": embed_all(crops, ckpt(k, "cnn", cnn_t, 0, "final"))[sel],
            "Transformer, untrained": embed_all(crops, ckpt(k, "tf", 0.1, 0, "init"))[sel]}
    g4 = tg["group4"][sel]; names = {0: "small, down-first", 1: "small, up-first", 2: "large, down-first", 3: "large, up-first"}
    rng = np.random.default_rng(7)
    queries = np.concatenate([rng.choice(np.flatnonzero(g4 == c), 2, replace=False) for c in range(4)])
    wav = crops.float().cpu().numpy()[sel][:, ::6]       # 600 -> 100 points
    gal = []
    purity = {}
    for rname, X in reps.items():
        Z = StandardScaler().fit_transform(X)
        nn = NearestNeighbors(n_neighbors=11).fit(Z)
        _, I = nn.kneighbors(Z)
        purity[rname] = float(np.mean(g4[I[:, 1:]] == g4[:, None]))
        for q in queries:
            for rank, j in enumerate(I[q, :6]):
                gal.append(dict(rep=rname, query=int(q), rank=rank, group=names[int(g4[j])],
                                same_group=bool(g4[j] == g4[q]), wave=[round(float(v), 3) for v in wav[j]]))
    json.dump(dict(cnn_temp=cnn_t, purity=purity, queries=[int(q) for q in queries], gallery=gal),
              open(os.path.join(HERE, "examples.json"), "w"))
    rows = []
    for rname, X in reps.items():
        Y = TSNE(2, perplexity=30, random_state=0, init="pca").fit_transform(StandardScaler().fit_transform(X))
        for (a, b), c in zip(Y, g4):
            rows.append(dict(rep=rname, x=round(float(a), 2), y=round(float(b), 2), group=names[int(c)]))
    pd.DataFrame(rows).to_csv(os.path.join(HERE, "tsne_0212.csv"), index=False)
    print("purity (share of 10 nearest neighbours in the same 4-group):", {a: round(b, 3) for a, b in purity.items()})


if __name__ == "__main__":
    {"dryrun": dryrun, "score": score, "examples": examples, "verdict": lambda: verdict(json.load(open(os.path.join(HERE, "h4_date_thresholds.json"))))}[sys.argv[1]]()
