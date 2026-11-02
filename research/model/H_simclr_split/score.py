"""Scoring for H (see PLAN.md). Usage:
    python research/model/H_simclr_split/score.py dryrun
    python research/model/H_simclr_split/score.py real
"""
import sys, json, glob, os
import numpy as np, pandas as pd, torch
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import balanced_accuracy_score, r2_score

sys.path.insert(0, os.getcwd())
from ssl_central import data as D, embed as E, config as C

HERE = os.path.dirname(os.path.abspath(__file__))
TEST_RUN = C.HOLDOUT_RUN_ID   # run 0218
SEEDS = range(5)
CELLS = {"cnn": (3e-4, 0.2), "tf": (3e-4, 0.1)}
rng = np.random.default_rng(0)


def skill(y, p):
    K = len(np.unique(y)); return (balanced_accuracy_score(y, p) - 1 / K) / (1 - 1 / K)


def load():
    m, W, EM, RM = D.load_all()
    m = m.reset_index(drop=True)
    tier = (m.snr >= C.TIER).to_numpy()
    blk = pd.to_datetime(m.wall_time).dt.floor("30min").astype("int64").to_numpy()
    amp = 10 ** m.m8_log_amp.to_numpy()
    big = (amp >= 5000).astype(int)
    pol = (m.m8_polarity.to_numpy() > 0).astype(int)
    tg = dict(polarity=pol, lobes=np.minimum(m.m8_n_lobes.to_numpy(), 4).astype(int),
              duration=np.log10(m.m8_dur10.clip(lower=0.05).to_numpy()), magnitude=big, group4=big * 2 + pol)
    return m, W, tier, blk, tg


def test_scores(X, m, tier, blk, tg):
    sel = tier & (m.run_id.to_numpy() == TEST_RUN)
    Xs, g = X[sel], blk[sel]
    out = {}
    for name, y in tg.items():
        ys = y[sel]
        if name == "duration":
            p = cross_val_predict(make_pipeline(StandardScaler(), KNeighborsRegressor(15)), Xs, ys, groups=g, cv=GroupKFold(5))
            out[name] = r2_score(ys, p)
        else:
            p = cross_val_predict(make_pipeline(StandardScaler(), KNeighborsClassifier(15)), Xs, ys, groups=g, cv=GroupKFold(5))
            out[name] = skill(ys, p)
    # L2 date leakage inside the test run
    yd = m.date.to_numpy()[sel]
    p = cross_val_predict(make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)), Xs, yd, groups=g, cv=GroupKFold(5))
    out["L2_date"] = skill(yd, p)
    # L1 is-0204 across all runs, one space; report the worse of LR and kNN
    y = (m.run_id.to_numpy()[tier] == 0).astype(int); Xa, ga = X[tier], blk[tier]
    vals = []
    for clf in (LogisticRegression(max_iter=2000), KNeighborsClassifier(15)):
        p = cross_val_predict(make_pipeline(StandardScaler(), clf), Xa, y, groups=ga, cv=GroupKFold(5))
        vals.append(skill(y, p))
    out["L1_is0204"] = max(vals)
    return out


def boot_ci(d, blocks, n=1000):
    """Block bootstrap over test-run blocks is approximated by bootstrapping seeds x jackknife-free mean;
    with per-seed gains we report mean and a seed-level t interval plus min/max."""
    d = np.asarray(d); se = d.std(ddof=1) / np.sqrt(len(d))
    return d.mean(), d.mean() - 2.776 * se, d.mean() + 2.776 * se   # t(0.975, df=4)


def dryrun():
    m, W, tier, blk, tg = load()
    morph = m[[c for c in m.columns if c.startswith("m8_")]].fillna(0).to_numpy(float)
    noise = rng.standard_normal((len(m), 64))
    leak = np.c_[noise, np.eye(4)[m.run_id.to_numpy()] * 5]
    res = {"planted_signal_morph8": test_scores(morph, m, tier, blk, tg),
           "null_noise": test_scores(noise, m, tier, blk, tg),
           "planted_leak_runonehot": test_scores(leak, m, tier, blk, tg)}
    json.dump(res, open(os.path.join(HERE, "dryrun.json"), "w"), indent=1)
    for k, v in res.items():
        print(k, {a: round(b, 3) for a, b in v.items()})


def real():
    m, W, tier, blk, tg = load()
    crops = E.eval_crops(W, m.fs_eff.to_numpy())
    rows = []
    for arch, (lr, t) in CELLS.items():
        for s in SEEDS:
            tag = f"h_{arch}_s{s}"
            r = {"arch": arch, "seed": s}
            for which in ("final", "init"):
                enc = E.load_encoder(E.ckpt_path(tag, which))
                X = E.embed_views(enc, crops)
                r.update({f"{which}_{k}": v for k, v in test_scores(X, m, tier, blk, tg).items()})
            rows.append(r); print(r, flush=True)
    df = pd.DataFrame(rows); df.to_csv(os.path.join(HERE, "scores_per_seed.csv"), index=False)
    summ = []
    for arch, g in df.groupby("arch"):
        for k in ["polarity", "lobes", "duration", "magnitude", "group4", "L1_is0204", "L2_date"]:
            d = (g[f"final_{k}"] - g[f"init_{k}"]).to_numpy()
            mu, lo, hi = boot_ci(d, None)
            summ.append(dict(arch=arch, score=k, trained=g[f"final_{k}"].mean(), untrained=g[f"init_{k}"].mean(),
                             gain=mu, lo=lo, hi=hi, seeds_positive=int((d > 0).sum())))
    S = pd.DataFrame(summ); S.to_csv(os.path.join(HERE, "scores_summary.csv"), index=False)
    print(S.round(3).to_string(index=False))


if __name__ == "__main__":
    {"dryrun": dryrun, "real": real}[sys.argv[1]]()
