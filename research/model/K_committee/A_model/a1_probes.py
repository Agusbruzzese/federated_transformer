"""A1 + A4: reproduce M5 test probe scores (m5.probe_all: kNN k=15 / linear, scaling chosen on the validation day)
and break them down per seed and per test day, for SimCLR-TF trained/init and SimCLR-CNN trained/init.

Outputs: a1_probe_scores.csv (long), a1_sanity.csv (vs M5 scores_abs.csv), a4_variability.csv (SD across seeds/days).
"""
from common import *   # noqa
import pandas as pd

m, W = m4.load(); tg = m4.targets(m); te = (m.role == "test").to_numpy()
day = m.date.astype(str).to_numpy()[te]; DAYS = sorted(np.unique(day))
rows = []
for meth in ("simclr_tf", "simclr_cnn"):
    for which in ("final", "init"):
        for s in SEEDS:
            X = emb(meth, s, which)
            for k in ("knn", "lin"):
                for nm in m5.PRIMARY:
                    p, scaled, _ = m5.probe_all(X, m, tg, k, nm); y = tg[nm][te]
                    rows.append(dict(rep=meth, which=which, seed=s, probe=k, score=nm, day="ALL", scaled=scaled, value=m4.metric(nm, y, p)))
                    for d in DAYS:
                        i = day == d
                        rows.append(dict(rep=meth, which=which, seed=s, probe=k, score=nm, day=d, scaled=scaled, value=m4.metric(nm, y[i], p[i])))
        print(meth, which, "done", flush=True)
R = pd.DataFrame(rows); R.to_csv(os.path.join(HERE, "a1_probe_scores.csv"), index=False)

# ---- sanity vs M5
ref = pd.read_csv(os.path.join(ROOT, "research/model/M5_long/scores_abs.csv"))
mine = R[(R.which == "final") & (R.day == "ALL")].rename(columns={"value": "mine"})
S = mine.merge(ref.rename(columns={"value": "m5"}), on=["rep", "seed", "probe", "score"])
S["absdiff"] = (S.mine - S.m5).abs(); S.to_csv(os.path.join(HERE, "a1_sanity.csv"), index=False)
print("sanity rows", len(S), "max |diff|", S.absdiff.max())

# ---- variability
out = []
for (rep, which, k, nm), g in R.groupby(["rep", "which", "probe", "score"]):
    a = g[g.day == "ALL"].sort_values("seed").value.to_numpy()
    pd_ = g[g.day != "ALL"].pivot(index="seed", columns="day", values="value")          # seeds x days
    out.append(dict(rep=rep, which=which, probe=k, score=nm, mean=a.mean(), sd_seeds=a.std(ddof=1), min=a.min(), max=a.max(),
                    sd_days_of_seedmean=pd_.mean(0).std(ddof=1),                          # day-to-day SD of the 5-seed mean
                    mean_within_seed_sd_days=pd_.std(1, ddof=1).mean(),
                    sd_seeds_within_day=pd_.std(0, ddof=1).mean(),
                    **{f"day_{d}": pd_[d].mean() for d in DAYS}))
V = pd.DataFrame(out)
# paired trained - init per seed (same seed's init)
dl = []
for (rep, k, nm), g in R[R.day == "ALL"].groupby(["rep", "probe", "score"]):
    f = g[g.which == "final"].set_index("seed").value; i = g[g.which == "init"].set_index("seed").value; d = f - i
    dl.append(dict(rep=rep, which="final-init", probe=k, score=nm, mean=d.mean(), sd_seeds=d.std(ddof=1), min=d.min(), max=d.max()))
V = pd.concat([V, pd.DataFrame(dl)], ignore_index=True)
V.to_csv(os.path.join(HERE, "a4_variability.csv"), index=False)
pd.set_option("display.width", 250); print(V.round(4).to_string())
