"""A7: sensitivity to training length. No intermediate M5 checkpoints exist (only init.pt / final.pt), so we use
(i) the M5 per-epoch loss curves (log.json, 1000 'epochs' of 10 steps) and (ii) the M4 SimCLR-TF checkpoints
(same architecture, same data split, same views, T=0.1, ~1000 steps, cosine without warm-up; data/m4/ckpt/simclr_tf_c0_s*)
as a '1k-step' point vs M5 '10k-step'.
Outputs: a7_loss_curves.csv, a7_steps.csv (probe scores, RankMe), a7_agreement.csv
"""
from common import *   # noqa
import json, itertools, pandas as pd

# (i) loss curves
rows = []
for meth in ("simclr_tf", "simclr_cnn"):
    for s in SEEDS:
        c = np.array(json.load(open(os.path.join(ROOT, f"data/m5/ckpt/{meth}_s{s}/log.json")))["curve"])  # mean loss per 10-step block
        r = dict(rep=meth, seed=s)
        for st in (100, 500, 1000, 2000, 5000, 7500, 10000): r[f"loss@{st}"] = c[st // 10 - 10: st // 10].mean()   # mean of last 100 steps
        r["drop_last_half"] = r["loss@5000"] - r["loss@10000"]
        rows.append(r)
LC = pd.DataFrame(rows); LC.to_csv(os.path.join(HERE, "a7_loss_curves.csv"), index=False)
pd.set_option("display.width", 250); print(LC.round(3).to_string())
m4c = [json.load(open(os.path.join(ROOT, f"data/m4/ckpt/simclr_tf_c0_s{s}/log.json"))) for s in SEEDS]
print("M4 simclr_tf c0 final-epoch loss", [round(d["loss"][-1], 3) for d in m4c], "epochs", m4c[0]["epochs"], "n_train", m4c[0]["n_train"])

# (ii) M4 1k-step vs M5 10k-step
m, W = m4.load(); tg = m4.targets(m); te = (m.role == "test").to_numpy()
E = {}
for s in SEEDS: E[f"m4_1k_s{s}"] = m4_tf_emb(s); E[f"m5_10k_s{s}"] = emb("simclr_tf", s, "final")
rows = []
for k, X in E.items():
    r = dict(rep=k.rsplit("_s", 1)[0], seed=int(k.rsplit("_s", 1)[1]), rankme_test=rankme(X[te]))
    for pk in ("knn", "lin"):
        for nm in m5.PRIMARY:
            p, _, _ = m5.probe_all(X, m, tg, pk, nm); r[f"{pk}_{nm}"] = m4.metric(nm, tg[nm][te], p)
    rows.append(r); print(r, flush=True)
D = pd.DataFrame(rows); D.to_csv(os.path.join(HERE, "a7_steps.csv"), index=False)
print(D.groupby("rep").agg(["mean", "std"]).drop(columns="seed").round(4).T.to_string())

N = {k: knn_sets(v[te]) for k, v in E.items()}; ag = []
for i, j in itertools.combinations(SEEDS, 2):
    for tag in ("m4_1k", "m5_10k"):
        a, b = f"{tag}_s{i}", f"{tag}_s{j}"; ag.append(dict(group=f"{tag}_pairs", a=a, b=b, cka=linear_cka(E[a][te], E[b][te]), knn15=knn_overlap(N[a], N[b])))
for i in SEEDS:
    a, b = f"m4_1k_s{i}", f"m5_10k_s{i}"; ag.append(dict(group="m4_1k_vs_m5_10k_same_seed", a=a, b=b, cka=linear_cka(E[a][te], E[b][te]), knn15=knn_overlap(N[a], N[b])))
AG = pd.DataFrame(ag); AG.to_csv(os.path.join(HERE, "a7_agreement.csv"), index=False)
print(AG.groupby("group")[["cka", "knn15"]].agg(["mean", "min", "max"]).round(4).to_string())
