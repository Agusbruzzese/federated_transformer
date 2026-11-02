"""Post-hoc exploratory (after reviewer): shape families = Louvain (seed 0) on the shape-only graph corr>=0.9, families >=20 events.
Writes families.csv, families.json and gallery_families.png.   python research/model/C2_lookalikes/families.py"""
import os, sys, json, numpy as np, pandas as pd, networkx as nx
ROOT = os.getcwd(); sys.path[:0] = [ROOT, f"{ROOT}/research/model/C1_clustering"]; import c1
HERE = os.path.dirname(os.path.abspath(__file__))
m, W = c1.m4.load(); x = c1.raw_crops(); P = np.load(os.path.join(ROOT, "data", "c1", "aligned_corr.npy")); n = len(P)
A = P >= 0.9; np.fill_diagonal(A, False)
com = sorted(nx.community.louvain_communities(nx.from_numpy_array(A.astype(np.uint8)), seed=0), key=len, reverse=True)
fam = -np.ones(n, int); k = 0
for c in com:
    if len(c) >= 20: fam[list(c)] = k; k += 1
amp = 10 ** m.m8_log_amp.to_numpy(); dur = m.m8_dur10.to_numpy(); C2 = pd.read_csv(os.path.join(HERE, "c2_labels.csv"))
rows = []
for f in range(k):
    i = fam == f; sub = P[np.ix_(np.flatnonzero(i), np.flatnonzero(i))]
    rows.append(dict(family=f, size=int(i.sum()), amp_median=float(np.median(amp[i])), amp_p90_p10=float(np.quantile(amp[i], .9) / np.quantile(amp[i], .1)),
                     dur_median=float(np.median(dur[i])), median_corr=float(np.median(sub[np.triu_indices(len(sub), 1)])),
                     polarity_pos=float((m.m8_polarity.to_numpy()[i] > 0).mean()), c2a_groups_inside=int(len(np.unique(C2.C2a.values[i & (C2.C2a.values >= 0)])))))
pd.DataFrame({"event_id": m.event_id, "family": fam}).to_csv(os.path.join(HERE, "families.csv"), index=False)
json.dump(dict(n_families=k, coverage=float((fam >= 0).mean()), families=rows), open(os.path.join(HERE, "families.json"), "w"), indent=1)
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
t = np.arange(600) / 50; fig, axs = plt.subplots(2, 4, figsize=(16, 7)); rng = np.random.default_rng(0)
for ax, r in zip(axs.flat, rows):
    i = np.flatnonzero(fam == r["family"]); xn = x[i] / np.abs(x[i]).max(1, keepdims=True)
    for j in rng.choice(len(i), min(40, len(i)), replace=False): ax.plot(t, xn[j], lw=0.5, alpha=0.4, color="0.4")
    ax.plot(t, np.median(xn, 0), color="C3", lw=1.6)
    ax.set_title(f"family {r['family']}: n={r['size']}, peak {r['amp_median']:.0f} (p90/p10 {r['amp_p90_p10']:.1f}x)", fontsize=9)
for ax in axs.flat[len(rows):]: ax.axis("off")
fig.suptitle("Shape families (post-hoc, exploratory): members normalised to their peak; red = median shape"); fig.tight_layout()
fig.savefig(os.path.join(HERE, "gallery_families.png"), dpi=90)
print(json.dumps(dict(n_families=k, coverage=float((fam >= 0).mean())), indent=1)); print(pd.DataFrame(rows).round(2).to_string(index=False))
