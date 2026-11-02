"""A3: seed-to-seed representational agreement on test-day events (noise floor for central-vs-federated similarity).

Linear CKA (centred, Kornblith 2019) on raw 64-d encoder outputs; 15-NN overlap = mean over events of
|N15_a(i) & N15_b(i)| / 15, neighbours by Euclidean distance on per-feature z-scored embeddings, self excluded.
Chance level for the overlap = 15/(n-1).
Groups: TF trained pairs (10), TF init pairs (10), TF trained vs own init (5), TF trained vs other seed's init (20),
CNN trained pairs (10), TF trained vs CNN trained (25), and references: trained TF vs raw-crop PCA-64
(PCA fit on training-day crops, as M5 B4) and vs z-scored morph8 features.
Output: a3_pairs.csv (one row per pair), a3_summary.csv.
"""
from common import *   # noqa
import itertools, pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

m, W = m4.load(); te = (m.role == "test").to_numpy(); tr = (m.role == "train").to_numpy()
E = {}
for meth, tag in (("simclr_tf", "tf"), ("simclr_cnn", "cnn")):
    for which in ("final", "init"):
        for s in SEEDS: E[f"{tag}_{which}_s{s}"] = emb(meth, s, which)[te]
raw = m5.crops().cpu().numpy(); E["raw_pca64"] = PCA(64, random_state=0).fit(raw[tr]).transform(raw)[te]
morph = m[[c for c in m.columns if c.startswith("m8_")]].fillna(0).to_numpy(float); E["morph8"] = StandardScaler().fit(morph[tr]).transform(morph)[te]
N = {k: knn_sets(v) for k, v in E.items()}
n = te.sum(); chance = 15 / (n - 1)

pairs = []
def add(group, a, b): pairs.append(dict(group=group, a=a, b=b, cka=linear_cka(E[a], E[b]), knn15=knn_overlap(N[a], N[b])))
for i, j in itertools.combinations(SEEDS, 2):
    add("tf_final_vs_tf_final", f"tf_final_s{i}", f"tf_final_s{j}")
    add("tf_init_vs_tf_init", f"tf_init_s{i}", f"tf_init_s{j}")
    add("cnn_final_vs_cnn_final", f"cnn_final_s{i}", f"cnn_final_s{j}")
for i in SEEDS:
    add("tf_final_vs_own_init", f"tf_final_s{i}", f"tf_init_s{i}")
    add("cnn_final_vs_own_init", f"cnn_final_s{i}", f"cnn_init_s{i}")
    add("tf_final_vs_raw_pca64", f"tf_final_s{i}", "raw_pca64")
    add("tf_final_vs_morph8", f"tf_final_s{i}", "morph8")
    add("tf_init_vs_raw_pca64", f"tf_init_s{i}", "raw_pca64")
    for j in SEEDS:
        if i != j: add("tf_final_vs_other_init", f"tf_final_s{i}", f"tf_init_s{j}")
        add("tf_final_vs_cnn_final", f"tf_final_s{i}", f"cnn_final_s{j}")
P = pd.DataFrame(pairs); P.to_csv(os.path.join(HERE, "a3_pairs.csv"), index=False)
S = P.groupby("group", sort=False).agg(n=("cka", "size"), cka_mean=("cka", "mean"), cka_min=("cka", "min"), cka_max=("cka", "max"), cka_sd=("cka", "std"),
                                       knn_mean=("knn15", "mean"), knn_min=("knn15", "min"), knn_max=("knn15", "max"), knn_sd=("knn15", "std")).reset_index()
S["knn_chance"] = chance; S.to_csv(os.path.join(HERE, "a3_summary.csv"), index=False)
pd.set_option("display.width", 250); print("n test", n, "chance overlap", round(chance, 4)); print(S.round(4).to_string())
print(P[P.group == "tf_final_vs_tf_final"].round(4).to_string())
