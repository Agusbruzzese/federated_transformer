"""Reviewer E: data-design checks of protocol v1 (report 08). Read-only on everything outside E_review.
    .venv/bin/python research/model/K_committee/E_review/check_design.py
"""
import json, os, sys
import numpy as np, pandas as pd
ROOT = os.getcwd(); sys.path.insert(0, ROOT); sys.path.insert(0, "research/model/M4_methods")
import m4
HERE = os.path.dirname(os.path.abspath(__file__))
m, _ = m4.load(); day = m.date.astype(str)
cnt = day.value_counts().sort_index(); run = m.groupby(day).run_id.first(); role = m.groupby(day).role.first()
FJ = json.load(open("research/model/K_committee/protocol/folds.json"))
out = {}
print(pd.DataFrame(dict(n=cnt, run=run, m4_role=role)).to_string())
# 1. recompute folds with folds.py's exact rule (copied, writes nothing)
SERVER, TUNE, N_CLIENTS = ["2025-02-18", "2025-02-20"], "2025-02-10", 10
rest = [d for d in cnt.index if d not in SERVER + [TUNE]]
large = [d for d in rest if cnt[d] >= 200 and d != "2025-02-19"]
folds = {0: [], 1: [], 2: []}; k = 0
for r in sorted(run[large].unique()):
    for d in [d for d in large if run[d] == r]:
        folds[k % 3].append(d); k += 1
same = True
for f, ev in folds.items():
    days = [d for d in rest if d not in ev]; clients = [[d] for d in days]
    while len(clients) > N_CLIENTS:
        size = [sum(cnt[d] for d in c) for c in clients]
        for i in np.argsort(size, kind="stable"):
            c = clients[i]; rr = run[c[0]]
            nb = [j for j in (i - 1, i + 1) if 0 <= j < len(clients) and run[clients[j][0]] == rr]
            if nb:
                j = min(nb, key=lambda j: (size[j], j)); a, b = sorted((i, j))
                clients[a] = clients[a] + clients[b]; del clients[b]; break
    fj = FJ["folds"][str(f)]
    same &= (fj["eval_days"] == ev) and ([c["days"] for c in fj["clients"]] == clients)
out["folds_json_reproduced"] = bool(same)
# 2. disjointness and cross-fold structure
pool_days = set(pd.read_parquet("data/k_committee/C_federated/bgpool_day_meta.parquet").day.unique())
rows = []
for f in range(3):
    fj = FJ["folds"][str(f)]; cd = [d for c in fj["clients"] for d in c["days"]]; ev = fj["eval_days"]
    assert not set(cd) & set(ev) and not set(cd) & set(SERVER) and TUNE not in cd and TUNE not in ev and not set(ev) & set(SERVER)
    other_eval = [d for g in range(3) if g != f for d in FJ["folds"][str(g)]["eval_days"]]
    N = fj["train_events"]; n = np.array([c["n"] for c in fj["clients"]]); X, R = 2_560_000, 200
    bk = np.minimum(64, n); sk = np.maximum(1, np.round(X / R * n / N / bk)).astype(int)
    rows.append(dict(fold=f, eval_days=" ".join(d[5:] for d in ev), n_train=N,
                     clients_that_are_other_folds_eval=" ".join(sorted(d[5:] for d in cd if d in other_eval)),
                     client_days_m4_role_test=" ".join(sorted(d[5:] for d in cd if role[d] == "test")),
                     client_days_without_day_noise_pool=" ".join(sorted(d[5:] for d in cd if d not in pool_days)),
                     epochs_per_event_central=round(X / N, 1),
                     fed_b_k="/".join(map(str, bk)), fed_s_k="/".join(map(str, sk)),
                     fed_examples_over_X=round(R * (sk * bk).sum() / X, 4), fed_grad_steps=int(R * sk.sum()),
                     fed_epochs_min_max=f"{(R*sk*bk/n).min():.0f}-{(R*sk*bk/n).max():.0f}"))
D = pd.DataFrame(rows); print(D.T.to_string())
out["server_epochs_per_event"] = round(512000 / FJ["server_events"], 1)
# 3. tuning runs (fold-0 clients) vs other folds' eval days
t_cl = {d for c in FJ["folds"]["0"]["clients"] for d in c["days"]}
out["tuning_clients_overlap_eval_days"] = {g: sorted(set(FJ["folds"][str(g)]["eval_days"]) & t_cl) for g in (1, 2)}
# 4. generator / families training days (13 M4/M5 training days, D's B1) vs eval days
gen_days = sorted(d for d in cnt.index if role[d] == "train")
out["generator_fit_days_that_are_eval_days"] = {f: sorted(set(FJ["folds"][str(f)]["eval_days"]) & set(gen_days)) for f in range(3)}
out["generator_fit_days_that_are_server_days"] = sorted(set(SERVER) & set(gen_days))
out["eval_days_seen_by_M5_central_reference"] = sorted(d for f in range(3) for d in FJ["folds"][str(f)]["eval_days"] if role[d] == "train")
D.to_csv(os.path.join(HERE, "check_design.csv"), index=False)
json.dump(out, open(os.path.join(HERE, "check_design.json"), "w"), indent=1); print(json.dumps(out, indent=1))
