"""Fold and client construction for PROTOCOL v1 (report 08). Deterministic; no randomness except the IID permutation.
Rules (fixed before any training):
  server days   : Feb 18, Feb 20 (run 0218)                         -> server pretraining only
  tuning day    : Feb 10                                            -> equal tuning budget only; never a client, never evaluated
  eval days     : the large (>= 200 events) remaining days except Feb 19 (same run as the server),
                  assigned to 3 folds round-robin within each run, in date order
  clients       : for fold f, all remaining days that are not server, tuning or fold-f eval days.
                  While there are more than 10 clients, merge the SMALLEST client into its chronologically adjacent
                  client of the same run (the smaller of the two neighbours; ties -> earlier). If none exists, the next smallest.
"""
import json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.getcwd()); sys.path.insert(0, "research/model/M4_methods")
import m4
m, _ = m4.load(); day = m.date.astype(str)
cnt = day.value_counts().sort_index(); run = m.groupby(day).run_id.first()
SERVER, TUNE, N_CLIENTS = ["2025-02-18", "2025-02-20"], "2025-02-10", 10
rest = [d for d in cnt.index if d not in SERVER + [TUNE]]
large = [d for d in rest if cnt[d] >= 200 and d != "2025-02-19"]
folds = {0: [], 1: [], 2: []}
# round-robin within run, continuing the counter across runs so folds stay balanced
k = 0
for r in sorted(run[large].unique()):
    for d in [d for d in large if run[d] == r]:
        folds[k % 3].append(d); k += 1
out = dict(server=SERVER, server_events=int(cnt[SERVER].sum()), tuning_day=TUNE, tuning_events=int(cnt[TUNE]), folds={})
for f, ev in folds.items():
    days = [d for d in rest if d not in ev]
    clients = [[d] for d in days]
    while len(clients) > N_CLIENTS:
        size = [sum(cnt[d] for d in c) for c in clients]
        for i in np.argsort(size, kind="stable"):
            c = clients[i]; rr = run[c[0]]
            nb = [j for j in (i - 1, i + 1) if 0 <= j < len(clients) and run[clients[j][0]] == rr]
            if nb:
                j = min(nb, key=lambda j: (size[j], j)); a, b = sorted((i, j))
                clients[a] = clients[a] + clients[b]; del clients[b]; break
    out["folds"][f] = dict(eval_days=ev, eval_events=int(cnt[ev].sum()),
                           clients=[dict(days=c, n=int(cnt[c].sum()), run=int(run[c[0]])) for c in clients],
                           train_events=int(sum(cnt[d] for c in clients for d in c)))
json.dump(out, open("research/model/K_committee/protocol/folds.json", "w"), indent=1)
print(cnt.to_string()); print(json.dumps(out, indent=1))
