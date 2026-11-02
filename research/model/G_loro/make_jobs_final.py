"""After selection.json is frozen: seeds 1-4 of each selected cell + the CNN no-stretch ablation (seeds 0-4)."""
import json, os
from ssl_central_v3 import config as C
S = json.load(open(os.path.join(C.OUT, "selection.json")))
jobs = []
for k in C.RUNS:
    for a in C.ARCHS:
        s = S[f"{k}_{a}"]
        jobs += [dict(k=k, arch=a, lr=s["lr"], temp=s["temp"], seed=sd) for sd in C.SEEDS if sd > 0]
    s = S[f"{k}_cnn"]
    jobs += [dict(k=k, arch="cnn", lr=s["lr"], temp=s["temp"], seed=sd, nostretch=True) for sd in C.SEEDS]
json.dump(jobs, open(os.path.join(C.RES, "logs", "jobs_final.json"), "w"), indent=0)
print(len(jobs), "jobs")
