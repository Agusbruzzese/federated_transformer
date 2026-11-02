"""G (independent recompute), stage 1: embed client + eval events for each checkpoint, with own code.
Run from repo root:  .venv/bin/python research/model/K_committee/G_recompute/g_embed.py
Writes embeddings to research/model/K_committee/G_recompute/emb/<tag>.npz (never overwrites).
Also records sha256(final.pt) vs summary.json provenance.final and fold checks.
"""
import os, sys, json, hashlib
import numpy as np, torch

sys.path.insert(0, os.getcwd()); sys.path.insert(0, "research/model/M4_methods")
import m4
from ssl_central import models as Mo, augment as A

OUT = "research/model/K_committee/G_recompute"
RUNS = "data/k_committee/protocol/runs3"
ARMS = ["C256", "F-nat", "S+F", "F-X+P", "S+C"]
folds = json.load(open("research/model/K_committee/protocol/folds.json"))
m, W = m4.load()
day = m.date.astype(str).to_numpy()
os.makedirs(f"{OUT}/emb", exist_ok=True)


def idx_for(f):
    cd = sorted({d for c in folds["folds"][str(f)]["clients"] for d in c["days"]})
    ed = folds["folds"][str(f)]["eval_days"]
    return np.flatnonzero(np.isin(day, cd)), np.flatnonzero(np.isin(day, ed)), cd, ed


@torch.no_grad()
def embed(path, ii):
    enc = Mo.make("tf"); enc.load_state_dict(torch.load(path, map_location="cpu")["enc"]); enc = enc.cuda().eval()
    out = []
    fs = m.fs_eff.to_numpy(np.float32)
    for k in range(0, len(ii), 1024):
        j = ii[k:k + 1024]
        x = A.eval_view(torch.tensor(W[j], device="cuda"), torch.tensor(fs[j], device="cuda"))
        out.append(enc(x).double().cpu().numpy())
    return np.concatenate(out)


checks = []
for arm in ARMS:
    for f in range(3):
        ci, ei, cd, ed = idx_for(f)
        for s in range(10):
            tag = f"v21_{arm}_f{f}_s{s}"; d = f"{RUNS}/{tag}"
            sm = json.load(open(f"{d}/summary.json"))
            sha = hashlib.sha256(open(f"{d}/final.pt", "rb").read()).hexdigest()
            fold_ok = str(sm["args"]["fold"]) == str(f)
            trained_days = sorted({x for c in sm["plan"].get("clients_days", []) for x in c})
            checks.append(dict(tag=tag, fold_summary=sm["args"]["fold"], fold_ok=fold_ok,
                               sha_ok=sha == sm["provenance"]["final"], sha=sha,
                               trained_days_eq_fold_clients=trained_days == cd,
                               trained_days_overlap_eval=sorted(set(trained_days) & set(ed)),
                               init=sm["provenance"].get("init"), arm_summary=sm["args"]["arm"], kind=sm["args"]["kind"]))
            fn = f"{OUT}/emb/{tag}.npz"
            if os.path.exists(fn):
                continue
            np.savez(fn, client=embed(f"{d}/final.pt", ci), eval=embed(f"{d}/final.pt", ei), ci=ci, ei=ei)
            print(tag, fold_ok, checks[-1]["sha_ok"], flush=True)

fn = f"{OUT}/integrity_checks.json"
assert not os.path.exists(fn)
json.dump(checks, open(fn, "w"), indent=1)
print("fold_ok all:", all(c["fold_ok"] for c in checks), " sha_ok all:", all(c["sha_ok"] for c in checks),
      " trained==fold clients all:", all(c["trained_days_eq_fold_clients"] for c in checks),
      " any overlap:", any(c["trained_days_overlap_eval"] for c in checks))
