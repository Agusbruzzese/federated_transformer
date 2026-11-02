"""SimCLR training of one (fold, arch, lr, T, seed[, nostretch]) cell on fold k's training runs (snr>=20).

Same loop as ssl_central.train (F), with the fold split (J1), per-fold pool (J2) and an optional no-stretch view.
    python -m ssl_central_v3.train --jobs jobs.json --shard 0 --nshard 3
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time

import numpy as np
import torch

from ssl_central import augment as A
from ssl_central import models as Mo
from . import config as C
from . import data as D

PARTS_FULL = ("crop", "stretch", "amp", "noise", "offset")
PARTS_NOSTRETCH = ("crop", "amp", "noise", "offset")


def tag(k, arch, lr, temp, seed, nostretch=False):
    return f"f{k}_{arch}_lr{lr:g}_t{temp:g}_s{seed}" + ("_nostretch" if nostretch else "")


def ckpt(t, which):
    return os.path.join(C.OUT, "ckpt", t, f"{which}.pt")


def train(k, arch, lr, temp, seed, nostretch=False, dev="cuda"):
    t_ = tag(k, arch, lr, temp, seed, nostretch)
    od = os.path.join(C.OUT, "ckpt", t_)
    if os.path.exists(os.path.join(od, "final.pt")):
        return None
    os.makedirs(od, exist_ok=True)
    m, W = D.load_all()
    idx = D.train_index(m, k)
    D.firewall(m, k, idx)
    bg, _ = D.fold_pool(k)
    s = m.iloc[idx]
    T = dict(W=torch.tensor(W[idx], device=dev), fs=torch.tensor(s.fs_eff.to_numpy(np.float32), device=dev),
             ev0=torch.tensor(s.ev_i0.to_numpy(), device=dev), ev1=torch.tensor(s.ev_i1.to_numpy(), device=dev),
             sig=torch.tensor(s.sigma_ev.to_numpy(np.float32), device=dev),
             t=torch.tensor(s.t_s.to_numpy(np.float64), device=dev), bg=torch.tensor(bg, device=dev))
    ids = s.event_id.to_numpy(); N = len(ids)
    parts = PARTS_NOSTRETCH if nostretch else PARTS_FULL
    torch.manual_seed(seed); np.random.seed(seed)
    enc = Mo.make(arch).to(dev); head = Mo.Head().to(dev)
    torch.save(dict(arch=arch, enc=enc.state_dict()), os.path.join(od, "init.pt"))
    opt = torch.optim.AdamW(list(enc.parameters()) + list(head.parameters()), lr=lr, weight_decay=1e-4)
    bs = min(C.BATCH, N); spe = N // bs; total = spe * C.EPOCHS
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: 0.5 * (1 + math.cos(math.pi * min(i, total) / total)))
    g = torch.Generator(device=dev); g.manual_seed(seed)
    gp = torch.Generator(); gp.manual_seed(seed)
    log = dict(fold=k, arch=arch, lr=lr, temp=temp, seed=seed, nostretch=nostretch, parts=list(parts), n_train=N,
               n_pool=len(bg), steps_per_epoch=spe, epochs=C.EPOCHS,
               train_ids_sha256=hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest(),
               loss=[], guard_frac=[])
    t0 = time.time()
    for ep in range(C.EPOCHS):
        perm = torch.randperm(N, generator=gp).to(dev)
        L, G = [], []
        enc.train(); head.train()
        for b in range(spe):
            ii = perm[b * bs:(b + 1) * bs]
            args = (T["W"][ii], T["fs"][ii], T["ev0"][ii], T["ev1"][ii], T["sig"][ii], T["bg"], g)
            v1 = A.train_view(*args, parts=parts); v2 = A.train_view(*args, parts=parts)
            loss, gf = Mo.nt_xent(head(enc(v1)), head(enc(v2)), T["t"][ii], temp)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
            L.append(loss.item()); G.append(gf)
        log["loss"].append(float(np.mean(L))); log["guard_frac"].append(float(np.mean(G)))
    log["total_s"] = time.time() - t0
    torch.save(dict(arch=arch, enc=enc.state_dict(), head=head.state_dict()), os.path.join(od, "final.pt"))
    json.dump(log, open(os.path.join(od, "log.json"), "w"), indent=1)
    print(f"[{t_}] n={N} pool={len(bg)} loss {log['loss'][0]:.3f} -> {log['loss'][-1]:.3f} "
          f"guard {np.mean(log['guard_frac']):.1e} {log['total_s']:.0f}s", flush=True)
    return log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True); ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshard", type=int, default=1)
    a = ap.parse_args()
    jobs = json.load(open(a.jobs))
    for j in jobs[a.shard::a.nshard]:
        train(**j)


if __name__ == "__main__":
    main()
