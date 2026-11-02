"""SimCLR training of one (arch, lr, temperature, seed) cell on the TRAIN split (SNR >= 20).

    python -m ssl_central.train --arch cnn --lr 1e-3 --temp 0.1 --seed 0 [--epochs 100] [--subset 64] [--tag x]
Saves <OUT>/ckpt/<tag>/{init.pt, final.pt, log.json}. No validation or holdout event is ever loaded here.
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

from . import augment as A
from . import config as C
from . import data as D
from . import models as Mo


def cell_tag(arch, lr, temp, seed):
    return f"{arch}_lr{lr:g}_t{temp:g}_s{seed}"


def load_train(dev, subset=None, seed=0):
    m, W, EM, RM = D.load_all()
    idx = D.split_index(m, "train")
    D.firewall(m, idx)
    if subset:
        idx = np.sort(np.random.default_rng(seed).choice(idx, subset, replace=False))
    s = m.iloc[idx]
    T = dict(W=torch.tensor(W[idx], device=dev), fs=torch.tensor(s.fs_eff.to_numpy(np.float32), device=dev),
             ev0=torch.tensor(s.ev_i0.to_numpy(), device=dev), ev1=torch.tensor(s.ev_i1.to_numpy(), device=dev),
             sig=torch.tensor(s.sigma_ev.to_numpy(np.float32), device=dev),
             t=torch.tensor(s.t_s.to_numpy(np.float64), device=dev),
             bg=torch.tensor(np.load(os.path.join(C.OUT, "bgpool.npy")), device=dev))
    return T, s.event_id.to_numpy()


def train(arch, lr, temp, seed, epochs=C.EPOCHS, subset=None, tag=None, dev="cuda", verbose=True):
    tag = tag or cell_tag(arch, lr, temp, seed)
    od = os.path.join(C.OUT, "ckpt", tag); os.makedirs(od, exist_ok=True)
    T, ids = load_train(dev, subset, seed)
    N = len(ids)
    torch.manual_seed(seed); np.random.seed(seed)
    enc = Mo.make(arch).to(dev); head = Mo.Head().to(dev)
    torch.save(dict(arch=arch, enc=enc.state_dict()), os.path.join(od, "init.pt"))
    opt = torch.optim.AdamW(list(enc.parameters()) + list(head.parameters()), lr=lr, weight_decay=C.WD)
    bs = min(C.BATCH, N)
    spe = N // bs
    total = spe * epochs
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda k: 0.5 * (1 + math.cos(math.pi * min(k, total) / total)))
    g = torch.Generator(device=dev); g.manual_seed(seed)
    gp = torch.Generator(); gp.manual_seed(seed)
    log = dict(arch=arch, lr=lr, temp=temp, seed=seed, n_train=N, steps_per_epoch=spe, epochs=epochs,
               n_params_enc=Mo.n_params(enc), n_params_head=Mo.n_params(head),
               config_sha256=hashlib.sha256(open(C.__file__, "rb").read()).hexdigest(),
               train_ids_sha256=hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest(),
               loss=[], guard_frac=[], epoch_s=[])
    t_all = time.time()
    for ep in range(epochs):
        t0 = time.time()
        perm = torch.randperm(N, generator=gp).to(dev)
        L, G = [], []
        enc.train(); head.train()
        for b in range(spe):
            ii = perm[b * bs:(b + 1) * bs]
            args = (T["W"][ii], T["fs"][ii], T["ev0"][ii], T["ev1"][ii], T["sig"][ii], T["bg"], g)
            v1 = A.train_view(*args); v2 = A.train_view(*args)
            loss, gf = Mo.nt_xent(head(enc(v1)), head(enc(v2)), T["t"][ii], temp)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
            L.append(loss.item()); G.append(gf)
        if dev == "cuda":
            torch.cuda.synchronize()
        log["loss"].append(float(np.mean(L))); log["guard_frac"].append(float(np.mean(G)))
        log["epoch_s"].append(time.time() - t0)
        if verbose and (ep % 10 == 0 or ep == epochs - 1):
            print(f"[{tag}] ep {ep:3d} loss {log['loss'][-1]:.4f} guard {log['guard_frac'][-1]:.2e} "
                  f"{log['epoch_s'][-1]:.2f}s", flush=True)
    log["total_s"] = time.time() - t_all
    log["max_mem_gb"] = torch.cuda.max_memory_allocated() / 1e9 if dev == "cuda" else None
    torch.save(dict(arch=arch, enc=enc.state_dict(), head=head.state_dict()), os.path.join(od, "final.pt"))
    json.dump(log, open(os.path.join(od, "log.json"), "w"), indent=1)
    return log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True); ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--temp", type=float, required=True); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=C.EPOCHS); ap.add_argument("--subset", type=int, default=None)
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()
    lg = train(a.arch, a.lr, a.temp, a.seed, a.epochs, a.subset, a.tag)
    print(f"done {lg['total_s']:.1f}s, mem {lg['max_mem_gb']:.2f} GB, final loss {lg['loss'][-1]:.4f}")


if __name__ == "__main__":
    main()
