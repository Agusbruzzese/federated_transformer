"""OOF embedding assembly (J6), PCA per fold (J12), fold-init control and within-space sets (D1).

    python -m ssl_central_v3.embed
Every event (all SNR) of run k is embedded by fold k's models. Writes data/ssl_central_v3/emb/*.npy.
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch
from sklearn.decomposition import PCA

from ssl_central import embed as FE
from ssl_central import models as Mo
from . import config as C
from . import data as D
from .train import ckpt, tag

EMB = os.path.join(C.OUT, "emb")


def sel():
    return json.load(open(os.path.join(C.OUT, "selection.json")))


def main():
    os.makedirs(EMB, exist_ok=True)
    m, W = D.load_all()
    S = sel()
    crops = FE.eval_crops(W, m.fs_eff.to_numpy())
    run = m.run_id.to_numpy()
    N = len(m)
    out = {}

    def put(name, k, X):
        out.setdefault(name, np.full((N, X.shape[1]), np.nan, np.float32))
        out[name][run == k] = X[run == k]

    init_check = {}
    for k in C.RUNS:
        te = run == k
        for arch in C.ARCHS:
            s_ = S[f"{k}_{arch}"]
            for seed in C.SEEDS:
                t_ = tag(k, arch, s_["lr"], s_["temp"], seed)
                Xs = FE.embed_views(FE.load_encoder(ckpt(t_, "final")), crops)
                put(f"{arch}_ssl_s{seed}", k, Xs)
                np.save(os.path.join(EMB, f"ws_f{k}_{arch}_s{seed}.npy"), Xs.astype(np.float32))   # D1 within-space
                Xr = FE.embed_views(FE.load_encoder(ckpt(t_, "init")), crops[torch.from_numpy(te).to(crops.device)])
                Z = np.zeros((N, Xr.shape[1]), np.float32); Z[te] = Xr; put(f"{arch}_rand_s{seed}", k, Z)
                init_check.setdefault((arch, seed), []).append(
                    torch.load(ckpt(t_, "init"), map_location="cpu")["enc"])
                # D1 fold-specific init control
                torch.manual_seed(100 + 10 * k + seed)
                enc = Mo.make(arch).to("cuda").eval()
                Xf = FE.embed_views(enc, crops[torch.from_numpy(te).to(crops.device)])
                Z = np.zeros((N, Xf.shape[1]), np.float32); Z[te] = Xf; put(f"{arch}_foldinit_s{seed}", k, Z)
                if arch == "cnn":
                    t_ns = tag(k, arch, s_["lr"], s_["temp"], seed, nostretch=True)
                    if os.path.exists(ckpt(t_ns, "final")):
                        put(f"cnnns_ssl_s{seed}", k, FE.embed_views(FE.load_encoder(ckpt(t_ns, "final")), crops))
        tr = D.train_index(m, k)
        D.firewall(m, k, tr)
        Xc = crops.cpu().numpy()
        pca = PCA(64, random_state=0).fit(Xc[tr])
        put("pca64", k, pca.transform(Xc).astype(np.float32))
    # J3: the seed-s init must be identical across folds
    same = {f"{a}_s{s}": all(all(torch.equal(sd[q], L[0][q]) for q in sd) for sd in L[1:])
            for (a, s), L in init_check.items()}
    for name, X in out.items():
        assert not np.isnan(X).any(), name
        np.save(os.path.join(EMB, f"{name}.npy"), X)
    m[["event_id", "gi", "run_id", "snr"]].to_parquet(os.path.join(EMB, "index.parquet"))
    json.dump(dict(init_identical_across_folds=same, names=sorted(out)), open(os.path.join(EMB, "info.json"), "w"),
              indent=1)
    print("written", sorted(out)); print("init identical across folds:", same)


if __name__ == "__main__":
    main()
