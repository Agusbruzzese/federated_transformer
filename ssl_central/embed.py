"""Frozen-embedding export (deterministic central 12 s crop, I3) plus P3 query views."""
from __future__ import annotations

import os

import numpy as np
import torch

from . import augment as A
from . import config as C
from . import models as Mo


def load_encoder(path, dev="cuda"):
    ck = torch.load(path, map_location=dev)
    enc = Mo.make(ck["arch"]).to(dev); enc.load_state_dict(ck["enc"]); enc.eval()
    return enc


@torch.no_grad()
def embed_views(enc, views, bs=2048):
    return np.concatenate([enc(views[i:i + bs]).float().cpu().numpy() for i in range(0, len(views), bs)])


@torch.no_grad()
def eval_crops(W, fs, dev="cuda", bs=4096):
    out = []
    for i in range(0, len(W), bs):
        out.append(A.eval_view(torch.tensor(W[i:i + bs], device=dev),
                               torch.tensor(fs[i:i + bs].astype(np.float32), device=dev)))
    return torch.cat(out)


def ckpt_path(tag, which):
    return os.path.join(C.OUT, "ckpt", tag, f"{which}.pt")
