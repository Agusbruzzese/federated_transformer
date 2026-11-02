"""Encoders (64-d embedding) and the shared projection head. No BatchNorm anywhere."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import config as C


class CNN(nn.Module):
    """1D CNN, GroupNorm(8) + GELU, 4 stride-2 stages then global average pooling."""

    def __init__(self, emb=C.EMB, ch=(32, 64, 96, 128, 128), ks=(9, 7, 7, 5, 5), st=(2, 2, 2, 2, 1)):
        super().__init__()
        L, cin = [], 1
        for c, k, s in zip(ch, ks, st):
            L += [nn.Conv1d(cin, c, k, s, k // 2), nn.GroupNorm(8, c), nn.GELU()]
            cin = c
        self.body = nn.Sequential(*L)
        self.out = nn.Linear(cin, emb)

    def forward(self, x):                     # x [B, 600]
        h = self.body(x[:, None, :]).mean(-1)
        return self.out(h)


class PatchTransformer(nn.Module):
    """Patch embedding (0.5 s patches -> 24 tokens), learnable positions, pre-LN encoder, mean pooling."""

    def __init__(self, emb=C.EMB, patch=25, d=64, depth=4, heads=4, ff=256):
        super().__init__()
        self.patch = patch
        n = C.CROP_N // patch
        self.embed = nn.Linear(patch, d)
        self.pos = nn.Parameter(torch.randn(1, n, d) * 0.02)
        layer = nn.TransformerEncoderLayer(d, heads, ff, dropout=0.0, activation="gelu", batch_first=True,
                                           norm_first=True)
        self.enc = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.out = nn.Linear(d, emb)

    def forward(self, x):
        B = x.shape[0]
        t = self.embed(x.view(B, -1, self.patch)) + self.pos
        h = self.norm(self.enc(t)).mean(1)
        return self.out(h)


class Head(nn.Module):
    def __init__(self, emb=C.EMB, proj=C.PROJ):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(emb, emb), nn.ReLU(), nn.Linear(emb, proj))

    def forward(self, h):
        return self.net(h)


def make(arch):
    return {"cnn": CNN, "tf": PatchTransformer}[arch]()


def n_params(m):
    return sum(p.numel() for p in m.parameters())


def nt_xent(z1, z2, t_s, temp, guard_s=C.GUARD_S):
    """SimCLR NT-Xent. z1, z2 [B,d] projections of two views; t_s [B] float64 event centroid times (s).
    False-negative guard: pairs of DIFFERENT events with |dt| < guard_s are removed from each other's
    negatives (all four view combinations). Returns (loss, fraction of negative pairs masked)."""
    B = z1.shape[0]
    z = F.normalize(torch.cat([z1, z2]), dim=1)
    sim = z @ z.t() / temp
    t = torch.cat([t_s, t_s])
    near = (t[:, None] - t[None, :]).abs() < guard_s
    idx = torch.arange(2 * B, device=z.device)
    pos = torch.cat([idx[B:], idx[:B]])
    same = torch.zeros_like(near); same[idx, idx] = True; same[idx, pos] = True
    guard = near & ~same
    sim = sim.masked_fill(guard | torch.eye(2 * B, dtype=torch.bool, device=z.device), float("-inf"))
    loss = F.cross_entropy(sim, pos)
    n_neg = 2 * B * (2 * B - 2)
    return loss, guard.sum().item() / n_neg
