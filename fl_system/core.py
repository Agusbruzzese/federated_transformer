"""Shared core of the real federated system (report 25). VERBATIM copies of the frozen simulator's functions
(research/model/K_committee/protocol/fedsim3.py: lam, build, Stream, train_view_day, the local step loop, the aggregation
formula), so that a client process computes exactly what fedsim3 computed, WITHOUT importing study code (fedsim3 reads the
pooled noise file at import, which a client must never touch). Equality is proven by test T1 (report 26), not assumed.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from ssl_central import augment as A, config as C, models as Mo

WD, WARM, T_NCE, REF_BATCH = 1e-4, 0.05, 0.1, 256


def lam(i, steps):                                                   # fedsim3.lam (m5's LambdaLR, float-valued step index)
    wu = int(WARM * steps)
    return (i + 1) / wu if i < wu else 0.5 * (1 + math.cos(math.pi * (i - wu) / (steps - wu)))


def build(seed, dev):                                                # fedsim3.build
    torch.manual_seed(seed); np.random.seed(seed)
    return Mo.make("tf").to(dev), Mo.Head().to(dev)


def params(enc, head): return list(enc.parameters()) + list(head.parameters())


def get_weights(enc, head):
    return [p.detach().cpu().numpy().copy() for p in params(enc, head)]


@torch.no_grad()
def set_weights(enc, head, arrays):
    ps = params(enc, head); assert len(ps) == len(arrays)
    for p, a in zip(ps, arrays):
        assert tuple(p.shape) == tuple(a.shape); p.copy_(torch.as_tensor(a, device=p.device))


class Stream:                                                        # fedsim3.Stream
    """m5's batch order: a fresh randperm per epoch (CPU generator), N // bs batches, drop_last."""
    def __init__(self, n, bs, seed, dev):
        self.n, self.bs, self.dev = n, bs, dev; self.spe = n // bs; self.b = self.spe
        self.gp = torch.Generator(); self.gp.manual_seed(seed); self.perm = None
    def next(self):
        if self.b >= self.spe:
            self.perm = torch.randperm(self.n, generator=self.gp).to(self.dev); self.b = 0
        ii = self.perm[self.b * self.bs:(self.b + 1) * self.bs]; self.b += 1; return ii


def train_view_day(W, fs_eff, ev0, ev1, sigma_ev, bg, g, st, ct):   # fedsim3.train_view_day (verbatim)
    dev = W.device; B = W.shape[0]
    s = A._u(B, *C.STRETCH, g, dev); c = A.crop_centres(ev0, ev1, fs_eff, s, g); x = A.render(W, fs_eff, c, s)
    x = x * A._u(B, *C.AMP, g, dev)[:, None]
    k = st + (torch.rand(B, generator=g, device=dev) * ct).long().clamp(max=ct - 1)
    o = torch.randint(0, bg.shape[1] - C.CROP_N + 1, (B,), generator=g, device=dev)
    n = torch.gather(bg[k], 1, o[:, None] + torch.arange(C.CROP_N, device=dev)[None, :])
    x = x + n * (sigma_ev * A._u(B, *C.NOISE_ALPHA, g, dev))[:, None]
    Am = x.abs().amax(1, keepdim=True)
    off = A._u(B, -C.OFFSET, C.OFFSET, g, dev)[:, None]; dr = A._u(B, -C.DRIFT, C.DRIFT, g, dev)[:, None]
    ramp = torch.linspace(-0.5, 0.5, C.CROP_N, device=dev)[None, :]
    return A.maxnorm(x + Am * (off + dr * ramp))


def plan(n_k, N, X, R, batch):
    """fedsim3: bk = min(batch, n_k); sk = max(1, round(X / R * n_k / N / bk)) (same float64 operation order)."""
    n = np.array([n_k]); bk = np.minimum(batch, n); sk = np.maximum(1, np.round(X / R * n / N / bk)).astype(int)
    return int(sk[0]), int(bk[0])


def local_round(enc, head, opt, T, bg, g, stream, r, R, sk, steps_ref, lr):
    """fedsim3 inner loop for one client in round r (0-based). T: own events (tensors), bg: own pool."""
    L, gf, lrs = [], [], []; enc.train(); head.train()
    for j in range(sk):
        i_eq = (r * sk + j) * steps_ref / (R * sk)
        for pg in opt.param_groups: pg["lr"] = lr * lam(i_eq, steps_ref)
        lrs.append(opt.param_groups[0]["lr"])
        ii = stream.next()
        args = (T["W"][ii], T["fs"][ii], T["ev0"][ii], T["ev1"][ii], T["sig"][ii], bg, g)
        v1, v2 = train_view_day(*args, T["st"][ii], T["ct"][ii]), train_view_day(*args, T["st"][ii], T["ct"][ii])
        loss, frac = Mo.nt_xent(head(enc(v1)), head(enc(v2)), T["t"][ii], T_NCE)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); L.append(loss.item()); gf.append(frac)
    return dict(loss_first=L[0], loss_last=L[-1], loss_mean=float(np.mean(L)), guard=float(np.mean(gf)), lr_first=lrs[0], lr_last=lrs[-1])


def aggregate(vecs, examples):
    """fedsim3 FedAvg: new = sum_k (sk_k bk_k / sum) * v_k, accumulated in client order k = 0..K-1, float32."""
    ex = np.asarray(examples); new = torch.zeros_like(vecs[0])
    for k in range(len(vecs)):
        new += (ex[k] / ex.sum()) * vecs[k]
    return new


def flat(arrays, dev):
    return torch.cat([torch.as_tensor(a, device=dev).reshape(-1) for a in arrays])


def unflat(vec, shapes):
    out, i = [], 0
    for s in shapes:
        k = int(np.prod(s)); out.append(vec[i:i + k].reshape(s).cpu().numpy().copy()); i += k
    return out
