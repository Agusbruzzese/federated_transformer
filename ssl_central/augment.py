"""GPU rendering of encoder inputs (12 s, 50 Hz, max-normalised) and SimCLR view augmentations.

Allowed (PREREG): random 12 s crop, SNR-matched real background noise, amplitude U(0.8, 1.25),
time-stretch U(0.85, 1.18), small offset / linear drift. Forbidden and NOT implemented: sign flip,
time reversal, < 4 Hz masking.
"""
from __future__ import annotations

import torch

from . import config as C

_I = None


def _grid(device):
    global _I
    if _I is None or _I.device != device:
        _I = torch.arange(C.CROP_N, device=device, dtype=torch.float32) - (C.CROP_N - 1) / 2.0
    return _I


def render(W, fs_eff, centre, stretch):
    """Linear interpolation of windows W [B,1050] onto a 600-sample 50 Hz grid centred at source index
    `centre` [B], with time-stretch factor `stretch` [B] (>1 = slower event, i.e. longer)."""
    r = (fs_eff / C.FS_OUT / stretch)[:, None]
    p = centre[:, None] + _grid(W.device)[None, :] * r
    p = p.clamp(0, C.WIN_N - 1 - 1e-4)
    i0 = p.floor().long(); f = p - i0
    x0 = torch.gather(W, 1, i0); x1 = torch.gather(W, 1, (i0 + 1).clamp(max=C.WIN_N - 1))
    return x0 + f * (x1 - x0)


def maxnorm(x):
    return x / x.abs().amax(1, keepdim=True).clamp_min(1e-6)


def eval_view(W, fs_eff):
    B = W.shape[0]
    c = torch.full((B,), C.CENTRE, device=W.device)
    return maxnorm(render(W, fs_eff, c, torch.ones(B, device=W.device)))


def heldout_view(W, fs_eff, sign):
    """P3 held-out augmentation: +-0.5 s shift (sign [B] in {-1,+1}) and 0.9x stretch together, no noise."""
    B = W.shape[0]
    c = C.CENTRE + sign * 0.5 * fs_eff
    return maxnorm(render(W, fs_eff, c, torch.full((B,), 0.9, device=W.device)))


def _u(B, lo, hi, g, dev):
    return lo + (hi - lo) * torch.rand(B, generator=g, device=dev)


def crop_centres(ev0, ev1, fs_eff, stretch, g):
    """Random crop centre (I4): contain [ev0, ev1) if it fits, else keep the window centre in the middle half."""
    dev = ev0.device; B = ev0.shape[0]
    half = (C.CROP_N - 1) / 2.0 * fs_eff / C.FS_OUT / stretch
    lo_w, hi_w = half, C.WIN_N - 1 - half
    lo_c = torch.maximum(lo_w, (ev1 - 1).float() - half)
    hi_c = torch.minimum(hi_w, ev0.float() + half)
    alt_lo = torch.maximum(lo_w, C.CENTRE - half / 2); alt_hi = torch.minimum(hi_w, C.CENTRE + half / 2)
    bad = lo_c > hi_c
    lo = torch.where(bad, alt_lo, lo_c); hi = torch.where(bad, alt_hi, hi_c)
    return lo + (hi - lo) * torch.rand(B, generator=g, device=dev)


def train_view(W, fs_eff, ev0, ev1, sigma_ev, bg, g, parts=("crop", "stretch", "amp", "noise", "offset")):
    """One augmented view per row. bg: background pool [M, 640] (unit sigma) on the same device."""
    dev = W.device; B = W.shape[0]
    s = _u(B, *C.STRETCH, g, dev) if "stretch" in parts else torch.ones(B, device=dev)
    if "crop" in parts:
        c = crop_centres(ev0, ev1, fs_eff, s, g)
    else:
        c = torch.full((B,), C.CENTRE, device=dev)
    x = render(W, fs_eff, c, s)
    if "amp" in parts:
        x = x * _u(B, *C.AMP, g, dev)[:, None]
    if "noise" in parts:
        k = torch.randint(0, bg.shape[0], (B,), generator=g, device=dev)
        o = torch.randint(0, bg.shape[1] - C.CROP_N + 1, (B,), generator=g, device=dev)
        n = torch.gather(bg[k], 1, o[:, None] + torch.arange(C.CROP_N, device=dev)[None, :])
        x = x + n * (sigma_ev * _u(B, *C.NOISE_ALPHA, g, dev))[:, None]
    if "offset" in parts:
        A = x.abs().amax(1, keepdim=True)
        off = _u(B, -C.OFFSET, C.OFFSET, g, dev)[:, None]
        dr = _u(B, -C.DRIFT, C.DRIFT, g, dev)[:, None]
        ramp = torch.linspace(-0.5, 0.5, C.CROP_N, device=dev)[None, :]
        x = x + A * (off + dr * ramp)
    return maxnorm(x)
