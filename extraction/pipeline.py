"""Per-block processing: detection (C) + feature path (B) + windows, masks, morphology (D, fs-corrected)."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from . import params as PR
from .blockio import SlotClock, load_block
from .featpath import feature_path
from .morph import morph_feats
from ._vendor import detector as CD

STEP = PR.DOWN / PR.UP          # slots per output sample (48.32)


def make_blocks(ixr: pd.DataFrame, core=PR.CORE_CHUNKS):
    """[(core_i0, core_i1)] positions covering the run."""
    n = len(ixr)
    return [(a, min(a + core, n)) for a in range(0, n, core)]


def detect_block(v, slot):
    """C's detect_stream on a stitched block. t is A's exact slot index (relative), so C's dropout re-insertion
    uses A's slots and C's times (slot/2416) map back to slots exactly: slot = slot[0] + t*2416."""
    ev, _ = CD.detect_stream(v, (slot - slot[0]).astype(float))
    s0 = int(slot[0])
    for e in ev:
        for k in ("onset", "offset", "centroid_t", "peak_t", "e_on", "e_off"):
            e["slot_" + k] = s0 + e[k] * CD.FS_RAW
    return ev


def cut(r, mask, J0, j_start, n):
    """Window r[j_start - J0 : +n] with zero padding outside; returns (w, repaired mask, n_outside)."""
    k0 = j_start - J0
    w = np.zeros(n, np.float32)
    rm = np.ones(n, bool)
    a, b = max(k0, 0), min(k0 + n, len(r))
    if b > a:
        w[a - k0:b - k0] = r[a:b]
        rm[a - k0:b - k0] = mask[a:b]
    return w, rm, n - max(b - a, 0)


def process_block(job, v=None, slot=None):
    """job: dict(run, run_id, ixr (run's chunk index), core=(i0, i1), ctx). v/slot: optional preloaded
    (e.g. injected) arrays for exactly the chunks [core0 - ctx, core1 + ctx) (tests)."""
    t_start = time.time(); cpu0 = time.process_time()
    ixr = job["ixr"]; c0, c1 = job["core"]; ctx = job.get("ctx", PR.CONTEXT_CHUNKS)
    n = len(ixr)
    i0, i1 = max(0, c0 - ctx), min(n, c1 + ctx)
    if v is None:
        B = load_block(ixr, i0, i1)
        v, slot = B["v"], B["slot"]
    clock = SlotClock(ixr)
    core_s0 = int(ixr.slot_start.iloc[c0])
    core_s1 = int(ixr.slot_start.iloc[c1]) if c1 < n else int(slot[-1]) + 1
    run_s_first = int(ixr.slot_start.iloc[0])
    ev = detect_block(v, slot)
    F = feature_path(v, slot, clock, extra_event_slots=[(e["slot_onset"], e["slot_offset"]) for e in ev])
    r, mask, J0 = F["r"], F["mask"], F["J0"]
    half = PR.WIN_N // 2
    rows, W, EM, RM, excl = [], [], [], [], []
    for e in ev:
        sc = e["slot_centroid_t"]
        if not (core_s0 <= sc < core_s1):
            continue
        jc = sc / STEP
        w0 = int(np.rint(jc)) - half
        w, rm_, n_out = cut(r, mask, J0, w0, PR.WIN_N)
        j_on = int(np.rint(e["slot_onset"] / STEP)); j_off = int(np.rint(e["slot_offset"] / STEP))
        ev_mask = np.zeros(PR.WIN_N, bool)
        ev_mask[max(j_on - w0, 0):max(min(j_off - w0, PR.WIN_N), 0)] = True
        # mask_frac over the event extent [onset, offset) on the feature grid (B's definition)
        ka, kb = max(j_on - J0, 0), min(max(j_off - J0, j_on - J0 + 1), len(mask))
        mf = float(mask[ka:kb].mean()) if kb > ka else 1.0
        fs_eff, fs_min, fs_max = clock.fs_eff_mean(w0 * STEP, (w0 + PR.WIN_N) * STEP)
        s_c_int = int(np.floor(sc))
        wall = clock.wall_time(np.array([s_c_int]))[0] + np.timedelta64(int(round((sc - s_c_int) / clock.fs_local(s_c_int) * 1e9)), "ns")
        meta = dict(run=job["run"], run_id=job["run_id"], chunk=int(clock.chunk(s_c_int)),
                    slot_centroid=float(sc), slot_onset=float(e["slot_onset"]), slot_offset=float(e["slot_offset"]),
                    wall_time=wall, snr=float(e["snr"]), peak_z=float(e["peak_z"]),
                    dur_s=float((e["slot_offset"] - e["slot_onset"]) / (fs_eff * STEP)),
                    has_dropout=bool(e["has_dropout"]), mask_frac=mf, win_mask_frac=float(rm_.mean()),
                    n_neighbours_in_window=int(e["n_neighbours_in_window"]),
                    fs_eff=fs_eff, fs_eff_min=fs_min, fs_eff_max=fs_max, fs_local=float(clock.fs_local(s_c_int)),
                    truncated=bool(n_out > 0), n_outside=int(n_out),
                    win_start_j=int(w0), centre_offset_samples=float(jc - np.rint(jc)),
                    ev_i0=int(max(j_on - w0, 0)), ev_i1=int(min(j_off - w0, PR.WIN_N)),
                    near_run_edge=bool(i0 == 0 and sc - run_s_first < 60 * 2416 or c1 == n and slot[-1] - sc < 60 * 2416),
                    block_core0=int(c0))
        if mf > PR.MASK_FRAC_MAX:
            excl.append(meta)
            continue
        m21 = morph_feats(w, fs_eff)
        cc = PR.WIN_N // 2 - PR.CROP8_N // 2
        m8 = morph_feats(w[cc:cc + PR.CROP8_N], fs_eff)
        meta.update({f"m_{k}": float(x) for k, x in zip(_names(), m21)})
        meta.update({f"m8_{k}": float(x) for k, x in zip(_names(), m8)})
        rows.append(meta); W.append(w); EM.append(ev_mask); RM.append(rm_)
    stats = dict(run=job["run"], core0=c0, core1=c1, n_chunks=i1 - i0, n_det_block=len(ev),
                 n_core=len(rows) + len(excl), n_kept=len(rows), n_excluded=len(excl),
                 core_hours=(core_s1 - core_s0) / 2416.0 / 3600, n_split=F["n_split"], n_filled=F["n_filled"],
                 feat_masked_frac=float(mask.mean()), wall_s=time.time() - t_start,
                 cpu_s=time.process_time() - cpu0)
    E = lambda a, shape, dt: np.stack(a) if a else np.zeros(shape, dt)
    return dict(rows=rows, excl=excl, W=E(W, (0, PR.WIN_N), np.float32), EM=E(EM, (0, PR.WIN_N), bool),
                RM=E(RM, (0, PR.WIN_N), bool), stats=stats)


def _names():
    from .morph import MORPH_NAMES
    return MORPH_NAMES
