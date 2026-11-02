"""Stitched block loading on workstream A's run-global ADC slot grid.

Every sample gets an exact run-global slot index: slot = slot_start(chunk) + A's within-chunk slot
(bsd_io.slots_from_timestamps). The chunk index's slot_start is built from the exact full-scan
n_missing and the tick-derived inter-chunk slots, so slots are identical whichever block loads a chunk
(verified in the self-tests). Wall time of a slot s in chunk c: wall_start(c) + (s - slot_start(c)) / fs_local(c)
(A's piecewise-linear PC-clock fit; +-0.35 s absolute).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ._vendor import bsd_io

UP, DOWN = 25, 1208


def chunk_index() -> pd.DataFrame:
    ix = bsd_io.list_chunks()
    ix = ix[ix.paired].copy()
    runs = sorted(ix.run.unique())
    ix["run_id"] = ix.run.map({r: i for i, r in enumerate(runs)}).astype(int)
    return ix.reset_index(drop=True)


def run_label(run: str) -> str:
    return run[4:12]          # e.g. '20250204'


def load_block(ixr: pd.DataFrame, i0: int, i1: int):
    """Load chunks with positions i0..i1-1 of one run's index frame `ixr` (sorted by index).
    Returns dict(v float64 raw 24-bit counts, slot int64 run-global, chunks DataFrame of the loaded chunks)."""
    V, S = [], []
    rows = ixr.iloc[i0:i1]
    prev_last = None
    for r in rows.itertuples():
        _, v, info = bsd_io.load_chunk(r.values_path, r.time_path, fs=r.fs_local, return_info=True)
        s = int(r.slot_start) + info["slots"]
        if prev_last is not None:
            # exact stitching (as in bsd_io.load_span): must agree with the index's slot_start
            expect = prev_last + int(r.interchunk_slots)
            if expect != s[0]:
                raise AssertionError(f"slot mismatch at {r.run} #{r.index}: {expect} vs {s[0]}")
        prev_last = int(s[-1])
        V.append(v); S.append(s)
    return dict(v=np.concatenate(V), slot=np.concatenate(S), chunks=rows.reset_index(drop=True))


class SlotClock:
    """Map run-global slots -> (chunk row, fs_local, wall time) using the run's chunk index."""

    def __init__(self, ixr: pd.DataFrame):
        self.s0 = ixr.slot_start.to_numpy(np.int64)
        self.fs = ixr.fs_local.to_numpy(float)
        self.idx = ixr["index"].to_numpy()
        self.wall = ixr.wall_start.to_numpy("datetime64[ns]")

    def pos(self, slot):
        return np.clip(np.searchsorted(self.s0, slot, side="right") - 1, 0, len(self.s0) - 1)

    def chunk(self, slot):
        return self.idx[self.pos(slot)]

    def fs_local(self, slot):
        return self.fs[self.pos(slot)]

    def wall_time(self, slot):
        p = self.pos(slot)
        dt_ns = np.rint((np.asarray(slot, float) - self.s0[p]) / self.fs[p] * 1e9).astype("int64")
        return self.wall[p] + dt_ns.astype("timedelta64[ns]")

    def fs_eff_mean(self, slot_a: float, slot_b: float) -> tuple[float, float, float]:
        """Slot-weighted mean, min and max of fs_local*25/1208 over [slot_a, slot_b)."""
        pa, pb = self.pos(int(np.floor(slot_a))), self.pos(int(np.ceil(slot_b)) - 1)
        if pa == pb:
            f = self.fs[pa] * UP / DOWN
            return f, f, f
        edges = np.r_[slot_a, self.s0[pa + 1:pb + 1], slot_b].astype(float)
        w = np.diff(edges)
        fs = self.fs[pa:pb + 1] * UP / DOWN
        return float(np.average(fs, weights=w)), float(fs.min()), float(fs.max())
