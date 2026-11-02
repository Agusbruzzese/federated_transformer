"""Minimal I/O helpers for the TMR dataset (workstream B, signal characterisation).
Format (verified here): Values = little-endian uint32 holding a 24-bit ADC code (100000 samples);
Time = little-endian uint64 host timestamps in microseconds (monotonic, per-run epoch)."""
import os, re, glob
from datetime import datetime
import numpy as np

ROOT = os.environ.get("TMR_DATA_ROOT") or "/home/augustin/Documents/federated_transformer/Spintronic Sensor Dataset for Vehicle Detection and Car Model Recognition (14 Days, Unlabeled with Timestamps)"
PAT = re.compile(r"S0C0 BIN Values - (\d{6}) Start@ (\d{8}) - (\d\d)h(\d\d)m(\d\d)s Stop@")
GAP_US = 1000  # a sample interval > 1 ms marks a dropout


def list_chunks():
    """Return list of dicts (run, idx, start datetime, values path, time path) sorted by time."""
    out = []
    for ri, run in enumerate(sorted(d for d in os.listdir(ROOT) if d.startswith("Run"))):
        rdir = os.path.join(ROOT, run)
        names = os.listdir(rdir)
        tmap = {n.split(" Start@")[0].split(" - ")[-1]: n for n in names if n.startswith("S0C0 BIN Time")}
        for n in names:
            m = PAT.match(n)
            if not m:
                continue
            idx = m.group(1)
            dt = datetime.strptime(m.group(2) + m.group(3) + m.group(4) + m.group(5), "%Y%m%d%H%M%S")
            tn = tmap.get(idx)
            out.append(dict(run=ri, idx=int(idx), start=dt, vpath=os.path.join(rdir, n),
                            tpath=os.path.join(rdir, tn) if tn else None))
    out.sort(key=lambda d: (d["start"], d["run"], d["idx"]))
    return out


def load(ch, time=True):
    v = np.fromfile(ch["vpath"], "<u4").astype(np.float64)
    t = np.fromfile(ch["tpath"], "<u8").astype(np.int64) if (time and ch["tpath"]) else None
    return v, t


def segments(t, min_len=0):
    """Contiguous (no dropout) index ranges [a, b) given timestamps."""
    br = np.where(np.diff(t) > GAP_US)[0] + 1
    edges = np.r_[0, br, len(t)]
    return [(a, b) for a, b in zip(edges[:-1], edges[1:]) if b - a >= min_len]
