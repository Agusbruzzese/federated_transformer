"""Minimal I/O for workstream C (event detection). Values: <u4 holding 24-bit ADC codes; Time: <u8 DEVICE TICKS
(not microseconds; 1-5 % fast vs wall clock, workstream A F3). True sample rate ~2416 Hz from the mains line."""
import os, re, glob, datetime as dt
import numpy as np

ROOT = "/home/augustin/Documents/federated_transformer/Spintronic Sensor Dataset for Vehicle Detection and Car Model Recognition (14 Days, Unlabeled with Timestamps)"
PAT = re.compile(r"S0C0 BIN Values - (\d+) Start@ (\d{8}) - (\d\d)h(\d\d)m(\d\d)s Stop@ (\d{8}) - (\d\d)h(\d\d)m(\d\d)s\.bsd")

def runs():
    return sorted(d for d in glob.glob(os.path.join(ROOT, "Run *")) if os.path.isdir(d))

_cache = {}
def index(run):
    """DataFrame-like list of (idx, start_dt, stop_dt, values_path, time_path) sorted by idx."""
    if run in _cache: return _cache[run]
    out = []
    for f in os.listdir(run):
        m = PAT.match(f)
        if not m: continue
        g = m.groups()
        s = dt.datetime.strptime(g[1] + g[2] + g[3] + g[4], "%Y%m%d%H%M%S")
        e = dt.datetime.strptime(g[5] + g[6] + g[7] + g[8], "%Y%m%d%H%M%S")
        out.append((int(g[0]), s, e, os.path.join(run, f), os.path.join(run, f.replace("BIN Values", "BIN Time"))))
    out.sort()
    _cache[run] = out
    return out

def run_start(run):
    b = os.path.basename(run)
    return dt.datetime.strptime(b[4:22], "%Y%m%d %Hh%Mm%Ss")

def load_chunks(run, i0, n):
    """Load n consecutive chunks starting at chunk index i0. Returns (x float64 counts, t = Time ticks/1e6 (device ticks, NOT seconds: 1-5 % fast, workstream A F3; only used for dropout detection via ratios), chunk_id per sample)."""
    idx = index(run)
    pos = {c[0]: k for k, c in enumerate(idx)}
    xs, ts, cs = [], [], []
    for i in range(i0, i0 + n):
        if i not in pos: continue
        _, _, _, vp, tp = idx[pos[i]]
        x = np.fromfile(vp, "<u4").astype(np.float64)
        t = np.fromfile(tp, "<u8").astype(np.float64) / 1e6
        m = min(len(x), len(t))
        xs.append(x[:m]); ts.append(t[:m]); cs.append(np.full(m, i, np.int32))
    return np.concatenate(xs), np.concatenate(ts), np.concatenate(cs)

def find_chunk(run, when):
    """chunk index whose start <= when (datetime)."""
    best = None
    for c in index(run):
        if c[1] <= when: best = c[0]
        else: break
    return best
