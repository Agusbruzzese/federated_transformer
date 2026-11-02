"""
bsd_io.py -- validated I/O for the TMR spintronic sensor dataset (workstream A).

File format (verified, see REPORT.md):
  * "S0C0 BIN Values - NNNNNN Start@ ... .bsd": little-endian uint32, one per sample,
    100 000 samples (400 000 B) per full chunk. Only the low 24 bits are used (top byte is
    always 0): a 24-bit ADC code. Observed codes 11.19M..11.97M (0xAAxxxx..0xB6xxxx), i.e.
    far from both rails. Offset-binary vs two's-complement cannot be decided from the data
    (the code never crosses 0x800000) -- the two interpretations differ only by a constant,
    so for features just use float counts and remove the baseline.
  * "S0C0 BIN Time - ...": little-endian uint64, one per sample (800 000 B / chunk),
    device timer ticks, nominally microseconds, origin = logger start (first sample of a run
    is at ~4.5-16 s). Strictly increasing within and across chunks of a run.
    The tick rate is NOT a trustworthy microsecond: relative to the PC wall clock it is off by
    +1 % .. +5 % depending on run and drifts ~1 % with time of day. Use it only to
    (a) order samples and (b) locate dropouts (gaps). Use the ADC sample grid for time.
  * ADC grid: the ADC samples uniformly at fs ~= 2415 Hz (PC-clock referenced; nominal 2400).
    Samples are lost in dropouts (mostly bursts of 106 missing samples = one ~44.5 ms timer
    gap; also 1-4-sample gaps). We reconstruct an integer "slot" index per sample.

Public API
  list_chunks(root, cache=True, refresh=False) -> DataFrame (one row per chunk)
  load_chunk(values_path, time_path, fs=FS_NOMINAL_EFFECTIVE, return_info=False)
        -> (t_seconds, values)  [t on the reconstructed ADC slot grid, 0 at first sample]
  load_span(root, start, end, index=None) -> Span(t (datetime64[ns]), values, gaps DataFrame, ...)
  slots_from_timestamps(t_ticks) -> (slots int64, base_dt float)

Run `python bsd_io.py` for the self-test.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd

DEFAULT_ROOT = ("/home/augustin/Documents/federated_transformer/Spintronic Sensor Dataset for Vehicle "
                "Detection and Car Model Recognition (14 Days, Unlabeled with Timestamps)")
HERE = os.path.dirname(os.path.abspath(__file__))
INDEX_PATH = os.path.join(HERE, "chunk_index.parquet")

VALUES_DTYPE = np.dtype("<u4")
TIME_DTYPE = np.dtype("<u8")
SAMPLES_PER_CHUNK = 100_000
ADC_BITS = 24
FS_CONFIG = 2400.0                 # "Sampling Rate :: 2400" in Configuration - Hardware.cfg
FS_NOMINAL_EFFECTIVE = 2415.0      # measured vs PC wall clock (2413.9-2417.8 Hz per run)
EDGE_N = 2000                      # timestamps read at each end of a Time file for the index
GAP_FACTOR = 1.5                   # dt > GAP_FACTOR * median(dt) => at least one sample missing

_PAT = re.compile(r"S0C0 BIN (Values|Time) - (\d+) Start@ (\d{8}) - (\d\d)h(\d\d)m(\d\d)s "
                  r"Stop@ (\d{8}) - (\d\d)h(\d\d)m(\d\d)s\.bsd$")
_RUN_PAT = re.compile(r"Run (\d{8}) (\d\d)h(\d\d)m(\d\d)s - (\d{8}) (\d\d)h(\d\d)m(\d\d)s$")


def _dt(d, h, m, s):
    return datetime.strptime(f"{d}{h}{m}{s}", "%Y%m%d%H%M%S")


def parse_chunk_name(fn: str):
    """Return dict(kind, index, fn_start, fn_stop) or None for non-chunk files."""
    m = _PAT.match(os.path.basename(fn))
    if not m:
        return None
    return dict(kind=m.group(1), index=int(m.group(2)),
                fn_start=_dt(*m.group(3, 4, 5, 6)), fn_stop=_dt(*m.group(7, 8, 9, 10)))


# --------------------------------------------------------------------------- timestamps -> slots
LATE_MEAN = 0.24   # mean lateness (slots) of the first stamp after a dropout (measure_postgap.py)


def slots_from_timestamps(t, base_dt: float | None = None):
    """Reconstruct the integer ADC-slot index of every sample from device timestamps
    ("clean-anchor" estimator, v2).

    Timestamp model (measured, measure_postgap.py): uniform ADC grid + white stamp jitter
    (~2 ticks rms, 0.005 slot) ; the FIRST stamp after a dropout is late by 0..0.9 slot
    (mean 0.24). A delta dt > 1.5*median marks a 'gap' (it may also be a late stamp with no
    loss). Samples whose incoming delta is a gap are 'late'; all others are 'clean'.
      * clean -> clean consecutive, no gap: +1 slot
      * between clean samples a < c (with late samples in between):
            slots(c) - slots(a) = max(c - a, rint((t_c - t_a) / base))
      * a late sample j followed by a clean sample without a gap: slot(c) - 1 (exact)
      * a late sample followed by another gap (bursts with spacing 1): rint of the
        lateness-corrected position, clamped to stay strictly between its neighbours.
    Validated on synthetic data (synthetic_slot_test.py). Returns (slots int64 from 0, base_dt).
    """
    t = np.asarray(t, dtype=np.int64)
    n = len(t)
    if n < 2:
        return np.zeros(n, np.int64), float("nan")
    d = np.diff(t)
    med = np.median(d)
    gap = d >= GAP_FACTOR * med
    if base_dt is None:
        base_dt = float(d[~gap].mean())
    late = np.zeros(n, bool); late[1:] = gap
    C = np.flatnonzero(~late)                       # clean samples (sample 0 is clean)
    dc = np.diff(C)
    inc = np.maximum(dc, np.rint((t[C[1:]] - t[C[:-1]]) / base_dt).astype(np.int64))
    inc[dc == 1] = 1                                # consecutive clean samples: no gap between them
    slots = np.empty(n, np.int64)
    slots[C] = np.concatenate([[0], np.cumsum(inc)])
    L = np.flatnonzero(late)
    if len(L):
        prev_c = C[np.searchsorted(C, L) - 1]
        for j, a in zip(L, prev_c):
            if j + 1 < n and not gap[j] and not late[j + 1]:
                slots[j] = slots[j + 1] - 1
            else:
                s = slots[a] + int(np.rint((t[j] - t[a]) / base_dt - LATE_MEAN))
                lo = slots[j - 1] + 1
                s = max(s, lo)
                if j + 1 < n and not late[j + 1]:
                    s = min(s, slots[j + 1] - 1)
                slots[j] = s
        # enforce strict monotonicity (only reachable in pathological bursts)
        for i in np.flatnonzero(np.diff(slots) < 1) + 1:
            for k in range(i, n):
                if slots[k] > slots[k - 1]:
                    break
                slots[k] = slots[k - 1] + 1
    return slots, base_dt


def _slots_v1(t, base_dt=None, window=8):
    """Superseded v1 estimator (kept only for the synthetic comparison)."""
    t = np.asarray(t, dtype=np.int64); n = len(t)
    d = np.diff(t); med = np.median(d); ok = d < GAP_FACTOR * med
    if base_dt is None:
        base_dt = float(d[ok].mean())
    inc = np.ones(n - 1, np.int64)
    gaps = np.flatnonzero(~ok); nxt = np.append(gaps[1:], n - 1)
    for g, g2 in zip(gaps, nxt):
        w = int(min(window, g2 - g)); m = np.arange(1, w + 1)
        est = np.rint((t[g + m] - t[g]) / base_dt) - (m - 1)
        inc[g] = max(1, int(np.median(est)))
    return np.concatenate([[0], np.cumsum(inc)]), base_dt


# --------------------------------------------------------------------------- index
def _edge_stats(time_path, size):
    n = size // 8
    with open(time_path, "rb") as f:
        a = np.frombuffer(f.read(8 * min(EDGE_N, n)), TIME_DTYPE).astype(np.int64)
        f.seek(max(0, n - EDGE_N) * 8)
        b = np.frombuffer(f.read(), TIME_DTYPE).astype(np.int64)
    d = np.concatenate([np.diff(a), np.diff(b)])
    if len(d) == 0:
        return a[0], b[-1], np.nan, 0.0
    ok = d < GAP_FACTOR * np.median(d)
    base = d[ok].mean()
    miss = max(0.0, (b[-1] - a[0]) / base - (n - 1))
    return int(a[0]), int(b[-1]), float(base), float(miss)


def _first_value(p):
    with open(p, "rb") as f:
        return int(np.frombuffer(f.read(4), VALUES_DTYPE)[0])


def _fit_wall_clock(slot_start, wall_s, knot_every=40):
    """Piecewise-linear least squares wall_seconds ~ f(slot) with knots every `knot_every`
    chunks (~28 min), two passes with 3-sigma clipping. Returns knot (slots, seconds)."""
    x = np.asarray(slot_start, float); y = np.asarray(wall_s, float)
    nk = max(2, int(np.ceil(len(x) / knot_every)) + 1)
    kx = np.linspace(x.min(), x.max(), nk)
    use = np.ones(len(x), bool)
    for _ in range(2):
        A = np.zeros((len(x), nk))
        pos = np.clip(np.searchsorted(kx, x, side="right") - 1, 0, nk - 2)
        f = (x - kx[pos]) / (kx[pos + 1] - kx[pos])
        A[np.arange(len(x)), pos] = 1 - f
        A[np.arange(len(x)), pos + 1] = f
        ky, *_ = np.linalg.lstsq(A[use], y[use], rcond=None)
        r = y - A @ ky
        use = np.abs(r) < max(3 * r[use].std(), 1.0)
    return kx, ky


def list_chunks(root: str = DEFAULT_ROOT, cache: bool = True, refresh: bool = False) -> pd.DataFrame:
    """One row per (run, index) chunk.

    Columns: run, run_start, run_stop (from folder name), index, fn_start, fn_stop,
      values_path, time_path, values_size, time_size, paired, n_samples, truncated,
      value_first, t_first, t_last (device ticks), base_dt (ticks/slot, from 2x2000 edge samples),
      n_missing_est (dropped samples inside chunk from the edge scan, +-10), n_missing (exact,
      from full_scan.py when available; else = n_missing_est), missing_exact, n_drop_events,
      max_gap_slots, interchunk_slots (slots between the
      previous chunk's last sample and this chunk's first, 1 = contiguous),
      slot_start (cumulative ADC slot of first sample within the run),
      wall_start (fitted PC-clock datetime of first sample), fs_local (fitted Hz),
      wall_resid_s (filename start - fitted, s).
    Only file names/sizes, 4 bytes of each Values file and 2x16 KB of each Time file are read.
    """
    if cache and not refresh and os.path.exists(INDEX_PATH):
        return pd.read_parquet(INDEX_PATH)
    rows = {}
    for run in sorted(os.listdir(root)):
        rd = os.path.join(root, run)
        rm = _RUN_PAT.match(run)
        if not (os.path.isdir(rd) and rm):
            continue
        for fn in os.listdir(rd):
            p = parse_chunk_name(fn)
            if p is None:
                continue
            key = (run, p["index"])
            r = rows.setdefault(key, dict(run=run, run_start=_dt(*rm.group(1, 2, 3, 4)),
                                          run_stop=_dt(*rm.group(5, 6, 7, 8)), index=p["index"],
                                          fn_start=p["fn_start"], fn_stop=p["fn_stop"],
                                          values_path=None, time_path=None, values_size=0,
                                          time_size=0, dup=0, name_mismatch=False))
            col = "values" if p["kind"] == "Values" else "time"
            if r[f"{col}_path"] is not None:
                r["dup"] += 1
            r[f"{col}_path"] = os.path.join(rd, fn)
            r[f"{col}_size"] = os.path.getsize(os.path.join(rd, fn))
            if (p["fn_start"], p["fn_stop"]) != (r["fn_start"], r["fn_stop"]):
                r["name_mismatch"] = True
    df = pd.DataFrame(list(rows.values())).sort_values(["run", "index"]).reset_index(drop=True)
    df["paired"] = df.values_path.notna() & df.time_path.notna()
    df["n_samples"] = np.where(df.paired, df.values_size // 4, 0)
    df["paired"] &= (df.values_size // 4 == df.time_size // 8)
    df["truncated"] = df.n_samples < SAMPLES_PER_CHUNK
    edge = [(_edge_stats(r.time_path, r.time_size) if r.paired else (np.nan,) * 4) for r in df.itertuples()]
    df[["t_first", "t_last", "base_dt", "n_missing_est"]] = pd.DataFrame(edge, index=df.index)
    df["t_first"] = df.t_first.astype("int64"); df["t_last"] = df.t_last.astype("int64")
    df["value_first"] = [(_first_value(p) if isinstance(p, str) else -1) for p in df.values_path]
    # exact dropout counts from the full scan (full_scan.py), when available
    fsc = os.path.join(HERE, "full_scan_chunks.parquet")
    df["n_missing"] = np.nan
    if os.path.exists(fsc):
        fx = pd.read_parquet(fsc)[["run", "index", "n_missing", "n_drop_events", "max_gap_slots"]]
        df = df.drop(columns="n_missing").merge(fx, on=["run", "index"], how="left")
    df["missing_exact"] = df.n_missing.notna()
    df["n_missing"] = df.n_missing.fillna(df.n_missing_est)
    # slot bookkeeping + wall-clock model per run
    df["interchunk_slots"] = 1
    df["slot_start"] = 0
    df["wall_start"] = pd.NaT
    df["fs_local"] = np.nan
    df["wall_resid_s"] = np.nan
    for run, g in df.groupby("run"):
        ii = g.index.values
        base = g.base_dt.values
        prev_last = np.roll(g.t_last.values, 1)
        ic = np.rint((g.t_first.values - prev_last) / np.where(np.isfinite(base), base, 418.0)).astype(np.int64)
        ic[0] = 0
        ic = np.maximum(ic, 1); ic[0] = 0
        span = (g.n_samples.values - 1) + np.rint(g.n_missing.fillna(0).values).astype(np.int64)
        ss = np.cumsum(np.concatenate([[0], span[:-1]])) + np.cumsum(ic)
        df.loc[ii, "interchunk_slots"] = np.maximum(ic, 1) if len(ic) else ic
        df.loc[ii[0], "interchunk_slots"] = 0
        df.loc[ii, "slot_start"] = ss
        wall = (g.fn_start - g.fn_start.iloc[0]).dt.total_seconds().values
        fit_ok = ~g.truncated.values | (np.arange(len(g)) < len(g) - 1)
        kx, ky = _fit_wall_clock(ss[fit_ok], wall[fit_ok])
        pred = np.interp(ss, kx, ky, left=np.nan, right=np.nan)
        # extrapolate linearly at the ends (e.g. truncated last chunk)
        lo = ss < kx[0]; hi = ss > kx[-1]
        pred[lo] = ky[0] + (ss[lo] - kx[0]) * (ky[1] - ky[0]) / (kx[1] - kx[0])
        pred[hi] = ky[-1] + (ss[hi] - kx[-1]) * (ky[-1] - ky[-2]) / (kx[-1] - kx[-2])
        seg = np.clip(np.searchsorted(kx, ss, side="right") - 1, 0, len(kx) - 2)
        fs = (kx[seg + 1] - kx[seg]) / (ky[seg + 1] - ky[seg])
        df.loc[ii, "wall_start"] = g.fn_start.iloc[0] + pd.to_timedelta(pred, unit="s")
        df.loc[ii, "fs_local"] = fs
        df.loc[ii, "wall_resid_s"] = wall - pred
    df["wall_start"] = pd.to_datetime(df.wall_start)
    df.attrs["knots"] = "piecewise-linear, 40-chunk knots"
    if cache:
        df.to_parquet(INDEX_PATH)
    return df


# --------------------------------------------------------------------------- chunk loading
def read_raw(values_path, time_path):
    """Raw arrays exactly as stored: (t uint64 device ticks, v uint32 ADC codes)."""
    v = np.fromfile(values_path, VALUES_DTYPE)
    t = np.fromfile(time_path, TIME_DTYPE)
    if len(v) != len(t):
        raise ValueError(f"length mismatch {len(v)} values vs {len(t)} timestamps")
    if len(v) and int(v.max()) >> ADC_BITS:
        raise ValueError("value uses bits above 24 -- unexpected format")
    return t, v


_FS_CACHE: dict | None = None


def chunk_fs(time_path) -> float:
    """fs_local (Hz) of the chunk from the cached index; FS_NOMINAL_EFFECTIVE (with a warning)
    if the chunk is not in the index."""
    global _FS_CACHE
    if _FS_CACHE is None:
        if os.path.exists(INDEX_PATH):
            ix = list_chunks()
            _FS_CACHE = dict(zip(ix.time_path, ix.fs_local))
        else:
            _FS_CACHE = {}
    fs = _FS_CACHE.get(time_path)
    if fs is None or not np.isfinite(fs):
        import warnings
        warnings.warn(f"{os.path.basename(time_path)} not in chunk index; using fs={FS_NOMINAL_EFFECTIVE}")
        return FS_NOMINAL_EFFECTIVE
    return float(fs)


def load_chunk(values_path, time_path, fs: float | None = None, time_base: str = "slots",
               return_info: bool = False):
    """Load one chunk.

    fs: sampling rate for the slot grid; default None = this chunk's fitted fs_local from the
        index (2407-2421 Hz). Pass a number only to override.
    time_base='slots' (default): t = slot/fs seconds from the first sample, where slot is the
        reconstructed ADC sample index (gaps appear as jumps of >1 slot). Uniform grid.
    time_base='device': t = (ticks - ticks[0]) * 1e-6 (device clock, biased by +1..+5 %).
    values: float64 raw 24-bit ADC counts (unsigned code). Physical units (V, T) are not
        determinable: Vref and sensor transfer function are not in the config files.
    return_info=True adds a dict(slots, base_dt, gaps=[(sample_idx, n_missing)], ticks).
    """
    t, v = read_raw(values_path, time_path)
    slots, base = slots_from_timestamps(t)
    if fs is None:
        fs = chunk_fs(time_path)
    if time_base == "slots":
        ts = slots / float(fs)
    elif time_base == "device":
        ts = (t.astype(np.int64) - int(t[0])) * 1e-6
    else:
        raise ValueError(time_base)
    vals = v.astype(np.float64)
    if not return_info:
        return ts, vals
    inc = np.diff(slots)
    g = np.flatnonzero(inc > 1)
    info = dict(slots=slots, base_dt=base, ticks=t, gaps=list(zip(g + 1, inc[g] - 1)),
                n_missing=int(inc.sum() - (len(inc))), fs=float(fs))
    return ts, vals, info


@dataclass
class Span:
    t: np.ndarray                 # datetime64[ns], estimated PC wall-clock time per sample
    values: np.ndarray            # float64 raw ADC counts
    slot: np.ndarray              # int64 global ADC slot within its run (uniform grid index)
    run: np.ndarray               # int8 run number per sample (0..3)
    fs: float                     # sample-count-weighted mean fitted rate over the span (Hz)
    gaps: pd.DataFrame = field(default_factory=pd.DataFrame)  # start, end, duration_s, n_missing, kind

    def to_frame(self):
        return pd.DataFrame({"t": self.t, "value": self.values, "slot": self.slot, "run": self.run})


def load_span(root: str = DEFAULT_ROOT, start=None, end=None, index: pd.DataFrame | None = None,
              min_gap_report_samples: int = 1) -> Span:
    """Load and stitch all samples with estimated wall time in [start, end).

    Time axis: t = wall_start(chunk) + (slot - slot_start(chunk)) / fs_local(chunk) -- the ADC
    grid anchored to the fitted PC clock (anchor uncertainty ~0.5 s absolute, <1 ms relative
    within a chunk). Gaps (dropouts inside chunks, between chunks, between runs / outside
    data) are returned in .gaps; no interpolation is done.
    """
    idx = list_chunks(root) if index is None else index
    start = pd.Timestamp(start); end = pd.Timestamp(end)
    idx = idx[idx.paired]
    wall_end = idx.wall_start + pd.to_timedelta((idx.n_samples + idx.n_missing) / idx.fs_local, unit="s")
    sel = idx[(wall_end > start) & (idx.wall_start < end)]
    runs = sorted(idx.run.unique())
    T, V, S, R, gaps, fsl = [], [], [], [], [], []
    prev = None   # (run, index, exact slot of last sample) -- exact stitching of contiguous chunks
    for r in sel.itertuples():
        _, v, info = load_chunk(r.values_path, r.time_path, fs=r.fs_local, return_info=True)
        s0 = r.slot_start
        if prev is not None and prev[0] == r.run and prev[1] == r.index - 1:
            s0 = prev[2] + int(r.interchunk_slots)      # exact, from device ticks at the join
        slot = s0 + info["slots"]
        prev = (r.run, r.index, int(slot[-1]))
        t = np.datetime64(r.wall_start, "ns") + (((slot - r.slot_start) / r.fs_local) * 1e9).astype("timedelta64[ns]")
        m = (t >= np.datetime64(start, "ns")) & (t < np.datetime64(end, "ns"))
        if not m.any():
            continue
        for (k, nm) in info["gaps"]:
            if nm >= min_gap_report_samples and m[k] and m[k - 1]:
                gaps.append(dict(start=t[k - 1], end=t[k], n_missing=nm, kind="intra-chunk dropout"))
        T.append(t[m]); V.append(v[m]); S.append(slot[m]); R.append(np.full(m.sum(), runs.index(r.run), np.int8))
        fsl.append((r.fs_local, int(m.sum())))
    if not T:
        gaps.append(dict(start=start, end=end, n_missing=np.nan, kind="no data"))
        return Span(np.array([], "datetime64[ns]"), np.array([]), np.array([], np.int64), np.array([], np.int8),
                    float("nan"), pd.DataFrame(gaps))
    t = np.concatenate(T); v = np.concatenate(V); s = np.concatenate(S); rr = np.concatenate(R)
    # boundaries between chunks: detect slot jumps / run changes / overlaps
    ends = np.cumsum([len(x) for x in T])[:-1]
    for e in ends:
        if rr[e] != rr[e - 1]:
            gaps.append(dict(start=t[e - 1], end=t[e], n_missing=np.nan, kind="between runs"))
        elif s[e] - s[e - 1] > 1:
            gaps.append(dict(start=t[e - 1], end=t[e], n_missing=int(s[e] - s[e - 1] - 1), kind="inter-chunk"))
    # enforce monotone time at chunk joins (anchor jitter could create tiny overlaps)
    bad = np.flatnonzero(np.diff(t) <= np.timedelta64(0, "ns"))
    if len(bad):
        keep = np.ones(len(t), bool)
        for i in bad:
            keep[i + 1] = False
            gaps.append(dict(start=t[i], end=t[i + 1], n_missing=1, kind="dropped at join (non-monotone time)"))
        t, v, s, rr = t[keep], v[keep], s[keep], rr[keep]
    if t[0] - np.datetime64(start, "ns") > np.timedelta64(1, "s"):
        gaps.append(dict(start=start, end=t[0], n_missing=np.nan, kind="no data"))
    if np.datetime64(end, "ns") - t[-1] > np.timedelta64(1, "s"):
        gaps.append(dict(start=t[-1], end=end, n_missing=np.nan, kind="no data"))
    G = pd.DataFrame(gaps, columns=["start", "end", "n_missing", "kind"])
    if len(G):
        G["start"] = pd.to_datetime(G.start); G["end"] = pd.to_datetime(G.end)
        G["duration_s"] = (G.end - G.start).dt.total_seconds()
        G = G.sort_values("start").reset_index(drop=True)
    fs_w = float(np.average([f for f, _ in fsl], weights=[n for _, n in fsl]))   # sample-weighted
    return Span(t, v, s, rr, fs_w, G)


# --------------------------------------------------------------------------- self-test
def _selftest(root=DEFAULT_ROOT):
    import time
    t0 = time.time()
    idx = list_chunks(root, refresh=not os.path.exists(INDEX_PATH))
    checks = []

    def check(name, cond):
        checks.append((name, bool(cond)))
        print(("PASS " if cond else "FAIL ") + name)

    check("4 runs", idx.run.nunique() == 4)
    check("every chunk paired (Values & Time, equal sample counts)", idx.paired.all())
    check("no duplicate (run,index)", not idx.duplicated(["run", "index"]).any() and (idx.dup == 0).all())
    check("filename Start/Stop identical for Values & Time", not idx.name_mismatch.any())
    check("indices contiguous 0..N-1 per run",
          all((g["index"].values == np.arange(len(g))).all() for _, g in idx.groupby("run")))
    check("only the final chunk of a run may be truncated",
          all(not g.truncated.values[:-1].any() for _, g in idx.groupby("run")))
    check("timestamps strictly increasing across chunks",
          all((g.t_first.values[1:] > g.t_last.values[:-1]).all() for _, g in idx.groupby("run")))
    fgap = np.concatenate([((g.fn_start.values[1:] - g.fn_stop.values[:-1]) / np.timedelta64(1, "s")).astype(int)
                           for _, g in idx.groupby("run")])
    check("filename start[i+1] - stop[i] in {0,1} s", set(np.unique(fgap).tolist()) <= {0, 1})
    check("fitted wall-clock residual |r| < 2.5 s (1 s filename quantisation) for all full chunks",
          idx.loc[~idx.truncated, "wall_resid_s"].abs().max() < 2.5)
    check("fitted fs within 2400..2430 Hz", idx.fs_local.between(2400, 2430).all())
    check("first values are 24-bit and far from rails",
          (idx.value_first < 2 ** 24).all() and idx.value_first.between(2 ** 23 + 2 ** 20, 2 ** 24 - 2 ** 20).all())
    # full load of a few chunks
    rng = np.random.default_rng(0)
    pick = idx[~idx.truncated].iloc[rng.choice((~idx.truncated).sum(), 6, replace=False)]
    for r in pick.itertuples():
        ts, v, info = load_chunk(r.values_path, r.time_path, return_info=True)
        tag = f"{r.run[4:12]}#{r.index}"
        check(f"{tag}: 100000 samples, top byte 0", len(v) == 100000 and v.max() < 2 ** 24)
        check(f"{tag}: slots strictly increasing", (np.diff(info["slots"]) >= 1).all())
        check(f"{tag}: index n_missing == recomputed ({r.n_missing:.0f} vs {info['n_missing']}; edge est {r.n_missing_est:.0f})",
              r.n_missing == info["n_missing"])
        check(f"{tag}: load_chunk default fs == fs_local ({info['fs']:.3f})", abs(info["fs"] - r.fs_local) < 1e-9)
        tick = info["ticks"].astype(np.int64); sl = info["slots"]
        res = np.concatenate([(tick[k:k+5000] - np.polyval(np.polyfit(sl[k:k+5000], tick[k:k+5000], 1), sl[k:k+5000]))
                              / info["base_dt"] for k in range(0, len(tick), 5000)])
        check(f"{tag}: slot grid explains ticks (resid {res.std():.3f} slot rms, {np.mean(np.abs(res) > 0.5):.1e} > 0.5 slot)",
              res.std() < 0.05 and np.mean(np.abs(res) > 0.5) < 1e-3)
    # span stitching across a chunk boundary
    r = pick.iloc[0]
    sp = load_span(root, r.wall_start + pd.Timedelta(seconds=30), r.wall_start + pd.Timedelta(seconds=90), index=idx)
    dts = np.diff(sp.t).astype(np.int64) / 1e9
    check(f"load_span: monotone time, median dt {np.median(dts)*1e6:.1f} us ~ 1/fs",
          (dts > 0).all() and abs(np.median(dts) - 1 / sp.fs) < 1e-6)
    check("load_span: sample count + missing ~ duration*fs",
          abs(len(sp.values) + sp.gaps.n_missing.fillna(0).sum() - 60 * sp.fs) < 0.01 * 60 * sp.fs)
    # span across a run boundary reports it
    sp2 = load_span(root, "2025-02-11 16:29:00", "2025-02-12 14:43:00", index=idx)
    check("load_span across runs reports a >20 h 'no data'/'between runs' gap",
          (sp2.gaps.duration_s > 20 * 3600).any())
    check("index: exact n_missing available for every chunk (full_scan.py)", idx.missing_exact.all())
    check("load_span: no samples dropped at chunk joins",
          not (sp.gaps.kind == "dropped at join (non-monotone time)").any())
    ok = all(c for _, c in checks)
    print(f"\n{sum(c for _, c in checks)}/{len(checks)} checks passed in {time.time()-t0:.1f}s")
    return ok


if __name__ == "__main__":
    import sys
    if "--refresh" in sys.argv:
        list_chunks(refresh=True)
    sys.exit(0 if _selftest() else 1)
