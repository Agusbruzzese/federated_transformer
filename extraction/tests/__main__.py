"""Self-tests: python -m extraction.tests   (plain asserts; exits non-zero on failure)."""
from __future__ import annotations

import os
import sys
import time
import traceback
import warnings

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "1")
warnings.filterwarnings("ignore", category=RuntimeWarning)

import numpy as np
import pandas as pd
from scipy import signal

from .. import params as PR
from ..blockio import SlotClock, chunk_index, load_block
from ..featpath import feature_path, resample_to_grid
from ..morph import MORPH_NAMES, morph_feats
from ..pipeline import STEP, detect_block, process_block
from ..run import dedupe
from .._vendor import bsd_io
from .._vendor import features as DF

IX = chunk_index()
RUNS = sorted(IX.run.unique())
RESULTS = []


def ixr_of(rid):
    return IX[IX.run == RUNS[rid]].sort_values("index").reset_index(drop=True)


def check(name, cond, detail=""):
    RESULTS.append((name, bool(cond)))
    print(("PASS " if cond else "FAIL ") + name + (f"  [{detail}]" if detail else ""), flush=True)


# ------------------------------------------------------------------ 1. 24-bit loader (+ legacy int16 regression)
def test_loader_24bit():
    rng = np.random.default_rng(1)
    ixr = ixr_of(0)
    found = None
    for pos in rng.choice(len(ixr) - 1, 80, replace=False):
        r = ixr.iloc[pos]
        raw = np.fromfile(r.values_path, "<u4")
        # signed int16 of the low word wraps when the code crosses 65536*k + 32768 (0x7FFF -> 0x8000)
        if len(np.unique((raw.astype(np.int64) + 32768) >> 16)) > 1:
            found = pos
            break
    check("found a chunk whose code crosses an int16 wrap point (65536k + 32768)", found is not None)
    B = load_block(ixr, found, found + 1)
    raw = np.fromfile(ixr.iloc[found].values_path, "<u4")
    v = B["v"]
    check("loader returns the full 24-bit code (== uint32 & 0xFFFFFF, top byte 0)",
          np.array_equal(v, (raw & 0xFFFFFF).astype(float)) and (raw >> 24).max() == 0)
    check("codes are 24-bit scale (> 2^23), far above the int16 range", v.min() > 2 ** 23 and v.max() < 2 ** 24,
          f"{v.min():.0f}..{v.max():.0f}")
    leg = DF.legacy_read_int16(raw)
    jl = np.abs(np.diff(leg)).max(); jv = np.abs(np.diff(v)).max()
    check("REGRESSION legacy int16 bug: legacy reader jumps ~65536 here, the loader does not",
          jl > 30000 and jv < 30000, f"max step legacy {jl:.0f}, loader {jv:.0f}")
    # legacy reader bytes-level equivalence (documents what the bug was)
    leg2 = np.frombuffer(raw.tobytes(), "<i2").astype(np.float64)[0::2]
    check("legacy int16 reader == low 16 bits signed (bug reproduced exactly)", np.array_equal(leg, leg2))


# ------------------------------------------------------------------ 2. effective-fs bookkeeping
def test_fs():
    rng = np.random.default_rng(2)
    x = rng.standard_normal(50000).cumsum()
    j, y = resample_to_grid(x, 0, padtype="constant")
    y0 = signal.resample_poly(x, PR.UP, PR.DOWN)
    n = min(len(y), len(y0))
    check("resample_to_grid(slot_a=0, constant) == scipy resample_poly(25,1208)",
          j == 0 and np.allclose(y[:n], y0[:n], atol=1e-9 * np.abs(y0).max()) and abs(len(y) - len(y0)) <= 1)
    # phase-exact placement on the run-global grid, arbitrary segment starts
    fs, f = 2412.7, 1.3
    errs = []
    for sa in rng.integers(0, 10 ** 7, 5):
        s = np.arange(sa, sa + 60000)
        xs = np.sin(2 * np.pi * f * s / fs)
        j, y = resample_to_grid(xs, int(sa))
        jj = j + np.arange(len(y))
        truth = np.sin(2 * np.pi * f * jj * STEP / fs)
        errs.append(np.abs(y - truth)[20:-20].max())
    check("phase-exact resampling: sample j sits at slot j*1208/25 (max err < 1e-3 of a 1.3 Hz sine)",
          max(errs) < 1e-3, f"max err {max(errs):.1e}")
    # a 1.3 Hz sine resampled from each rate has exactly fs*25/1208 samples per second of true time
    for fsl in (2407.1, 2416.0, 2421.4):
        n_per_s = fsl * PR.UP / PR.DOWN
        s = np.arange(0, int(fsl * 30))
        j, y = resample_to_grid(np.sin(2 * np.pi * f * s / fsl), 0)
        # zero crossings per true second on the output, using the recorded effective rate
        zc = np.flatnonzero(np.diff(np.sign(y[50:-50])) > 0)
        per = np.diff(zc).mean() / n_per_s
        check(f"fs_local {fsl}: period measured with fs_eff={n_per_s:.3f} Hz is 1/1.3 s", abs(per - 1 / f) < 2e-3,
              f"{per:.4f} s")
    # SlotClock on a synthetic 2-chunk index
    ixs = pd.DataFrame(dict(slot_start=[0, 100000], fs_local=[2410.0, 2420.0], index=[0, 1],
                            wall_start=pd.to_datetime(["2025-01-01 00:00:00", "2025-01-01 00:00:41.4937759"], format="ISO8601")))
    ck = SlotClock(ixs)
    m, lo, hi = ck.fs_eff_mean(75000, 125000)
    exp = (2410 * 25000 + 2420 * 25000) / 50000 * PR.UP / PR.DOWN
    check("fs_eff over a window spanning two fs_local = slot-weighted mean; min/max recorded",
          abs(m - exp) < 1e-9 and abs(lo - 2410 * 25 / 1208) < 1e-9 and abs(hi - 2420 * 25 / 1208) < 1e-9)
    wt = ck.wall_time(np.array([100000 + 2420]))[0]
    check("wall time = wall_start(chunk) + (slot - slot_start)/fs_local",
          abs((wt - np.datetime64("2025-01-01T00:00:42.4937759")) / np.timedelta64(1, "us")) < 1)
    # real index: effective rates in range
    fe = IX.fs_local * PR.UP / PR.DOWN
    check("all effective rates within 49.8-50.15 Hz (fs_local 2407-2421.5)", fe.between(49.8, 50.15).all(), f"{fe.min():.3f}-{fe.max():.3f}")
    # morph time features on the true time base
    ok = True; det = []
    for fse in (49.83, 50.0, 50.11):
        t = (np.arange(PR.WIN_N) - PR.WIN_N // 2) / fse
        sd = 1.0 / 2.3548                      # FWHM 1.0 s gaussian (bandwidth << 5 Hz)
        g = 10000 * np.exp(-0.5 * (t / sd) ** 2)
        m = dict(zip(MORPH_NAMES, morph_feats(g, fse)))
        d = 1e4 * (-t / sd) * np.exp(0.5 - 0.5 * (t / sd) ** 2)       # derivative of gaussian: t_pp = 2*sd
        md = dict(zip(MORPH_NAMES, morph_feats(d, fse)))
        # dur50/t_pp are sample-quantised: up to 1 sample per side (2/fs); t_spread is continuous
        ok &= abs(m["dur50_main"] - 1.0) <= 2 / fse and abs(abs(md["t_pp"]) - 2 * sd) <= 2 / fse
        ok &= abs(m["t_spread"] - sd / np.sqrt(2)) < 0.003
        det.append(f"{fse}: FWHM {m['dur50_main']:.3f} t_pp {md['t_pp']:.3f} t_spread {m['t_spread']:.4f}")
    check("morph time features in true seconds at fs_eff (FWHM 1.0 s, t_pp 2sd, t_spread sd/sqrt2)", ok, "; ".join(det))
    # D's original function would mis-smooth at ~50 Hz (its 5 Hz filter is fixed for 100 Hz): documents E2
    t = (np.arange(PR.WIN_N) - PR.WIN_N // 2) / 50.0
    narrow = 1e4 * np.exp(-0.5 * (t / 0.08) ** 2)
    a_e = morph_feats(narrow, 50.0)[0]; a_d = DF.morph_feats(narrow, fs=50.0)[0]
    check("E2: D's fixed-100 Hz smoother used at 50 Hz attenuates a 0.19 s pulse more than the fs-correct one",
          a_d < a_e - 0.02, f"log_amp E {a_e:.3f} vs D-at-50Hz {a_d:.3f}")


# ------------------------------------------------------------------ 3. synthetic event injected into real background
def pulse(slot, s0, fsl, A, tau):
    t = (slot - s0) / fsl
    return A * (t / tau) * np.exp(0.5 - 0.5 * (t / tau) ** 2)       # bipolar, |peak| = A at t = +-tau


def test_injection():
    ixr = ixr_of(1)
    rng = np.random.default_rng(3)
    ok_all, det = True, []
    n_ok = 0; n_try = 0
    for c0 in (700, 2100, 3500):
        job = dict(run=RUNS[1], run_id=1, ixr=ixr, core=(c0, c0 + 4), ctx=3)
        B = load_block(ixr, c0 - 3, c0 + 7)
        v, slot = B["v"], B["slot"]
        ev = detect_block(v, slot)
        cs = np.array([e["slot_centroid_t"] for e in ev])
        s_lo, s_hi = int(ixr.slot_start[c0]), int(ixr.slot_start[c0 + 4])
        # a place in the core >= 15 s from any existing detection; one site forced onto a chunk boundary
        cands = [int(ixr.slot_start[c0 + 2])] + list(rng.integers(s_lo + 40000, s_hi - 40000, 200))
        site = next((s for s in cands if len(cs) == 0 or np.abs(cs - s).min() > 15 * 2416), None)
        if site is None:
            continue
        fsl = float(ixr.fs_local[c0])
        for A, tau in ((15000, 0.8), (3000, 0.5)):
            n_try += 1
            vi = v + pulse(slot, site, fsl, A, tau)
            res = process_block(job, vi, slot)
            df = pd.DataFrame(res["rows"])
            hit = df[(df.slot_centroid - site).abs() < 3 * 2416] if len(df) else df
            if len(hit) != 1:
                ok_all = False; det.append(f"site {site}: {len(hit)} matches"); continue
            h = hit.iloc[0]; k = hit.index[0]
            cen_err = (h.slot_centroid - site) / fsl
            win_centre = (h.win_start_j + PR.WIN_N // 2) * STEP
            wc_err = (win_centre - site) / fsl
            w = res["W"][k]
            amp_ok = abs(np.abs(w).max() / A - 1) < 0.15
            good = abs(cen_err) <= 0.5 and abs(wc_err) <= 0.5 and amp_ok and h.snr > 5
            n_ok += good; ok_all &= good
            det.append(f"A={A}: centroid err {cen_err:+.3f} s, window-centre err {wc_err:+.3f} s, "
                       f"peak {np.abs(w).max():.0f}, snr {h.snr:.0f}")
            if A == 15000:
                check("window shape (1050,), float32, finite; masks bool (1050,)",
                      w.shape == (PR.WIN_N,) and w.dtype == np.float32 and np.isfinite(w).all()
                      and res["EM"][k].dtype == bool and res["RM"][k].shape == (PR.WIN_N,))
                em = res["EM"][k]
                check("evmask marks [onset, offset] and contains the window centre",
                      em[PR.WIN_N // 2] and em.sum() == h.ev_i1 - h.ev_i0)
    check(f"synthetic events injected into real background recovered, window centred within +-0.5 s ({n_ok}/{n_try})",
          ok_all and n_try >= 4, "; ".join(det))


# ------------------------------------------------------------------ 4. block overlaps: no duplicates, same events
def test_overlap():
    ixr = ixr_of(2)
    a0, n, k = 1000, 60, 10
    big = process_block(dict(run=RUNS[2], run_id=2, ixr=ixr, core=(a0, a0 + n), ctx=3))
    Db = pd.DataFrame(big["rows"] + big["excl"])
    parts = []
    for c in range(a0, a0 + n, k):
        r = process_block(dict(run=RUNS[2], run_id=2, ixr=ixr, core=(c, c + k), ctx=3))
        parts += r["rows"] + r["excl"]
    Ds = pd.DataFrame(parts).sort_values("slot_centroid").reset_index(drop=True)
    keep = dedupe(Ds, ixr)
    Dk = Ds[keep].reset_index(drop=True)
    o = (Dk.slot_onset.to_numpy()[1:] < Dk.slot_offset.to_numpy()[:-1])
    check("no two kept events overlap in [onset, offset] after block stitching",
          not o.any(), f"{(~keep).sum()} duplicate(s) removed from {len(Ds)}")
    def nearest(a, b):
        return np.array([np.abs(b - x).min() for x in a]) / 2416 if len(a) and len(b) else np.full(len(a), np.inf)
    cb = Db.slot_centroid.to_numpy(); cs = Dk.slot_centroid.to_numpy()
    ds, db = nearest(cs, cb), nearest(cb, cs)
    un = sorted(np.r_[Dk.snr[ds >= 0.5], Db.snr[db >= 0.5]].round(1).tolist())
    hi_s, hi_b = Dk[Dk.snr >= 20], Db[Db.snr >= 20]
    hi_ok = (ds[Dk.snr.to_numpy() >= 20] < 0.5).all() and (db[Db.snr.to_numpy() >= 20] < 0.5).all()
    check("6 blocks (core 10 chunks, 5 internal boundaries) vs one 60-chunk block: every SNR>=20 event matched "
          "(centroid < 0.5 s), total count within 10 %",
          hi_ok and abs(len(Dk) - len(Db)) <= max(1, 0.10 * len(Db)),
          f"{len(Dk)} vs {len(Db)} events ({len(hi_s)} vs {len(hi_b)} SNR>=20); unmatched SNRs {un}")
    # unit check of the dedupe rule
    t = pd.DataFrame(dict(slot_centroid=[100., 105., 300.], slot_onset=[90., 95., 290.], slot_offset=[110., 115., 310.],
                          block_core0=[0, 1, 1]))
    fake = pd.DataFrame(dict(slot_start=[0, 104]))
    k = dedupe(t, fake)
    check("dedupe keeps exactly one copy, from the block owning the centroid midpoint (102.5 < 104 -> block 0)",
          k.tolist() == [True, False, True])


# ------------------------------------------------------------------ 5. dropout masking on a known chunk
def test_dropouts():
    gaps = pd.read_parquet(os.path.join(bsd_io.HERE, "full_scan_gaps.parquet"))
    ixr = ixr_of(0)
    run = RUNS[0]
    g = gaps[gaps.run == run]
    per = g.groupby("index").agg(n5=("n_missing", lambda s: (s >= 5).sum()), n14=("n_missing", lambda s: ((s >= 1) & (s <= 4)).sum()))
    pick = per[(per.n5 >= 3) & (per.n5 <= 12) & (per.n14 >= 1)].index
    ci = int(pick[len(pick) // 2]) if len(pick) else int(per[per.n5 >= 3].index[0])
    pos = int(np.flatnonzero(ixr["index"].to_numpy() == ci)[0])
    B = load_block(ixr, pos, pos + 1)
    v, slot = B["v"], B["slot"]
    clock = SlotClock(ixr)
    F = feature_path(v, slot, clock, return_debug=True)
    gc = g[g["index"] == ci]
    check(f"chunk {ci}: splits == A's exact dropouts >= 5 samples ({(gc.n_missing >= 5).sum()})",
          F["n_split"] == int((gc.n_missing >= 5).sum()), f"E {F['n_split']}")
    check(f"chunk {ci}: filled 1-4-sample dropouts == A's ({((gc.n_missing >= 1) & (gc.n_missing <= 4)).sum()})",
          F["n_filled"] == int(((gc.n_missing >= 1) & (gc.n_missing <= 4)).sum()), f"E {F['n_filled']}")
    lost = np.diff(slot) - 1
    J0 = F["J0"]; hole = F["hole"]; mask = F["mask"]
    ok_hole, ok_guard = True, True
    for b in np.flatnonzero(lost >= 5) + 1:
        inside = np.arange(int(np.ceil(slot[b - 1] / STEP + 1e-9)), int(np.floor(slot[b] / STEP - 1e-9)) + 1) - J0
        inside = inside[(inside >= 0) & (inside < len(hole))]
        ok_hole &= hole[inside].all()
        gz = np.arange(int(np.floor(slot[b - 1] / STEP - 0.15 * 50)), int(np.ceil(slot[b] / STEP + 0.15 * 50)) + 1) - J0
        gz = gz[(gz >= 0) & (gz < len(mask))]
        ok_guard &= mask[gz].all()
    check("every output sample inside a >=5-sample dropout is a bridged hole", ok_hole)
    check("flag mask covers +-0.15 s around every split", ok_guard)
    # nothing flagged far (> 0.2 s) from a split
    jj = (np.arange(len(mask)) + J0) * STEP
    gap_slots = np.r_[[slot[b - 1] for b in np.flatnonzero(lost >= 5) + 1], [slot[b] for b in np.flatnonzero(lost >= 5) + 1]]
    far = np.array([np.abs(gap_slots - s).min() for s in jj]) > 0.2 * 2416 if len(gap_slots) else np.ones(len(jj), bool)
    check("no sample flagged farther than 0.2 s from a split", not mask[far].any(), f"flagged frac {mask.mean():.4f}")
    check("feature series finite after repair", np.isfinite(F["r"]).all())
    # C's has_dropout flag agrees: detections overlapping a dropout are flagged
    ev = detect_block(v, slot)
    bad = [e for e in ev if not e["has_dropout"] and any(e["slot_onset"] <= s <= e["slot_offset"]
                                                         for s in slot[np.flatnonzero(lost >= 1) + 1])]
    check("C's has_dropout set on every detection whose extent contains a dropout", len(bad) == 0,
          f"{len(ev)} detections")


# ------------------------------------------------------------------ 6. stream edge -> truncated
def test_truncated():
    ixr = ixr_of(1)
    B = load_block(ixr, 0, 4)
    v, slot = B["v"], B["slot"]
    fsl = float(ixr.fs_local[0])
    site = int(slot[0] + 4.0 * fsl)
    res = process_block(dict(run=RUNS[1], run_id=1, ixr=ixr, core=(0, 1), ctx=3), v + pulse(slot, site, fsl, 15000, 0.8), slot)
    df = pd.DataFrame(res["rows"])
    hit = df[(df.slot_centroid - site).abs() < 2 * 2416]
    ok = len(hit) == 1 and bool(hit.iloc[0].truncated)
    if ok:
        k = hit.index[0]; no = int(hit.iloc[0].n_outside)
        ok = no > 0 and np.isfinite(res["W"][k]).all() and res["RM"][k][:no].all() and (res["W"][k][:no] == 0).all()
    check("event 4 s after run start: truncated=True, padded samples zero and flagged in repmask", ok)


def main():
    t0 = time.time()
    for f in (test_loader_24bit, test_fs, test_injection, test_overlap, test_dropouts, test_truncated):
        print(f"--- {f.__name__}", flush=True)
        try:
            f()
        except Exception:                # noqa: BLE001
            traceback.print_exc()
            RESULTS.append((f.__name__ + " raised", False))
    n_ok = sum(c for _, c in RESULTS)
    print(f"\n{n_ok}/{len(RESULTS)} checks passed in {time.time() - t0:.0f} s")
    sys.exit(0 if n_ok == len(RESULTS) else 1)


if __name__ == "__main__":
    main()
