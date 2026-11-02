"""v2 end-to-end run of the recommended chain (preproc.py) + detector (det2.py) on
  (A) the 11 contiguous 2-h windows, and (B) every 8th chunk of every run (each chunk its own span).
Detection path: 5 Hz LP -> 50 Hz -> 20 s running median (2-pass, events masked) -> robust sigma
(60 s blocks) -> hysteresis at 5, 6 and 8 sigma.
Saves  data/v2_events.parquet  (all thresholds, one row per event, with 'thr')
       data/v2_snippets.pkl    (8 sigma events fully inside one dropout-free segment: mains-free raw at
                                2416 Hz for [a-6 s, e+6 s], plus the 50 Hz baseline of that stretch)
       data/v2_quiet.pkl       (quiet 60 s mains-free raw stretches for injection tests)
       data/v2_spans.parquet   (per span: hours, sigma, n_lost, day/hour, run)
Usage: OMP_NUM_THREADS=1 python run_pipeline.py [STEP]"""
import os, sys
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "1")
import pickle
import numpy as np, pandas as pd
from datetime import datetime, timedelta
from multiprocessing import Pool
import io_utils as io, preproc as P, det2 as D

STEP = int(sys.argv[1]) if len(sys.argv) > 1 else 8
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
WIN = [("20250205 0700", "weekday AM"), ("20250205 1630", "weekday PM"), ("20250206 0200", "night"),
       ("20250208 1100", "Saturday"), ("20250213 0700", "weekday AM"), ("20250213 1630", "weekday PM"),
       ("20250219 0700", "weekday AM"), ("20250219 1230", "weekday midday"), ("20250225 0700", "weekday AM"),
       ("20250225 1630", "weekday PM"), ("20250226 0200", "night")]
THRS = (5.0, 6.0, 8.0)
SNIP_S = 6.0


def process(job):
    kind, label, chunks = job
    v = np.concatenate([np.fromfile(c["vpath"], "<u4") for c in chunks]).astype(float)
    t = np.concatenate([np.fromfile(c["tpath"], "<u8") for c in chunks]).astype(np.int64)
    t0 = chunks[0]["start"]
    C = P.clean_span(v, t, lp_list=(P.DETECT_LP_HZ,), keep_raw=True)
    y = C["g50"][P.DETECT_LP_HZ]
    m = C["mask50"]
    # pass 1
    b1 = P.baseline(y, m)
    r1 = y - b1
    sig1 = P.robust_sigma(r1, m)
    ev1 = D.detect(r1, sig1, m, k_hi=5.0)
    em = np.zeros(len(y), bool)
    for a, e in ev1:
        em[max(0, a - 100):e + 100] = True
    # pass 2: baseline with events bridged
    b = P.baseline(y, m, event_mask=em)
    r = y - b
    sig = P.robust_sigma(r, m)
    rows = []
    snippets = []
    for thr in THRS:
        for a, e in D.detect(r, sig, m, k_hi=thr):
            d = D.describe(r, a, e, sig, m)
            d1 = D.shape(r1[a:e])  # same extent, single-pass baseline (for the distortion check)
            d.update(kind=kind, span=label, thr=thr, a=a, e=e, t_s=a / 50.0,
                     start=t0 + timedelta(seconds=a / 50.0), run=chunks[0]["run"],
                     amax_1pass=d1["amax"], span_len_s=len(y) / 50.0)
            rows.append(d)
            if thr == 8.0:
                # raw snippet if [a-6s, e+6s] lies in one segment
                sa = int((a / 50.0 - SNIP_S) * P.FS_ADC)
                sb = int((e / 50.0 + SNIP_S) * P.FS_ADC)
                for g0, xr in C["raw"]:
                    if g0 <= sa and sb <= g0 + len(xr):
                        ja, jb = max(0, a - int(SNIP_S * 50)), e + int(SNIP_S * 50)
                        snippets.append(dict(span=label, a=a, e=e, x=xr[sa - g0:sb - g0].copy(),
                                             base50=b[ja:jb].astype(np.float32), off50=ja, sa_s=sa / P.FS_ADC,
                                             sig=float(np.median(sig[a:e]))))
                        break
    # quiet stretches (contiguous windows only): 60 s with no 5-sigma event and no mask
    quiet = []
    if kind == "window":
        busy = m.copy()
        for a, e in D.detect(r, sig, m, k_hi=5.0):
            busy[max(0, a - 250):e + 250] = True
        L = 3000
        cand = [c for c in range(0, len(y) - L, L) if not busy[c:c + L].any()]
        rng = np.random.default_rng(len(label))
        for c in rng.permutation(cand)[:4]:
            sa, sb = int(c / 50 * P.FS_ADC), int((c + L) / 50 * P.FS_ADC)
            for g0, xr in C["raw"]:
                if g0 <= sa and sb <= g0 + len(xr):
                    quiet.append(dict(span=label, x=xr[sa - g0:sb - g0].copy(), sig50=float(np.median(sig[c:c + L]))))
                    break
    span = dict(kind=kind, span=label, run=chunks[0]["run"], start=t0, hours=len(y) / 50 / 3600,
                masked_frac=float(m.mean()), sig_med=float(np.nanmedian(sig)), n_lost=C["n_lost"],
                day=t0.date(), hour=t0.hour)
    return rows, snippets, quiet, span


if __name__ == "__main__":
    Cs = io.list_chunks()
    jobs = []
    for ts, lab in WIN:
        a = datetime.strptime(ts, "%Y%m%d %H%M")
        sel = [c for c in Cs if a <= c["start"] < a + timedelta(hours=2)]
        jobs.append(("window", f"{ts} {lab}", sel))
    for run in sorted(set(c["run"] for c in Cs)):
        rc = [c for c in Cs if c["run"] == run and c["tpath"]]
        for c in rc[::STEP]:
            if os.path.getsize(c["vpath"]) == 400000:
                jobs.append(("chunk", f"r{run}_{c['idx']:06d}", [c]))
    print(len(jobs), "spans", flush=True)
    rows, snips, quiet, spans = [], [], [], []
    with Pool(5) as p:
        for i, (r_, s_, q_, sp) in enumerate(p.imap_unordered(process, jobs, chunksize=4)):
            rows += r_; snips += s_; quiet += q_; spans.append(sp)
            if i % 500 == 0:
                print(i, flush=True)
    ev = pd.DataFrame(rows)
    ev.to_parquet(os.path.join(OUT, "v2_events.parquet"))
    pd.DataFrame(spans).to_parquet(os.path.join(OUT, "v2_spans.parquet"))
    with open(os.path.join(OUT, "v2_snippets.pkl"), "wb") as f:
        pickle.dump(snips, f)
    with open(os.path.join(OUT, "v2_quiet.pkl"), "wb") as f:
        pickle.dump(quiet, f)
    print(ev.groupby(["kind", "thr"]).size(), len(snips), "snippets", len(quiet), "quiet")
