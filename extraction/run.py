"""Entry point: python -m extraction.run --out data/events [--procs 16] [--runs 0,1,2,3] [--max-blocks N]

Outputs (per run r = 0..3, label = run start date):
  run{r}_{label}_meta.parquet   one row per kept event: metadata + 19 morph scalars (m_*, 21 s window)
                                + 19 on the central ~8 s crop (m8_*); column `row` = row in the arrays below
  run{r}_{label}_windows.npy    float32 [N, 1050]  baseline-subtracted feature-path window (counts)
  run{r}_{label}_evmask.npy     bool    [N, 1050]  True inside C's [onset, offset]
  run{r}_{label}_repmask.npy    bool    [N, 1050]  True where a sample is dropout-affected (hole bridged or
                                                   within 0.15 s of a split) or padded beyond a stream edge
  excluded.parquet              events with mask_frac > 0.2 (metadata only)
  blocks.parquet                per-block statistics and timings
  manifest.json                 code version, parameters, dataset fingerprint, counts, runtime
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import hashlib
import json
import os
import platform
import resource
import subprocess
import sys
import time

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import numpy as np
import pandas as pd

from . import params as PR
from .blockio import chunk_index, run_label
from .pipeline import make_blocks, process_block

PKG = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(PKG)
_IX = None


def _init():
    global _IX
    import warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    _IX = chunk_index()


def _work(arg):
    run, run_id, core = arg
    ixr = _IX[_IX.run == run].sort_values("index").reset_index(drop=True)
    return process_block(dict(run=run, run_id=run_id, ixr=ixr, core=core))


def dedupe(df: pd.DataFrame, ixr: pd.DataFrame) -> np.ndarray:
    """Keep-mask removing duplicate detections of one event by two adjacent blocks.
    df: events of one run sorted by slot_centroid. Two consecutive events from different blocks whose
    [onset, offset] overlap are one event; the copy kept is the one from the block whose core contains the
    midpoint of the two centroids."""
    keep = np.ones(len(df), bool)
    c = df.slot_centroid.to_numpy(); on = df.slot_onset.to_numpy(); off = df.slot_offset.to_numpy()
    blk = df.block_core0.to_numpy()
    s0 = ixr.slot_start.to_numpy()
    for i in range(len(df) - 1):
        j = i + 1
        if blk[i] != blk[j] and on[j] < off[i]:
            mid = 0.5 * (c[i] + c[j])
            owner = blk[j] if mid >= s0[blk[j]] else blk[i]
            keep[i if owner == blk[j] else j] = False
    return keep


def package_hash():
    h = hashlib.sha256()
    files = sorted(glob.glob(os.path.join(PKG, "**", "*.py"), recursive=True))
    for f in files:
        h.update(os.path.relpath(f, PKG).encode()); h.update(open(f, "rb").read())
    return h.hexdigest(), len(files)


def git_info():
    try:
        head = subprocess.check_output(["git", "-C", REPO, "rev-parse", "HEAD"], text=True).strip()
        st = subprocess.check_output(["git", "-C", REPO, "status", "--porcelain", "--", "extraction"], text=True)
        return dict(head=head, extraction_tracked_clean=(st.strip() == ""), status_extraction=st.strip()[:2000])
    except Exception as e:           # noqa: BLE001
        return dict(error=str(e))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/events")
    ap.add_argument("--procs", type=int, default=16)
    ap.add_argument("--runs", default="0,1,2,3")
    ap.add_argument("--max-blocks", type=int, default=0, help="debug: only the first N blocks per run")
    a = ap.parse_args(argv)
    out = os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)
    t0 = time.time(); started = dt.datetime.now().isoformat(timespec="seconds")
    ix = chunk_index()
    runs = sorted(ix.run.unique())
    sel = [int(x) for x in a.runs.split(",")]
    jobs = []
    for rid in sel:
        ixr = ix[ix.run == runs[rid]].sort_values("index").reset_index(drop=True)
        bl = make_blocks(ixr)
        if a.max_blocks:
            bl = bl[:a.max_blocks]
        jobs += [(runs[rid], rid, b) for b in bl]
    jobs.sort(key=lambda j: -(j[2][1] - j[2][0]))
    print(f"{len(jobs)} blocks, {a.procs} processes", flush=True)
    from multiprocessing import Pool
    res = []
    with Pool(a.procs, initializer=_init) as p:
        for k, r in enumerate(p.imap_unordered(_work, jobs, chunksize=1)):
            res.append(r)
            if k % 50 == 0:
                print(f"  {k + 1}/{len(jobs)} blocks, {time.time() - t0:.0f} s", flush=True)
    wall_pipeline = time.time() - t0
    res.sort(key=lambda r: (r["stats"]["run"], r["stats"]["core0"]))      # deterministic order
    ru = resource.getrusage(resource.RUSAGE_CHILDREN)
    counts = {}
    all_excl, all_stats = [], []
    for rid in sel:
        run = runs[rid]; lab = run_label(run)
        ixr = ix[ix.run == run].sort_values("index").reset_index(drop=True)
        R = [r for r in res if r["stats"]["run"] == run]
        rows = [x for r in R for x in r["rows"]]
        df = pd.DataFrame(rows)
        W = np.concatenate([r["W"] for r in R]); EM = np.concatenate([r["EM"] for r in R])
        RM = np.concatenate([r["RM"] for r in R])
        o = np.argsort(df.slot_centroid.to_numpy(), kind="stable")
        df = df.iloc[o].reset_index(drop=True); W, EM, RM = W[o], EM[o], RM[o]
        keep = dedupe(df, ixr)
        n_dup = int((~keep).sum())
        df = df[keep].reset_index(drop=True); W, EM, RM = W[keep], EM[keep], RM[keep]
        df["wall_time"] = pd.to_datetime(df.wall_time)
        df.insert(0, "event_id", [f"{lab}_{int(s):011d}" for s in df.slot_centroid])
        df.insert(1, "row", np.arange(len(df)))
        df["date"] = df.wall_time.dt.strftime("%Y-%m-%d")
        df["hour"] = df.wall_time.dt.hour.astype(int)
        df["snr_ge20"] = df.snr >= 20; df["snr_ge30"] = df.snr >= 30
        assert df.event_id.is_unique
        assert W.shape == (len(df), PR.WIN_N) and W.dtype == np.float32 and np.isfinite(W).all()
        base = os.path.join(out, f"run{rid}_{lab}")
        df.to_parquet(base + "_meta.parquet", index=False)
        np.save(base + "_windows.npy", W); np.save(base + "_evmask.npy", EM); np.save(base + "_repmask.npy", RM)
        ex = [x for r in R for x in r["excl"]]
        all_excl += ex
        all_stats += [r["stats"] for r in R]
        st = pd.DataFrame([r["stats"] for r in R])
        counts[f"run{rid}_{lab}"] = dict(
            hours=float(st.core_hours.sum()), n_kept=len(df), n_excluded_mask_frac=len(ex), n_duplicates_removed=n_dup,
            n_snr_lt5=int((df.snr < 5).sum()), n_snr_ge5=int((df.snr >= 5).sum()),
            n_snr_ge20=int(df.snr_ge20.sum()), n_snr_ge30=int(df.snr_ge30.sum()),
            n_has_dropout=int(df.has_dropout.sum()), n_truncated=int(df.truncated.sum()))
        print(lab, counts[f"run{rid}_{lab}"], flush=True)
    ex = pd.DataFrame(all_excl)
    if len(ex):
        ex["wall_time"] = pd.to_datetime(ex.wall_time)
        ex = ex.sort_values(["run_id", "slot_centroid"]).reset_index(drop=True)
    ex.to_parquet(os.path.join(out, "excluded.parquet"), index=False)
    pd.DataFrame(all_stats).to_parquet(os.path.join(out, "blocks.parquet"), index=False)
    tot = {k: sum(c[k] for c in counts.values()) for k in next(iter(counts.values()))}
    ph, nf = package_hash()
    from ._vendor import bsd_io
    idx_md5 = hashlib.md5(open(bsd_io.INDEX_PATH, "rb").read()).hexdigest()
    import scipy, sklearn  # noqa: F401
    man = dict(
        created=started, finished=dt.datetime.now().isoformat(timespec="seconds"),
        command=" ".join([sys.executable, "-m", "extraction.run"] + (argv if argv is not None else sys.argv[1:])),
        code=dict(git=git_info(), package_sha256=ph, package_n_files=nf,
                  note="extraction/ is not committed; package_sha256 (sha256 over sorted relative paths + contents of "
                       "extraction/**/*.py) identifies the code version",
                  vendored={os.path.basename(f): open(f).readlines()[1].split(":")[-1].strip()
                            for f in sorted(glob.glob(os.path.join(PKG, "_vendor", "*.py"))) if "__init__" not in f}),
        params=PR.as_dict(),
        dataset=dict(root=bsd_io.DEFAULT_ROOT, chunk_index=os.path.relpath(bsd_io.INDEX_PATH, REPO),
                     chunk_index_md5=idx_md5, n_chunks=int(len(ix)),
                     total_bytes=int(ix.values_size.sum() + ix.time_size.sum()), runs=runs),
        counts=dict(per_run=counts, total=tot, tiers=list(PR.TIERS)),
        files=sorted(os.path.basename(f) for f in glob.glob(os.path.join(out, "*"))),
        runtime=dict(wall_s=time.time() - t0, wall_pipeline_s=wall_pipeline,
                     cpu_children_user_s=ru.ru_utime, cpu_children_sys_s=ru.ru_stime,
                     cpu_blocks_sum_s=float(sum(s["cpu_s"] for s in all_stats)), procs=a.procs,
                     data_hours=float(sum(s["core_hours"] for s in all_stats))),
        env=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
                 scipy=__import__("scipy").__version__, host_cpus=os.cpu_count()),
        schema=dict(windows="float32 [N,1050] baseline-subtracted counts on the run-global ~50 Hz grid "
                            "(sample k at slot (win_start_j+k)*1208/25); effective rate fs_eff per event",
                    evmask="bool [N,1050] C's [onset, offset]", repmask="bool [N,1050] dropout-affected or padded",
                    morph="m_* on the 21 s window, m8_* on the central 402 samples; time features in true seconds"),
    )
    json.dump(man, open(os.path.join(out, "manifest.json"), "w"), indent=1, default=str)
    print(json.dumps(man["counts"]["total"]), f"wall {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
