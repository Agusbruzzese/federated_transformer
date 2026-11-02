"""Step 2 of protocol v2 (report 09, section 2.3): own-day noise pools for the days C did not build
(Feb 7, 13, 19, 25: client days in the fold design; Feb 10: tuning/monitoring), merged with C's pools.

Rule (fixed in report 09 before this script was run):
  strict pool (ssl_central I5 rule, C's per-day block choice, 6 blocks/day); if a day has < N_MIN = 50 strict
  segments, add its relaxed segments (<= 10 % flagged samples, bridged). The relaxed rule accepts every
  strict-clean stretch too, so segments are de-duplicated by (run, j), keeping the strict copy.
C's code paths are reused unchanged (noisepool._job via ssl_central.bgpool, noisepool._job_relaxed).
C's files are read, never modified. Output: data/k_committee/protocol/bgpool_day_v2.{npy,meta.parquet}, counts json.
    .venv/bin/python research/model/K_committee/protocol/noisepool_v2.py [procs=12]
"""
import hashlib, json, os, sys, time
from multiprocessing import Pool
import numpy as np, pandas as pd
ROOT = os.getcwd(); sys.path[:0] = [ROOT, os.path.join(ROOT, "research/model/K_committee/C_federated")]
import noisepool as NP
from ssl_central.bgpool import _job
NEW_DAYS = ["2025-02-07", "2025-02-10", "2025-02-13", "2025-02-19", "2025-02-25"]
N_MIN, BPD = 50, 6
OUT = os.path.join(ROOT, "data", "k_committee", "protocol"); HERE = os.path.dirname(os.path.abspath(__file__))


def day_of_job(ix, job):
    run, (c0, c1), _ = job; ixr = ix[ix.run == run].sort_values("index").reset_index(drop=True)
    return pd.to_datetime(ixr.wall_start).dt.strftime("%Y-%m-%d").iloc[c0]


def collect(res, rule):
    segs = [s for S, _ in res for s in S]; prov = pd.DataFrame([q for _, P in res for q in P])
    if len(prov):
        prov["day"] = pd.to_datetime(prov.wall_time).dt.strftime("%Y-%m-%d"); prov["rule"] = rule
    return segs, prov


def main():
    procs = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    from extraction.blockio import chunk_index
    ix = chunk_index()
    NP.TRAIN_DAYS = NEW_DAYS                                     # C's planner, same block rule, new days only
    jobs = NP.plan(BPD); t0 = time.time(); print(len(jobs), "strict blocks", flush=True)
    with Pool(procs) as p: res = p.map(_job, jobs)
    segs, prov = collect(res, "strict")
    strict_counts = prov.day.value_counts().reindex(NEW_DAYS, fill_value=0) if len(prov) else pd.Series(0, index=NEW_DAYS)
    low = [d for d in NEW_DAYS if strict_counts[d] < N_MIN]; print("strict counts", strict_counts.to_dict(), "relax:", low, flush=True)
    if low:
        rj = [j for j in jobs if day_of_job(ix, j) in low]
        with Pool(procs) as p: rres = p.map(NP._job_relaxed, rj)
        rs, rp = collect(rres, "relaxed"); segs += rs; prov = pd.concat([prov, rp], ignore_index=True)
    # merge with C's pool
    bgC = np.load(os.path.join(NP.OUT, "bgpool_day.npy")); mC = pd.read_parquet(os.path.join(NP.OUT, "bgpool_day_meta.parquet"))
    if "rule" not in mC: mC["rule"] = "strict"
    X = np.concatenate([bgC, np.stack(segs)]); M = pd.concat([mC, prov], ignore_index=True)
    order = np.argsort((M.rule != "strict").to_numpy(), kind="stable")          # strict copies first
    X, M = X[order], M.iloc[order].reset_index(drop=True)
    keep = ~M.duplicated(["run", "j"], keep="first").to_numpy()
    X, M = X[keep], M[keep].reset_index(drop=True)
    o = np.lexsort((M.j.to_numpy(), M.run.astype(str).to_numpy(), M.day.to_numpy())); X, M = X[o], M.iloc[o].reset_index(drop=True)
    os.makedirs(OUT, exist_ok=True)
    np.save(os.path.join(OUT, "bgpool_day_v2.npy"), X.astype(np.float32)); M.to_parquet(os.path.join(OUT, "bgpool_day_v2_meta.parquet"))
    tab = M.groupby(["day", "rule"]).size().unstack(fill_value=0)
    tab["total"] = tab.sum(1)
    sha = hashlib.sha256(open(os.path.join(OUT, "bgpool_day_v2.npy"), "rb").read()).hexdigest()
    out = dict(n_min=N_MIN, new_days=NEW_DAYS, strict_counts_new=strict_counts.astype(int).to_dict(), relaxed_days_new=low,
               duplicates_removed=int((~keep).sum()), total_segments=int(len(M)), sha256=sha, secs=round(time.time() - t0),
               per_day=tab.astype(int).to_dict("index"), days_below_n_min_after_rule=[d for d, r in tab.iterrows() if r.total < N_MIN])
    json.dump(out, open(os.path.join(HERE, "noisepool_v2.json"), "w"), indent=1); print(tab.to_string()); print(json.dumps({k: v for k, v in out.items() if k != "per_day"}, indent=1))


if __name__ == "__main__":
    main()
