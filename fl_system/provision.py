"""Provision one federation (report 25, A1): split the data into one folder per client ("site") and one server folder, so
that every process can only see its own data. Run with the STUDY environment (.venv) from the repo root:
    .venv/bin/python fl_system/provision.py --fold 0 [--server-seed 0]
Output (never overwritten): data/fl_system/fold<F>/
  client_<k>/ events.npz (W float32 [n,1050], fs_eff, ev_i0, ev_i1, sigma_ev, t_s, days), noise.npz (pool rows of the client's
              own days + per-day start/count), info.json (k, days, n, sha of each file)
  server/     server_init_s<seed>.pt (copy of the pinned study server checkpoint, hash-checked), info.json
  federation.json  K, n_k, N_total, fold, client days, provenance (folds.json, pool and m4 data hashes)
Client membership and event order are exactly fedsim3's natural partition (sorted m4 rows), asserted below.
"""
import argparse, hashlib, json, os, shutil, sys
import numpy as np, pandas as pd
ROOT = os.getcwd(); sys.path[:0] = [ROOT, os.path.join(ROOT, "research/model/M4_methods")]
import m4
PROT = os.path.join(ROOT, "research/model/K_committee/protocol"); DATA = os.path.join(ROOT, "data/k_committee/protocol")
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()


def add_server_seed(fold_dir, seed):
    """Copy another pinned study server checkpoint into an existing server folder (new file only, hash-checked)."""
    src = os.path.join(DATA, "runs3", f"v21_server_s{seed}", "final.pt"); pins = json.load(open(os.path.join(PROT, "server_hashes.json")))
    assert sha(src) == pins[str(seed)], "server checkpoint differs from the pinned hash"
    dst = os.path.join(fold_dir, "server", f"server_init_s{seed}.pt"); assert not os.path.exists(dst); shutil.copyfile(src, dst); print(dst, sha(dst)[:16])


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "add-server-seed":
        return add_server_seed(sys.argv[2], int(sys.argv[3]))
    ap = argparse.ArgumentParser(); ap.add_argument("--fold", type=int, required=True); ap.add_argument("--server-seed", type=int, default=0)
    ap.add_argument("--version", default="v2", help="output folder suffix (fold0 of 2026-10-13 = v1, superseded: object arrays)")
    a = ap.parse_args(); out = os.path.join(ROOT, "data/fl_system", f"fold{a.fold}_{a.version}"); assert not os.path.exists(out), f"refusing to overwrite {out}"
    F = json.load(open(os.path.join(PROT, "folds.json"))); f = F["folds"][str(a.fold)]
    m, W = m4.load(); day = m.date.astype(str).to_numpy()
    pool = np.load(os.path.join(DATA, "bgpool_day_v2.npy")); pday = pd.read_parquet(os.path.join(DATA, "bgpool_day_v2_meta.parquet")).day.to_numpy()
    forbidden = set(f["eval_days"]) | set(F["server"]) | {F["tuning_day"]}
    fed = dict(fold=a.fold, clients=[], provenance=dict(folds=sha(os.path.join(PROT, "folds.json")), pool=sha(os.path.join(DATA, "bgpool_day_v2.npy"))))
    for k, c in enumerate(f["clients"]):
        idx = np.flatnonzero(np.isin(day, c["days"]))                       # == fedsim3.partition(fold, "natural")[k], sorted m4 rows
        assert not (set(day[idx]) & forbidden), f"client {k}: forbidden days"
        d = os.path.join(out, f"client_{k:02d}"); os.makedirs(d); s = m.iloc[idx]
        np.savez(os.path.join(d, "events.npz"), W=W[idx].astype(np.float32), fs_eff=s.fs_eff.to_numpy(np.float32), ev_i0=s.ev_i0.to_numpy(),
                 ev_i1=s.ev_i1.to_numpy(), sigma_ev=s.sigma_ev.to_numpy(np.float32), t_s=s.t_s.to_numpy(np.float64),
                 day=np.array(s.date.astype(str).tolist(), dtype="U10"), event_id=np.array(s.event_id.astype(str).tolist(), dtype="U32"))   # no object arrays (no pickle)
        days = sorted(set(s.date.astype(str))); rows, st, ct = [], {}, {}
        for dd in days:                                                     # same per-day block layout as fedsim3.day_pool
            r = np.flatnonzero(pday == dd); assert len(r) > 0; st[dd] = sum(len(x) for x in rows); ct[dd] = len(r); rows.append(r)
        np.savez(os.path.join(d, "noise.npz"), pool=pool[np.concatenate(rows)], days=np.array(days, dtype="U10"), start=np.array([st[x] for x in days]),
                 count=np.array([ct[x] for x in days]))
        info = dict(k=k, days=c["days"], n=int(len(idx)), events_sha=sha(os.path.join(d, "events.npz")), noise_sha=sha(os.path.join(d, "noise.npz")))
        json.dump(info, open(os.path.join(d, "info.json"), "x"), indent=1); fed["clients"].append(dict(k=k, days=c["days"], n=int(len(idx))))
    sd = os.path.join(out, "server"); os.makedirs(sd)
    src = os.path.join(DATA, "runs3", f"v21_server_s{a.server_seed}", "final.pt"); pins = json.load(open(os.path.join(PROT, "server_hashes.json")))
    assert sha(src) == pins[str(a.server_seed)], "server checkpoint differs from the pinned hash"
    dst = os.path.join(sd, f"server_init_s{a.server_seed}.pt"); shutil.copyfile(src, dst)
    json.dump(dict(server_days=F["server"], init=os.path.basename(dst), init_sha=sha(dst)), open(os.path.join(sd, "info.json"), "x"), indent=1)
    fed.update(K=len(fed["clients"]), N_total=int(sum(c["n"] for c in fed["clients"])))
    json.dump(fed, open(os.path.join(out, "federation.json"), "x"), indent=1); print(json.dumps({k: v for k, v in fed.items() if k != "clients"}), [c["n"] for c in fed["clients"]])


if __name__ == "__main__":
    main()
