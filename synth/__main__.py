"""python -m synth fit | sample --n N [--family F --polarity P] [--seed S] [--out PATH]"""
import argparse
import os

import numpy as np

from . import generator as G

ap = argparse.ArgumentParser(prog="python -m synth")
sub = ap.add_subparsers(dest="cmd", required=True)
f = sub.add_parser("fit", help="fit on training days (bandwidth: --bw, default = the one chosen by synth.validate)")
f.add_argument("--bw", type=float, default=None)
f.add_argument("--all-days", action="store_true", help="fit on every day (no held-out days left for validation)")
s = sub.add_parser("sample", help="draw synthetic events from data/synth/model.npz")
s.add_argument("--n", type=int, required=True)
s.add_argument("--family", type=int); s.add_argument("--polarity", type=int, choices=(-1, 1))
s.add_argument("--seed", type=int, default=0); s.add_argument("--out")
a = ap.parse_args()
path = os.path.join(G.OUT, "model.npz"); os.makedirs(G.OUT, exist_ok=True)

if a.cmd == "fit":
    bw = a.bw if a.bw is not None else (float(np.load(path)["bw"]) if os.path.exists(path) else 0.2)
    m, x = G.load_events()
    g = G.Generator.fit(m, x, bw=bw, roles=("train", "val", "test") if a.all_days else ("train",))
    g.save(path); print(g.summary().to_string(index=False)); print("bw", bw, "->", path)
else:
    g = G.Generator.load(path)
    key = (a.family, a.polarity) if a.family is not None else None
    if key is not None and key not in g.strata:
        raise SystemExit(f"no stratum {key}; available: {g.keys}")
    X, info = g.sample(a.n, key=key, rng=a.seed)
    out = a.out or os.path.join(G.OUT, f"sample_{a.n}_seed{a.seed}.npz")
    np.savez_compressed(out, x=X, **{c: info[c].to_numpy() for c in info.columns})
    print(f"{a.n} events -> {out}"); print(info.groupby(["family", "polarity"]).size().to_string())
