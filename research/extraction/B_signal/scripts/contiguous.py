"""Build contiguous 100 Hz series (mains-regressed, 20 Hz LP, dropout-aware) for selected 2-h windows,
then run the transient detector on them (for rates, inter-arrival times, overlaps)."""
import numpy as np, pandas as pd
from datetime import datetime, timedelta
from multiprocessing import Pool
import io_utils as io, chunk_stats as cs, events as E
WIN = [("20250205 0700", "weekday AM"), ("20250205 1630", "weekday PM"), ("20250206 0200", "night"),
       ("20250208 1100", "Saturday"), ("20250213 0700", "weekday AM"), ("20250213 1630", "weekday PM"),
       ("20250219 0700", "weekday AM"), ("20250219 1230", "weekday midday"), ("20250225 0700", "weekday AM"),
       ("20250225 1630", "weekday PM"), ("20250226 0200", "night")]
C = io.list_chunks()
def z_of(ch):
    r, _, _, z, bad = cs.one(ch); return z + (r["p50"] - 11_500_000), bad  # absolute level (offset 11.5M) restored
if __name__ == "__main__":
    out = {}
    with Pool(5) as p:
        for ts, lab in WIN:
            t0 = datetime.strptime(ts, "%Y%m%d %H%M"); t1 = t0 + timedelta(hours=2)
            sel = [c for c in C if t0 <= c["start"] < t1]
            zs = p.map(z_of, sel)
            z = np.concatenate([q[0] for q in zs]).astype(np.float32)
            out[f"{ts}|{lab}"] = z
            out[f"{ts}|{lab}|bad"] = np.concatenate([q[1] for q in zs])
            print(ts, lab, len(sel), "chunks", len(z) / 100 / 3600, "h", flush=True)
    np.savez_compressed("../data/contiguous_lp100.npz", **{k.replace(" ", "_").replace("|", "__"): v for k, v in out.items()})
