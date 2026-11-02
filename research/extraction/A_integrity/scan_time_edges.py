"""For every Time file read the first and last N=2000 timestamps (32 KB/file) to estimate the
per-chunk base sampling interval and, from the chunk's total device duration, the number of
'missing' sample slots (gaps). Output: time_edge_scan.parquet"""
import os, numpy as np, pandas as pd
from multiprocessing import Pool
HERE = os.path.dirname(os.path.abspath(__file__))
N = 2000
def one(args):
    run, idx, path, size = args
    n = size // 8
    with open(path, "rb") as f:
        a = np.frombuffer(f.read(8 * min(N, n)), "<u8").astype(np.int64)
        f.seek(max(0, n - N) * 8); b = np.frombuffer(f.read(), "<u8").astype(np.int64)
    d = np.concatenate([np.diff(a), np.diff(b)])
    med = np.median(d); ok = d < 1.5 * med
    base = d[ok].mean()
    dur = b[-1] - a[0]
    return dict(run=run, index=idx, n=n, t_first=a[0], t_last=b[-1], base_dt=base, base_dt_head=np.diff(a)[np.diff(a) < 1.5 * med].mean(),
                base_dt_tail=np.diff(b)[np.diff(b) < 1.5 * med].mean(), dt_std=d[ok].std(),
                n_missing_est=dur / base - (n - 1), edge_gaps=int((~ok).sum()))
if __name__ == "__main__":
    df = pd.read_parquet(os.path.join(HERE, "raw_header_scan.parquet"))
    T = df[df.kind == "Time"]
    with Pool(4) as p:
        res = p.map(one, list(zip(T.run, T["index"], T.path, T["size"])), chunksize=64)
    out = pd.DataFrame(res).sort_values(["run", "index"])
    out.to_parquet(os.path.join(HERE, "time_edge_scan.parquet"))
    print(out.describe().T)
