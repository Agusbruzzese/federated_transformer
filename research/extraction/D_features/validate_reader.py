"""PROTOCOL item 5: validate D's provisional reader (io_detect.py) against A's bsd_io.py."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, "../A_integrity")
import bsd_io
from io_detect import list_chunks as d_list, load_chunk as d_load
idx = bsd_io.list_chunks()
mine = d_list()
print("A chunks:", len(idx), "paired:", int(idx.paired.sum()), "| D chunks:", len(mine))
rng = np.random.default_rng(0)
sel = idx[idx.paired].sample(40, random_state=0)
res = []
for r in sel.itertuples():
    ta, va, info = bsd_io.load_chunk(r.values_path, r.time_path, return_info=True)
    vd, td = d_load(r.values_path)
    res.append(dict(run=r.run, index=r.index, same_values=bool(np.array_equal(va, vd)), n=len(vd),
                    n_missing=info["n_missing"], fs_local=r.fs_local))
df = pd.DataFrame(res); print(df.describe(include="all").T[["count", "mean", "min", "max"]])
print("all values identical:", df.same_values.all())
print("dropout samples per chunk: median %.0f, mean %.1f, max %d" % (df.n_missing.median(), df.n_missing.mean(), df.n_missing.max()))
fsl = idx.fs_local.dropna()
print("fs_local (A): median %.1f Hz, range %.1f-%.1f; D assumes 2400 -> time-scale error %.2f%%" %
      (fsl.median(), fsl.min(), fsl.max(), 100 * (fsl.median() / 2400 - 1)))
# D's list vs A's paired list
dset = set((m[0], m[1]) for m in mine)
runs = sorted(idx.run.unique())
aset = set((runs.index(r.run), r.index) for r in idx[idx.paired].itertuples())
print("D-only chunks:", len(dset - aset), " A-paired not in D:", len(aset - dset))
print("chunks with any dropout: %.1f%%; total missing fraction %.4f%%" %
      (100 * (idx.n_missing_est > 0).mean(), 100 * idx.n_missing_est.sum() / idx.n_samples.sum()))
