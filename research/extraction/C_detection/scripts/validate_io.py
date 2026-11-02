"""Validate cio.py (and detector.insert_dropouts slot reconstruction) against workstream A's bsd_io.py."""
import sys, os; sys.path.insert(0, '.'); sys.path.insert(0, '../A_integrity')
import numpy as np, pandas as pd, cio, detector as D, bsd_io as B
rng = np.random.default_rng(0)
idx = B.list_chunks(cache=True)
rows = []
for ri, r in enumerate(cio.runs()):
    ci = cio.index(r)
    for k in rng.choice(len(ci) - 1, 3, replace=False):
        c = ci[k]
        x, t, _ = cio.load_chunks(r, c[0], 1)
        tb, vb = B.read_raw(c[3], c[4])
        same_v = np.array_equal(x, vb.astype(float)); same_t = np.array_equal(np.round(t * 1e6).astype(np.uint64), tb)  # cio returns ticks/1e6 (NOT seconds; used only as ratios)
        # dropout slot reconstruction: mine vs A's
        _, slot, ins = D.insert_dropouts(x, t)
        a = B.slots_from_timestamps(tb)
        a = np.asarray(a[0] if isinstance(a, tuple) else a)
        n_missing_A = int(a[-1] - a[0] + 1 - len(a))
        rows.append(dict(run=ri, chunk=c[0], values_equal=same_v, time_equal=same_t, n=len(x),
                         missing_mine=int(ins.sum()), missing_A=n_missing_A,
                         slots_equal=bool(np.array_equal(slot - slot[0], a - a[0]))))
df = pd.DataFrame(rows); print(df.to_string()); df.to_csv('data/validate_io.csv', index=False)
print('all values equal', df.values_equal.all(), '| all time equal', df.time_equal.all(),
      '| slot recon identical', df.slots_equal.mean(), '| max |missing diff|', (df.missing_mine - df.missing_A).abs().max())
