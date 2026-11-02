"""Extended value/time sanity scan on a stratified random subset (~300 chunks) plus the 100
chunks with the most estimated missing samples. Loads full chunks only for this subset."""
import os, numpy as np, pandas as pd
from multiprocessing import Pool
HERE = os.path.dirname(os.path.abspath(__file__))
LAG = 106
def one(a):
    run, idx, pt, pv, why = a
    t = np.fromfile(pt, "<u8").astype(np.int64); v = np.fromfile(pv, "<u4").astype(np.int64)
    d = np.diff(t); med = np.median(d); ok = d < 1.5 * med; base = d[ok].mean()
    slots = np.concatenate([[0], np.cumsum(np.maximum(1, np.rint(d / base)).astype(np.int64))])
    # timestamp noise model: residual from a line in slot space (local, 2000-sample windows to absorb drift)
    res = []
    for k in range(0, len(t) - 2000, 2000):
        x = slots[k:k+2000]; y = t[k:k+2000]; p = np.polyfit(x, y, 1); res.append(y - np.polyval(p, x))
    res = np.concatenate(res)
    dd = d[ok] - d[ok].mean(); ac1 = np.dot(dd[:-1], dd[1:]) / np.dot(dd, dd)
    dv = np.diff(v); mad = 1.4826 * np.median(np.abs(dv - np.median(dv))) + 1e-9
    same = np.concatenate([[0], (dv == 0).astype(int), [0]]); e = np.flatnonzero(np.diff(same)); runs = e[1::2] - e[::2]
    aa = v[1:-1] - v[:-2]; bb = v[1:-1] - v[2:]
    spikes = int(np.sum((np.sign(aa) == np.sign(bb)) & (np.minimum(np.abs(aa), np.abs(bb)) > 20 * mad)))
    big = np.flatnonzero(d > 20 * med)   # the ~44 ms class
    jumps = np.abs(dv[big]) if len(big) else np.array([])
    # reference distributions for jumps: lag-1 and lag-106 absolute diffs
    ref1 = np.median(np.abs(dv)); ref106 = np.median(np.abs(v[LAG:] - v[:-LAG]))
    # high-pass noise estimate: MAD of second difference / sqrt(6)
    d2 = np.diff(v, 2); sig = 1.4826 * np.median(np.abs(d2 - np.median(d2))) / np.sqrt(6)
    return dict(run=run, index=idx, why=why, n=len(v), base_dt=base, max_gap_ms=d.max() / 1e3, n_gap_small=int(((d > 1.5 * med) & (d <= 20 * med)).sum()),
                n_gap_44=len(big), missing_slots=int(slots[-1] - (len(t) - 1)), ts_resid_std=res.std(), ts_resid_maxabs=np.abs(res).max(), dt_ac1=ac1,
                vmin=v.min(), vmax=v.max(), top_byte=int((v >> 24).max()), n_zero=int((v == 0).sum()), max_same_run=int(runs.max() if len(runs) else 0),
                n_spikes=spikes, noise_sigma=sig, gap_jump_med=np.median(jumps) if len(jumps) else np.nan, ref_lag1=ref1, ref_lag106=ref106,
                ptp=v.max() - v.min(), low4_chi2=float(((np.bincount(v & 15, minlength=16) - len(v) / 16) ** 2 / (len(v) / 16)).sum()))
if __name__ == "__main__":
    h = pd.read_parquet(os.path.join(HERE, "raw_header_scan.parquet")); e = pd.read_parquet(os.path.join(HERE, "time_edge_scan.parquet"))
    T = h[h.kind == "Time"][["run", "index", "path", "fn_start"]].rename(columns={"path": "pt"})
    V = h[h.kind == "Values"][["run", "index", "path"]].rename(columns={"path": "pv"})
    j = T.merge(V, on=["run", "index"]).merge(e, on=["run", "index"])
    j = j[j.n == 100000]
    rng = np.random.default_rng(1)
    j["hb"] = j.fn_start.dt.hour
    rand = j.loc[np.concatenate([rng.choice(g.index.values, min(4, len(g)), replace=False) for _, g in j.groupby(["run", "hb"])])]
    top = j.nlargest(100, "n_missing_est")
    jobs = [(r.run, r["index"], r.pt, r.pv, "random") for _, r in rand.iterrows()] + [(r.run, r["index"], r.pt, r.pv, "top_missing") for _, r in top.iterrows()]
    with Pool(4) as p: res = p.map(one, jobs)
    out = pd.DataFrame(res); out.to_parquet(os.path.join(HERE, "sanity_scan.parquet"))
    pd.set_option("display.width", 250)
    print(out.groupby("why").describe().T.to_string())
