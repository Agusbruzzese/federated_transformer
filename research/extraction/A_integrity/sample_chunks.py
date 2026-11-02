"""Stratified deep inspection of a subset of chunks (full sample load only for the subset)."""
import numpy as np, pandas as pd, os
HERE = os.path.dirname(os.path.abspath(__file__))
df = pd.read_parquet(os.path.join(HERE, "raw_header_scan.parquet"))
T = df[df.kind == "Time"].set_index(["run", "index"]); V = df[df.kind == "Values"].set_index(["run", "index"])
j = T.join(V, lsuffix="_t", rsuffix="_v").reset_index()
j["hour"] = j.fn_start_t.dt.hour
rng = np.random.default_rng(0)
# stratify: for each run x 4-hour block of day, pick 2-3 chunks at random -> ~ 60-70 chunks
sel = []
for (run, hb), g in j.groupby([j.run, j.hour // 4]):
    sel += list(rng.choice(g.index.values, size=min(3, len(g)), replace=False))
sel = sorted(set(sel) | {j.index[-1]})  # include truncated last chunk
rows = []
for i in sel:
    r = j.loc[i]
    t = np.fromfile(r.path_t, "<u8").astype(np.int64); v = np.fromfile(r.path_v, "<u4").astype(np.int64)
    d = np.diff(t); med = np.median(d)
    gapmask = d > 2 * med
    dv = np.diff(v)
    mad = np.median(np.abs(dv - np.median(dv))) * 1.4826 + 1e-9
    # longest run of identical consecutive values
    same = np.concatenate([[0], (dv == 0).astype(int), [0]])
    edges = np.flatnonzero(np.diff(same)); runs = edges[1::2] - edges[::2]
    # isolated spikes: sample deviates from both neighbours in same direction by > 20 MAD
    a = v[1:-1] - v[:-2]; b = v[1:-1] - v[2:]
    spikes = np.sum((np.sign(a) == np.sign(b)) & (np.minimum(np.abs(a), np.abs(b)) > 20 * mad))
    rows.append(dict(run=r.run[4:12], index=r["index"], start=r.fn_start_t, n=len(v), nt=len(t),
                     wall_dur=(r.fn_stop_t - r.fn_start_t).total_seconds(), dev_dur=(t[-1] - t[0]) / 1e6,
                     dt_med=med, dt_mean_nogap=d[~gapmask].mean(), dt_p01=np.percentile(d, 0.1), dt_p999=np.percentile(d[~gapmask], 99.9),
                     dt_min=d.min(), n_nonpos=(d <= 0).sum(), n_gaps=gapmask.sum(), gap_ms=list(np.round(d[gapmask] / 1e3, 2)),
                     gap_pos=list(np.flatnonzero(gapmask)),
                     vmin=v.min(), vmax=v.max(), vmed=np.median(v), vstd=v.std(), top_byte_max=int((v >> 24).max()),
                     n_unique=len(np.unique(v)), diff_mad=mad, max_abs_diff=np.abs(dv).max(), max_same_run=runs.max() if len(runs) else 0,
                     n_spikes=int(spikes), lsb_odd_frac=(v & 1).mean(), n_zero=(v == 0).sum(), n_ffffff=(v == 0xFFFFFF).sum()))
out = pd.DataFrame(rows)
out.to_parquet(os.path.join(HERE, "sample_chunk_stats.parquet"))
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40); pd.set_option("display.max_rows", 200)
print(out.drop(columns=["gap_ms", "gap_pos"]).to_string())
print(out[["run", "index", "gap_ms", "gap_pos"]].to_string())
