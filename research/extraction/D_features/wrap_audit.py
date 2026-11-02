"""How often does the legacy int16 low-word reader wrap (65536-count jumps)?
(a) on D's event / background windows (10 s), (b) on 300 random whole chunks (41.7 s, ~ legacy 60-s windows)."""
import numpy as np, pandas as pd, json
from features import legacy_read_int16
from io_detect import list_chunks, load_chunk

SCR = "/tmp/claude-1000/-home-augustin-Documents-federated-transformer/8ccb016b-de96-4655-ab49-d3318afeb915/scratchpad/"
RAW = np.load(SCR + "RAW.npy", mmap_mode="r"); meta = pd.read_parquet(SCR + "meta.parquet")
wr = np.zeros(len(meta), int)
for i in range(len(meta)):
    w = legacy_read_int16(np.asarray(RAW[i], float) + meta.dc_baseline[i])
    wr[i] = int(np.sum(np.abs(np.diff(w)) > 32768))
meta["wraps"] = wr
out = {"windows_with_wrap_frac": meta.assign(w=meta.wraps > 0).groupby("kind").w.mean().to_dict(),
       "event_wrap_frac_by_peak_tertile": meta[meta.kind == "event"].assign(
           t=pd.qcut(meta.peak[meta.kind == "event"], 3, labels=["low", "mid", "high"]), w=lambda d: d.wraps > 0)
           .groupby("t", observed=True).w.mean().to_dict(),
       "event_wrap_frac_by_run": meta[meta.kind == "event"].assign(w=lambda d: d.wraps > 0).groupby("run").w.mean().to_dict()}
ch = list_chunks(); rng = np.random.default_rng(0)
sel = rng.choice(len(ch), 300, replace=False)
cw = []
for j in sel:
    v, _ = load_chunk(ch[j][3])
    w = legacy_read_int16(v)
    cw.append(dict(run=ch[j][0], wraps=int(np.sum(np.abs(np.diff(w)) > 32768)), code_min=v.min(), code_max=v.max()))
cw = pd.DataFrame(cw)
out["random_chunks_with_wrap_frac"] = float((cw.wraps > 0).mean())
out["random_chunks_with_wrap_by_run"] = cw.assign(w=cw.wraps > 0).groupby("run").w.mean().to_dict()
out["random_chunks_median_wraps_if_any"] = float(cw.wraps[cw.wraps > 0].median()) if (cw.wraps > 0).any() else 0
json.dump(out, open("wrap_audit.json", "w"), indent=1, default=float)
print(json.dumps(out, indent=1, default=float))
