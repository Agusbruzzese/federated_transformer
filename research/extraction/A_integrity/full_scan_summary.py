"""Summarise full_scan_*.parquet: exact dropouts per day / run, size histogram, max gap,
value sanity over ALL chunks. Also compares exact n_missing vs edge estimate."""
import os, numpy as np, pandas as pd
import bsd_io as b
HERE = os.path.dirname(os.path.abspath(__file__))
C = pd.read_parquet(os.path.join(HERE, "full_scan_chunks.parquet"))
G = pd.read_parquet(os.path.join(HERE, "full_scan_gaps.parquet"))
idx = b.list_chunks()
C = C.merge(idx[["run", "index", "wall_start", "fs_local", "n_missing_est", "truncated", "t_first", "t_last"]],
            on=["run", "index"], suffixes=("", "_idx"))
print("chunks scanned", len(C), "| n==n_v all:", (C.n == C.n_v).all(), "| t_first/t_last match index:",
      ((C.t_first == C.t_first_idx) & (C.t_last == C.t_last_idx)).all())
print("timestamps non-positive deltas:", C.n_nonpos.sum(), "| min dt ticks:", C.dt_min.min())
print("\n--- VALUES over ALL %d chunks (%.0f M samples)" % (len(C), C.n.sum() / 1e6))
print("vmin %d  vmax %d  top_byte max %d  zeros %d  max identical run %d  isolated spikes(>20 MAD) %d"
      % (C.vmin.min(), C.vmax.max(), C.top_byte.max(), C.n_zero.sum(), C.max_same_run.max(), C.n_spikes.sum()))
print("chunks with spikes:", (C.n_spikes > 0).sum())
print("\n--- DROPOUTS (exact)")
D = G[G.n_missing > 0]; L = G[G.n_missing == 0]
print("dropout events %d, missing samples %d (%.3f %% of %d slots); late-stamp-only events %d"
      % (len(D), D.n_missing.sum(), 100 * D.n_missing.sum() / (C.n.sum() + D.n_missing.sum()), C.n.sum() + D.n_missing.sum(), len(L)))
h = D.n_missing.value_counts().sort_index()
print("size histogram (n_missing: count):", h.to_dict())
bins = [0, 1, 2, 5, 20, 99, 103, 106, 110, 1000, 10**9]
print("binned:", pd.cut(D.n_missing, bins).value_counts().sort_index().to_dict())
i = G.dt_ticks.idxmax(); r = G.loc[i]
print("TRUE max timestamp gap: %d ticks = %.2f ms(ticks) , %d missing samples (%s #%d)" % (r.dt_ticks, r.dt_ticks / 1e3, r.n_missing, r.run[4:12], r["index"]))
print("max n_missing single event:", D.n_missing.max())
print("ambiguous dt ratios in (1.3,1.7) (potential hidden 1-sample losses):", C.n_ambiguous_ratio.sum())
# burst spacing
D = D.sort_values(["run", "index", "pos"])
sp = D.groupby(["run", "index"]).pos.diff().dropna()
print("spacing between consecutive dropouts in a chunk: min %d, fraction <=1: %.4f, <=8: %.4f, <=40: %.3f"
      % (sp.min(), (sp <= 1).mean(), (sp <= 8).mean(), (sp <= 40).mean()))
print("spacing-1 events (sample sandwiched between two dropouts; slot +-1 ambiguous):", int((sp <= 1).sum()))
# exact vs edge estimate
e = C.n_missing - C.n_missing_est
print("\nedge estimate error (est-exact): mean %.2f sd %.2f p1 %.1f p99 %.1f max|.| %.1f"
      % (-e.mean(), e.std(), np.percentile(-e, 1), np.percentile(-e, 99), e.abs().max()))
# per run
C["run_s"] = C.run.str[4:12]
pr = C.groupby("run_s").agg(chunks=("n", "size"), samples=("n", "sum"), missing=("n_missing", "sum"),
                             events=("n_drop_events", "sum"), max_gap=("max_gap_slots", "max"))
pr["pct"] = 100 * pr.missing / (pr.samples + pr.missing)
print("\nper run:\n", pr.round(3).to_string())
# per day (assign by chunk start wall time)
G2 = D.merge(C[["run", "index", "wall_start", "fs_local"]], on=["run", "index"])
G2["date"] = (G2.wall_start + pd.to_timedelta(G2.pos / G2.fs_local, unit="s")).dt.date
C["date"] = C.wall_start.dt.date
pdd = C.groupby("date").agg(samples=("n", "sum"))
dd = G2.groupby("date").agg(events=("n_missing", "size"), missing=("n_missing", "sum"), max_missing=("n_missing", "max"),
                            ev_1_5=("n_missing", lambda x: int((x <= 5).sum())),
                            ev_100_106=("n_missing", lambda x: int(x.between(100, 106).sum())),
                            ev_other=("n_missing", lambda x: int(((x > 5) & ~x.between(100, 106)).sum())))
dd = pdd.join(dd).fillna(0)
dd["pct_missing"] = 100 * dd.missing / (dd.samples + dd.missing)
print("\nper day (chunk-start date):\n", dd.round(3).to_string())
dd.to_csv(os.path.join(HERE, "dropouts_per_day_exact.csv"))
h.to_csv(os.path.join(HERE, "dropout_size_histogram_exact.csv"))
