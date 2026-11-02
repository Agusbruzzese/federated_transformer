"""C2 traceable verification (post-audit 124215): tolerance guarantee, look-alike abundance, cross-run/day recurrence
(non-partition reproducibility). Writes verify.json.   python research/model/C2_lookalikes/verify.py"""
import os, sys, json, numpy as np, pandas as pd
ROOT = os.getcwd(); sys.path[:0] = [ROOT, f"{ROOT}/research/model/M4_methods"]; import m4
HERE = os.path.dirname(os.path.abspath(__file__))
m, _ = m4.load(); L = pd.read_csv(os.path.join(HERE, "c2_labels.csv")); assert (L.event_id.values == m.event_id.values).all()
P = np.load(os.path.join(ROOT, "data", "c1", "aligned_corr.npy")); la = m.m8_log_amp.values; ld = np.log10(m.m8_dur10.values)
run = m.run_id.values; day = np.asarray(m.date.astype(str).to_numpy(), dtype=object); t = m.t_s.to_numpy(np.float64); rng = np.random.default_rng(4)
D = np.maximum(np.maximum((1 - P) / 0.1, np.abs(la[:, None] - la[None]) / np.log10(1.5)), np.abs(ld[:, None] - ld[None]) / np.log10(1.5))
np.fill_diagonal(D, 9); A = D <= 1; out = {}
for k in ("C2a", "C2b"):
    l = L[k].values; w = [0, 0, 1]
    for g in np.unique(l[l >= 0]):
        i = np.flatnonzero(l == g); w = [max(w[0], 10 ** np.ptp(la[i])), max(w[1], 10 ** np.ptp(ld[i])), min(w[2], P[np.ix_(i, i)].min())]
    out[f"{k}_worst_within_group"] = dict(amp_ratio=w[0], dur_ratio=w[1], min_corr=w[2])
k = A.sum(1); out["partners"] = dict(ge1=float((k >= 1).mean()), ge5=float((k >= 5).mean()), ge10=float((k >= 10).mean()), median=float(np.median(k)))
other_run = (A & (run[:, None] != run[None])).any(1); other_day = (A & (day[:, None] != day[None])).any(1)
far = (A & (np.abs(t[:, None] - t[None]) > 3600)).any(1)                       # partner >1 h away (not the same convoy/minute)
def boot(x):
    b = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(2000)]; return [float(x.mean()), float(np.quantile(b, .025)), float(np.quantile(b, .975))]
out["recurrence"] = dict(lookalike_in_other_run=boot(other_run), lookalike_on_other_day=boot(other_day), lookalike_more_than_1h_apart=boot(far))
out["recurrence_per_run"] = {int(r): float(other_run[run == r].mean()) for r in np.unique(run)}
# chance reference: same but with amplitude/duration tolerance only (shape ignored) -> how much shape adds
A2 = np.maximum(np.abs(la[:, None] - la[None]) / np.log10(1.5), np.abs(ld[:, None] - ld[None]) / np.log10(1.5)) <= 1; np.fill_diagonal(A2, False)
out["share_of_amp_dur_matches_that_also_match_shape"] = float(A.sum() / A2.sum())
json.dump(out, open(os.path.join(HERE, "verify.json"), "w"), indent=1, default=float); print(json.dumps(out, indent=1, default=float))
