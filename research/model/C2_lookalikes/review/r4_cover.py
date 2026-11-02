"""Greedy prototype library: events covered (within tolerance d<=1) by k prototypes; held-out-run coverage."""
import sys, json, numpy as np
sys.path[:0] = ['.', 'research/model/M4_methods']; import m4
m, W = m4.load(); A = np.load('research/model/C2_lookalikes/review/A_main.npy'); run = m.run_id.to_numpy(); n = len(A)
C = A | np.eye(n, dtype=bool)
def greedy(C, K):
    cov = np.zeros(C.shape[1], bool); protos = []; curve = []
    for _ in range(K):
        g = (C[:, ~cov]).sum(1); j = int(g.argmax()); protos.append(j); cov |= C[j]; curve.append(cov.mean())
    return protos, curve
p, c = greedy(C, 400); out = {f'k{k}': round(float(c[k - 1]), 3) for k in (10, 25, 50, 100, 200, 400)}
lo = {}
for g in range(4):
    iA, iB = np.flatnonzero(run != g), np.flatnonzero(run == g); pA, _ = greedy(C[np.ix_(iA, iA)], 100)
    lo[g] = {k: round(float(C[np.ix_(iA[pA[:k]], iB)].any(0).mean()), 3) for k in (25, 100)}
out['loro_heldout_coverage_by_protos_from_other_runs'] = lo
json.dump(out, open('research/model/C2_lookalikes/review/r4.json', 'w'), indent=1); print(out)
