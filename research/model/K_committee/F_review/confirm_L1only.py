"""Reviewer F: behaviour of analyze.controls() when ONLY L1 fails (report 14 A2 pre-stated response). Output: confirm_L1only.json (x)."""
import json, os, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(os.path.dirname(HERE), "protocol"), os.getcwd(), "research/model/M4_methods"]
import analyze as AN
rng = np.random.default_rng(3)
rows = []
for a, ss, eff in (("C256", range(10), 0), ("C256rep", range(10, 20), 0), ("INIT", range(10), -.3), ("C4", range(10, 20), -.12), ("C5", range(10, 20), -.05)):
    for s in ss:
        for d in AN.DAYS:
            for k in ("LOBES", "POLARITY", "DURATION"): rows.append(dict(arm=a, seed=s, fold=AN.FOLD_OF_DAY[d], day=d, metric=k, value=.6 + eff + rng.normal(0, .01)))
S = pd.DataFrame(rows)
for it in range(2000):
    zm = rng.normal(0, .03, 10); zr = rng.normal(.02, .03, 10)
    M = pd.DataFrame([dict(arm=a, seed=s, fold=f, erank=20., Z=z, Q=0., Q_null_q95=.05) for a, ss, zz in (("C256", range(10), zm), ("C256rep", range(10, 20), zr))
                      for s, z in zip(ss, zz) for f in range(3)] + [dict(arm="L2_0.03", seed=s, fold=f, erank=20., Z=.5, Q=.5, Q_null_q95=.05) for s in range(10, 20) for f in range(3)])
    c = AN.controls(S, M, [0.03])
    if not c["criteria"]["L1"]: break
out = dict(iteration=it, criteria=c["criteria"], ALL_PASS=c["ALL_PASS"], gate_status=c["gate_status"], m_leak=c["m_leak"], L1=dict(U=c["L1"]["U"], delta=c["L1"]["delta"]))
json.dump(out, open(os.path.join(HERE, "confirm_L1only.json"), "x"), indent=1, default=str); print(json.dumps(out, indent=1, default=str))
