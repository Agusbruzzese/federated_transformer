"""Committee C: budget accounting for each arm (same formula as fedsim.run). Writes accounting.csv."""
import os, json, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); P = json.load(open(os.path.join(HERE, "partitions.json")))
X = 10_000 * 256; rows = []
def plan(n, b, R):
    n = np.array(n); N = n.sum(); bk = np.minimum(b, n); sk = np.maximum(1, np.round(X / R * n / N / bk)).astype(int)
    return dict(N=int(N), R=R, b=b, b_k="/".join(map(str, bk)), s_k="/".join(map(str, sk)), ex_round=int((sk * bk).sum()),
                ex_total=int(R * (sk * bk).sum()), ex_total_over_X=round(R * (sk * bk).sum() / X, 4), grad_steps=int(R * sk.sum()),
                seq_depth=int(R * sk.max()), epochs_per_event=round(R * (sk * bk).sum() / N, 1),
                epochs_small_vs_large=f"{(R * sk * bk / n).min():.0f}-{(R * sk * bk / n).max():.0f}")
for part in ("P13", "P11"):
    n = [len(c) for c in P[part]["natural"]]
    for arm, nn, b, R in [("central_b256", [sum(n)], 256, 1), ("central_b64", [sum(n)], 64, 1),
                          ("fed_b64_R200", n, 64, 200), ("fed_b64_R50", n, 64, 50), ("fed_b256cap_R200", n, 256, 200)]:
        rows.append(dict(part=part, arm=arm, **plan(nn, b, R)))
    if P[part]["server"]:
        Xs = 2000 * 256; ns = len(P[part]["server"])
        rows.append(dict(part=part, arm="server_pretrain_2000x256", N=ns, R=1, b=256, ex_total=Xs, grad_steps=2000, epochs_per_event=round(Xs / ns, 1)))
df = pd.DataFrame(rows); df.to_csv(os.path.join(HERE, "accounting.csv"), index=False); print(df.to_string(index=False))
