"""Render the markdown tables for REPORT.md from the result CSVs.  python report_tables.py > report_tables.md"""
import json
import os

import numpy as np
import pandas as pd

H = os.path.dirname(os.path.abspath(__file__))
f3 = lambda x: "–" if pd.isna(x) else f"{x:.3f}"


def ci(r):
    return f"{f3(r.value if 'value' in r else r.skill)} [{f3(r.lo)}, {f3(r.hi)}]"


print("## Selection (validation day, seed 0)\n")
S = pd.read_csv(os.path.join(H, "tab_selection.csv"))
print("| arch | lr | T | log_amp R² | log_dur R² | polarity skill | n_lobes skill | **mean** | final SSL loss |")
print("|---|---|---|---|---|---|---|---|---|")
for r in S.itertuples():
    print(f"| {r.arch} | {r.lr if not pd.isna(r.lr) else '–'} | {r.temp if not pd.isna(r.temp) else '–'} | {f3(r.val_log_amp)} | "
          f"{f3(r.val_log_dur)} | {f3(r.val_polarity)} | {f3(r.val_n_lobes)} | **{f3(r.val_mean)}** | {f3(r.final_loss)} |")

V = pd.read_csv(os.path.join(H, "tab_P1_values.csv"))
T = pd.read_csv(os.path.join(H, "tab_P1_tests.csv"))
REPS = ["cnn_ssl_mean3", "cnn_rand_mean3", "tf_ssl_mean3", "tf_rand_mean3", "morph8", "morph21", "pca64", "snr"]
for tier, tg in (("snr20", "y"), ("snr30", "y"), ("all", "y"), ("snr20", "y21")):
    print(f"\n## P1 values, tier {tier}, targets {'m8 (primary)' if tg == 'y' else 'm_ 21 s (secondary)'}: point [95% block-bootstrap CI] (seed SD)\n")
    print("| representation | log_amp R² | log_dur R² | polarity skill | n_lobes skill | mean |")
    print("|---|---|---|---|---|---|")
    v = V[(V.tier == tier) & (V.targets == tg)]
    for rep in REPS:
        q = v[v.rep == rep].set_index("target")
        cells = []
        for t in ("log_amp", "log_dur", "polarity", "n_lobes"):
            r = q.loc[t]
            sd = f" ({r.seed_sd:.3f})" if not pd.isna(r.get("seed_sd", np.nan)) else ""
            cells.append(f"{f3(r.value)} [{f3(r.lo)}, {f3(r.hi)}]{sd}")
        print(f"| {rep} | " + " | ".join(cells) + f" | {f3(q.value.mean())} |")
    print(f"\n## P1 tests, tier {tier}, targets {tg}: Δ = SSL − comparator (mean over seeds) [95% CI], one-sided p, Holm p\n")
    print("| arch | vs | target | Δ | seed SD of Δ | 95% CI | p | p Holm | pass |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in T[(T.tier == tier) & (T.targets == tg)].itertuples():
        ok = "yes" if (r.delta > 0 and r.p_holm < 0.05) else "no"
        print(f"| {r.arch} | {r.vs} | {r.target} | {r.delta:+.3f} | {r.delta_seed_sd:.3f} | [{r.lo:+.3f}, {r.hi:+.3f}] | "
              f"{r.p_one_sided:.4f} | {r.p_holm:.4f} | {ok if r.vs == 'rand' else '(report)'} |")

L = pd.read_csv(os.path.join(H, "tab_P2.csv")) if os.path.exists(os.path.join(H, "tab_P2.csv")) else None
if L is not None:
    RE2 = ["cnn_ssl_mean3", "tf_ssl_mean3", "cnn_rand_mean3", "tf_rand_mean3", "morph8", "morph21", "pca64", "snr"]
    for (tier, sub, probe), l in L.groupby(["tier", "subset", "probe"], sort=False):
        tg = [t for t in ("date", "run_id", "hbin", "has_dropout", "is0204") if t in set(l.target)]
        print(f"\n## P2 skill, tier {tier}, {sub}, probe {probe}: skill [95% CI] (seed SD) | null mean ± SD (max)\n")
        print("| representation | n | " + " | ".join(tg) + " |")
        print("|---|---|" + "---|" * len(tg))
        for rep in RE2:
            q = l[l.rep == rep].set_index("target")
            if len(q) == 0:
                continue
            cells = []
            for t in tg:
                r = q.loc[t]
                sd = f" ({r.seed_sd:.3f})" if not pd.isna(r.get("seed_sd", np.nan)) else ""
                nl = f"<br>null {r.null_mean:+.3f} ± {r.null_sd:.3f} ({r.null_max:+.3f})" if not pd.isna(r.null_mean) else ""
                cells.append(f"{r.skill:+.3f} [{r.lo:+.3f}, {r.hi:+.3f}]{sd}{nl}")
            print(f"| {rep} | {int(q.n.iloc[0])} | " + " | ".join(cells) + " |")
        if "run_id" in tg:
            q = l[(l.target == "run_id")]
            print("\nRun-0204 recall in the 4-class run problem: " + ", ".join(
                f"{r.rep} {r.recall_0204:.3f}" for r in q.itertuples() if r.rep in RE2 and not pd.isna(r.recall_0204)))

P3 = pd.read_csv(os.path.join(H, "tab_P3.csv"))
print("\n## P3 retrieval@1 (holdout SNR≥20, gallery = clean central crops, cosine)\n")
print(P3.groupby(["rep", "query"]).R_at_1.agg(["mean", "std"]).round(3).to_string())
P4 = pd.read_csv(os.path.join(H, "tab_P4.csv"))
print("\n## P4 cross-seed 10-NN Jaccard (holdout SNR≥20)\n")
print(P4.groupby("rep").jaccard.agg(["mean", "std"]).round(3).to_string())
if os.path.exists(os.path.join(H, "tab_P5.csv")):
    P5 = pd.read_csv(os.path.join(H, "tab_P5.csv"))
    print("\n## P5 KMeans silhouette / bootstrap ARI (holdout SNR≥20, mean over seeds)\n")
    a = P5.groupby(["rep", "k"])[["silhouette", "boot_ari"]].mean().round(3)
    print(a.unstack("rep").to_string())
if os.path.exists(os.path.join(H, "results_verdict.json")):
    print("\n## Verdict JSON\n\n```\n" + json.dumps(json.load(open(os.path.join(H, "results_verdict.json"))), indent=1) + "\n```")
