"""G stage 3: compare per-cell values with the protocol table; recompute Q-F, Q-S, interaction, S+F vs C256 from G's values."""
import os, json
import numpy as np, pandas as pd
from scipy import stats

OUT = "research/model/K_committee/G_recompute"
P = "research/model/K_committee/protocol"
g = pd.read_parquet(f"{OUT}/g_scores.parquet")
p = pd.read_parquet(f"{P}/scoring/scores_combined_20261012_211200.parquet")
res_p = json.load(open(f"{P}/results_main_20261012_211202.json"))
folds = json.load(open(f"{P}/folds.json"))
DAYS = [d for f in range(3) for d in folds["folds"][str(f)]["eval_days"]]
MARGIN = dict(LOBES=0.04, POLARITY=0.02)
report = {}

# ---- per-cell agreement
key = ["arm", "seed", "fold", "day", "metric"]
mg = g.merge(p, on=key, how="left", suffixes=("_g", "_p"))
assert mg.value_p.notna().all() and len(mg) == len(g) == 5 * 30 * 3 * 2
mg["ad"] = (mg.value_g - mg.value_p).abs()
agg = mg.groupby(["arm", "metric"]).ad.agg(["max", "mean", lambda x: int((x > 1e-6).sum())]).rename(columns={"<lambda_0>": "n_gt_1e-6"})
print(agg); report["per_cell"] = {f"{a}|{mm}": r.to_dict() for (a, mm), r in agg.iterrows()}
report["per_cell_overall_max"] = float(mg.ad.max())
# also check that the protocol table uses the same fold for each day
chk = p[p.arm.isin(g.arm.unique())].groupby("day").fold.unique()
report["protocol_day_fold_unique"] = {d: list(map(int, v)) for d, v in chk.items()}


def grid(arm, met, src=g):
    t = src[(src.arm == arm) & (src.metric == met)].pivot_table(index="seed", columns="day", values="value", aggfunc="count")
    assert (t.values == 1).all()
    y = src[(src.arm == arm) & (src.metric == met)].pivot(index="seed", columns="day", values="value")[DAYS]
    assert y.shape == (10, 9) and np.isfinite(y.values).all()
    return y.values


def outcome(L, U, m):
    if L > -m:
        if U < m: return "EQ"
        if L >= m: return "SUP"
        return "NI"
    return "INC" if U > -m else "INF"


def two_stratum(A, B, m):
    nA, nB, D = A.shape[0], B.shape[0], A.shape[1]
    dd = A.mean(0) - B.mean(0); delta = dd.mean()
    MA, MB = A.mean(1), B.mean(1)
    sp = (((MA - MA.mean()) ** 2).sum() + ((MB - MB.mean()) ** 2).sum()) / (nA + nB - 2)
    v1 = sp * (1 / nA + 1 / nB); df1 = nA + nB - 2
    v2 = np.var(dd - delta, ddof=1) / D; df2 = D - 1
    se = np.sqrt(v1 + v2); df = (v1 + v2) ** 2 / (v1 ** 2 / df1 + v2 ** 2 / df2)
    t = stats.t.ppf(0.95, df); L, U = delta - t * se, delta + t * se
    return dict(delta=delta, se=se, df=df, L=L, U=U, p_sup=float(stats.t.sf(delta / se, df)), outcome=outcome(L, U, m))


def label(o):
    v = list(o.values())
    if "INF" in v: return "INFERIOR"
    if "INC" in v: return "INCONCLUSIVE"
    if all(x == "EQ" for x in v): return "EQUIVALENT"
    return "NON-INFERIOR"


def contrast(a, b):
    r = {mt: two_stratum(grid(a, mt), grid(b, mt), MARGIN[mt]) for mt in MARGIN}
    return r, label({k: v["outcome"] for k, v in r.items()})


cmp = {}
for a, b in [("F-nat", "C256"), ("S+F", "F-nat"), ("S+F", "F-X+P"), ("S+F", "C256"), ("S+C", "C256")]:
    name = f"{a} vs {b}"; r, lab = contrast(a, b)
    pp = res_p["contrasts"][name]
    cmp[name] = dict(G_label=lab, P_label=pp["label"], metrics={
        mt: {k: (r[mt][k], pp["metrics"][mt][k]) for k in ["delta", "se", "df", "L", "U", "p_sup", "outcome"]} for mt in MARGIN})
    print(name, "G:", lab, "P:", pp["label"])
    for mt in MARGIN:
        print("  ", mt, " ".join(f"{k}={r[mt][k]:.5f}/{pp['metrics'][mt][k]:.5f}" for k in ["delta", "se", "df", "L", "U"]),
              r[mt]["outcome"], pp["metrics"][mt]["outcome"])
report["contrasts"] = cmp

# ---- Q-S: Holm over 4 one-sided tests
tests = {f"S+F vs {b}|{mt}": cmp[f"S+F vs {b}"]["metrics"][mt] for b in ["F-nat", "F-X+P"] for mt in MARGIN}
pv = {k: v["p_sup"][0] for k, v in tests.items()}
order = sorted(pv, key=pv.get); rej = {}; stop = False
for i, k in enumerate(order):
    if not stop and pv[k] <= 0.05 / (4 - i): rej[k] = True
    else: stop = True; rej[k] = False
helps = any(all(rej[f"S+F vs {b}|{mt}"] and tests[f"S+F vs {b}|{mt}"]["delta"][0] >= MARGIN[mt] / 2 for b in ["F-nat", "F-X+P"]) for mt in MARGIN)
not_help = cmp["S+F vs F-nat"]["G_label"] == "EQUIVALENT"
verdict = "HELPS" if helps else ("DOES NOT HELP" if not_help else "INCONCLUSIVE")
report["QS"] = dict(p=pv, holm_reject_G=rej, holm_reject_P=res_p["QS"]["holm_reject"], verdict_G=verdict, verdict_P=res_p["QS"]["verdict"])
print("QS", pv, rej, verdict, res_p["QS"])

# ---- interaction (S+F - F-nat) - (S+C - C256), report 14 A4
def interaction(mt):
    Y = {a: grid(a, mt) for a in ["S+F", "F-nat", "S+C", "C256"]}
    I = Y["S+F"].mean(1) - Y["F-nat"].mean(1) - Y["S+C"].mean(1) + Y["C256"].mean(1)
    dd = Y["S+F"].mean(0) - Y["F-nat"].mean(0) - Y["S+C"].mean(0) + Y["C256"].mean(0)
    n, D = len(I), len(dd); delta = dd.mean()
    v1 = np.var(I, ddof=1) / n; v2 = np.var(dd, ddof=1) / D
    se = np.sqrt(v1 + v2); df = (v1 + v2) ** 2 / (v1 ** 2 / (n - 1) + v2 ** 2 / (D - 1)); t = stats.t.ppf(0.95, df)
    return dict(delta=delta, se=se, df=df, L=delta - t * se, U=delta + t * se, p_sup=float(stats.t.sf(delta / se, df)))
report["interaction"] = {}
for mt in MARGIN:
    r = interaction(mt); pp = res_p["interaction"][mt]
    report["interaction"][mt] = {k: (r[k], pp[k]) for k in r}
    print("interaction", mt, " ".join(f"{k}={r[k]:.5f}/{pp[k]:.5f}" for k in r))

# ---- same statistics from the PROTOCOL table, through G's stats code (isolates stats-code agreement)
report["stats_on_protocol_table"] = {}
for a, b in [("F-nat", "C256"), ("S+F", "F-nat"), ("S+F", "F-X+P"), ("S+F", "C256")]:
    for mt in MARGIN:
        r = two_stratum(grid(a, mt, p), grid(b, mt, p), MARGIN[mt]); pp = res_p["contrasts"][f"{a} vs {b}"]["metrics"][mt]
        report["stats_on_protocol_table"][f"{a} vs {b}|{mt}"] = max(abs(r[k] - pp[k]) for k in ["delta", "se", "df", "L", "U"])
print("max |stat diff| when G's stats run on protocol table:", report["stats_on_protocol_table"])

fn = f"{OUT}/g_results.json"; assert not os.path.exists(fn)
json.dump(report, open(fn, "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
