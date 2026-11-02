"""Committee B: variance components, margins, power simulation and planted-control dry run.

    python research/model/K_committee/B_stats/stats_design.py           # needs scores_day.csv, leak.csv, pairs_day.csv

Outputs (this folder):
  varcomp.csv        per metric: sigma_seed, sigma_resid (seed x day), sigma_day, sigma_armday (from other arms), SSL gain
  margins.csv        proposed margin per metric with the numbers behind it
  power.csv          simulated P(declare) for n seeds x D days x true Delta x inference method
  controls.csv       planted-control dry run with the proposed TOST / NI procedure on the real score matrix
  controls_splits.csv  all 35 balanced 4-vs-4 central-vs-central splits
All RNGs fixed. Model per metric (balanced design, arms a, seeds s nested in arm, days d crossed with arm):
  Y[a,s,d] = mu_a + u_d + v[a,d] + w[a,s] + e[a,s,d]
"""
import os, itertools, json
import numpy as np, pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
S = pd.read_csv(os.path.join(HERE, "scores_day.csv"))
L = pd.read_csv(os.path.join(HERE, "leak.csv"))
P = pd.read_csv(os.path.join(HERE, "pairs_day.csv"))

# composite primary metrics: mean of kNN and linear probe (pre-specified)
comp = []
for t in ("polarity", "lobes", "duration"):
    x = S[S.metric.isin([f"knn_{t}", f"lin_{t}"])].groupby(["model", "arm", "seed", "day"], as_index=False).value.mean()
    x["metric"] = f"P_{t}"; comp.append(x)
S = pd.concat([S] + comp, ignore_index=True)
PRIMARY = ["P_lobes", "P_polarity", "P_duration"]
SECOND = ["knn_lobes", "lin_lobes", "knn_polarity", "lin_polarity", "knn_duration", "lin_duration", "rankme", "uniformity", "alignment"]


def mat(arm, metric, seeds=None):
    x = S[(S.arm == arm) & (S.metric == metric)]
    if seeds is not None: x = x[x.seed.isin(seeds)]
    return x.pivot_table(index="seed", columns="day", values="value").sort_index()


# ------------------------------------------------------------------ 1. variance components (central only, 8 seeds x 5 days)
def two_way(Y):
    """Two-way ANOVA without replication; returns sigma_seed^2, sigma_resid^2, sigma_day^2 (method of moments, floored at 0)."""
    n, D = Y.shape; g = Y.mean(); rs = Y.mean(1); cs = Y.mean(0)
    msS = D * ((rs - g) ** 2).sum() / (n - 1); msD = n * ((cs - g) ** 2).sum() / (D - 1)
    res = Y - rs[:, None] - cs[None, :] + g; msE = (res ** 2).sum() / ((n - 1) * (D - 1))
    return max((msS - msE) / D, 0), msE, max((msD - msE) / n, 0), msS


def armday(Yx, Yc, s2e_x, s2e_c):
    """sigma^2 of the arm x day interaction: var_d(Delta_d) - resid/nx - resid/nc, floored at 0."""
    dd = Yx.mean(0) - Yc.mean(0)
    return max(dd.var(ddof=1) - s2e_x / Yx.shape[0] - s2e_c / Yc.shape[0], 0), dd.mean()


rows = []
for met in PRIMARY + SECOND:
    Yc = mat("central", met).to_numpy(); s2w, s2e, s2d, msS = two_way(Yc)
    r = dict(metric=met, n_seeds=Yc.shape[0], n_days=Yc.shape[1], central_mean=Yc.mean(), sigma_seed=np.sqrt(s2w),
             sigma_resid=np.sqrt(s2e), sigma_day=np.sqrt(s2d), sd_seedmean=np.sqrt(msS / Yc.shape[1]))
    for arm in ("simclr_cnn", "vicreg_tf", "tf_short1k", "tf_init"):
        Yx = mat(arm, met).to_numpy(); _, s2ex, _, _ = two_way(Yx); s2v, dm = armday(Yx, Yc, s2ex, s2e)
        r[f"sigma_armday_{arm}"] = np.sqrt(s2v); r[f"delta_{arm}"] = dm
    r["ssl_gain_vs_init"] = -r["delta_tf_init"]
    r["sigma_armday_max"] = max(r[f"sigma_armday_{a}"] for a in ("simclr_cnn", "vicreg_tf", "tf_short1k"))
    rows.append(r)
VC = pd.DataFrame(rows); VC.to_csv(os.path.join(HERE, "varcomp.csv"), index=False)
print(VC[["metric", "central_mean", "sigma_seed", "sigma_resid", "sigma_day", "sigma_armday_max", "ssl_gain_vs_init",
          "delta_simclr_cnn", "delta_vicreg_tf", "delta_tf_short1k"]].round(4).to_string())


# ------------------------------------------------------------------ 2. inference procedures on a (seed x day) matrix pair
def tost_stats(YA, YC):
    """Two-stratum (seed, day) SE for Delta = mean(YA) - mean(YC). Returns delta, se_cons, df_cons, se_unb, df_unb.
    cons: SE^2 = S1(1/nA+1/nC) + S2/D  (overestimates by resid*(1/nA+1/nC)/D; conservative)
    unb : subtracts the residual term (unbiased), floored at the larger of the two strata."""
    nA, D = YA.shape; nC = YC.shape[0]
    mA, mC = YA.mean(1), YC.mean(1)
    S1 = (((mA - mA.mean()) ** 2).sum() + ((mC - mC.mean()) ** 2).sum()) / (nA + nC - 2); df1 = nA + nC - 2
    dd = YA.mean(0) - YC.mean(0); S2 = dd.var(ddof=1); df2 = D - 1
    rA = YA - mA[:, None] - YA.mean(0)[None] + YA.mean(); rC = YC - mC[:, None] - YC.mean(0)[None] + YC.mean()
    S3 = ((rA ** 2).sum() + (rC ** 2).sum()) / ((nA - 1) * (D - 1) + (nC - 1) * (D - 1)); df3 = (nA - 1 + nC - 1) * (D - 1)
    k = 1 / nA + 1 / nC
    a, b, c = S1 * k, S2 / D, S3 * k / D
    v_c = a + b; df_c = v_c ** 2 / (a ** 2 / df1 + b ** 2 / df2)
    v_u = max(a + b - c, a, b); df_u = v_u ** 2 / (a ** 2 / df1 + b ** 2 / df2 + c ** 2 / df3)
    return dd.mean(), np.sqrt(v_c), df_c, np.sqrt(v_u), df_u


def decide(delta, se, df, m, alpha=0.05):
    t = stats.t.ppf(1 - alpha, df); lo, hi = delta - t * se, delta + t * se
    return dict(delta=delta, lo90=lo, hi90=hi, equiv=bool(lo > -m and hi < m), noninf=bool(lo > -m),
                inferior=bool(hi < -m), superior=bool(lo > 0), different=bool(lo > 0 or hi < 0))


# ------------------------------------------------------------------ 3. margins
def se_design(r, n, D, s_v):
    return np.sqrt(2 * r.sigma_seed ** 2 / n + s_v ** 2 / D + 2 * r.sigma_resid ** 2 / (n * D))


GRID = [0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10]
mrows = []
for _, r in VC.iterrows():
    se0 = se_design(r, 5, 8, 0.0); sev = se_design(r, 5, 8, r.sigma_armday_max)
    # TOST power approx at Delta=0 with df ~ 10: need m >= (t_.95 + t_.90(two-sided->.95 per side ~)) * SE; use t_.95 + z_.90
    need = (stats.t.ppf(0.95, 10) + stats.norm.ppf(0.90)) * sev     # ~80 % power for TOST at Delta = 0
    floor = next((g for g in GRID if g >= need), np.nan)
    ceiling = 0.25 * abs(r.ssl_gain_vs_init) if r.metric.startswith(("P_", "knn_", "lin_")) else np.nan
    mrows.append(dict(metric=r.metric, sd_cc_n5_D8=se0, se_n5_D8_with_armday=sev, m_needed_80pct=need, floor_grid=floor,
                      ssl_gain=r.ssl_gain_vs_init, ceiling_quarter_gain=ceiling,
                      feasible=bool(np.isnan(ceiling) or floor <= ceiling)))
MG = pd.DataFrame(mrows)
# proposed margins (decision recorded here, argued in the report): primary composites
PROPOSED = {"P_lobes": 0.04, "P_polarity": 0.02, "P_duration": 0.02}
MG["proposed"] = MG.metric.map(PROPOSED)
MG.to_csv(os.path.join(HERE, "margins.csv"), index=False)
print(MG.round(4).to_string())


# ------------------------------------------------------------------ 4. power simulation
def simulate(r, m, n, D, true_delta, s_v, nsim, rng, boot_B=400, t_df=None):
    """Return P(equiv), P(noninf) for cons / unb t-procedures and hierarchical bootstrap (days, then seeds within arm)."""
    out = {k: 0 for k in ("eq_cons", "eq_unb", "eq_boot", "ni_cons", "ni_unb", "ni_boot")}
    draw = (lambda size: rng.standard_t(t_df, size) * np.sqrt((t_df - 2) / t_df)) if t_df else (lambda size: rng.standard_normal(size))
    for _ in range(nsim):
        u = draw(D) * r.sigma_day
        YC = u[None] + draw((n, 1)) * r.sigma_seed + draw((n, D)) * r.sigma_resid
        YA = true_delta + u[None] + draw(D)[None] * s_v + draw((n, 1)) * r.sigma_seed + draw((n, D)) * r.sigma_resid
        d, sc, dfc, su, dfu = tost_stats(YA, YC)
        for tag, se, df in (("cons", sc, dfc), ("unb", su, dfu)):
            x = decide(d, se, df, m); out[f"eq_{tag}"] += x["equiv"]; out[f"ni_{tag}"] += x["noninf"]
        di = rng.integers(0, D, (boot_B, D)); sa = rng.integers(0, n, (boot_B, n)); sc_ = rng.integers(0, n, (boot_B, n))
        bA = YA[sa[:, :, None], di[:, None, :]].mean((1, 2)); bC = YC[sc_[:, :, None], di[:, None, :]].mean((1, 2))
        lo, hi = np.quantile(bA - bC, [0.05, 0.95])
        out["eq_boot"] += (lo > -m) and (hi < m); out["ni_boot"] += lo > -m
    return {k: v / nsim for k, v in out.items()}


if __name__ == "__main__":
    rng = np.random.default_rng(20261009); prow = []
    NSIM = int(os.environ.get("NSIM", 2000))
    for met in PRIMARY:
        r = VC[VC.metric == met].iloc[0]; m = PROPOSED[met]
        for s_v_tag, s_v in (("armday0", 0.0), ("armday_meas", r.sigma_armday_max)):
            for n in (3, 5, 8):
                for D in (4, 6, 8):
                    for td_tag, td in (("0", 0.0), ("m/2", -m / 2), ("-m", -m), ("+m", m)):
                        res = simulate(r, m, n, D, td, s_v, NSIM, rng)
                        prow.append(dict(metric=met, margin=m, armday=s_v_tag, sigma_armday=s_v, n_seeds=n, n_days=D, true_delta=td_tag, **res))
            print("power", met, s_v_tag, flush=True)
    # robustness: heavy tails (t, 4 df) at the recommended design
    for met in PRIMARY:
        r = VC[VC.metric == met].iloc[0]; m = PROPOSED[met]
        for td_tag, td in (("0", 0.0), ("-m", -m)):
            res = simulate(r, m, 5, 8, td, r.sigma_armday_max, NSIM, rng, t_df=4)
            prow.append(dict(metric=met, margin=m, armday="armday_meas_t4", sigma_armday=r.sigma_armday_max, n_seeds=5, n_days=8, true_delta=td_tag, **res))
    PW = pd.DataFrame(prow); PW.to_csv(os.path.join(HERE, "power.csv"), index=False)

    # ------------------------------------------------------------------ 5. planted controls on the real matrix (5 days)
    crow = []
    def run(label, arm, seedsA, ref_seeds, expect):
        for met in PRIMARY:
            YA = mat(arm, met, seedsA).to_numpy(); YC = mat("central", met, ref_seeds).to_numpy()
            d, sc, dfc, su, dfu = tost_stats(YA, YC)
            x = decide(d, sc, dfc, PROPOSED[met])
            crow.append(dict(control=label, arm=arm, metric=met, nA=len(YA), nC=len(YC), margin=PROPOSED[met], se=sc, df=dfc, expect=expect, **x))
    A4, C4 = [0, 2, 4, 6], [1, 3, 5, 7]
    # C1 central vs central (disjoint seeds) -> must be equivalent
    for met in PRIMARY:
        YA = mat("central", met, A4).to_numpy(); YC = mat("central", met, C4).to_numpy(); d, sc, dfc, _, _ = tost_stats(YA, YC)
        crow.append(dict(control="C1 central{0,2,4,6} vs central{1,3,5,7}", arm="central", metric=met, nA=4, nC=4, margin=PROPOSED[met], se=sc, df=dfc,
                         expect="equiv", **decide(d, sc, dfc, PROPOSED[met])))
    run("C2 1k-step training (5 seeds) vs central (8)", "tf_short1k", None, None, "not equiv on lobes")
    run("C3 untrained init (5) vs central (8)", "tf_init", None, None, "inferior on all")
    for tau in (0.1, 0.25, 0.5, 1.0):
        run(f"C4 central + noise tau={tau} (seeds 0,2,4,6) vs central (1,3,5,7)", f"noise{tau}", A4, C4,
            "equiv" if tau == 0.1 else "fail")
    run("C6 other recipe: SimCLR-CNN (5) vs central (8)", "simclr_cnn", None, None, "descriptive")
    run("C6 other recipe: VICReg-TF (5) vs central (8)", "vicreg_tf", None, None, "descriptive")
    CT = pd.DataFrame(crow)

    # leakage controls: day-identity skill, one value per seed; NI test 'arm not leakier than central by > m_leak'
    M_LEAK = 0.03
    lrow = []
    def leak_test(label, arm, seedsA, ref, expect):
        a = L[(L.arm == arm) & (L.seed.isin(seedsA))].day_id_skill.to_numpy(); c = L[(L.arm == "central") & (L.seed.isin(ref))].day_id_skill.to_numpy()
        d = a.mean() - c.mean(); se = np.sqrt(a.var(ddof=1) / len(a) + c.var(ddof=1) / len(c))
        df = se ** 4 / ((a.var(ddof=1) / len(a)) ** 2 / (len(a) - 1) + (c.var(ddof=1) / len(c)) ** 2 / (len(c) - 1))
        hi = d + stats.t.ppf(0.95, df) * se
        lrow.append(dict(control=label, arm=arm, metric="day_id_skill", nA=len(a), nC=len(c), margin=M_LEAK, delta=d, se=se, df=df,
                         hi90=hi, lo90=d - stats.t.ppf(0.95, df) * se, not_leakier=bool(hi < M_LEAK), leakier=bool(d - stats.t.ppf(0.95, df) * se > 0), expect=expect))
    leak_test("L1 central vs central", "central", A4, C4, "not leakier")
    for kap in (0.005, 0.01, 0.02, 0.05):
        leak_test(f"L2 planted day offset kappa={kap}", f"leak{kap}", A4, C4, "leakier")
    for arm in ("simclr_cnn", "vicreg_tf", "tf_short1k", "tf_init"):
        leak_test(f"L3 {arm}", arm, range(8), range(8), "descriptive")
    LT = pd.DataFrame(lrow)
    pd.concat([CT, LT], ignore_index=True).to_csv(os.path.join(HERE, "controls.csv"), index=False)

    # CKA / kNN-overlap to central: Delta = mean cross-arm similarity - mean within-central similarity, per day
    def sim_delta(arm, seedsA, ref, col):
        names_a = [f"{arm}_s{s}" if arm != "central" else f"central_s{s}" for s in seedsA]
        if arm.startswith(("noise", "leak")): names_a = [f"{arm}_central_s{s}" for s in seedsA]
        if arm == "tf_short1k": names_a = [f"short_s{s}" for s in seedsA]
        if arm == "tf_init": names_a = [f"init_s{s}" for s in seedsA]
        if arm == "simclr_cnn": names_a = [f"cnn_s{s}" for s in seedsA]
        if arm == "vicreg_tf": names_a = [f"vicreg_s{s}" for s in seedsA]
        refn = [f"central_s{s}" for s in ref]
        x = P[P.a.isin(names_a) & P.b.isin(refn)].groupby("day")[col].mean()
        w = P[P.a.isin(refn) & P.b.isin(refn)].groupby("day")[col].mean()
        return (x - w).mean(), (x - w).min(), (x - w).max(), x.mean(), w.mean()
    srow = []
    for col in ("cka", "knn_overlap"):
        for lab, arm, sa in (("C1", "central", A4), ("C2", "tf_short1k", range(5)), ("C3", "tf_init", range(5)), ("C4", "noise0.1", A4),
                             ("C4", "noise0.25", A4), ("C4", "noise0.5", A4), ("C6", "simclr_cnn", range(5)), ("C6", "vicreg_tf", range(5))):
            d, lo, hi, x, w = sim_delta(arm, sa, C4, col)
            srow.append(dict(control=lab, arm=arm, metric=col, delta=d, day_min=lo, day_max=hi, cross=x, within_central=w))
    pd.DataFrame(srow).to_csv(os.path.join(HERE, "controls_similarity.csv"), index=False)

    # all 35 balanced 4-vs-4 central splits (dependent; descriptive pass rate)
    sp = []
    for A in itertools.combinations(range(8), 4):
        if 0 not in A: continue
        B = [s for s in range(8) if s not in A]
        for met in PRIMARY:
            YA = mat("central", met, list(A)).to_numpy(); YC = mat("central", met, B).to_numpy(); d, sc, dfc, _, _ = tost_stats(YA, YC)
            sp.append(dict(split=str(A), metric=met, **decide(d, sc, dfc, PROPOSED[met])))
    SP = pd.DataFrame(sp); SP.to_csv(os.path.join(HERE, "controls_splits.csv"), index=False)
    print(pd.concat([CT, LT], ignore_index=True)[["control", "metric", "delta", "lo90", "hi90", "equiv", "noninf", "inferior", "not_leakier", "leakier", "expect"]].round(4).to_string())
    print(pd.DataFrame(srow).round(4).to_string())
    print(SP.groupby("metric")[["equiv", "noninf"]].mean())
