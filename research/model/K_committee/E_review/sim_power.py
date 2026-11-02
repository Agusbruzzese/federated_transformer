"""Reviewer E: (A) re-run of B's power_fast model at D = 9 (n = 5, 10); (B) the same two-stratum TOST applied to the
ACTUAL cross-fitted design (9 days nested in 3 folds, one training run per arm x seed x fold, init seeds shared
across folds), under three readings of "S_seed" that the protocol leaves open.
    .venv/bin/python research/model/K_committee/E_review/sim_power.py
"""
import os, numpy as np, pandas as pd
from scipy import stats
HERE = os.path.dirname(os.path.abspath(__file__))
VC = pd.read_csv(os.path.join(HERE, "../B_stats/varcomp.csv")).set_index("metric")
MARG = {"P_lobes": 0.04, "P_polarity": 0.02}
rng = np.random.default_rng(20261010); NS = 4000

def tost(dd, S1, n_eff, D, m, df1):
    a, b = S1 * 2 / n_eff, dd.var(1, ddof=1) / D; se = np.sqrt(a + b)
    df = (a + b) ** 2 / (a ** 2 / df1 + b ** 2 / (D - 1)); t = stats.t.ppf(0.95, df); d = dd.mean(1)
    return (d - t * se > -m) & (d + t * se < m)

# ---------------- (A) B's model, exactly (copied from power_fast.py), D = 8, 9, 13
rowsA = []
for met, m in MARG.items():
    r = VC.loc[met]
    for sv_tag, sv in (("0", 0.0), ("half", r.sigma_armday_max / 2), ("meas", r.sigma_armday_max)):
        for n in (5, 10):
            for D in (8, 9, 13):
                for td_tag, td in (("0", 0.0), ("-m", -m)):
                    wC = rng.standard_normal((NS, n, 1)) * r.sigma_seed; wA = rng.standard_normal((NS, n, 1)) * r.sigma_seed
                    YC = wC + rng.standard_normal((NS, n, D)) * r.sigma_resid
                    YA = td + rng.standard_normal((NS, 1, D)) * sv + wA + rng.standard_normal((NS, n, D)) * r.sigma_resid
                    S1 = (YA.mean(2).var(1, ddof=1) + YC.mean(2).var(1, ddof=1)) / 2; dd = YA.mean(1) - YC.mean(1)
                    rowsA.append(dict(metric=met, sigma_v=sv_tag, n=n, D=D, true_delta=td_tag, P_equiv=tost(dd, S1, n, D, m, 2 * n - 2).mean()))
A = pd.DataFrame(rowsA); A.to_csv(os.path.join(HERE, "sim_power_A_bmodel.csv"), index=False)
print(A.pivot_table(index=["metric", "sigma_v", "n"], columns=["true_delta", "D"], values="P_equiv").round(3).to_string())

# ---------------- (B) nested design: 3 folds x 3 days, run (arm, seed, fold); seeds shared across folds with corr rho
F, Df, D = 3, 3, 9
rowsB = []
def nested(met, m, n, sv, saf, rho, sseedA, td):
    r = VC.loc[met]; ss = r.sigma_seed
    u = rng.standard_normal((NS, 1, F, Df)) * r.sigma_day                              # day effects (cancel in pairing)
    def runs(s):                                                                       # run effect, corr rho across folds
        return (np.sqrt(rho) * rng.standard_normal((NS, n, 1, 1)) + np.sqrt(1 - rho) * rng.standard_normal((NS, n, F, 1))) * s
    YC = u + runs(ss) + rng.standard_normal((NS, n, F, Df)) * r.sigma_resid
    YA = td + u + rng.standard_normal((NS, 1, F, Df)) * sv + rng.standard_normal((NS, 1, F, 1)) * saf + runs(sseedA) \
         + rng.standard_normal((NS, n, F, Df)) * r.sigma_resid
    dd = (YA.mean(1) - YC.mean(1)).reshape(NS, D)
    res = {}
    # P5: seed = init seed, mean over all 9 days (n = 5)
    S1 = (YA.mean((2, 3)).var(1, ddof=1) + YC.mean((2, 3)).var(1, ddof=1)) / 2
    res["P5_seedmean"] = tost(dd, S1, n, D, m, 2 * n - 2).mean()
    # P15 naive: 15 seed-fold run means (each over its fold's 3 days), not centred within fold
    rA, rC = YA.mean(3).reshape(NS, n * F), YC.mean(3).reshape(NS, n * F)
    S1 = (rA.var(1, ddof=1) + rC.var(1, ddof=1)) / 2
    res["P15_naive"] = tost(dd, S1, n * F, D, m, 2 * (n * F) - 2).mean()
    # P15 centred within fold (fold/day effects removed), treated as 15 independent units
    cA = YA.mean(3) - YA.mean((1, 3), keepdims=True)[..., 0]; cC = YC.mean(3) - YC.mean((1, 3), keepdims=True)[..., 0]
    S1 = ((cA ** 2).sum((1, 2)) + (cC ** 2).sum((1, 2))) / (2 * (n * F - F))
    res["P15_centred"] = tost(dd, S1, n * F, D, m, 2 * (n * F - F)).mean()
    return res
for met, m in MARG.items():
    r = VC.loc[met]
    for n in (5, 10):
        for sv in (0.0, round(r.sigma_armday_max / 2, 4)):
            for saf in (0.0, m / 4, m / 2):
                for rho in (0.0, 1.0):
                    for sA in (r.sigma_seed, 0.03):
                        for td_tag, td in (("0", 0.0), ("-m", -m)):
                            res = nested(met, m, n, sv, saf, rho, sA, td)
                            rowsB.append(dict(metric=met, n=n, sigma_v=sv, sigma_armfold=saf, rho_seed_across_folds=rho,
                                              sigma_seed_armA=round(sA, 4), true_delta=td_tag, **res))
B = pd.DataFrame(rowsB); B.to_csv(os.path.join(HERE, "sim_power_B_nested.csv"), index=False)
pd.set_option("display.width", 250); print(B.round(3).to_string(index=False))
