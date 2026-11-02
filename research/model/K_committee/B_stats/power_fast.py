"""Committee B: fast vectorised power simulation (two-stratum TOST, conservative SE, Satterthwaite df).
Uses varcomp.csv (central SimCLR-TF M5 seeds 0-4 x 5 held-out days). Writes power_fast.csv.
    python research/model/K_committee/B_stats/power_fast.py
"""
import os, numpy as np, pandas as pd
from scipy import stats
HERE = os.path.dirname(os.path.abspath(__file__))
VC = pd.read_csv(os.path.join(HERE, "varcomp.csv")).set_index("metric")
MARG = {"P_lobes": 0.04, "P_polarity": 0.02, "P_duration": 0.03, "knn_lobes": 0.04}
rng = np.random.default_rng(20261010); NS = 4000; rows = []
for met, m in MARG.items():
    r = VC.loc[met]
    for sv_tag, sv in (("0", 0.0), ("half", r.sigma_armday_max / 2), ("meas", r.sigma_armday_max)):
        for n in (3, 5, 8, 10):
            for D in (3, 4, 6, 8, 13):
                for td_tag, td in (("0", 0.0), ("-m/2", -m / 2), ("-m", -m)):
                    sh = (NS,)
                    wC = rng.standard_normal((NS, n, 1)) * r.sigma_seed; wA = rng.standard_normal((NS, n, 1)) * r.sigma_seed
                    YC = wC + rng.standard_normal((NS, n, D)) * r.sigma_resid
                    YA = td + rng.standard_normal((NS, 1, D)) * sv + wA + rng.standard_normal((NS, n, D)) * r.sigma_resid
                    mA, mC = YA.mean(2), YC.mean(2)
                    S1 = (mA.var(1, ddof=1) + mC.var(1, ddof=1)) / 2
                    dd = YA.mean(1) - YC.mean(1); S2 = dd.var(1, ddof=1)
                    a, b = S1 * 2 / n, S2 / D; se = np.sqrt(a + b); df = (a + b) ** 2 / (a ** 2 / (2 * n - 2) + b ** 2 / (D - 1))
                    t = stats.t.ppf(0.95, df); d = dd.mean(1); lo, hi = d - t * se, d + t * se
                    rows.append(dict(metric=met, margin=m, armday=sv_tag, sigma_armday=sv, n_seeds=n, n_days=D, true_delta=td_tag,
                                     P_equiv=np.mean((lo > -m) & (hi < m)), P_noninf=np.mean(lo > -m), median_halfwidth=np.median(t * se)))
P = pd.DataFrame(rows); P.to_csv(os.path.join(HERE, "power_fast.csv"), index=False)
for met in MARG:
    x = P[(P.metric == met)]
    print(met, MARG[met]); print(x.pivot_table(index=["armday", "n_seeds"], columns=["true_delta", "n_days"], values="P_equiv").round(2).to_string())
