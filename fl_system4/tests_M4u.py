"""M4 unit tests (report 35), CPU only. U1 clipping, U2 per-site noise, U3 noise in the decoded SecAgg+ sum, U4 accountant."""
import json, os, sys, time
import numpy as np
from scipy import integrate, stats
R = os.getcwd(); sys.path.insert(0, R + "/fl_system4"); import dp
rng = np.random.default_rng(1); shapes = [(64, 32), (215744 - 64 * 32,)]; res = {}
# U1: norm <= C after clipping (relative tol 1e-6, float32), unchanged when below C
u1 = []
for C in (0.1, 1.0, 3.0):
    for s in (0.001, 0.01, 0.1):
        d = [(rng.standard_normal(sh) * s).astype(np.float32) for sh in shapes]; n0 = float(np.sqrt(sum((x.astype(np.float64) ** 2).sum() for x in d)))
        o, info = dp.privatize(d, 0, 1, 0, 100, 100, 10, C, 0.0); n1 = float(np.sqrt(sum((x.astype(np.float64) ** 2).sum() for x in o)))
        u1.append(dict(C=C, n0=n0, n1=n1, ok=(n1 <= C * (1 + 1e-6)) and (n0 > C or all(np.array_equal(a, b) for a, b in zip(o, d)))))
res["U1"] = dict(cases=u1, PASS=all(c["ok"] for c in u1))
# U2: per-site noise sd = z*C*n_max/(n_k*sqrt(K)) within 0.5 %, mean ~ 0, normal (KS p > 0.01), deterministic for a seed
z, C, K = 1.0, 1.2, 10; zero = [np.zeros(sh, np.float32) for sh in shapes]
o, info = dp.privatize(zero, 3, 7, 0, 122, 896, K, C, z); v = np.concatenate([x.ravel() for x in o]).astype(np.float64)
o2, _ = dp.privatize(zero, 3, 7, 0, 122, 896, K, C, z); o3, _ = dp.privatize(zero, 3, 8, 0, 122, 896, K, C, z)
target = z * C * 896 / (122 * np.sqrt(K)); ks = stats.kstest(v / target, "norm")
res["U2"] = dict(target_sd=target, sample_sd=float(v.std()), rel_err=float(v.std() / target - 1), mean_over_se=float(v.mean() / (target / np.sqrt(v.size))),
                 ks_p=float(ks.pvalue), same_seed_identical=all(np.array_equal(a, b) for a, b in zip(o, o2)),
                 other_round_differs=not np.array_equal(o[1], o3[1]))
res["U2"]["PASS"] = abs(res["U2"]["rel_err"]) < 5e-3 and abs(res["U2"]["mean_over_se"]) < 4 and res["U2"]["ks_p"] > 0.01 and res["U2"]["same_seed_identical"] and res["U2"]["other_round_differs"]
# U3: decoded sum of K noisy zero-updates through the exact SecAgg+ arithmetic: sd = z*C*n_max/sum(n) within 1 %
n = [106, 122, 192, 512, 640, 640, 704, 768, 832, 896]; nmax = max(n); cr = float(dp.secagg_clipping_range(C, z, K))
sites = []; over = 0
for k in range(K):
    o, _ = dp.privatize(zero, k, 1, 0, n[k], nmax, K, C, z); q = dp.secagg_site(o, n[k], nmax, cr, 4194304, 0, k, 1); sites.append(q); over += q[2]
dec = np.concatenate([x.ravel() for x in dp.secagg_decode(sites, cr, 4194304, 4294967296)]); t3 = z * C * nmax / sum(n)
res["U3"] = dict(target_sd=t3, sample_sd=float(dec.std()), rel_err=float(dec.std() / t3 - 1), over_clip=over, clipping_range=cr, PASS=abs(dec.std() / t3 - 1) < 0.01 and over == 0)
# U4: exact GDP eps vs independent numerical integration of the privacy-loss distribution L ~ N(mu^2/2, mu^2):
#     delta(eps) = E[(1 - exp(eps - L))_+]; and eps_gdp <= eps_rdp (RDP is an upper bound)
u4 = []
for zz in (0.3, 1, 3, 10, 30):
    a = dp.accountant(200, zz); mu = a["mu_zero_out"]; e = a["eps_zero_out"]
    f = lambda L: (1 - np.exp(e - L)) * stats.norm.pdf(L, mu ** 2 / 2, mu)
    dnum = integrate.quad(f, e, mu ** 2 / 2 + 40 * mu, limit=500, points=[mu ** 2 / 2])[0]
    u4.append(dict(z=zz, mu=mu, eps=e, eps_replace_one=a["eps_replace_one"], eps_rdp=a["eps_rdp_zero_out"], delta_numeric=dnum,
                   ok=abs(dnum / 1e-5 - 1) < 1e-3 and e <= a["eps_rdp_zero_out"] + 1e-9))
res["U4"] = dict(cases=u4, PASS=all(c["ok"] for c in u4))
res["ALL_PASS"] = all(res[k]["PASS"] for k in ("U1", "U2", "U3", "U4"))
p = time.strftime(R + "/fl_system4/tests_M4u_%Y%m%d_%H%M%S.json"); json.dump(res, open(p, "x"), indent=1, default=float)
print(json.dumps({k: (v["PASS"] if isinstance(v, dict) else v) for k, v in res.items()}), "->", p)
for c in u4: print(c)
print(res["U2"], res["U3"], sep="\n")
