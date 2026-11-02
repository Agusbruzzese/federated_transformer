"""M3a (correctness) and M3b (confidentiality) from a dump=true SecAgg+ run (report 30).
M3a: decoded weighted mean (server) vs exact sum_k n_k u_k / sum_k n_k from the sites' true updates; bound from the quantisation.
M3b: per site and round, Pearson correlation between the masked vector the server received and the site's true update; and a
     uniformity check of the masked values (they should look like uniform noise on [0, modulus))."""
import glob, json, os, re, sys, time
import numpy as np
run, R = sys.argv[1], int(sys.argv[2]); c, Q, MOD = 0.5, 2 ** 22, 2 ** 32
base = "data/fl_system/fold0_v2"; sdir = f"{base}/server/runs3/{run}"; fed = json.load(open(f"{base}/federation.json")); K = fed["K"]
mw = json.load(open(f"{sdir}/summary.json"))["max_weight"]; step = 2 * c / Q
flat = lambda f: np.concatenate([np.load(f)[f"arr_{i}"].reshape(-1).astype(np.float64) for i in range(len([k for k in np.load(f).files if k.startswith("arr_")]))])
res = dict(run=run, clipping_range=c, quantization_range=Q, max_weight=mw, rounds=[])
for r in range(1, R + 1):
    tu = sorted(glob.glob(f"{base}/client_*/runs3/{run}/true_update_r{r:03d}_node*.npz")); assert len(tu) == K, len(tu)
    U = {int(re.search(r"node(\d+)", f).group(1)): (flat(f), int(np.load(f)["n_examples"])) for f in tu}
    n = np.array([v[1] for v in U.values()], float); exact = sum(v[1] * v[0] for v in U.values()) / n.sum()
    dec = flat(f"{sdir}/decoded_r{r:03d}.npz"); err = float(np.abs(dec - exact).max())
    sum_ratio = sum(round(v[1] / mw * Q) / Q for v in U.values())
    bound = (K * step + sum(np.abs(v[0]).max() * 0.5 / Q for v in U.values())) / sum_ratio      # stochastic rounding + weight rounding
    corrs, ks = [], []
    for nid, (u, _) in U.items():
        m = flat(f"{sdir}/masked_r{r:03d}_node{nid}.npz")[1:]   # element 0 = the masked weight factor (factor_combine)
        corrs.append(float(np.corrcoef(m, u)[0, 1]))
        ks.append(float(np.max(np.abs(np.sort(m / MOD) - (np.arange(1, len(m) + 1) - 0.5) / len(m)))))   # KS distance to Uniform(0,1)
    res["rounds"].append(dict(round=r, max_abs_err=err, bound=float(bound), rel_err=float(err / np.abs(exact).max()), err_within_bound=bool(err <= bound),
                              max_abs_corr_masked_vs_true=float(np.max(np.abs(corrs))), ks_uniform_max=float(max(ks)), n_values=int(len(u))))
res["M3a_PASS"] = bool(all(x["err_within_bound"] for x in res["rounds"]))
res["M3b_PASS"] = bool(all(x["max_abs_corr_masked_vs_true"] < 0.01 for x in res["rounds"]))
out = f"fl_system3/tests_M3ab_{time.strftime('%Y%m%d_%H%M%S')}.json"; json.dump(res, open(out, "x"), indent=1); print(json.dumps(res, indent=1), "->", out)
