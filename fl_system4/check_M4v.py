"""M4 validity checks (report 35). usage: check_M4v.py v1 | calib | v2
v1:    dpsim (no DP) seeds 0-2 vs the REAL SecAgg+ runs m3c_SpF_s{s}: global vectors r1-3, 50, 100, 150, 200 and final.pt, max |diff| must be 0.0.
calib: C = median of all site-round update L2 norms of m4v1_s0 (= real m3c seed 0); written to fl_system4/calib_M4_<time>.json.
v2:    REAL fl_system4 (SecAgg+ + DP, 3 rounds) vs dpsim with the same config: global r1-3 and final.pt, max |diff| must be 0.0."""
import glob, json, os, sys, time
import numpy as np, torch
R = os.getcwd(); S = R + "/data/fl_system/fold0_v2/server"; what = sys.argv[1]
def sd_diff(a, b):
    A, B = torch.load(a, map_location="cpu"), torch.load(b, map_location="cpu")
    return max(float((A[m][k] - B[m][k]).abs().max()) for m in ("enc", "head") for k in A[m])
def gdiff(a, b): return float((torch.load(a) - torch.load(b)).abs().max())
if what == "v1":
    res = {}
    for s in (0, 1, 2):
        sim, real = f"{S}/dpsim/m4v1_s{s}", f"{S}/runs3/m3c_SpF_s{s}"
        d = {f"r{r:03d}": gdiff(f"{sim}/global_r{r:03d}.pt", f"{real}/global_r{r:03d}.pt") for r in (1, 2, 3, 50, 100, 150, 200)}
        d["final"] = sd_diff(sim + "/final.pt", real + "/final.pt"); res[s] = d
    out = dict(per_seed=res, V1_PASS=all(v == 0.0 for d in res.values() for v in d.values()))
elif what == "calib":
    norms = [s["norm"] for l in open(f"{S}/dpsim/m4v1_s0/rounds.jsonl") for s in json.loads(l)["sites"]]
    out = dict(rule="median of all site-round update L2 norms, dpsim m4v1_s0 (report 35)", n=len(norms), C=float(np.median(norms)),
               quantiles={q: float(np.quantile(norms, q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9)})
elif what == "v2":
    sim, real = f"{S}/dpsim/m4v2_sim_s0", f"{S}/runs4/m4v2_real_s0"
    d = {f"r{r:03d}": gdiff(f"{sim}/global_r{r:03d}.pt", f"{real}/global_r{r:03d}.pt") for r in (1, 2, 3)}; d["final"] = sd_diff(sim + "/final.pt", real + "/final.pt")
    rs = [json.loads(l) for l in open(real + "/rounds.jsonl")]; ss = [json.loads(l) for l in open(sim + "/rounds.jsonl")]
    noise_match = all(abs(a["noise_sd"] - b["noise_sd"]) == 0 and abs(a["scale"] - b["scale"]) == 0 for x, y in zip(rs, ss) for a, b in zip(x["sites"], y["sites"]))
    out = dict(diffs=d, per_site_scale_and_noise_identical=noise_match, real_summary_accountant=json.load(open(real + "/summary.json"))["accountant"],
               V2_PASS=all(v == 0.0 for v in d.values()) and noise_match)
p = time.strftime(f"{R}/fl_system4/{what}_M4_%Y%m%d_%H%M%S.json"); json.dump(out, open(p, "x"), indent=1); print(json.dumps(out)[:600], "->", p)
