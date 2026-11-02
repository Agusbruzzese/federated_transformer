"""M4 privacy-utility curve (report 35, PRE-REGISTERED; sha256 recorded in report 35 before any curve run).
For each noise multiplier z (C from calib), seeds 0-2, fold 0, frozen evaluator (3 evaluation days):
  - mean scores; epsilon (delta 1e-5; zero-out and replace-one adjacency) for T = 200 rounds;
  - vs PLAIN (study v21_S+F_f0_s{s}): seed-paired two-stratum 90 % CI, outcomes at the study margins, contrast label;
  - vs INIT (the pinned server checkpoint the run starts from, i.e. no federated training): paired 90 % CI; LEARNS(metric) if L > 0;
  - vs SECAGG-ONLY (m3c_SpF_s{s} = dpsim m4v1_s{s}): mean difference (descriptive).
Summary: z_star = largest z whose label vs plain is EQUIVALENT or NON-INFERIOR (None if none), with its epsilon."""
import glob, json, os, sys, time
import numpy as np
R = os.getcwd(); sys.path[:0] = [R + "/research/model/K_committee/protocol", R, R + "/research/model/M4_methods", R + "/fl_system4"]
import evaluator as E, evaluator2 as E2, analyze as AN, dp
S = R + "/data/fl_system/fold0_v2/server"; Z = [0.0, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]; SEEDS = (0, 1, 2)
MARG = dict(LOBES=0.04, POLARITY=0.02, DURATION=0.03)
def per_day(path):
    rows, _ = E2.score(E.load_encoder(path), 0, do_gate=False); out = {}
    for r in rows: out.setdefault(r["metric"], {})[r["day"]] = r["value"]
    return out
ztag = lambda z: f"m4c_z{z:g}_s"
calib = json.load(open(sorted(glob.glob(R + "/fl_system4/calib_M4_*.json"))[0])); C = calib["C"]
ref = dict(plain={s: per_day(f"{R}/data/k_committee/protocol/runs3/v21_S+F_f0_s{s}/final.pt") for s in SEEDS},
           init={s: per_day(f"{S}/server_init_s{s}.pt") for s in SEEDS}, secagg={s: per_day(f"{S}/runs3/m3c_SpF_s{s}/final.pt") for s in SEEDS})
days = sorted(ref["plain"][0]["LOBES"]); Y = lambda D, k: np.array([[D[s][k][d] for d in days] for s in SEEDS])
res = dict(prereg="report 35", C=C, days=days, reference_means={a: {k: float(Y(D, k).mean()) for k in MARG} for a, D in ref.items()}, curve=[])
for z in Z:
    runs = {s: f"{S}/dpsim/{ztag(z)}{s}" for s in SEEDS}
    recs = {s: [json.loads(l) for l in open(runs[s] + "/rounds.jsonl")] for s in SEEDS}
    valid = {s: dict(rounds=len(recs[s]), over_clip=sum(r["n_over_clip"] for r in recs[s]), clipped_frac=float(np.mean([x["scale"] < 1 for r in recs[s] for x in r["sites"]]))) for s in SEEDS}
    A = {s: per_day(runs[s] + "/final.pt") for s in SEEDS}; row = dict(z=z, accountant=dp.accountant(200, z) if z > 0 else None, validity=valid, metrics={})
    for k, m in MARG.items():
        pv = AN.paired(Y(A, k), Y(ref["plain"], k)); pi = AN.paired(Y(A, k), Y(ref["init"], k))
        row["metrics"][k] = dict(mean=float(Y(A, k).mean()), vs_plain=dict(delta=pv["delta"], CI90=[pv["L"], pv["U"]], outcome=E.outcome(pv["L"], pv["U"], m)),
                                 vs_init=dict(delta=pi["delta"], CI90=[pi["L"], pi["U"]], learns=bool(pi["L"] > 0)),
                                 vs_secagg_only_delta=float((Y(A, k) - Y(ref["secagg"], k)).mean()))
    row["label_vs_plain"] = E.contrast_label({k: row["metrics"][k]["vs_plain"]["outcome"] for k in ("LOBES", "POLARITY")})
    row["valid"] = all(v["rounds"] == 200 for v in valid.values()); res["curve"].append(row)
    print(f"z={z:g} eps={row['accountant']['eps_zero_out'] if z > 0 else 'inf'} label={row['label_vs_plain']} "
          + " ".join(f"{k}={row['metrics'][k]['mean']:.3f}(learns={row['metrics'][k]['vs_init']['learns']})" for k in MARG), flush=True)
ok = [r for r in res["curve"] if r["label_vs_plain"] in ("EQUIVALENT", "NON-INFERIOR")]
zs = max(ok, key=lambda r: r["z"]) if ok else None
res["summary"] = dict(z_star=zs["z"] if zs else None, eps_star_zero_out=(zs["accountant"]["eps_zero_out"] if zs and zs["z"] > 0 else None),
                      eps_star_replace_one=(zs["accountant"]["eps_replace_one"] if zs and zs["z"] > 0 else None))
p = time.strftime(R + "/fl_system4/results_M4_%Y%m%d_%H%M%S.json"); json.dump(res, open(p, "x"), indent=1); print(json.dumps(res["summary"]), "->", p)
