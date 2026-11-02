"""Scoring driver (report 14, A1/A2; steps 8 and 11). Frozen before training. Produces tidy tables (never overwritten):
  scores_<mode>_<stamp>.parquet  (arm, seed, fold, day, metric, value)
  models_<mode>_<stamp>.parquet  (arm, seed, fold, erank, Z, Z_unc, Q, Q_null_q95, gate fits, lin_converged)
Modes:
  dryrun  C256 seeds 0-9 and C256rep seeds 10-19 (each scored on its own fold, with gate); INIT seeds 0-9 (scores only);
          planted controls on C256rep embeddings: C4 (noise tau 0.5), C5 (tau* by bisection on fold 0 so that LOBES
          Delta(fold 0, C5 vs C256) = -1.25 m), L2_<kappa> (per-evaluation-day offsets, gate only)
  main    every main-group arm (seeds 0-9) and ablation arm (seeds 0-4) on its own fold, with gate (ablations without);
          S0 = server checkpoints 0-9 on every fold (descriptive, no gate); 15-NN overlap with C256 of the same seed/fold
    python score_all.py dryrun|main [threads=12]
"""
import json, os, sys, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "../../../.."))
os.chdir(ROOT); sys.path[:0] = [HERE, ROOT, os.path.join(ROOT, "research/model/M4_methods")]
import evaluator as E, evaluator2 as E2, analyze as AN, launch as LA
assert LA.fs.sha_file(E.FOLDS) == LA.fs.PIN_FOLDS, "folds.json differs from the pinned hash"
MANIFEST = {e["id"]: e for e in json.load(open(LA.MAN))}
TAU_C4, KAPPAS = 0.5, [0.005, 0.01, 0.02, 0.03, 0.05, 0.1]
MAIN_ARMS = ("C64", "F-nat", "F-IID", "F-X+P", "S+F", "S+C", "CJ"); ABL = ("F-nat-reset", "F-nat-R50")


def noise(tau, seed):
    def t(Etr, Eev, _days):
        r = np.random.default_rng(seed); sd = Etr.std(0)
        return Etr + tau * sd * r.standard_normal(Etr.shape), Eev + tau * sd * r.standard_normal(Eev.shape)
    return t


def offset(kappa, fold):
    def t(Etr, Eev, days):
        r = np.random.default_rng(500 + fold); u = {d: (lambda v: v / np.linalg.norm(v))(r.standard_normal(Eev.shape[1])) for d in sorted(set(days))}
        return Etr, Eev + kappa * np.linalg.norm(Etr.std(0)) * np.stack([u[d] for d in days])
    return t


def verify_run(tag):
    """Reviewer F, F2: the checkpoint must be exactly the manifest entry: same resolved CLI (tuned lr, pinned server init),
    no --max-rounds truncation, rounds_run == R, init hash == pinned server hash, final hash == sha256(final.pt)."""
    e = MANIFEST[tag]; want = vars(LA.fs.parser().parse_args(["run"] + LA.resolve(e))); s = E2.run_info(tag); got = s["args"]
    for k, v in want.items():
        if k in ("plan_only",): continue
        assert (abs(float(got[k]) - float(v)) < 1e-15) if k in ("lr", "server_lr") else got[k] == v, (tag, k, got[k], v)
    assert got["max_rounds"] == 0 and s["rounds_run"] == want["rounds"], (tag, "truncated run")
    if e["init"]:
        pins = json.load(open(os.path.join(LA.HERE, "server_hashes.json"))); assert s["provenance"]["init"] == pins[e["init"].split(":")[1]], (tag, "init")
    assert s["provenance"]["final"] == LA.fs.sha_file(os.path.join(E2.RUNS, tag, "final.pt")), (tag, "final.pt hash")


def emb_of(tag_or_seed, fold, init=False):
    enc = E2.init_encoder(tag_or_seed) if init else E.load_encoder(os.path.join(E2.RUNS, tag_or_seed, "final.pt"))
    if not init:
        verify_run(tag_or_seed)
        a = E2.run_info(tag_or_seed)["args"]; assert a["arm"] == "server" or str(a["fold"]) == str(fold), (tag_or_seed, fold)
    return E2.embed_fold(enc, fold)


def job(arm, seed, fold, emb, transform=None, do_gate=True, probes=True):
    if probes:
        rows, info = E2.score(None, fold, transform=transform, do_gate=do_gate, emb=emb)
    else:                                                     # gate only (L2)
        tr, ev, days, Etr, Eev = emb; Etr, Eev = transform(Etr, Eev, E.data()["day"][ev]) if transform else (Etr, Eev)
        rows, info = [], dict(fold=int(fold), erank=E2.erank(Eev)); info.update(E2.gate_core2(Eev, *E.gate_arrays(ev)))
    info.pop("per_probe", None)
    return [dict(arm=arm, seed=seed, **r) for r in rows], dict(arm=arm, seed=seed, **info)


def run_jobs(jobs, threads):
    S, M = [], []
    with ThreadPoolExecutor(threads) as ex:
        for rows, info in ex.map(lambda j: job(*j[:4], **j[4]), jobs): S += rows; M.append(info)
    return S, M


def dryrun(threads):
    E.data(); EMB = {}
    for f in range(3):
        for s in range(20): EMB["C256" if s < 10 else "C256rep", s, f] = emb_of(f"v21_{'C256' if s < 10 else 'C256rep'}_f{f}_s{s}", f)
        for s in range(10): EMB["INIT", s, f] = emb_of(s, f, init=True)
    jobs = [(a, s, f, EMB[a, s, f], {}) for (a, s, f) in EMB if a != "INIT"] + [("INIT", s, f, EMB["INIT", s, f], dict(do_gate=False)) for s in range(10) for f in range(3)]
    jobs += [("C4", s, f, EMB["C256rep", s, f], dict(transform=noise(TAU_C4, 100 + 10 * s + f), do_gate=False)) for s in range(10, 20) for f in range(3)]
    S, M = run_jobs(jobs, threads)
    # C5: bisection on fold 0 (rep + noise vs C256 main), then all folds
    base = pd.DataFrame(S); yb = base[(base.arm == "C256") & (base.fold == 0) & (base.metric == "LOBES")].value.mean()
    target = -1.25 * E.MARGIN["LOBES"]; lo, hi = 0.0, 0.5; trace = []
    for _ in range(8):
        tau = (lo + hi) / 2
        s5, _ = run_jobs([("C5", s, 0, EMB["C256rep", s, 0], dict(transform=noise(tau, 100 + 10 * s), do_gate=False)) for s in range(10, 20)], threads)
        d = pd.DataFrame(s5).query("metric == 'LOBES'").value.mean() - yb; trace.append((tau, float(d)))
        lo, hi = (tau, hi) if d > target else (lo, tau)
    tau5 = (lo + hi) / 2
    s5, _ = run_jobs([("C5", s, f, EMB["C256rep", s, f], dict(transform=noise(tau5, 100 + 10 * s + f), do_gate=False)) for s in range(10, 20) for f in range(3)], threads)
    S += s5
    s2, m2 = run_jobs([(f"L2_{k}", s, f, EMB["C256rep", s, f], dict(transform=offset(k, f), probes=False)) for k in KAPPAS for s in range(10, 20) for f in range(3)], threads)
    return S, M + m2, dict(tau_c5=tau5, bisection=trace)


def main_mode(threads):
    E.data(); jobs = []; overlap = []
    for f in range(3):
        for s in range(10):
            ref = emb_of(f"v21_C256_f{f}_s{s}", f)
            for a in MAIN_ARMS:
                e = emb_of(f"v21_{a}_f{f}_s{s}", f); jobs.append((a, s, f, e, {}))
                overlap.append(dict(arm=a, seed=s, fold=f, knn15_overlap_with_C256=E2.knn_overlap(e[4], ref[4])))
            rep = emb_of(f"v21_C256rep_f{f}_s{s + 10}", f)
            overlap.append(dict(arm="C256rep(s+10)", seed=s, fold=f, knn15_overlap_with_C256=E2.knn_overlap(rep[4], ref[4])))
            jobs.append(("S0", s, f, emb_of(f"v21_server_s{s}", f), dict(do_gate=False)))
        for s in range(5):
            for a in ABL: jobs.append((a, s, f, emb_of(f"v21_{a}_f{f}_s{s}", f), dict(do_gate=False)))
    S, M = run_jobs(jobs, threads)
    return S, M, dict(overlap=overlap)


if __name__ == "__main__":
    mode = sys.argv[1]; th = int(sys.argv[2]) if len(sys.argv) > 2 else 12; stamp = time.strftime("%Y%m%d_%H%M%S"); t0 = time.time()
    S, M, extra = dryrun(th) if mode == "dryrun" else main_mode(th)
    out = os.path.join(HERE, "scoring"); os.makedirs(out, exist_ok=True)
    for name, rows in (("scores", S), ("models", M)):
        p = os.path.join(out, f"{name}_{mode}_{stamp}.parquet"); assert not os.path.exists(p); pd.DataFrame(rows).to_parquet(p)
    json.dump(dict(extra, secs=round(time.time() - t0)), open(os.path.join(out, f"extra_{mode}_{stamp}.json"), "x"), indent=1, default=str)
    print("done", mode, stamp, round(time.time() - t0), "s")
