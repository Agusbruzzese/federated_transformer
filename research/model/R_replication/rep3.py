"""Ceiling test (report 45, pre-registered; follows report 44). Scores the models TRAINED on the low-SNR tier with probes and labels
from the CLEAN tier (SNR >= 20; the study's own evaluator data, m4.load unchanged). No low-SNR model has ever seen an SNR >= 20 event.
Question: was the server-pretraining gain on low-SNR clients hidden by noisy low-SNR labels (reading B), or absent (reading A)?
    rep3.py score [threads]   -> scoring/scores_rep3_<stamp>.parquet
    rep3.py analyze           -> results_rep3_<stamp>.json
"""
import json, os, sys, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import rep, rep2
ROOT, HERE, RUNS = rep.ROOT, rep.HERE, rep.RUNS
ARMS = {"C256": "rep10_C256_f{f}_s{s}", "C256rep": "rep10_C256rep_f{f}_s{s10}", "F-nat": "rep10_F-nat_f{f}_s{s}", "F-X+P": "rep10_F-X+P_f{f}_s{s}",
        "S+F": "rep10_S+F_f{f}_s{s}", "Sclean+F": "rep2_ScleanF_f{f}_s{s}", "S+F@lrT": "rep2_S+F_lrT_f{f}_s{s}"}


def score(threads=12):
    import evaluator as E, evaluator2 as E2
    from concurrent.futures import ThreadPoolExecutor
    C = E.data(); assert (C["m"].snr >= 20).all() and len(C["m"]) == 4375, "evaluator must use the CLEAN tier (m4.load unchanged)"
    jobs = []
    for f in range(3):
        for s in range(10):
            for arm, pat in ARMS.items():
                sd = s + 10 if arm == "C256rep" else s
                p = os.path.join(RUNS, pat.format(f=f, s=s, s10=s + 10), "final.pt"); jobs.append((arm, sd, f, E.load_encoder(p)))
            jobs.append(("S0noisy", s, f, E.load_encoder(rep2.init_path(("rep", s)))))
            jobs.append(("S0clean", s, f, E.load_encoder(rep2.init_path(("study", s)))))
            jobs.append(("INIT", s, f, E2.init_encoder(s)))
    embs = [(a, s, f, E2.embed_fold(enc, f)) for a, s, f, enc in jobs]
    def job(j):
        arm, seed, fold, emb = j; rows, _ = E2.score(None, fold, do_gate=False, emb=emb)
        return [dict(arm=arm, seed=seed, **r) for r in rows]
    S = []
    with ThreadPoolExecutor(threads) as ex:
        for rows in ex.map(job, embs): S += rows
    p = os.path.join(HERE, "scoring", time.strftime("scores_rep3_%Y%m%d_%H%M%S.parquet")); assert not os.path.exists(p); pd.DataFrame(S).to_parquet(p)
    print("scored", len(embs), "models with clean-tier probes ->", p)


def analyze():
    import evaluator as E, analyze as AN
    d = os.path.join(HERE, "scoring"); f = sorted(x for x in os.listdir(d) if x.startswith("scores_rep3_"))[-1]; S = pd.read_parquet(os.path.join(d, f))
    MAIN, REP = list(range(10)), list(range(10, 20)); C = lambda a, b, sa=MAIN, sb=MAIN: AN.contrast(S, a, b, sa, sb)
    valid = dict(C1=C("C256rep", "C256", REP, MAIN)["label"] == "EQUIVALENT", C3=C("INIT", "C256")["label"] == "INFERIOR")
    prim = {"S+F vs F-nat": C("S+F", "F-nat"), "S+F vs F-X+P": C("S+F", "F-X+P")}; qs = AN.q_s(prim)
    reading = {"HELPS": "B (gain hidden by noisy labels)", "DOES NOT HELP": "A (no gain)"}.get(qs["verdict"], "UNDECIDED")
    sec = {}
    for arm in ("Sclean+F", "S+F@lrT"):
        cs = {"S+F vs F-nat": C(arm, "F-nat"), "S+F vs F-X+P": C(arm, "F-X+P")}; sec[arm] = dict(QS=AN.q_s(cs)["verdict"], contrasts=cs)
    means = {a: {k: float(S[(S.arm == a) & (S.metric == k)].value.mean()) for k in ("LOBES", "POLARITY", "DURATION")} for a in sorted(S.arm.unique())}
    trained = [a for a in ARMS if a != "C256rep"]
    ceiling = dict(spread_lobes_trained=float(max(means[a]["LOBES"] for a in trained) - min(means[a]["LOBES"] for a in trained)),
                   gap_untrained_to_C256_lobes=float(means["C256"]["LOBES"] - means["INIT"]["LOBES"]))
    res = dict(prereg="report 45", scores=f, validity=valid, PROCEDURE_VALID=all(valid.values()), primary_QS=qs, READING=reading if all(valid.values()) else "NOT INTERPRETABLE",
               primary_contrasts=prim, secondary=sec, others={n: C(a, b) for n, (a, b) in dict(Fnat_vs_C256=("F-nat", "C256"), S0noisy_vs_Fnat=("S0noisy", "F-nat"),
                                                                                              S0clean_vs_Fnat=("S0clean", "F-nat")).items()},
               ceiling=ceiling, means=means)
    p = os.path.join(HERE, time.strftime("results_rep3_%Y%m%d_%H%M%S.json")); json.dump(res, open(p, "x"), indent=1, default=float)
    print(json.dumps(dict(PROCEDURE_VALID=res["PROCEDURE_VALID"], QS=qs["verdict"], READING=res["READING"], secondary={a: v["QS"] for a, v in sec.items()}, ceiling=ceiling)), "->", p)


if __name__ == "__main__":
    {"score": lambda: score(int(sys.argv[2]) if len(sys.argv) > 2 else 12), "analyze": analyze}[sys.argv[1]]()
