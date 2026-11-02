"""Why did the server-pretraining gain disappear on the low-SNR tier? (report 43, pre-registered; follows report 42)
Same machinery as rep.py (low-SNR tier installed into m4's cache; frozen protocol code unchanged; study commands).
  H-data  the server's own data were noisy -> low-SNR clients started from the STUDY's clean servers (v21_server_s*), lr 1e-3
  H-lr    S+F's lr (1e-3, tuned on the clean tier) is wrong here -> re-tune F-nat and S+F on the low-SNR tier with the study's
          own rule (tune.select: keep 3e-4 unless another lr beats it by > 0.02 on the tuning day); rerun at the new lrs if changed
  H-client neither restores the gain -> pretraining does not help noisy clients
    rep2.py tune          8 tuning runs (fold 0, seed 0; lrs of the study grid) + selection -> rep2_tuned.json
    rep2.py plan          stage-B manifest from the frozen rule -> rep2_manifest.json
    rep2.py launch        trains every stage-B run not yet finished (resumable)
    rep2.py train <id>    one run
    rep2.py score [th]    stage-B runs + server encoders alone (clean and noisy) -> scoring/*_rep2_*.parquet
    rep2.py analyze       pre-registered decisions -> results_rep2_*.json
"""
import json, os, subprocess, sys, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import rep
ROOT, HERE, PROT, OUT, RUNS = rep.ROOT, rep.HERE, rep.PROT, rep.OUT, rep.RUNS
MAN2, TUNED2 = os.path.join(HERE, "rep2_manifest.json"), os.path.join(HERE, "rep2_tuned.json")
STUDY_RUNS = os.path.join(ROOT, "data/k_committee/protocol/runs3")
GRID = [0.00015, 0.0003, 0.0006, 0.001]                                     # the study's tuning grid (manifest group "tuning")


def study_entry(sid):
    return next(x for x in json.load(open(os.path.join(PROT, "manifest_v2_1.json"))) if x["id"] == sid)


def init_path(init):
    """('rep', s): low-SNR server rep10_server_s (pins rep_server_hashes.json); ('study', s): clean study server (pins server_hashes.json)."""
    import fedsim3 as fs
    kind, s = init
    if kind == "rep":
        p = os.path.join(RUNS, f"rep10_server_s{s}", "final.pt"); pin = json.load(open(os.path.join(OUT, "rep_server_hashes.json")))[f"rep10_server_s{s}"]
    else:
        p = os.path.join(STUDY_RUNS, f"v21_server_s{s}", "final.pt"); pin = json.load(open(os.path.join(PROT, "server_hashes.json")))[str(s)]
    assert fs.sha_file(p) == pin, f"server checkpoint hash mismatch: {init}"; return p


def entries_tune():
    out = []
    for arm, init in (("F-nat", None), ("S+F", ("rep", 0))):
        for lr in GRID:
            cli = list(study_entry(f"v21_tune_{arm}_lr{lr}")["cli"]); rid = f"rep2_tune_{arm}_lr{lr}"; cli[cli.index("--tag") + 1] = rid
            out.append(dict(id=rid, arm=arm, lr=lr, cli=cli, init=init))
    return out


def train_entry(e):
    rep.install(); import fedsim3 as fs
    fs.RUNS = RUNS; cli = list(e["cli"])
    if e["init"]: cli += ["--init", init_path(tuple(e["init"]))]
    fs.run(fs.parser().parse_args(["run"] + cli))
    json.dump(dict(id=e["id"], tier=[rep.SNR_LO, rep.SNR_HI], init=e["init"]), open(os.path.join(RUNS, e["id"], "tier.json"), "x"))


def run_all(entries, logname):
    log = open(os.path.join(OUT, logname), "a")
    for e in entries:
        if os.path.exists(os.path.join(RUNS, e["id"], "final.pt")): continue
        if os.path.exists(os.path.join(RUNS, e["id"])):
            os.rename(os.path.join(RUNS, e["id"]), os.path.join(RUNS, e["id"] + f".interrupted_{time.strftime('%Y%m%d_%H%M%S')}"))
        t0 = time.time(); r = subprocess.run([sys.executable, os.path.abspath(__file__), "train_json", json.dumps(e)],
                                             stdout=open(os.path.join(OUT, f"log_{e['id']}.txt"), "w"), stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        log.write(json.dumps(dict(id=e["id"], rc=r.returncode, secs=round(time.time() - t0), t=time.strftime("%Y-%m-%d %H:%M:%S"))) + "\n"); log.flush()
        print(e["id"], r.returncode, round(time.time() - t0), flush=True); assert r.returncode == 0, f"{e['id']} failed"


def tune():
    run_all(entries_tune(), "launch_log_rep2.jsonl")
    if os.path.exists(TUNED2): return
    rep.install(); import evaluator as E, evaluator2 as E2, tune as T
    res = dict(rule=f"study rule tune.select (keep {T.DEFAULT} unless > +{T.DELTA_SEL})", arms={})
    for e in entries_tune():
        res["arms"].setdefault(e["arm"], {})[str(e["lr"])] = E2.score_tuning(E.load_encoder(os.path.join(RUNS, e["id"], "final.pt")))
    res["chosen"] = {a: T.select({float(k): v["score"] for k, v in d.items()}) for a, d in res["arms"].items()}
    json.dump(res, open(TUNED2, "x"), indent=1); print("chosen", res["chosen"])


def plan():
    assert not os.path.exists(MAN2); ch = json.load(open(TUNED2))["chosen"]; out = []
    for f in range(3):                                                        # H-data: clean study server -> low-SNR clients, lr 1e-3
        for s in range(10):
            cli = list(study_entry(f"v21_S+F_f{f}_s{s}")["cli"]); cli[cli.index("--lr") + 1] = "0.001"; rid = f"rep2_ScleanF_f{f}_s{s}"
            cli[cli.index("--tag") + 1] = rid; out.append(dict(id=rid, arm="Sclean+F", seed=s, fold=f, cli=cli, init=["study", s]))
    lr_changed = dict(F=ch["F-nat"] != 0.0003, S=ch["S+F"] != 0.001)
    for arm, key, src, init, cur in (("F-nat", "F", "F-nat", None, 0.0003), ("S+F", "S", "S+F", "rep", 0.001)):
        if not lr_changed[key]: continue                                      # same lr as report 42: the existing runs ARE this arm
        for f in range(3):
            for s in range(10):
                cli = list(study_entry(f"v21_{src}_f{f}_s{s}")["cli"]); cli[cli.index("--lr") + 1] = str(ch[src]); rid = f"rep2_{arm}_lrT_f{f}_s{s}"
                cli[cli.index("--tag") + 1] = rid
                out.append(dict(id=rid, arm=f"{arm}@lrT", seed=s, fold=f, cli=cli, init=([init, s] if init else None)))
    json.dump(dict(chosen=ch, lr_changed=lr_changed, runs=out), open(MAN2, "x"), indent=1); print(len(out), "runs; lr_changed", lr_changed)


def launch():
    run_all(json.load(open(MAN2))["runs"], "launch_log_rep2.jsonl")


def score(threads=12):
    rep.install(); import evaluator as E, evaluator2 as E2
    from concurrent.futures import ThreadPoolExecutor
    E.data(); jobs = []
    for e in json.load(open(MAN2))["runs"]:
        jobs.append((e["arm"], e["seed"], e["fold"], E2.embed_fold(E.load_encoder(os.path.join(RUNS, e["id"], "final.pt")), e["fold"]), True))
    for f in range(3):
        for s in range(10):
            jobs.append(("S0clean", s, f, E2.embed_fold(E.load_encoder(init_path(("study", s))), f), False))
            jobs.append(("S0noisy", s, f, E2.embed_fold(E.load_encoder(init_path(("rep", s))), f), False))
    def job(j):
        arm, seed, fold, emb, gate = j; rows, info = E2.score(None, fold, do_gate=gate, emb=emb); info.pop("per_probe", None)
        return [dict(arm=arm, seed=seed, **r) for r in rows], dict(arm=arm, seed=seed, **info)
    S, M = [], []
    with ThreadPoolExecutor(threads) as ex:
        for rows, info in ex.map(job, jobs): S += rows; M.append(info)
    d = os.path.join(HERE, "scoring"); st = time.strftime("%Y%m%d_%H%M%S")
    for name, rows in (("scores", S), ("models", M)):
        p = os.path.join(d, f"{name}_rep2_{st}.parquet"); assert not os.path.exists(p); pd.DataFrame(rows).to_parquet(p)
    print("scored", len(jobs), st)


def analyze():
    import evaluator as E, analyze as AN
    d = os.path.join(HERE, "scoring")
    last = lambda pre: sorted(f for f in os.listdir(d) if f.startswith(pre))[-1]
    S = pd.concat([pd.read_parquet(os.path.join(d, last("scores_rep10_"))), pd.read_parquet(os.path.join(d, last("scores_rep2_")))], ignore_index=True)
    S = S[S.arm != "S+F_secagg"]; MAIN = list(range(10)); man = json.load(open(MAN2)); ch = man["chosen"]
    C = lambda a, b, **kw: AN.contrast(S, a, b, MAIN, MAIN, **kw)
    # H-data: the full study QS rule with the clean-server arm, plus clean vs noisy server inside S+F
    cs_d = {"S+F vs F-nat": C("Sclean+F", "F-nat"), "S+F vs F-X+P": C("Sclean+F", "F-X+P")}
    qs_d = AN.q_s(cs_d); clean_vs_noisy = C("Sclean+F", "S+F")
    h_data = qs_d["verdict"] == "HELPS" and clean_vs_noisy["metrics"]["LOBES"]["p_sup"] < 0.05
    # H-lr: re-tuned lrs; if unchanged, refuted by the study's own rule without new runs
    F_arm = "F-nat@lrT" if man["lr_changed"]["F"] else "F-nat"; S_arm = "S+F@lrT" if man["lr_changed"]["S"] else "S+F"
    if man["lr_changed"]["F"] or man["lr_changed"]["S"]:
        c = C(S_arm, F_arm); rej = E.holm([c["metrics"][k]["p_sup"] for k in E.PRIMARY])
        helps_lr = any(r and c["metrics"][k]["delta"] >= E.MARGIN[k] / 2 for r, k in zip(rej, E.PRIMARY))
        h_lr = dict(tested=True, contrast=c, holm_reject=dict(zip(E.PRIMARY, rej)), HELPS=helps_lr)
    else:
        h_lr = dict(tested=False, note="study tuning rule kept both lrs on the low-SNR tier -> H-lr refuted without new runs", HELPS=False)
    h_client = (not h_data) and (not h_lr["HELPS"]) and qs_d["verdict"] != "HELPS"
    s0 = C("S0clean", "S0noisy")
    res = dict(prereg="report 43", tuned=json.load(open(TUNED2)), lr_changed=man["lr_changed"],
               H_data=dict(SUPPORTED=h_data, QS_with_clean_server=qs_d, contrasts=cs_d, clean_vs_noisy_server_inside_SF=clean_vs_noisy),
               H_lr=h_lr, H_client=dict(SUPPORTED=h_client), S0_clean_vs_noisy_alone=s0,
               Sclean_vs_C256=C("Sclean+F", "C256")["label"],
               means={a: {k: float(S[(S.arm == a) & (S.metric == k)].value.mean()) for k in ("LOBES", "POLARITY", "DURATION")}
                      for a in sorted(S.arm.unique())})
    p = os.path.join(HERE, time.strftime("results_rep2_%Y%m%d_%H%M%S.json")); json.dump(res, open(p, "x"), indent=1, default=float)
    print(json.dumps(dict(H_data=h_data, H_lr=h_lr["HELPS"], H_lr_tested=h_lr["tested"], H_client=h_client, chosen=ch)), "->", p)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "tune": tune()
    elif cmd == "plan": plan()
    elif cmd == "launch": launch()
    elif cmd == "train_json": train_entry(json.loads(sys.argv[2]))
    elif cmd == "score": score(int(sys.argv[2]) if len(sys.argv) > 2 else 12)
    elif cmd == "analyze": analyze()
