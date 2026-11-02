"""Low-SNR replication (report 41, pre-registered). Re-runs the study's key comparisons on events NO model has ever been trained
or scored on: the 10 <= SNR < 20 tier of the same 14 days (the study used SNR >= 20 only).

Method: the frozen protocol code is used UNCHANGED. Every protocol module reads its events through m4.load(), which caches
(m, W); install() fills that cache with the low-SNR tier (same columns, same role/blk derivation as m4.load) BEFORE any protocol
module touches the data. Run commands are copied verbatim from the frozen study manifest (manifest_v2_1.json); only the tag, the
output folder (data/k_committee/replication/runs) and the server checkpoint (retrained on the low-SNR server days) change.

    .venv/bin/python research/model/R_replication/rep.py manifest            # writes rep_manifest.json (once)
    .venv/bin/python research/model/R_replication/rep.py launch              # trains every run not yet finished (resumable)
    .venv/bin/python research/model/R_replication/rep.py train <id>          # one run (used by launch, one process per run)
    .venv/bin/python research/model/R_replication/rep.py provision           # fl_system fold-0 folders for the SecAgg+ arm
    .venv/bin/python research/model/R_replication/rep.py score [threads]     # frozen evaluator on every run -> scoring/*.parquet
    .venv/bin/python research/model/R_replication/rep.py analyze             # pre-registered decisions -> results_rep_*.json
"""
import json, os, shutil, subprocess, sys, time
import numpy as np, pandas as pd
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../.."))
os.chdir(ROOT); HERE = os.path.join(ROOT, "research/model/R_replication"); PROT = os.path.join(ROOT, "research/model/K_committee/protocol")
sys.path[:0] = [PROT, ROOT, os.path.join(ROOT, "research/model/M4_methods")]
import m4
OUT = os.path.join(ROOT, "data/k_committee/replication"); RUNS = os.path.join(OUT, "runs"); MAN = os.path.join(HERE, "rep_manifest.json")
SNR_LO, SNR_HI = 10, 20
ARMS = {"C256": ("dryrun", range(10)), "C256rep": ("dryrun", range(10, 20)), "C64": ("main", range(10)), "F-nat": ("main", range(10)),
        "F-X+P": ("main", range(10)), "S+F": ("main", range(10))}
FED_REP = os.path.join(ROOT, "data/fl_system/fold0_rep10/federation.json")


def install():
    """Low-SNR tier into m4's cache; identical derivation to m4.load except the SNR band."""
    assert "m" not in m4._CACHE, "m4 cache already filled (would mix tiers)"
    from ssl_central import data as D
    m, W, _, _ = D.load_all()
    keep = ((m.snr >= SNR_LO) & (m.snr < SNR_HI)).to_numpy()
    m = m[keep].reset_index(drop=True); W = W[keep]
    day = m.date.astype(str).to_numpy()
    m["role"] = np.where(np.isin(day, m4.TEST_DAYS), "test", np.where(day == m4.VAL_DAY, "val", "train"))
    m["blk"] = pd.to_datetime(m.wall_time).dt.floor("30min").astype("int64")
    assert (m.snr < 20).all() and (m.snr >= 10).all() and len(m) == 5284, len(m)
    m4._CACHE["m"], m4._CACHE["W"] = m, W
    return m, W


def manifest():
    assert not os.path.exists(MAN), "manifest exists (never overwritten)"
    study = json.load(open(os.path.join(PROT, "manifest_v2_1.json"))); chosen = json.load(open(os.path.join(PROT, "tuned_lr.json")))["chosen"]
    out = []
    for s in range(10):                                                       # servers first (S+F needs them)
        e = next(x for x in study if x["id"] == f"v21_server_s{s}"); cli = list(e["cli"]); cli[cli.index("--tag") + 1] = f"rep10_server_s{s}"
        out.append(dict(id=f"rep10_server_s{s}", arm="server", seed=s, fold=None, cli=cli, init=None, from_study=e["id"]))
    for arm, (grp, seeds) in ARMS.items():
        for f in range(3):
            for s in seeds:
                sid = f"v21_{arm}_f{f}_s{s}"; e = next(x for x in study if x["id"] == sid); cli = list(e["cli"])
                i = cli.index("--lr") + 1
                if cli[i].startswith("TUNED:"): cli[i] = str(chosen[cli[i].split(":")[1]])
                rid = f"rep10_{arm}_f{f}_s{s}"; cli[cli.index("--tag") + 1] = rid
                out.append(dict(id=rid, arm=arm, seed=s, fold=f, cli=cli, init=(f"rep10_server_s{int(e['init'].split(':')[1])}" if e["init"] else None), from_study=sid))
    json.dump(out, open(MAN, "x"), indent=1); print(len(out), "runs ->", MAN)


def train(rid):
    install(); import fedsim3 as fs
    fs.RUNS = RUNS; os.makedirs(RUNS, exist_ok=True)
    e = next(x for x in json.load(open(MAN)) if x["id"] == rid); cli = list(e["cli"])
    if e["init"]:
        p = os.path.join(RUNS, e["init"], "final.pt"); pins = json.load(open(os.path.join(OUT, "rep_server_hashes.json")))
        assert fs.sha_file(p) == pins[e["init"]], f"{rid}: server checkpoint hash mismatch"; cli += ["--init", p]
    a = fs.parser().parse_args(["run"] + cli); summ, _, _ = fs.run(a)
    json.dump(dict(id=rid, tier=[SNR_LO, SNR_HI], n_events_tier=int(len(m4._CACHE["m"]))), open(os.path.join(RUNS, rid, "tier.json"), "x"))


def launch():
    import fedsim3 as fs
    os.makedirs(RUNS, exist_ok=True); log = open(os.path.join(OUT, "launch_log.jsonl"), "a")
    for e in json.load(open(MAN)):
        if os.path.exists(os.path.join(RUNS, e["id"], "final.pt")): continue
        if os.path.exists(os.path.join(RUNS, e["id"])):                        # interrupted run: keep its folder as a record, retrain
            os.rename(os.path.join(RUNS, e["id"]), os.path.join(RUNS, e["id"] + f".interrupted_{time.strftime('%Y%m%d_%H%M%S')}"))
        if e["arm"] != "server" and not os.path.exists(os.path.join(OUT, "rep_server_hashes.json")):
            pins = {f"rep10_server_s{s}": fs.sha_file(os.path.join(RUNS, f"rep10_server_s{s}", "final.pt")) for s in range(10)}
            json.dump(pins, open(os.path.join(OUT, "rep_server_hashes.json"), "x"), indent=1)
        t0 = time.time(); r = subprocess.run([sys.executable, os.path.abspath(__file__), "train", e["id"]], stdout=open(os.path.join(OUT, f"log_{e['id']}.txt"), "w"),
                                             stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        log.write(json.dumps(dict(id=e["id"], rc=r.returncode, secs=round(time.time() - t0), t=time.strftime("%Y-%m-%d %H:%M:%S"))) + "\n"); log.flush()
        print(e["id"], r.returncode, round(time.time() - t0), flush=True)
        assert r.returncode == 0, f"{e['id']} failed (see log)"


def provision():
    install(); sys.path.insert(0, os.path.join(ROOT, "fl_system")); import provision as P
    sys.argv = ["provision.py", "--fold", "0", "--version", "rep10"]; P.main()
    pins = json.load(open(os.path.join(OUT, "rep_server_hashes.json"))); sd = os.path.join(os.path.dirname(FED_REP), "server")
    os.rename(os.path.join(sd, "server_init_s0.pt"), os.path.join(sd, "STUDY_server_init_s0_high_snr_NOT_USED.pt"))   # provision.py copies the study server
    json.dump(dict(note="server inits = replication servers rep10_server_s0..9 (low-SNR server days); the study copy was renamed, not used",
                   pins=pins), open(os.path.join(sd, "rep_info.json"), "x"), indent=1)
    for s in range(10):
        src = os.path.join(RUNS, f"rep10_server_s{s}", "final.pt"); dst = os.path.join(os.path.dirname(FED_REP), "server", f"server_init_s{s}.pt")
        import fedsim3 as fs; assert fs.sha_file(src) == pins[f"rep10_server_s{s}"]; assert not os.path.exists(dst); shutil.copyfile(src, dst)


def score(threads=12):
    install(); import evaluator as E, evaluator2 as E2
    from concurrent.futures import ThreadPoolExecutor
    E.data(); man = [e for e in json.load(open(MAN)) if e["arm"] != "server"]; jobs = []
    for e in man:
        enc = E.load_encoder(os.path.join(RUNS, e["id"], "final.pt")); jobs.append((e["arm"], e["seed"], e["fold"], E2.embed_fold(enc, e["fold"]), True))
    for f in range(3):
        for s in range(10): jobs.append(("INIT", s, f, E2.embed_fold(E2.init_encoder(s), f), False))
    def job(j):
        arm, seed, fold, emb, gate = j; rows, info = E2.score(None, fold, do_gate=gate, emb=emb); info.pop("per_probe", None)
        return [dict(arm=arm, seed=seed, **r) for r in rows], dict(arm=arm, seed=seed, **info)
    S, M = [], []
    with ThreadPoolExecutor(threads) as ex:
        for rows, info in ex.map(job, jobs): S += rows; M.append(info)
    sec = secagg_scores(E, E2); S += sec
    d = os.path.join(HERE, "scoring"); os.makedirs(d, exist_ok=True); st = time.strftime("%Y%m%d_%H%M%S")
    for name, rows in (("scores", S), ("models", M)):
        p = os.path.join(d, f"{name}_rep10_{st}.parquet"); assert not os.path.exists(p); pd.DataFrame(rows).to_parquet(p)
    print("scored", len(jobs), "models +", len(sec), "SecAgg rows ->", d, st)


def secagg_scores(E, E2):
    rows = []
    for s in range(10):
        p = os.path.join(os.path.dirname(FED_REP), "server", "dpsim", f"rep10_secagg_s{s}", "final.pt")
        r, _ = E2.score(E.load_encoder(p), 0, do_gate=False); rows += [dict(arm="S+F_secagg", seed=s, **x) for x in r]
    return rows


def analyze():
    import evaluator as E, analyze as AN
    d = os.path.join(HERE, "scoring"); st = sorted(f for f in os.listdir(d) if f.startswith("scores_rep10_"))[-1][len("scores_rep10_"):-8]
    S = pd.read_parquet(os.path.join(d, f"scores_rep10_{st}.parquet")); M = pd.read_parquet(os.path.join(d, f"models_rep10_{st}.parquet"))
    MAIN, REP = list(range(10)), list(range(10, 20))
    c1 = AN.contrast(S, "C256rep", "C256", REP, MAIN); c3 = AN.contrast(S, "INIT", "C256", MAIN, MAIN)
    zr, zm = AN.zbar(M, "C256rep", REP), AN.zbar(M, "C256", MAIN); m_leak = E.m_leak_rule(zr, zm); L1 = E.gate_test(zr, zm, m_leak)
    valid = dict(C1_EQUIVALENT=c1["label"] == "EQUIVALENT", C3_INFERIOR=c3["label"] == "INFERIOR", gate_calibrated=L1["label"] == "PASS")
    cs = {f"{a} vs {b}": AN.contrast(S, a, b, MAIN, MAIN) for a, b in (("F-nat", "C256"), ("S+F", "F-nat"), ("S+F", "F-X+P"), ("S+F", "C256"),
                                                                        ("C64", "C256"), ("F-nat", "C64"), ("F-X+P", "C256"))}
    gate_qf = AN.gate(M, "F-nat", "C256", MAIN, MAIN, m_leak) if valid["gate_calibrated"] else None
    qs = AN.q_s(cs)
    f0 = AN.F["folds"]["0"]["eval_days"]; Y = lambda arm, k: AN.grid(S, arm, k, MAIN, f0)
    sec = {k: dict(**AN.paired(Y("S+F_secagg", k), Y("S+F", k))) for k in ("LOBES", "POLARITY", "DURATION")}
    for k in sec: sec[k]["outcome"] = E.outcome(sec[k]["L"], sec[k]["U"], E.MARGIN[k])
    sec_label = E.contrast_label({k: sec[k]["outcome"] for k in E.PRIMARY})
    claims = dict(
        QF_federated_equivalent_to_central=dict(study="EQUIVALENT, gate PASS", rep=cs["F-nat vs C256"]["label"], gate=(gate_qf or {}).get("label"),
                                                REPLICATED=cs["F-nat vs C256"]["label"] == "EQUIVALENT" and (gate_qf or {}).get("label") == "PASS"),
        QS_server_pretraining_helps=dict(study="HELPS", rep=qs["verdict"], REPLICATED=qs["verdict"] == "HELPS"),
        SF_vs_central=dict(study="NON-INFERIOR", rep=cs["S+F vs C256"]["label"], REPLICATED=cs["S+F vs C256"]["label"] in ("NON-INFERIOR", "EQUIVALENT")),
        SecAgg_equivalent_to_plain=dict(study="EQUIVALENT (report 36)", rep=sec_label, REPLICATED=sec_label == "EQUIVALENT"))
    res = dict(prereg="report 41", scores_stamp=st, validity=valid, PROCEDURE_VALID=all(valid.values()), m_leak=m_leak, L1=L1, controls=dict(C1=c1, C3=c3),
               claims=claims, contrasts=cs, gate_QF=gate_qf, QS=qs, secagg_fold0=sec,
               collapsed={a: AN.collapsed(M, a, MAIN) for a in ("C256", "C64", "F-nat", "F-X+P", "S+F")},
               sensitivity=dict(QF_seed_paired=AN.contrast(S, "F-nat", "C256", MAIN, MAIN, method="paired")["label"],
                                QS_seed_paired=AN.q_s({f"{a} vs {b}": AN.contrast(S, a, b, MAIN, MAIN, method="paired") for a, b in (("S+F", "F-nat"), ("S+F", "F-X+P"))})["verdict"]),
               means={a: {k: float(S[(S.arm == a) & (S.metric == k)].value.mean()) for k in ("LOBES", "POLARITY", "DURATION")}
                      for a in ("C256", "C256rep", "C64", "F-nat", "F-X+P", "S+F", "S+F_secagg", "INIT")})
    p = os.path.join(HERE, time.strftime("results_rep_%Y%m%d_%H%M%S.json")); json.dump(res, open(p, "x"), indent=1, default=float)
    print(json.dumps(dict(PROCEDURE_VALID=res["PROCEDURE_VALID"], **{k: (v["rep"], v["REPLICATED"]) for k, v in claims.items()})), "->", p)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "manifest": manifest()
    elif cmd == "launch": launch()
    elif cmd == "train": train(sys.argv[2])
    elif cmd == "provision": provision()
    elif cmd == "score": score(int(sys.argv[2]) if len(sys.argv) > 2 else 12)
    elif cmd == "analyze": analyze()
