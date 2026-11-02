"""Training-tier study (report 47, pre-registered; follows report 46). At FIXED compute (the study's budgets), should sites train on
  T20  clean events only (SNR >= 20; = the study's runs v21_*),
  T10  all events (SNR >= 10; NEW runs here), or
  TLO  noisy events only (10 <= SNR < 20; = the replication runs rep10_*)?
All models are scored with RELIABLE labels (the study's evaluator on the SNR >= 20 tier, unchanged). Same machinery as rep.py:
for training, the tier is installed into m4's cache; frozen protocol code; study commands verbatim.
    rep4.py manifest | launch | train <id> | score [threads] | analyze
"""
import json, os, subprocess, sys, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import rep
ROOT, HERE, PROT = rep.ROOT, rep.HERE, rep.PROT
OUT = os.path.join(ROOT, "data/k_committee/tier10"); RUNS = os.path.join(OUT, "runs"); MAN = os.path.join(HERE, "rep4_manifest.json")
STUDY_RUNS = os.path.join(ROOT, "data/k_committee/protocol/runs3")
ARMS = ("C256", "F-nat", "S+F")
N_T10 = 9659


def install_t10():
    m4 = rep.m4; assert "m" not in m4._CACHE
    from ssl_central import data as D
    m, W, _, _ = D.load_all(); keep = (m.snr >= 10).to_numpy(); m = m[keep].reset_index(drop=True); W = W[keep]
    day = m.date.astype(str).to_numpy()
    m["role"] = np.where(np.isin(day, m4.TEST_DAYS), "test", np.where(day == m4.VAL_DAY, "val", "train"))
    m["blk"] = pd.to_datetime(m.wall_time).dt.floor("30min").astype("int64")
    assert (m.snr >= 10).all() and len(m) == N_T10, len(m)
    m4._CACHE["m"], m4._CACHE["W"] = m, W


def manifest():
    assert not os.path.exists(MAN)
    study = json.load(open(os.path.join(PROT, "manifest_v2_1.json"))); chosen = json.load(open(os.path.join(PROT, "tuned_lr.json")))["chosen"]; out = []
    for s in range(10):
        e = next(x for x in study if x["id"] == f"v21_server_s{s}"); cli = list(e["cli"]); cli[cli.index("--tag") + 1] = f"t10_server_s{s}"
        out.append(dict(id=f"t10_server_s{s}", arm="server", seed=s, fold=None, cli=cli, init=None))
    for arm in ARMS:
        for f in range(3):
            for s in range(10):
                e = next(x for x in study if x["id"] == f"v21_{arm}_f{f}_s{s}"); cli = list(e["cli"]); i = cli.index("--lr") + 1
                if cli[i].startswith("TUNED:"): cli[i] = str(chosen[cli[i].split(":")[1]])
                rid = f"t10_{arm}_f{f}_s{s}"; cli[cli.index("--tag") + 1] = rid
                out.append(dict(id=rid, arm=arm, seed=s, fold=f, cli=cli, init=(f"t10_server_s{int(e['init'].split(':')[1])}" if e["init"] else None)))
    json.dump(out, open(MAN, "x"), indent=1); print(len(out), "runs")


def train(rid):
    install_t10(); import fedsim3 as fs
    fs.RUNS = RUNS; os.makedirs(RUNS, exist_ok=True); e = next(x for x in json.load(open(MAN)) if x["id"] == rid); cli = list(e["cli"])
    if e["init"]:
        p = os.path.join(RUNS, e["init"], "final.pt"); pin = json.load(open(os.path.join(OUT, "t10_server_hashes.json")))[e["init"]]
        assert fs.sha_file(p) == pin, "server hash mismatch"; cli += ["--init", p]
    fs.run(fs.parser().parse_args(["run"] + cli))
    json.dump(dict(id=rid, tier="SNR>=10", n_events_tier=N_T10), open(os.path.join(RUNS, rid, "tier.json"), "x"))


def launch():
    import fedsim3 as fs
    os.makedirs(RUNS, exist_ok=True); log = open(os.path.join(OUT, "launch_log.jsonl"), "a")
    for e in json.load(open(MAN)):
        if os.path.exists(os.path.join(RUNS, e["id"], "final.pt")): continue
        if os.path.exists(os.path.join(RUNS, e["id"])):
            os.rename(os.path.join(RUNS, e["id"]), os.path.join(RUNS, e["id"] + f".interrupted_{time.strftime('%Y%m%d_%H%M%S')}"))
        if e["arm"] != "server" and not os.path.exists(os.path.join(OUT, "t10_server_hashes.json")):
            json.dump({f"t10_server_s{s}": fs.sha_file(os.path.join(RUNS, f"t10_server_s{s}", "final.pt")) for s in range(10)},
                      open(os.path.join(OUT, "t10_server_hashes.json"), "x"), indent=1)
        t0 = time.time(); r = subprocess.run([sys.executable, os.path.abspath(__file__), "train", e["id"]], stdout=open(os.path.join(OUT, f"log_{e['id']}.txt"), "w"),
                                             stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        log.write(json.dumps(dict(id=e["id"], rc=r.returncode, secs=round(time.time() - t0), t=time.strftime("%Y-%m-%d %H:%M:%S"))) + "\n"); log.flush()
        assert r.returncode == 0, f"{e['id']} failed"


def score(threads=12):
    import evaluator as E, evaluator2 as E2
    from concurrent.futures import ThreadPoolExecutor
    C = E.data(); assert (C["m"].snr >= 20).all() and len(C["m"]) == 4375, "scoring must use the clean tier"
    jobs = []
    for f in range(3):
        for s in range(10):
            for arm in ARMS:
                for tier, path in (("T20", os.path.join(STUDY_RUNS, f"v21_{arm}_f{f}_s{s}")), ("T10", os.path.join(RUNS, f"t10_{arm}_f{f}_s{s}")),
                                   ("TLO", os.path.join(rep.RUNS, f"rep10_{arm}_f{f}_s{s}"))):
                    jobs.append((f"{arm}|{tier}", s, f, os.path.join(path, "final.pt")))
    def job(j):
        arm, seed, fold, p = j; rows, info = E2.score(None, fold, do_gate=arm.startswith("S+F") and not arm.endswith("TLO"), emb=E2.embed_fold(E.load_encoder(p), fold))
        info.pop("per_probe", None); return [dict(arm=arm, seed=seed, **r) for r in rows], dict(arm=arm, seed=seed, **info)
    S, M = [], []
    with ThreadPoolExecutor(threads) as ex:
        for rows, info in ex.map(job, jobs): S += rows; M.append(info)
    st = time.strftime("%Y%m%d_%H%M%S"); d = os.path.join(HERE, "scoring")
    for name, rows in (("scores", S), ("models", M)):
        p = os.path.join(d, f"{name}_rep4_{st}.parquet"); assert not os.path.exists(p); pd.DataFrame(rows).to_parquet(p)
    print("scored", len(jobs), st)


def analyze():
    import evaluator as E, analyze as AN
    d = os.path.join(HERE, "scoring"); last = lambda pre: sorted(x for x in os.listdir(d) if x.startswith(pre))[-1]
    S = pd.read_parquet(os.path.join(d, last("scores_rep4_"))); M = pd.read_parquet(os.path.join(d, last("models_rep4_"))); MAIN = list(range(10))
    C = lambda a, b: AN.contrast(S, a, b, MAIN, MAIN)
    res = dict(prereg="report 47", contrasts={}, rule={})
    for arm in ARMS:
        for a, b in (("T10", "T20"), ("TLO", "T20"), ("T10", "TLO")):
            res["contrasts"][f"{arm}: {a} vs {b}"] = C(f"{arm}|{a}", f"{arm}|{b}")
    prim = res["contrasts"]["S+F: T10 vs T20"]; gate = AN.gate(M, "S+F|T10", "S+F|T20", MAIN, MAIN, 0.03)
    L = prim["metrics"]["LOBES"]; label = prim["label"]
    if label in ("SUPERIOR",) or (label in ("EQUIVALENT", "NON-INFERIOR") and L["L"] > 0):
        rule = "USE ALL EVENTS (SNR >= 10)" if gate["label"] == "PASS" else "UNDECIDED (gain, but nuisance gate fails)"
    elif label in ("EQUIVALENT", "NON-INFERIOR"):
        rule = "NO FILTER NEEDED (noisy events harmless)" if gate["label"] == "PASS" else "FILTER TO SNR >= 20 (nuisance gate fails)"
    elif label == "INFERIOR" or L["U"] < 0:
        rule = "FILTER TO SNR >= 20"
    else:
        rule = "UNDECIDED (INCONCLUSIVE)"
    res.update(primary=dict(contrast=prim, gate=gate, RULE=rule),
               secondary_labels={k: v["label"] for k, v in res["contrasts"].items()},
               means={a: {k: float(S[(S.arm == a) & (S.metric == k)].value.mean()) for k in ("LOBES", "POLARITY", "DURATION")} for a in sorted(S.arm.unique())})
    p = os.path.join(HERE, time.strftime("results_rep4_%Y%m%d_%H%M%S.json")); json.dump(res, open(p, "x"), indent=1, default=float)
    print(json.dumps(dict(primary=label, gate=gate["label"], RULE=rule, secondary=res["secondary_labels"])), "->", p)


if __name__ == "__main__":
    c = sys.argv[1]
    {"manifest": manifest, "launch": launch, "train": lambda: train(sys.argv[2]), "score": lambda: score(int(sys.argv[2]) if len(sys.argv) > 2 else 12),
     "analyze": analyze}[c]()
