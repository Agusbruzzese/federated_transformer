"""Run manifest entries only (report 14 A1). Before training each entry: resolve lr (tuned_lr.json) and server init
(server_hashes.json, sha checked), build the plan with fedsim3 --plan-only and check K, batch rule, rounds and
examples within tolerance. Skips entries already completed; never overwrites. Appends to launch_log.jsonl.
    python launch.py check                        # plan-check every entry (lr placeholders -> 3e-4 for the check)
    python launch.py run <group> [jobs=4] [ids...]
    python launch.py pin-servers                  # after the server group: write server_hashes.json (refuses overwrite)
"""
import json, os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "../../../.."))
os.chdir(ROOT); sys.path[:0] = [HERE]                         # fedsim3 resolves paths from the repo root
import fedsim3 as fs
PY = os.path.join(ROOT, ".venv/bin/python"); MAN = os.path.join(HERE, "manifest_v2_1.json")
LOG = os.path.join(HERE, "launch_log.jsonl")


def resolve(e, for_check=False):
    cli = list(e["cli"]); i = cli.index("--lr") + 1
    if cli[i].startswith("TUNED:"):
        if for_check: cli[i] = "3e-4"
        else:
            t = json.load(open(os.path.join(HERE, "tuned_lr.json")))["chosen"]; cli[i] = str(t[cli[i].split(":")[1]])
    if e["init"]:
        s = int(e["init"].split(":")[1]); path = os.path.join(fs.RUNS, f"v21_server_s{s}", "final.pt")
        if not for_check:
            pins = json.load(open(os.path.join(HERE, "server_hashes.json")))
            assert fs.sha_file(path) == pins[str(s)], f"{e['id']}: server checkpoint hash mismatch"
        cli += ["--init", path]
    return cli


def check_plan(e, cli):
    plan, _, _ = fs.run(fs.parser().parse_args(["run", "--plan-only"] + cli)); x = e["expect"]
    assert plan["K"] == x["K"], (e["id"], "K", plan["K"])
    assert all(b == min(x["batch"], n) for b, n in zip(plan["b"], plan["n"])), (e["id"], "batch rule")
    assert abs(plan["examples_total"] / x["examples"] - 1) <= x["tolerance"], (e["id"], "examples", plan["examples_total"])
    return plan


def done(e): return os.path.exists(os.path.join(fs.RUNS, e["id"], "summary.json"))


def run_one(e):
    cli = resolve(e); check_plan(e, cli); t0 = time.time()
    r = subprocess.run([PY, os.path.join(HERE, "fedsim3.py"), "run"] + cli, cwd=ROOT, capture_output=True, text=True)
    rec = dict(id=e["id"], rc=r.returncode, secs=round(time.time() - t0, 1), cli=cli, when=time.strftime("%F %T"), err=r.stderr[-2000:] if r.returncode else "")
    with open(LOG, "a") as f: f.write(json.dumps(rec) + "\n")
    print(e["id"], "rc", r.returncode, f"{rec['secs']}s", flush=True); return rec


def main():
    cmd = sys.argv[1]; man = json.load(open(MAN))
    if cmd == "check":
        seen = {}
        for e in man:
            key = (e["arm"], e["fold"], e["group"] == "tuning")
            if key not in seen: seen[key] = check_plan(e, resolve(e, for_check=True))["examples_total"]
        print(f"plan check passed for {len(seen)} distinct (arm, fold) configurations covering {len(man)} entries")
    elif cmd == "pin-servers":
        p = os.path.join(HERE, "server_hashes.json")
        pins = {str(e["seed"]): fs.sha_file(os.path.join(fs.RUNS, e["id"], "final.pt")) for e in man if e["group"] == "server"}
        json.dump(pins, open(p, "x"), indent=1); print(p)
    elif cmd == "run":
        group = sys.argv[2]; jobs = int(sys.argv[3]) if len(sys.argv) > 3 else 4; ids = set(sys.argv[4:])
        todo = [e for e in man if e["group"] == group and (not ids or e["id"] in ids) and not done(e)]
        print(f"{len(todo)} runs to do in group {group}", flush=True)
        with ThreadPoolExecutor(jobs) as ex: res = list(ex.map(run_one, todo))
        bad = [r["id"] for r in res if r["rc"]]; print("failed:", bad)


if __name__ == "__main__":
    main()
