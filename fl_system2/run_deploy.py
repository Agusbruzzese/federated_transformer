"""Deploy one federation on Flower's current runtime on this machine (report 30, M1): 1 SuperLink + K SuperNodes (one per site,
node-config data-path = the site's own folder), then `flwr run fl_system2 <connection>`. Stops all processes at the end.
    .venv-fl/bin/python fl_system2/run_deploy.py --fed data/fl_system/fold0_v2/federation.json --run <name> --seed 0 --rounds 3 \
        --examples 16384 --batch 64 --lr 1e-3 --init data/fl_system/fold0_v2/server/server_init_s0.pt [--resume] [--conn fl2]
        [--kill-site k --kill-at-round r]
Exit 0 only if the run finished all rounds (final.pt + last_round == rounds). Logs: <fold>/server/runs2/<run>/deploy*/.
"""
import argparse, json, os, signal, subprocess, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); BIN = os.path.join(ROOT, ".venv-fl/bin")
ENV = dict(os.environ, PATH=BIN + os.pathsep + os.environ["PATH"])


def start(cmd, log):
    return subprocess.Popen(cmd, cwd=ROOT, env=ENV, stdin=subprocess.DEVNULL, stdout=open(log, "w"), stderr=subprocess.STDOUT, start_new_session=True)


def stop(p):
    try: os.killpg(p.pid, signal.SIGTERM)
    except ProcessLookupError: pass


def teardown(procs, timeout=20):
    """SIGTERM every process group, wait, then SIGKILL survivors (stale SuperNodes otherwise re-register on the next SuperLink)."""
    for p in procs: stop(p)
    t0 = time.time()
    while time.time() - t0 < timeout and any(p.poll() is None for p in procs): time.sleep(0.5)
    for p in procs:
        if p.poll() is None:
            try: os.killpg(p.pid, signal.SIGKILL)
            except ProcessLookupError: pass


def flower_procs():
    out = subprocess.run(["ps", "-eo", "pid,args"], capture_output=True, text=True).stdout.splitlines()
    return [l.strip() for l in out if any(x in l for x in ("flower-superlink", "flower-supernode", "flwr-clientapp", "flwr-serverapp", "flower-superexec"))
            and "run_deploy" not in l]


def port_free(port):
    import socket
    with socket.socket() as s:
        return s.connect_ex(("127.0.0.1", port)) != 0


def main():
    ap = argparse.ArgumentParser()
    for k, t in (("--fed", str), ("--run", str), ("--seed", int), ("--rounds", int), ("--examples", int), ("--batch", int), ("--lr", float)):
        ap.add_argument(k, type=t, required=True)
    ap.add_argument("--init", default=""); ap.add_argument("--resume", action="store_true"); ap.add_argument("--conn", default="fl2")
    ap.add_argument("--superlink-args", default="--insecure"); ap.add_argument("--supernode-args", default="--insecure")
    ap.add_argument("--kill-site", type=int); ap.add_argument("--kill-at-round", type=int); ap.add_argument("--trace", action="store_true"); ap.add_argument("--db", default="file", choices=["file", "memory"])
    ap.add_argument("--app", default="fl_system2", choices=["fl_system2", "fl_system3", "fl_system4"]); ap.add_argument("--extra-run-config", default="")
    ap.add_argument("--register-keys", default="", help="path template of site public keys, e.g. fl_system2/pki/keys/site_{k}.pub (TLS+auth runs)")
    a = ap.parse_args(); fed = os.path.abspath(a.fed); base = os.path.dirname(fed); F = json.load(open(fed))
    left = flower_procs(); assert not left, f"refusing to start: Flower processes still running: {left[:3]}"      # pre-flight
    assert all(port_free(p) for p in (8190, 9091, 9092, 9093) + tuple(9400 + c["k"] for c in F["clients"])), "Flower ports still in use"
    out = os.path.join(base, "server", {"fl_system2": "runs2", "fl_system3": "runs3", "fl_system4": "runs4"}[a.app], a.run); d = os.path.join(out, "deploy_resume" if a.resume else "deploy"); os.makedirs(d, exist_ok=False)
    db = os.path.join(d, "superlink_state.db") if a.db == "file" else ":flwr-in-memory-state:"   # file DB: less lock contention
    sl = start([os.path.join(BIN, "flower-superlink"), "--database", db, "--port", "8190"] + a.superlink_args.split(), os.path.join(d, "superlink.log"))
    time.sleep(6); nodes = {}
    if a.register_keys:                                                  # M2: authenticated SuperNodes must be registered first
        for c in F["clients"]:
            kf = a.register_keys.replace("{k}", f"{c['k']:02d}")
            r = subprocess.run([os.path.join(BIN, "flwr"), "supernode", "register", kf, a.conn], cwd=ROOT, env=ENV, capture_output=True, text=True, stdin=subprocess.DEVNULL)
            open(os.path.join(d, "register.log"), "a").write(f"== {kf} rc {r.returncode}\n{r.stdout[-400:]}{r.stderr[-400:]}\n")
            assert r.returncode == 0, f"register failed for {kf}: {r.stderr[-300:]}"
    for c in F["clients"]:
        k = c["k"]; nc = f'data-path="{os.path.join(base, f"client_{k:02d}")}" code-root="{ROOT}"'
        if a.trace: nc += f' trace-file="{os.path.join(d, f"trace_site_{k:02d}.txt")}"'
        nodes[k] = start([os.path.join(BIN, "flower-supernode"), "--superlink", "127.0.0.1:9092", "--port", str(9400 + k), "--node-config", nc]
                         + a.supernode_args.replace("{k}", f"{k:02d}").split(), os.path.join(d, f"supernode_{k:02d}.log"))   # {k}: per-site key files
    time.sleep(8)
    rc = (f'rounds={a.rounds} examples={a.examples} batch={a.batch} lr={a.lr} seed={a.seed} fed="{fed}" '
          f'init="{os.path.abspath(a.init) if a.init else ""}" run-name="{a.run}" code-root="{ROOT}" resume={"true" if a.resume else "false"} {a.extra_run_config}'.strip())
    t0 = time.time(); run = start([os.path.join(BIN, "flwr"), "run", os.path.join(ROOT, a.app), a.conn, "--stream", "--run-config", rc], os.path.join(d, "flwr_run.log"))
    killed = None; lr_path = os.path.join(out, "last_round.json")
    while run.poll() is None:
        if a.kill_site is not None and killed is None and os.path.exists(lr_path) and json.load(open(lr_path))["round"] >= a.kill_at_round - 1:
            time.sleep(1.0); os.killpg(nodes[a.kill_site].pid, signal.SIGKILL); killed = time.time() - t0; print(f"SIGKILL SuperNode {a.kill_site} at t={killed:.1f}s", flush=True)
        txt = open(os.path.join(d, "flwr_run.log")).read()
        if "] done" in txt or "Traceback" in txt or "RuntimeError" in txt: time.sleep(3); break
        time.sleep(1)
    teardown([run] + list(nodes.values()) + [sl])
    assert not flower_procs(), f"teardown left Flower processes: {flower_procs()[:3]}"
    done = os.path.exists(os.path.join(out, "final.pt")) and json.load(open(lr_path))["round"] == a.rounds if os.path.exists(lr_path) else False
    res = dict(finished=done, wall_s=round(time.time() - t0, 1), last_round=json.load(open(lr_path))["round"] if os.path.exists(lr_path) else 0, killed_at_s=killed)
    json.dump(res, open(os.path.join(d, "deploy.json"), "x"), indent=1); print(json.dumps(res)); sys.exit(0 if done else 1)


if __name__ == "__main__":
    main()
