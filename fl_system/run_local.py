"""Launch one federation on this machine: 1 server process + K client processes (each with only its own folder).
On real separate machines, run server.py on the server host and client.py on each site with --server <host:port>.
    .venv-fl/bin/python fl_system/run_local.py --fed data/fl_system/fold0/federation.json --run <tag> --seed 0 --rounds 200 \
        --examples 2560000 --batch 64 --lr 1e-3 --init data/fl_system/fold0/server/server_init_s0.pt [--port 8080]
        [--trace] [--die-client k --die-at-round r] [--resume]
Logs: <fold>/server/runs/<run>/{server.log, client_<k>.log}; exit code 0 only if the server finished all rounds.
"""
import argparse, json, os, socket, subprocess, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); PY = os.path.join(ROOT, ".venv-fl/bin/python")


def wait_port(port, timeout=120):
    t0 = time.time()
    while time.time() - t0 < timeout:
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", port)) == 0: return True
        time.sleep(0.5)
    return False


def main():
    ap = argparse.ArgumentParser()
    for k, t in (("--fed", str), ("--run", str), ("--seed", int), ("--rounds", int), ("--examples", int), ("--batch", int), ("--lr", float)):
        ap.add_argument(k, type=t, required=True)
    ap.add_argument("--init", default=""); ap.add_argument("--port", type=int, default=8080); ap.add_argument("--trace", action="store_true")
    ap.add_argument("--die-client", type=int, default=None); ap.add_argument("--die-at-round", type=int, default=None); ap.add_argument("--resume", action="store_true")
    a = ap.parse_args(); fed = json.load(open(a.fed)); base = os.path.dirname(os.path.abspath(a.fed))
    out = os.path.join(base, "server", "runs", a.run); os.makedirs(out, exist_ok=True); sfx = "_resume" if a.resume else ""
    scmd = [PY, os.path.join(ROOT, "fl_system/server.py"), "--fed", a.fed, "--run", a.run, "--seed", str(a.seed), "--rounds", str(a.rounds),
            "--examples", str(a.examples), "--batch", str(a.batch), "--lr", str(a.lr), "--init", a.init, "--address", f"0.0.0.0:{a.port}"] + (["--resume"] if a.resume else [])
    srv = subprocess.Popen(scmd, cwd=ROOT, stdout=open(os.path.join(out, f"server{sfx}.log"), "w"), stderr=subprocess.STDOUT)
    assert wait_port(a.port), "server did not open its port"
    cl = []
    for c in fed["clients"]:
        k = c["k"]; folder = os.path.join(base, f"client_{k:02d}")
        cmd = [PY, os.path.join(ROOT, "fl_system/client.py"), "--folder", folder, "--run", a.run, "--seed", str(a.seed), "--server", f"127.0.0.1:{a.port}"]
        if a.trace: cmd += ["--trace-files", os.path.join(out, f"trace_client_{k:02d}{sfx}.txt")]
        if a.die_client == k and a.die_at_round: cmd += ["--die-at-round", str(a.die_at_round)]
        cl.append(subprocess.Popen(cmd, cwd=ROOT, stdout=open(os.path.join(out, f"client_{k:02d}{sfx}.log"), "w"), stderr=subprocess.STDOUT))
    while srv.poll() is None:                                     # watchdog: a client that exits early stops the run (no silent hang)
        dead = [k for k, p in enumerate(cl) if p.poll() is not None and p.returncode != 0]
        if dead:
            print(f"client(s) {dead} exited early; stopping server", flush=True); srv.terminate(); time.sleep(5); srv.kill(); break
        time.sleep(2)
    rc = srv.wait()
    for p in cl:
        try: p.wait(timeout=60)
        except subprocess.TimeoutExpired: p.kill()
    codes = [p.returncode for p in cl]; print(json.dumps(dict(server_rc=rc, client_rc=codes, out=out))); sys.exit(0 if rc == 0 else 1)


if __name__ == "__main__":
    main()
