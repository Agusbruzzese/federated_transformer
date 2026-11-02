"""Run one federation as real containers (report 28): 1 server + K clients, each its own container (filesystem + network namespace),
on a private Docker network; data mounted per container (server: federation.json + its server folder; client k: only its folder).
Writes a NEW compose file per run (never overwrites; the legacy docker-compose.yml is not touched).
    sg docker -c ".venv-fl/bin/python fl_system/containers/run_compose.py --fed data/fl_system/fold0_v2/federation.json --run <name> \
        --seed 0 --rounds 3 --examples 16384 --batch 64 --lr 1e-3 --init server_init_s0.pt [--resume] [--kill-client k --kill-at-round r]"
Exit 0 only if the server finished all rounds. Logs + net samples go to <fold>/server/runs/<run>/containers*/.
"""
import argparse, json, os, re, subprocess, sys, threading, time
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); IMAGE = "fl-base:2026-10-13"
PYFL = f"{ROOT}/.venv-fl/bin/python"
RO = [f"{ROOT}/.venv-fl", f"{ROOT}/.venv", f"{ROOT}/fl_system", f"{ROOT}/ssl_central"]


def sh(cmd, **kw): return subprocess.run(cmd, shell=True, capture_output=True, text=True, **kw)


def compose_yaml(project, a, fed, base, clients):
    vol = lambda p, mode="ro": f"      - {p}:{p}:{mode}"
    common = ["    image: " + IMAGE, '    user: "1000:1000"', f"    working_dir: {ROOT}", "    networks: [flnet]", "    gpus: all"]
    srv_cmd = [PYFL, f"{ROOT}/fl_system/server.py", "--fed", fed, "--run", a.run, "--seed", str(a.seed), "--rounds", str(a.rounds),
               "--examples", str(a.examples), "--batch", str(a.batch), "--lr", str(a.lr), "--init", os.path.join(base, "server", a.init),
               "--address", "0.0.0.0:8080"] + (["--resume"] if a.resume else [])
    y = [f"name: {project}", "networks:", "  flnet:", "    driver: bridge", "    internal: true", "services:", "  flserver:"] + common + [
        "    command: " + json.dumps(srv_cmd), "    volumes:"] + [vol(p) for p in RO] + [vol(fed), vol(os.path.join(base, "server"), "rw")] + [
        "    healthcheck:", '      test: ["CMD", "bash", "-c", "echo > /dev/tcp/127.0.0.1/8080"]', "      interval: 2s", "      retries: 60"]
    for c in clients:
        k = c["k"]; folder = os.path.join(base, f"client_{k:02d}")
        cmd = [PYFL, f"{ROOT}/fl_system/client.py", "--folder", folder, "--run", a.run, "--seed", str(a.seed), "--server", "flserver:8080"]
        y += [f"  client_{k:02d}:"] + common + ["    command: " + json.dumps(cmd), "    depends_on:", "      flserver:", "        condition: service_healthy",
                                                 "    volumes:"] + [vol(p) for p in RO] + [vol(folder, "rw")]
    return "\n".join(y) + "\n"


def main():
    ap = argparse.ArgumentParser()
    for k, t in (("--fed", str), ("--run", str), ("--seed", int), ("--rounds", int), ("--examples", int), ("--batch", int), ("--lr", float), ("--init", str)):
        ap.add_argument(k, type=t, required=True)
    ap.add_argument("--resume", action="store_true"); ap.add_argument("--kill-client", type=int); ap.add_argument("--kill-at-round", type=int)
    a = ap.parse_args(); fed = os.path.abspath(a.fed); base = os.path.dirname(fed); F = json.load(open(fed))
    out = os.path.join(base, "server", "runs", a.run); os.makedirs(out, exist_ok=True)
    tag = "containers_resume" if a.resume else "containers"; cdir = os.path.join(out, tag); os.makedirs(cdir, exist_ok=False)
    project = re.sub(r"[^a-z0-9_-]", "", f"fl-{a.run}{'-resume' if a.resume else ''}".lower())
    yml = os.path.join(cdir, "compose.yml"); open(yml, "x").write(compose_yaml(project, a, fed, base, F["clients"]))
    t0 = time.time(); r = sh(f"docker compose -f {yml} up -d"); assert r.returncode == 0, r.stderr
    srv = f"{project}-flserver-1"; net = []; killed = None; ips = {}; dead_since = None
    while True:
        st = sh(f"docker inspect -f '{{{{.State.Status}}}} {{{{.State.ExitCode}}}}' {srv}").stdout.split()
        if st and st[0] == "running":
            if not ips:                                                      # capture IPs while the containers run
                ips = {svc: sh(f"docker inspect -f '{{{{range .NetworkSettings.Networks}}}}{{{{.IPAddress}}}}{{{{end}}}}' {project}-{svc}-1").stdout.strip()
                       for svc in ["flserver"] + [f"client_{c['k']:02d}" for c in F["clients"]]}
            s = sh(f"docker exec {srv} cat /sys/class/net/eth0/statistics/rx_bytes /sys/class/net/eth0/statistics/tx_bytes").stdout.split()
            lr = json.load(open(os.path.join(out, "last_round.json")))["round"] if os.path.exists(os.path.join(out, "last_round.json")) else 0
            if len(s) == 2: net.append(dict(t=round(time.time() - t0, 1), rx=int(s[0]), tx=int(s[1]), last_round=lr))
            if a.kill_client is not None and killed is None and lr >= a.kill_at_round - 1:
                # round kill_at_round - 1 is committed, so round kill_at_round is under way: kill the client container mid-round
                time.sleep(0.3); killed = sh(f"docker kill {project}-client_{a.kill_client:02d}-1").returncode
                print(f"docker kill client_{a.kill_client:02d} after committed round {lr} (rc {killed})", flush=True)
        elif st:
            break
        dead = [c for c in [f"client_{c['k']:02d}" for c in F["clients"]]
                if sh(f"docker inspect -f '{{{{.State.Status}}}}' {project}-{c}-1").stdout.strip() == "exited"]
        if dead and st and st[0] == "running":                               # backup watchdog (same as run_local.py)
            dead_since = dead_since or time.time()
            if time.time() - dead_since > 60: print(f"watchdog: {dead} exited, stopping server", flush=True); sh(f"docker stop {srv}")
        time.sleep(0.5)
    rc = int(st[1]); logs = sh(f"docker compose -f {yml} ps -a --format json").stdout
    for svc in ["flserver"] + [f"client_{c['k']:02d}" for c in F["clients"]]:
        lg = sh(f"docker compose -f {yml} logs --no-color {svc}"); open(os.path.join(cdir, f"{svc}.log"), "x").write(lg.stdout + lg.stderr)
    codes = {c: sh(f"docker inspect -f '{{{{.State.ExitCode}}}}' {project}-{c}-1").stdout.strip() for c in ["flserver"] + [f"client_{c['k']:02d}" for c in F["clients"]]}
    sh(f"docker compose -f {yml} down")
    json.dump(dict(server_rc=rc, exit_codes=codes, ips=ips, net_samples=net, wall_s=round(time.time() - t0, 1), killed=killed), open(os.path.join(cdir, "run.json"), "x"), indent=1)
    print(json.dumps(dict(server_rc=rc, exit_codes=codes, ips=ips, wall_s=round(time.time() - t0, 1), last_net=net[-1] if net else None)), flush=True)
    sys.exit(0 if rc == 0 else 1)


if __name__ == "__main__":
    main()
