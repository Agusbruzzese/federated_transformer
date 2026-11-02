"""Full-stack release test (report 38): Flower's current runtime as real containers with EVERYTHING on at once:
TLS + SuperNode authentication (M2) + SecAgg+ (M3: app fl_system5/app = fl_system3 + round-continuity guard, report 38) + an emulated WAN per site (tc netem).
  - flserver: SuperLink (TLS cert, --enable-supernode-auth) and the ServerApp it spawns; sees only federation.json + server folder.
  - site_kk:     one SuperNode per site; sees only its own data folder, the CA certificate and ITS OWN key pair.
  - private internal Docker network; site uplink netem delay/rate; SuperLink egress netem delay/rate.
  - traffic: eth0 rx/tx bytes of every site container sampled at every committed round (= the site <-> server network hop).
Modes: run (default) | intruders (TLS+auth SuperLink with site_00 registered; tries plaintext / foreign CA / unregistered key).
    sg docker -c ".venv-fl/bin/python fl_system5/run_stack.py --run <name> --seed 0 --rounds 200 --examples 2560000 [--resume]
                  [--kill-site k --kill-at-round r]"
Exit 0 only if all rounds finished. Everything goes to <fold>/server/runs3/<run>/stack*/ (new folder per attempt; never overwrites)."""
import argparse, json, os, re, shlex, subprocess, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); IMAGE = "fl-base:2026-10-13"; BIN = f"{ROOT}/.venv-fl/bin"
PKI = f"{ROOT}/fl_system2/pki"; FED = f"{ROOT}/data/fl_system/fold0_v2/federation.json"; BASE = os.path.dirname(FED)
CODE = [f"{ROOT}/.venv-fl", f"{ROOT}/.venv", f"{ROOT}/fl_system", f"{ROOT}/ssl_central"]
ENV = ["      PATH: " + BIN + ":/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin", "      HOME: /tmp", "      PYTHONUNBUFFERED: '1'"]


def sh(cmd, **kw): return subprocess.run(cmd, shell=True, capture_output=True, text=True, **kw)


def vol(p, mode="ro"): return f"      - {p}:{p}:{mode}"


def svc(name, command, volumes, gpu, extra=()):
    y = [f"  {name}:", "    image: " + IMAGE, '    user: "1000:1000"', f"    working_dir: {ROOT}", "    networks: [flnet]", "    cap_add: [NET_ADMIN]",
         "    environment:"] + ENV + list(extra) + ["    command: " + json.dumps(command), "    volumes:"] + [vol(*v) if isinstance(v, tuple) else vol(v) for v in volumes]
    return y + (["    gpus: all"] if gpu else [])


def superlink_cmd(cdir, tls=True):
    c = [f"{BIN}/flower-superlink", "--database", f"{cdir}/superlink_state.db", "--port", "8190"]
    return c + (["--ssl-ca-certfile", f"{PKI}/ca.crt", "--ssl-certfile", f"{PKI}/server.pem", "--ssl-keyfile", f"{PKI}/server.key", "--enable-supernode-auth"] if tls else ["--insecure"])


def supernode_cmd(k, key=None, ca=f"{PKI}/ca.crt", insecure=False):
    key = key or f"{PKI}/keys/site_{k:02d}"
    nc = f'data-path="{BASE}/client_{k:02d}" code-root="{ROOT}"'
    c = [f"{BIN}/flower-supernode", "--superlink", "flserver:9092", "--node-config", nc]
    return c + (["--insecure", "--auth-supernode-private-key", key, "--auth-supernode-public-key", key + ".pub"] if insecure
                else ["--root-certificates", ca, "--auth-supernode-private-key", key, "--auth-supernode-public-key", key + ".pub"])


def site_volumes(k, key=None, ca=f"{PKI}/ca.crt"):
    key = key or f"{PKI}/keys/site_{k:02d}"
    return CODE + [ca, key, key + ".pub", (f"{BASE}/client_{k:02d}", "rw")]


def compose(project, cdir, sites, tls=True, intruders=()):
    y = [f"name: {project}", "networks:", "  flnet:", "    driver: bridge", "    internal: true", "services:"]
    y += svc("flserver", superlink_cmd(cdir, tls), CODE + [f"{ROOT}/fl_system5/app", f"{PKI}/ca.crt", f"{PKI}/server.pem", f"{PKI}/server.key", FED, (f"{BASE}/server", "rw")]
             + [f"{PKI}/keys/site_{k:02d}.pub" for k in range(10)], False,
             ["    healthcheck:", '      test: ["CMD", "bash", "-c", "echo > /dev/tcp/127.0.0.1/9092"]', "      interval: 2s", "      retries: 60"])
    for k in sites:
        y += svc(f"site_{k:02d}", supernode_cmd(k), site_volumes(k), True, ["    profiles: [sites]"])
    for name, cmd, vols in intruders:
        y += svc(name, cmd, vols, False, ["    profiles: [intruders]"])
    return "\n".join(y) + "\n"


def counters(ctr):
    s = sh(f"docker exec {ctr} cat /sys/class/net/eth0/statistics/rx_bytes /sys/class/net/eth0/statistics/tx_bytes").stdout.split()
    return (int(s[0]), int(s[1])) if len(s) == 2 else None


def register(project, cdir, keys):
    out = []
    for kf in keys:
        r = sh(f"docker exec -e FLWR_HOME={cdir}/flwr_home {project}-flserver-1 {BIN}/flwr supernode register {kf} rel")
        out.append(dict(key=os.path.basename(kf), rc=r.returncode, tail=(r.stdout + r.stderr)[-300:]))
    return out


def write_flwr_home(cdir):
    os.makedirs(f"{cdir}/flwr_home")
    open(f"{cdir}/flwr_home/config.toml", "x").write(f'[superlink]\ndefault = "rel"\n\n[superlink.rel]\naddress = "127.0.0.1:8190"\nroot-certificates = "{PKI}/ca.crt"\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="run", choices=["run", "intruders"]); ap.add_argument("--run", required=True)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--rounds", type=int, default=200); ap.add_argument("--examples", type=int, default=2560000)
    ap.add_argument("--batch", type=int, default=64); ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--resume", action="store_true")
    ap.add_argument("--kill-site", type=int); ap.add_argument("--kill-at-round", type=int)
    ap.add_argument("--site-netem", default="delay 20ms rate 50mbit"); ap.add_argument("--link-netem", default="delay 20ms rate 1gbit")
    ap.add_argument("--stall-min", type=float, default=20.0, help="stop if no round is committed for this many minutes")
    a = ap.parse_args(); F = json.load(open(FED)); sites = [c["k"] for c in F["clients"]]
    out = f"{BASE}/server/runs3/{a.run}"; os.makedirs(out, exist_ok=True)
    cdir = os.path.join(out, "stack_intruders" if a.mode == "intruders" else ("stack_resume" if a.resume else "stack")); os.makedirs(cdir, exist_ok=False)
    project = re.sub(r"[^a-z0-9_-]", "", f"rel-{a.run}-{os.path.basename(cdir)}".lower()); write_flwr_home(cdir)
    yml = f"{cdir}/compose.yml"; t0 = time.time(); res = dict(mode=a.mode, args=vars(a))
    if a.mode == "intruders":
        U = f"{PKI}/keys/site_unlisted"
        intr = [("intr_plaintext", supernode_cmd(1, insecure=True), site_volumes(1)),
                ("intr_foreign_ca", supernode_cmd(1, ca=f"{PKI}/other_ca.crt"), site_volumes(1, ca=f"{PKI}/other_ca.crt")),
                ("intr_unregistered", supernode_cmd(0, key=U), CODE + [f"{PKI}/ca.crt", U, U + ".pub"]),
                ("intr_plaintext_nokeys", [f"{BIN}/flower-supernode", "--superlink", "flserver:9092", "--insecure"], CODE)]
        open(yml, "x").write(compose(project, cdir, [0], intruders=intr))
    else:
        open(yml, "x").write(compose(project, cdir, sites))
    try:
        r = sh(f"docker compose -f {yml} up -d --wait flserver"); assert r.returncode == 0, r.stderr
        SL = f"{project}-flserver-1"
        res["register"] = register(project, cdir, [f"{PKI}/keys/site_{k:02d}.pub" for k in ([0] if a.mode == "intruders" else sites)])
        assert all(x["rc"] == 0 for x in res["register"]), res["register"]
        r = sh(f"docker compose -f {yml} --profile sites {'--profile intruders' if a.mode == 'intruders' else ''} up -d"); assert r.returncode == 0, r.stderr
        if a.mode == "intruders":
            time.sleep(60); res["containers"] = {}
            for name in ("site_00", "intr_plaintext", "intr_foreign_ca", "intr_unregistered", "intr_plaintext_nokeys"):
                st = sh(f"docker inspect -f '{{{{.State.Status}}}} {{{{.State.ExitCode}}}}' {project}-{name}-1").stdout.split()
                lg = sh(f"docker logs {project}-{name}-1"); txt = lg.stdout + lg.stderr; open(f"{cdir}/{name}.log", "x").write(txt)
                res["containers"][name] = dict(status=st[0], exit_code=int(st[1]), log_tail=txt[-400:])
            nl = sh(f"docker exec -e FLWR_HOME={cdir}/flwr_home {SL} {BIN}/flwr supernode list rel"); res["supernode_list"] = (nl.stdout + nl.stderr)[-2500:]
            c = res["containers"]
            res["PASS"] = (c["site_00"]["status"] == "running" and all(c[n]["status"] == "exited" and c[n]["exit_code"] != 0
                                                                         for n in ("intr_plaintext", "intr_foreign_ca", "intr_unregistered"))
                         and "online" in res["supernode_list"] and res["supernode_list"].count("online") == 1)
            fl = open(f"{cdir}/flserver.log").read() if os.path.exists(f"{cdir}/flserver.log") else ""
            res["plaintext_nokeys_note"] = "status of a keyless plaintext node is judged by the SuperLink: it must never appear online (count above)"
        else:
            ctrs = {k: f"{project}-site_{k:02d}-1" for k in sites}
            res["netem"] = {k: sh(f"docker exec -u 0 {c} tc qdisc add dev eth0 root netem {a.site_netem}").returncode for k, c in ctrs.items()}
            res["netem"]["superlink"] = sh(f"docker exec -u 0 {SL} tc qdisc add dev eth0 root netem {a.link_netem}").returncode
            assert all(v == 0 for v in res["netem"].values()), res["netem"]
            res["netem_check"] = sh(f"docker exec {ctrs[sites[0]]} tc qdisc show dev eth0").stdout.strip()
            # isolation: what each container can see of the fold and of the key folder
            res["isolation"] = {k: dict(fold=sh(f"docker exec {c} ls {BASE}").stdout.split(),
                                        keys=sh(f"docker exec {c} ls {PKI}/keys").stdout.split(), pki=sh(f"docker exec {c} ls {PKI}").stdout.split())
                                for k, c in ctrs.items()}
            res["isolation"]["superlink"] = dict(fold=sh(f"docker exec {SL} ls {BASE}").stdout.split(), keys=sh(f"docker exec {SL} ls {PKI}/keys").stdout.split(),
                                                 pki=sh(f"docker exec {SL} ls {PKI}").stdout.split())
            time.sleep(8)
            rc = (f'rounds={a.rounds} examples={a.examples} batch={a.batch} lr={a.lr} seed={a.seed} fed="{FED}" '
                  f'init="{BASE}/server/server_init_s{a.seed}.pt" run-name="{a.run}" code-root="{ROOT}" resume={"true" if a.resume else "false"}')
            run = subprocess.Popen(["docker", "exec", "-e", f"FLWR_HOME={cdir}/flwr_home", SL, f"{BIN}/flwr", "run", f"{ROOT}/fl_system5/app", "rel", "--stream",
                                    "--run-config", rc], stdin=subprocess.DEVNULL, stdout=open(f"{cdir}/flwr_run.log", "w"), stderr=subprocess.STDOUT)
            lrp = f"{out}/last_round.json"; last = -1; t_last = time.time(); samples = []; killed = None; c0 = {k: counters(c) for k, c in ctrs.items()}
            samples.append(dict(t=0.0, round=json.load(open(lrp))["round"] if os.path.exists(lrp) else 0, c=c0))
            while True:
                lr = json.load(open(lrp))["round"] if os.path.exists(lrp) else 0
                if lr != last:
                    last, t_last = lr, time.time(); samples.append(dict(t=round(time.time() - t0, 1), round=lr, c={k: counters(c) for k, c in ctrs.items()}))
                if a.kill_site is not None and killed is None and lr >= a.kill_at_round - 1:
                    time.sleep(5); killed = dict(rc=sh(f"docker kill {ctrs[a.kill_site]}").returncode, after_round=lr, t=round(time.time() - t0, 1))
                    print("killed", killed, flush=True)
                txt = open(f"{cdir}/flwr_run.log").read()
                if "] done" in txt or "Traceback" in txt or "RuntimeError" in txt or run.poll() is not None: time.sleep(3); break
                if time.time() - t_last > a.stall_min * 60: res["stalled"] = True; break
                time.sleep(1)
            res["killed"] = killed; res["traffic_samples"] = samples
    finally:
        for s in sh(f"docker compose -f {yml} --profile sites --profile intruders ps -a --format '{{{{.Service}}}}'").stdout.split():
            lg = sh(f"docker compose -f {yml} logs --no-color {s}"); open(f"{cdir}/{s}.log", "w").write(lg.stdout + lg.stderr)
        sh(f"docker compose -f {yml} --profile sites --profile intruders down -t 10")
        res["finished"] = os.path.exists(f"{out}/final.pt") and json.load(open(f"{out}/last_round.json"))["round"] == a.rounds if os.path.exists(f"{out}/last_round.json") else False
        res["wall_s"] = round(time.time() - t0, 1); json.dump(res, open(f"{cdir}/stack.json", "x"), indent=1, default=str)
        print(json.dumps({k: res.get(k) for k in ("mode", "finished", "wall_s", "killed", "PASS", "stalled")}), flush=True)
    sys.exit(0 if (res.get("PASS") if a.mode == "intruders" else res["finished"]) else 1)


if __name__ == "__main__":
    main()
