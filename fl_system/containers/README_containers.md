# Running the federation as Docker containers (validated in report 29)

**Prerequisites**
- Docker with the Compose plugin.
- The NVIDIA Container Toolkit, for GPU access in the containers.
- Your user in the `docker` group. Until your next login, prefix commands with `sg docker -c "..."`.

```bash
# build the small base image (Ubuntu 24.04 + python3.12, same version as the host)
docker build -t fl-base:2026-10-13 fl_system/containers

# one federation: 1 server + 10 client containers on a private network (generates a new compose file per run)
.venv-fl/bin/python fl_system/containers/run_compose.py --fed data/fl_system/fold0_v2/federation.json --run <name> \
    --seed 0 --rounds 200 --examples 2560000 --batch 64 --lr 1e-3 --init server_init_s0.pt

# after a failure: restart with --resume (same --run name)
# isolation probe:     bash fl_system/containers/check_isolation.sh
```

**Outputs** go to `data/fl_system/fold0_v2/server/runs/<name>/`:
- `final.pt`, `rounds.jsonl` and the global checkpoints;
- `containers/`, which holds `compose.yml`, one log per container, and `run.json` (exit codes, IPs, network samples).

**Mounts.** The environments `.venv-fl` and `.venv` and the code are mounted read-only at their host paths, which keeps the image small and the libraries identical. In a production image you would install `research/requirements_fl.lock` into the image instead.

**Isolation.** Each client container mounts only its own folder; the server mounts only `federation.json` and `server/`.

**Real multi-machine deployment.** On each host, run the same image, with the client folder produced by that site and `--server <server-host>:8080`. You also need TLS before using any network you don't control. See `fl_system/README.md`.
