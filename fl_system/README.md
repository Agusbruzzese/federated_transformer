# fl_system: the federated system (runbook)

The real federated version of the study's best recipe: a server pretrained on its own data, then FedAvg over sites (clients), with each site using only its own data and noise. It is validated as **bit-identical** to the frozen study simulator (reports 25-27).

## Files

| file | role |
|---|---|
| `provision.py` | Splits a fold into client folders and a server folder (study env `.venv`); `add-server-seed` adds another pinned server checkpoint |
| `core.py` | Local training step, schedule, batching and FedAvg formula (verbatim copies of the simulator) |
| `server.py` | Flower server: strict FedAvg (all clients, no partial rounds), checkpoint every round, `--resume` |
| `client.py` | Flower client: own data, own optimiser and RNG state, per-round state for exact resume |
| `run_local.py` | Runs 1 server and K clients as separate processes on one machine, with a watchdog and fault injection |
| `tests/` | T1-T6 acceptance tests and their JSON results |

The environment is `.venv-fl`: Flower 1.39, with numpy, torch, scipy, sklearn and pandas pinned to the study environment. The lock file is `research/requirements_fl.lock`.

## On one machine (what was validated)

```
.venv/bin/python fl_system/provision.py --fold 0 --server-seed 0          # once; creates data/fl_system/fold0_v2/
.venv-fl/bin/python fl_system/run_local.py --fed data/fl_system/fold0_v2/federation.json --run <name> --seed 0 \
    --rounds 200 --examples 2560000 --batch 64 --lr 1e-3 --init data/fl_system/fold0_v2/server/server_init_s0.pt
```

A full run takes about 5 minutes on the RTX 4060. The output goes to `data/fl_system/fold0_v2/server/runs/<name>/`:
- `final.pt`
- `rounds.jsonl` (per-round losses, bytes, per-client metrics)
- `global_rNNN.pt`
- `summary.json` (provenance hashes)

## On separate machines (deployment)

1. **Data at each site.** In a real deployment, each site runs the extraction pipeline (`extraction/`) on its own sensor stream and builds its own-day noise pool. It then creates a client folder with the same layout as `provision.py` writes:
   - `events.npz`
   - `noise.npz`
   - `info.json` (`k`, `days`, `n`, file hashes)

   No raw data leaves the site.
2. **Server host.** Put the pretrained server checkpoint in a `server/` folder. Write `federation.json` with `K`, `N_total` and each client's `n`; the sites report these counts, not data. Start the server:

   ```
   .venv-fl/bin/python fl_system/server.py --fed federation.json --run <name> --seed 0 --rounds 200 --examples 2560000 \
       --batch 64 --lr 1e-3 --init server/server_init_s0.pt --address 0.0.0.0:8080
   ```
3. **Each site.** Start the client:

   ```
   .venv-fl/bin/python fl_system/client.py --folder <client folder> --run <name> --seed 0 --server <server-host>:8080
   ```
   The server waits until all K clients are connected, then starts round 1.
4. **If a site fails.** The server stops the run without aggregating the partial round. Restart the failed client, then restart the server with `--resume`. Clients restore their state from the last committed round, and the result is bit-identical to an uninterrupted run (test T4).
5. **Traffic.** About 17.3 MB per round in total (0.86 MB per model, each direction, per client), so 3.45 GB per 200-round run. Measured gRPC overhead is about 1 %.

## Before a real deployment (not done here)

- **TLS.** Flower supports certificates (`certificates=` in `start_server` / `start_client`); they are not configured here.
- **Privacy.** There is no secure aggregation or differential privacy. The study's nuisance gate measures day information; it is not a privacy guarantee.
- **API deprecation.** `start_server` and `flwr.compat.client.app.start_client` are deprecated in Flower 1.39. The migration path is the SuperLink/SuperNode runtime, with the client state (optimiser, RNG, stream) kept in `Context.state`. T1-T6 must then be re-run.
- **Hardware.** The validated deployment used one GPU on one machine. On separate machines with different GPUs or library builds, bit-identity is not expected. Equivalence within the study's margins should be re-checked, as in T2.
