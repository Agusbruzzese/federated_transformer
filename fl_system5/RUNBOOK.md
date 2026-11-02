# Deployment runbook — federated SSL encoder (release configuration)

Configuration validated in reports 31–39. It uses:
- **Flower 1.39 runtime**: SuperLink plus one SuperNode per site;
- **TLS** with **site authentication** (each site has a registered ECDSA key);
- **SecAgg+**, so the server decodes only the sum of the site updates;
- **strict rounds**: every site in every round, nothing partial and nothing skipped;
- **exact resume** after any failure.

The app to deploy is `fl_system5/app`. It is the M3 app `fl_system3` plus the round-continuity guard (report 38).

Do **not** enable differential privacy. With 10 sites it destroys the model at every useful privacy level (report 37).

---

## 1. Roles and what each machine holds

| Role | Runs | Holds | Never holds |
|---|---|---|---|
| Server (operator) | `flower-superlink`; the ServerApp (spawned by it) | `federation.json`, `server/` (pinned server checkpoint, run outputs), server TLS certificate + key, CA certificate, the sites' **public** keys | any site data, any site private key, the CA private key |
| Site *k* (10 sites) | `flower-supernode` | its own `client_kk/` folder (events, noise pool, info), the CA certificate, **its own** key pair | other sites' data or keys |
| Certificate authority (offline) | — | CA private key | — (keep it off both server and sites) |

Environment on every machine: Python 3.12, the `.venv-fl` environment (Flower 1.39 + the pinned numeric stack of `.venv`), and the code folders `fl_system/` and `ssl_central/`. The server also needs `fl_system5/app/`.

## 2. One-time setup

1. **Certificates.** Issue the server certificate from the organisation's CA. Its SAN must contain the server's DNS name, which every site will use. `fl_system2/pki/make_pki.sh` shows the procedure, but it produces **test material only**: do not deploy those keys.
2. **Site keys.** Each site generates its own ECDSA P-384 key pair (as in `make_pki.sh`), keeps the private key, and sends only the `.pub` file to the operator.
3. **Data provisioning.** Run `fl_system/provision.py` once per fold (study environment `.venv`). It writes `client_kk/` folders, which are handed to each site, and a `server/` folder that stays with the operator. Pinned server checkpoints are added with `provision.py add-server-seed <fold_dir> <seed>`, which checks their hashes.
4. **Ports.** Sites need outbound access to the server's Fleet API (`9092`). The operator CLI needs the control API (port `8190` in this setup). Nothing on the sites needs to accept inbound connections.

## 3. Starting a federation

On the server:

```bash
flower-superlink --database <dir>/superlink_state.db --port 8190 \
  --ssl-ca-certfile ca.crt --ssl-certfile server.pem --ssl-keyfile server.key \
  --enable-supernode-auth
# register every site's public key (operator CLI; connection 'rel' in $FLWR_HOME/config.toml
# with root-certificates = ca.crt)
for k in 00 01 02 03 04 05 06 07 08 09; do flwr supernode register keys/site_$k.pub rel; done
```

Use a file-backed `--database`. The in-memory store lost replies under load (report 31).

On each site *k*:

```bash
flower-supernode --superlink <server-dns>:9092 --root-certificates ca.crt \
  --auth-supernode-private-key site_kk --auth-supernode-public-key site_kk.pub \
  --node-config 'data-path="/path/to/client_kk" code-root="/path/to/code"'
```

Start the run from the operator machine:

```bash
flwr run fl_system5/app rel --stream --run-config \
 'rounds=200 examples=2560000 batch=64 lr=0.001 seed=0 fed="/path/federation.json"
  init="/path/server/server_init_s0.pt" run-name="<new name>" code-root="/path/to/code" resume=false'
```

This is the study's best recipe, S+F: server pretraining followed by FedAvg with 200 rounds and learning rate 1e-3. `run-name` must be new. The server refuses to overwrite an existing run.

## 4. Monitoring

- **Progress:** `<fold>/server/runs3/<run>/rounds.jsonl` gets one line per committed round (time, loss, maximum update, `n_over_clip`, which must stay 0), and `last_round.json` holds the last committed round.
- **Expected values on the test bench:** about 40 s per round with SecAgg+ and about 2.2 h per run. Per site per round, about 1.0 MB down and 1.9 MB up. SecAgg+ sends 64-bit masked integers, so the upload is twice the model size. Report 39 has the measured figures.
- **Checkpoints:** `global_rNNN.pt` is written for rounds 1–3 and then every 50 rounds, `global_last.pt` is updated every round, and `final.pt` and `summary.json` are written at the end.

## 5. Failures and recovery

| Situation | What the system does | What the operator does |
|---|---|---|
| A site dies or disconnects mid-round | SecAgg+ cannot unmask, so the round is **not** committed. The guard stops the ServerApp at the next round ("would follow committed round N"). No partial or skipped round is possible. | Restore the site, then run the **same** `flwr run` with `resume=true`. Each site reloads its own state for round N from its folder, and the result is bit-identical to an uninterrupted run (reports 38–39). |
| SuperLink store error ("reply message objects are unavailable") | The round is replayed in full, at most `max-retries` = 2 times, with an identical result (report 31). | Nothing. If it keeps happening, check disk and the database. |
| Unregistered, plaintext or wrongly-certified site tries to join | Refused at the TLS handshake or at activation (reports 32, 38). | Investigate. |
| Stale processes after a crash | Old SuperNodes can re-register as dead nodes. | Stop every Flower process before restarting. `run_deploy.py` and `run_stack.py` do this automatically. |

## 6. Verifying a deployment

1. **Exactness.** Before production, run 3 rounds and compare the global weights with the exact simulator: `fl_system4/dpsim.py` with `--clipping-range 0.5` and the same schedule. The maximum |diff| must be 0.0 (`check_M4v.py` shows how).
2. **Model quality.** Score the final model with the frozen evaluator (`research/model/K_committee/protocol/evaluator.py` and `evaluator2.py`). Compare it with the study's S+F model for the same seed. SecAgg+ runs are EQUIVALENT within the study margins (report 36), but they are not bit-identical to plain runs.
3. **Local reproduction of the whole stack.** In containers, with emulated WAN links:
   `sg docker -c ".venv-fl/bin/python fl_system5/run_stack.py --run <name> --rounds 3 --examples 38400"`.

## 7. Security notes and limits

- **TLS** protects the site–server link, and **authentication** keeps unregistered sites out.
- **SecAgg+** hides each site's update from the server, as long as no more than 3 of 10 parties collude with an honest-but-curious server (reconstruction threshold 7).
- **What nothing here protects:** the final model still depends on all sites together. A formal privacy bound would need record-level DP and a different loss (report 37).
- **Deterministic seeds.** Stochastic-rounding seeds are fixed so that runs are reproducible. They do not weaken the masking. A deployment that does not need bit-reproducibility can drop the seeding.
- **Study scope.** The validated scope is the study's scope: one TMR sensor, 14 winter days, and morphology probes (no vehicle labels). Generalising to other sensors or seasons needs new evaluation.
