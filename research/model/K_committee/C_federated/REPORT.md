# Committee member C (federated-learning engineer): report, 2026-10-09

_Saved verbatim by the orchestrator from the agent's hand-back. Orchestrator check: `selftest.json` shows identical loss curves for m5.train and fedsim central, max_abs_weight_diff 0.0; `determinism.txt` max abs diff 0.0. No committee processes were left running._

## 1 Summary
- **Simulator built and proven against M5.** `fedsim.py` is an in-process FedAvg SimCLR simulator. It reuses M5's components unchanged: PatchTransformer and Head, `augment.train_view` operations, `nt_xent` with the 30 s guard, AdamW at 3e-4 / 1e-4, and the warm-up plus cosine schedule.
  - **Proof that only the topology differs:** `fedsim.py selftest` runs `m5.train` for 40 steps into a scratch directory, then runs fedsim's `central` arm (a single client, so FedAvg is the identity). The two are **bit-identical**: same loss curve, max absolute weight difference 0.0 (`selftest.json`).
  - Two federated runs with the same seed are also bit-identical (`determinism.txt`).
- **Recommended data design (P11).**
  - The server owns run 0218: Feb 18 and Feb 20, 434 events.
  - 10 clients come from the other 11 training days (2 202 events).
  - The same 2 202 events are used by every arm, central included.
- **Noise pools rebuilt per day.** Every event draws background noise only from stretches recorded on its own day, in every arm. Each event therefore has exactly the same augmentation distribution in central, federated-natural, federated-IID and server training, and no client touches another client's noise.
  - Under the strict I5 rule, Feb 9 and 11 give 0 segments and Feb 8 gives 1, because run 0204 has dense dropouts.
  - A documented relaxed fallback for Feb 6, 8, 9 and 11 fixes this. Every day now has at least 35 segments (5 014 in total).
- **Cost of the full design.** Measured cost is about 7–8 GPU-hours sequentially, including tuning and ablations. A federated round of the primary design takes 1.95 s; peak GPU memory is ≤ 272 MB per process.
- **What I changed after D's input:**
  - The existing synthetic generator and all existing checkpoints are rejected as server data.
  - The batch rule is explicit: `b_k = min(64, n_k)`, so every client has at least one step per epoch. This avoids M5's non-terminating loop when N < 256.
  - The merge rule is fixed in advance.
  - A central small-batch control and a federated-IID control are included.
  - Every arm gets an equal tuning budget on Feb 10 only.

## 2 Client partitions
Split is M4/M5: test days Feb 7, 13, 19, 25; validation Feb 10; SNR ≥ 20; 2 636 training events.

**Training days (`per_day_counts.csv`):**

| Day | Events | Run |
|---|---|---|
| 02-04 Tue | 47 | 0 |
| 02-05 | 357 | 0 |
| 02-06 | 387 | 0 |
| 02-08 Sat | 46 | 0 |
| 02-09 Sun | 35 | 0 |
| 02-11 | 300 | 0 |
| 02-12 | 61 | 1 |
| 02-14 | 297 | 1 |
| 02-18 | 144 | 2 |
| 02-20 | 290 | 2 |
| 02-24 | 53 | 3 |
| 02-26 | 366 | 3 |
| 02-27 | 253 | 3 |

Per run: run 0 = 1 172, run 1 = 358, run 2 = 434, run 3 = 672. Seven days are below 256 events.

**Partition rules (`partition.py`, written before any training):**
- **P11 (recommended).** Run 0218 belongs to the server. Each of the other days is one client, except the weekend (Feb 8 + 9), which is one client.
- **P13** (only if there is no real server data). One client per day. A day with fewer than 60 events merges into its adjacent same-run training day: Feb 4 into Feb 5, Feb 8 with Feb 9, Feb 24 into Feb 26.
- **IID control** for each: identical client sizes, events assigned by a seeded permutation (seed 0).

**P11 natural clients:**

| Client | Days | Run | n | Family TVD vs pool | Positive polarity | Median σ_ev | Night fraction |
|---|---|---|---|---|---|---|---|
| 0 | 02-04 | 0 | 47 | 0.409 | 0.23 | 147 | 0.06 |
| 1 | 02-05 | 0 | 357 | 0.028 | 0.29 | 190 | 0.06 |
| 2 | 02-06 | 0 | 387 | 0.025 | 0.33 | 187 | 0.05 |
| 3 | 02-08 + 02-09 | 0 | 81 | 0.327 | 0.26 | 113 | 0.17 |
| 4 | 02-11 | 0 | 300 | 0.105 | 0.31 | 178 | 0.01 |
| 5 | 02-12 | 1 | 61 | 0.283 | 0.33 | 154 | 0.10 |
| 6 | 02-14 | 1 | 297 | 0.047 | 0.31 | 159 | 0.10 |
| 7 | 02-24 | 3 | 53 | 0.276 | 0.25 | 127 | 0.04 |
| 8 | 02-26 | 3 | 366 | 0.033 | 0.33 | 189 | 0.07 |
| 9 | 02-27 | 3 | 253 | 0.127 | 0.36 | 182 | 0.05 |

- The small clients are the partial run-start days and the weekend. They are dominated by family 3 (40–45 %, against about 15 % overall). They also have lower σ and lower amplitude.
- Per-family fractions are in `hetero_*.csv`.

**P13 natural** client sizes: 404, 387, 81, 61, 300, 297, 144, 290, 419, 253 (range 61–419).

**Heterogeneity, natural vs IID (`hetero_summary.json`):**

| Measure | P11 natural | P11 IID | P13 natural | P13 IID |
|---|---|---|---|---|
| Mean / max family TVD | 0.166 / 0.409 | 0.077 / 0.140 | 0.126 / 0.328 | 0.068 / 0.131 |
| Family χ² p-value | 3e-17 | 0.82 | 3e-13 | 0.52 |
| Family Cramér's V | 0.116 | 0.059 | 0.099 | 0.058 |
| σ_ev Kruskal–Wallis H (p) | 103 (4e-18) | 7.8 (0.56) | 91 (1e-15) | 19 (0.02) |
| CV of client sizes | 0.62 | 0.62 | 0.47 | 0.47 |
| Clients from a single run | 10/10 | 0/10 | 10/10 | 0/10 |

**False-negative guard, measured.** On natural clients the guard masks 0.39 % of negative pairs, against 0.033 % for IID and 0.034 % for central. Same-day events fall within 30 s of each other more often, so natural clients lose 10× more negatives; the loss is still small.

## 3 Equal-conditions specification (primary design, P11)
**Held equal across arms**
- Data: the same 2 202 client events.
- Model: architecture, initial weights (built by `torch.manual_seed(seed)`, the same for every arm, seeds 0–4), and the full M5 augmentation chain with own-day noise per event.
- Loss: NT-Xent, T = 0.1, 30 s guard.
- Optimiser: AdamW with peak lr 3e-4, wd 1e-4.
- LR schedule shape: 5 % warm-up, then cosine to 0, over the budget.
- Evaluation: a single shared protocol, set by committee B.

**Compute budget:** X = 10 000 × 256 = 2 560 000 single-event examples, each giving 2 views. Equal examples means equal FLOPs. Gradient steps cannot all be equal; the table reports them.

| Arm | Batch | Rounds R | Steps per round | Total examples | Gradient steps | Sequential depth |
|---|---|---|---|---|---|---|
| C256: central (M5) | 256 | – | – | 2 560 000 | 10 000 | 10 000 |
| C64: central small-batch control | 64 | – | – | 2 560 000 | 40 000 | 40 000 |
| F-nat: FedAvg, natural | min(64, n_k): 47/64/64/64/64/61/64/53/64/64 | 200 | s_k = 6/32/35/7/27/6/27/6/33/23 | 2 548 400 (−0.45 %) | 40 400 | 7 000 |
| F-IID: FedAvg, IID control | same sizes | 200 | same | same | same | same |

**How local work is set.** Client k runs s_k = round(X/R · n_k/N / b_k) local steps per round. This is equal local epochs (about 5.8 per round), so every event is seen about 1 160 times in every arm (`accounting.csv`): 1 106–1 200 epochs across clients, against 1 163 central.

**Learning rate.** M5's lambda is evaluated on a central-equivalent step axis: i = (r·s_k + j)·10 000 / (R·s_k). Warm-up covers the first 10 rounds; cosine reaches 0 at round 200.

**Participation.** Cross-silo: all 10 clients take part in every round. There is no client sampling.

**Aggregation.** FedAvg weighted by examples processed (proportional to n_k), server lr 1.

**Optimiser state.** Each client keeps its own persistent AdamW moments (realistic cross-silo). Ablations:
- **Moments reset each round.**
- **FedAdam** (server Adam on the pseudo-gradient). In the smoke run with η_s = 1e-3 the client drift was 10–12× the server step, so FedAdam needs its own lr tuning or must be excluded.

**SimCLR-specific costs, measured rather than hidden.** Negatives are local only. Local batch 64 gives 126 negatives per anchor (92 for the 47-event client), against 510 in M5. The comparison is decomposed into three contrasts:
- **C256 → C64:** the effect of batch size / number of negatives.
- **C64 → F-IID:** the effect of local steps plus averaging.
- **F-IID → F-nat:** the effect of non-IID data.

**Variants and tuning.**
- A "practical" variant, F-b256cap (b_k = min(256, n_k)), is an ablation (`accounting.csv`).
- R = 50 is a communication ablation.
- Tuning (D's point 5): every arm gets the same 4-configuration grid on Feb 10, seed 0 only (peak lr {3e-4, 1e-3} × {R or steps as given, R = 50 for FL / batch fixed for central}). The selection metric is fixed before any run.
- There is no early stopping; the final weights are used, as in M5.

**Noise for central.** Central uses the same per-event own-day pools, so its noise is not a pooled advantage.

## 4 Pretrained-server arm
**Options considered:**
- **S0: existing `data/synth/model.npz` or existing checkpoints.** REJECTED. They were fitted or trained on the clients' days, and the C2 families come from all days including test. This is leakage (D's blocker).
- **S1: real server days (RECOMMENDED).** Run 0218 (Feb 18 + 20, 434 events) is excluded from every client and from the central data. Its own noise pool has 1 082 segments (603 + 479).
  - Caveat: test day Feb 19 lies between the two server days in the same run. Every run contains a test day, so this cannot be avoided. Report test results per day, with Feb 19 separately.
- **S2: generator refitted on server days only.** Feasible, but weak. Only 6 family×polarity strata reach the 30-event minimum (346 of 434 events), against 12 for all training days. The families would also have to be re-derived on server data, and the generator's noise switched to the server pool. It is worth adding only as a secondary arm testing "volume from a generator", and only if S1 shows a pretraining effect.

**Budget.** The server is pretrained by fedsim's `server` arm (central topology on the server events). P = 2 000 steps × 256 = 512 000 examples, about 1 180 epochs per event, matching the clients' per-event exposure. The client phase then runs the identical F-nat specification, with a fresh warm-up and cosine. The total is P + X = 3.07M examples.

**Controls:**
- **C+P:** central training (C256) initialised from the same server checkpoint, one checkpoint per seed, shared. This isolates the effect of federation given pretraining.
- **F-nat at budget X + P from random init (R = 240):** an equal-total-compute control.

Results are reported under both equal client-phase compute and equal total compute.

## 5 Prototype: what was run and measured
Smoke tests used P13/P11 with the R = 200, b = 64 schedule, stopped after 1–4 rounds (`smoke_stdout.txt`, `runs/smoke_*/rounds.jsonl`). Total GPU use was under 5 minutes.

| Run | Seconds per round | Peak memory |
|---|---|---|
| F-nat P13 (201 steps of 64 per round) | 1.93–2.00 | 134 MB |
| F-IID P13 | 2.00–2.07 | 134 MB |
| Server pretrain P11 (100 steps of 256) | 1.85–1.93 | 253 MB |
| Fed-pretrained, reset moments + FedAdam (P11) | 1.97–2.05 | 131 MB |
| Central b256 | 18.6 ms per step | 272 MB |
| Central b64 | 9.6 ms per step | 104 MB |
| 4 concurrent F-nat processes | 5.15 each | – |

- Four concurrent processes give 1.5× the throughput of running them one after another.
- An earlier run measured 29 s per round. That was CPU contention from the 14-process noise-pool build running at the same time, not the simulator; I discarded it.

**Full-run estimates:**

| Run | Time |
|---|---|
| C256 | 3.1 min |
| C64 | 6.4 min |
| F (R = 200) | 6.5 min |
| Server pretrain | 0.6 min |
| S + F | 7.1 min |
| S + C256 | 3.7 min |
| F at budget X + P | 7.8 min |

- About 41 minutes per seed for the 7 core arms. Five seeds take about 3.4 h sequentially, or about 2.3 h with 4 concurrent processes.
- Ablations (reset moments, FedAdam, R = 50, b256cap; F-nat × 5 seeds): about 2.2 h.
- Equal tuning grid: about 2 h.
- **Total: about 7–8 GPU-hours** sequentially (about 5 h wall-clock). Probe evaluation is not included.
- Noise pools: a one-off 19 min plus 3 min CPU on 14 processes.

**Legacy code (`client/`, `server/`).** Not reusable. It is a Flower gRPC/docker federation of an MLP on 70 EMD features with KMeans pseudo-labels, and it imports the missing `pipeline` / `clustering` modules. Only its FedProx idea carries over, and that is a one-line ablation if wanted. The in-process simulator gives bitwise determinism that the Flower setup cannot.

## 6 Risks and monitoring
`rounds.jsonl` already logs, every round:
- per-client loss (first, last and mean local step), guard fraction, and update norm;
- global step norm and client drift (mean ‖θ_k − θ̄‖ divided by the global step norm);
- seconds per round and peak memory;
- on the global model, at `--eval-every`: NT-Xent on Feb 10 with fixed augmented views (a held-out yardstick that is comparable across arms), effective rank of the 64-d embedding, and linear CKA to the M5 central checkpoint.
- The CKA reference saw all 13 days, so it is for monitoring only, not evaluation.

**Risks:**
1. **Client drift.** Small clients (47–81 events) run about 6 epochs per round, about 1 160 epochs in total. Expect overfitting and dominance by family 3. The drift ratio was 0.57–0.64 in the first rounds. Watch for per-client loss collapse on the small clients while the validation loss rises.
2. **Collapse.** Two of five SimCLR seeds collapsed in C1. Flag a run if effective rank falls below half of the central value or if validation loss rises for 10 consecutive evaluations. These are reported and never acted on: no restarts or early stopping, for fairness.
3. **Divergence.** Watch for NaN, or a loss above ln(2b − 1); the FedAdam variant is the most exposed.
4. **Persistent Adam moments vs averaging mismatch.** The reset ablation covers this.
5. **Feb 19 adjacency** for the server arm (see §4).
6. **Relaxed-noise days.** Feb 6, 8, 9 and 11 add bridged dropout segments (mean mask fraction 4.5–8.8 %). This could amplify run-0204 identifiability, which is already a known leak. Run the leakage probe per arm.
7. **Equal-examples rounding.** F totals 2 548 400 against 2 560 000; report it as is, or set R so the totals match exactly.

## 7 Files created
In this folder:
- `partition.py`, `partitions.json`, `per_day_counts.csv`, `hetero_P13_natural.csv`, `hetero_P13_iid.csv`, `hetero_P11_natural.csv`, `hetero_P11_iid.csv`, `hetero_summary.json`
- `noisepool.py`: per-day pools; reuses `ssl_central.bgpool._job`, plus the `relaxed` fallback
- `accounting.py`, `accounting.csv`
- `fedsim.py`: arms `central`, `fed`, `server`, plus `--init`, `--opt persist|reset`, `--server fedavg|fedadam`, `selftest`
- `selftest.json` (bit-identical to M5), `determinism.txt`, `smoke_stdout.txt`

In `data/k_committee/C_federated/`:
- `bgpool_day.npy` (5 014 × 640) and `bgpool_day_meta.parquet` (day, rule strict/relaxed, σ, mask fraction)
- `runs/smoke_*/`: `rounds.jsonl`, `summary.json`, `final.pt`
- `selftest_m5/`: scratch output of the M5 equivalence run

No existing files were modified, and nothing was committed.

---
## Post-review fix (orchestrator, 2026-10-10)
- **Bug.** Reviewer E's B1 is confirmed: `load_vec` used `vector_to_parameters`, which aliases the global vector, so the federated arms trained the clients sequentially. The buggy file is kept as `fedsim_v0_aliasing_bug.py`.
- **Fix.** `load_vec` now copies the weights in under `no_grad`. Clients are aggregated in a fixed order. A test-only option, `--reverse-clients`, was added.
- **Proof of the fix:** `test_fedavg.py` → `test_fedavg.json` (control C0, PASS).
  - C0a: identical start weights for every client in every round.
  - C0b: bit-identical client results and global weights when clients are visited in reverse order.
  - C0c: global = example-weighted mean of the clients (max |diff| 1.6e-7).
  - E's `test_fedavg_alias.json` shows that v0 fails C0a, with 10 distinct start points.
- **K = 1 still bit-identical to M5.** `fedsim.py selftest` was re-run → `selftest.json` (identical: true, weight difference 0.0); the previous result is kept in `selftest_v0.json`.
- **Invalid smoke-run numbers.** The drift ratios and smoke-run observations in §1–§6 above were produced by v0 and are invalid; the timings are unaffected.
