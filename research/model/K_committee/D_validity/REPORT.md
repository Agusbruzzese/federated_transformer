# Committee member D (validity critic): report, 2026-10-09

_Saved verbatim by the orchestrator from the agent's hand-back. Orchestrator spot-checks:_
- _B3 confirmed: `m5.py:84` sets `spe = N // bs`, so with N < 256 the inner loop never runs and `while step < STEPS` never ends._
- _B1 confirmed: the strata sizes in `data/synth/model.npz` sum to 2 299, the training-day (client-day) events._

## 1. Summary: ranked threats

Facts I verified are tagged **[V]**, with file:line or a script output. Hypotheses are tagged **[H]**.

**BLOCKERS**

- **B1. The synthetic generator is central data under another name.** It cannot serve as the "server-owned data" for arm 3. **[V]**
  - `data/synth/model.npz`: the strata sizes sum to 2 299. That is exactly `S_synth/results.json` n.train, i.e. all 13 training days, which are the clients' days.
  - Its strata are the C2 families. `families.py` runs Louvain on `aligned_corr.npy` over all 4 375 SNR≥20 events, and `families.csv` contains all 1 412 test events.
  - Its noise colouring filter is fitted on pooled client events (`generator.py:172`). Its noise rows come from 7 client days.
  - The smoothed bootstrap resamples latent codes of real client events. Synthetic events sit closer to the training events than test events do (NN ratio 0.937, `S_synth/results.json` V4).
  - `sim14d/insample` was fitted on **all days, test included** (report 07 l.67).
  - **Fix:**
    - Server data = real days assigned to no client and no test.
    - If synthetic data is wanted, refit the generator on server days only: families re-derived on server days, noise from server days, bandwidth chosen on server days.
    - Add a central arm trained on clients ∪ server data, so that arm 3 has a data-matched upper bound.

- **B2. Existing checkpoints must not be the "pretrained server".** **[V]**
  - `data/c1`: `c1_train.py:1` trained on all SNR≥20 events of all days, and l.11 swaps in the full noise pool, which includes test days.
  - `data/m5`, `data/ssl_central` and `data/c2` were trained on client days.
  - **Fix:** pretrain the server from scratch on server-only data, then hash-pin the checkpoint.

- **B3. The central training loop cannot run on most clients.** **[V]**
  - `m5.py:84-88`: `spe = N // bs`, and the step counter only advances inside `for b in range(spe)`.
  - With N < 256 the loop never terminates. `fl_recipe_checks.py` reproduces this: N=35 and N=255 → no termination.
  - 7 of 13 training days have fewer than 256 SNR≥20 events.
  - Choosing min(256, N) instead changes the number of negatives per anchor from 510 to 68–504. That confounds "federation" with "small contrastive batch".
  - **Fix:**
    - Pre-register the client batch rule.
    - Add an IID-FL control: same client sizes, events assigned at random.
    - Add a central small-batch control.

**MAJOR**

- **M1. The shared noise pool is cross-client data sharing.** **[V]** Fix in §3.
  - After M4's filter, the pool has 1 008 segments from 7 days only. Feb 14 alone is 40.5 %.
  - 6 of 13 client days have **no** segments: Feb 9, 11, 18, 20, 24, 26.
- **M2. The central recipe was tuned post-hoc on the test days; FL gets no tuning.** **[V]** This favours central.
  - lr 3e-4 was chosen in F on Feb 26 (`data/ssl_central/selection.json`), using models whose training data included test days Feb 7, 13 and 25.
  - T=0.1 came from M4's validation (Feb 10).
  - The move to 10k steps with warm-up was motivated by M4's test results (M5 PREREG C1 calls this "post-selection").
  - **Fix:**
    - Give FL an equal, pre-registered tuning budget on the validation day only (about 6 configurations, matching F's grid plus M4's).
    - Treat everything as exploratory.
- **M3. The partition is strongly non-IID in the labels' own marginals, not just in volume.** **[V]** See §4.
  - Low-volume days have median duration 6.3–6.7 s, against about 3.1 s on full weekdays. Their family-mix TVD vs the global mix is 0.29–0.41.
  - How clients are weighted therefore moves the effective training distribution away from the test distribution, which is all weekdays.
- **M4. The test has almost no room to separate central from federated.** **[V]**
  - From M5 `scores_abs.csv`, the gap between central SSL and an untrained network is:
    - lobes: +0.17 (kNN) and +0.10 (lin);
    - duration and polarity: +0.00 to +0.05;
    - SimCLR-CNN duration under kNN is **below** its own init (0.771 vs 0.820).
  - The seed SD on lobes is 0.02–0.04. The M5 review computed about 5 % power for a +0.05 effect.
  - "Federated ≈ central" will be the default outcome even if federation hurts. **[H, highly probable]**
- **M5. Run-0204 dropout artefacts mark specific clients.** **[V]**
  - On client days Feb 6, 8, 9 and 11, 61–93 % of SNR≥20 events contain dropouts. That is 768 events, 29 % of training.
  - Client-identity leakage tests are therefore confounded with the artefact, and SimCLR-CNN already carries extra run-0204 information.
- **M6. The test days have been reused at least 4 times:** M4, M5, synthetic-generator validation, and C1/C2. All days have been used for something, so no confirmatory claim is possible. **[V]**
- **M7. Batch composition differs between arms.** **[H]**
  - Central batches mix days, so negatives from other days make day and run cues usable as a shortcut.
  - Single-day client batches remove that shortcut.
  - Federated training may therefore leak *less* for structural reasons, not because of privacy.
  - The IID-FL control separates the two.

**MINOR**

- The false-negative guard is not a confound. **[V]** It masks 0.03 % of pairs centrally and 0.2–1.3 % per client.
- No BatchNorm (`models.py:1`), so FedAvg needs no BN-statistics handling. **[V]**
- The C2 families are defined using test events. **[V]** This is the same for every arm, but it is a circular target.
- Generator bandwidth was chosen with the central M5 SimCLR-TF embedding (`validate.py:35,101-104`). **[V]**

## 2. Leakage audit

| Path | Test-day info? | Evidence | Favours |
|---|---|---|---|
| (a) Noise pool, raw file | **Yes**: 922 segments from Feb 13 and Feb 25 | `bgpool_meta.parquet` (`day_confounds.json`) | — |
| (a) Noise pool, as M4/M5 use it | No: filtered (`m4.py:41-45`) | 1 008 segments remain | Neither, **if** the FL code also calls `m4.bgpool()`. Loading `bgpool.npy` directly leaks, as C1 did. |
| (b) Generator `model.npz` | Fit on training days only (n = 2 299). Strata = families built with test events | `families.py`, `families.csv` | Arm 3, if used |
| (b) `sim14d/insample` | **Yes**: fitted on all days | report 07 l.67 | Arm 3 (direct test leakage) |
| (b) Generator noise filter and pool | Training days only | `generator.py:53-58,172` | Arm 3, via client noise |
| (c) Hyperparameters | lr chosen on Feb 26, with models trained on test-day events. Step count from M4 test results | `selection.json`, M5 PREREG C1 | **Central** |
| (d) Normalisation | None during training: per-view max-norm (`augment.py`), per-event σ. Probe z-scoring and PCA fitted on training days at evaluation time | `m5.py:146-150` | Neither, if the evaluator is identical across arms |
| (e) C2 families | Louvain over all 4 375 events, including test | `families.py` | Neither, as an evaluation target. Arm 3 if used for training |
| Existing checkpoints | c1 = all days plus test-day noise | `c1_train.py:1,11` | Arm 3, if reused |

## 3. Cross-client sharing audit (federated setting)

| Shared object | Status | Fix |
|---|---|---|
| Background noise pool | Mixes 7 days' raw background. 6 clients have none of their own | Build a pool per client from its own day (rerun `bgpool.py` per day, same filters). The central arm uses the **same per-event own-day noise**, so the augmentation is identical across arms |
| False-negative guard | Local by event time | OK; nothing to change |
| sigma_ev, crop and normalisation | Per event | OK |
| Global statistics | None in training | Forbid adding any (D_features' "normalise globally" must not enter the FL training path) |
| HP selection | Central selection on pooled data | FL selection uses only the server's or validation day's data, under a declared budget |
| Generator as server data | Built from all clients' data plus test-defined families | B1 fix |
| Evaluator (probes fitted on all training-day embeddings) | Centralised by design | Declare it an **oracle evaluator** outside the FL system, identical for all arms |
| Extraction and detector constants | Global design constants, applied per block | Acceptable; declare them |

## 4. Partition confounds (SNR≥20, `day_confounds.csv`)

The 13 training days are Feb 4, 5, 6, 8, 9, 11, 12, 14, 18, 20, 24, 26, 27.

| Measure | Values |
|---|---|
| Events per day | 35 (Sun Feb 9), 46 (Sat Feb 8), 47 (Feb 4), 53 (Feb 24), 61 (Feb 12), 144 (Feb 18), 253–387 (the other 7). Max/min ratio 11.1 |
| Days with fewer than 256 events | Feb 4, 8, 9, 12, 18, 24, 27 |
| Partial days | Run starts Feb 4, 12, 18 and 24 cover only 11–23 h (6-h-bin TVD 0.59–0.65). Run ends Feb 11, 14, 20 and 27 stop at 12–16 h |
| Duration median | 6.3–6.7 s on low-volume days vs 2.8–3.4 s on full weekdays (log_amp 3.3–3.6 vs 4.1–4.3) |
| Family-mix TVD vs global | 0.18–0.41 on low-volume days; 0.02–0.12 otherwise |
| Polarity (positive share) | 0.20–0.36. CRITIQUE A6 notes that polarity tracks time of day and direction |
| Dropout share (SNR≥20) | 0.61–0.93 on Feb 6, 8, 9 and 11. 0.04–0.47 otherwise. Test day Feb 7: 0.83 |
| Noise (sigma_ev median) | 102–190 counts |
| Weight of the 6 low-volume days | 386/2 636 = 14.6 % under size-weighted FedAvg or central training; 46 % under uniform client weights |

13 days do not split into 10 clients without a merge rule. That rule must be pre-registered: for example, merge the run-start partial day with the next day of its run.

## 5. Central-model weaknesses that matter for the comparison

- **Circular targets.** Polarity, lobes and duration are morph8 descriptors, so they are not vehicle information (M5 C8; CRITIQUE A6). Any central-vs-FL gap is a gap in waveform-descriptor recoverability.
- **Saturation.** The untrained CNN already reaches polarity 0.843/0.885 and duration 0.820/0.821 (kNN/lin). Only lobes has about 0.1–0.17 of headroom.
- **No same-space baseline for the TF.** B2 is unpooled tokens, and there is no pooled TF-init baseline. FL vs central needs a shared per-seed init, with each arm scored against that init in its own space. Never pool spaces: the CRITIQUE and v3 lesson is that pooling makes the network identifiable.
- **Architecture choice.** TF and CNN tie (M5); the TF is not better than the CNN (H2). The CNN's window above its init is smaller, and it carries extra run-0204 information. Choose the architecture with the largest central-minus-init window, decided on the validation day before any FL run.
- **10k steps on 2 636 events is about 970 epochs.** Central may be over-trained or saturated, so a compute-matched FL comparison could hide or create gaps. Report curves and two budgets: matched gradient steps, and matched samples seen per client.
- **The recipe is untuned for FL** (M2). Its AdamW schedule and state handling across rounds are not defined.
- **The decision rules have about 5 % power** (M5 review). The hierarchical bootstrap with 4 test days effectively requires the same sign on all 4 days.

**Discriminating metrics:**
- lobes (kNN and lin);
- duration (lin, TF);
- gap closure G = (FL − init)/(central − init), with a CI;
- calibrated conditional leakage of client, day and is-0204 identity, conditioned on morph8, hour and has_dropout, with a day-permutation null.

Polarity and kNN duration will not discriminate.

## 6. Claims allowed under each outcome

**FL ≈ central (no significant difference):**
- **Allowed:**
  - "Within ±δ" only if TOST equivalence at a pre-registered δ passes.
  - Otherwise: "no detectable difference; the test could detect differences ≥ X".
    - Rough estimate: with 5 seeds the lobes SE of the difference is ≈ 0.025, so X ≈ 0.06, about 35–60 % of the central-vs-random window.
- **Not allowed:**
  - "Federated learning is as good as central".
  - Any statement about vehicles.
  - Generalisation to other partitions, client counts or sensors.

**FL < central:**
- **Allowed:**
  - "This FL recipe under this partition loses X on these morph-descriptor proxies".
  - Attribution to non-IID-ness only if the IID-FL control closes the gap.
  - Attribution to batch size only if the central small-batch control reproduces it.
- **Not allowed:** "Federation inherently hurts". The recipe was tuned for central training (M2).

**FL > central:**
- Treat as suspect first: leakage, a tuning asymmetry, or central over-training.
- Allowed only if it holds across seeds and both probes, and is not explained by the IID-FL control.

**Arm 3 > arm 2:**
- **Allowed:**
  - "Server pretraining on disjoint data helps".
  - This only holds if the server data is verified disjoint (B1) and the gain also beats a central model on clients ∪ server data, or the server-only model alone.
- **Not allowed:** any claim if the generator or its families were fitted on client or test days.

**Leakage differences:**
- Allowed only against a calibrated null and conditioned on the dropout and hour confounds (M5, M7).

**All outcomes:** exploratory, post-selection, SNR≥20 only.

## 7. Mandatory controls and pre-registration items

**Controls:**
1. **Negative controls:**
   - the shared per-seed random init (same-space baseline for all arms);
   - FL with 0 rounds;
   - shuffled-client null for client-leakage tests.
2. **IID-FL:** same 10 client sizes, events assigned at random. This separates federation from heterogeneity.
3. **Central small-batch control:** batch = median client batch.
4. **Central on clients ∪ server data:** the upper bound for arm 3.
5. **Server-only pretrained model:** the floor for arm 3.
6. **Planted controls, run through the full FL path and the evaluator before training:**
   - one client with a planted artefact (e.g. a synthetic offset) must become detectable by the leakage test;
   - planted noisy-lobe and duration features must pass the probes (as in the M5 dry run).
7. **Equivalence calibration:** two independent central runs with different seeds must come out "equivalent" at δ. If they don't, the test cannot certify equivalence.

**Pre-registration items:**
- Client partition and the merge rule for 13 days into 10 clients.
- Client weighting.
- Batch rule for N < 256 (and fix the infinite loop).
- Per-client own-day noise pool, used identically in all arms.
- Server data identity, hashes and disjointness checks (days, events, noise segments, families).
- Compute budgets: matched steps and matched samples.
- Local epochs, rounds, optimiser-state policy.
- Equal tuning budget, on the validation day only.
- Final-round checkpoint, with no early stopping in any arm.
- At least 5 seeds per arm, preferably 10, with inits shared across arms.
- Architecture chosen on the validation day by the size of the central-vs-init window.
- Primary metric (lobes gap closure G), δ and the TOST rule.
- Hierarchical bootstrap with the same draws across arms.
- Leakage tests conditioned on morph8, hour and has_dropout, with a day-permutation null.
- An assertion in code that `m4.bgpool()` (filtered) is used and that `bgpool.npy` is never loaded raw.
- Hashes of every input and of the code.
- The "exploratory" status and the claims list in §6, fixed in advance.

## 8. Files created (all in `research/model/K_committee/D_validity/`)

- `day_confounds.py`: per-day volume, coverage, dropout, noise, label marginals, family and hour TVD, noise-pool provenance. Outputs `day_confounds.csv` and `day_confounds.json`.
- `fl_recipe_checks.py`: m5 loop non-termination for N < 256, false-negative-guard mask fractions, negatives per client batch. Output `fl_recipe_checks.json`.

No existing file was modified. Nothing was committed.
