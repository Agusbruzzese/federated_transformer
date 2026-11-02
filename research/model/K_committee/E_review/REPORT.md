# Committee member E (independent reviewer): review of protocol v1, 2026-10-10

_Saved verbatim by the orchestrator from the agent's hand-back. Orchestrator check of B1: an independent minimal test (`torch.nn.utils.vector_to_parameters` on a Linear layer, then an in-place step) shows that the parameter shares memory with the global vector and that the global vector changes. Confirmed. The fedsim self-test (K = 1) could not detect it._

## 1. Verdict: accept, but only after the required changes below. The protocol must not be frozen as it stands.

The most serious finding is a bug in the simulator. As shipped, `fedsim.py` does not run FedAvg for K > 1: each client starts from the previous client's end point, so the "federated" arms are really sequential training. Every federated arm and every smoke-run number from C depends on this code.

**Required changes, most important first:**
1. Fix the parameter aliasing in `fedsim.load_vec` (B1). Add tests that exercise the federated code path (B1 fix), re-run the smoke runs, and recompute the drift figures.
2. Build per-day noise pools for Feb 7, 13, 19 and 25 under a relaxed-rule threshold that is written down before the pools are built. Then correct §4.2 (M1).
3. Define the seed stratum and the estimand for the cross-fitted design: n = 5 init-seed means over all 9 days, an estimand conditional on the 3 fixed partitions, and per-fold Δ with a heterogeneity flag (M2).
4. Redefine the leakage gate: the unit of analysis, how the permutation null enters, which version is primary, a recalibrated margin, and no privacy wording (M3).
5. Fix the tuning asymmetry for S+C and F-X+P (M4).
6. Write a complete decision table, with explicit tests for the attribution and the S+F vs F-X+P claims (M5).
7. Do the minor fixes (m1–m9).

## 2. Findings

### BLOCKER

**B1. The simulator does not implement FedAvg for K > 1. Verified.**
- **Cause.** `fedsim.py` (`load_vec`, which calls `torch.nn.utils.vector_to_parameters`) sets `param.data = vec[...].view_as(param)`. Each client's parameters therefore become views of `gvec`, and `opt.step()` modifies `gvec` in place. Client k then loads the weights client k−1 left behind.
- **Proof** (`test_fedavg_alias.py`, which runs `fedsim.run` unchanged: P11, 10 clients, 1 round, 1 local step each; output `test_fedavg_alias.json`):
  - The 10 clients started from **10 different** global vectors, with norms rising from 39.5262 to 39.5410.
  - With a copying `load_vec`, all 10 start from one vector.
  - The final global weights differ by up to 1.9e-3 after a single round.
  - Mean loss 4.078 with aliasing vs 4.500 without. The aliased run is lower because each client inherits the previous client's progress.
- **Why the existing proofs missed it.** The self-test has a single client (K = 1), where aliasing has no effect, and the determinism test cannot detect it either.
- **Consequences.**
  - Report 08 claim 3, "the arms differ ONLY in training topology", is unsupported for every federated arm.
  - C's drift ratios (0.57–0.64) and smoke observations come from sequential training.
- **Fix.**
  - In `load_vec`, write into the parameters under `torch.no_grad()` with `p.copy_(slice)`, or pass `v.clone()`.
  - Add unit tests:
    - (a) assert that all K clients start each round from bit-identical vectors (hash check);
    - (b) K identical clients with the same data and RNG must reproduce the single-client run to float tolerance;
    - (c) with 1 local SGD step, the result must equal the example-weighted mean of K independently computed one-step updates;
    - (d) log a trace of the learning rate per step.
  - Make these a dry-run gate (new control "C0").

### MAJOR

**M1. Own-day noise pools are missing for 4 client days. Verified.**
- `bgpool_day_meta.parquet` holds only the 13 old training days. Feb 7, 13, 19 and 25 are clients in the fold design (Feb 19 in every fold), and `noisepool.py:7` says "Test/val days are never touched".
- `fedsim.day_pool` asserts "no noise segments" and stops. The `setup` firewall (`assert role == {"train"}`) also rejects these days (they have m4 role "test").
- Report 08 §4.2 ("every event draws noise only from its own day") is therefore false for the final design.
- Feb 7 belongs to the dropout-dense run 0204 (83 % of its events have dropouts), so it will probably need the relaxed rule. The relaxed rule is currently a hand-picked list of days.
- **Fix.**
  - Pre-register a threshold rule (for example, "relaxed if the strict rule yields fewer than 35 segments") and build the 4 pools.
  - Replace the role firewall with a per-fold one: client events ∩ (eval ∪ server ∪ tuning) = ∅.
  - Implement the assertion that raw `bgpool.npy` is never loaded. §4.2 says it exists; it does not.

**M2. The statistics do not yet fit the nested, shared-seed design. Verified by simulation (`sim_power.py`, `sim_power_B_nested.csv`).**
- **Contradiction on seeds.** B assumed "seeds are independent per fold" (B §4). The protocol uses the same `torch.manual_seed(seed)` in every fold and one server checkpoint per seed shared across folds, so the seed effects are correlated across folds (ρ > 0, and ρ ≈ 1 for the S arms).
- **The "S_seed" reading changes the result.** Lobes, n = 5, σ_v = 0.03:
  - n = 15 uncentred seed×fold run means: power falls from 0.80 to 0.42, because fold/day difficulty (σ_day 0.088) enters S_seed.
  - n = 15 run means centred within fold, with ρ = 1 and an FL seed SD of 0.03: type I 0.07–0.15 (lobes) and up to 0.17 (polarity).
  - n = 5 init-seed means over all 9 days: type I ≤ 0.05 whenever σ_af = 0.
- **Arm × fold interaction.** Each fold has a different client partition, so an arm × fold interaction σ_af is plausible. At σ_af = m/2, type I is 0.07–0.14 for lobes and 0.13–0.18 for polarity under every reading. "Fold effects cancel within the day pairing" (08 l.240) is true only for fold main effects.
- **Fix.**
  - Specify that S_seed is computed on the 5 init-seed means over all 9 days. That estimator is valid whether ρ is 0 or 1. Alternatively, use distinct seeds per fold, with a server checkpoint per seed and fold.
  - Declare the estimand conditional on the 3 fixed partitions, in line with §4.9.
  - Report per-fold Δ and pre-state a heterogeneity flag: if the per-fold Δ span more than m, label the result "partition-dependent".
- **D = 9 re-simulation (B's model, 4 000 simulations).** The interpolation (about 0.75) is reasonable.
  - Lobes, σ_v = 0.030: n = 5 power **0.773**, n = 10 **0.877**.
  - Lobes, measured σ_v = 0.061: n = 5 **0.137**, n = 10 **0.177**.
  - Polarity: ≥ 0.998 in every cell.
  - Type I ≤ 0.048 in every cell.

**M3. The leakage gate is ill-defined and mislabelled.**
- **Unit of analysis.** "One value per seed, Welch" (08 l.157) no longer holds: there are 3 eval days per fold and 15 seed×fold values, and fold-specific day triplets differ in difficulty. The unit must be specified.
- **Role of the null.** How the "block-permutation null" enters the test is not stated.
- **Primary version.** It is not stated whether the unconditioned or the morph8/hour/dropout-conditioned version is the gate.
- **Margin.** m_leak = 0.03 was calibrated on 5 pooled days (`leak.csv`: central −0.029, CNN +0.056). It does not transfer to 3-day folds. Central skill is also below 0, so the metric is not chance-centred.
- **What it measures.** Day identity on *unseen* days measures how strongly day nuisance is encoded, not privacy. B's table row "fails the privacy claim" is not supported.
- **Report 08 vs B.** 08 §2.5 cites A's in-sample, 13-day 0.027 as "the baseline", but the gate is B's held-out definition.
- **L2 target.** L2 plants a "per-client offset" (08 l.275), but the gate is evaluated on eval days, which are not clients. L2 must plant per-eval-day offsets.
- **Fix.** Specify:
  - the statistic: null-centred skill per (seed, fold), averaged over folds per seed, n = 5;
  - the primary version;
  - a margin re-derived in the dry run (L2 grid and/or a C256 vs SimCLR-CNN fold run);
  - the name "nuisance-encoding gate".
  - If privacy is a goal, add client identity on held-out within-day blocks (B's option b3) as a secondary.

**M4. Tuning is asymmetric (§4.4).**
- S+F gets its own learning-rate search, but S+C inherits C256's learning rate. The interaction (S+F − F-nat) − (S+C − C256) is therefore biased towards "helps federation".
- F-X+P (+20 % budget) inherits F-nat's learning rate while S+F is tuned, which biases S+F vs F-X+P, the comparison §4.9 requires.
- C64 inherits a learning rate tuned for batch 256 at 4× the steps, which confounds C256→C64 with mis-tuning.
- **Fix.**
  - Tune S+C, and preferably C64 and F-X+P (12 more runs, about 1 h).
  - Alternatively, give S+F F-nat's learning rate.
  - Make "ties go to 3e-4" operational: keep 3e-4 unless another learning rate beats it by more than a pre-stated δ. Selection on one seed and one day is otherwise noise-driven (σ_resid lobes 0.027).

**M5. The decision rules are incomplete.**
- **Overlapping rows.** "LEAKIER (lower > 0)" can co-occur with "leakage OK (upper < 0.03)".
- **Missing rows:**
  - mixed outcomes across the two primaries (for example, polarity equivalent and lobes inconclusive);
  - superiority;
  - leakage failing while the primaries are inconclusive.
- **Missing tests.**
  - §4.9 makes "S+F helps" conditional on beating F-X+P, but §4.1 defines no test for it.
  - "F-IID closes the gap" and "C64 reproduces it" have no statistical definition.
- **Undeclared roles.** It is not stated which of the 6 contrasts against C256 are confirmatory; only F-nat vs C256 and S+F vs F-nat should be.
- **Fix.**
  - Write an exhaustive, mutually exclusive table.
  - Define the attribution tests, for example: "non-IID explains it" = F-IID vs C256 NI at m AND F-nat − F-IID upper bound < 0.
  - Add a one-sided S+F > F-X+P test to the Q-S family (Holm).
  - Declare the remaining contrasts descriptive.

### MINOR

- **m1. Tuning leak, low severity. Verified.**
  - The tuning runs train on the fold-0 clients, which include **all 6** eval days of folds 1 and 2 (6, 13, 26 / 7, 14, 27).
  - Selection uses only Feb 10 scores, so information flows only through a choice among 4 learning rates per arm, about 2 bits.
  - Fix: report fold 0 alone as a sensitivity analysis (its eval days are clean with respect to tuning), or tune on the 6 days that are never evaluated.
- **m2. Exposure figure is wrong for folds. Verified (`check_design.csv`).**
  - "Each event seen ~1 160 times" no longer holds: central is 995 / 1 009 / 927 epochs per fold, federated 909–1 024.
  - The server stays at 1 180, so the "matching exposure" rationale for P = 512 000 breaks.
  - Federated examples are 0.9945–0.9995 of X.
  - Fix: correct the text and choose P to match, or state the mismatch.
- **m3. Dry-run controls are too easy.**
  - C3 (Δ −0.33) and C4 at τ = 0.5 (Δ −0.13, more than 3m) are trivial failures.
  - C1 passes with probability ≈ 1 at σ_v = 0, so it can hardly fail.
  - Fix: add a near-margin planted control (τ calibrated so lobes Δ ≈ −m to −1.5m) that must not come out equivalent.
- **m4. Free and missing controls.**
  - The server checkpoint alone (the floor D requested) can be evaluated at no training cost. Add it.
  - D's "central on clients ∪ server" upper bound is absent. Either add it or state that S+C replaces it.
  - Report F-IID − C64 descriptively as a sanity check, without requiring it to be zero.
- **m5. Single points of failure.**
  - One server checkpoint per seed feeds 6 runs (3 folds × S+F/S+C), so a collapsed server seed propagates.
  - Tuning uses one seed.
  - Pre-state how collapsed runs enter the TOST (included, with a sensitivity analysis without them).
- **m6. The synthetic probe is partly contaminated.** The generator was fitted on 6 of the 9 eval days plus both server days, with families built from all days. This slightly favours the S arms. Keep it descriptive and say so.
- **m7. Monitoring uses contaminated references.** Monitoring uses the pooled `m4.bgpool()` (which contains other folds' eval days) and a CKA reference from M5 s0, which was trained on 6 eval days. Keep both monitoring-only, and use the fold's C256 as the "15-NN overlap with central" reference.
- **m8. Some recipe asymmetry remains.** T, the 10k steps and the learning rate were chosen using Feb 7, 13, 25 and 26, which are now eval days. This favours central, so an "inferior" result is anti-conservative. State it in §4.9.
- **m9. Text errors.**
  - The dry run is "section 7" (it is §4.7).
  - The compute budget is in "section 10" (no section 10 exists; it is §7), and the 10-seed option points to "section 10" as well.
  - F-X+P's R = 240 and the fresh warm-up after pretraining are not stated.
  - `margins.csv` proposes 0.02 for duration, while report 08 uses 0.03.
  - The lobe margin of 0.04 is below B's own 80 %-power floor at the measured σ_v (0.070, `margins.csv`), so "power-checked" is true only for σ_v ≤ 0.03.

### Checked and found sound
- `folds.json` is reproduced exactly by `folds.py`'s rule.
- Server, tuning, client and eval days are disjoint in every fold.
- No eval day comes from run 0218. Feb 19 as a client is acceptable.
- All numbers in report 08 §§2 and 4.2 match A, B and C's files and `folds.json`.
- The §7 compute plan matches C's timings: 39.9 min per seed-fold for 7 arms gives about 10 h sequential, against the claimed 12 h.
- Using an oracle evaluator that fits on client events is acceptable because it is identical for every arm.
- The aggregation weights and the learning-rate axis mapping in `fedsim` are correct on reading.

## 3. Checks run and their results
- **`check_design.py`:** folds reproduced (true); disjointness asserts pass; tuning overlap and generator overlap as above; 4 client days have no noise pool; per-fold accounting.
- **`sim_power.py`, part A:** B's model at D = 8/9/13 reproduces B's D = 8/13 values (lobes 0.710 vs 0.70, 0.927 vs 0.93); D = 9 as in M2.
- **`sim_power.py`, part B:** nested 3×3 design; 3 readings of S_seed × ρ ∈ {0, 1} × σ_af × FL seed SD (results in M2).
- **`test_fedavg_alias.py`:** a tiny GPU run (under 1 min, GPU idle afterwards). It proves B1.

## 4. Files created
All in this folder:
- `check_design.py`, `check_design.csv`, `check_design.json`
- `sim_power.py`, `sim_power_A_bmodel.csv`, `sim_power_B_nested.csv`
- `test_fedavg_alias.py`, `test_fedavg_alias.json`
- `_runs/`: symlinks to the per-day pools, plus the two 1-round test runs (1.8 MB, safe to delete).

Nothing was committed. One side effect: importing `fedsim` may have refreshed `C_federated/__pycache__`.
