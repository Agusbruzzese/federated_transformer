# Pre-registration v3: central SSL, tightened evaluation (phase G)

_Orchestrator, 2026-10-01. Written before any phase G code or result exists. It answers the F audit
(`F_central/AUDIT_20260930_213853.md`) and the orchestrator's own admission that v2's criteria were too loose
(ORCHESTRATION_LOG, "Audit F"). The phase F results are known to the author, so v3 is a **confirmatory
re-test with stricter rules**, not a blind first look. Once G starts, this file is frozen: its sha256 is
logged, and deviations are listed in the G report._

## What changes vs v2 (issue → fix)

| F-audit issue | Fix in v3 |
|---|---|
| #2: log_amp is predictable from SNR alone | P1 target `log_amp` → **`amp_resid`**: the residual of `m8_log_amp` after a linear fit on log10(SNR), fitted inside each probe training fold. |
| #3: 3 seeds; unstable random-init baseline | **5 seeds** per architecture per fold, each paired with its own random-init. |
| #4: selection on one day of 366 events | **Per-fold selection by leave-one-day-out CV** of the P1 probe over training-run events (§Selection). |
| #5: P2 tolerances could not fail; has_dropout null broken | **Absolute and paired P2 criteria** (§P2). has_dropout is tested **within run 0204 only**, so day prevalence can't stand in for it. |
| #6: SSL doubles run-0204 identifiability | A **new criterion, P2c**: SSL must not raise is-0204 skill by more than 0.05 above its own random-init. |
| #8: all-SNR verdicts were not meaningful | Secondary tiers are **descriptive only**: no verdict. |
| #9: single-run holdout | **Leave-one-run-out (LORO)** over all 4 runs. Every event is embedded by the model that never saw its run. |
| #10: CNN duration loss is an untested hypothesis | **Ablation:** the CNN trained without time-stretch (secondary). |
| #7: evaluation-code edits unverifiable | The sha256 of every `ssl_central*/` source file is logged at freeze, before evaluation, and again at the end. |
| P2 runtime (HGB took 6 h) | The primary leakage probe is **logistic regression** (fast; 50 permutations). HGB is secondary, with 10 permutations. |

## Data
- E's `data/events/`, tier `snr ≥ 20` (4 375 events).
- P1 targets come from `m8_*`; the 30-min blocks are the clustering unit (both as in v2 and addendum 1).
- Runs: 0204, 0212, 0218, 0224.
- **Folds:** for k in the 4 runs, the test set is run k and training uses the other three runs. All training-run events are used for SSL; there is no separate validation day.

## Models and training (unchanged from v2 unless stated)
- **Architectures:** the CNN (GroupNorm) and the patch transformer from phase F, same code and sizes.
- **Recipe:** the same augmentations, NT-Xent and <30 s guard. 100 epochs, batch 256, AdamW (weight decay 1e-4), cosine schedule, embedding dimension 64.
- **Grid per fold:** lr {1e-3, 3e-4} × T {0.1, 0.2}.
- **Selection:** for each grid cell (seed 0), embed the fold's training events. Run the P1 probe (kNN k = 15) with **leave-one-day-out CV** over the training days; mean over the 4 targets with `amp_resid`. Pick the best cell per architecture per fold. **Test-run events are never used for selection.**
- **Final models:** the selected cell × 5 seeds (0–4). The random-init baseline is each seed's own initial weights.
- **Out-of-fold (OOF) embeddings:** each event is embedded by its fold's models, the ones that did not see its run. So there are 5 OOF embedding sets per architecture (one per seed), plus 5 random-init sets.
- **Ablation (secondary, CNN only):** the selected CNN cell with time-stretch disabled, 5 seeds, LORO.

## Baselines
- (a) morph8 scalars;
- (b) PCA-64 of the evaluation crop, fitted per fold on training runs;
- (c) random-init CNN and random-init transformer;
- (d) SNR alone.

## Statistics
- **Clusters:** 30-min blocks.
- **CIs:** block bootstrap, 1000 resamples, seed 0.
- **Seeds:** metrics are per seed; the paired difference SSL − random-init is taken by seed index, then averaged over 5 seeds before bootstrapping. The seed SD and the min/max of the per-seed Δ are reported.
- **Skill** = (balanced accuracy − 1/K)/(1 − 1/K).
- **Holm correction** within each family (the P1 targets; the P2 criteria).

## Primary evaluation (on OOF embeddings, all four runs pooled)

**P1, physical probes.**
- Probe: kNN k = 15, z-scored, 5-fold CV grouped by block.
- Targets: `amp_resid` (R²), `log_dur` = log10 `m8_dur10` (R²), `polarity` (skill), `n_lobes` capped at 4 (skill).
- A target **passes** if pooled Δ(SSL − random-init) > 0 with a Holm-adjusted p < 0.05 **and** the per-fold Δ is positive in **≥ 3 of 4** held-out runs.
- **P1 passes** if ≥ 3 of 4 targets pass.

**P2, leakage.**
- Probe: logistic regression (L2, C = 1, z-scored), CV grouped by block. Null: 50 permutations at group level.
- Each criterion is tested on the 5-seed mean against its threshold, with a bootstrap CI reported.

| | Criterion | Threshold |
|---|---|---|
| P2a | date skill (pooled, 18 dates) | ≤ **0.03** |
| P2b | run skill **excluding run 0204** (3 classes) | ≤ **0.03** |
| P2c | is-0204 skill (binary, all runs) | ≤ (own random-init is-0204 skill + **0.05**), and ≤ morph8's |
| P2d | has_dropout skill **within run 0204** (grouped by block) | ≤ (own random-init + **0.05**) |

**P2 passes** if all four hold.

## Verdict
- **"Positive"** = P1 and P2 both pass.
- **"Learns but leaks"** = P1 passes, P2 fails.
- **"No evidence of learning"** = P1 fails.
- **CNN vs transformer:** reported as the paired difference in pooled P1 mean, with a CI. A preference is stated only if the CI excludes 0; otherwise the result is "tie". P2 is used as the tie-breaker.

## Secondary (descriptive, no verdict)
- Per-fold P1/P2 tables, so it is visible which run drives what.
- HGB leakage probe with 10 permutations.
- P3 retrieval@1 and P4 cross-seed Jaccard.
- SNR ≥ 30 and all-SNR tiers.
- The CNN no-stretch ablation, which tests the duration hypothesis.
- Comparison with the phase F numbers.

## Explicitly not claimed
- Car-model recognition.
- Generalisation beyond this sensor.
- That the selected hyperparameters are optimal.
