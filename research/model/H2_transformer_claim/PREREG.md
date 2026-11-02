# Pre-registration H2 (v2, after audit AUDIT_20261001_224245): is the SimCLR transformer better than the alternatives at encoding event shape?

_Orchestrator, 2026-10-01. Written before any H2 model is trained. It applies the lessons from G and the critique:
one embedding space per held-out run, comparison against each model's own untrained weights, a dry run on
planted controls, no amplitude/SNR target, and no per-fold re-selection. The H pilot (single split, test run 0218)
is known to the author. Its numbers motivated H2 and are **not** evidence for it._

## Claim under test
"A SimCLR-trained transformer encoder produces event embeddings that encode the shape of vehicle events better than
the alternatives."

## Shape scores (targets; all on the test run, SNR ≥ 20)
These are fixed from the hand-made morph8 columns of E's extraction:
- **S1 polarity:** skill, up-first vs down-first.
- **S2 lobe count:** skill, capped at 4.
- **S3 duration:** R² of log10 `m8_dur10`.
- **S4 magnitude group:** skill, small (< 5 000 counts) vs large.
- **S5 4 groups:** skill, magnitude × polarity.

**Probe:** kNN, k = 15, z-scored, 5-fold CV grouped by 30-min block **within the held-out run**. Skill = chance-corrected balanced accuracy.

## Competitors
- **TF:** the SimCLR patch transformer (phase F architecture, about 207k parameters). **Fixed cell:** lr 3e-4, T 0.1.
- **CNN:** the SimCLR CNN (about 211k parameters, parameter-matched), lr 3e-4. It is trained at **T 0.1 and T 0.2**, and for each held-out run it is scored at **whichever temperature is best on that run's own test scores (an oracle choice).** This deliberately favours the CNN, so a TF win is conservative.
- **TF-init:** the transformer's own untrained weights (same seed).
- **PCA-64:** PCA of the evaluation crop, fitted on training runs only.
- **RAW:** the evaluation crop itself (600 samples, max-normalised).
- **Reference (not a competitor):** morph8. It contains the targets, so it is the ceiling.

## Data and replication
- **Leave-one-run-out:** 4 folds over runs 0204, 0212, 0218, 0224. A model is trained on 3 runs and scored only on the 4th, in its own embedding space.
- **5 seeds** per (architecture, temperature, fold): 4 folds × 5 seeds × (TF + CNN at T 0.1 and T 0.2) = 60 trainings. TF at T 0.2 is also trained, for a sensitivity check only (another 20).
- 100 epochs, batch 256, the existing `ssl_central` recipe, hyperparameters fixed in advance. The background pool excludes the test run.

## Hypotheses and decision rules
**Unit of evidence:** the paired difference TF − competitor per (held-out run, seed). Each run contributes the mean over its 5 seeds.

A score counts as "TF better than X" only if all three hold:
1. The mean difference is **≥ 0.02**.
2. The difference is **> 0 in all 4 held-out runs** (seed means). This is the decisive replication criterion: 4/4 runs agreeing.
3. A **hierarchical bootstrap** p < 0.05 after Holm correction over the 5 scores. Each resample draws 4 runs with replacement, then 30-min blocks within each run; 2 000 resamples, seed 0.

The opposite rule defines "X better than TF". Anything else is "no clear difference".

| H | Statement | Supported if | Refuted if |
|---|---|---|---|
| **H1** | Training teaches TF shape | TF > TF-init on ≥ 4 of 5 scores | TF > TF-init on ≤ 2 |
| **H2** | TF beats the CNN | TF > CNN on ≥ 3 of 5, **and** CNN > TF on none | CNN > TF on ≥ 2 |
| **H3** | TF beats non-learned representations | TF > PCA-64 **and** TF > RAW on ≥ 4 of 5 each | either ≤ 2 |
| **H4** | TF's shape knowledge isn't bought with leakage | in TF's space, **across all runs**: is-0204 skill ≤ TF-init + 0.05, **and** date skill within the held-out run ≤ 0.03 | either limit exceeded |

The leakage probes use the worse (higher) of logistic regression and kNN.

**Overall verdict:**
- **"Proven (on this data)"** = H1, H2, H3 and H4 all supported.
- **"Partly supported"** = H1 and H3 supported, but H2 or H4 not.
- **"Not supported"** otherwise.

## Controls and examples
- **Dry run before training.** Planted signal (morph8 as the embedding) must score high. Gaussian noise must score ≈ 0. Noise plus a one-hot run column must fail H4.
- **Sensitivity (descriptive):** TF at T 0.2, and TF − CNN at matched temperature.
- **Examples (descriptive, fixed now):**
  1. **Nearest-neighbour galleries:** 8 query events from held-out run 0218 (2 per 4-group, chosen by seed 7). For each, show the 5 nearest test-run neighbours in TF (seed 0), CNN (seed 0, its oracle temperature) and TF-init (seed 0).
  2. **Neighbour purity:** the share of the 10 nearest neighbours in the same 4-group, averaged over all test events, per run and competitor.
  3. **A 2-D map** (t-SNE, seed 0) of run-0218 embeddings for TF, CNN and TF-init, coloured by 4-group.

## v2 amendments (from the audit, made before training; these override the text above where they conflict)

**A1. Crop and inputs.**
- Every competitor sees the same deterministic central 12 s evaluation crop (600 samples at 50 Hz, phase F choice I3), and all are probed on the same events.
- The morph8 targets are computed on the central ~8 s inside that crop.
- Training and evaluation use the SNR ≥ 20 tier.
- The augmentations are phase F's. The background pool excludes the test run.
- Embedding dimension: 64 for TF, CNN and PCA; 600 for RAW.

**A2. Primary vs secondary scores.**
- **Primary family:** S1 polarity, S2 lobe count, S3 duration.
- **Secondary (reported, not counted):** S4 magnitude group and S5 4-group. S4 is amplitude-derived and S5 is a function of S1 and S4.
- The 5 000-count threshold is the valley in the amplitude histogram, where a 2-component Gaussian mixture had the best BIC (data analysis of 2026-10-01).

**A3. Decision rule per score and competitor.** With 4 runs, the smallest possible exact sign-test p is 1/16 = 0.0625, so no run-level p < 0.05 is attainable. "TF better than X on a score" now requires all of:
- (a) mean Δ ≥ 0.02 over runs;
- (b) Δ > 0 in **all 4** held-out runs (seed means);
- (c) in **≥ 3 of 4** runs, the run's own 98.3 % CI lies above 0. The CI is a block bootstrap (2 000 resamples over test-run 30-min blocks, seeds averaged, seed 0). 98.3 % is Bonferroni over the 3 primary scores.

The exact sign-test p (minimum 0.0625) is reported, alongside effect sizes and intervals. The reverse rule defines "X better than TF".

**A4. Hypotheses on the primary family.**
- **H1:** TF > TF-init on ≥ 2 of 3, and TF-init > TF on none.
- **H2:** TF > CNN on ≥ 2 of 3, and CNN > TF on none. The CNN is shown at its oracle temperature, chosen **after averaging seeds, per run**, and also at T 0.1 and T 0.2 fixed. The H2 verdict uses the oracle; the fixed-temperature results are reported.
- **H3:** TF > PCA-64 and TF > RAW, each on ≥ 2 of 3.

**A5. H4 leakage, exact procedure.** For each fold model (and its own init), embed ALL SNR ≥ 20 events of all runs in that one model's space.
- **is-0204:** binary skill. Logistic regression is primary; kNN is reported. GroupKFold(5) by 30-min block, 20 shuffled-label nulls. **Limit:** mean over (fold, seed) of TF − TF-init ≤ 0.05.
- **date within the held-out run:** logistic regression skill. **Limit:** ≤ morph8's own within-run date skill + 0.03. morph8 is computed in the dry run, and the threshold is set from it before any TF result.
- H4 is reported per fold, with **the 0204 fold separately**.

**A6. Verdict wording.**
- **"Supported on this data"** = H1, H2, H3 and H4 all hold.
- **"Partly supported"** = H1 and H3 hold, but H2 or H4 doesn't.
- **"Not supported"** otherwise.

The word "proven" is not used.

**A7. Pilot contamination.**
- The hyperparameters were selected in phase F (validation day Feb 26), not in the H pilot.
- Run 0218 was the H pilot's test run, so its fold is reported separately. Every verdict is also given **excluding 0218**, as a 3-run check: it must not reverse.
- **The examples use run 0212** instead of 0218: galleries (2 queries per 4-group, seed 7) and the t-SNE map (perplexity 30, seed 0).

**A8. Dry-run numeric thresholds** (must pass before training):
- morph8 as the embedding: S1 ≥ 0.90.
- Gaussian noise: |score| ≤ 0.10 on all scores, is-0204 ≤ 0.10.
- Noise plus a one-hot run column: is-0204 ≥ 0.90.

**A9. Seeds.** Training seeds 0–4; GroupKFold is deterministic (sorted groups); bootstrap seed 0; t-SNE seed 0.

## Not claimed
- Car-model or vehicle-type recognition.
- That polarity equals direction of travel.
- Anything beyond this sensor and these 4 runs.
- Statistical power is limited by **4 held-out runs**; the 4/4 rule is strict on purpose.

**A10 (2026-10-01, after the dry run and BEFORE any SSL model was scored).** The dry run (`dryrun.json`, first attempt) exposed two rule bugs:
1. **The noise check** treated a negative kNN R² as failure. Pure noise gives duration R² of −0.06 to −0.11, which is the expected sign for a no-signal regression. Fixed: the noise check requires skill scores ≤ 0.10 in absolute value and duration R² ≤ 0.10. It must show no *positive* signal.
2. **The date limit** "morph8 + 0.03" fell below chance, because morph8's within-run date skill is negative on 3 runs (−0.11 to −0.16, a grouped-CV artefact with only 3 days). Fixed: limit = **max(0.03, morph8 + 0.03)** per run.

Nothing else changed.
