# Independent review: M5, longer-trained SSL vs random and raw baselines

_Reviewer: independent and adversarial, with no stake in the outcome._

**Inputs read:**
- PREREG.md, including amendments C1–C10. The current sha256 prefix is d214fd055615c339, which matches the log.
- AUDIT_20261002_084233.md and m5.py (with m4.py).
- All result files: dryrun, gates, train.log, scores_abs, scaling_choice, comparisons, leakage, verdict and hashes.
- The M4 review and the tail of ORCHESTRATION_LOG.

**My code and outputs:**
- Scripts: `review/scripts/` (`dump_emb.py`, `recompute.py`, `q3_check.py`, `mae_gate.py`).
- Results: `review/recompute_q2.json`, `review/recompute_q3.json`, `review/mae_gate.json`.
- Everything uses the frozen checkpoints only. Nothing was retrained.

## Bottom line (≤120 words)
The numbers reproduce exactly, and "no SSL method beats every baseline" stands. But the official reading over-states both negatives.

- **Q1:** SSL clearly adds lobe-count information over random networks (+0.07 to +0.17). It adds little to polarity and duration, which random convolutional networks already nearly saturate. The ≥2-of-3, both-probes, Holm rule has about 5 % power against a uniform +0.05 gain, so "no evidence of value" is a property of the rule as much as of the data.
- **Q3:** the SimCLR-TF flag (Δ 0.053) lies inside a day-permutation null whose 95th percentile is 0.07–0.10. The is-0204 skills are all at or below chance, so the flags are not calibrated.
- **Bug:** the verdict code has a variant-matching bug. It does not change Q1 or Q2.

All results remain post-selection, on SNR ≥ 20 events, with morph-derived targets.

## 1. Recomputed numbers vs. the pipeline

For this check I wrote my own code:
- probes: a ridge with LOO-GCV via SVD, a numpy kNN (k = 15), and my own R²/skill functions;
- my own validation-day scaling choice;
- my own hierarchical bootstrap (B = 4 000, a different RNG);
- my own PCA, z-scoring and CV loop for Q3.

### Q2 and Q1 cells (duration, R² of log dur10)

| Comparison | Pipeline Δ [95 % CI], Holm p | Mine Δ [95 % CI], raw p | Per-day Δ (Feb 7/13/19/25) |
|---|---|---|---|
| SimCLR-TF − VICReg, lin | 0.0426 [0.0155, 0.0722], 0.045 | **0.0426** [0.0150, 0.0715], 0.0017 | .031/.039/.049/.049 |
| SimCLR-TF − SimCLR-CNN, lin | 0.0232 [0.0088, 0.0389], 0.028 | **0.0232** [0.0089, 0.0388], 0.0017 | .014/.024/.027/.025 |
| SimCLR-TF − B1 (random CNN), lin | 0.0523 [0.0290, 0.0760], 0.0066 | **0.0523** [0.0301, 0.0763], 0.0002 | .059/.034/.053/.063 |
| SimCLR-TF − B1, kNN | −0.0035 [−0.0389, 0.0356], 1.0 | **−0.0035** [−0.0383, 0.0369], 0.86 | +.018/−.012/−.012/−.003 |

- My validation-chosen scaling flags match `scaling_choice.csv` seed for seed.
- **Q2 consequence:** under the linear probe, SimCLR-TF beats both VICReg and the CNN on duration, on all 4 days. Under kNN the gaps are not significant (0.021 and 0.046). The "three-way tie" is therefore a probe disagreement, not equality.

### Q3: SimCLR-CNN conditional run information
My own implementation gives per-seed Δ of 0.134 / 0.126 / 0.105 / 0.142 / 0.127, a mean of **0.127**. That matches leakage.csv exactly. SimCLR-TF seed 0 gives 0.055, against the pipeline's 0.0554.

### MAE gate (G1)
I reproduce model 0.740–0.754, interpolation 1.315 and the zero-predictor 0.96. See section 3.

## 2. Were the rules applied as written?

### Holm family
- 66 tests per probe, confirmed: 16 SSL×baseline pairs plus 6 SSL×SSL pairs, × 3 scores. The family is computed separately per probe.
- **Caveat 1:** whenever B3 picks its PCA variant (duration under both probes, lobes under kNN), `B3_raw_pca` is the same matrix as `B4_pca64`. About 12 tests per probe are exact duplicates. This makes Holm slightly stricter.
- **Caveat 2:** the minimum attainable p is 1/10 001. Its Holm floor is 0.0066, and about 27–35 tests per probe sit at or near that floor.

### Q1, baseline variant chosen on validation
- The choice is applied per probe and score, using the validation mean over seeds, as C4 says.
- **Bug in `verdict()`:** it looks up the baseline by `R.b.str.startswith(bl)` *without filtering by probe*.
  - When kNN and the linear probe pick different variants, two names come back and the cell is silently forced to "no difference".
  - The affected cells are B2 polarity, B2 duration and B3 lobes.
- I re-derived the verdict with a per-probe match:
  - **SimCLR-TF vs B2, duration** becomes a **win** (kNN +0.077, Holm 0.027; lin +0.064, Holm 0.016). The pipeline reports 0.
  - **MAE vs B2, polarity** becomes a **loss**. The pipeline reports 1 B2 loss; the correct count is 2.
  - **Q1's conclusion is unchanged:** no method beats B1 on ≥ 2 scores. The best is SimCLR-CNN, with lobes only.
- `verdict.json` should be regenerated after the fix. The Q2 tallies are unaffected: I re-derived +2/+2/+2/−6 by hand.

### Hierarchical bootstrap with 4 days
It behaves sensibly here:
- The hierarchical SD is close to the block-only SD (0.0145 vs 0.0140; 0.0118 vs 0.0098).
- Bootstrap p values agree with a 3-df t-test on the four per-day Δ (0.0024–0.004 vs 0.0017–0.0002).

Two structural properties should be stated:
- **Positive on all 4 days is effectively required.** If any one test day has Δ ≤ 0, then 1/256 of the resamples consist only of that day, so p ≥ about 0.008. The Holm critical raw p is only about 0.0015, so such a comparison cannot win.
- **The procedure is not "conservative" as C3 claims.** Four clusters understate between-day variance by about 25 %, and the block stage partly offsets that. It is roughly calibrated, not conservative.

### Deviations from the PREREG
None of these changes a headline, but all should be disclosed:
- **Shuffled-label null missing.** C6 adds a day-level shuffled-label null to the dry run. In `dryrun()`, `yshuf`/`dmap` are computed and then discarded, so no null was ever run or saved. This omission matters for Q3 (below).
- **C8:** the SNR ≥ 30 secondary tier is not implemented.
- **Q3 identity:** test-day identity (4 classes) is not reported.
- **C3:** the effective-cluster counts are not written to a file.
- **C9:** no end-of-run m5.py hash was logged. m5.py legitimately changed with C10 (current prefix c41059dfc9dc14d3, vs 108863eb at freeze). The manifest hash matches.
- **Ref (morph8)** test scores are not in `scores_abs.csv`.

### C10 legitimacy
I judge C10 **legitimate**:
- **The original gate was broken.** It normalised interpolation by the true patch's spread, giving MSE 787 against 0.75. That compares a well-scaled predictor with a badly scaled one, so the gate could not fail.
- **The fix made the gate harder,** not easier.
- **It was made before any test scoring,** and the stopped run wrote nothing.
- **The pass does not depend on the particular fix.** I tried two fairer comparators the authors did not use:

| Comparator | Comparator MSE | Model 0.743 as a ratio |
|---|---|---|
| Zero-predictor | 0.96 | 0.77 |
| Interpolation shrunk by its optimal scalar (k = 0.24) | 0.922 | 0.81 |

Both ratios are under the 0.90 bar, for all 5 seeds.

Remaining concern: the corrected comparison treats the two predictors unequally.
- The model can shrink its predictions toward zero.
- The interpolation is forced to unit variance. That is why it scores 1.31, worse than predicting zeros.

The reported "0.74 vs 1.31" therefore exaggerates the margin. The honest margin is about 0.74 vs 0.92–0.96.

## 3. Fairness

### Q1: is "no evidence SSL adds value over random features" fair?
Only half fair.

**Power of the rule.** I used the observed hierarchical-bootstrap SDs for the SSL-vs-B1/B2 cells: polarity 0.018/0.014, lobes 0.035/0.037, duration 0.028/0.018 (kNN/lin). Holm's critical raw p in this family is about 0.0015 (z ≈ 3.15). For a uniform true gain Δ on every score and probe, the chance of beating *one* random baseline on ≥ 2 of 3 scores under both probes is:

| True gain Δ (every cell) | P(win vs one baseline) |
|---|---|
| 0.03 | ≈ 0 |
| **0.05** | **0.01–0.05** |
| 0.10 | 0.70–0.77 |
| 0.15 | ≈ 1 |

The method then has to do this against **both** B1 and B2.

**Ceiling effects.** B1 is already at polarity 0.885 (lin) and duration 0.82, so a +0.10 polarity gain is nearly impossible. In practice the rule needs about +0.12 on lobes *and* duration under both probes against a random CNN.

**What the data actually show:**
- **Lobes:** SimCLR-TF and SimCLR-CNN are +0.07 to +0.17 over B1 and B2 under both probes, and all 8 of these 95 % CIs exclude 0. The kNN gaps are Holm-significant. SimCLR-TF's linear gap over B1 just misses (raw p 0.0027, Holm 0.076). SimCLR-CNN's linear gap over B1 passes (Holm 0.045).
- **Polarity and duration:** the gains over B1 are small (≤ 0.05) or null (duration kNN −0.003).

**Fair wording:** "SSL adds lobe-count information that random networks lack. On polarity and duration, an untrained CNN is already as good under kNN. No method clears the pre-registered all-baselines bar." "No evidence that SSL adds value" is too strong as a scientific summary, even if it is the pre-registered sentence.

### G1: is the corrected MAE gate meaningful?
- **What it shows:** the MAE optimised. Its final shape MSE is 0.74, against 0.985–1.015 for its own init. It predicts about 23 % of the per-patch shape variance (1 − 0.743/0.96), so it is no longer the M4 "never trained" model.
- **What it does not show:** that the representation is useful. MAE ranks last (−6), loses to B1/B2 on polarity and lobes, and passes G2 on only 3/5 seeds.
- **The right reading:** "MAE trained but its pooled embedding is poor at this scale". That is not "masked modelling is worse". Pooling was already shown in M4 to hurt MAE, and the unpooled-token secondary was pre-registered but not reported.

### Q3: is the flag calibrated?
No.

- **Every is-0204 skill is at or below chance under day-grouped CV.** morph8 is −0.141 (−0.088 with leave-one-day-out), and the embeddings alone range from −0.23 to +0.02.
- **SimCLR-CNN's Δ = 0.127 means** "[emb, morph8] reaches chance (−0.007), where morph8 alone anti-predicts". It does not mean run identity is decodable on unseen days.
  - SimCLR-CNN alone: +0.002.
  - With leave-one-day-out: alone 0.030, combined 0.022.
- **Nulls (mine):**
  - 19 Gaussian noise dimensions give Δ = −0.009 ± 0.010, so the effect is not pure dilution.
  - A day-level permutation null (is-0204 assigned to 8 random days of 18, 60 permutations) gives:

| Embedding | Mean Δ | SD | 95th percentile | Max |
|---|---|---|---|---|
| SimCLR-CNN s0 | 0.017 | 0.049 | 0.101 | 0.110 |
| B1 s0 | 0.013 | 0.031 | 0.070 | 0.097 |

- **Calibrated against these nulls:**
  - SimCLR-CNN's 0.134 exceeds all 60 permutations (p ≈ 0.016, single seed).
  - **SimCLR-TF's 0.053 is well inside the null.**
- **The flag's CI resamples only the 5 seeds.** It ignores the day-level variability, which dominates, so a "lower bound > 0" is nearly automatic.

**Fair wording:**
- "No representation, including morph8, identifies run 0204 on held-out days above chance."
- "SimCLR-CNN's embedding changes how morph8 generalises across days (Δ beyond a permutation null). SimCLR-TF's flag is within noise."

### Do the scores mean anything about vehicles?
Not directly.
- **The targets are morph-derived:** polarity sign, lobe count and log duration. These are waveform-shape descriptors, and random features already capture two of them almost perfectly.
- **No vehicle label (class, axles, speed, length) exists anywhere in the pipeline.** Duration is plausibly related to speed or length, and lobes to axle or structure, but neither link has been validated.
- **The tests are narrow:** SNR ≥ 20 (about 29 % of events), in-distribution days that were already seen in M4.

## 4. What CAN and CANNOT be concluded

**CAN** (exploratory, post-selection, SNR ≥ 20, in-distribution days, morph-derived targets):
- All four SSL methods and both random networks beat raw and PCA crops by a wide margin, especially under linear probes (lobes +0.38 to +0.46, duration R² +0.65 to +0.72).
- **SimCLR (both backbones) adds lobe-count information** over untrained CNN and transformer features (+0.07 to +0.17; every 95 % CI excludes 0).
- **On polarity and duration, an untrained CNN is near the SSL level** (within 0.05; duration kNN equal).
- **With 10 k steps the transformer catches up with the CNN.** SimCLR-TF ≈ SimCLR-CNN on lobes (Δ 0.005/0.012), and SimCLR-TF is best on duration under the linear probe (beating VICReg and the CNN on all 4 days).
- **The MAE now optimises,** shown robustly against fairer comparators. Its pooled embedding is the weakest of the SSL methods and below the random networks on polarity and lobes.

**CANNOT:**
- That SSL "adds no value over random features". The rule had about 5 % power for +0.05 gains, and the lobe gain is real.
- That SimCLR-TF, VICReg and SimCLR-CNN are equivalent. They tie under the rule because the probes disagree.
- That SimCLR-TF carries extra run information. Its Δ is within a permutation null.
- That any model leaks run identity to unseen days. Every is-0204 skill is ≤ chance.
- That masked modelling is an inferior objective. Only one pooled-embedding recipe was tested.
- Anything about vehicle types, the SNR < 20 events, unseen runs, or optimal hyperparameters.

## 5. Is data size the binding constraint? Next step

**Pretraining data is not the main constraint.**
- The evaluation is saturated: B1 reaches polarity 0.885 and duration 0.82 with zero training, so there is little headroom for any SSL gain on 2 of 3 targets.
- The test has only 4 day-clusters, which forces "positive on all 4 days".
- The targets are functions of the same morph features.

More pretraining events would not create headroom on targets that random convolutions already solve. A cheap secondary step is to pretrain on all SNR tiers, which would roughly triple the unlabeled data, since SSL needs no clean targets. It would not fix the evaluation.

**Single most valuable next step:** obtain **external labels for a few hundred events** that are not functions of morph8, from a co-located reference such as a camera, a traffic counter or a WIM/axle log, for example vehicle class or axle count and speed.

Then score SSL against B1/B2 as **label-efficiency curves** (10, 30, 100 and 300 labels) with **run-held-out** splits. That is the regime where SSL is supposed to matter, it removes the target circularity, and it is the only route to a statement about vehicles.

If that is impossible, the fallback is to pre-register a fresh evaluation with ≥ 8 held-out days (not 4) and a test family restricted to SSL vs B1/B2 only. That would raise power by roughly 2× in SE terms.
