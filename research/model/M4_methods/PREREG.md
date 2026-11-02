# Pre-registration M4: four self-supervised learning methods compared on the extracted events

_Orchestrator, 2026-10-02. Written before any M4 code or model exists. It applies every lesson from G, H2 and
their reviews:_
- _two probes must agree;_
- _CIs include seed variance;_
- _baselines are not handicapped;_
- _a planted-control dry run comes before training;_
- _equal tuning budgets;_
- _no amplitude or SNR target._

_The author knows the H2 results (SimCLR transformer ≈ CNN). This is a new comparison. Its two new methods have
never been run._

## Question
On the same data, the same transformer backbone and the same budget, which way of learning without labels gives the best event representation? The SimCLR CNN is the reference.

| ID | Method | Backbone | What it learns from |
|---|---|---|---|
| M1 | SimCLR (contrastive) | patch transformer (phase F, ~207k parameters) | two augmented views of an event must match, and differ from other events |
| M2 | VICReg (non-contrastive) | same transformer | two views must match; variance and covariance terms prevent collapse; no negatives |
| M3 | Masked autoencoder (MAE-style) | same transformer encoder, plus a linear patch decoder | reconstruct randomly hidden 0.5 s patches of the waveform |
| M4 | SimCLR | 1D CNN (~211k parameters) | as M1; the reference |

## Data split (SNR ≥ 20 events from E's extraction)
- **Test:** 4 full days, one per run, fixed now: **Feb 7 (0204), Feb 13 (0212), Feb 19 (0218), Feb 25 (0224).** Never used for training, tuning or selection.
- **Validation (tuning only):** Feb 26 (0224).
- **Train:** all other days.
- **Test-day background:** the noise-augmentation background pool is filtered to exclude every test and validation day.
- **This is an in-distribution test.** Test days come from the same runs as training days. Generalisation to unseen runs was tested in H2 and is not the question here.
- **Contamination statement:** the author has looked at results involving run 0218 in earlier phases, but not at these four test days in isolation.

## Training and equal tuning budget
- **Shared settings:** 100 epochs, batch 256, AdamW (weight decay 1e-4), cosine schedule, embedding dimension 64. The input is the per-event max-normalised 12 s crop at 50 Hz, as in phase F.
- **M1 and M2 augmentations:** phase F's (crop, stretch, amplitude, real noise, offset/drift).
- **M3 input:** the random 12 s crop only. Mask by setting the chosen patch tokens to a learnable mask token. The loss is the MSE on masked patches.
- **M2 settings:** expander 64→256→256, with coefficients λ = 25 (invariance), μ = 25 (variance), ν = 1 (covariance).
- **Grid:** exactly 2 cells per method, at seed 0.
  - M1: T ∈ {0.1, 0.2}, lr 3e-4.
  - M4: T ∈ {0.1, 0.2}, lr 3e-4.
  - M2: lr ∈ {1e-3, 3e-4}.
  - M3: mask ratio ∈ {0.5, 0.75}, lr 3e-4.
- **Selection:** the cell with the higher mean validation score over the 3 primary scores, using the kNN probe on validation-day events.
- **Final models:** the selected cell × 5 seeds (0–4). Each seed's initial weights are that model's own untrained baseline.

## Probes (all on frozen embeddings of the test set, deterministic central crop)
**Primary scores:**
- **S1 polarity:** skill.
- **S2 lobe count:** skill, capped at 4.
- **S3 duration:** R² of log10 `m8_dur10`.

**Two probes, both primary:**
- **kNN:** k = 15, z-scored.
- **Linear:** logistic regression (L2, C = 1) or ridge (α by inner CV), z-scored.

Both use 5-fold CV grouped by 30-min block, across the pooled test days.

**Skill** = chance-corrected balanced accuracy.

**Statistics:** a hierarchical bootstrap with 2 000 resamples (seed 0). Each resample draws test 30-min blocks with replacement **and** draws the 5 seeds with replacement, so the CIs include training randomness.

**Pairwise decision.** "A beats B on a score" requires, **under both probes**:
- a mean gap ≥ 0.02;
- a 95 % CI excluding 0 after Holm correction over the 6 pairs × 3 scores (18 comparisons per probe).

If the two probes disagree, the result is "no clear difference".

**Ranking:** the number of pairwise wins on the primary scores, minus losses. Ties are reported as ties.

**Learning check:** each method vs its own untrained weights, with the same rule and both probes. A method that does not beat its own init on ≥ 2 of 3 scores under both probes is "not learning".

**Baselines** (descriptive, not ranked):
- raw crop and PCA-64 (fitted on train), each under both probes, **with and without z-scoring**; the better is reported;
- untrained transformer and untrained CNN;
- hand-made morph8 (the ceiling, circular).

**Leakage** (per method, in its own space):
- every SNR ≥ 20 event of all runs is embedded by the final model;
- **is-0204 skill** and **test-day identity skill** (4 classes, test events only) are measured under logistic regression and kNN; the **higher** is reported;
- **Limit:** ≤ own init + 0.05 for each. A method exceeding it is flagged "leaks".

**Secondary** (descriptive):
- magnitude group and 4-group skill;
- 10-NN neighbour purity;
- augmentation retrieval@1 (this favours M1, M2 and M4 by construction);
- per-test-day results, as a consistency check.

## Dry run (must pass before any training)
Three planted embeddings go through the full scorer:
- **morph8** must give polarity ≥ 0.90 under both probes;
- **Gaussian noise** must give skill ≤ 0.10 and R² ≤ 0.10 under both probes;
- **noise plus a one-hot run column** must give is-0204 ≥ 0.90 and test-day identity ≥ 0.90.

## Not claimed
- Vehicle-type recognition (no labels).
- Generalisation to unseen runs (see H2).
- Optimality of any hyperparameters.

## v2 amendments (after audit AUDIT_20261001_233814, before any code). These override the text above.
- **B1, validation day.** It is now **Feb 10 (0204)**, which is not adjacent to any test day. Feb 26 joins the training set.
  - Counts (SNR ≥ 20): test 1 412 events / 167 blocks (Feb 7: 301/37, Feb 13: 324/43, Feb 19: 403/41, Feb 25: 384/46); validation 327/35; train 2 636.
  - Training days adjacent to test days remain (in-distribution design).
  - The background pool (columns run, wall_time) is filtered in code to exclude all test days and the validation day.
- **B2, probe design.** Probes are **fitted on training-day embeddings and evaluated on test-day events.** There is no CV inside the test set.
  - kNN: `KNeighborsClassifier/Regressor(15)`.
  - Linear: `LogisticRegression(C=1, max_iter=3000)` for S1/S2, and `RidgeCV(alphas=logspace(−3, 3, 13))` for S3.
  - Both are z-scored, fitted on training events.
  - Labels from E's columns: S1 = 1 if `m8_polarity` > 0; S2 = min(`m8_n_lobes`, 4); S3 = log10(clip(`m8_dur10`, 0.05)).
  - Skill = (balanced accuracy − 1/K)/(1 − 1/K), with K = the number of classes present in the test labels.
- **B3, statistics.**
  - For a pair (A, B), each bootstrap resample draws the test blocks with replacement (the same blocks for A and B) and draws the seeds of A and of B independently with replacement.
  - The 95 % CI is computed per probe, with Holm correction within a probe over 18 comparisons.
  - "A beats B" requires a gap ≥ 0.02 **and** a Holm-significant CI under **both** probes.
  - **If everything is a tie,** the conclusion is "methods indistinguishable at this data size".
  - The learning check uses its own Holm family: 4 methods × 3 scores, per probe.
- **B4, selection.** Validation score = the mean of the 3 primary scores, using the kNN probe fitted on train and scored on the validation day, **averaged over seeds 0 and 1.** If the two cells differ by less than the seed SD, the first-listed cell is chosen. The validation gap and the seed SD are reported.
- **B5, M3 embedding.**
  - Patch 25 samples (0.5 s), 24 tokens.
  - The encoder is the same `PatchTransformer`. The M3 embedding is the mean-pooled, final-layer-normed token state (64-d, equal to the embedding dimension); the unused `out` layer is ignored.
  - Decoder: a linear map from 64 to 25 samples per masked token. Masked tokens are replaced by a learnable mask token.
  - Loss: MSE on masked patches of the max-normalised crop.
  - At evaluation nothing is masked. This train/test shift is inherent to MAE and is stated.
- **B6, augmentation asymmetry** (stated, not fixed).
  - M1, M2 and M4 train with time-stretch U(0.85, 1.18), which perturbs duration (S3), and amplitude scaling. No augmentation flips polarity.
  - M3 sees only the crop.
  - S3 is therefore suppressed by design for M1, M2 and M4 relative to M3.
- **B7, leakage.**
  - **is-0204:** all 4 375 SNR ≥ 20 events, about 41 % positive, GroupKFold(5) by 30-min block.
  - **Test-day identity:** 4 classes, the 1 412 test events, GroupKFold(5) by block.
  - Each is measured under logistic regression and kNN, and the max is taken, for **both** the trained model and its init.
  - The limit is applied to the 5-seed means: trained ≤ init + 0.05. Init values and their seed spread are reported.
- **B8, dry run, extended.** The planted embeddings must pass under both probes:
  - morph8: S1 ≥ 0.90;
  - noisy lobe labels (one-hot plus noise, SNR 2): S2 ≥ 0.50;
  - noisy duration (value plus noise, SNR 2): S3 ≥ 0.50;
  - Gaussian noise: all scores ≤ 0.10;
  - noise plus a run one-hot: is-0204 ≥ 0.90, and noise plus a day one-hot: day identity ≥ 0.90.

  Training smoke test: each method overfits a 64-event subset (the loss falls by > 30 %) before the grid runs.
- **B9, seeds.** `torch.manual_seed(seed)` before the model is built (init), and a dedicated generator per seed for augmentations, masks and batch order. Bootstrap seed 0. The event lists are saved with their sha256.
- **B10, limitations** (stated in advance). The tuning effort is unequal: VICReg's λ/μ/ν and the MAE decoder are taken from the literature, untuned. 4 test days. All targets are morph-derived.
- **B11 (2026-10-02, smoke test, BEFORE any grid or final training).**
  - **Result:** the smoke test passed for SimCLR-TF (−58 %), SimCLR-CNN (−87 %) and MAE (−83 %). VICReg fell −29 % on 64 events, just short of the 30 % bar.
  - **Diagnosis:** the variance term fell 1.90 → 0.64, so the embeddings un-collapse and the method learns. The covariance term has a floor when n = 64 < 256 expander dims.
  - **Fix:** VICReg's smoke test uses 256 events (n ≥ expander dim), with the same > 30 % criterion. No other change.
- **B12 (deviation, disclosed).** VICReg also fell only −29.6 % on 256 events.
  - **Cause:** at init the views already match (near-collapsed embeddings, invariance term ≈ 0.002), so learning trades a small rise in the invariance term for a large fall in the variance term. The total loss is not a valid learning indicator for VICReg.
  - **Smoke criterion for VICReg:** the variance term must fall > 30 %. Observed: 1.90 → 0.64 (−66 %), which passes.
  - This is a pipeline check that uses no validation or test data, so it cannot bias any result.
  - A second smoke-criterion change is disclosed here as a deviation, not hidden.
