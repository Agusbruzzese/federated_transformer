# Pre-registration M5: longer training, gated MAE, random and raw baselines in the ranking, calibrated leakage

_Orchestrator, 2026-10-02. Written before any M5 code or model exists. It implements the M4 reviewer's
recommendations (`M4_methods/review/REVIEW.md`)._

## Questions (in priority order)
- **Q1:** Does any self-supervised method beat **every** non-learned or untrained representation at encoding event shape?
- **Q2:** Among the SSL methods that actually trained, which is best?
- **Q3:** Do they carry run identity **beyond** what the physical event features already carry?

## Honesty statement
- M5 reuses M4's split, so the test days have already been seen in M4's results. No fresh test data exists; all 340 h have been used somewhere.
- The test results are therefore **confirmatory of M4's direction, not blind**.
- M5's rules are fixed before any M5 result.

## Data split (same as M4)
- **Tier:** SNR ≥ 20.
- **Test:** Feb 7, 13, 19, 25 (1 412 events, 167 blocks).
- **Validation:** Feb 10 (327).
- **Train:** all other days (2 636).
- **Background pool:** excludes the test and validation days.

## Representations

| ID | Representation | Seeds | Kind |
|---|---|---|---|
| S1 | SimCLR, transformer (T 0.1, lr 3e-4) | 5 | SSL |
| S2 | VICReg, transformer (lr 3e-4; λ/μ/ν 25/25/1; expander 64→256→256) | 5 | SSL |
| S3 | MAE, transformer, **standard recipe**: mask 0.75; per-patch normalised targets; 2-layer decoder 64→128→25 with GELU; loss on masked patches; input = random crop | 5 | SSL (**gated**, below) |
| S4 | SimCLR, CNN (T 0.1, lr 3e-4) | 5 | SSL |
| B1 | untrained CNN (same architecture as S4, init of seeds 0–4) | 5 | baseline |
| B2 | untrained transformer, **unpooled tokens** (24 × 64 = 1 536-d, init of seeds 0–4) | 5 | baseline |
| B3 | raw evaluation crop (600-d) | 1 | baseline |
| B4 | PCA-64 of the crop (fitted on train) | 1 | baseline |
| Ref | morph8, the 19 hand-made features | 1 | ceiling only, not ranked |

- **Hyperparameters:** fixed from M4's selection. No grid.
- **Embeddings:** S1, S2 and S4 use the 64-d `out` embedding. S3 uses pooled normed tokens (64-d). S3's unpooled tokens are reported as secondary.

## Training (all SSL)
- **10 000 optimiser steps:** batch 256 gives 10 steps per epoch, so 1 000 epochs.
- **Schedule:** AdamW (weight decay 1e-4); linear warm-up over the first 5 % of steps, then cosine decay.
- **Augmentations:** M4's.
- **Seeds:** 0–4. `torch.manual_seed(seed)` before the model is built; one generator per seed for augmentations and batch order.

## Gates (checked on train/validation only, before any test scoring)
- **G1, MAE gate.**
  - On the validation-day events: the S3 masked-patch MSE (normalised targets, mask 0.75, fixed mask seed) must be **< the MSE of linear interpolation** across the same masked patches, on the same normalised targets.
  - A seed that fails is **"not trained"**. If ≥ 3 of 5 seeds fail, S3 is excluded from Q2 and reported as "did not train under this recipe".
- **G2, learning sanity (all SSL).** The validation-day kNN mean over the 3 primary scores must exceed the method's own untrained weights. Failures are reported.
- **G3, logging.** Every number quoted in the report must exist in a saved file.

## Probes (as M4 B2)
- Fitted on training-day embeddings, evaluated on test days.
- **kNN:** k = 15, z-scored.
- **Linear:** `LogisticRegression(C=1, max_iter=3000)` for polarity and lobes; `RidgeCV(alphas=logspace(-3, 3, 13))` for duration.
- **Scaling for baselines B3/B4:** each uses z-scored or unscaled, whichever is better on the **validation day** under that probe. It is chosen before test scoring, so it favours the baselines without peeking at test.
- **Targets:** S1 polarity (skill), S2 lobes (skill, capped at 4), S3 duration (R² of log10 dur10). Skill = (balanced accuracy − 1/K)/(1 − 1/K).

## Statistics and decision rules
- **Bootstrap:** 2 000 resamples (seed 0) over test 30-min blocks, the same blocks for both members of a pair. Seeds are drawn independently with replacement for seeded representations; single-seed baselines use the block resampling only.
- **"X beats Y on a score":** under **both** probes, a mean gap ≥ 0.02 and a Holm-adjusted p < 0.05 with the 95 % CI excluding 0. The Holm family is all 28 pairs × 3 scores, per probe.
- **Q1:** an SSL method "beats the baselines" if it beats **each** of B1–B4 on ≥ 2 of 3 scores, and loses to none on any score.
- **Q2:** rank the SSL methods that passed G1 by pairwise wins minus losses among themselves. Ties are reported as ties.
- **If no SSL method beats all baselines,** the conclusion is: **"no evidence that self-supervised training adds value over random or raw features at this data size"**.

## Q3: calibrated leakage
- **Conditional run information:** Δ = skill(is-0204 | [embedding, morph8]) − skill(is-0204 | morph8 alone).
  - All 4 375 events; GroupKFold(5) by block; logistic regression and kNN; the max over the two probes.
  - Embeddings and morph8 are z-scored, and morph8 enters with equal weight via PCA-whitened concatenation (the embedding is PCA-reduced to ≤ 32 dims to avoid dimension dominance).
- **Flag "carries extra run information"** if Δ > 0.05 (5-seed mean).
- **Also reported:** raw is-0204 skill per representation, plus morph8's, raw's and PCA's for calibration; test-day identity (4 classes).

## Dry run (must pass before training)
Planted controls under both probes:
- morph8: polarity ≥ 0.90;
- noisy lobes: lobe skill ≥ 0.50;
- noisy duration: R² ≥ 0.50;
- noise: all ≤ 0.10.

Conditional-leakage controls:
- a planted run column on top of noise must give Δ ≥ 0.50;
- a copy of morph8 as the "embedding" must give Δ ≤ 0.05.

## Not claimed
- Vehicle types.
- Generalisation to unseen runs.
- Optimal hyperparameters.

## v2 amendments (after audit AUDIT_20261002_084233, before any M5 code). These override the text above.
- **C1, status.** All M5 results are **"exploratory, post-selection"**. The test days were seen in M4, and hyperparameters come from M4. No confirmatory claim is made.
- **C2, p-values.** Two-sided bootstrap p = (1 + 2·min(#{Δ*≤0}, #{Δ*≥0}))/(B + 1), with **B = 10 000**.
  - The Holm correction is applied to these p within each probe's family.
  - **"Y loses to X"** uses the identical rule with the sign reversed. Anything else is "no clear difference".
  - **Family per probe:** the SSL-vs-baseline pairs (4 SSL × 4 baselines) plus the SSL-vs-SSL pairs (6) = 22 pairs × 3 scores = 66 tests. Baseline-vs-baseline pairs are not tested.
- **C3, hierarchical bootstrap.**
  - Stage 1 draws the 4 test days with replacement. Stage 2 draws 30-min blocks with replacement within each drawn day. The same draw is used for both members of a pair.
  - Seeds are drawn independently with replacement per seeded representation.
  - Effective clusters: 4 days and 167 blocks, reported. With only 4 days, CIs are wide; this is accepted as conservative.
- **C4, dimension matching and scaling, applied to every representation.**
  - B2 (1 536-d) and B3 (600-d) also enter as PCA-64 versions (fitted on train).
  - **For Q1, an SSL method must beat the better of {full, PCA-64}** for B2 and B3, chosen per score and probe **on the validation day**.
  - For **every** representation and probe, z-scored vs unscaled is chosen on the validation day.
- **C5, gates.**
  - **G1:** the mask generator seed is 12345. Pass if the S3 validation masked MSE ≤ **0.90 ×** the linear-interpolation MSE. The S3 exclusion rule (≥ 3 of 5 seeds fail) is kept as stated.
  - **G2:** validation kNN mean ≥ own init + **0.02**.
- **C6, Q3 leakage (revised).**
  - Δ = skill(is-0204 | [emb19, morph8]) − skill(is-0204 | morph8). emb19 = the embedding PCA-reduced to 19 dims (the size of morph8); both blocks are z-scored.
  - **CV grouped by day** (18 groups), logistic regression **primary** and kNN reported. CI = bootstrap over the 5 seeds' Δ (seed resampling).
  - Δ is also computed for B1–B4 (calibration).
  - **Flag** "carries extra run information" if the mean Δ > 0.05 **and** the CI's lower bound > 0.
  - The dry run adds shuffled-label (day-level) nulls.
- **C7.** The final-step checkpoint is evaluated. Training curves are saved per run.
- **C8, limitations stated.**
  - The targets come from the morph features (circular w.r.t. Ref).
  - The SNR ≥ 20 restriction applies to every conclusion. The SNR ≥ 30 tier is reported as secondary.
  - B1/B2 share inits with S4/S1 by design.
- **C9, pinning.** `data/events/manifest.json` sha256 and the sha256 of `m5.py` are logged at freeze and at the end.
- **C10 (2026-10-02, after the gates ran, BEFORE any test scoring; the scoring run was stopped and wrote no output).**
  - **The bug:** G1 as implemented normalised interpolation by the TRUE patch spread. Near-flat true patches make the interpolation error explode (MSE 787 vs the model's 0.75), so the gate was degenerate and passed trivially.
  - **Corrected G1:** both predictors are judged on predicting each masked patch's **shape**.
    - The model's normalised prediction is compared with the true normalised patch.
    - The interpolation is normalised by **its own** mean and spread (all-constant interpolations are set to 0).
    - Patches whose true spread is < 1e-3 are excluded from both.
  - **The rest is unchanged:** pass if model MSE ≤ 0.90 × interpolation MSE; mae_excluded if ≥ 3 of 5 seeds fail.
  - **G2 is unchanged.** As recorded: SimCLR-TF 5/5 pass, VICReg 5/5, MAE 3/5, SimCLR-CNN 3/5. The G2 failures don't exclude anything; they are reported.
