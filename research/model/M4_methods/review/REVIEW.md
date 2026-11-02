# Independent review: M4, four SSL methods on vehicle-event waveforms

_Reviewer: independent, adversarial, no stake. Inputs: PREREG.md (v1 + B1–B12), m4.py, dryrun/smoke/selection/scores_abs/comparisons/leakage/verdict, train.log, AUDIT_20261001_233814.md, H2 review. My own code and outputs are in `review/scripts/` and `review/recompute.json`._

## Bottom line (≤120 words)
The pipeline applied its own rules correctly, and my recomputations reproduce its numbers. The scientific reading still has to be much narrower than the verdict suggests.

- **What the data support:** SimCLR beats VICReg on lobe count at this budget. SimCLR-CNN gives the best absolute embeddings.
- **"CNN not learning"** reflects a strong untrained CNN, not a failed CNN.
- **"MAE learns nothing"** comes from an MAE that never trained. Its reconstruction error (0.062–0.065) is barely below predicting zeros (0.069), and linear interpolation does twice as well (0.035).
- **The leakage flag is uncalibrated.** Hand-made morph8 features, which involve no training, reach is-0204 skill 0.096. That is above SimCLR-TF's "leaking" 0.087.
- **SimCLR-TF is barely above simple baselines.** On lobes it is only marginally above raw-kNN and the random CNN.

## 1. Recomputed numbers vs. the pipeline
I wrote my own probes for this check: a numpy kNN (k = 15, majority vote/mean), sklearn `balanced_accuracy_score(adjusted=True)`, my own block × seed hierarchical bootstrap (B = 2000, different RNG seed), and per-day gaps. I used the same frozen checkpoints and the same train→test protocol.

| Comparison | Probe | Pipeline Δ [95% CI] | Mine Δ [95% CI], raw p | Per-day Δ (Feb 7/13/19/25) |
|---|---|---|---|---|
| SimCLR-TF − VICReg, lobes | kNN | 0.077 [0.031, 0.121] | 0.077 [0.030, 0.122], 0.0015 | .10/.06/.10/.07 |
| SimCLR-TF − VICReg, lobes | lin | 0.122 [0.079, 0.167] | 0.121 [0.075, 0.164], 0.0005 | .12/.10/.12/.16 |
| CNN − VICReg, duration | kNN | −0.015 [−0.050, 0.020] | −0.015 [−0.050, 0.020] | −.07/+.04/−.02/−.01 |
| CNN − VICReg, duration | lin | 0.034 [0.014, 0.056] | 0.034 [0.015, 0.056] | all positive |
| CNN − init, polarity | kNN | 0.028, Holm p 0.11 | 0.028 [0.002, 0.055], raw p 0.033 | .05/−.01/.04/.03 |
| CNN − init, polarity | lin | 0.011 | 0.011 [−0.006, 0.030] | mixed |
| CNN − init, duration | kNN | −0.005 | −0.005 [−0.037, 0.031] | ≈0 |
| CNN − init, duration | lin | 0.057 | 0.057 [0.040, 0.077] | all positive |
| CNN − init, lobes | kNN / lin | 0.158 / 0.130 | 0.158 / 0.130 | all positive |
| MAE − init, pol / lobes / dur | kNN | .037 / .027 / −.005 | .037 / .027 / −.005, all CIs include 0 | – |
| MAE − init, pol / lobes / dur | lin | .017 / .004 / −.028 | .017 / .004 / −.028 (dur CI < 0) | – |

The verdicts reproduce:
- **SimCLR-TF > VICReg on lobes** is robust. It holds under both probes and on all 4 test days.
- **CNN learns on 1/3 scores** (lobes only). Polarity misses because the linear gain is below 0.02 and the kNN gain is not Holm-significant. Duration misses because the probes disagree: the linear gain is +0.057, the kNN gain is −0.005.
- **MAE learns on 0/3 scores.** Its duration score (linear probe) is significantly *worse* than its own initial weights.

**Additional checks that change the interpretation** (all mine):

| Check | Value |
|---|---|
| MAE masked-patch MSE, trained, 50% mask (eval crops; seeds 0, 1) | 0.065, 0.062 |
| Same, if you predict **zeros** | 0.069 |
| Same, if you **linearly interpolate** from visible samples | 0.035 |
| MAE unpooled tokens (24×64, flattened): lobes linear, final vs init | 0.597 vs 0.603 (no gain) |
| MAE unpooled tokens: duration, final − init (kNN / lin) | −0.024 / −0.049 (worse) |
| is-0204 leakage skill: raw / PCA-64 / **morph8** | 0.012 / 0.038 / **0.096** |
| is-0204: SimCLR-TF / VICReg / CNN / MAE (final, pipeline) | 0.087 / 0.057 / 0.136 / 0.030 |

## 2. Fairness of each conclusion

### "SimCLR-CNN not learning"
The rule was applied correctly, but the label is misleading. The **untrained** CNN is already a strong feature extractor (random convolutions plus pooling act as random local features):

| Score (kNN / lin) | CNN at init | SimCLR-TF trained | Raw |
|---|---|---|---|
| Lobes | 0.47 / 0.57 | 0.52 / 0.585 | 0.50 / 0.26 |
| Duration | 0.815 / 0.821 | – | – |
| Polarity | 0.843 / 0.885 | – | – |

SimCLR training adds +0.13 to +0.16 on lobes (robust on every day). It adds little on polarity, which is near ceiling at about 0.87–0.90. On duration the probes disagree (+0.057 linear, ≈0 kNN), and the time-stretch augmentation actively pushes duration out of the embedding (B6).

The correct statement is: *"SimCLR adds lobe information on top of a strong convolutional prior; most of the CNN's polarity and duration skill is architectural, not learned."* The trained CNN still has the best absolute scores on 5 of 6 probe×score cells. "Learning check" here measures the gain over the method's own init, so it cannot be compared across backbones whose inits differ this much. The random TF init is weak, which makes its "learning" easy to show.

### "MAE learns nothing"
This is **not a valid finding about MAE**. It is evidence that this MAE was never trained to a working state.

- **Reconstruction:** the final masked MSE (0.062–0.065) is only 5–10% below predicting zeros (0.069), and nearly 2× worse than naive linear interpolation (0.035). The training-loss drop 0.16 → 0.064 in train.log is almost entirely the random decoder learning to output ≈0.
- **The readout is not the cause.** The unpooled 1 536-d token maps carry more information than the pooled embedding (lobes linear 0.60 vs 0.35), but trained = init there too. Mean pooling hurts absolute MAE scores (pooled lobes 0.31/0.35 vs raw-kNN 0.50), but it does not explain the zero gain.
- **Likely causes:**
  - **Budget:** 2 636 events / 256 per batch ≈ 10 steps per epoch, so about **1 000 optimiser steps** in total. MAE/SimMIM normally needs 10–100× more.
  - **Recipe:** no warm-up, a linear decoder, and no per-patch target normalisation.
  - **Untuned grid:** only mask ratio was varied.
- **The smoke test passed (−83%)** only because 64 events can be memorised. It cannot detect this failure.

The right wording is "M3 as implemented did not train (it never beat a trivial reconstruction baseline)". MAE's −7 rank should be excluded from any method-level claim.

### Leakage limit (+0.05 over init)
The limit is uncalibrated and conflates information with leakage.
- Run 0204's events are physically distinguishable: hand-made morph8 features, which have no training and cannot learn run identity, give is-0204 skill **0.096**.
- Any representation that captures morphology better than its init will therefore "leak" by this rule. So the flag rewards the methods that learned least (MAE, whose is-0204 fell, and VICReg) and penalises the ones that learned most.
- The absolute values are small: is-0204 ≤ 0.14, and test-day identity is at chance (≤ 0) for every model.
- The max over two probes adds further upward bias.

SimCLR-TF (0.087) is below morph8. The CNN (0.136) is somewhat above it, which is mildly interesting but not evidence of a shortcut. A meaningful test would measure run identity **after conditioning on the targets**, for example within strata matched on morph8, or as a residual after regressing out morph8. The "leaks" flags for SimCLR-TF and the CNN should be reported as "exceeds an uncalibrated limit", not as leakage.

### Baselines: raw / PCA kNN lobes ≈ 0.50 vs SimCLR-TF 0.52
- **Under kNN,** the learned transformer is not meaningfully better than raw waveforms on lobes (+0.02) or polarity (0.873 vs 0.855). It clearly wins only on duration (0.85 vs 0.59).
- **Under the linear probe,** SimCLR-TF is far ahead of raw (lobes 0.585 vs 0.26). Its gain is mainly *linear accessibility*, not new information.
- **It roughly equals random-feature baselines:**
  - the untrained CNN (lobes 0.47/0.57);
  - the untrained transformer's unpooled tokens (lobes linear 0.60).
- **Baselines were descriptive only:** single seed, no CI, not entered into the ranking. So the study cannot claim that any SSL method beats raw waveforms or random networks with statistical support.

What this implies: at this data size and budget, **inductive bias (convolution, no pooling bottleneck) matters as much as the SSL objective**. The headline "method X beats method Y" sits on top of a small margin over trivial representations.

## 3. Were the decision rules applied as written?
- **Pairwise rule:** gap ≥ 0.02, Holm p < 0.05 and CI excluding 0, under both probes. Applied correctly; I re-derived every win in verdict.json from comparisons.csv.
  - Ranks: CNN +4 (lobes vs TF and VICReg; polarity and lobes vs MAE), TF +3, VICReg 0, MAE −7. Correct.
  - Minor: the CI is the unadjusted 95% CI, combined with a Holm-adjusted p. This is stricter than or equal to the PREREG's "CI after Holm", so it is acceptable.
  - Minor: `wins()` returns +1 if no rows exist (vacuous `all`). This was not triggered here.
- **Learning check:** separate Holm family of 12 per probe (B3). Applied correctly. VICReg passes duration by a hair (Holm p 0.045 / 0.0475).
- **Selection (B4):** I checked the gaps against the seed SD.

  | Method | Cell 0 | Cell 1 | Gap | Seed SD | Choice |
  |---|---|---|---|---|---|
  | SimCLR-TF | 0.747 | 0.721 | – | 0.009 | c0 |
  | CNN | – | – | 0.008 | 0.007 | c0 |
  | VICReg | 0.648 | 0.712 | – | – | c1 |
  | MAE | – | – | 0.031 | 0.015 | c0 |

  All choices are correct. The seed SD from n = 2 is near-meaningless, but it was pre-registered.
- **Dry run:** passed and is sound. (Morph8 polarity is circular, as stated.)
- **Smoke test, B11/B12:** procedurally weak, substantively harmless.
  - `smoke.json` still says `"PASS": false`, including for the 256-event re-run (−29.6%). The gate formally failed twice.
  - B11's diagnosis (a covariance floor at n < 256) was **falsified** by its own re-run: 256 events gave the same −29.6%.
  - B12 then switched to a VICReg-specific criterion (variance term −66%). That number, "1.90 → 0.64", is quoted identically in B11 (64 events) and B12 (256 events), and it appears in **no saved artefact**, so I cannot verify it.
  - Why it is still legitimate:
    - The smoke test is a pipeline check that uses no validation or test data.
    - The variance term is a more sensible VICReg signal than the total loss.
    - On the full training set VICReg's total loss fell about 41% (43.5 → 25.3, train.log), so it would have passed the original bar anyway.
    - VICReg passes the downstream learning check on 3/3 scores.

  Verdict: legitimate, but the variance-term log should be saved and smoke.json updated. The real lesson is that the smoke test failed to catch the MAE failure. The gate checked "the loss falls", not "the loss beats a trivial baseline".
- **Other:**
  - Equal tuning effort was not achieved (acknowledged in B10).
  - The total budget of about 1 000 steps is very short for every method. This was not acknowledged.

## 4. What CAN and CANNOT be concluded

**CAN (at 2 636 training events, about 1 000 steps, in-distribution test days):**
- SimCLR (either backbone) gives better lobe-count embeddings than this VICReg configuration. The result is robust across probes and on all 4 days.
- The SimCLR-CNN embeddings have the best absolute lobe and duration scores. SimCLR-CNN beats SimCLR-TF on lobes (Δ ≈ 0.10–0.12).
- SimCLR-TF and VICReg improve substantially over a (weak) random transformer.
- No model separates the 4 test days beyond chance.

**CANNOT:**
- That MAE (masked modelling) is a worse SSL objective. This MAE never learned to reconstruct.
- That the CNN "does not learn". It learns lobes; its other skills are architectural.
- That SimCLR-TF or the CNN leak run identity. The limit is uncalibrated, and morph8 exceeds SimCLR-TF.
- That any SSL method beats raw-waveform kNN or random-network features with statistical support. The baselines were never tested.
- Any ranking of objectives independent of backbone or training length. The objective (SimCLR vs VICReg vs MAE) and the backbone (CNN vs TF) are both confounded with training adequacy.
- Generalisation to unseen runs, vehicle types, or optimal hyperparameters (as the PREREG states).

## 5. The single most valuable next experiment
Run a **training-adequacy re-run with gates that cannot be passed trivially**, and promote the baselines to ranked competitors.

1. **Budget.** Train all four methods for a fixed budget of about **10 000 optimiser steps**, about 10× the current one, with a short warm-up.
2. **MAE gate.** Give MAE a pre-registered gate: masked MSE on the validation day must be below the linear-interpolation baseline (0.035). Use the standard recipe fixes (per-patch normalised targets, a 2-layer transformer decoder, mask ratio 0.75). If it cannot pass the gate, MAE is reported "not trained", not "worse".
3. **Ranking.** Score everything with the frozen M4 scorer. Enter the **untrained CNN, untrained-TF unpooled tokens, and raw-kNN** into the same Holm-corrected pairwise ranking.
4. **Leakage.** Recalibrate leakage against morph8, or compute it conditional on the targets.

This one run tells you whether *any* SSL objective beats random features at this data size. That is the prerequisite for every method claim in this study, and it is the question the current verdict silently assumes is answered.

---
_Reproduction:_
- `review/scripts/emb.py` embeds all final and init models, plus the MAE token maps.
- `rev.py` runs my probes and bootstrap (output in `recompute.json`).
- `leak.py` computes the baseline leakage and the MAE trivial-reconstruction baselines.
- `maeloss.py` measures the trained MAE's masked MSE.

Run them from the repo root with `.venv/bin/python <script> <scratch.npz> …`.
