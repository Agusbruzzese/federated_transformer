# Independent review of H2: "a SimCLR transformer encodes vehicle-event shape better than the alternatives"

_Reviewer: an independent agent with no part in H2. I modified nothing outside `review/`. My scripts are `recompute.py` (own kNN on the GPU, own skill and R², own block folds, own linear probe; it does not import `h2.py`) and `deltas.py`. Outputs are in `recompute_scores.csv` and `recompute_deltas.csv`. Compute was about 6 min of GPU and CPU time._

## Bottom line

The experiment does **not** show that the transformer (TF) is better than the alternatives. Calling it "Partly supported" is a correct mechanical reading of the frozen rules, but the parts it supports are the easy ones:
- **H1 (training helps):** robust.
- **H3 (beats raw and PCA):** a low bar. PCA is also handicapped by z-scoring.

Against the one serious competitor, a parameter-matched CNN, the TF:
- ties on polarity;
- loses on lobe count in 4/4 runs;
- wins on duration by +0.024 R², which only just clears the 0.02 margin and **reverses (−0.019, 0/4 runs) under a linear probe.**

H4 passes only because amendment A5 switched to a logistic-regression probe. Under v1's "worse of LR/kNN" rule, the is-0204 gap is 0.064 (> 0.05), which replicates the pilot's +0.07. An honest summary: *the TF learns shape, and it is no better than a CNN.*

---

## 1. Recomputed numbers vs h2.py

**Method.** The seeds are averaged. "Random" folds assign the 30-min blocks to 5 folds at random. "Sorted" folds are time-contiguous groups of blocks. Both differ from sklearn's GroupKFold. The kNN is my own (k = 15, z-scored on the training fold, majority vote or mean).

### 1a. TF vs CNN (oracle temperature per run and score, as in h2.py) on runs other than 0212

| Run / score | h2.py | mine, kNN (random folds) | mine, kNN (sorted folds) | mine, **linear probe** |
|---|---|---|---|---|
| 0224 polarity | +0.004 | +0.015 | +0.007 | −0.012 |
| 0224 lobes | −0.076 | −0.092 | −0.104 | −0.119 |
| 0224 duration | +0.030 | +0.031 | +0.032 | −0.020 |
| 0204 polarity | +0.001 | +0.009 | −0.002 | −0.013 |
| 0204 lobes | −0.029 | −0.033 | −0.050 | −0.079 |
| 0204 duration | +0.021 | +0.021 | +0.020 | −0.026 |
| **Mean over 4 runs:** duration | +0.0239 (4/4 runs > 0) | +0.024 (4/4) | +0.020 (4/4; 0212 only +0.007) | **−0.019 (0/4)** |
| **Mean over 4 runs:** lobes | −0.050 (0/4) | −0.055 (0/4) | −0.055 (0/4) | **−0.106 (0/4)** |
| **Mean over 4 runs:** polarity | +0.001 (3/4) | +0.005 (3/4) | +0.002 (2/4) | −0.009 (0/4) |

**Agreement with h2.py.** The kNN absolute scores match within about 0.01 to 0.02 on random folds. On sorted folds they differ by up to 0.06, the largest being TF lobes on 0204: 0.313 vs 0.373. Lobe count is therefore sensitive to the fold assignment, by an amount comparable to the TF−CNN gaps.

**The TF's only "win" (duration) is fragile:**
- It sits exactly at the 0.02 margin under my sorted folds.
- It reverses with a linear probe. Under a linear probe (logistic regression C = 1, ridge α = 1, untuned) **the CNN beats the TF on all three primary scores in all four runs.**

### 1b. TF vs its own untrained weights (H1)

| Run | Score | h2.py | Mine, kNN (random) | Mine, linear |
|---|---|---|---|---|
| 0224 | polarity | +0.262 | +0.265 | +0.083 |
| 0224 | lobes | +0.194 | +0.182 | +0.256 |
| 0224 | duration | +0.097 | +0.094 | +0.080 |
| 0204 | polarity | +0.220 | +0.226 | +0.093 |
| 0204 | lobes | +0.170 | +0.152 | +0.251 |
| 0204 | duration | +0.093 | +0.087 | +0.068 |

H1 replicates under every probe and fold scheme, positive in 4/4 runs.

### 1c. Baselines h2.py does not use

**An untrained CNN**, a fair non-learned baseline with convolutional inductive bias. TF trained minus random CNN, kNN:
- polarity +0.054, 4/4 runs;
- lobes +0.081, 4/4;
- duration +0.023, 4/4, but 0212 only +0.006.

With a linear probe the margins are +0.021, +0.052 and +0.042. A random CNN already reaches polarity 0.79–0.85, against 0.58–0.67 for the random TF.

**PCA-64 without z-scoring.** Z-scoring equalises all 64 components and inflates the noise directions. Without it:

| Score | PCA-64, z-scored | PCA-64, not z-scored |
|---|---|---|
| polarity | 0.63–0.70 | 0.78–0.83 |
| lobes | 0.10–0.17 | 0.28–0.36 |
| duration | 0.29–0.40 | 0.59–0.66 |

TF minus non-z-scored PCA (sorted folds) is still positive in 4/4 runs, but much smaller:
- polarity +0.06 to +0.08 (reported +0.20);
- lobes +0.03 to +0.09 (reported +0.25);
- duration +0.13 to +0.18 (reported +0.43).

H3 survives, but its reported margins are inflated by a strawman PCA. My linear-probe numbers for RAW and PCA (negative R²) come from an untuned 600-dimensional ridge and are not meaningful.

### 1d. H4 (from `leakage.csv`, recomputed independently of verdict())

**is-0204 gap, TF − TF-init, mean over 20 (fold, seed) pairs:**

| Probe | Gap | Per fold: 0204 / 0212 / 0218 / 0224 |
|---|---|---|
| **LR** (A5 primary) | **0.037** | 0.022 / 0.038 / 0.040 / 0.049 |
| **kNN** | **0.066** | 0.050 / 0.083 / 0.070 / 0.062 |
| **max(LR, kNN)** (v1 rule) | **0.064** | |

The 0218-fold kNN gap of **0.070 reproduces the pilot's +0.070 exactly.** The pilot used max(LR, kNN). The "0.037 vs pilot 0.07" discrepancy is therefore entirely the change of probe, not a change in the model.

**Under LR, 0.037 is robust.** A hierarchical bootstrap over runs and then seeds gives 95 % CI [0.025, 0.049], with P(gap > 0.05) ≈ 0.02. It is not robust to the probe choice.

**The run-identity signal is real.** The TF's is-0204 LR skill exceeds its own shuffled-label null maximum in 20/20 seeds, by a mean of +0.08 over the null mean. It is small, but it is there.

## 2. Was the pre-registration followed? Were A10 and the bug fix legitimate?

**Decision rules (A3/A4) were applied as written.** I re-derived every point delta from the absolute rows (exact match). I also hand-checked the decisive cells:
- cnn_oracle duration: mean 0.0239; 4/4 runs > 0; lower CI bound > 0 in 3/4 (0212 −0.016) → "TF better". ✓
- cnn_oracle lobes: all 4 runs negative; upper CI bound < 0 in 2/4 (0218, 0224) → "no clear difference". ✓
- H2 therefore fails with 1 "TF better" out of 3. ✓

**Minor deviations, none of which changes the verdict:**
1. **Oracle granularity.** A4 says the oracle is chosen "per run"; h2.py chooses it per run *and per score*. I re-ran with one temperature per run (best mean over the primary scores): duration TF +0.036, lobes −0.050, polarity +0.004. The verdict is the same.
2. **The excl-0218 check** uses "≥ 2 of 3 CIs", an unwritten but natural adaptation.
3. **examples() uses yet another oracle rule** (mean over the primary scores). This is descriptive only.
4. **h2.py's docstring cites the pre-A10 hash.** This is cosmetic.
5. **The CIs cover only the test blocks.** Predictions are fixed and seeds averaged, so training-seed variance is not in the interval. For CNN lobes, the per-seed SD is 0.02–0.06, as large as the effects. The CIs are too narrow for the TF−CNN comparisons.

**A10 (dry-run fixes).** Its timing is plausible:
- PREREG, dryrun.json and the thresholds are stamped 22:46;
- training finished at 23:03;
- scoring ran at 23:20.

1. **Noise-check fix.** Legitimate and neutral for the SSL results: it only affects the noise control.
2. **Date limit `max(0.03, morph8 + 0.03)`.** Logically justified: A5's limit was below chance on 3 runs, which made it degenerate. **It was not result-neutral in hindsight, though.** Under the unamended A5 limit, run 0212 fails: TF date skill −0.0761 vs limit −0.0777, a miss of 0.0016. I see no evidence that the change was made with that knowledge. The bigger issue is that the date probe is broken: every representation, pure noise included, scores −0.02 to −0.16. **The date half of H4 has essentially no power to fail.** For what it is worth, the TF's date skill exceeds TF-init's in 4/4 runs (+0.006 to +0.037).

**A5 switch from "worse of LR/kNN" to "LR primary".**
- *Procedure:* it was made before training, on the auditor's advice, so it is procedurally legitimate.
- *Effect:* it is the switch that decides H4. The author knew from the pilot that the max-rule gap was +0.07 (a fail).
- *Reporting:* A5 promises that "kNN is reported", but verdict.json and the decisions files report only the LR gap.

This should be disclosed next to H4 ✓.

**The verdict() dtype bug fix** is a legitimate code fix. No score changed, and the decisions are consistent with the score rows.

## 3. Over- and under-claims

**H1 ✓ is correct but easy.** A random patch transformer is a weak starting point (polarity 0.58–0.67, below raw at 0.76). A large gain over it says little about the TF relative to other designs. A random CNN is already 0.17–0.24 better on polarity.

**H2 ✗ is correct, and it is the decisive result for the claim as worded.** The write-up should not soften it:
- the CNN's lobe advantage is consistent in direction (4/4 runs, every probe and fold scheme, −0.03 to −0.12);
- the TF's duration edge depends on the probe;
- if anything, the evidence tilts toward the CNN.

Saying "no clear difference" for lobes is correct under the rules but under-states the consistency. Saying "TF better on duration" is correct under the rules but over-states its robustness.

**H3 ✓ is a low bar, and the margins are over-stated.**
- RAW is an unaligned 600-d Euclidean kNN.
- PCA is crippled by z-scoring.
- A fair non-learned baseline would be a random CNN, or PCA without z-scoring. Against it the TF still wins in 4/4 runs, but by 0.02–0.08 rather than 0.10–0.43.

**H4 ✓ is probe-dependent.**
- It passes under LR (0.037, CI [0.025, 0.049]).
- It fails under kNN (0.066) and under the v1 rule (0.064).
- The date half cannot fail.

Better wording: "run identifiability rises by 0.04 (LR) to 0.07 (kNN) over init, on an absolute scale of ≤ 0.09; borderline."

**The verdict label "Partly supported" over-claims in plain English.** The claim is "better than the alternatives". The only real alternative (the CNN) is not beaten. A reader will take "partly supported" as "partly true". It should read **"Not supported against the CNN; training effect and non-learned baselines confirmed."**

**Are the scores meaningful?** They are three coarse proxies taken from hand features:
- **polarity:** the sign of the first lobe;
- **lobe count:** threshold-dependent, with a morph8 kNN ceiling of only 0.60–0.73, so it is noisy;
- **duration:** with a ±15 % time-stretch augmentation that pushes the encoders to be partly invariant to it.

High skill shows that embedding neighbourhoods respect these 3 descriptors, not "shape" in a richer sense. A representation could score near the ceiling by encoding three scalars. The probe also matters: kNN favours isotropic, locally uniform geometry, and on these data it seems to favour the TF. The linear probe favours the CNN. Neither was justified as the "right" reading of "encodes".

**Is the CNN oracle fair?** Yes, and it barely matters:
- At the CNN's own phase-F temperature (T 0.2), the CNN is ahead on lobes by 0.049 (0/4 runs TF-positive), and the TF is ahead on duration by +0.042.
- At matched T 0.1, the lobes gap is −0.021 (1/4) and duration +0.024.
- The TF is insensitive to temperature (T 0.2 − T 0.1 ≤ 0.018 everywhere).
- The oracle's bias toward the CNN is ≤ 0.03 and does not decide H2.

**Is lobe count simply favoured by convolution?** The evidence is consistent with that, but it is not proven:
- A random CNN already beats a random TF on lobes by 0.06–0.10, and beats raw.
- Most of the CNN's SSL gain falls on lobes, as the critique noted.
- The TF tokenises into 0.5 s linear patches (25 samples) and mean-pools, which plausibly blurs sub-patch zero crossings.

## 4. What the experiment CAN and CANNOT conclude

**CAN (on 4 runs, this sensor, SNR ≥ 20, ~2.6–3.7 k training events, 100 epochs, ~207 k parameters):**
- SimCLR training gives the patch transformer much more polarity, lobe and duration structure than its random init. This holds in 4/4 held-out runs, under kNN and linear probes and every fold scheme tried.
- TF embeddings beat raw waveforms and PCA on these proxies in 4/4 runs. The fair margin is modest.
- A parameter-matched SimCLR CNN is at least as good as the TF:
  - equal on polarity;
  - better on lobe count, consistently;
  - duration depends on the probe (TF +0.02 with kNN, CNN +0.02 with a linear probe).
- TF training raises run-0204 identifiability slightly, by +0.04 (LR) to +0.07 (kNN), on a small absolute scale.

**CANNOT:**
- That the transformer encodes event shape *better than the alternatives*. It does not beat the CNN.
- That the TF's advantage over a non-learned baseline is large, once a random CNN is the baseline.
- Anything about scaling (data, epochs, model size): only one size and one schedule were tested.
- Anything about vehicles, travel direction or other sensors. Statistical power is limited to n = 4 runs (minimum sign-test p = 0.0625).
- That date leakage is absent. The date probe cannot detect it.

## 5. The single most informative next experiment

**A pre-registered tokenisation ablation of the transformer**, keeping the claim a "pure transformer":
- **Design:** patch size 25 (current) vs 10 vs 5 samples at about 207 k parameters, trained with the identical recipe on the 4 folds × 5 seeds, against the existing CNN.
- **Probes:** score kNN and a linear probe as **co-primary**. A TF claim should then require winning under both.

**Why this, over more data or longer training:**
- Lobe count is the only consistent TF deficit, and 0.5 s tokenisation is the obvious mechanism.
- If small patches close the lobe gap while keeping the duration edge, H2 could flip to "TF ≥ CNN on 2/3". The claim could then be restated about patch size.
- If they don't, the "convolution favours lobes" explanation gains support, and the claim is dead for this data size.

More data is not available beyond 4 runs. A data-fraction curve (25/50/100 %) with 3× epochs is the second choice: it could show only a trend, not a reversal at the current data size.
