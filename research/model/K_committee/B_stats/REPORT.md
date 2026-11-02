# Committee member B (statistics / experimental design): report, 2026-10-10

_Saved verbatim by the orchestrator from the agent's hand-back. Orchestrator checks:_
- _`controls_prelim.csv` matches §7: C1 lobes −0.011 [−0.050, 0.027]; C3 lobes −0.332; C4 τ=0.5 lobes −0.132 (inferior)._
- _`power_fast.csv` exists, with the cells shown._
- _No processes were left running and the GPU was idle after the hand-back._

All results below come from the M5 SimCLR-TF checkpoints (seeds 0–4) scored per seed and per day on 5 already-seen days (4 test days plus the val day). They are exploratory. The run is not complete: the 8-seed rescoring did not happen, the planted-leak grid is too strong, and the bootstrap-calibration simulation was not run (see §8).

## 1. Summary of recommendations

**Estimand**
- Δ is the day-averaged difference in a probe score between an arm and central (C256), on the same events, in each model's own embedding space.
- The probe is refit on each arm's own embedding.
- Unit of analysis: (seed, day). The estimand averages over seeds and over the population of days.

**Claims**
- "Federated is equivalent to central" is tested by TOST: the 90 % CI must lie inside ±m.
- "Federated is not worse than central" is a non-inferiority test: the lower 95 % one-sided bound must be above −m.
- "No extra leakage" is a non-inferiority test on day-identity skill: the upper bound must be below +0.03.

**Primary family**
- Lobe count and polarity, each a composite (mean of kNN and linear-probe skill), with margins **m = 0.04** (lobes) and **m = 0.02** (polarity).
- Conjunctive equivalence uses an intersection–union test (IUT), which needs no multiplicity correction.
- Duration is demoted to secondary, with m = 0.03 (justification in §3).

**Inference**
- Use a two-stratum t-procedure: SE² = S_seed·(1/n_A + 1/n_C) + S_day/D, with Satterthwaite degrees of freedom.
- This is preferred over the hierarchical bootstrap, whose percentile intervals with D ≤ 4 days rest on very few distinct resamples (35 multisets for 4 days).

**What drives power**
- The **number of evaluation days and the arm × day interaction** dominate. Seeds matter less.
- The measured seed SD is small: lobes 0.007, polarity ≈ 0.
- The seed × day residual is lobes 0.027, polarity 0.008.
- The arm × day interaction between *different recipes* is large for lobes: 0.03–0.06.

**Design**
- **10 seeds per core arm**, with init seeds paired across arms. This is affordable at about 7 min per run, per member C.
- **At least 8 evaluation days, ideally 13**, via 3-fold cross-fitting over the non-server days. Each fold retrains every arm.
- With only the 4 exhausted test days (3 clean days for the S-arms, since Feb 19 lies inside the server's run 0218), lobe equivalence at m = 0.04 is not demonstrable unless the federated arm × day interaction is near zero. The result must be labelled exploratory.

## 2. Estimands and hypotheses (aligned with C's P11 design)

**Arms and contrasts**
- Arms: C256 (reference), C64, F-IID, F-nat, S+F, S+C, and F-nat at equal total budget.
- Every arm trains on the same 2 202 client events.
- New central seeds are required. The M5 checkpoints are only a proxy for the variance components.

**Per metric k**
- Y[a,s,d] is the score of arm a, seed s, on evaluation day d.
- The score is computed on day d's events, with the probe fitted on the training-client events in that arm's embedding.
- Probe settings: StandardScaler always on (no validation-chosen scaling); kNN k = 15; logistic C = 1 in float64 with tol 1e-8, or Ridge for duration.
- Δ_k(A,C) = (1/D) Σ_d [Ȳ_A·d − Ȳ_C·d]. Days are weighted equally.

**Model**
- Y = μ_a + u_d + v_ad + w_as + e_asd.
- u_d cancels in the paired-day difference.
- Var(Δ̂) = 2σ_w²/n + σ_v²/D + 2σ_e²/(nD).

**Primary questions**
- **Q-F (federation):** for F-nat vs C256, H0: |Δ_k| ≥ m_k for some k ∈ {lobes, polarity}. Rejected by IUT, with each TOST at α = 0.05. A non-inferiority-only fallback is reported if equivalence fails.
- **Q-S (server pretraining):**
  - S+F vs F-nat: one-sided superiority (Δ > 0) on the primary metrics, Holm over 2 metrics.
  - S+F vs C256: non-inferiority at m_k.
  - **Confound:** S-arms see 434 extra server events. "Pretraining helps federation specifically" is the interaction (S+F − F-nat) − (S+C − C256). It is reported with a CI; it is secondary because it has roughly half the power.
- **Leakage gate:** Δ in day-identity skill (logistic, GroupKFold by 30-min block, over evaluation days) must have an upper 95 % bound below 0.03.
  - Leakage has one value per seed, so use Welch's t across seeds.
  - Under C's design, client identity on clients' own held-out tails needs option b3 (§4). Otherwise use day identity, which is computable.

**Decomposition contrasts** (C256→C64→F-IID→F-nat): estimated with 90 % CIs and labelled descriptive. No claims rest on them.

**Seed pairing:** shared init seeds 0–4 allow a seed-paired analysis as a sensitivity check. The unpaired two-stratum SE is the conservative primary.

## 3. Margins and their empirical justification

Sources: `varcomp.csv` and `margins.csv`, computed from `scores_day.csv` (M5 SimCLR-TF seeds 0–4 × 5 held-out days).

| Metric | Central mean | σ_seed | σ_seed×day | σ_day (cancels) | Arm×day σ (CNN / VICReg / 1k-step) | SSL gain vs own init | SD of central–central Δ (n = 5, D = 8) | Proposed m |
|---|---|---|---|---|---|---|---|---|
| P_lobes | 0.615 | 0.007 | 0.027 | 0.088 | 0.029 / 0.029 / 0.061 | 0.332 | 0.008 | **0.04** (12 % of gain) |
| P_polarity | 0.894 | 0.000 | 0.008 | 0.003 | 0.009 / 0.006 / 0.000 | 0.143 | 0.002 | **0.02** (14 % of gain) |
| P_duration | 0.841 | 0.007 | 0.006 | 0.029 | 0.014 / 0.019 / 0.016 | **0.053** | 0.005 | 0.03 (secondary) |

**Rule used to set margins**
- **Floor:** m must exceed the margin that gives 80 % TOST power at n = 5, D = 8 under the measured interaction.
- **Ceiling:** m must be at most ¼ of the SSL gain over the model's own init. Federated must keep at least ~75 % of what SSL learns.

**Polarity**
- Floor 0.012–0.015, ceiling 0.036, so 0.02 is feasible.

**Lobes**
- With zero interaction, the floor is about 0.02.
- With the 1k-step model's interaction of 0.06, the floor is about 0.07–0.08 against a ceiling of 0.083. The margin is feasible only at the edge.
- I propose 0.04 and make the evaluation-day count, not the margin, absorb this (§5).
- If only 4 days are available, the honest margin is about 0.08: still under ¼ of the gain, but coarse.

**Duration is demoted**
- The SSL gain is only 0.053. Member A measured +0.042 (kNN).
- The 1k-step model scores **higher** than the 10k model (+0.020). The metric is not monotone in training, so equivalence on it says little.
- The ¼-gain ceiling is 0.013, which is below the attainable floor of 0.025.

**Context for the size of these margins**
- Real recipe differences measured: CNN − TF lobes −0.016, VICReg − TF lobes −0.058, 1k − 10k lobes −0.034. A 0.04 lobe margin therefore separates "same recipe" from "a different SSL method".
- Member A's seed SDs on the 4 test days (lobes 0.037 kNN, 0.029 linear) are per-probe and per-seed. They are consistent with my σ_seed + σ_e/√D decomposition.

**Leakage margin**
- m_leak = 0.03. Central seed SD is 0.013.
- SimCLR-CNN sits +0.056 above TF, a known "leakier" recipe, and must count as a failure.
- Member A's null has q95 ≈ 0.

**Linear-probe stability**
- With always-on scaling, max_iter 3000 vs tol 1e-10 changes the lobe skill by ≤ 0.003 (checked on seeds 0–1).
- Member A's 0.046 discrepancies most likely come from M5's validation-chosen *unscaled* fits. Fix the scaling and use a converged solver.

## 4. Evaluation-data options

Source: `eval_options.csv`, `day_counts.csv`. Counts are SNR ≥ 20 events, 18 calendar days, 4 375 events.

| Option | Evaluation days (events) | Training per fold | Cost | Status |
|---|---|---|---|---|
| a. Reuse the 4 test days | 4 (1 412); 3 clean for S-arms | 2 202 (C's P11) | ×1 | Exploratory only. Seen ≥ 4 times; Feb 19 is in the server's run. |
| b1. 4 fresh large days | 4 (1 410) | ~2 638 before the server split | ×1 | Confirmatory, but still D = 4, and it removes client days. |
| b1′. 8 held out (4 old + 4 fresh) | 8 (2 822) | **1 226** | ×1 | Training halved; not recommended. |
| **b2. K-fold cross-fitting over the 13 large days** | 13 (4 133) | K = 3: 2 775–3 223 (before removing the server run); K = 4: 3 193–3 497 | ×K | **Recommended.** Every day is evaluated once, by models that never saw it. |
| b3. Within-day temporal holdout (last 25 % of blocks) | 18 days, only **537** events (~30 per day) | 3 838 | ×1 | Enables client-identity leakage; per-day noise is large. Secondary or leakage use only. |
| c. Synthetic events (`synth/`) | unlimited | — | ×1 | Known family, stretch and SNR labels; C2ST 0.60. Secondary only (relative). |

**Recommendation**
- Keep the server run (Feb 18 + 20) fixed.
- Run 3-fold cross-fitting over the remaining large days, about 4 evaluation days per fold. Each fold's clients are its ~10 remaining days, merged by a written rule.
- Total D ≈ 11–12 clean days, with fold as a blocking factor. Days are nested in folds; seeds are independent per fold.
- Cost: 3 folds × 7 arms × 5 seeds × 7 min ≈ 12 GPU-hours. With 10 seeds, about 25 h.
- If that is refused, use option a, declared exploratory, with the lobe margin relaxed to 0.08.

## 5. Power simulation

**Method** (`power_fast.py`, `power_fast.csv`)
- Gaussian model above, using the measured σ_seed and σ_e.
- Arm × day interaction σ_v ∈ {0, half of measured, measured max}.
- Two-stratum conservative t-TOST at α = 0.05, 4 000 simulations per cell, seed 20261010.
- "Day" here means a clean evaluation day.

**P(declare equivalence)**, lobes (m = 0.04), at true Δ = 0 (power) / Δ = −m (type I):

| σ_v | n \ D | 3 | 4 | 6 | 8 | 13 |
|---|---|---|---|---|---|---|
| 0 | 3 | .30/.02 | .47/.02 | .73/.02 | .86/.02 | .96/.03 |
| 0 | 5 | .67/.02 | .85/.02 | .97/.01 | .99/.02 | 1.0/.02 |
| 0 | 8 | .91/.02 | .99/.01 | 1.0/.02 | 1.0/.02 | 1.0/.03 |
| 0 | 10 | .95 | 1.0 | 1.0 | 1.0 | 1.0 |
| 0.030 | 5 | .17/.04 | .25/.04 | .47/.03 | .70/.03 | .93/.04 |
| 0.030 | 10 | .25/.05 | .34/.05 | .62/.04 | .82/.04 | .98/.04 |
| 0.061 | 10 | .05 | .05 | .07 | .14 | .39 |

**Polarity** (m = 0.02):
- σ_v = 0: n = 5 gives power .98–1.0 for D ≥ 3.
- Measured σ_v = 0.009: n = 5 gives .56 / .76 / .97 / 1.0 for D = 3 / 4 / 6 / 8. n = 10 gives .62 / .82 / .98 / 1.0.

**Duration** (m = 0.03): measured σ_v, n = 5 gives .39 / .57 / .83 / .94 / .98 for D = 3 / 4 / 6 / 8 / 13.

**Type I error** (true Δ = ±m): at most 0.06 in every cell. The procedure is close to nominal and slightly conservative.

**Reading**
- Going from 5 to 10 seeds helps only when the interaction is small.
- Days are the binding constraint.
- With option a (D = 3–4) and a realistic interaction, lobe equivalence has under 35 % power. **An "inconclusive" result is then the expected outcome, not evidence of a gap.**

## 6. Multiplicity and decision table

**Families and α**
- Family Q-F: α = 0.05, IUT over {lobes, polarity}.
- Family Q-S: α = 0.05, Holm over 2 metrics for S+F > F-nat.
- Leakage is a gate (IUT component) in both families.
- Secondary metrics get 90 % CIs with no claims:
  - duration;
  - per-probe scores;
  - kNN-overlap to central (within-central 0.753 on these days; A: 0.699);
  - CKA, RankMe, alignment, uniformity;
  - synthetic family and stretch probes;
  - the decomposition contrasts.
- RankMe is not a quality metric: noise raised it from 24.6 to 45.5.

**Decision table (per contrast vs C256)**

| Result on the primary metrics (both) plus the leakage gate | Conclusion |
|---|---|
| Both 90 % CIs inside ±m, and leakage upper bound < 0.03 | **Equivalent** |
| Both lower bounds > −m (at least one CI not inside ±m), leakage OK | **Non-inferior** (could be better) |
| Any upper bound < −m | **Inferior** on that metric |
| Any lower bound < −m with upper bound > −m | **Inconclusive** (state the CI; say "not shown", not "equal") |
| Primaries pass, leakage upper bound ≥ 0.03 | **Equivalent or non-inferior but leakier** (fails the privacy claim) |
| Leakage lower bound > 0 | **Leakier** (positive evidence) |

**Q-S**
- "Pretraining helps": Holm-adjusted one-sided p < 0.05 for S+F > F-nat, *and* the point estimate is at least m/2.
- "Does not help": TOST equivalence of S+F vs F-nat.
- Otherwise: inconclusive.
- "Helps federation specifically" needs the interaction CI to lie above 0.

## 7. Planted controls (`controls_prelim.csv`, `leak.csv`, `pairs_day.csv`; 5 days)

| Control | Expected | Observed (Δ [90 % CI]) |
|---|---|---|
| C1: central {0,2,4} vs {1,3} | equivalent | Polarity +0.002 [−.006, .010] ✓. Duration 0.000 [−.019, .018] ✓. **Lobes −0.011 [−.050, .027]: not shown (CI crosses −0.04).** With 3 vs 2 seeds and 5 days the procedure cannot pass lobes. This illustrates §5; it is not a failure of the method. Re-run as 4 vs 4 with seeds 5–7 (trained, not scored). |
| C2: 1k-step training (2 seeds) | fail on lobes | Lobes −0.034 [−.095, .027]: not equivalent ✓. Polarity −0.016 [−.024, −.007]: fails ✓. Duration +0.020 (better; non-monotone). |
| C3: untrained init | inferior | Lobes −0.33, polarity −0.14: **inferior** ✓. Duration −0.053 [−.078, −.029]: not equivalent. |
| C4: embedding noise τ = 0.25 / 0.5 / 1.0 SD | τ = 0.25 near the margin; larger τ fails | Lobes −0.077 / −0.13 / −0.27, all fail ✓. Polarity −0.006 (equivalent) / −0.017 / −0.051. Graded as expected. |
| C6: other recipes (descriptive) | — | CNN: lobes inconclusive, polarity equivalent. VICReg: lobes −0.058 [−.092, −.023], not equivalent. |
| L1: central leakage | not leakier | Day-identity skill −0.029 ± 0.013. |
| L2: planted per-day offset | detected | κ ≥ 0.1·‖σ‖ saturates (skill ≈ 1.0) ✓, but too strong to calibrate. The grid κ = 0.005–0.05 is coded but not run. |
| L3: SimCLR-CNN | leakier | +0.056 above TF, so it would fail m_leak = 0.03 ✓. |
| kNN-overlap to central | ordering | Within central 0.753; noise 0.5 → 0.682; 1k-step 0.594; CNN 0.578; init 0.177. Monotone ✓. The noise-planted pairs include their own source seed, which biases them upward and must be excluded. |

**Required dry-run sequence before unblinding the real arms**
1. Re-measure all controls on the new P11 central seeds.
2. C1 must pass on both primary metrics.
3. C3 and C4 at τ ≥ 0.5 must fail.
4. L2 at the smallest κ that the block-permutation null detects must be flagged.

## 8. Limitations

- **Not done:**
  - Central seeds 5–7 (10k steps) and 1k-step seeds 2–4 were trained but not scored. A wait-loop bug kept `score_matrix.py` from re-running.
  - The full `stats_design.py` run was not done: 8-seed variance components, all 35 4-vs-4 splits, hierarchical-bootstrap calibration and heavy-tail robustness.
  - The weak planted-leak grid was not run.
- Variance components come from 5 seeds × 5 days on *M5* models, not P11 models. σ_v is estimated from 3 other recipes on 5 days, so it is very imprecise. I used it as a range, not a point value.
- All 5 days are already seen, and the val day contributed to M5 selection, so every number here is exploratory.
- `scores_day.csv` used max_iter = 3000 in float32 for the linear probe; the converged-solver change was made afterwards. The measured effect is ≤ 0.003.
- The Gaussian model ignores the nesting of days in folds under option b2 (fold effects cancel within fold) and the seed pairing (which makes the analysis conservative).
- The targets are morphology-derived (circular). Valid for relative Δ only.

## 9. Files created

All in this folder:

**Scripts**
- `train_extra.py`: extra M5-recipe checkpoints.
- `score_matrix.py`: per-seed, per-day probes, label-free metrics, CKA/kNN-overlap and leakage; also builds the planted variants.
- `stats_design.py`: variance components, margins, TOST/NI procedures, controls. Its power and control sections were not run at full scale.
- `power_fast.py`: the power simulation in §5.
- `eval_options.py`: the data accounting in §4.

**Outputs**
- `scores_day.csv`, `leak.csv`, `pairs_day.csv`: from the earlier run, with 5 M5 seeds and 2 short-training seeds.
- `varcomp.csv`, `margins.csv`, `power_fast.csv`, `controls_prelim.csv`, `eval_options.csv`, `day_counts.csv`.

**Checkpoints** (in `data/k_committee/B_stats/`)
- `full/ckpt/simclr_tf_s{5,6,7}`
- `short/ckpt/simclr_tf_s{0..4}`
- `emb.npz`

Nothing was committed, and no files outside B_stats or data/k_committee were modified.
