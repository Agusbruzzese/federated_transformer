# Committee member A (model analyst): what the central SimCLR-TF model is, 2026-10-09

_Saved verbatim by the orchestrator from the agent's hand-back. Orchestrator spot-check: the `a3_summary.csv` noise floor (TF seed pairs: 15-NN overlap 0.699 [0.686, 0.715], CKA 0.851) and the `a4_variability.csv` lobe rows match the text._

All scripts and outputs are in this folder. Run them from the repo root with `.venv/bin/python research/model/K_committee/A_model/<script>`. No existing file was modified and nothing was committed. All seeds are fixed. Embeddings are cached in `cache/` (27 MB; delete it to recompute).

## 1. Summary of findings

1. **Sanity check: kNN reproduces exactly; one linear target does not.** Using M5's own `m5.emb` and `m5.probe_all`:
   - kNN: all 30 M5 rows (simclr_tf and simclr_cnn × 5 seeds × 3 targets) match `scores_abs.csv` with |diff| = 0.
   - Linear probe, polarity and duration: exact.
   - Linear probe, lobes: 5 of 10 rows differ, by up to 0.046. The worst case is TF s1 (0.716 vs 0.670), where validation picked "scaled" for me and "unscaled" for M5. Logistic regression on the 5-class lobe target is numerically fragile; the cause is undiagnosed.
   - Source: `a1_probes.py` → `a1_sanity.csv`.
2. **Probe means and seed SD (SimCLR-TF trained, 10k steps, test days, n = 1412):**

   | Target | kNN mean ± seed SD | Linear mean ± seed SD |
   |---|---|---|
   | polarity | 0.880 ± 0.005 | 0.904 ± 0.006 |
   | lobes | 0.624 ± 0.037 | 0.694 ± 0.029 |
   | log dur10 | 0.817 ± 0.011 | 0.874 ± 0.008 |

   Day-to-day SD of the 5-seed mean is much larger than the seed SD for lobes (0.101 kNN, 0.075 linear) and duration (0.041, 0.026), but not for polarity (0.007, 0.009). Feb 7 is the hard day for lobes (kNN 0.451 vs 0.63–0.67 on the other days). Source: `a4_variability.csv`.
3. **Trained minus own init, per seed (TF):**
   - kNN: lobes +0.348 ± 0.023, polarity +0.191 ± 0.063, duration +0.042 ± 0.012.
   - Linear: lobes +0.378, polarity +0.089, duration +0.064.
   - Every seed is positive on every target.
   - For contrast, the CNN's untrained network is already strong. Its trained − init gaps: kNN lobes +0.163, polarity +0.031, duration −0.050.
   - Source: `a4_variability.csv`.
4. **Label-free quality, trained vs init (TF, mean ± SD over 5 seeds):**

   | Metric | Trained | Init |
   |---|---|---|
   | RankMe (out of 64) | 25.3 ± 0.5 | 4.1 ± 1.1 |
   | Uniformity | −2.385 ± 0.067 | −0.205 ± 0.128 |
   | Alignment ratio | 0.061 ± 0.002 | 0.446 ± 0.057 |

   - Raw alignment does not separate the two (0.098 vs 0.073) because the init embeddings are nearly collapsed. Only the ratio is informative.
   - The trained/init separation is huge relative to seed SD, so these metrics are good "did it train" checks.
   - Source: `a2_labelfree.py` → `a2_labelfree.csv`.
5. **Noise floor: agreement between the 10 pairs of trained TF seeds:**
   - Linear CKA: mean 0.851 (min 0.786, max 0.888).
   - 15-NN overlap: mean 0.699 (min 0.686, max 0.715). Chance is 0.011.
   - **CKA is not discriminative here.** Untrained init pairs reach the same CKA (0.854, range 0.639–0.972) but much lower 15-NN overlap (0.400).
   - Trained vs its own init: CKA 0.140, overlap 0.147. This is identical to trained vs another seed's init (0.140 / 0.147), so the init seed leaves no trace.
   - TF vs CNN (trained): CKA 0.723, overlap 0.537.
   - Source: `a3_agreement.py` → `a3_summary.csv`, `a3_pairs.csv`.
6. **Synthetic ground truth (4000 events, generator seed 20261009, 5-fold CV), TF trained vs init vs raw PCA-64:**

   | Target | TF trained | TF init | Raw PCA-64 |
   |---|---|---|---|
   | family skill | 0.927 ± 0.011 | 0.744 | 0.770 |
   | polarity skill | 0.953 ± 0.007 | 0.861 | 0.815 |
   | log-stretch R² | 0.473 ± 0.029 | 0.355 | 0.128 |
   | log-SNR R² | 0.762 ± 0.010 | 0.667 | 0.065 |

   - Paired trained − init deltas are all positive: +0.183, +0.093, +0.118, +0.095.
   - CNN contrast: trained − init is +0.021 family, +0.029 polarity, −0.029 stretch, +0.109 SNR.
   - Source: `a5_synth.py` → `a5_synth.csv`, `a5_synth_summary.csv`, `a5_synth_deltas.csv`.
7. **Client (calendar-day) identifiability, 13 training days, 2655 events, 343 blocks.** Balanced-accuracy skill:

   | Representation | Skill |
   |---|---|
   | TF trained | 0.027 ± 0.013 (range 0.010–0.046) |
   | TF init | −0.019 ± 0.003 |
   | morph8 | −0.009 |
   | raw PCA-64 | 0.004 |
   | CNN trained | 0.059 ± 0.008 |
   | CNN init | 0.008 |

   - **The null is not centred at 0.** The block-permutation null (50 permutations) has mean −0.016 and 95th percentile ≈ 0.000 for TF s0. TF s0 at 0.033 lies above all 50 null draws (null max 0.002). morph8 (−0.009) is inside its own null (q95 −0.0015).
   - So SSL training adds a small, detectable day signal: about +0.04 above the null mean for TF and about +0.075 for CNN.
   - The signal is concentrated on run-0 days (Feb 5, 6, 8, 9). Excluding run 0 (7 days), TF trained scores −0.018 vs init −0.041 and morph8 −0.038 (no null was computed for this subset).
   - Source: `a6_identifiability.py` → `a6_identifiability.csv`, `a6_null.csv`; `a6b_perday.py` → `a6b_perday_recall.csv`, `a6b_no0204.csv`.
8. **Training length changes the representation more than the seed does.** M4 SimCLR-TF (about 1k steps, same architecture, data and views) vs M5 (10k steps), same seed:
   - Same-seed CKA 0.786 and overlap 0.485, both below the 10k cross-seed floor (0.851 / 0.699).
   - Training longer raises RankMe (14.9 → 25.3), kNN lobes (0.528 → 0.624) and linear lobes (0.585 → 0.694).
   - It lowers kNN duration (0.847 → 0.817).
   - M5 loss is still falling at 10k steps (loss at step 5000 minus step 10000 = 0.152 ± 0.006 for TF), so the model is not converged.
   - Source: `a7_steps.py` → `a7_steps.csv`, `a7_agreement.csv`, `a7_loss_curves.csv`.

## 2. Methods

- **Data.** `m4.load()`: SNR ≥ 20 events, 4375 in total. Test = Feb 7/13/19/25 (1412 events), val = Feb 10, train = 13 days (2655 events). Encoder input is `eval_view` (600 samples, max-normalised), via `m5.crops()`. The embedding is the 64-d encoder output before the projection head; only the encoder is saved. Checkpoints: `data/m5/ckpt/{simclr_tf,simclr_cnn}_s{0..4}/{final,init}.pt`, loaded with `m5.build` / `m5.ck`.
- **A1/A4.** `m5.probe_all` unchanged:
  - kNN (k = 15) or Logistic(C = 1) / RidgeCV.
  - StandardScaler on or off, chosen on the validation day; fit on train, predict test.
  - Targets from `m4.targets` (polarity, lobes min(n, 4), log10 dur10). Metric from `m4.metric` (balanced-accuracy skill, or R²).
  - Per-day scores are the same test predictions split by day.
  - "sd_seeds" = SD over the 5 seeds of the all-test score. "sd_days_of_seedmean" = SD over the 4 days of the 5-seed mean. Also reported: mean within-seed SD over days and mean within-day SD over seeds.
- **A2.** On test events:
  - RankMe = exp(entropy of s/Σs) of the uncentred embedding matrix. A centred variant is also in the CSV.
  - Alignment: two independent `augment.train_view` views with all parts (crop, stretch, amp, noise, offset), using the `m4.bgpool()` training-day noise pool exactly as M5 trains. 5 draws (generator seeds 1000–1004), identical views for every model. Embeddings L2-normalised, mean ‖z1 − z2‖².
  - Uniformity = log mean exp(−2‖zi − zj‖²) over all i < j of the L2-normalised eval-view embeddings.
  - align_ratio = alignment / mean pairwise squared distance.
- **A3.**
  - Linear CKA, centred, on raw embeddings.
  - 15-NN overlap: neighbours by Euclidean distance on per-feature z-scored embeddings, self excluded; mean over events of |shared| / 15.
  - References: raw PCA-64 (fit on training-day crops, as M5 B4) and z-scored morph8.
- **A5.**
  - `Generator.load("data/synth/model.npz").sample(4000, rng=20261009)`, max-normalised, then embedded.
  - StratifiedKFold(5, shuffle, rs = 0) on family × polarity. StandardScaler + LogisticRegression(C = 1) for family (7 classes) and polarity skill; RidgeCV(logspace(−3, 3, 13)) for log stretch, log stretch minus its family mean, and log snr_peak. Out-of-fold scores.
  - Baselines: PCA-64 fit on real training crops; raw 600-d crop.
- **A6.**
  - Training days only; y = date.
  - StandardScaler + LogisticRegression(C = 1), GroupKFold(5) over 30-min blocks (`m.blk`). Score: `m4.skill`.
  - Null: each block keeps its size but gets the day label of a randomly permuted block, 50 permutations.
  - A6b gives per-day recall minus 1/K, and repeats the analysis without run 0 (`run_id == 0` = Feb 4–11).
- **A7.**
  - Loss@N = mean loss over steps N−100..N, from `log.json` (each curve entry is a 10-step mean).
  - M4 checkpoints: `data/m4/ckpt/simclr_tf_c0_s*` (the cell M4 selected; 100 epochs × 10 steps; cosine schedule without warm-up), probed with the same `probe_all`.

## 3. Tables

**A. Probe scores, SimCLR-TF trained (`a4_variability.csv`)**

| Probe | Target | Mean | SD seeds | SD days (seed-mean) | Feb 07 | Feb 13 | Feb 19 | Feb 25 |
|---|---|---|---|---|---|---|---|---|
| kNN | polarity | 0.880 | 0.005 | 0.007 | 0.876 | 0.882 | 0.888 | 0.872 |
| kNN | lobes | 0.624 | 0.037 | 0.101 | 0.451 | 0.649 | 0.633 | 0.668 |
| kNN | duration | 0.817 | 0.011 | 0.041 | 0.774 | 0.868 | 0.798 | 0.824 |
| lin | polarity | 0.904 | 0.006 | 0.009 | 0.911 | 0.908 | 0.891 | 0.907 |
| lin | lobes | 0.694 | 0.029 | 0.075 | 0.612 | 0.625 | 0.743 | 0.752 |
| lin | duration | 0.874 | 0.008 | 0.026 | 0.838 | 0.899 | 0.871 | 0.880 |

**TF init (kNN / lin):**
- polarity 0.689 / 0.815
- lobes 0.276 / 0.316
- duration 0.775 / 0.810

**B. Agreement on test days (`a3_summary.csv`, `a7_agreement.csv`)**

| Group | n | CKA mean [min, max] | 15-NN overlap mean [min, max] |
|---|---|---|---|
| TF trained seed pairs (noise floor) | 10 | 0.851 [0.786, 0.888] | 0.699 [0.686, 0.715] |
| TF init pairs | 10 | 0.854 [0.639, 0.972] | 0.400 [0.328, 0.480] |
| CNN trained pairs | 10 | 0.880 [0.853, 0.900] | 0.678 [0.661, 0.691] |
| TF trained vs own init | 5 | 0.140 [0.112, 0.179] | 0.147 [0.137, 0.154] |
| TF trained vs other init | 20 | 0.140 | 0.147 |
| TF trained vs CNN trained | 25 | 0.723 | 0.537 |
| TF trained vs raw PCA-64 | 5 | 0.477 | 0.171 |
| TF trained vs morph8 | 5 | 0.451 | 0.305 |
| M4 1k-step TF pairs | 10 | 0.928 | 0.683 |
| M4 1k vs M5 10k, same seed | 5 | 0.786 [0.710, 0.856] | 0.485 [0.474, 0.503] |

Chance-level overlap is 0.0106.

**C. Label-free metrics (`a2_labelfree.csv`, mean ± SD over 5 seeds)**

| Rep | RankMe | Uniformity | Alignment | Align ratio |
|---|---|---|---|---|
| TF trained | 25.30 ± 0.54 | −2.385 ± 0.067 | 0.098 ± 0.003 | 0.061 ± 0.002 |
| TF init | 4.05 ± 1.09 | −0.205 ± 0.128 | 0.073 ± 0.054 | 0.446 ± 0.057 |
| CNN trained | 27.77 ± 1.85 | −1.331 ± 0.200 | 0.040 ± 0.008 | 0.052 ± 0.002 |
| CNN init | 8.98 ± 1.67 | −0.109 ± 0.022 | 0.013 ± 0.005 | 0.219 ± 0.068 |

**D. Synthetic probes (`a5_synth_summary.csv`, mean ± SD over 5 seeds)**

| Rep | Family skill | Polarity skill | log-stretch R² | log-SNR R² |
|---|---|---|---|---|
| TF trained | 0.927 ± 0.011 | 0.953 ± 0.007 | 0.473 ± 0.029 | 0.762 ± 0.010 |
| TF init | 0.744 ± 0.023 | 0.861 ± 0.022 | 0.355 ± 0.008 | 0.667 ± 0.008 |
| CNN trained | 0.953 ± 0.005 | 0.956 ± 0.003 | 0.475 ± 0.051 | 0.797 ± 0.006 |
| CNN init | 0.932 ± 0.008 | 0.927 ± 0.010 | 0.503 ± 0.010 | 0.688 ± 0.011 |
| raw PCA-64 | 0.770 | 0.815 | 0.128 | 0.065 |
| raw 600-d | 0.791 | 0.805 | 0.116 | 0.034 |

Within-family log stretch gives practically the same numbers as pooled log stretch (TF trained 0.477).

**E. Day identifiability skill (`a6_identifiability.csv`, `a6_null.csv`, `a6b_no0204.csv`)**

| Rep | 13 days | Null mean / q95 | Without run 0 (7 days) |
|---|---|---|---|
| TF trained | 0.027 ± 0.013 | −0.016 / 0.000 (s0) | −0.018 ± 0.006 |
| TF init | −0.019 ± 0.003 | −0.020 / −0.002 (s0) | −0.041 ± 0.006 |
| CNN trained | 0.059 ± 0.008 | – | 0.013 ± 0.014 |
| CNN init | 0.008 ± 0.005 | – | −0.028 ± 0.005 |
| morph8 | −0.009 | −0.019 / −0.002 | −0.038 |
| raw PCA-64 | 0.004 | – | – |

**F. Training length (`a7_steps.csv`; TF, mean ± SD)**

| Metric | M4 ~1k steps | M5 10k steps |
|---|---|---|
| RankMe | 14.9 ± 1.1 | 25.3 ± 0.5 |
| kNN polarity | 0.873 | 0.880 |
| kNN lobes | 0.528 ± 0.022 | 0.624 ± 0.037 |
| kNN duration | 0.847 | 0.817 |
| lin polarity | 0.885 | 0.904 |
| lin lobes | 0.585 ± 0.030 | 0.694 ± 0.029 |
| lin duration | 0.867 | 0.874 |

TF loss at steps 1k / 5k / 10k: 1.62 / 0.90 / 0.75.

## 4. Implications for a fair central vs federated comparison

1. **Use the seed noise floor.** A federated model is "as close to central as another central seed" if, against each central seed:
   - its 15-NN overlap lies within about [0.686, 0.715] (mean 0.699; pairwise SD 0.010);
   - its CKA lies within about [0.79, 0.89].
2. **15-NN overlap is the informative similarity metric. CKA is nearly saturated and confounded:**
   - untrained network pairs reach the same CKA (0.85) as trained pairs;
   - TF vs CNN, two different architectures, still reach 0.72.
   - CKA should not be a primary endpoint.
3. **Shared init is irrelevant to similarity.** Trained vs own init equals trained vs another init (0.140 / 0.147). Federated and central models that share an init gain no artificial similarity from it.
4. **Match training budget exactly.** At the same seed, 1k vs 10k steps moves the representation (overlap 0.485) much further than a seed change does (0.699). The 10k model is not converged. A federated run with fewer effective steps or a different schedule will look "different" for reasons unrelated to federation. Compare at matched total optimiser steps and the same schedule, or report a central learning curve. A central checkpoint every 1k steps would be needed; none exist now.
5. **Probe endpoints and margins:**
   - **Polarity is near-saturated** (seed SD 0.005–0.006, day SD 0.007–0.009). It can only detect large degradations.
   - **Lobes is the most training-sensitive target** (trained − init +0.35), but it is noisy (seed SD 0.03–0.04, day SD 0.075–0.10, Feb 7 an outlier).
   - **Duration** has moderate sensitivity (+0.04 to +0.06) and seed SD about 0.01. It moves the wrong way with longer training under kNN.
   - **Use per-day paired differences.** Day variance dominates seed variance for lobes and duration, so tests should pair by day.
   - **Prefer kNN or a converged linear solver.** The linear lobe probe is numerically unstable (finding 1).
6. **Synthetic probes are the cleanest ground truth.** Family skill (TF seed SD 0.011, trained − init +0.18) and log-stretch R² (seed SD 0.029, +0.12) are the most sensitive. Synthetic polarity (0.953) is near ceiling.
7. **Label-free metrics are coarse:**
   - **RankMe** is informative: seed SD 0.54 against a trained − init gap of 21, and it also separates 1k from 10k steps (14.9 vs 25.3).
   - **Uniformity** is useful (seed SD 0.07).
   - **Raw alignment** is uninformative without normalisation; use the alignment ratio.
8. **Leakage baseline for federation:**
   - Central TF day-skill is 0.027 (range 0.010–0.046), against a null centred at −0.016 with q95 ≈ 0. **Compare against the null, not against 0.**
   - morph8 sits inside its null. Most of the excess signal is in run-0 days.
   - A federated model should be compared to the TF 5-seed range with the same probe and null. The CNN (0.059) would be a stricter contrast.

## 5. Limitations

- **Sanity check is incomplete.** The linear-lobe probe differs from M5 in 5/10 rows, by up to 0.046. I did not diagnose the cause (possibly lbfgs convergence or sklearn and float differences).
- **Test days are not held out.** They were already used in M4/M5, so everything here is exploratory. Probe targets are morph-derived, not external labels.
- **Small samples.** Only 5 seeds and 4 test days, so the SDs are imprecise. The SD of an SD from n = 5 is about ±35 %.
- **Single augmentation setting.** Alignment uses the M5 training augmentations (one setting, 5 draws). The values depend on augmentation strength.
- **Synthetic data has limits.** The generator is fitted on training days and is only "congruent" with real data (C2ST 0.60, report 06). Family 6 is rare. Stretch is defined relative to the family template. 12 % of real events are not modelled. The log-SNR R² depends on noise level, which max-normalisation preserves.
- **A6 caveats:**
  - The null used 50 permutations, for 3 representations only. TF seeds 1–4 were judged against the s0 null.
  - Day labels are confounded with traffic volume per day (35–387 events) and with run-0 dropout artefacts.
  - The run-0-excluded analysis has no null.
- **A7 caveats.** The 1k-step comparison uses M4 checkpoints with a different schedule (no warm-up, epoch-based cosine), so length and schedule are confounded. No intermediate M5 checkpoints exist.
- **Shared CPU.** A5 and A6 ran while C_federated/noisepool.py was occupying the CPU; results are unaffected, but runtime was slower.

## 6. Files created (all in this folder)

- **Scripts:** `common.py`, `a1_probes.py`, `a2_labelfree.py`, `a3_agreement.py`, `a5_synth.py`, `a6_identifiability.py`, `a6b_perday.py`, `a7_steps.py`
- **Outputs:** `a1_probe_scores.csv`, `a1_sanity.csv`, `a4_variability.csv`, `a2_labelfree.csv`, `a3_pairs.csv`, `a3_summary.csv`, `a5_synth.csv`, `a5_synth_summary.csv`, `a5_synth_deltas.csv`, `a6_identifiability.csv`, `a6_null.csv`, `a6_stdout.txt`, `a6b_perday_recall.csv`, `a6b_no0204.csv`, `a6b_stdout.txt`, `a7_loss_curves.csv`, `a7_steps.csv`, `a7_agreement.csv`, `a7_stdout.txt`
- **Cache:** `cache/*.npy` (embeddings, 27 MB)
