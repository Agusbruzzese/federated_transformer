"""Phase F (central SSL) configuration.

Governing documents: research/model/PREREG_central.md (v2, sha256 d322a199b4b88aa5...) and
PREREG_central_addendum1.md (sha256 94297f2551859d2f...). Everything below that the PREREG fixes is copied
from it. Everything the PREREG leaves open is an IMPLEMENTATION CHOICE, fixed here BEFORE any encoder was
trained or evaluated. The sha256 of this file is logged at training start (research/model/F_central/logs).

Implementation choices (not specified by the PREREG), fixed a priori:
 I1  Run numbering: PREREG "Run 1..4" = run_id 0..3 = runs 0204, 0212, 0218, 0224 (files run0..run3).
 I2  Encoder input: a 12 s crop = 600 samples on an exact 50 Hz grid (linear interpolation from the event's
     fs_eff grid; ratio fs_eff/50 is <= 0.23 % from 1), per-view max|x| normalised as the LAST step.
 I3  Evaluation crop: deterministic, stretch 1, centred on the window centre (index 524.5 = C's energy
     centroid), same resampling to 50 Hz.
 I4  Random crop rule: the crop must contain C's [onset, offset] mask when it fits; otherwise the window
     centre must lie in the middle half of the crop (D §8: crops must not remove the main lobe).
 I5  Background pool: 12.8 s stretches of B's feature path (same chain as E) from TRAINING days only, with no
     detection of any SNR within 3 s, no dropout flag, and max|x| <= 6 robust sigma; stored unit-sigma.
     Added at alpha*sigma_ev, alpha ~ U(0, 1), sigma_ev = robust (MAD) sigma of the event window outside its
     [onset, offset] (+-1 s) and outside the repair mask ("SNR-matched": the view's noise floor stays within
     sqrt(2) of the event's own).
 I6  Order of view augmentations: stretch+crop -> amplitude scale U(0.8, 1.25) (before noise, so it moves the
     event-to-noise ratio) -> noise -> offset U(-0.05, 0.05)*A and end-to-end linear drift U(-0.05, 0.05)*A
     (A = crop max|x|) -> max-normalise.
 I7  Encoder embedding (64-d, before the projection head) is what is evaluated.
 I8  Schedule: cosine decay per optimiser step to 0, no warm-up; batches drop_last; final epoch weights
     (no early stopping of any kind).
 I9  kNN probes (P1, selection): features z-scored with the training fold's mean/SD, Euclidean, uniform
     weights, k = 15; sklearn GroupKFold(5) (deterministic). R^2 and skill computed on pooled out-of-fold
     predictions. Skill = (balanced accuracy - 1/K)/(1 - 1/K), K = classes present.
 I10 Selection metric: mean of [R2 log_amp, R2 log_dur, skill polarity, skill n_lobes] on the validation day
     (Feb 26, SNR>=20), 5-fold CV grouped by 30-min block, seed 0 of each grid cell. Ties -> lower lr*temp.
     The 3 final seeds are 0, 1, 2 (seed 0 = the grid run itself).
 I11 Random-init baseline seed s = the SSL model's own initial weights for seed s (saved before step 1).
 I12 P1 test: one-sided block-bootstrap p = (1 + #{Delta* <= 0}) / (1 + B), Delta = mean over the 3 seeds of
     (metric_SSL_s - metric_rand_s), OOF predictions fixed, 30-min blocks of the holdout resampled with
     replacement, B = 1000, numpy seed 0; Holm over the 4 targets; pass = Delta_hat > 0 and Holm p < 0.05.
 I13 P2 probe: HistGradientBoostingClassifier(max_iter=200, lr=0.1, random_state=0), exactly E's leakage probe,
     so the morphology baseline (a) is comparable with E's table. kNN (k=15) leakage is reported as secondary.
     Pass decided on the point estimate (mean over the 3 seeds). Null: labels permuted at the CV-group level
     (30-min block) or within day (has_dropout), as E did, 20 permutations.
 I14 Baseline (a) = the 19 m8_* scalars; (b) PCA-64 fitted on the TRAIN split's evaluation crops (max-normalised);
     (d) log10(snr).
 I15 Secondary tiers (SNR>=30, all-SNR): the SNR>=20-trained encoders are re-evaluated on those event subsets
     (not retrained; E recommends against all-SNR training without a blip filter).
"""
from __future__ import annotations

import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVENTS = os.path.join(REPO, "data", "events")
OUT = os.path.join(REPO, "data", "ssl_central")
RES = os.path.join(REPO, "research", "model", "F_central")

# ---- split (PREREG) ----
TRAIN_RUN_IDS = (0, 1, 3)          # Runs 1, 2, 4
VAL_DATE = "2025-02-26"            # Run 4, selection only
HOLDOUT_RUN_ID = 2                 # Run 3 (Feb 18-20), weakly held out
TIER = 20.0                        # E's snr column (addendum A2)
GUARD_S = 30.0                     # false-negative guard

# ---- input ----
WIN_N = 1050
CROP_N = 600                       # 12 s at 50 Hz
FS_OUT = 50.0
CENTRE = (WIN_N - 1) / 2.0

# ---- augmentations (PREREG / D §8) ----
STRETCH = (0.85, 1.18)
AMP = (0.8, 1.25)
NOISE_ALPHA = (0.0, 1.0)
OFFSET = 0.05
DRIFT = 0.05
BG_LEN = 640

# ---- training (PREREG) ----
EPOCHS = 100
BATCH = 256
WD = 1e-4
EMB = 64
PROJ = 64
GRID_LR = (1e-3, 3e-4)
GRID_T = (0.1, 0.2)
SEEDS = (0, 1, 2)

# ---- evaluation ----
K_NN = 15
N_BOOT = 1000
N_NULL = 20
TARGETS = ("log_amp", "log_dur", "polarity", "n_lobes")
TARGET_KIND = {"log_amp": "reg", "log_dur": "reg", "polarity": "cls", "n_lobes": "cls"}
