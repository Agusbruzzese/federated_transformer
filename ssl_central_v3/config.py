"""Phase G (central SSL, leave-one-run-out, tightened evaluation) configuration.

Governing document: research/model/PREREG_central_v3.md (sha256 e6424ecd1c0a8abe...). Everything v3 fixes is
copied from it. Phase F's implementation choices I1-I15 (ssl_central/config.py docstring) are REUSED unless v3
overrides them. The choices below (J*) fill what v3 leaves open; they were written BEFORE any phase G model was
trained or evaluated, and this file's sha256 is logged at freeze (research/model/G_loro/logs/hashes_at_freeze.txt).

Reused from F unchanged: I1 (run numbering), I2/I3 (12 s 50 Hz input, deterministic central evaluation crop),
I4 (crop rule), I5 noise scaling (alpha*sigma_ev), I6 (augmentation order), I7 (64-d encoder output is evaluated),
I8 (cosine per step, drop_last, final weights), I9 (kNN k=15, z-scored with training-fold stats, uniform, GroupKFold(5)
deterministic, metrics on pooled OOF predictions, skill with K = classes present), I11 (random-init = own init),
I12 (one-sided block-bootstrap p, B=1000, numpy seed 0), I14 (morph8 = the 19 m8_* scalars; SNR = log10 snr),
I15 (secondary tiers re-evaluated, not retrained). Overridden by v3: I10 (selection), I13 (P2 probe), the split.

 J1  Folds: fold k tests run_id k (0..3 = 0204, 0212, 0218, 0224); SSL training uses ALL snr>=20 events of the
     other three runs (all their days, including Feb 26). No validation day.
 J2  Background pool: NOT rebuilt. F's pool (training days of 0204/0212/0224, Feb 26 excluded) is FILTERED by its
     `run` column, dropping every stretch from the test run. Fold sizes: 0204 -> 1630, 0212 -> 1080,
     0218 -> 1930 (pool never had 0218), 0224 -> 1150. Asserted in data.fold_pool().
 J3  Initialisation: torch.manual_seed(seed) immediately before building the encoder (as F), so the seed-s init is
     identical across folds (and to F's seed-s init). Random-init baseline for seed s = that init. Final seed 0 is
     the grid run itself (as I10).
 J4  Selection (v3 §Selection): seed-0 model of each grid cell embeds the fold's TRAINING-run events (snr>=20,
     evaluation crop); kNN probe with leave-one-DAY-out CV over the training days; amp_resid fitted inside each
     LODO training fold (J5); metrics on pooled OOF predictions; score = mean of [R2 amp_resid, R2 log_dur,
     skill polarity, skill n_lobes]. Best score wins; ties -> lower lr*T. Test-run event_ids are asserted absent.
 J5  amp_resid: inside every probe training fold, OLS m8_log_amp ~ a + b*log10(snr) is fitted on the training-fold
     events only; the target for BOTH training- and test-fold events is m8_log_amp - (a + b*log10 snr) with that
     fit. R2 is computed on the pooled OOF (true residual, predicted residual) pairs; the bootstrap reuses those
     fixed pairs. No test-fold amplitude or SNR enters the fit.
 J6  P1 pooled OOF matrix: row i = embedding of event i by the fold model that tests i's run (same seed index).
     kNN/GroupKFold(5) by 30-min block over all 4 375 events; z-scoring with each CV training fold's stats.
 J7  Per-fold Delta (P1 "per-fold Delta > 0 in >= 3 of 4 runs"): the metric recomputed on the pooled-probe OOF
     predictions restricted to run k's events, Delta = mean over the 5 seeds of (SSL_s - rand_s).
 J8  P1 test: Delta = mean_s(SSL_s - rand_s) on the pooled metric; one-sided block-bootstrap p (I12) over the
     30-min blocks of all events; Holm over the 4 targets per architecture. Target passes iff Delta > 0 and
     p_holm < 0.05 and per-fold Delta > 0 in >= 3 of 4 runs. P1 passes iff >= 3 of 4 targets pass.
 J9  P2 probe: StandardScaler + LogisticRegression(penalty L2, C=1, lbfgs, max_iter=2000), fitted per CV fold;
     GroupKFold(5) by 30-min block. Skill = chance-corrected balanced accuracy, K = classes present.
 J10 P2 null: 50 permutations (rng seed 1000+p). date / run / is-0204 are constant within a block -> block-level
     label permutation (F's permute_labels). has_dropout varies within a block -> labels permuted WITHIN each
     30-min block (respects the CV group; keeps block prevalence, so the null absorbs prevalence tracking).
 J11 P2 decision on point estimates of the 5-seed means: P2a date skill <= 0.03; P2b run skill (runs 0212/0218/0224,
     3 classes) <= 0.03; P2c is-0204 skill <= mean rand is-0204 skill + 0.05 AND <= morph8 is-0204 skill;
     P2d has_dropout skill within run 0204 <= mean rand skill + 0.05. Bootstrap (1000, blocks) CIs reported; a
     descriptive one-sided "exceeds threshold" bootstrap p, Holm over the 4 criteria, is reported but does not
     decide (v3: "tested on the 5-seed mean against its threshold").
 J12 Baseline (b) PCA-64: per fold, fitted on the fold's training-run snr>=20 evaluation crops, applied to the test
     run; assembled OOF like J6.
 J13 CNN vs transformer: d = mean_s(P1mean_cnn_s - P1mean_tf_s), seeds paired by index, P1mean = mean over the 4
     pooled targets; 95% block-bootstrap percentile CI. "tie" unless the CI excludes 0.
 J14 Secondary tiers: snr>=30 and all-SNR subsets of the OOF embeddings (models trained on snr>=20, I15); P1 kNN and
     P2 LR with 10 permutations; descriptive only; log_dur flagged degenerate at all-SNR (addendum A1).
 J15 Ablation: per fold, the selected CNN cell trained with time-stretch disabled (crop jitter, amplitude, noise,
     offset/drift kept; stretch fixed at 1), seeds 0-4. Compared with the stretched CNN by seed pairing on every
     P1 target (focus log_dur), bootstrap CI; it shares the stretched CNN's random-init (same seed).
 J16 P3 retrieval@1 and P4 cross-seed 10-NN Jaccard computed per fold on the test run's snr>=20 events with that
     fold's models (F's recipes, fold pool), then averaged over folds.
 J17 HGB leakage (secondary): F's HGB probe (I13), 10 permutations, only if time allows; scope reported.
 D1  Diagnostic fixed BEFORE results (descriptive, not a criterion): pooling embeddings from 4 different fold
     networks may itself make runs/dates identifiable. Control "foldinit": untrained encoders with fold-specific
     init seed 100 + 10*k + s, assembled exactly like J6, run through P1/P2. Also "within-space": each fold's seed-s
     SSL model embeds ALL events (incl. its own training runs) and P2a/P2b are computed in that single space.
"""
from __future__ import annotations

import os

from ssl_central import config as F

REPO = F.REPO
EVENTS = F.EVENTS
F_OUT = F.OUT
OUT = os.path.join(REPO, "data", "ssl_central_v3")
RES = os.path.join(REPO, "research", "model", "G_loro")
PREREG = os.path.join(REPO, "research", "model", "PREREG_central_v3.md")
PREREG_SHA_PREFIX = "e6424ecd1c0a8abe"

RUNS = (0, 1, 2, 3)
RUN_NAME = {0: "0204", 1: "0212", 2: "0218", 3: "0224"}
TIER = 20.0

EPOCHS = F.EPOCHS
BATCH = F.BATCH
GRID_LR = (1e-3, 3e-4)
GRID_T = (0.1, 0.2)
SEEDS = (0, 1, 2, 3, 4)
ARCHS = ("cnn", "tf")

K_NN = 15
N_BOOT = 1000
N_PERM = 50
N_PERM_SEC = 10
LR_C = 1.0
TARGETS = ("amp_resid", "log_dur", "polarity", "n_lobes")
KIND = {"amp_resid": "reg", "log_dur": "reg", "polarity": "cls", "n_lobes": "cls"}
P2A_THR = 0.03
P2B_THR = 0.03
P2C_MARGIN = 0.05
P2D_MARGIN = 0.05
