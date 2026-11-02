# Pre-registration: central self-supervised model (phase F), v2

_Orchestrator, 2026-09-30. v1 was audited (`AUDIT_20260930_111722.md`) before any extraction or training, and v2
applies issues 2–11. Two audit points are rejected:_
- _#1, "no model or results": this document is deliberately a pre-registration._
- _#12, the human pack is "the AI's own judgement": the pack is **unlabelled**, and labels would come from the
  user (a human)._

_The SHA-256 of this file is recorded in `../extraction/ORCHESTRATION_LOG.md` when training starts. It may not
be edited afterwards; deviations go in the phase F report as a numbered list._

## Goal
Learn event embeddings from unlabeled vehicle events on **all local data, centrally** (no federation). Measure
whether the embeddings capture vehicle-related structure rather than time, run or sensor artefacts. This is the
reference that federated training will later be compared against. Car-model recognition is **not** claimed:
there are no labels.

## Inputs (from phase E, the extraction pipeline, SYNTHESIS §3)
- 24-bit codes read correctly, on the fs_local time base, through B's feature chain (LP 8 Hz, ~50 Hz with the
  effective rate recorded, two-pass 20 s median subtracted).
- 21 s windows centred on the energy centroid, with the [onset, offset] mask.
- Dropouts: events with mask_frac ≤ 0.2 are included, repaired as the pipeline does it, and flagged
  `has_dropout`. Events above 0.2 are excluded.

## Data split
- **Train:** Runs 1, 2 and 4, except the validation day.
- **Validation (model selection only):** Feb 26 (Run 4), excluded from training.
- **Holdout:** Run 3 (Feb 18–20). Never used for training, selection or early stopping.
  - **Exposure statement:** C viewed Run 3 once during exploration (`zoom_busy.png`, `sanity_r3.png`). No
    threshold was scored on it, but the detector design saw its shapes. Holdout results are therefore labelled
    **"weakly held out"**. The leakage analysis (P2) also uses all runs, so it doesn't depend on Run 3 alone.
- **Tier:** SNR ≥ 20 (provisional). SNR ≥ 30 and all-SNR are secondary analyses.

## Models, hyperparameters and selection
- **CNN:** a 1D-CNN encoder with GroupNorm (no BatchNorm).
- **Transformer:** a patch-embedding transformer encoder, parameter count matched to the CNN within ±20 %.
- **Identical budget for both:**
  - grid lr ∈ {1e-3, 3e-4} × temperature ∈ {0.1, 0.2};
  - 100 epochs, batch 256, AdamW with weight decay 1e-4 and cosine schedule;
  - embedding dimension 64, projection head dimension 64.
- **Selection metric:** mean kNN R² over the four P1 targets on the **validation day** (not the SSL loss).
  The best grid cell per architecture is then trained with 3 seeds.
- **Baselines** (no SSL; seeds matched where random):
  - (a) morphology scalars. These are the **ceiling reference** for P1, since they contain the targets and are
    circular.
  - (b) PCA of the input window, 64 dimensions.
  - (c) **the random-init CNN and random-init transformer, 3 seeds each.** These are the null an SSL model
    must beat.
  - (d) **SNR alone:** a 1-D predictor, the reference for how much amplitude leaks through noise-floor shape.

## SSL recipe
- SimCLR / NT-Xent. The two views are augmentations of the same event.
- **False-negative guard:** within a batch, events whose centroids are < 30 s apart are not used as negatives
  for each other, because they overlap or are consecutive vehicles.
- **Augmentations:**
  - a random 12 s crop from the 21 s window;
  - additive real background noise, SNR-matched;
  - amplitude scaling U(0.8, 1.25);
  - time-stretch U(0.85, 1.18);
  - a small baseline offset or linear drift.
- **Forbidden augmentations:** sign flip, time reversal, masking of the < 4 Hz band.
- The input is per-event max-normalised. `log_amp` and SNR are never given to the encoder.

## Statistics
- **Clusters:** 30-min blocks. Run 3 has about 100; the phase F report states the number and the
  events per block.
- **CIs:** block bootstrap, 1000 resamples, seed 0.
- **Seeds:** each metric is computed per seed, the difference between the SSL model and the random-init
  baseline is paired by seed index, and the mean over 3 seeds is bootstrapped over blocks. The SD across
  seeds is reported alongside.
- **Skill** = (balanced accuracy − 1/K) / (1 − 1/K), chance-corrected.
- **Multiplicity:** Holm correction across the 4 P1 targets.

## Evaluation

### Primary

| # | Metric (on frozen embeddings) | Pass criterion |
|---|---|---|
| **P1** | **Physical probes on the holdout.** kNN (k = 15) regression or classification, 5-fold CV grouped by block. Targets: **log_amp** = log10 max\|x\| of the feature window (R²); **log_dur** = log10 of `dur10` (R²); **polarity** = sign of the sample at max\|x\| (skill); **n_lobes** = `features.morph_feats` lobe count, capped at 4 (skill). | The SSL model beats **its own random-init baseline (c)** on ≥ 3 of 4 targets, each with a Holm-adjusted p < 0.05. Also report it against (a) and (d). |
| **P2** | **Leakage across all runs.** Skill for date, run and 6-h hour-bin (CV grouped by 30-min block), and for `has_dropout` (CV grouped by day). Uses embeddings of all events: train, validation and holdout. | Date and run skill ≤ the morphology baseline (a) + 0.05; `has_dropout` skill ≤ 0.2. Tolerances are stated as conventions, not derived. The shuffled-label null is reported. |

**Verdict:**
- **Positive** only if P1 **and** P2 pass.
- "Learns structure but leaks" if P1 passes and P2 fails.
- "No evidence of learning" if P1 fails.
- **CNN vs transformer:** decided on P1 (mean R²/skill over targets), then P2.

### Secondary (reported, not part of the verdict)
- **P3 augmentation retrieval@1.** This is the SSL objective itself, so it is a sanity check only. It is also
  measured on a **held-out augmentation** that is never used in training: a ±0.5 s shift plus 0.9× stretch
  applied together.
- **P4 cross-seed kNN-graph Jaccard** (k = 10), reported next to the random-init models' value.
- **P5 cluster structure** at k ∈ {3…10} (silhouette, bootstrap ARI), reported next to the random-init models'
  embeddings. The phase-randomised null is dropped (audit issue 11).
- **Human pack:** if the user labels it, report kNN V-vs-N AUC (leave-one-out) as an external check.
- **Tiers:** repeat P1 and P2 on the SNR ≥ 30 and all-SNR tiers.
