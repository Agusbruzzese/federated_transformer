# Pre-registration C1: clustering events by magnetic signature (similar shape, amplitude and duration)

_Orchestrator, 2026-10-02. Written before any C1 code or result._

## User's definition of similarity
Two events are similar when the magnetic print left by whatever passed the sensor is alike: **almost identical shape, very similar amplitude, similar duration.** No vehicle labels are needed or used.

## Data
- **Events:** all SNR ≥ 20 events from **all days** (whole timeline, 4 375 events). SSL training also uses all of them; there is no held-out test set. Lower-SNR detections are excluded because they are mostly non-vehicle or noise (C, E).
- **Signal used for similarity:** the central 12 s, 50 Hz, baseline-subtracted crop **in counts**, without max-normalisation, so amplitude is kept.

## Reference similarity (the definition made measurable)
**Aligned waveform distance (AWD).**
- d(x, y) = minimum over time shifts |s| ≤ 1 s of the Euclidean distance between x and y shifted by s.
- It is computed on log-compressed amplitude: x̃ = sign(x)·log1p(|x|/σ₀), with σ₀ = 100 counts (≈ noise σ). Without this, the 5k–60k counts of the largest events would swamp everything else.
- A small AWD therefore means the same shape, amplitude and duration at once.

## Representations clustered (all compared)
- **R1 AWD:** the precomputed distance matrix (the definition itself).
- **R2 SimCLR-TF + scale:** the SimCLR transformer embedding (64-d, 5 seeds, 10 k steps, trained on all events, M5 recipe), PCA to 8 dims and z-scored, plus z-scored log10 peak amplitude and log10 duration (`m8_dur10`). These are the 10 dims used for clustering. The 2 scale dims are added because the encoder sees max-normalised input.
- **R3 morph8:** the 19 hand-made features, z-scored.

## Clustering
- **Main method:** HDBSCAN (sklearn), `min_cluster_size` = 15 (sensitivity 10 and 25), default `min_samples`, metric precomputed (R1) or Euclidean (R2, R3).
- Points labelled noise stay unclustered; this means "no close look-alikes".

## Validation without labels (all reported; decisions below)
- **V1, tightness in physical units.** Per cluster:
  - the coefficient of variation of peak amplitude and of duration;
  - the median pairwise **aligned correlation** of the waveforms.

  These are reported as size-weighted means and compared with a **random grouping of the same cluster sizes** (100 draws).
- **V2, split-half reproducibility.**
  - Split the events into odd and even calendar days and cluster each half independently.
  - Transfer labels from half A to half B: each B event takes the label of its nearest A event in the same representation, and becomes noise if that neighbour is noise.
  - Compute the ARI against B's own clustering on events clustered in both, and the same B→A; average them.
- **V3, seed stability (R2 only).** The mean pairwise ARI over the 5 SSL seeds.
- **V4, null.** Same pipeline on each representation with every column independently permuted (R2, R3), or on events shuffled within blocks of time shifts (R1: AWD of phase-randomised waveforms). It must find ≤ 1 cluster or far fewer than the real data.
- **V5, coverage.** The fraction of events assigned to a cluster (not noise).

## Decision rules (fixed now)
A representation's clustering is **"valid"** if all of:
- **V1:** median aligned correlation within clusters ≥ 0.80, and amplitude CV and duration CV are both lower than the random-grouping 5th percentile;
- **V2:** split-half ARI ≥ 0.50;
- **V4:** the null gives fewer than half as many clusters as the real data;
- **R2 only:** V3 seed ARI ≥ 0.50.

**Recommended representation:** among the valid ones, the highest V2 reproducibility. Ties are broken by V1 correlation. If none is valid, the report says so and gives the best partial result.

## Deliverables
- A cluster label for every SNR ≥ 20 event.
- A per-cluster profile: size, median amplitude and duration, polarity share, hour-of-day distribution.
- A gallery figure per representation showing, for each of the largest clusters, its members overlaid.
- Every quoted number saved in a file.

## Dry run (before real data is clustered)
- **Planted clusters:** 5 synthetic templates × 60 noisy copies (amplitude ±10 %, shift ±0.3 s), plus 300 noise events. R1 and R3-style clustering must recover the 5 groups with ARI ≥ 0.80 and mark the noise as noise ≥ 70 %.
- **Null:** pure noise events. They must give ≤ 1 cluster.

## Not claimed
- That clusters correspond to vehicle types.
- That the cluster boundaries are natural if V2 fails.

## v2 amendments (after audit AUDIT_20261002_102430, before any C1 code). These override the text above.
- **D1, circularity.**
  - R1's V1 correlation is non-independent of AWD. R2 and R3's amplitude and duration CVs are non-independent, because those variables are clustering inputs (R2) or included (R3).
  - Each is reported with that label. The decision relies on V2, V4 and coverage, plus V1 **relative to random grouping** (D2).
- **D2, V1 correlation.**
  - The signal is the raw-count 600-sample crop. The measure is the maximum over shifts |s| ≤ 50 samples (1 s) of the Pearson correlation.
  - Valid requires the size-weighted cluster median to be ≥ 0.80 **and** above the 95th percentile of the same statistic under random grouping with the same sizes (100 draws).
- **D3, reproducibility.**
  - V2 (odd vs even days) is kept. For R2 it is flagged optimistic, since SSL saw both halves.
  - Added **V2b, leave-one-run-out:** cluster 3 runs, transfer labels to the 4th, average the ARI over 4 folds. Reported for all representations.
  - ARI is reported both on jointly clustered events and with noise as a class, together with the excluded fraction.
- **D4, confounds** (descriptive). Cramér's V between cluster label and run, 6-h hour bin, and SNR tertile.
- **D5, validity also requires:**
  - coverage V5 ≥ 30 % of events;
  - the largest cluster ≤ 50 % of clustered events;
  - **V4 null ≤ 1 cluster.**
- **D6, nulls.** R1: phase-randomised waveforms (spectrum and amplitude kept). R2 and R3: a **multivariate Gaussian with the real covariance** (the harder null, replacing column permutation).
- **D7, sensitivity** (reported, not decisive): `min_cluster_size` 10 and 25, and σ₀ 50 and 200 for R1. HDBSCAN is deterministic. R2 uses PCA-8 (the explained variance is reported).
- **D8, crop.** E's 21 s window → the 600-sample, 50 Hz crop centred on the energy centroid (the `eval_view` geometry), **not** max-normalised, in counts. Dropouts are already repaired by E; overlapping events are not separated (a limitation).
- **D9, dry run** (harder):
  - 5 templates × 60 copies with amplitude ×U(0.5, 2), shift ±0.5 s and SNR 3–30, plus 20 overlap events (two templates summed) and 300 noise events;
  - run through R1 (AWD) and through a vector representation (R3-style features from the synthetic waveforms);
  - recover the templates with ARI ≥ 0.70, and assign ≥ 60 % of the noise to noise;
  - pure noise must give ≤ 1 cluster.

## v3 amendment D10 (2026-10-02, after the D9 dry run FAILED, BEFORE any real-data clustering)
- **The D9 failure is kept on record** in `dryrun_D9_FAILED.json`.
  - R1 found 0 clusters.
  - R3 found the templates (ARI 0.71) but put the noise into clusters (noise-as-noise 0.02).
- **Diagnosis.**
  - Planted copies at SNR 3–30 had a median within-template AWD of 90, close to the between-template AWD of 112.
  - Gaussian "noise events" were mutually closest (AWD 30), so they formed the densest region.
- **Why D9 was unrealistic for this dataset.**
  - The real set contains only SNR ≥ 20 events by construction, so copies at SNR 3–20 do not occur.
  - The real non-repeating events are not white noise. They are real magnetic events without look-alikes.
- **D10 dry run, which replaces D9 for the go/no-go decision:**
  - templates are 5 real events with peak > 15k counts;
  - 60 copies each, with amplitude ×U(0.5, 2) and shift ±0.5 s (both kept from D9);
  - **SNR U(20, 300)** (matches the real tier);
  - 20 overlap events;
  - **300 "singleton" events:** real events, phase-randomised. Each has a real spectrum and amplitude but a unique shape.
- **Thresholds are unchanged:** ARI ≥ 0.70, singletons-as-noise ≥ 60 %, for both R1 and R3. The null gives ≤ 1 cluster for:
  - (a) the original pure-Gaussian null;
  - (b) a new null of 400 phase-randomised real events.
- **Disclosed limitation.** The pipeline is **not** shown to work below SNR 20, or when noise-like events dominate. The D9 result shows it fails there.
- **Pre-check disclosed.** An ad-hoc run of this design gave R1 ARI 0.92 / singletons 0.85, and R3 0.93 / 0.74. This amendment was written after seeing that.

## v4 amendment D11 (2026-10-02, after audit AUDIT_20261002_103059, BEFORE any real-data clustering). Stricter, never looser.
- **V2 is decisive in both forms.** Valid requires `ari_joint` ≥ 0.50 for **both** V2 (odd/even day) and V2b (leave-one-run-out).
  - `ari_with_noise` and `frac_joint` are reported next to every value.
  - The audit's month-boundary concern does not apply: all 18 event days are in Feb 2025.
- **R2 validity is assessed per seed.** The full validation (V1, V2, V2b, null, confounds) runs for all 5 seeds. R2 is valid only if **≥ 3 of 5 seeds** are valid **and** V3 seed ARI ≥ 0.50.
- **Nulls use 10 draws per representation** (each R2 seed gets its own Gaussian null). The null check passes if **≥ 9 of 10 draws give ≤ 1 cluster**. The maximum is reported.
- **Amplitude-matched control (reported, not decisive).** It answers whether the clusters are more than amplitude bins.
  - For each cluster, an equal-size group is drawn from **outside** the cluster, matched to each member's amplitude by 20-quantile bin (100 draws).
  - The within-group aligned correlation is reported against that null's 95th percentile.
- **Disclosures, not changes:**
  - the D10 dry run is a sanity check, not a blind test (audit #2);
  - the dry-run R3 uses `morph_feats` on 50 Hz crops, not the stored `m8_*` columns, and R2 is not dry-run;
  - dry-run template recall and the fate of the overlap events are now reported (not gating, because they were added after the pass);
  - AWD treats energy shifted out of the window as mismatch (zero-padded shift); this is the operational definition;
  - V1 correlation is circular for R1, and the CVs are circular for R2/R3 (D1).
