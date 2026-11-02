# Pre-registration C2: groups of look-alike events (exploratory)

_Orchestrator, 2026-10-02. Written after C1 and before any C2 code or result._

## Status
C2 is **exploratory and post-hoc**: it was designed after seeing C1. The C1 seeds that collapse (2 and 4) are already known, and the collapse rule below was written knowing that.

## User's definition, made into an explicit tolerance
Events i and j are **look-alikes** when all three of these hold:
- **Shape:** aligned correlation ≥ 0.90. This is the maximum Pearson correlation over shifts |s| ≤ 1 s of the 600-sample, 50 Hz crop in counts; it is `data/c1/aligned_corr.npy` from C1.
- **Amplitude:** a peak ratio of ≤ 1.5×, measured as |Δ m8_log_amp| ≤ log10 1.5.
- **Duration:** a duration ratio of ≤ 1.5×, measured as |Δ log10 m8_dur10| ≤ log10 1.5 (durations clipped at 0.05 s).

These combine into one tolerance distance, d_ij = max((1 − corr)/0.10, |Δlog amp|/log10 1.5, |Δlog dur|/log10 1.5).

d_ij ≤ 1 means i and j are look-alikes.

- **Main tolerance:** (0.90, 1.5×, 1.5×).
- **Sensitivity:** strict (0.95, 1.25×, 1.25×) and loose (0.85, 2×, 2×).

## Grouping methods

**C2a, direct (no learning).**
- Run complete-linkage agglomerative clustering on d, cut at 1. With complete linkage, **every pair inside a group is a look-alike**, so tightness holds by construction.
- Groups with fewer than 10 members become "no look-alike group" (noise).

**C2b, SSL consensus then direct.**
- **Collapse rule:** a C1 R2 seed is excluded if its largest cluster holds more than 50 % of clustered events, or if it has fewer than 5 clusters.
- **Consensus:** co-association is the fraction of kept seeds in which i and j share a non-noise cluster. Run average linkage on 1 − co-association, cut at 0.5, keep groups of ≥ 10 members; the rest is noise.
- **Refinement:** inside each consensus group, apply C2a (complete linkage on d, cut at 1, size ≥ 10).

## Validation
- **T, tightness.** Report the size-weighted p90/p10 of peak and of duration, and the median within-group correlation. These hold by construction, so they are reported, not tested.
- **R, reproducibility: leave-one-run-out (4 folds) and odd/even days.**
  - Group A alone and B alone.
  - Transfer A's labels to B: a B event gets the label of the A group whose medoid (the minimum summed d) is nearest, but only if d to that medoid is ≤ 1; otherwise it becomes noise.
  - Report ARI on jointly grouped events (`ari_joint`), `ari_with_noise` and `frac_joint`, **per fold**.
  - **For C2b's LORO, the SSL is retrained without the held-out run.** That is seeds 0, 1 and 3 per fold, with the M5 recipe, 10k steps, on the other runs' events. HDBSCAN, the collapse rule and the consensus are applied per fold, and the held-out run's events are embedded with the fold models.
  - For C2b's odd/even split, the C1 models are used; this is flagged as optimistic.
- **N, null.** Run C2a on phase-randomised waveforms, recomputing the correlation; amplitude and duration are kept. Use 10 draws, and report the count of groups and the coverage.
- **Coverage:** report the fraction of events in a group. There is no threshold, because most passages are expected to have no look-alike.

## Decision rules (fixed now)
A method is **reliable** if all of these hold:
- the mean LORO `ari_joint` is ≥ 0.50, with no fold below 0.30;
- the mean LORO `frac_joint` is ≥ 0.30;
- the null mean group count is ≤ 10 % of the real count;
- the method has ≥ 5 groups.

**Preference:** if both methods are reliable, take the higher LORO `ari_joint`; if they are within 0.05, take the higher coverage. C2b must beat C2a for the claim "SSL helps find look-alikes".

## Dry run (must pass before real data)
- **Data:** C1's D10 synthetic set (5 real templates × 60 copies with amplitude ×U(0.8, 1.25), shift ±0.5 s, SNR 20–300; 20 overlaps; 300 phase-randomised singletons).
- **Pass criteria for C2a:** template ARI ≥ 0.70 on templates, with recall reported, and singletons-as-noise ≥ 0.80.
- **Amplitude range:** narrowed from ×U(0.5, 2) to ×U(0.8, 1.25), because C2's definition says that 2× apart are **not** look-alikes. The D10 ×0.5–2 set is also reported, where splitting is expected.
- **Null:** 400 phase-randomised real events give ≤ 1 group.
- **C2b is not dry-run,** because it needs SSL.

## Deliverables
- `c2_labels.csv`.
- Per-group profiles.
- A gallery of the largest groups, plus 12 random groups.
- `results.json`, `REPORT.md`.
- A human-readable summary for the user.

## Not claimed
- Vehicle types.
- That events without a group are noise; they are only unique within this dataset.

## v2 amendments (after AUDIT_20261002_112628, before any C2 code or result). These override the text above.
- **E1, event set.**
  - Main set: the C1 set, i.e. all 4 375 SNR ≥ 20 events from all 4 runs, including run 0204.
  - Secondary: the SNR ≥ 30 subset (3 534 events), reported.
  - Group composition by run and 6-h hour bin is reported. A group with more than 80 % of its events from one run is flagged.
- **E2, signal conventions.**
  - Correlation is **signed**: an inverted print is not a look-alike.
  - Durations range 0.36–8.1 s in this set, so no clipping occurs.
  - The shift is ±50 samples at fs_eff ≈ 50 Hz (≈ 1 s).
  - Crop: 600 samples (12 s) is the main crop, since all durations are ≤ 8.1 s. The central 402 samples (≈ 8 s) are a sensitivity check for C2a.
- **E3, reproducibility controls.**
  - Per fold, the number of B groups and the joint-set size are reported, plus a **random baseline**: B's group labels permuted among its grouped events, giving the ARI of the transferred labels against the permuted labels (100 draws, p95).
  - Reliability additionally needs, for every LORO fold, `ari_joint` > that fold's random p95 and a joint set of ≥ 50 events.
- **E4, harder nulls** (reported; the decision still uses the phase null).
  - Noise-only background windows (`bgpool`, 1 930 windows, rendered with the same crop).
  - Time-reversed real events, each paired with the real set and counted only for groups mixing originals with reversals. This is reported descriptively only.
- **E5, C2b controls.**
  - **(a) Matched-coverage random pre-partition.** Events are randomly partitioned into the same group sizes as the C2b consensus, then refined with C2a.
  - **(b) All-5-seed C2b** as a sensitivity check.
  - **The claim "SSL helps"** needs C2b to beat (a) on LORO `ari_joint` **and** on coverage.
  - **Fold fallback:** if fewer than 2 seeds survive the collapse rule in a fold, that fold's ARI is set to 0.
  - The collapse thresholds and the choice of seeds 0, 1 and 3 are post-hoc. This is stated in the REPORT, and C2 stays exploratory.
- **E6, dry run made harder.** The pass criteria above apply to the main dry run. In addition:
  - an SNR 20–40 variant is reported, with per-template recall;
  - the ×0.5–2 variant is reported.
- **E7, seeds and parameters.**
  - numpy `default_rng(2)` for C2.
  - The C2b HDBSCAN uses mcs 15, as in C1.
  - The gallery of 12 random groups uses `rng(3)`.
  - The tolerance sensitivity results are reported as group count and coverage; the verdict uses the main tolerance only.
- **Thresholds:** 0.50 / 0.30 are conventional ARI cut-offs carried over from C1. They are acknowledged as not derived from first principles.

## v3 amendment F1 (after the C2 dry run FAILED, `dryrun_v2_FAILED.json`; before any real-data C2 result, written without an ad-hoc rerun)
- **The result.** Main ARI was 0.58, with 10 groups for 5 templates; singletons-as-noise was 1.00, overlaps-as-noise 1.00, and per-template recall 0.85–1.00.
- **Cause: a specification error.** Planted amplitudes of ×U(0.8, 1.25) span 1.56×, more than the 1.5× tolerance, so complete linkage **must** split each template. A planted "look-alike class" has to lie within the tolerance to be recoverable as one group.
- **New main dry run.** Amplitude ×U(0.9, 1.11), a 1.23× span; everything else unchanged. The pass rule is unchanged: ARI ≥ 0.70, singletons ≥ 0.80, null ≤ 1.
- **Now reported for every variant: homogeneity** (sklearn `homogeneity_score` on templates), i.e. whether groups are pure. Splitting a template is acceptable under the user's definition; merging two templates is not.
- **Disclosed property.** Under a hard tolerance, a population with a continuous spread wider than the tolerance gets cut into tolerance-width bins. On real data, the group boundaries inside such a continuum are arbitrary, and LORO reproducibility is the test of whether they are.
