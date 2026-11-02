# C1: clustering events by magnetic signature (report)

_Orchestrator, 2026-10-02._
- Pre-registration: `PREREG.md`, final sha d6c432b5 (amendments D1–D11).
- Code `c1.py`: sha 34e36b06.
- Pins: `pins_at_run.txt`, in this order: c1.py, PREREG.md, data/events/manifest.json.

## What was done
- **Events:** all 4 375 events at SNR ≥ 20, over the whole timeline (18 event days, Feb 4–27 2025, 4 runs).
- **SSL:** SimCLR-TF trained on all of them (5 seeds, 10k steps; loss 6.0 → 0.70–0.81).
- **Representations clustered with HDBSCAN (mcs 15):**
  - **R1:** aligned waveform distance on log-compressed counts;
  - **R2:** SimCLR-TF PCA-8 plus log amplitude and log duration (5 seeds);
  - **R3:** the 19 morph8 hand features.

## Dry run
- **D9 FAILED** (`dryrun_D9_FAILED.json`): R1 found 0 clusters; R3 labelled white noise as clusters.
- **D10:** redesigned after the failure, with SNR 20–300 and phase-randomised real events as singletons. It **passed** (`dryrun.json`): R1 ARI 0.91, R3 ARI 0.93, singletons-as-noise 0.89 / 0.72, nulls 0 clusters.
- **D10 is post-hoc.** Treat it as a smoke test, not validation. The method is **not** shown to work below SNR 20.

## Pre-registered outcome (`results.json`)

| Representation | Clusters | Coverage | Largest | V1 corr (random p95 / amp-matched p95) | V2 odd/even ARI (frac joint) | V2b LORO ARI | Null (≤1 cluster in 9/10 draws) | Valid |
|---|---|---|---|---|---|---|---|---|
| R1 AWD σ₀=100 | 6 | **0.16** | 0.45 | 0.913 (0.70 / 0.64) | 0.56 (0.37) | **0.45** | **8/10** | no |
| R2 SimCLR s0 | 17 | 0.33 | 0.32 | 0.873 (0.49 / 0.57) | 0.91 (0.25) | 0.89 | 9/10 | yes |
| R2 s1 | 16 | 0.40 | 0.38 | 0.865 | 0.79 | 0.92 | 10/10 | yes |
| R2 s2 | 9 | 0.48 | **0.56** | **0.773** | **0.11** | 0.58 | 10/10 | no |
| R2 s3 | 16 | 0.37 | 0.28 | 0.857 | 0.93 | 0.78 | 10/10 | yes |
| R2 s4 | 2 | 0.84 | **0.95** | **0.648** | 0.96 | 0.51 | 10/10 | no |
| R2 overall | 3/5 seeds valid; seed ARI **0.485** (< 0.50) | | | | | | | **no** |
| R3 morph8 | 17 | 0.63 | 0.40 | 0.881 (0.69 / 0.69) | 1.00 (0.54) | 0.89 | 10/10 | **yes, recommended** |

Bold marks the failing checks.

## Caveats found after the run (orchestrator's own recomputation)
- **R3's validity is largely an artefact of discrete features.** Each R3 cluster is an almost pure combination of `m8_polarity` × `m8_first_lobe_sign` × `m8_n_lobes`, split further by amplitude. These combinations form separable islands, which is why the split-half ARI is about 1.0 (recomputed independently: 0.9998 and 1.0).
- **R3 is coarse.** Its three largest clusters hold 2 184 of 2 771 clustered events (79 %). Within-cluster amplitude CV is 0.52, against 0.33–0.41 for R2. The galleries show R3's big clusters spanning roughly 10k–70k counts peak, so they do **not** meet "very similar amplitude".
- **R2 clusters look closest to the user's definition** (gallery): tight amplitude and shape. But they are not seed-stable enough by the pre-registered bar (0.485 vs 0.50), and 2 of 5 seeds collapse.
- **R1 clusters are the tightest in shape** (corr 0.913), but it covers only 16 % of events and is not reproducible across runs.

## Independent review and audit
See `review/` and `AUDIT_20261002_105403.md` (ACCEPT WITH FIXES). The responses are below.

### Independent reviewer (`review/`; numbers recomputed, labels bit-identical, correlation matrix matches to 1.6e-6)
**R3's pass is an artefact.**
- Removing the 3 discrete columns gives 4 clusters with the largest at 92 % (orchestrator re-check: 4, 0.921). Odd/even ARI is then 0.01, so R3 would fail.
- Jittering the discrete columns makes the result collapse.
- A null that keeps the discrete columns gives 2 clusters.
- The reviewer reports ARI 0.90 between discrete-only HDBSCAN and R3. The orchestrator's own run gives 0.40 (12 islands), likely because of differences in lobe capping. The conclusion is the same, since R3 clusters are 93 % pure on the discrete combination.

**Tightness against the user's definition** (size-weighted):

| | Peak p90/p10 | IQR log₁₀ amp | Duration p90/p10 | Pairs with corr ≥ 0.9 |
|---|---|---|---|---|
| R1 | 2.8× | 0.21 | 1.72 | 0.65 |
| R2 s0 | 2.8× | 0.19 | 1.57 | 0.54 |
| R3 | 3.9× | 0.32 | 2.07 | 0.47 |
| Random | 13.3× | 0.45 | 3.64 | 0.16 |

**R2's seed instability is training collapse.** Seeds 0, 1 and 3 agree pairwise at ARI 0.86–0.92. Seed 4 found 2 clusters, and the pairs with it score only 0.07–0.20.

## Verdict
- **Under the pre-registered rules, R3 is the only valid representation.** On inspection, that validity comes from discrete features and does **not** mean "near-identical events": R3 is a coarse shape-family taxonomy of polarity × first-lobe sign × lobe count, split by amplitude.
- **No representation achieved pre-registered validity together with groups tight to the user's definition.** The best look-alike groups (R2 seeds 0, 1 and 3; R1) are 2–3× wide in peak amplitude and cover 16–40 % of events.
- **Confidence:**
  - high that R3 is a coarse taxonomy;
  - moderate that R2 s0/s1/s3 clusters are real, reproducible look-alike groups (ARI ~0.9 across days and runs, but SSL saw all data);
  - low on any finer claim.
- **Audit follow-ups not done:** a copula null, per-fold LORO, SSL retrained without the held-out run, and log-CV. These are listed for C2.

## Proposed C2 (exploratory, needs its own pre-registration)
1. Pre-state a collapse rule, e.g. largest cluster > 50 %.
2. Build a consensus over the non-collapsed R2 seeds.
3. Split each group hierarchically by aligned distance until peak p90/p10 ≤ 1.5× and median corr ≥ 0.9.
4. Use a null that keeps the discrete features, and per-fold LORO.
