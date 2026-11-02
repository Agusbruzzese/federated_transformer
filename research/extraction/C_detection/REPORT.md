# Workstream C: event detection and segmentation

_Revised 2026-09-30 after external audit `AUDIT_20260930_021556.md` (REJECT) and the orchestrator's required fixes 1–7. Saved by the orchestrator from the agent's hand-back; revision 1 is kept as `REPORT_rev1.md`._

_**Orchestrator verification** (2026-09-30), against the files:_
- _`data/tab_ranking_templates.csv`: the ranking holds for all template sets_
- _`data/tab_segrepr_rule.csv`: L = 21 s for the clean tier_
- _`data/tab_bias_summary.csv`: about 4.1k clean vs 5.5k SNR ≥ 20 events over 14 days_
- _`figs/human_pack/`: 80 IDs, titles show only the ID, y-axis values hidden; the labels file is empty. `sheet_00.png` was inspected visually._

_**Orchestrator corrections (kept):**_
1. _The "hand labels" are **not human labels**. They are the AI agent's own visual judgement of plotted candidates. Read "annotator" throughout as "single AI annotator, no human review"._
2. _The SNR ≥ 30 cut was fitted on 100 tuning labels, of which only **20** passed the cut (tuning precision 0.90 [0.70, 0.97])._
3. _The clean tier keeps only 59 % of labelled vehicles (relative recall, itself circular; see §6.2). It is biased toward high-SNR, large, morning events (quantified in §6.8)._

Folder: `research/extraction/C_detection/`. Detector: `detector.py`. Every number comes from scripts in `scripts/`; `run_all.sh` lists the order. Split (PROTOCOL.md): tuning uses Runs 1, 2 and 4, and **Run 3 is the holdout**.

CIs used:
- cluster bootstrap (span × realisation, 200 resamples, seed 0) for injection metrics
- Wilson intervals for labelled proportions
- **exact Garwood Poisson intervals for count-based rates**, including all zero-count cells

## 1. Summary (confidence in brackets; see §3 for what is exploratory)

1. **Detector: HYST‑BP.** [medium]
   - Front end: FIR low-pass + decimation to 50.33 Hz on the true ~2416 Hz sample grid, with dropouts masked.
   - Detection channel: 0.5 s median, two-pass masked 60 s median baseline, 4 s moving-average high-pass, rolling robust σ.
   - Rule: hysteresis on a 0.2 s RMS envelope, 5σ on / 2σ off, merge < 1 s, valley split, drop < 0.3 s.
   - Every event is emitted with its SNR.
   - The design was developed on the tuning runs; several stages were added after looking at data (§3).
2. **Headline, holdout (Run 3): HYST‑BP vs the fair fixed-window method FIXEDLOC** (4 s windows on the same locally normalised channel).
   - F1 0.803 [0.785, 0.820] vs 0.645 [0.633, 0.655]. **ΔF1 = +0.157 [0.144, 0.173]**. Tuning: +0.145 [0.133, 0.155]. [medium-high]
   - Injection F1 covers all injections at SNR ≥ 6 dB, at thresholds matched to ≤ 0.2 false alarms/h on noise surrogates.
   - The gain comes from segmentation, not detection. FIXEDLOC reaches the same recall@SNR8 (0.99). It splits 70 % of events across windows and captures 0.86 of their energy, vs 7 % and 0.96 for HYST‑BP.
   - This comparison satisfies PROTOCOL's "beat the fixed-window baseline on the holdout".
3. **The legacy DC-relative fixed windows are illustrative only.** [low as evidence about the lost code]
   - This is a reconstruction (24 min mean as a proxy for the 120 min DC, global MAD, k = 5), not the lost code.
   - Its near-zero F1 (0.006) is driven by that global DC/MAD choice: drift inflates the MAD, so no injected event crosses threshold.
4. **The ranking survives templates not selected by HYST.** [medium]
   - Template sets selected by workstream B's detector (104) or by WAVELET (98), and their subsets never selected by HYST (46 and 42), all give:
     HYST‑BP ≳ CUSUM‑BP > WAVELET ≈ BDET > FIXEDLOC ≈ STA/LTA‑BP > legacy.
   - HYST‑BP beats WAVELET, BDET and FIXEDLOC by +0.10 to +0.16 in every set (CIs exclude 0).
   - **HYST‑BP vs CUSUM‑BP is a tie on the holdout with independent templates** (ΔF1 +0.009 to +0.014, CIs include 0).
   - All templates are high-SNR real events scaled down, so low-SNR recall remains optimistic for real distant vehicles.
5. **Real-data precision needs an SNR tier.** [low–medium: AI-only labels, SNR visible on the sheets, cut fitted on 20 passing items]
   - Raw HYST‑BP detections at k = 5: AI-labelled strict precision 0.40 [0.23, 0.59] (holdout, n = 25).
   - Clean tier (SNR ≥ 30): 0.96 [0.87, 0.99] on 50 holdout items.
   - A blind human-labelling pack (80 items, SNR hidden) is prepared in `figs/human_pack/`. It is not yet labelled, and these figures should not be relied on until it is.
6. **The clean tier is a biased subset.** [high, descriptive]
   - 48 % of clean events fall at 07–09 h, vs 20 % of all detections. 20–23 h is 3 % vs 13 %.
   - Amplitude (KS 0.69) and duration (KS 0.59) differ strongly from all detections; polarity does not (p = 0.66).
   - SNR ≥ 20 adds +35 % events (16.3 vs 12.1/h, about 5.5k vs 4.1k over 14 days) at lower AI-labelled precision (0.88 on tuning labels).
7. **Segment length: the pre-registered rule, as written, gives L = 21 s**, not 12 s. [medium]
   - The rule is the 95th percentile of the 1–99 % energy extent including ramps, clean tier. It is a lower bound, because the extent is capped by ±10 s of context.
   - Truncation with ramps (clean tier): 42 % at 12 s, 33 % at 16 s, 22 % at 20 s.
   - At 21 s, about 26 % of windows contain another clean event and about 40 % contain any event.
   - `detector.py` now defaults to 21 s. 12 s (core-only: 1.8 % segment truncation) is a documented alternative that departs from the rule.
8. **Plausibility.**
   - Weekday night/peak rate ratio: clean tier 0.0075 [0.003, 0.016] (tuning; exact CI); raw HYST‑BP 0.24 [0.20, 0.28], which fails.
   - Holdout clean tier: 0 night events in 1.88 h, so night rate < 1.96/h and ratio < 0.030.
   - The criterion is partly a consequence of the SNR cut, so it is weak evidence.
   - The weekend is untestable on the holdout. [low–medium]
9. **Dropouts, boundaries, cost.** [high]
   - Dropouts are masked at 50 Hz; detection starts coincide with them only at chance level.
   - Clean-tier rate is 9.7/h in dropout-dense Run 1 (≥ Feb 6) vs 13.7/h elsewhere, so there is no inflation.
   - 15 % of clean segments straddle chunk boundaries, so streams must be stitched.
   - Cost: ~2 CPU-s per hour of data.

## 2. Data facts used (verified here and by workstream A)

- **Values** are `<u4` holding 24-bit codes. **Time** is `<u8` device ticks, not µs; ticks run 1–5 % fast.
- `cio.py` Values/Time arrays are identical to A's `bsd_io.read_raw` on 12 random chunks (`data/validate_io.csv`). The dropout-slot reconstruction differs from A's refined estimator by 1–13 slots in 5 of 12 chunks (< 1 %, 1 slot = 0.4 ms).
- **fs ≈ 2416 Hz**, estimated from the mains line. All timing uses the sample clock. Wall clock = filename start + sample offset / 2416.
- **Dropouts** are ~105-sample real losses.

## 3. Exploratory vs confirmatory

`PREREGISTRATION.md` was written before any detector was run, but after exploratory viewing that included Run 3 (item E1). The following table lists every post-hoc change; anything built on these is **exploratory**.

| # | change / decision | when (relative to results) | status |
|---|---|---|---|
| E1 | Run 3 viewed in exploration (`zoom.py`, `figs/zoom_busy.png`, `figs/sanity_r3.png`) | before PROTOCOL.md; informed the pre-registration's shape description | exploratory contact with the holdout |
| E2 | Front end: Time-clock 20 ms bins → FIR + decimation on the sample grid; dropout masking | after data checks, before scoring | exploratory (design) |
| E3 | 0.5 s median in the detection channel (3 s periodic pulser) | after viewing tuning data | exploratory (design) |
| E4 | 4 s high-pass (HYST‑BP) added as a new detector | after the first tuning scores (surrogate FA plateau) | exploratory: **the recommended detector was not pre-registered** |
| E5 | CUSUM cap and forward∩backward rule | after the first tuning runs | exploratory (bug fixes) |
| E6 | Surrogate redesign twice (phase randomisation → median-Welch Gaussian, smoothed baseline) | after the first tuning scores | exploratory |
| E7 | Fairness variants STALTA‑BP, CUSUM‑BP, FIXEDLOC; BDET (B's detector) | after the first tuning scores / on request | exploratory |
| E8 | Operating-point rule (most sensitive threshold with surrogate FA ≤ 0.2/h) | pre-registered | confirmatory in form; audit notes surrogates miss non-Gaussian interference |
| E9 | Tie-break onset/offset error (HYST‑BP over CUSUM‑BP, FIXEDLOC) | pre-registered | confirmatory in form; applied to a detector added post hoc (E4) |
| E10 | Hand-check redesigned (union → per-detector samples) | after the first sheet | exploratory |
| E11 | Four extra holdout spans added (count < 200) | after the holdout candidate count was seen (no labels yet) | exploratory holdout change |
| E12 | SNR ≥ 30 post-filter: grid 8 SNR × 7 widths on 100 tuning labels, 20 passing | **after the holdout raw precision (0.40) was seen**; rule chosen by the author | exploratory; its holdout check (n = 50) is AI-labelled and not SNR-blind |
| E13 | Segment representation: 12 s from segment extent (first version) | deviated from the pre-registered rule | withdrawn; the rule as written gives 21 s (§6.6) |
| E14 | Independent template sets (B, WAVELET) | after the audit | robustness check of E4's ranking |
| C1 | Injection results at frozen settings on the holdout (Run 3), all template sets | after freezing (`data/frozen_params.json`) | **confirmatory** for the detector comparison, given E1/E11 |
| C2 | Diurnal and weekday/weekend criterion | pre-registered | confirmatory in form; weak because of the SNR cut (§6.3) |

Confirmation on data not yet touched (unused Run 3 days, or other Run 2 days) with human labels is still outstanding.

## 4. Evaluation set

| split | spans (20 min each, ±2 min context) | hours |
|---|---|---|
| tune | Run 1: Wed 5 07:45, 12:30, 17:15; Thu 6 02:30; Fri 7 08:00; Sat 8 10:00; Sun 9 03:00, 16:00. Run 2: Thu 13 08:00, 18:00. Run 4: Tue 25 13:00; Wed 26 04:00 | 4.0 |
| holdout (Run 3) | Tue 18 17:30, 22:00; Wed 19 07:30, 08:10, 12:30, 16:00, 17:15; Thu 20 03:00, 07:50, 12:00 | 3.3 |

- Large-sample pass: 6 chunks every 24 over all runs, giving 1227 blocks, 64.6 h counted, about ¼ of 338 h.
- Not tested on the holdout: the weekend (only Feb 8–9, tuning) and the dense-dropout regime (Run 1 ≥ Feb 6, tuning).

## 5. Methods (unchanged; see scripts)

- **Detectors** (`detectors_all.py`):

  | Detector | Rule |
  |---|---|
  | FIXED (legacy reconstruction) | DC-relative, global MAD. |
  | FIXEDLOC | 4 s windows on the locally normalised high-pass channel. |
  | HYST, HYST‑BP | Hysteresis on the envelope; HYST‑BP on the high-passed channel. |
  | STALTA, STALTA‑BP | STA/LTA energy ratio. |
  | CUSUM, CUSUM‑BP | Capped two-sided CUSUM. |
  | WAVELET | Gaussian-derivative bank. |
  | BDET | B's chain re-implemented on C's 50 Hz stream: 5 Hz LP, two-pass 20 s median, 60 s MAD σ, 8σ/2σ. Not B's original pipeline (audit issue 9). |

- **Injection:**
  - Template sets: 61 HYST-selected, 104 BDET@8-selected, 98 WAVELET-selected. All come from tuning runs, are isolated, have SNR > 40 and are tapered to unit peak.
  - Each template is time-stretched ×1, 1.5 or 2.5.
  - Injected at peak SNR {2 … 40} × local σ into free slots of real spans. Tuning: 1536 isolated + 960 pair injections per setting; holdout: 1280 + 800.
  - F1 counts extra fragments as false positives; the split rate is also reported.
- **False-alarm null:** Gaussian surrogates with the median-Welch PSD, 24 h (tuning) and 20 h (holdout) per setting.
- **AI labels:** V/M/F/N/U codes (§6.2). Sheets showed SNR and duration in their titles, so those labels are not SNR-blind.

## 6. Results

### 6.1 Injection at frozen operating points (holdout, HYST-selected templates; `data/tab_holdout_summary.csv`)

| detector | thr | FA/h surrogate [exact 95 % CI] | recall@SNR8 | SNR90 | F1 [95 % CI] | onset/offset MAE (s) | energy capture | split rate | pairs resolved at 3 s / 5 s |
|---|---|---|---|---|---|---|---|---|---|
| **HYST‑BP** | 5σ | 0 [0, 0.185] | 0.99 | 6.6 | **0.803 [0.785, 0.820]** | 0.70 / 0.77 | 0.96 | 0.07 | 0.79 / 1.00 |
| CUSUM‑BP | 800 | 0 [0, 0.185] | 0.98 | 7.0 | 0.783 [0.763, 0.802] | 1.44 / 1.76 | 0.98 | 0.03 | 0.00 / 0.00 |
| WAVELET | 10 | 0 [0, 0.185] | 0.62 | 10.6 | 0.699 [0.674, 0.732] | 2.64 / 2.95 | 1.00 | 0.02 | 0.00 / 0.06 |
| BDET (B, re-impl.) | 6σ | 0.05 [0.001, 0.28] | 0.58 | 12.9 | 0.673 [0.643, 0.700] | 0.64 / 0.54 | 0.95 | 0.03 | 0.50 / 0.93 |
| **FIXEDLOC (fair fixed windows)** | 5 | 0 [0, 0.185] | 0.99 | 6.0 | **0.645 [0.633, 0.655]** | 1.29 / 1.31 | 0.86 | 0.70 | 0.79 / 1.00 |
| STALTA‑BP | 30 | 0 [0, 0.185] | 0.55 | ∞ | 0.642 [0.607, 0.669] | 0.55 / 0.52 | 0.95 | 0.01 | 0.26 / 0.26 |
| BDET (B native) | 8σ | 0 [0, 0.185] | 0.25 | 16.8 | 0.560 [0.525, 0.597] | 0.70 / 0.56 | 0.94 | 0.01 | 0.54 / 0.85 |
| HYST | 10 | 0.10 [0.012, 0.36] | 0.07 | 15.9 | 0.479 | 0.89 / 0.74 | 0.97 | 0.05 | 0.64 / 0.84 |
| STALTA | 30 | 0 [0, 0.185] | 0.30 | ∞ | 0.479 | 0.47 / 0.56 | 0.92 | 0.00 | 0.16 / 0.18 |
| CUSUM | 8000 | 0.65 [0.35, 1.11] (fails FA) | 0.11 | 38 | 0.343 | 181 / 28 | 1.00 | 0.01 | 0 / 0 |
| FIXED legacy (reconstruction, illustrative) | k=5 | 0 [0, 0.185] | 0.00 | ∞ | 0.006 [0.002, 0.011] | 2.10 / 1.54 | 0.82 | 0.25 | 0 / 0 |

- Tuning values are within ±0.03 (`data/tab_tune_summary.csv`).
- "0 FA/h" only means < 0.15–0.19/h: `data/tab_poisson_fa.csv`.
- The surrogates are Gaussian and miss the non-Gaussian interference seen in the AI hand-check (§6.2), so the matched-FA design does not control real false alarms (audit issue 8).

### 6.1b Ranking robustness to template selection (`data/tab_ranking_templates.csv`)

| split | templates (n) | F1: HYST‑BP / CUSUM‑BP / WAVELET / BDET / FIXEDLOC | ΔF1 HYST‑BP − CUSUM‑BP | ΔF1 HYST‑BP − FIXEDLOC |
|---|---|---|---|---|
| tune | HYST (61) | 0.795 / 0.770 / 0.670 / 0.664 / 0.650 | +0.025 [0.014, 0.039] | +0.145 [0.135, 0.157] |
| tune | BDET (104) | 0.787 / 0.767 / 0.657 / 0.662 / 0.647 | +0.020 [0.008, 0.032] | +0.140 [0.127, 0.151] |
| tune | BDET not in HYST set (46) | 0.800 / 0.781 / 0.658 / 0.663 / 0.659 | +0.018 [0.002, 0.034] | +0.140 [0.126, 0.151] |
| tune | WAVELET (98) | 0.794 / 0.768 / 0.668 / 0.654 / 0.646 | +0.026 [0.015, 0.038] | +0.148 [0.138, 0.160] |
| tune | WAVELET not in HYST set (42) | 0.805 / 0.784 / 0.696 / 0.658 / 0.657 | +0.021 [0.007, 0.036] | +0.148 [0.135, 0.160] |
| holdout | HYST (61) | 0.803 / 0.783 / 0.699 / 0.673 / 0.645 | +0.019 [0.008, 0.030] | +0.157 [0.144, 0.173] |
| holdout | BDET (104) | 0.793 / 0.783 / 0.685 / 0.669 / 0.638 | +0.009 [−0.004, 0.025] | +0.154 [0.139, 0.167] |
| holdout | BDET not in HYST set (46) | 0.789 / 0.776 / 0.684 / 0.670 / 0.635 | +0.012 [−0.007, 0.032] | +0.154 [0.138, 0.171] |
| holdout | WAVELET (98) | 0.808 / 0.794 / 0.699 / 0.667 / 0.648 | +0.014 [0.001, 0.026] | +0.160 [0.147, 0.174] |
| holdout | WAVELET not in HYST set (42) | 0.809 / 0.796 / 0.695 / 0.670 / 0.651 | +0.013 [−0.002, 0.027] | +0.158 [0.142, 0.173] |

- Overlap between sets: 58 of 104 BDET templates and 56 of 98 WAVELET templates are also HYST templates (peak within 1 s).
- The ranking HYST‑BP ≳ CUSUM‑BP > WAVELET ≈ BDET > FIXEDLOC holds in all 10 cases.
  - WAVELET vs BDET swaps in some sets.
  - FIXEDLOC vs STALTA‑BP swaps on the holdout BDET/WAVELET sets.
- HYST‑BP is preferred over CUSUM‑BP on segmentation quality: onset/offset error, pair resolution 0.79 vs 0.00 at 3 s, and cost.

### 6.2 AI hand-check of real detections (not human; not SNR-blind)

Holdout, 25 per detector (`data/tab_hand_precision.csv`). Strict precision = V/n, with U counted as not-V.
- HYST‑BP raw: 0.40 [0.23, 0.59], relative recall 0.98.
- FIXEDLOC: 0.32.
- CUSUM‑BP: 0.36.
- BDET@8: 0.80 [0.63, 0.91], relative recall 0.94 (n = 30, `data/tab_tier_compare.csv`).
- Legacy: 0.76, relative recall 0.43.
- Clean tier: 0.96 [0.87, 0.99] (n = 50). Detections failing the cut: 0.15 [0.05, 0.36] (n = 20).

Relative recall is computed against the V-labelled pool drawn from these same detectors, so it is circular: it ranks detectors against each other but does not estimate true recall (audit issue 5).

**Blind human pack prepared, not labelled** (`figs/human_pack/`, 4 sheets, IDs H000–H079):
- 40 clean-tier, 20 HYST‑BP failing the cut, and 20 BDET@8 holdout candidates (fresh draw, seed 7), shuffled.
- Titles show only an ID; y-axis values and σ lines are hidden.
- Key: `data/human_pack_key.csv`. Empty labels file: `data/human_pack_labels.csv`. Codes are in `figs/human_pack/README.md`.
- Residual leak: the noisiness of the auto-scaled background still hints at SNR.

### 6.3 Plausibility with exact CIs (`data/tab_poisson_night.csv`; weekdays only)

| split | tier | night events / hours | night rate (/h) [exact CI] | night/peak ratio [exact CI] | < 0.2? |
|---|---|---|---|---|---|
| tune | clean SNR≥30 | 7 / 9.47 | 0.74 [0.30, 1.52] | 0.0075 [0.003, 0.016] | yes |
| tune | SNR≥20 | 51 / 9.47 | 5.4 [4.0, 7.1] | 0.052 [0.037, 0.070] | yes |
| tune | HYST‑BP raw | 314 / 9.47 | 33.2 [29.6, 37.1] | 0.24 [0.20, 0.28] | no |
| tune | BDET@8 | 71 / 9.47 | 7.5 [5.9, 9.5] | 0.074 [0.055, 0.097] | yes |
| tune | FIXEDLOC | 423 / 9.47 | 44.7 [40.5, 49.2] | 0.17 [0.15, 0.20] | borderline |
| tune | FIXED legacy | 714 / 9.47 | 75.4 [70.0, 81.2] | 0.49 [0.43, 0.57] | no |
| holdout | clean SNR≥30 | **0 / 1.88** | 0 [0, 1.96] | 0 [0, 0.030] | yes |
| holdout | SNR≥20 | 6 / 1.88 | 3.2 [1.2, 6.9] | 0.044 [0.015, 0.109] | yes |
| holdout | HYST‑BP raw | 47 / 1.88 | 25.0 [18.3, 33.2] | 0.26 [0.17, 0.40] | no |
| holdout | BDET@8 | 13 / 1.88 | 6.9 [3.7, 11.8] | 0.096 [0.046, 0.190] | yes |

- The criterion is partly mechanical: any high-SNR cut suppresses night detections of the low-amplitude interference.
- Weekend: clean tier 07–10 h rate is 5.1/h vs 56/h on weekdays (tuning only). Untestable on the holdout.

### 6.4 Stability (`data/tab_stability.csv`, tuning)

- **HYST‑BP raw** passes (< 15 % count change, ≥ 90 % matched) for all secondary parameters. It **fails** for:
  - k_on ±25 %: +66 % / −31 %
  - pulse_med ×0.75: +20 %
  - Every detector's main threshold also fails this test.
- **Clean tier** passes for k_on, hp_s, pulse_med and the SNR cut ×1.25. It **fails** the pre-registered criterion for the SNR cut ×0.75 (+12 % count, 0.895 matched < 0.90).

### 6.5 Boundaries, dropouts, cost

- 15 % of clean segments straddle a chunk boundary, so streams must be stitched.
- Dropout handling (`data/dropout_test.csv`): masking at 50 Hz removes step artefacts (p99 0.7–2.5σ, vs 8–10σ for raw interpolation).
- Clean-tier rate: 9.7/h in dense-dropout Run 1 vs 13.7/h elsewhere. 77 % of clean events there carry `has_dropout` and should be repaired, not dropped.
- Cost per hour of data: ~2 CPU-s (I/O 0.41, front end 1.78, detection 0.015).
- The front end's edge behaviour (median/uniform filters at stream ends) has not been tested; always pass ≥ 60 s of context (audit issue 12).

### 6.6 Segment length: pre-registered rule applied as written (`data/tab_segrepr_rule.csv`, `data/tab_segrepr.csv`)

- **The rule:** smallest L covering the 95th percentile of the 1–99 % energy extent **including ramps**.
- Extent is computed on ±10 s context, clipped halfway to neighbours.

| tier | extent p50 | extent p95 | **L from the rule** |
|---|---|---|---|
| clean SNR≥30 (n=783) | 5.8 s | 20.8 s | **21 s** |
| SNR≥20 (n=1056) | 9.1 s | 21.6 s | 22 s |
| all HYST‑BP (n=3425) | 15.2 s | 21.7 s | 22 s |

- **The rule gives L > 12 s.**
- Both caveats (the ±10 s cap and noise inflation) are one-line limitations and don't change the rule's outcome:
  - The extent is capped by the ±10 s context (segment + 20 s), so these values are lower bounds.
  - For low-SNR events the extent is inflated by noise energy.

| L | ramp-inclusive truncation, clean / all | segment (2σ) truncation, clean | another clean event in window, clean | any event in window, clean |
|---|---|---|---|---|
| 12 s | **42 % / 79 %** | 1.8 % | 12 % | 22 % |
| 16 s | **33 % / 73 %** | 0.4 % | 20 % | 32 % |
| 20 s | **22 % / 50 %** | 0.0 % | 25 % | 39 % |
| 22 s | 11 % / 13 % | 0.0 % | 27 % | 42 % |
| 24 s | 5 % / 4 % | 0.0 % | 29 % | 45 % |

- **Recommendation following the rule:** a 21 s window centred on the energy centroid (`detector.DEFAULTS['window_s'] = 21`), with the [onset, offset] mask and `n_neighbours_in_window` kept.
- 12 s (core-only) is an alternative that departs from the pre-registered rule. Workstream D should choose knowing that a 21 s window has about 26 % contamination by another clean event.
- Contamination counts only events inside the same sampled block, so it is underestimated near block edges (audit issue 11).

### 6.7 Close pairs and merges

HYST‑BP resolves both vehicles in:

| separation | resolved |
|---|---|
| 1 s | 8 % |
| 2 s | 31 % |
| 3 s | 79 % |
| 5–8 s | 100 % |

Pairs closer than 2 s come out as one merged segment.

### 6.8 Clean-tier selection bias (`data/tab_bias_summary.csv`, `data/tab_bias_categorical.csv`, `figs/clean_tier_bias.png`)

| | all HYST‑BP (SNR≥5) | SNR≥20 | clean SNR≥30 |
|---|---|---|---|
| events/h (sampled 64.6 h) | 53.0 | 16.3 | 12.1 |
| est. events, full 14 days (338 h) | ~17.9k | ~5.5k | ~4.1k |
| \|peak\| p10 / p50 / p90 (counts) | 453 / 924 / 20.1k | 1.2k / 10.6k / 46.0k | 2.5k / 17.1k / 53.7k |
| local σ p50 (counts) | 75 | 81 | 91 |
| duration p10 / p50 / p90 (s) | 0.5 / 1.5 / 6.5 | 1.2 / 5.7 / 8.0 | 3.3 / 6.2 / 8.3 |
| share 07–09 h / 00–05 h / 20–23 h | 20 % / 15 % / 14 % | 39 % / 7 % / 6 % | 48 % / 2 % / 3 % |
| share 16–19 h | 17 % | 10 % | 8 % |
| positive dominant polarity | 36 % | 34 % (χ² p = 0.10) | 35 % (p = 0.66) |
| Run 1 / 2 / 3 / 4 | 48 / 17 / 15 / 20 % | 42 / 17 / 17 / 24 % | 41 / 17 / 18 / 24 % |
| KS vs all (amplitude / duration / σ) | – | 0.54 / 0.47 / 0.09 | 0.69 / 0.59 / 0.17 |

- SNR ≥ 20 adds 273 events in the sample, +35 % over SNR ≥ 30 (about +1.4k over 14 days).
- Its AI-labelled tuning precision is 0.88 (not validated on the holdout).
- Interpretation: the clean tier over-represents the morning peak and large, long signatures. It under-represents evening and night events. The σ comparison shows this is not because clean events come from quieter periods.
- The "all" column is itself about 60 % non-vehicle (AI hand-check), so its distributions are a contaminated reference.
- The clean tier is **not representative** of all vehicles passing the sensor. Workstream D should treat SNR tier and hour as confounds.

## 7. Failure modes

- A 3 s periodic pulser and irregular dips produce sub-second low-SNR detections.
- Low-frequency wander.
- Merges below 2 s.
- Parked vehicles produce baseline steps.
- Ambiguous ~1 s monopolar lobes at SNR 5–20 are of unknown nature.

## 8. Recommendation (conditional on human labels)

1. Use `detector.detect_stream(x, t)` on stitched chunks with ≥ 60 s of context. Keep every event with its SNR, `has_dropout` and `n_neighbours_in_window`.
2. For training, start from the SNR ≥ 30 clean tier, but record its bias (§6.8). Consider SNR ≥ 20 (+35 %) if volume or coverage of the day matters more than purity.
3. Final tier thresholds should come from the human-labelled pack.
4. Window: 21 s (pre-registered rule), centred on the energy centroid, with mask. 12 s is a documented, non-pre-registered alternative.
5. CUSUM‑BP is statistically tied with HYST‑BP on injection F1 but segments much worse. FIXEDLOC reaches the same recall but fragments events.

## 9. Limitations and outstanding items

- **Labels are AI-only and were not SNR-blind.** The human pack is unlabelled. All precision figures are low–medium confidence.
- **Injection templates are high-SNR real events scaled down,** so low-SNR recall is optimistic for real distant vehicles. True recall is unknown; a human census of one full span is still needed.
- **Holdout coverage and confirmation.** The holdout is Run 3 only and has no weekend or dense-dropout regime. It was touched in exploration (E1) and extended after a count (E11). Fresh-span confirmation is outstanding.
- **Minor items:**
  - BDET is a re-implementation, not B's original chain.
  - F1 penalises fragmentation; split rates are reported separately.
  - The stability of `k_on` fails the criterion.
  - The legacy FIXED is a reconstruction.
  - Hand-label CSVs are AI inputs and can't be regenerated.
  - `hand_candidates.csv` was regenerated once after a design change. The labelled version equals `hand_candidates.labelled_backup.csv`, and the current file is identical to it (the list of detectors used was pinned before BDET was added).

## Files
`research/extraction/C_detection/`
- `detector.py` (window default is now 21 s)
- `scripts/`: new files are `ranking.py`, `poisson_ci.py`, `bias.py`, `human_pack.py`; `templates.py`, `inject_eval.py` and `segment_repr.py` were modified
- `data/`: new tables `tab_ranking_templates.csv`, `tab_poisson_*.csv`, `tab_bias_*.csv`, `tab_segrepr_rule.csv`, `human_pack_key.csv`, `human_pack_labels.csv`
- `figs/human_pack/` (sheets + README)
- `figs/clean_tier_bias.png`
