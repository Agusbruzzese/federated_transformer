# Workstream C: event detection and segmentation

_Saved by the orchestrator from the agent's hand-back; the harness blocked the subagent from writing it. On 2026-09-30 the orchestrator checked these tables against the data files, and they match:_
- _`data/tab_holdout_summary.csv`: HYST‑BP F1 0.803 [0.785, 0.820]; legacy 0.006_
- _`data/tab_paired.csv`: holdout ΔF1 vs legacy +0.796 [0.775, 0.814]_
- _`data/tab_postfilter.csv` and `data/tab_tier_compare.csv`: holdout clean-tier precision 0.96 (48/50)_

_**Orchestrator corrections:**_
1. _The "hand labels" are **not human labels**. They are the AI agent's own visual judgement of plotted candidates. Read "single annotator" throughout as "single AI annotator, no human review"._
2. _The SNR ≥ 30 cut was fitted on 100 tuning labels, of which only **20** passed the cut (tuning precision 0.90 [0.70, 0.97])._
3. _The clean tier keeps only 59 % of vehicles (relative recall). It is biased toward high-SNR, near-lane, large events, which matters for how representative the training set is (see workstream D and the synthesis)._

Folder: `research/extraction/C_detection/`. Recommended detector: `detector.py`. All numbers below come from the
scripts in `scripts/`; `run_all.sh` lists the order. Hand labels (AI-annotated, see the correction above) are stored in `data/*labels*.csv`.
Split, following PROTOCOL.md: tuning uses Runs 1, 2 and 4, and **Run 3 is the holdout**. 95 % CIs are cluster
bootstraps (200 resamples, seed 0) or Wilson intervals for labelled proportions.

## 1. Summary (confidence in brackets)

1. **Recommended detector: HYST‑BP plus a two-tier output.** [high]
   - Front end: FIR low-pass + decimation to 50.33 Hz on the true ~2416 Hz sample grid, with dropouts masked.
   - Detection channel: 0.5 s median (removes a 3 s periodic pulse interferer), two-pass masked 60 s median
     baseline, 4 s moving-average high-pass, rolling robust σ.
   - Rule: hysteresis on a 0.2 s RMS envelope, on 5σ / off 2σ, merge < 1 s, valley split, drop < 0.3 s.
   - Every event is emitted with its SNR. The **clean tier (SNR ≥ 30)** is the training set for contrastive
     learning.
2. **Injection, holdout (Run 3), at matched false-alarm rate (≤ 0.2/h on noise surrogates).**
   - HYST‑BP: F1 (all SNR ≥ 6 dB) **0.803 [0.785, 0.820]**, recall@SNR8 0.99, SNR90 = 6.6, onset/offset
     MAE 0.70 / 0.77 s, median energy capture 0.96.
   - Legacy fixed windows (DC-relative, k = 5): **F1 0.006 [0.002, 0.011]**. With its threshold re-tuned: 0.08.
   - Paired ΔF1 (HYST‑BP − legacy) = +0.80 [0.78, 0.81]. The PROTOCOL condition "beat the fixed-window baseline
     on the holdout" is met. [high]
   - Workstream B's detector (BDET): F1 0.673 at the matched 6σ and 0.560 at its native 8σ. ΔF1 vs HYST‑BP
     = −0.13 [−0.15, −0.11]. [high]
3. **Surrogate-matched thresholds are not enough on real data.** [high]
   - Blind hand-check at k = 5: HYST‑BP strict precision is only 0.40 [0.23, 0.59] (holdout, n = 25), though
     its relative recall is 0.98.
   - The errors are sub-second low-SNR lobes and wander in non-Gaussian interference.
   - An SNR ≥ 30 post-filter was chosen on **tuning-run labels only** (n = 100). On fresh holdout labels
     (n = 50) it gives strict precision **0.96 [0.87, 0.99]** and "contains a vehicle" 1.00 [0.93, 1.00].
     The cost is that relative recall falls to 0.59.
   - BDET@8 sits in between: precision 0.80 [0.63, 0.91], relative recall 0.94.
4. **Plausibility (clean tier).** [high]
   - Weekday night / peak-hour rate = 0.008 [0.005, 0.011], peak at 08–09 h.
   - Weekend 07–10 h rate is 5.1/h vs 56/h on weekdays. The same pattern holds on the holdout.
   - Raw HYST‑BP at k = 5 fails the night criterion (0.24), which agrees with the hand-check.
5. **Segment representation.** A **12 s window centred on the energy centroid**. [medium-high]
   - Clean-tier segment truncation is 1.8 % and core truncation 0.3 %.
   - 12 % of windows contain another clean event (22 % any event), so the [onset, offset] mask and a
     neighbour count are kept per event. This agrees with B's 12–15 s.
6. **Dropouts, boundaries and cost.** [high]
   - Dropouts create no excess events: clean-tier rate is 9.7/h in dropout-dense Run 1 (≥ Feb 6) vs 13.7/h
     elsewhere.
   - 15 % of clean segments straddle a chunk boundary, so streams must be stitched.
   - Cost: ~2 CPU-s per hour of data (I/O 0.4, front end 1.8, detection 0.015).

## 2. Data facts used (verified here and by workstream A)

- **Values** are `<u4` holding 24-bit codes. **Time** is `<u8` device ticks, not µs; ticks run 1–5 % fast.
  `cio.py` agrees exactly with A's `bsd_io.read_raw` on 12 random chunks (`scripts/validate_io.py`,
  `data/validate_io.csv`).
- **True fs ≈ 2416 Hz**, estimated from the 50 Hz mains line (2414–2418 Hz). All timing uses the sample clock.
  Wall clock = filename start + sample offset / 2416.
- **Dropouts** are ~105-sample gaps.
  - My slot reconstruction differs from A's refined estimator by ≤ 13 slots per chunk (< 1 % of missing slots;
    1 slot = 0.4 ms).
  - A 50 Hz phase check confirmed the gaps are real loss, not a timestamp artefact.

## 3. Pre-registration and deviations

`PREREGISTRATION.md` was written after exploration only (format checks and raw-data plots) and before any
detector was run or scored. It has not been edited since. Deviations:

1. **Run 3 was used for exploration before PROTOCOL.md arrived.** `figs/zoom_busy.png` and one sanity plot
   (`figs/sanity_r3.png`) show Run 3, and they informed the pre-registration's description of event shape.
   - No parameter, threshold or design choice was scored on Run 3.
   - All tuning and development was then moved to Runs 1, 2 and 4.
2. **Front end changed after data checks.** The pre-registered Time-clock 20 ms bins were replaced by
   FIR + decimation on the sample grid (the Time clock is invalid). Detection-channel stages were added:
   - a 0.5 s median, because a 3 s periodic 0.1–0.15 s pulser was found (`figs/debug_pulses.png`)
   - a 4 s high-pass, because low-frequency wander dominated false alarms. HYST‑BP was added as a new entry
     alongside the pre-registered HYST, which is kept.
3. **Fairness variants added.** STA/LTA‑BP, CUSUM‑BP and FIXEDLOC (fixed windows on the locally normalised
   channel) were added. At the coordinator's request, BDET (B's detector) was added at its matched and native
   thresholds.
4. **CUSUM needed two corrections.** The statistic was capped at 1.5h, because the uncapped version never
   resets on this road. The event is the forward ∩ backward active region.
5. **Surrogates were changed twice, both times on tuning data only.**
   - FFT phase randomisation leaked real transients' low-frequency power.
   - They were replaced by Gaussian noise with a **median-Welch** PSD on a 300 s-smoothed baseline.
6. **Hand-check redesign.** The union-of-detectors design was abandoned, because FIXED and CUSUM segments are
   minutes long and the union merged many vehicles into one candidate. It was replaced by per-detector random
   samples, de-duplicated across detectors (IoU > 0.8) and shown blind. Four holdout spans were added to reach
   ≥ 200 candidates.
7. **Post-filter.** No detector met the pre-registered "hand precision ≥ 0.90" at its surrogate-matched
   threshold. An SNR post-filter was therefore fitted on 100 new tuning-run labels and then validated on 70 new
   holdout labels. The holdout summary precision (0.40) had been seen before the filter was designed; the
   filter's parameter was chosen from tuning labels only.
8. **F1 definition.** F1 counts extra fragments of one injected event as false positives. The split rate is
   also reported separately.

## 4. Evaluation set

| split | spans (20 min each, ±2 min context) | hours |
|---|---|---|
| tune | Run 1: Wed 5 07:45, 12:30, 17:15; Thu 6 02:30; Fri 7 08:00; **Sat 8 10:00; Sun 9 03:00, 16:00**. Run 2: Thu 13 08:00, 18:00. Run 4: Tue 25 13:00; Wed 26 04:00 | 4.0 |
| holdout (Run 3) | Tue 18 17:30, 22:00; Wed 19 07:30, 08:10, 12:30, 16:00, 17:15; Thu 20 03:00, 07:50, 12:00 | 3.3 |

- Large-sample pass (`scripts/diurnal.py`): blocks of 6 chunks every 24 chunks over all runs. That is
  1227 blocks, 85 h processed and 64.6 h counted, about ¼ of the data.
- Limitation: the only weekend (Feb 8–9) is in the tuning runs, so weekday/weekend effects can't be tested on
  the holdout.

## 5. Methods

- **Detectors** (`detectors_all.py`). Onset and offset rules:

  | Detector | Rule |
  |---|---|
  | FIXED | Legacy. DC = mean of a 24 min context (a proxy for the 120 min DC), 4 s windows, flag if max\|y−DC\| > k·MAD_global. Window edges. |
  | FIXEDLOC | Same 4 s windows on the locally normalised high-pass channel. |
  | HYST | Hysteresis on \|z\|/σ envelope (0.2 s RMS), on k / off 2σ. Crossings. |
  | HYST‑BP | As HYST on the 4 s high-passed channel. **Recommended.** |
  | STALTA / STALTA‑BP | STA 0.5 s / LTA 30 s. Onset = trigger − STA, offset = ratio < 2. |
  | CUSUM / CUSUM‑BP | Two-sided capped Page CUSUM on u² (ν = 4). Onset = last zero before alarm; offset from the time-reversed pass. |
  | WAVELET | Max over Gaussian-derivative filters (orders 1–2, scales 0.15–1.2 s), hysteresis, boundaries refined on the envelope. |
  | BDET | B's chain: 5 Hz 4th-order zero-phase LP, two-pass 20 s median, 60 s-block MAD σ, 8σ/2σ, merge 1 s, min 0.4 s. |

- **Synthetic injection** (`inject_eval.py`).
  - Templates: 61 isolated, high-SNR (> 40) real events from tuning runs, tapered to unit peak
    (`figs/templates.png`). Each is time-stretched ×1, 1.5 or 2.5 to mimic slow or distant vehicles.
  - Injected into free slots (≥ 10 s from any real detection) of real background spans, at peak SNR
    ∈ {2, 3, 4, 5, 6, 8, 10, 15, 20, 40} × local σ.
  - Volume: tuning 1536 isolated injections + 960 in pairs (separations 1–8 s); holdout 1280 + 800.
  - Match: the detection contains the injected peak (one-to-one). Ground-truth core = 5–95 % cumulative energy.
- **False-alarm (FA) null.** 6 Gaussian surrogates per span with the median-Welch PSD: 24 h (tuning) and
  20 h (holdout) per setting. Operating point = the most sensitive threshold with FA ≤ 0.2/h on the tuning
  surrogates (`data/frozen_params.json`).
- **Hand-check.** Candidate plots with ±15 s context and the segment shaded; detector identity hidden.
  - Labels: V = one vehicle, well segmented; M = merged or over-long (≥ 2 vehicles, or > 3× the transient);
    F = fragment; N = no vehicle-like transient; U = uncertain.
  - Strict precision = V/n. "Contains vehicle" = (V+M+F)/n. U is counted as not-V throughout (conservative).
  - All labelling is the agent's own visual judgement, single AI annotator, with no ground truth (see §9).

## 6. Results

### 6.1 Injection at the operating points

Holdout, Run 3 (`data/tab_holdout_summary.csv`); tuning values in `data/tab_tune_summary.csv` are within
±0.03.

| detector | thr | FA/h surrogate | recall@SNR6 | recall@SNR8 | SNR90 | F1 (all, ≥6 dB) [95 % CI] | onset / offset MAE (s) | energy capture | pairs resolved at 3 s / 5 s | CPU-s per h |
|---|---|---|---|---|---|---|---|---|---|---|
| FIXED legacy | k=5 | 0 | 0.00 | 0.00 | ∞ | 0.006 [0.002, 0.011] | 2.10 / 1.54 | 0.82 | 0.00 / 0.00 | 0.004 |
| FIXED (FA-matched) | k=3 | 0 | 0.00 | 0.00 | ∞ | 0.081 [0.058, 0.107] | 1.21 / 1.56 | 0.83 | 0.00 / 0.01 | 0.004 |
| FIXEDLOC | 5 | 0 | 0.90 | 0.99 | 6.0 | 0.645 [0.633, 0.655] | 1.29 / 1.31 | 0.86 | 0.79 / 1.00 | 0.008 |
| HYST | 10 | 0.10 | 0.03 | 0.07 | 15.9 | 0.479 [0.451, 0.506] | 0.89 / 0.74 | 0.97 | 0.64 / 0.84 | 0.006 |
| STALTA | 30 | 0 | 0.13 | 0.30 | ∞ | 0.479 [0.452, 0.504] | 0.47 / 0.56 | 0.92 | 0.16 / 0.18 | 0.010 |
| CUSUM | 8000 | 0.65 | 0.09 | 0.11 | 38.2 | 0.343 [0.317, 0.369] | 181 / 28 | 1.00 | 0 / 0 | 0.19 |
| WAVELET | 10 | 0 | 0.31 | 0.62 | 10.6 | 0.699 [0.674, 0.732] | 2.64 / 2.95 | 1.00 | 0.00 / 0.06 | 0.14 |
| STALTA‑BP | 30 | 0 | 0.33 | 0.55 | ∞ | 0.642 [0.607, 0.669] | 0.55 / 0.52 | 0.95 | 0.26 / 0.26 | 0.018 |
| CUSUM‑BP | 800 | 0 | 0.82 | 0.98 | 7.0 | 0.783 [0.763, 0.802] | 1.44 / 1.76 | 0.98 | 0.00 / 0.00 | 0.19 |
| BDET (B, FA-matched) | 6σ | 0.05 | 0.37 | 0.58 | 12.9 | 0.673 [0.643, 0.700] | 0.64 / 0.54 | 0.95 | 0.50 / 0.93 | 0.10 |
| BDET (B, native) | 8σ | 0 | 0.12 | 0.25 | 16.8 | 0.560 [0.525, 0.597] | 0.70 / 0.56 | 0.94 | 0.54 / 0.85 | 0.10 |
| **HYST‑BP** | **5σ** | **0** | **0.86** | **0.99** | **6.6** | **0.803 [0.785, 0.820]** | **0.70 / 0.77** | **0.96** | **0.79 / 1.00** | 0.014 |

- **Paired ΔF1, HYST‑BP minus each detector** (holdout, `data/tab_paired.csv`):

  | vs | ΔF1 [95 % CI] |
  |---|---|
  | CUSUM‑BP | +0.019 [0.008, 0.032] |
  | WAVELET | +0.10 [0.08, 0.13] |
  | BDET | +0.13 [0.11, 0.15] |
  | FIXEDLOC | +0.16 [0.14, 0.17] |
  | FIXED legacy | +0.80 [0.78, 0.81] |

  The tuning runs give the same signs and sizes.
- **Pre-registered decision rule.**
  - Recall@SNR8: FIXEDLOC 0.99, HYST‑BP 0.99 and CUSUM‑BP 0.98 tie (< 3 pp).
  - FIXEDLOC fails the energy-capture constraint (0.86 < 0.9) and splits 70 % of events into windows.
  - The tie-break (onset/offset error) picks HYST‑BP over CUSUM‑BP (0.70/0.77 vs 1.44/1.76 s). HYST‑BP is also
    the only one that resolves close pairs well.
- **Dropout strata** (tuning; `data/tab_tune_strata.csv`): HYST‑BP F1 is 0.786 in Run 1 ≥ Feb 6 (dense
  dropouts) vs 0.802 elsewhere; surrogate FA is 0 vs 0.14/h. The holdout has only the low-dropout stratum.
- **Recall vs SNR and threshold sweep:** `figs/inj_tune_recall.png` and `figs/inj_holdout_recall.png`.
- **Close pairs** (HYST‑BP, holdout): both vehicles resolved in 8 % of pairs at 1 s, 31 % at 2 s, 79 % at 3 s,
  100 % at 5 s and 8 s. The fragment rate for single events is 7 %. Below 2 s, pairs come out as one merged
  segment.

### 6.2 Blind hand-check (real data)

- Sample: 306 candidates, 227 of them holdout (`data/tab_hand_precision.csv`). Each detector contributes 25
  holdout samples at its operating point.

| detector (holdout, n=25) | V | M | F | N | U | strict precision [95 % CI] | contains vehicle | relative recall (n_V=114) |
|---|---|---|---|---|---|---|---|---|
| FIXED legacy | 19 | 0 | 6 | 0 | 0 | 0.76 [0.57, 0.89] | 1.00 | 0.43 [0.34, 0.52] |
| FIXED (FA-matched) | 20 | 0 | 4 | 1 | 0 | 0.80 [0.61, 0.91] | 0.96 | 0.54 |
| FIXEDLOC | 8 | 0 | 9 | 6 | 2 | 0.32 [0.17, 0.52] | 0.76 | 0.99 |
| HYST | 18 | 5 | 0 | 0 | 2 | 0.72 [0.52, 0.86] | 1.00 | 0.90 |
| STALTA | 18 | 0 | 1 | 2 | 4 | 0.72 [0.52, 0.86] | 0.92 | 0.82 |
| CUSUM | 5 | 6 | 0 | 12 | 2 | 0.20 [0.09, 0.39] | 0.52 | 0.80 |
| WAVELET | 10 | 11 | 0 | 2 | 2 | 0.40 [0.23, 0.59] | 0.92 | 0.91 |
| STALTA‑BP | 16 | 0 | 1 | 3 | 5 | 0.64 [0.45, 0.80] | 0.88 | 0.82 |
| CUSUM‑BP | 9 | 3 | 0 | 8 | 5 | 0.36 [0.20, 0.55] | 0.68 | 0.98 |
| HYST‑BP (k=5, raw) | 10 | 0 | 3 | 8 | 4 | 0.40 [0.23, 0.59] | 0.68 | 0.98 [0.94, 1.00] |

- **Two-tier validation** (`data/tab_postfilter.csv`, `data/tab_tier_compare.csv`).
  - Rule fitted on 100 tuning labels (`data/tab_postfilter_grid_tune.csv`). A width criterion added nothing;
    SNR alone discriminates. The tuning trade-off:

    | SNR cut | strict precision | V kept |
    |---|---|---|
    | 10 | 0.59 | 1.00 |
    | 15 | 0.78 | 0.78 |
    | 20 | 0.88 | 0.69 |
    | 30 | 0.90 | 0.56 |

  - Holdout results by tier:

    | tier (holdout) | strict precision | contains vehicle | relative recall | events/h |
    |---|---|---|---|---|
    | **HYST‑BP + SNR ≥ 30 (clean)** | **0.96 [0.87, 0.99]** (n=50) | 1.00 [0.93, 1.00] | 0.59 [0.50, 0.67] | 15.0 |
    | HYST‑BP detections failing the filter | 0.15 [0.05, 0.36] (n=20) | 0.30 | – | – |
    | BDET@8 (B native) | 0.80 [0.63, 0.91] (n=30) | 0.80 | 0.94 [0.88, 0.97] | 27.9 |
    | HYST‑BP raw | 0.40 | 0.68 | 0.98 | 59.7 |

- **Short detections < 0.4 s (B's dropout concern).** Only 1 of 250 sampled holdout detections was shorter
  than 0.4 s (`data/tab_hand_precision_minlen.csv`), so precision is unchanged with or without them. The
  0.5 s median and 50 Hz dropout masking remove that artefact class.

### 6.3 Plausibility: diurnal and weekday/weekend

Source: `data/tab_diurnal.csv`, `figs/diurnal.png`. Pre-registered pass criteria: night (00–05) < 20 % of the
weekday peak-hour rate, and weekend 07–10 h < weekday.

| detector | split | events/h | night/peak [95 % CI] | pass night | weekday / weekend 07–10 (per h) |
|---|---|---|---|---|---|
| **HYST‑BP + SNR≥30** | tune | 11.7 | **0.008 [0.005, 0.011]** | yes | 56.1 / 5.1 |
| **HYST‑BP + SNR≥30** | holdout | 14.2 | **0.000 [0, 0]** | yes | 49.5 / – |
| HYST‑BP raw | tune | 53.1 | 0.24 [0.17, 0.30] | no | 97.7 / 44.5 |
| BDET@8 | tune | 21.3 | 0.074 [0.054, 0.105] | yes | 64.1 / 18.8 |
| FIXED legacy | tune | 74.1 | 0.49 [0.32, 0.61] | no | 116.7 / 89.9 |

The legacy method's rate is driven by drift and dropouts: 103/h in dropout-dense Run 1 vs 45/h elsewhere.

### 6.4 Stability (±25 %, one parameter at a time; `data/tab_stability.csv`, tuning spans)

- **HYST‑BP raw.** Stable (count change < 15 %, ≥ 90 % matched) for k_off, env, merge, min_dur, split
  parameters, hp_s, baseline/σ windows, mask parameters and split on/off. Not stable for:
  - k_on: +66 % / −31 %, because the amplitude distribution is a continuum that is dense at low SNR. The same
    holds for every detector's main threshold.
  - pulse_med ×0.75: +20 %, because pulses leak through.
- **Clean tier.** Stable for every perturbation of k_on, hp_s and pulse_med, and for the SNR cut ×1.25
  (−6 %). The SNR cut ×0.75 is marginal: +12 %, 0.895 matched.

### 6.5 Boundaries, dropouts and gaps (`data/tab_boundaries_dropouts.csv`, `data/dropout_test.csv`)

- **Chunk boundaries.** 8 % of all HYST‑BP segments and 15 % of clean segments straddle a chunk boundary, so
  stitched processing is required. `detect_stream` works on concatenated chunks, which join exactly (A).
- **Dropouts.**
  - Raw linear interpolation over re-inserted slots leaves step artefacts: median 1.0–1.75σ, p99 8–10σ.
    Plain concatenation leaves p99 5–6.6σ.
  - **Masking 50 Hz samples whose FIR support touches a dropout** reduces this to median 0.06–0.09σ and
    p99 0.7–2.5σ.
  - Detection starts coincide with dropouts only at chance level.
  - 50 % (raw) and 77 % (clean) of events in Run 1 ≥ Feb 6 contain a dropout, simply because events are ~6–9 s
    long and dropouts occur every 2.4–5 s. They are flagged `has_dropout` and should be **repaired, not
    discarded**, as B also recommends.
- **Slow drift.** Handled by the two-pass masked baseline plus the high-pass. The legacy DC-relative
  normalisation is the main reason FIXED fails: drift inflates the global MAD.

### 6.6 Segment representation (`data/tab_segrepr.csv`, `figs/segment_repr.png`; clean tier, n = 783)

- Clean-tier durations: segment (down to 2σ, ramps included) median 6.2 s, p95 9.1 s, p99 10.7 s. The
  1–99 % core energy extent is p95 5.8 s.

| L (s) | segment truncated | core truncated | another clean event in window | any event in window |
|---|---|---|---|---|
| 8 | 23 % | 1.5 % | 4.5 % | 13 % |
| 10 | 5.7 % | 0.8 % | 7.5 % | 17 % |
| **12** | **1.8 %** | **0.3 %** | **11.9 %** | **22 %** |
| 16 | 0.4 % | 0.1 % | 20 % | 32 % |

- L = 12 s is the smallest length with segment truncation < 5 %, per the pre-registered rule. It matches B's
  12–15 s.
- Recommended representation: a fixed 12 s window centred on the energy centroid (604 samples at 50.33 Hz, or
  28 992 at 2416 Hz), plus the variable [onset, offset] mask, `n_neighbours_in_window` and `has_dropout`.
- Pad with zeros only at stream edges.
- Variable length alone is not recommended: segment durations vary 3× with speed and SNR, and fixed-length
  input is needed downstream anyway.

### 6.7 Cost (`data/tab_cost.csv`)

- Per hour of data: I/O 0.41 CPU-s, front end + preprocessing 1.78 s, HYST‑BP detection 0.015 s.
- The detection rules themselves cost 0.005 s (legacy FIXED) to 0.16 s (CUSUM‑BP) per hour.
- The whole 14-day dataset is ≈ 10 CPU-min.

## 7. Failure modes

- **Non-Gaussian interference.** A 3 s periodic pulser (+700–900 counts, 0.1–0.15 s, nights and weekends)
  and irregular dips produce sub-second low-SNR detections. The 0.5 s median removes the pulses; the rest is
  handled by the SNR tier.
- **Low-frequency wander.** Strong 0.01–0.1 Hz wander in some spans (e.g. Run 4, 04:00). Without the
  high-pass, it makes surrogate-matched thresholds 2× higher (HYST needs 10σ).
- **Merges.** Vehicles < 2 s apart merge into one segment; 2–3 s apart are resolved ~30–80 % of the time.
  Merged M-type segments are rare for HYST‑BP (0/25) but common for WAVELET and CUSUM.
- **Parked or stopping vehicles.** These produce baseline steps (segments up to tens of s) and are flagged by
  duration.
- **Ambiguous ~1 s monopolar lobes at SNR 5–20.** Possibly distant vehicles. These are excluded from the clean
  tier; their true nature is unknown.

## 8. Recommendation

1. Use `detector.detect_stream(x, t)` on stitched chunks with ≥ 60 s context either side. Default parameters
   are in `detector.DEFAULTS`.
2. Keep all events with their SNR. **Train on `clean` (SNR ≥ 30)**, which gives ≈ 12–14 events/h, about 3–4k
   events over 14 days.
3. If more volume is needed, relax the cut to SNR ≥ 20. On tuning labels this gives 0.88 precision and keeps
   69 % of V events; this cut has not been validated on the holdout.
4. Segment with a 12 s window centred on the energy centroid. Keep the mask and flags, and repair
   `has_dropout` events.
5. For a higher-recall alternative with moderate precision, B's detector at 8σ (precision 0.80, relative recall
   0.94) is a reasonable middle tier. On injections, its sensitivity is lower (SNR90 16.8 vs 6.6).

## 9. Confidence and limitations

- **Hand labels:** single AI annotator (the agent itself), no human review, no ground truth. U (uncertain) was
  13–30 % of low-SNR candidates, and every precision figure depends on that judgement.
- **Injection realism:** SNR was scaled from high-SNR templates, with time stretching. Real small events may
  differ in shape, so injection recall at low SNR is optimistic for real distant vehicles.
- **Weekend:** only in the tuning runs.
- **Holdout coverage:** Run 3 has low dropout density, so the holdout does not test the dense-dropout regime
  (the tuning strata do).
- **Post-filter:** chosen after the holdout summary precision of raw HYST‑BP had been seen. Its parameter came
  from tuning labels only and was validated on fresh holdout labels.
- **Legacy baseline:** the legacy FIXED is a reconstruction (24 min DC proxy, 4 s windows), not the lost code.
