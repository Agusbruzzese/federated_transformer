# Workstream B: Signal characterisation of the TMR vehicle sensor (revision 2)

_Saved by the orchestrator from the agent's hand-back; revision 1 is kept as `REPORT_rev1.md`. On 2026-09-30 the orchestrator spot-checked this revision against the saved outputs, and they match:_
- _the injection correlation tables in `data/out_synth.txt`_
- _the per-run Mann-Whitney p-values and the repaired-event statistics in `data/out_v2_analysis.txt`_
- _`data/v2_events.parquet`, which holds 38 102 rows at thresholds 5, 6 and 8σ_

_**Orchestrator note:** the 8 Hz minimax claim (S12) holds for amplitudes of at least 1500 counts. At 700 counts and 1× speed, 2 Hz beats 8 Hz by 0.06 in correlation (0.890 vs 0.831), because noise dominates._


_**Orchestrator addendum after audit AUDIT_20260930_015100** (the workstream is closed under the stopping rule in ORCHESTRATION_LOG):_
- _**S12 (8 Hz feature cut-off) is downgraded to medium.** The 4× templates are synthetic speed-ups, and the real events in `cutoff2.py` were selected by a 5 Hz detector. The choice between 5, 8 and 10 Hz is therefore handed to workstream D as a hyperparameter, to be decided on D's label-free criteria._
- _**Amplitude classes.** "Large" (>5k) is an amplitude-selected subset, not a demonstrated population (S6–S8). The hourly rates are relative to this detector and threshold._
- _**Blips.** Read "logger blips" in S10 as "short detections correlated with dropout load". The mechanism is unproven._
- _**Run-0204 null result (§3.4).** It has low power (n = 40 vs 290); no effect sizes are reported._
- _**Duration uncertainty.** Durations carry about ±0.3 % from per-chunk fs variation, not ±0.1 %._
- _**Post-hoc criteria.** Criteria were set after seeing the data; nothing was pre-registered._
- _**Divide vs subtract (S11).** Untested downstream; D may check it._
- _**Rejected audit points.**_
  - _Issue 6's "missing out files": `out_baseline.txt` and `out_v2_analysis.txt` exist in `data/`. The auditor omitted them for size._
  - _Issue 7's objection to filling 1–4-sample gaps: A recommends exactly that._

## 0. Scope

**Sources.** Scripts are in `scripts/`, figures in `figures/`, and derived data in `data/`.
- Every quoted number comes from a saved output in `data/out_*.txt`.
- The `*.npz` and `*.pkl` files (about 250 MB) are git-ignored and can be regenerated. Run `OMP_NUM_THREADS=1 python chunk_stats.py 8 5` (about 5 min), then `python run_pipeline.py 8` (about 5 min).

**Clock.**
- Time uses the sample slot index at **fs = 2416 Hz**. Ticks are used only to find lost samples.
- The absolute rate is known to about ±0.1 %; all durations and rates inherit that uncertainty.
- The circularity concern about deriving fs from the mains line (audit issue 2) is settled independently by A (REPORT §5.4): mains and wall clock agree within −0.1 ± 1.2 Hz across 428 chunks.
- Nothing in this report uses 2400 Hz or µs ticks.

**Two processing versions.**
- **v1** was the first characterisation, on a 100 Hz, 20 Hz-LP series. Its outputs are figs 1–9 and `out_psd*`, `out_baseline`, `out_events`.
- **v2** is the recommended chain run end to end (§4, `preproc.py` and `run_pipeline.py`). It produces figs 10–13 and `out_v2_analysis`, `out_cutoff2`, `out_synth*`.
- **Wherever the two disagree, v2 is authoritative.**

**Obsolete scripts.** `quicklook.py` (tick-derived 2381 Hz) and `cutoff_test.py` (compared against the 20 Hz series) are kept for provenance only.

**Orchestrator caveat (resolved).** In v1, events cut by the 41 s chunk edges biased extent low. §3.4 now uses only events at least 5 s from any span edge or dropout mask, taken mainly from the contiguous windows.

## 1. Summary (confidence in brackets)

| # | Finding | Conf. |
|---|---|---|
| S1 | **fs ≈ 2416 Hz (not 2400 Hz), and Time ticks are not µs.** The mains-implied rate is 2416.3 ± 2.8 Hz (p1–p99: 2407.6–2421.8) over 3679 chunks, confirmed against the wall clock by A (§5.4). Tick-derived rates range 2290–2405 Hz. | high |
| S2 | **The baseline wanders strongly.** It sits at 11.47M–11.85M counts and moves 46k–157k p-p within a day (310k during the run-0212 settling hour). The large moves are smooth ramps of up to about 4k counts/min, not steps. The diurnal component is weak, 34k p-p. | high |
| S3 | **Noise floor.** After mains removal it is 31 counts/√Hz at 1–5 Hz, sloping f^−0.33 to 10 Hz and f^−0.40 from 10 to 45 Hz. It reads 17 at 20 Hz, 11.6 at 100 Hz, and 0.8 at 1 kHz (ADC filter roll-off above about 300 Hz). Below 0.5 Hz it rises like 1/f. The noise σ of the chain at quiet times is 98 counts after LP 5 Hz and 104 after LP 8 Hz (p10–p90: 68–160). Per span the detection σ has median 106 (p10–p90: 73–177), about 40 % higher by day. | high |
| S4 | **Mains hum.** The 50 Hz line has median amplitude 1433 counts (range 686–1886) and a diurnal cycle from about 970 to about 1650. Harmonics: 150 Hz ≈ 633, 250 Hz ≈ 345, 350 Hz ≈ 116, even harmonics 52–65. The hum comes in bursts. Overlapping-block regression leaves about 34 counts. | high |
| S5 | **Other tones, all fixed in frequency across runs and hours.** 9.45 Hz family: 9.45/18.87/28.3/37.7 Hz at ≈40/26/2/0.2 counts. 0.732 Hz family (1.465, 2.197 Hz): 21–25 counts each. A 1.978 Hz line: ≈23 counts. Also 604 Hz (fs/4, 49 counts) and 1208 Hz (fs/2). Origin unknown. | high (existence), low (origin) |
| S6 | **Large events (>5k counts; contiguous windows, n=194) are slow.** Median peak 20.2k (p5–p95: 8.5k–56k). Extent above 2σ 5.1 s (2.4–11.7 s); 90 %-energy duration 1.55 s (0.83–3.7 s). Rise time 0.29 s (0.14–1.25 s); FWHM 0.54 s (0.28–1.0 s). 99 % of the excess energy lies below 1.52 Hz (median event) and below 2.4 Hz (p95 event). **These bandwidth statements hold only for the large class.** | high |
| S7 | **Large events are mostly bipolar.** Patterns: +− 70 %, −+− 10 %, −+ 7 %, +−+− 6 %, unipolar 5 %. Small and medium events are mostly unipolar: − 75 %, + 17 %. Pattern labels depend on the sign convention and on the 20 % lobe threshold. Direction of travel is untested. | medium |
| S8 | **Amplitude distribution.** At 5σ, 6σ and 8σ triggers alike, an upper mode near 10^4.3 counts is separated by a minimum at 4.0k–6.3k. The lower population's peak moves with the trigger (10^2.9 at 5σ, 10^3.1 at 8σ). It may be a truncated continuum reaching down to the noise, so bimodality of the whole distribution cannot be claimed. Large events occur almost exclusively between 07 and 18 h. | medium |
| S9 | **Rush-hour overlap is not negligible.** 7.1 % of large events have another detection within 5 s (a lower bound, n=283). 17.7 % have an extent over 8 s, i.e. possible merged pairs or queueing. | medium |
| S10 | **Logger artefacts.** (a) Dropouts (A). (b) Short "blips": 93 % positive, 96 % in run 0204, 126/h in run 0204 vs 5–6/h elsewhere. Their rate rises from 0.06 to 1.95 per chunk with lost samples. Vehicle-like detections do not follow this. (c) 1–3-sample spikes of about 1.9k counts at a median 684/h. (d) Slow ramps. No saturation. | high (existence); medium (blips are logger-caused, since the evidence is correlational) |
| S11 | **Baseline handling.** Subtracting a baseline is required because of the ramps and day-to-day wander. Dividing by it is unjustified: the effect is ±1.6 % (±6 % from mid-scale) against event amplitudes spanning roughly 200×. A two-pass 20 s median keeps ≥99.5 % of the peak for real events and for unipolar events up to 12 s. A one-pass median fails beyond about 8 s on unipolar events. | high (subtract), medium (divide), high (median tests) |
| S12 | **Feature-path cut-off: LP 8 Hz.** This is the most robust choice across amplitude and speed. At 1.5k counts, 8 Hz costs ≤0.011 in median correlation relative to the best cut-off at each speed. 5 Hz loses 14–17 % of the peak on 4× faster events, whose 0.07 s rise matches real p5. Small events at real speed would prefer 2–3 Hz, but only because noise dominates. | medium-high |

## 2. Data and methods

**Reader** (`io_utils.py`).
- Values are `<u4` holding 24-bit codes; Time is `<u8` ticks. This matches A.
- **v2 slot logic** (`preproc.slots`, A's estimator): lost = round(dt/median) − 1 wherever dt > 1.5 × median.
  - 1–4 lost samples are filled by linear interpolation.
  - 5 or more lost samples split the segment.
  - Segments shorter than 0.5 s are masked.

**Samples.**
- Every 8th chunk of every run: 3679 spans, 42.6 h, covering 14–16 independent days per hour of day (§3.4).
- Eleven contiguous 2-h windows (22.0 h):
  - weekday 07–09 h, four times;
  - 16:30–18:30, three times;
  - 12:30 once;
  - a Saturday at 11 h;
  - 02–04 h, twice.

**v1 per-chunk statistics** (`chunk_stats.py`): raw percentiles, mains frequency and amplitude, Welch PSDs, and a 100 Hz, 20 Hz-LP series. They feed figs 1–4, 9 and `out_psd*` / `out_baseline`.

**v2 chain and detector** (`preproc.py`, `det2.py`, `run_pipeline.py`).
- Mains regression, 50 Hz + 20 harmonics: 10 s blocks at 50 % overlap, cross-faded with Hann weights, f0 estimated per block.
- LP 5 Hz (4th-order Butterworth, zero-phase, even padding), then `resample_poly(25, 1208)` to exactly 50 Hz.
- Pass 1: 20 s running median, then a 5σ detection. Pass 2: the median is recomputed with events ±2 s bridged.
- σ: MAD per 60 s block.
- Hysteresis: the core must exceed k·σ (k = 5, 6, 8); the extent runs while |r| > 2σ; events closer than 1 s are merged.
- Descriptors: extent, 90 %-energy duration, peak, p2p, SNR, 10–90 % rise on the leading edge of the dominant lobe, FWHM, lobe pattern (lobes ≥20 % of peak), mask fraction, and distance to the nearest mask or span edge.
- Durations are quantised to 20 ms, which produces the comb visible in fig12.

**Cut-off test** (`cutoff2.py`).
- Uses 886 events at 8σ whose [a−6 s, e+6 s] lies inside one dropout-free segment; 663 of them last ≥0.4 s.
- For each, the mains-free raw 2416 Hz snippet minus the pipeline baseline is the reference.
- Metrics on [a−1 s, e+1 s]: peak retention vs raw and vs LP 40 Hz; correlation with raw divided by the noise ceiling √((var_W − var_noise)/var_W); descriptors at each cut-off; excess variance above f (event periodogram minus the flank noise PSD).

**Injection** (`synth.py`, `synth_base_uni.py`).
- Templates: 40 real large events (LP 40 Hz, peak >10k, tapered).
- Speeds 1×, 2×, 4× (median true rise 0.28/0.14/0.07 s); amplitudes 700/1500/3000/20000 counts.
- Backgrounds: 29 real quiet 60 s mains-free raw stretches.
- Baseline tests: stretched real templates (2–16 s support) and unipolar half-sines (2–24 s), put through LP 5 Hz → 50 Hz → a 10/20/30/60 s median, one-pass and two-pass.

**Checks.** `gap_phase_check.py`: 238 gaps seen, 6 skipped near a chunk edge, 232 used. Also `mains_edge_check.py`, `spike_check.py` and `noise_vs_fc.py`.

## 3. Evidence

### 3.1 Clock (S1)
- Tick-derived fs is 2290 Hz (run 0212) to 2405 Hz and varies with hour. Mains measured against ticks reads 47.5–49.7 Hz, with r = 0.996 against the tick rate.
- The mains-implied rate is 2416.3 ± 2.8 Hz, p1–p99 2407.6–2421.8 (`chunk_stats.parquet`). A (§5.4) confirms it independently against the wall clock and finds a real day-time dip of 3–4 Hz in the ADC rate.
- **Dropouts are real losses.** Fitting the mains phase across 232 gaps rejects "no loss": the observed slip is 0.1–0.4 cycles, and the prediction error has median 0.07 cycles against a null of 0.16 (`out_gap_phase.txt`).

### 3.2 Baseline (S2, S11): fig1, fig2, fig8 (top), fig9; `out_baseline.txt`, `out_synth_uni.txt`
- **Level.** Per-run medians are 11.593M, 11.551M, 11.574M and 11.556M. The daily range is 57k–157k p-p (Feb 12: 310k while settling).
- **Drift speed.** Between sampled chunks 5.6 min apart, |slope| p50/p90/p99/p99.9 = 241/834/2165/6554 counts/min. Within one chunk, |last 5 s − first 5 s| p50/p90/p99 = 191/672/1948.
- **Structure function** (v1 contiguous windows; median |Δ| of 1-s medians):

  | Lag | 10 s | 20 s | 60 s | 5 min | 1 h |
  |---|---|---|---|---|---|
  | Counts | 56–175 | 85–250 | 170–610 | 0.6–2.8k | 2.6–20k |

- **The largest jumps are ramps** of 25–50k counts over about 12 min, some at night (fig9). They are not steps. The hourly-median diurnal profile spans 34k p-p.
- **Vehicle-associated offsets.** The 10-s median after an event minus the one before has median 360 counts; 13 % exceed 1000 and none exceeds 3000 (v1 windows). There is one lasting offset of about 5k, likely a vehicle stopping.
- **Distortion from the median (issue 9).**
  - **Bipolar real templates at 20k:** peak retention is 1.00 ± 0.01 for 2–16 s support, for every window from 10 to 60 s and for both passes.
  - **Unipolar half-sines at 20k, peak retained (area):**

    | Median | 8 s | 12 s | 16 s | 24 s |
    |---|---|---|---|---|
    | 1-pass, 20 s | 0.997 (0.99) | 0.75 (0.63) | 0.45 (0.29) | — |
    | 2-pass, 20 s | 1.00 | 1.01 | 0.80 (0.70) | 0.39 |
    | 2-pass, 30 s | — | — | 1.00 | 0.67 |
    | 2-pass, 60 s | — | — | — | 1.01 |

  - **Real 8σ events at least 5 s from edges:** the one-pass/two-pass peak ratio has median 0.99–1.01 in every duration bin, with p5 0.86–0.98. Among window events, 9.3 % exceed 8 s and 2.2 % exceed 12 s; only 11 % of those over 8 s are unipolar.
- **Why subtract but not divide.** The absolute code varies ±1.6 %, or ±6 % measured from mid-scale, while events span 650 to 140k counts. Sensor sensitivity versus bias point is unknown (A F8), so any per-baseline gain correction would be a guess.

### 3.3 Noise and spectrum (S3–S5): fig3, fig4; `out_psd.txt`, `out_psd_low.txt`, `out_noise_fc.txt`, `out_mains_edge.txt`
- **Floor** (quiet chunks, night; counts²/Hz):

  | f (Hz) | 1–2 | 5 | 10 | 20 | 40 | 100 | 300 | 600 | 1000 |
  |---|---|---|---|---|---|---|---|---|---|
  | Floor | 963 | 930 | 626 | 297 | 278 | 133 | 39 | 10 | 1 |

  - Day is at most 8 % higher.
  - Slopes: −0.33 (0.6–10 Hz) and −0.40 (10–45 Hz).
  - The 1 kHz value reflects the ADC filter, not the sensor.
- **Below 1 Hz** (41 s periodograms): 1385 counts²/Hz at 0.5 Hz, 5220 at 0.2 Hz, 14 590 at 0.03 Hz.
- **Line amplitudes**, from the quiet-chunk PSD: 50 Hz 1638; 100 Hz 65; 150 Hz 633; 200 Hz 52; 250 Hz 345; 350 Hz 116; 450–650 Hz 21–27; 604 Hz 49.
- **9.45 Hz family.** Prominence over the floor ×50 (9.45 Hz), ×130 (18.9), ×101 (28.3), ×90 (37.7). The frequency is stable (p10–p90: 9.42–9.45 Hz) and the line appears every hour.
- **Chain noise σ vs cut-off** (quiet 60 s stretches):

  | LP | 2 Hz | 3 Hz | 5 Hz | 8 Hz | 10 Hz | 15 Hz | 20 Hz |
  |---|---|---|---|---|---|---|---|
  | σ (counts) | 81 | 90 | 98 | 104 | 110 | 122 | 127 |

- **Mains block edges (issue 8)**, 60 dropout-free chunks:
  - Hard 10 s blocks: the residual 50 Hz envelope is 154 counts near block edges vs 105 elsewhere.
  - Overlapping cross-faded blocks: 87 vs 89, i.e. no edge effect.
  - The LP 5 Hz change across ±0.1 s at edges has median 59 counts, compared with 69–71 at random positions, so **block edges do not create blips**.
  - Residual line amplitudes: 50/150/250 Hz = 34/40/35 counts with overlapping blocks, vs 59/64/54 with hard blocks.
- **Event chunks vs quiet chunks** (PSD ratio of medians; a chunk-level measure, not per event): ×5187 at 0.5 Hz, ×1227 at 1 Hz, ×28 at 2 Hz, ×5.4 at 3 Hz, ×1.4 at 5 Hz, ≈1 above 5 Hz.

### 3.4 Events (S6–S9): figs 10–13 (v2); v1 figs 5–8 are illustrative only
**Detection counts (8σ, v2):** 3778 in the sampled chunks and 1341 in the windows. At 5σ the counts are 14 716 and 5153.

**Morphology tables** (issue 1: vehicle-like = ≥0.4 s, no mask overlap, at least 5 s from any span edge or mask). The exclusion keeps 74 % of window events but only 39 % of sampled-chunk events, which is why the windows are primary. Values are p5 / p50 / p95.

| | Windows, large (n=194) | Windows, small+medium ≤5k (n=216) | Sampled chunks, large (n=136) |
|---|---|---|---|
| Extent above 2σ (s) | 2.37 / 5.06 / 11.7 | 0.46 / 2.26 / 6.06 | 1.83 / 4.32 / 9.42 |
| 90 %-energy duration (s) | 0.83 / 1.55 / 3.69 | 0.30 / 1.45 / 4.18 | 0.68 / 1.46 / 2.52 |
| Peak (counts) | 8.5k / 20.2k / 56.3k | 685 / 1251 / 3619 | 7.6k / 19.6k / 51.0k |
| p2p (counts) | 13.4k / 34.7k / 92.2k | — | — |
| SNR | 48 / 141 / 407 | 8.2 / 10.5 / 28 | — |
| Rise (s) | 0.14 / 0.29 / 1.25 | 0.07 / 0.76 / 2.06 | 0.13 / 0.27 / 1.03 |
| FWHM (s) | 0.28 / 0.54 / 1.00 | 0.22 / 1.2 / 3.3 | — |
| Patterns | +− 0.70, −+− 0.10, −+ 0.07, +−+− 0.06, − 0.05 | − 0.76, + 0.17 | — |
| Dominant polarity negative | 63 % | 80 % | — |

- **Headline check v1 → v2.** Large-event peak, rise, FWHM and pattern mix are unchanged: 20k, 0.29 s, 0.54 s, +− about 70 %. Extent grew from a median of 4.0 s to 5.1 s (p95 6.6 → 11.7 s), because v1 was edge-truncated.
- **Day vs night** (windows): median peak 4.5k vs 754; share over 5k 49 % vs 0 %; median extent 3.7 s vs 0.96 s. Only 13 night events, so the night figures are weak.
- **Hourly rates** (issue 3; fig13; `out_v2_analysis.txt`). Sampled chunks, 8σ, ≥0.4 s, mask_frac ≤0.2, with 95 % bootstrap-by-day CIs. Every hour has 14–16 independent days and about 1.75 recorded hours.

  | Hour | All vehicle-like (/h) | Large (/h) |
  |---|---|---|
  | 00–06 | 5.7–12.4 (CIs within 1.7–23.6) | ≤0.6 |
  | 07 | 21 (15–27) | 5.7 (3.4–7.9) |
  | 08 | 64 (46–80) | 50 (34–64) |
  | 09 | 94 (76–111) | 81 (59–101) |
  | 10 | 29 (20–37) | 19 (12–27) |
  | 11–12 | 20–23 | 4.9–6.6 |
  | 13 | 33 (24–43) | 19 (11–29) |
  | 14 | 35 (28–43) | 16 (11–22) |
  | 15–17 | 21–22 | 4.6–8.2 |
  | 18–23 | 8–15 | 0–2.3 |

  - Contiguous windows, large events per 2 h: 47–66 on weekday mornings, 30 at weekday midday, 9–11 on weekday evenings, 0 at night.
  - Saturday 11–13 h had 1. **That is n = 1 day and anecdotal; no weekday/weekend contrast is claimed.**
- **Overlap (issue 13).** 7.1 % of large window events have another detection within 5 s (a lower bound, because the 1 s merge rule hides closer pairs). 17.7 % have an extent over 8 s, i.e. possible merged pairs or queueing.
- **Amplitude modes (issue 4).** Clean ≥0.4 s detections in the sampled chunks, in 0.2-dex bins from 10^2.2:

  | Trigger | n | Histogram |
  |---|---|---|
  | 5σ | 1654 | 0 15 323 642 255 76 44 **23** 27 61 75 68 32 12 1 |
  | 6σ | 1249 | 0 1 119 447 245 82 47 **23** 29 61 77 72 33 12 1 |
  | 8σ | 763 | 0 1 11 134 168 82 49 **24** 29 61 80 76 34 13 1 |

  - The minimum (bold, the 10^3.6–10^3.8 bin, i.e. 4.0k–6.3k) and the upper mode are identical at all three triggers.
  - The lower peak shifts with the trigger. Whether it is a real mode cannot be shown with this detector.
- **Per run (issue 5).** Large events, clean, at least 5 s from edges:

  | Run | n | Peak p50 | Extent p50 (s) | d90 p50 (s) | Rise p50 (s) | FWHM p50 (s) | +− share |
  |---|---|---|---|---|---|---|---|
  | 0 | 40 | 23.2k | 4.72 | 1.50 | 0.31 | 0.53 | 0.72 |
  | 1 | 74 | 24.9k | 5.20 | 1.50 | 0.32 | 0.56 | 0.68 |
  | 2 | 110 | 18.1k | 4.58 | 1.48 | 0.27 | 0.51 | 0.74 |
  | 3 | 106 | 18.5k | 4.70 | 1.55 | 0.26 | 0.52 | 0.66 |

  - Run 0 vs runs 1–3, Mann-Whitney: all p ≥ 0.24 (peak 0.48, extent 0.32, d90 0.59, rise 0.55, FWHM 0.96, lobes 0.24). **No difference in run-0204 morphology is detectable.**
  - Of run 0's 227 large events, 56 % are clean and 100 % are repairable (mask_frac ≤0.2). The 100 repaired events match the clean ones: peak 21.4k, extent 4.52 s, rise 0.30 s, +− 0.61.
  - **Recommendation: repair rather than discard.** Keep events with mask_frac ≤0.2 and a `has_dropout` flag.

### 3.5 Cut-off choice (S12, issue 6): fig10 (real), fig11 (synthetic); `out_cutoff2.txt`, `out_synth.txt`
**Real events vs raw 2416 Hz**, medians (p5):

| Class | Peak / LP40 at 2 / 5 / 8 / 15 Hz | Corr / noise ceiling | FWHM vs its 15 Hz value | Lobe count equal to 15 Hz (at 5 / 8 / 10 Hz) |
|---|---|---|---|---|
| Large (n=307) | 0.973 / 0.995 (0.970) / 0.995 / 0.996 | ≥0.999 from 3 Hz up | changes ≤3 % | 0.99 / 0.99 / 1.00 |
| Medium (n=133) | 0.88 / 0.90 (0.76) / 0.91 / 0.94 | 1.01 at every cut-off | changes 1–8 % | 0.73 / 0.81 / 0.88 |
| Small (n=223) | 0.69 / 0.77 / 0.81 / 0.86 | — | 1.12 s at 2 Hz → 0.47 s at 15 Hz | 0.54 / 0.65 / 0.71 |

- **Small events are noise-dominated.** The noise ceiling is 0.66, and the FWHM and lobe changes track added noise.
- **Excess variance above f is indistinguishable from zero above about 2 Hz in every class.** For small and medium events it comes out slightly negative (−1 % to −11 %), meaning the flank-noise model is only good to about ±10 %.
- **99 % excess bandwidth** (median/p95): large 1.52/2.40 Hz, medium 0.47/1.09 Hz, small 0.55/2.7 Hz. For small events this is a lower bound limited by SNR.
- **Real data cannot show whether small events carry fast content**, because their SNR is too low. Hence the injection test.

**Injection with known truth**, median correlation with truth:

| Case | 2 Hz | 3 Hz | 5 Hz | 8 Hz | 10 Hz | 15 Hz |
|---|---|---|---|---|---|---|
| 1500 counts, 1× | 0.974 | 0.974 | 0.970 | 0.963 | 0.960 | 0.948 |
| 1500 counts, 2× | 0.933 | 0.959 | 0.969 | 0.963 | 0.958 | 0.947 |
| 1500 counts, 4× | 0.758 | 0.877 | 0.950 | 0.965 | 0.965 | 0.961 |
| 3000 counts, 4× | 0.773 | 0.904 | 0.975 | 0.991 | 0.991 | 0.990 |
| 700 counts, 4× | 0.636 | 0.821 | 0.904 | 0.907 | 0.899 | 0.875 |

- **Peak retention at 5 Hz** is 0.83–0.87 for 4× templates at every amplitude, against 0.94–0.98 at 8 Hz.
- **Rise and FWHM errors at 8 Hz** are ≤0.13/0.13 for amplitude ≥1500 at every speed, and ≤0.24/0.21 at 700 counts.
- **Minimax choice: 8 Hz.** Its worst-case correlation loss is ≤0.011 against the best cut-off in every amplitude-speed cell with amplitude ≥1500, including the 20 000-count cells.
- **Why 4× matters.** It reproduces the p5 rise of real events (0.07 s at 4× vs large-event p5 0.13 s and small p5 0.04–0.07 s), so events that fast exist in the data.

### 3.6 Artefacts (S10): fig12; `out_v2_analysis.txt`, `out_spikes.txt`
**Blip vs vehicle without circular use of duration (issue 4).** Short (<0.4 s) vs vehicle-like detections at 8σ in the sampled chunks:

| | Short | Vehicle-like |
|---|---|---|
| Positive dominant polarity | 93 % | 37 % |
| Share in run 0204 | 96 % | 50 % |
| Share in spans losing >500 samples | 93 % | 40 % |
| Clean rate, run 0204 (/h) | 126.5 | 15.6 |
| Clean rate, runs 1–3 (/h) | 4.6–5.7 | 17.9–22.5 |

- **Per-chunk rate vs lost samples** (0 / ≤300 / 300–1k / 1–3k / >3k): short 0.06 / 0.09 / 1.11 / 1.95 / 1.0; vehicle-like 0.24 / 0.22 / 0.23 / 0.15 / 0.0.
- **Duration without any class cut** (8σ clean): runs 0212–0224 show almost no short mode; bins from 0.1 s read 3 0 14 47 30 against a long-duration hump of 33–64 per bin. Run 0204 has a mode of 822+ at 0.1–0.3 s.
- **Mechanism.** Blips are associated with logger dropout load. The causal mechanism is inferred, not proven.
- **Spikes.** 1–3 samples (0.4–1.2 ms), 1.7k–2.9k counts (p10–p99), median 684/h (p10–p90: 0–3039/h). A 2000-count, 2 ms impulse leaves 43 counts after LP 5 Hz and 68 after LP 8 Hz, both below σ.
- **No saturation.** Codes stay far from the rails.
- **Filtering pitfalls (verified in v1).** Filtering across a dropout, or odd-padding a segment that ends on a mains crest, creates false 0.2 s, ~1000-count events. v2 avoids both by splitting at dropouts, regressing mains out before the LP, and using even padding.

## 4. Recommended chain (as implemented in `scripts/preproc.py`)
1. **Clock.** Use the slot index at A's `fs_local` (or 2416 Hz); use ticks only to find gaps. Fill 1–4-sample losses and split at ≥5. Mask ±0.15 s around each gap and drop segments shorter than 0.5 s. Do not discard events that touch a small mask: keep those with mask_frac ≤0.2 and flag them `has_dropout` (§3.4).
2. **Mains.** Regress out 50 Hz + 20 harmonics per segment, in 10 s blocks with 50 % overlap cross-faded by Hann weights and f0 fitted per block. This leaves about 34 counts with no block-edge effects. A notch is unnecessary after an LP at ≤8 Hz (≥100 dB rejection at 50 Hz), but the regression also keeps the raw signal usable.
3. **Low-pass.** 4th-order Butterworth, zero-phase, even padding, per segment.
   - **Detection path: 5 Hz.** σ is 98 against 104 at 8 Hz; large events keep 99.5 % of their peak.
   - **Feature path: 8 Hz.** It is minimax-best across speeds and amplitudes (§3.5), preserves 4×-fast shapes (corr 0.965–0.997, peak 0.94–0.98), and suppresses the 9.45 Hz line by about 4 dB (≈40 → 25 counts). 10 Hz is an acceptable alternative (σ 110). Do not use ≤3 Hz for features.
4. **Resample.** `resample_poly(y, 25, 1208)` per segment gives exactly 50 Hz (Nyquist 25 Hz, over 3× the 8 Hz cut-off). 25 Hz is the minimum and only suits the 5 Hz path.
5. **Baseline.** Subtract a **20 s running median in two passes**: detect at 5σ, bridge events ±2 s, then recompute. Never divide.
   - It is exact for bipolar events of any length tested and for unipolar events up to 12 s. Only 2.2 % of events exceed 12 s, and only 11 % of those over 8 s are unipolar.
   - Use 30 s two-pass if long unipolar events matter (safe up to 16 s), at the cost of more wander (116–330 vs 85–250 counts).
6. **Detection.** Robust σ per 60 s block (median 106 counts). Trigger at 8σ, or 5–6σ to reach further down the small-event tail. Require ≥0.4 s. Near-lane vehicles sit above the 4–6k minimum.
7. **Windows.** 12–15 s for features: 600–750 samples at 50 Hz, covering the p95 extent of 11.7 s. Flag extents over 8 s, which are possibly merged (17.7 % of large events).
8. **Exclusions.** Skip the first hour of run 0212 for normalisation statistics. Expect run 0204 to be blip-heavy; blip rejection should use duration plus the dropout-load flag.

## 5. Open questions and limits
- **Origins unknown.** The small (~1k) class, the slow ramps, and the 0.73/1.98/9.45 Hz tones. Ground truth or temperature logs are needed.
- **Rates.** They come from a 1-in-8 sample (about 1.75 h per hour of day) plus 22 h of contiguous windows. There is one weekend day.
- **Detector.** Deliberately simple. Sub-5 s pairs are under-resolved, so overlap is a lower bound.
- **Direction of travel and polarity.** Untested; they depend on the sign convention.

## 6. File index
- **Figures (v1):**
  - fig1 baseline timeline; fig2 diurnal and drift
  - fig3 PSD and lines; fig4 low-frequency PSD
  - fig5/6 v1 event morphology and diurnal (superseded by fig12/13)
  - fig7 v1 gallery (random stratified pick, seed 7; illustrative)
  - fig8 contiguous rush hour and structure function; fig9 ramps
- **Figures (v2):** fig10 cut-off test on real events; fig11 injection tests; fig12 bimodality and duration by run; fig13 hourly rates with CIs.
- **Outputs:**
  - `out_psd.txt`, `out_psd_low.txt`, `out_baseline.txt`, `out_events.txt` (v1)
  - `out_cutoff2.txt`, `out_synth.txt`, `out_synth_uni.txt`, `out_v2_analysis.txt`
  - `out_mains_edge.txt`, `out_spikes.txt`, `out_noise_fc.txt`, `out_gap_phase.txt`
- **Data:** `v2_events.parquet` (all thresholds, with the `thr` column), `v2_spans.parquet`, `cutoff2_events.parquet`, `synth_lp.parquet`, `synth_base.parquet`, `chunk_stats.parquet`.
