# Workstream B: Signal characterisation of the TMR vehicle sensor

_Saved by the orchestrator from the agent's hand-back; the harness blocked the subagent from writing it. Orchestrator checks (2026-09-29):_
- _`data/events_classified.parquet` reproduces the class counts: 6824 = 4476 dropout-overlap + 1743 short blips + 605 vehicle-like._
- _The fig7 gallery was inspected visually. Large events are clean bipolar or multi-lobe shapes far above the noise. Small events are marginal. Blips sit on dropouts._
- **_Orchestrator caveat:_** _events are detected per sampled chunk (every 8th chunk), so events cut by the 41 s chunk edges exist (see the `edge` column; e.g. gallery panels starting at 0 s or ending at 41.4 s). Duration and extent statistics in §3.4 are therefore biased low for long events unless the edge events are excluded. The contiguous-window results (§3.4 rates, S9) are not affected._

All scripts are in `scripts/`, figures in `figures/`, and small derived data in `data/`. The `*.npz` files are git-ignored and can be regenerated.

Every number below uses **fs = 2416 Hz** (the ADC rate implied by the mains line) and a **sample-index time axis**. Time ticks are used only to locate dropouts, in agreement with workstream A (F3–F5). No reported number uses 2400 Hz or µs ticks. An early quick-look used a tick-derived 2381 Hz; it was discarded and every figure regenerated.

## 1. Summary (confidence in brackets)

| # | Finding | Conf. |
|---|---|---|
| S1 | **The ADC runs at about 2416 Hz, not 2400 Hz, and Time ticks are not µs.** The tick-derived rate is 2290–2405 Hz and tracks the apparent mains frequency (r = 0.996). Assuming mains = 50 Hz gives an ADC rate of 2416.3 ± 2.8 Hz (1–99 %: 2407.6–2421.8). The filename wall-clock gives 2413–2414 Hz for runs 1–3. | high |
| S2 | **The baseline sits at 11.47M–11.85M counts (about 3.2M above mid-scale) and wanders.** It moves 46k–160k counts p-p within a day. The largest changes are **smooth ramps** of up to about 4k counts/min over about 10 min, not steps. The diurnal part is weak: the hourly profile spans 34k p-p, with its minimum at 12–13 h. Run offsets differ by up to 42k. Run 0212 shows a +300k settling excursion in its first hour. | high |
| S3 | **The noise floor after mains removal is 33 counts/√Hz at 1–5 Hz.** It falls as f^−0.44 to 17 counts/√Hz at 20 Hz and 11.5 at 100 Hz. Above about 300 Hz it rolls off steeply because of the ADC's digital filter, reaching 0.8 at 1 kHz. Below 0.5 Hz it rises like 1/f. In the 0–5 Hz band σ is about 85–100 counts (118 by day, 82 at night). | high |
| S4 | **Mains hum dominates the raw signal.** The 50 Hz line has a median amplitude of 1430 counts (range 690–1890) and a diurnal cycle from about 970 (03–05 h) to about 1650 (12 h and 19 h). Odd harmonics are strongest: 150 Hz ≈ 630, 250 Hz ≈ 345, 350 Hz ≈ 115. Even harmonics are 50–65 counts. The amplitude comes in bursts, so the hum is not stationary. | high |
| S5 | **Other tonal lines are present in every run and at every hour, at fixed frequencies.** A 9.45 Hz line with harmonics at 18.9, 28.3 and 37.7 Hz has amplitudes of about 40, 26, 2 and 0.2 counts. A 0.732 Hz family (0.732, 1.465, 2.197 Hz, …) is about 21–25 counts per line, and a 1.978 Hz line is about 23 counts. The origin of these lines is unknown. Separately, lines at 604 Hz (fs/4) and 1208 Hz (fs/2) are ADC artefacts. | high (existence), low (origin) |
| S6 | **Vehicles produce slow, large signatures.** Large events (over 5k counts) have a median peak of 20k (p5–p95: 7.9k–59k; max 142k). Their 90 %-energy duration is 1.3 s (0.64–2.4 s), and they stay above 2σ for 4.0 s (1.7–6.6 s). The 10–90 % rise time is 0.29 s (0.13–1.0 s). For 95 % of events, 99 % of the energy is below 2.4 Hz. For event chunks as a whole, 99 % of the excess power is below 1.4 Hz. | high |
| S7 | **Large events are mostly bipolar.** Patterns are +− 68 %, −+ 8 %, −+− 8.5 %, unipolar 8 % and 4-lobe 5 %. The order of the signs may encode direction of travel; this is untested. | high (shapes), low (direction) |
| S8 | **Amplitudes are bimodal.** A large group (around 10^4.3 counts) appears almost only between 07 and 18 h: 57/h at 09 h, 32/h at 08 h, about 12/h at 13–14 h and about 0 at night. A small group (about 700–1500 counts, SNR 8–15) appears at all hours, at about 6/h at night and roughly twice that by day. It is probably a mix of far-lane traffic and non-vehicle disturbances. | high (bimodality), low (small-group origin) |
| S9 | **Overlaps are rare, even at the morning peak.** In 2-h contiguous windows from 07 to 09 h, the median gap between large events is 47–97 s. Only 1–3 pairs per window start less than 5 s apart (about 2–5 %), and 0–8 detections per window last longer than 8 s (merged events or queueing). | medium-high |
| S10 | **The main non-vehicle artefacts come from the logger.** (a) Dropouts of about 105 samples (see A). (b) "Logger blips" of 0.17–0.28 s and +670 to +1100 counts: 94 % are in chunks with dropouts, and their rate grows from 0.09 to 1.3 per chunk as dropouts increase. Run 0204 has 76/h, plus 200/h touching gaps; runs 1–3 have 5–10/h. (c) Impulsive spikes 1–3 samples wide and about 2000 counts, at a median of 600/h. (d) Slow ramps. There is no saturation. | high (a–c), medium (origin of d) |
| S11 | **Dividing by the DC level ("DC-relative" normalisation) is not justified; subtracting a slow baseline is essential.** Dividing changes amplitudes by only ±1.6 %, or ±6 % measured from mid-scale. Event amplitudes naturally vary about 100-fold, and there is no evidence that sensitivity depends on the bias point. | high (subtract), medium (divide unnecessary) |

## 2. Data and methods

- **Reader.** `io_utils.py` reads Values as `<u4` (24-bit codes with the top byte 0) and Time as `<u8` ticks, which matches A's `bsd_io.py`.
  - A tick step of more than 1000 ticks (about 2.4 samples) counts as a dropout.
  - Time = (sample index + samples lost in earlier gaps) / 2416.
- **Sample.** Every 8th chunk of every run: 3679 chunks, 42.3 h, covering all 18 days and all hours. `chunk_stats.py` computes, per chunk:
  - raw percentiles, standard deviation and first-difference noise;
  - mains frequency and amplitude, from a zero-padded FFT of the longest segment without dropouts;
  - Welch PSDs (mean- and median-averaged, nperseg 4096);
  - a 100 Hz series built within each dropout-free segment: regress out 50 Hz and 20 harmonics, apply a 4th-order zero-phase 20 Hz LP with even padding, then resample onto the gap-aware grid;
  - a dropout mask covering ±0.15 s around each gap and any segment shorter than 0.5 s.
- **Contiguous windows.** 11 windows of 2 h each, about 1900 chunks in total (`contiguous.py`):
  - weekday 07–09 h, four times;
  - 16:30–18:30, three times;
  - 12:30 once;
  - Saturday 11–13 h;
  - night 02–04 h, twice.
- **Detector** (`events.py`, for characterisation only):
  - Signal: 5 Hz LP, minus a 15 s running median.
  - Noise: robust σ per chunk.
  - Event: a core above 8σ, extended while the signal stays above 2σ; events less than 1 s apart are merged.
  - Descriptors: duration, 90 %-energy duration, peak, peak-to-peak, SNR, lobe sign pattern, 10–90 % rise time, FWHM, and 90/95/99 % energy bandwidth.
  - Bandwidth is measured on the 20 Hz series over the event ±1 s, with the baseline removed, a taper, and the noise floor subtracted.
- **Event classes.**
  - Dropout-overlap: touches the dropout mask.
  - Short blip: shorter than 0.4 s.
  - Vehicle-like: 0.4 s or longer and clean. Within this class, "large" means amax > 5000 counts.
- **Figure scripts:** `fig_baseline.py`, `fig_psd.py`, `fig_psd_low.py`, `fig_events.py`, `fig_gallery.py`, `fig_contiguous.py`, `fig_steps.py`.
- **Check scripts:** `gap_phase_check.py`, `cutoff_test.py`.
- **Reference chain:** `preproc.py` implements the recommended preprocessing.
- **Saved output:** `data/out_*.txt`.
- **To regenerate:** run `OMP_NUM_THREADS=1 python chunk_stats.py 8 5` (about 5 min), then `python contiguous.py` (about 2.5 min). Keep BLAS pinned to one thread; otherwise the regression step oversubscribes the CPUs and runs about 50 times slower.

## 3. Evidence

### 3.1 Clock and format (S1)
- **Format.** Codes span 11.42M–11.86M in the sample, with no saturated values and nothing at or above 2^24.
- **Tick rate.** The median tick step is 416–437. That puts the tick-derived rate between 2290 Hz (run 0212) and 2405 Hz, varying with the hour (2372–2388).
- **Mains.** Measured against ticks, mains sits at 47.5–49.7 Hz, and it correlates with the tick rate at r = 0.996. Rescaled to 50 Hz, this gives 2416.3 ± 2.8 Hz. The spread matches normal grid variation of ±0.05 Hz, which corresponds to about ±2.4 Hz.
- **Dropouts are real data losses.** A mains-phase fit before and after 232 gaps (`gap_phase_check.py`) rejects "no loss". The phase slips by 0.1–0.4 cycles, consistent with about 2.2 lost cycles (about 44 ms).

### 3.2 Baseline (S2): fig1, fig2, fig9, fig8 top
- **Levels.**
  - Median by run: 11.593M, 11.551M, 11.574M, 11.556M.
  - Range within a day: 46k–157k p-p; 310k on Feb 12 because of settling.
- **Drift speed.**
  - Between sampled chunks 5.6 min apart: median 241 counts/min, p90 834, p99 2165, p99.9 6554.
  - Within a 41 s chunk (last 5 s vs first 5 s): median 190 counts, p90 672, p99 1940.
- **Structure function.** Median |m(t+τ) − m(t)| over 1-s medians, quiet seconds only, 11 windows:

  | Lag τ | 1 s | 10 s | 20 s | 30 s | 60 s | 5 min | 20 min | 1 h |
  |---|---|---|---|---|---|---|---|---|
  | Counts | 27–55 | 56–175 | 85–250 | 116–330 | 170–610 | 0.6–2.8k | 1.6–10.7k | 2.6–20k |

  Up to τ = 20 s the wander stays at or below about 2σ.
- **What the big jumps are (fig9).**
  - Zooming into the 6 largest jumps shows smooth ramps of 25–50k counts over about 12 min, some at night, with vehicles on top.
  - They are not steps, so parked vehicles are not the explanation.
  - The cause is unknown; heat or a large slow-moving magnetic object are candidates.
- **Offsets left behind by vehicles.** For 368 large events, the 10-s median after minus before:
  - median 360 counts;
  - 13 % above 1000 counts, none above 3000;
  - median 1.6 % of the event peak.

  One gallery example (02-12 14:53) leaves a lasting offset of about 5k, likely a vehicle that stopped nearby. Such cases are rare.
- **Mains follows building load, not the sensor.** Its amplitude has a clean daily cycle (fig1, third panel).

### 3.3 Noise and spectrum (S3–S5): fig3, fig4
- **Noise floor in quiet chunks** (running-median floor; counts²/Hz):

  | Frequency | 1 Hz | 5 Hz | 10 Hz | 20 Hz | 40 Hz | 100 Hz | 300 Hz | 600 Hz | 1000 Hz |
  |---|---|---|---|---|---|---|---|---|---|
  | Floor | 1126 | 986 | 649 | 278 | 267 | 132 | 37 | 10 | 1 |

  - Slope: −0.44 over 0.6–10 Hz and −0.39 over 10–45 Hz.
  - Day and night differ by less than 10 % above 1 Hz.
- **Below 1 Hz** (41-s periodograms): 1.4k counts²/Hz at 0.5 Hz, 5.2k at 0.2 Hz and 14.6k at 0.03 Hz.
- **Line amplitudes (counts).**

  | Line | Amplitude |
  |---|---|
  | 50 Hz | 1640 |
  | 100 Hz | 65 |
  | 150 Hz | 633 |
  | 200 Hz | 52 |
  | 250 Hz | 345 |
  | 350 Hz | 116 |
  | 450–650 Hz | 21–27 |
  | 604 Hz (fs/4) | 49 |
  | 9.45 / 18.87 / 28.3 / 37.7 Hz | 40 / 26 / 2 / 0.2 (25–50× above the floor, every hour) |
  | 0.732 Hz family and 1.978 Hz | 21–25 each, stable to ±0.02 Hz in all runs and hours |

  The 0.73 and 1.98 Hz lines fall inside the vehicle band, but at about 0.25σ each they only matter for small events.
- **Where vehicle energy is.** The PSD ratio of event chunks to quiet chunks is 5000× at 0.5 Hz, 1200× at 1 Hz, 28× at 2 Hz, 5× at 3 Hz and 1.4× at 5 Hz. Above 5 Hz it is about 1. Of the excess power, 90 % is below 0.81 Hz, 99 % below 1.39 Hz and 99.9 % below 2.56 Hz.
- **Bandwidth of large events** (n = 259): 95 % of energy lies within 0.56–1.8 Hz and 99 % within 0.78–2.4 Hz. Rise times of at least 0.13 s imply a bandwidth of about 2.7 Hz or less, which agrees.

### 3.4 Events (S6–S9): fig5, fig6, fig7 (gallery of 20), fig8
- **Detections in the 42.3-h sample.** 6824 in total:
  - 4476 overlap dropouts, mostly run 0204;
  - 1743 short blips;
  - 605 vehicle-like.

  The dropout-overlap group includes real run-0204 vehicles, so vehicle rates for run 0204 are biased low (10/h versus 14–21/h in the other runs).
- **Vehicle-like events** (5th / 50th / 95th percentile):

  | Measure | 5th | 50th | 95th |
  |---|---|---|---|
  | Time above 2σ | 0.48 s | 2.4 s | 5.8 s |
  | 90 %-energy duration | 0.33 s | 1.2 s | 2.5 s |
  | Peak (amax) | 704 | 2.7k | 44k |
  | SNR | 8 | 18 | 280 |
  | Rise time | 0.08 s | 0.34 s | 1.23 s |
  | FWHM | 0.17 s | 0.55 s | 1.7 s |

- **Large events** (n = 259):

  | Measure | 5th | 50th | 95th |
  |---|---|---|---|
  | Peak (amax) | 7.9k | 20k | 59k |
  | Peak-to-peak | 10.7k | 31k | 87k |
  | SNR | 34 | 119 | 389 |
  | Time above 2σ | 1.7 s | 4.0 s | 6.6 s |
  | 90 %-energy duration | 0.64 s | 1.3 s | 2.4 s |
  | Rise time | 0.13 s | 0.29 s | 0.96 s |

- **Small events** (n = 346): peaks of 674 / 1118 / 3744 counts and durations of 0.43 / 1.5 / 3.9 s (5th / 50th / 95th percentile).
- **Day versus night.**
  - By day (07–19 h), the median peak is 9.0k, 54 % of events exceed 5k, and the median duration is 2.9 s.
  - At night, the median peak is 857, 3 % exceed 5k, and the median duration is 1.0 s.
  - Noise is about 40 % higher by day (118 vs 82 counts).
- **Large events per hour.** About 0 from 20 to 06 h, then 2.3 at 07 h, 32 at 08 h, 57 at 09 h, 14 at 10 h, 12 at 13–14 h and about 3 at 15–17 h.
  - Traffic is lopsided: large events arrive in the morning and after lunch, while evenings are dominated by small events. This fits one near lane (entering) plus one far or opposite lane.
  - The Saturday midday window has no large events at all.
- **Gallery (fig7).** Large events are smooth bipolar or multi-lobe shapes lasting 2–6 s, and a 2 Hz LP overlays the 20 Hz trace almost exactly. Small events sit only 8–15σ above a floor that contains bursts of leftover mains and spikes of about 1 ms.

### 3.5 Stationarity and artefacts (S10–S11)
- **Logger blips.**
  - Before masking, 66 % of short detections lie within 0.3 s of a dropout, against 16 % for random positions.
  - They survive per-segment filtering, mains regression and masking, and their rate still tracks dropout density. Per chunk: 0.09 / 0.12 / 0.32 / 1.3 / 1.0 for 0 / 1–2 / 3–5 / 6–10 / 11–20 gaps.
  - In run 0204 the daily count rises from 12 to 414, following the growth in dropouts.
  - In the raw waveform a blip looks like about 0.2 s of distorted mains with a DC shift of about +800 counts.
  - The best discriminator is duration: blips last under 0.4 s, whereas vehicles have p5 = 0.48 s and large events last at least 1.7 s.
- **Two filtering pitfalls that create about 1000-count, 0.2 s false events.** Both were verified.
  1. Filtering across a dropout joins two pieces of 50 Hz waveform with a phase jump.
  2. Odd padding at a segment edge that falls on a mains crest shifts the DC level by 2·x_end.

  The fix: split at dropouts, remove mains before low-pass filtering, and use even padding.
- **Spikes** (inline check on 150 chunks, script not saved): 0.4–1.2 ms wide, 1.7k–3.0k counts, median 600/h (p90 3200/h). A 5 Hz LP reduces a 2000-count, 2 ms spike to 43 counts or less.
- **No saturation.** Codes stay far from both rails.
- **Mains bursts.** Residual bursts of about ±1000 counts remain after regression. They do not matter once the signal is low-passed at 10 Hz or less; a 5 Hz filter attenuates 50 Hz by about 160 dB.
- **"DC-relative" normalisation.**
  - Subtracting the baseline is mandatory: day-to-day wander of about 100k is 5× a large event and 100× a small one.
  - Dividing is not justified. The code varies only ±1.6 %, or ±6 % measured from mid-scale, while events span 700–140k counts, and sensitivity is unknown (A F8). Dividing would also inject the unexplained ramps into the amplitude.
  - If amplitude invariance is wanted, normalise each event by its own peak or RMS and keep log-amplitude as a separate scalar feature.

## 4. Preprocessing recommendations
1. **Clock.** Use the sample index at A's `fs_local` (or 2416 Hz). Use ticks only to find gaps.
2. **Dropouts.** Split into dropout-free segments and never filter across a gap. Mask ±0.15 s around each gap, drop segments shorter than 0.5 s, and set a `has_dropout` flag on any window that touches the mask.
3. **Mains.** No notch is needed if the low-pass cut is 10 Hz or lower (at least 100 dB at 50 Hz). If any stage works above 10 Hz, regress out 50 Hz plus 20 harmonics per segment, in blocks of 10 s or less, with f0 estimated per chunk (`preproc.remove_mains`). A fixed notch cannot follow the ±0.1 Hz drift or the burst modulation.
4. **Low-pass.** Use a **4th-order Butterworth at 5 Hz, zero-phase, even padding.**
   - At 5 Hz, large events keep 99.8 % of their peak (p5 98.1 %) with waveform correlation 1.000. Noise drops from 115 to 85 counts, and the 9.45 Hz line is cut by 44 dB.
   - A 3 Hz cut still keeps 99.1 % (p5 96.3 %), but 2 Hz starts to clip fast events (p5 89.5 %).
   - 8 Hz is the highest sensible cut (noise 91 counts).
5. **Decimation.** Resample to **50 Hz** with `resample_poly(x, 25, 1208)` per segment, or interpolate onto a grid.
   - This gives at least 10× oversampling of the content below 2.5 Hz, which is good for EMD/HHT.
   - **25 Hz is the minimum.** Below 10 Hz the rise-time information starts to alias.
6. **Baseline.** Subtract a **20 s running median** on the 50 Hz series, then refine it: mask detected events and interpolate the baseline across them.
   - 20 s is at least 3× the p95 event extent (6.6 s).
   - It leaves 85–250 counts of wander, about 1–2σ.
   - A centred median removes linear ramps (up to 6.6k counts/min) exactly.
   - A 60 s window would leave 170–610 counts, more than a small event.
   - Subtract; never divide.
7. **Detection.** Use a robust σ per chunk (60–280 counts in the 0–5 Hz band, median 99).
   - Trigger at **8σ**, with a minimum duration of 0.4 s.
   - A peak above 5000 counts separates near-lane vehicles, which sit at 50σ or more; the gap between the two amplitude groups is at 3–5k.
8. **Event window.** Use **8–10 s centred on the energy centroid**, i.e. 400–500 samples at 50 Hz. Single-event windows are adequate because overlaps are 5 % or less. Flag merged extents longer than 8 s.
9. **Caveats.**
   - Exclude the first hour of run 0212 from normalisation statistics.
   - Run 0204 carries many blips and masked events.
   - The 0.73 and 1.98 Hz lines are constant tones of about 25 counts and matter only if small events are used.

## 5. Open questions
- What the small (about 1k count) events are. This needs ground truth.
- Where the slow ramps and the 0.73, 1.98 and 9.45 Hz tones come from. Temperature logs would help with the ramps.
- Rates are order-of-magnitude only: they come from a 1-in-8 sample plus 22 h of contiguous data with a deliberately simple detector.
- Whether the +− versus −+ order encodes direction of travel is untested.

## Files
Everything is under `research/extraction/B_signal/`.

- **Figures** (`figures/`):
  - `fig1_baseline_timeline.png`
  - `fig2_baseline_diurnal_drift.png`
  - `fig3_psd_noise.png`
  - `fig4_psd_lowfreq.png`
  - `fig5_event_morphology.png`
  - `fig6_event_diurnal.png`
  - `fig7_event_gallery.png`
  - `fig8_contiguous_rushhour.png`
  - `fig9_baseline_steps.png`
- **Data** (`data/`):
  - `chunk_stats.parquet`
  - `chunk_stats_enriched.parquet`
  - `events_classified.parquet`
  - `events_contiguous.parquet`
  - `lp100.npz` (57 MB; the dropout-aware 100 Hz series for 3679 chunks, which may help workstreams C and D)
  - `psd_summary.npz`
  - `out_*.txt`
- **Scripts:** `scripts/`, including `preproc.py`, the reference implementation of the chain above.
