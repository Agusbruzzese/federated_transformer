# Workstream C — pre-registered evaluation plan

Written after exploratory viewing of raw data (format check, 8 one-hour overviews, one zoomed busy span;
figs/look_raw.png, figs/overview_1h.png, figs/zoom_busy.png) and BEFORE any detector was run or scored.
It is not edited afterwards; deviations are listed in REPORT.md under "Deviations from pre-registration".

## Observations that shaped the plan (exploration only)
- Values are <u4 ADC counts (~1.15e7, 24-bit range), Time <u8 us since run start; ~1.2 gaps of ~44 ms per chunk.
- Strong 50 Hz mains ripple (+harmonics); vehicle energy < ~5 Hz. Front end = 20 ms time-binned mean (50 Hz output):
  a boxcar with exact nulls at 50 Hz and harmonics; gap-aware because it bins on timestamps.
- Vehicles: bipolar/multipolar signatures, ~1-2 s core, approach/departure ramps up to 5-10 s, amplitude ~1e3-2e5 counts.
- Slow drift of 1e3-1e4 counts over minutes; low-frequency wander ~±300 counts; 2 ms EMI spikes.
- Low-traffic road: ~3-40 events/h.

## Detectors (defaults chosen a priori; threshold swept for ROC)
(a) FIXED: old approach — DC baseline = mean of long context (proxy for 120-min DC), non-overlapping fixed windows (4 s),
    a window is an event if max|y-DC| > k*sigma_global (k=5). Onset/offset = window edges.
(b) HYST: masked two-pass rolling-median baseline (60 s), rolling robust sigma (MAD), envelope = 0.2 s moving RMS of z/sigma;
    on at k_on=5, off at k_off=2, merge gaps < 1 s, drop < 0.3 s. Onset/offset = k_off crossings.
(c) STALTA: energy z^2, STA 0.5 s, trailing LTA 30 s, on at ratio 8, off at 2. Onset = trigger point minus STA; offset = off-crossing.
(d) CUSUM: two-sided (forward + time-reversed) Page CUSUM on u^2=(z/sigma)^2 with drift nu=4 and threshold h (default 100);
    onset = last zero of forward statistic before alarm; offset = last zero of backward statistic; event = union.
(e) WAVELET: max over Gaussian-derivative (order 1,2) filters at scales 0.15-1.2 s of robust-normalized response; hysteresis
    on 5 / off 2 on the max response; boundaries refined as in (b).
(b),(c),(d),(e) share the same front end + baseline + sigma so differences are attributable to the detection rule.

## Evaluation set
~14 continuous spans of 20 min (+ context margins) from all 4 runs: night, morning rush, midday, evening, weekday vs weekend.

## Label-free criteria (pre-registered)
1. Synthetic injection. Templates: >=30 clear, isolated real events (peak SNR>40, no other event within ±20 s), tapered,
   unit peak. Backgrounds: real quiet spans (night/weekend), injections placed >=20 s away from any background detection
   (union of detectors at liberal threshold). SNR = peak|template| / robust sigma of detrended 50 Hz residual.
   SNR grid {3,4,5,6,8,10,15,20,40}. Match: detection interval contains the injected peak time; one-to-one.
   Metrics: recall(SNR), SNR50, SNR90; onset/offset error vs GT core (5%-95% cumulative energy); energy capture
   (fraction of template energy inside detected segment); split rate; merge rate for pairs at separations {1,2,3,5,8} s.
   False alarms: (i) FA/h on phase-randomized surrogates of the backgrounds (pure stationary noise with real PSD);
   (ii) new unmatched detections in injected backgrounds.
2. Operating point: each detector's main threshold chosen to give FA <= 0.2/h on surrogates (FIXED keeps k=5 as the legacy method,
   and is also reported at its matched-FA threshold). Detectors compared at that operating point.
3. Hand-verified set: >=200 candidates from the union of detectors on real spans, plotted with ±15 s context, labelled by eye
   V(vehicle) / N(not) / U(uncertain) without seeing which detector produced them. Precision per detector with Wilson 95% CI
   (U counted as not-V for a conservative bound and as V for an optimistic bound); pooled recall vs V-labelled union.
4. Plausibility: hourly event-rate curve over sampled days; expected weekday 07-09 h peak and 00-05 h low.
   Pass if night(00-05) rate < 20% of the weekday peak-hour rate, and weekend morning rate < weekday morning rate.
5. Stability: each parameter perturbed ±25% (one at a time); stable if event count changes < 15% and >= 90% of events keep a
   matched counterpart (interval overlap).
6. Cost: CPU seconds per hour of data (front end, detection separately; I/O reported separately).

## Decision rule (pre-registered)
Primary: highest recall at SNR 8 at the matched operating point, subject to hand-verified precision (conservative) >= 0.90
and energy capture median >= 0.9 at SNR >= 10. Ties (<3 pp recall): lower onset/offset error, then stability, then cost.
Segment representation: choose fixed-length window length L as the smallest length covering the 95th percentile of detected
core durations + ramps (energy 1-99%) of real events, centred on the energy centroid; compare against variable-length by
contamination (fraction of windows containing another event's energy >10%) and truncation (fraction of events whose 1-99%
energy extent exceeds L).
