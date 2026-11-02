# Extraction re-design — workstream briefs

Issued by the orchestrator on 2026-09-29. Each workstream works only inside its own folder,
writes `REPORT.md`, and must state hypotheses, pre-register evaluation criteria, quantify, and give
a confidence level per finding. Shared constraints: data is unlabeled; don't load the full 33 GB;
≤5 CPU processes and ≤10 GB RAM per agent; Python at `.venv/bin/python`.

Known facts at issue time: 4 runs, ~29.4k chunk pairs, ~41.7 s per chunk, `Hardware.cfg` says
2400 Hz, gain 8, differential DAC. Unverified hypothesis: Values = LE uint32 (100k samples),
Time = LE uint64 µs. Run 1 has 14563 Values vs 14564 Time files.

## A — Data integrity & I/O (`A_integrity/`)
1. Exact binary format of Values/Time (dtype, endianness, signedness, effective ADC bits, Time units/epoch), verified against filename Start/Stop and 2400 Hz. Contents of Config and Data Buffer files.
2. Actual sampling rate, jitter, dropped samples, clock drift.
3. Continuity: inter-chunk gaps/overlaps, unpaired files, duplicates, run boundaries; per-day coverage table (hours, gap count, largest gap) for all 14 days.
4. Value sanity on a stratified sample (~50 chunks): clipping, stuck values, glitches, sentinels, sign.
5. Deliver a validated loader `bsd_io.py` (`list_chunks`, `load_chunk`, `load_span` with gap reporting) plus a self-test and a cached `chunk_index.parquet`.

## B — Signal characterization (`B_signal/`)
1. Baseline/DC level and drift across minutes/hours/days (14-day timeline from per-chunk stats).
2. Noise: Welch PSD of quiet segments, noise floor, 50 Hz and harmonics, 1/f, useful event bandwidth.
3. Events: morphology (polarity, duration, amplitude, SNR, rise time), day vs night, overlap frequency; gallery of ~20 examples.
4. Artefacts and non-stationarity; is the "DC-relative" normalisation justified?
5. Recommended preprocessing chain (detrend method/time constant, filters, safe decimation) with numerical justification.

## C — Event detection & segmentation (`C_detection/`)
1. Evaluation spans covering night/rush/midday/evening, weekday vs weekend (7–8 Feb = Sat/Sun), several runs.
2. Compare ≥4 detectors: fixed sliding windows (old baseline), MAD threshold + hysteresis, STA/LTA, CUSUM/change-point, optionally template/wavelet.
3. Label-free evaluation: synthetic injection (recall/precision/onset error vs SNR), hand-verified sample precision, diurnal-rate plausibility, parameter stability, compute cost.
4. Handle overlapping vehicles, drift, chunk boundaries, gaps.
5. Recommend detector + parameters + segment representation; deliver `detector.py`.

## D — Feature representation (`D_features/`)
1. Audit EMD/HHT: IMF-count variability, mode mixing, stability under shifts/noise, EEMD/CEEMDAN, cost.
2. Compare: old 70-feature EMD/HHT set, wavelet features, STFT/spectral, time-domain morphology, raw resampled waveform.
3. Label-free criteria: augmentation invariance, redundancy/effective dimension, confound leakage (time-of-day/run predictability — lower is better), cluster structure and KMeans stability (ARI), physically meaningful proxies; valid contrastive augmentations.
4. Recommend representation, feature definitions, normalisation (per-client vs global, non-IID), and KMeans pseudo-labels vs SimCLR-style pairs; deliver `features.py`.

## E — Implement and run the extraction pipeline (`extraction/`, issued 2026-09-30)
Implement SYNTHESIS.md §3 exactly, as a tested module in the repo. Run it over all 14 days. Validate its outputs.
Deliverables: the `extraction/` package, tests, the data store in `data/events/`, and `research/extraction/E_pipeline/REPORT.md`.

## F — Central self-supervised model (`ssl_central/`, issued 2026-09-30)
Train and evaluate exactly as `research/model/PREREG_central.md` (v2) plus `PREREG_central_addendum1.md` specify.
Report every deviation. Deliverables: the `ssl_central/` package, trained checkpoints and embeddings (gitignored, under `data/`), and `research/model/F_central/REPORT.md` with the pre-registered verdict.

## G — Tightened central evaluation (`research/model/G_loro/`, issued 2026-10-01)
Run PREREG_central_v3.md exactly: LORO over 4 runs, 5 seeds, amp_resid, P2a–d, ablation. Report all deviations.
