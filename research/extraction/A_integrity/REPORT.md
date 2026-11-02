# Workstream A: Data integrity and I/O (revision 2, after audit AUDIT_20260929_213354)

_Saved by the orchestrator from the agent's hand-back; revision 1 is kept as `REPORT_rev1.md`. The orchestrator independently re-ran these on 2026-09-29, and they match:_
- _self-test 46/46_
- _188 102 dropout events, 16 462 690 missing samples, max 108_
- _fs_local mean 2416.35 Hz_

Dataset: `Spintronic Sensor Dataset ... (14 Days, Unlabeled with Timestamps)/`. 4 runs, 29 417 chunk pairs, 2.942 G samples, 33 GB.
Loader: `bsd_io.py`. The self-test (`python bsd_io.py`) passes 46/46 (`selftest_stdout.txt`). Index: `chunk_index.parquet`.

## 1. Summary

| # | Finding | Evidence base | Confidence |
|---|---|---|---|
| F1 | **Values** are little-endian `uint32`, 4n bytes, no header. Bits 31..24 are always 0, so each value is a 24-bit ADC code. 100 000 samples per chunk. | All 2.942 G samples (full scan) | Very high |
| F2 | **Time** is little-endian `uint64`, 8n bytes: device timer ticks with origin at logger start (the first sample is at 4.5 to 16.1 s). No zero or negative deltas anywhere (min Δ = 50 ticks). Increasing across chunks, no wrap. | Full scan | Very high |
| F3 | **Time ticks are not µs.** Tick rate against the PC clock is 1.007 to 1.051 per µs, depending on run and hour: run 0212 ≈1.05, others 1.01 to 1.03, with day/night swings. Using `Time/1e6` as seconds is 1 to 5 % wrong. | All chunks (edges) + mains | Very high |
| F4 | **fs ≈ 2416 Hz, not 2400.** The sample-weighted mean is 2416.3 Hz. Run means are 2417.6, 2416.0, 2415.4 and 2414.3 Hz. Chunk-level `fs_local` spans 2407 to 2421 Hz (p1 to p99 about 2408 to 2421). The per-chunk value is known to ±1.2 Hz (1σ, §6). The day-time dip of 3 to 4 Hz is a real change in the ADC rate, not a PC-clock or grid artefact (§5.4). | 96 stratified + 332 dense mains measurements; all-chunk wall-clock fit | **High** for the value and the attribution (revised; the grid behaviour is assumed from general knowledge, not measured) |
| F5 | **Dropouts, exact counts.** 188 102 dropout events lose 16 462 690 samples (0.557 % of ADC slots). Run 0204 loses 1.049 %; runs 0212, 0218 and 0224 lose 0.079, 0.064 and 0.065 %. Sizes: 83.3 % are 100 to 106 samples (104 and 105 dominate); 16.7 % are 1 to 5 samples; only 80 are 6 to 99 or 107 to 108. **The true maximum is 108 missing samples (45.53 ms of ticks).** Minimum spacing between dropouts is 4 samples. Separately, 16 621 "late stamp only" events lose nothing. | Full scan + validated estimator | **Very high** for counts and sizes; **high** that these are genuinely lost ADC samples (revised) |
| F6 | **Chunks are contiguous.** Indices run 0..N−1, every file is paired, none duplicated, and Values/Time lengths are equal. The join between chunks is one sample, or a dropout. Filename Stop[i] = Start[i+1] to within ±1 s. The only holes are between runs. | All files | Very high |
| F7 | **Values are clean over all 2.942 G samples.** Codes 11 185 946 to 11 969 218. No zeros. At most 3 identical values in a row. 1 isolated spike (>20 MAD) in the whole dataset, in run 0212 #3 during the start-up settling. High-frequency white-noise σ ≈ 62 counts (a robust estimate from the second difference; not a Welch PSD). A 50 Hz hum of ~3k counts peak-to-peak. | Full scan (values); 484 chunks (noise) | Very high (range, stuck values, spikes); medium (noise level) |
| F8 | Undeterminable from the data: offset-binary vs two's-complement (they differ only by a constant), physical units, and differential polarity. | — | High |

## 2. File format specification

```
S0C0 BIN Values - NNNNNN Start@ YYYYMMDD - HHhMMmSSs Stop@ YYYYMMDD - HHhMMmSSs.bsd
    <u4[n], n=100000 (last chunk of a run may be shorter); bits 31..24 == 0; bits 23..0 = ADC code
S0C0 BIN Time   - (same tail).bsd
    <u8[n], device timer ticks (0.95..0.99 µs each, drifting), origin = logger start
    No header/footer/padding: sizes are exactly 4n and 8n bytes.
NNNNNN : chunk index, contiguous from 000000 per Run folder.
Start@/Stop@ : PC local wall clock (1 s resolution); Stop[i] == Start[i+1] (±1 s).
Sample k of a chunk sits on ADC slot s_k (reconstructed); dropouts = slot jumps > 1.
```

**Configuration files**
- `Configuration - Hardware.cfg` is identical in every run: SerialNumber E5848104-50593539-302E3120-FF111639, COM15, Gain 8, Sampling Rate 2400 (nominal), DAC Mode DIFFERENTIAL, D0=True, D1..D3=False.
- `Configuration - Repository.cfg` uses CRLF line endings. It holds EndUser "INL GATE", Tool "Much After Barrier", Data Points per file 100000, and the original Windows data path. The runs differ only in the letter case of these names.

**Data Buffer files.** `S0C0 - Counts/Time Data Buffer.bsd` exist in runs 0204 and 0218 only. They hold the unflushed tail after the last chunk; its first stamp is exactly one sample after that chunk. Run 0204 has 80 896 valid samples followed by zero padding (82 944 vs 81 920 entries). Run 0218 has 46 080 samples. They are ignored.

**Other file notes**
- There is no Values/Time count mismatch in any run.
- Run 0224's last chunk (#6052) has 640 samples. The data ends at 12:35:56, while the folder and file names say 14:38:27. "The logger ran on without data" is an inference from the names only.

## 3. Hypotheses & criteria

**These were not pre-registered.** They were written down during the analysis, and the acceptance criteria below were applied after the fact. They are listed so each finding can be re-checked.

| H | Hypothesis | Test / criterion | Result |
|---|---|---|---|
| H1 | Values are `<u4` with 24-bit content | File size = 4n; top byte 0 in 100 % of samples; codes away from 0 and 2^24−1 | Pass (all samples) |
| H2 | Time is `<u8` in µs | File size = 8n and monotonic (pass). "µs" requires the tick rate against the PC clock to be within 1e-4 of 1: **fail** (0.95 to 0.99 µs per tick) | µs rejected |
| H3 | fs = 2400 Hz (from the config) | Rate against the PC clock and against the mains line within 0.1 % of 2400: **fail**, both give ≈2416 | Rejected |
| H4 | Large Δt steps are lost samples, not stamp glitches | (a) value jump across the gap ≈ lag-106 difference, not lag-1; (b) counting missing slots flattens the rate against the PC clock; (c) the 50 Hz phase stays continuous on the slot grid | Pass on all three |
| H5 | The slot estimator is exact | Synthetic data with the real stamp model: per-event size error = 0 in every scenario present in the data | Pass (§5.2) |
| H6 | No hidden 1-sample losses below the gap threshold | Borderline intervals (Δt/base 1.3 to 1.7): the samples after them must not sit 1 slot off | Pass: 0 of 3 077 |
| H7 | The daily fs drift is real ADC behaviour | Mains-derived fs tracks the PC-clock fs with slope ≈ 1 and shows the same day/night gap | Pass (slope 0.99 [0.91, 1.09]) |
| H8 | Chunks are contiguous and paired | Index 0..N−1, equal lengths, tick step at every join ≈ 1 sample or a dropout | Pass |

## 4. Methods (see §9 for run order)

- **`scan_headers.py`, `scan_time_edges.py`**: names, sizes, first and last samples, 2×2000 edge stamps per file.
- **`full_scan.py`**: every Time and Values file, one at a time, 5 processes. Per-chunk statistics go to `full_scan_chunks.parquet`; one row per event (dropout or late stamp) goes to `full_scan_gaps.parquet`.
- **Slot reconstruction, `slots_from_timestamps` v2**. The stamp model comes from `measure_postgap.py`: white stamp jitter of about 1.9 ticks rms (≈0.005 slot), and the first stamp after a dropout is late by 0 to 0.87 slot (mean 0.24). A Δt ≥ 1.5×median marks a gap. A sample whose incoming Δt is a gap is "late"; every other sample is "clean".
  - Two consecutive clean samples are one slot apart.
  - For clean samples a < c: slots(c) − slots(a) = max(c−a, rint((t_c−t_a)/base)).
  - A late sample followed by a clean one sits at slot(c) − 1.
  - A late sample that is followed directly by another gap is placed by its lateness-corrected position (rint of position − 0.24), clamped between its neighbours.
- **Wall-clock model**: per run, a piecewise-linear least-squares fit of filename start time against cumulative slot count. Knots every 40 chunks (about 28 min), two passes with 3σ clipping. This gives `wall_start` and `fs_local`.
- **`synthetic_slot_test.py`**: synthetic validation (§5.2).
- **`check_ambiguous.py`**: the hidden-loss test (H6).
- **`mains_check_v2.py`**: fills dropouts on the slot grid, then finds the 45–55 Hz peak with a zero-padded FFT (2^22 points) and parabolic interpolation, for the whole chunk and each half. fs = 50/f.
- **`coverage_and_figures.py`**: coverage table and figures 1 to 6. It now uses the exact dropout counts.
- **`sample_chunks.py`, `sanity_scan.py`**: earlier samples of 73 and 484 chunks. Superseded for counts; still the source of the noise and value-jump numbers.

## 5. Evidence

### 5.1 Dropouts (full scan: `full_scan_summary_stdout.txt`, `dropouts_per_day_exact.csv`, `dropout_size_histogram_exact.csv`)

**Size histogram (missing samples: events)**
- 1: 16 732
- 2: 6 808
- 3: 7 630
- 4: 163
- 5: 41
- 6 to 79: 70 in total
- 100: 2 568
- 101: 714
- 102: 174
- 103: 455
- 104: 28 091
- 105: 120 918
- 106: 3 728
- 107: 9
- 108: 1

**Other results**
- Largest gap in the whole dataset: 45 534 ticks, 108 samples (run 0204 #6870).
- Spacing between consecutive dropouts: minimum 4 samples; 4.8 % are ≤8 apart and 21 % are ≤40 apart (bursts). None are 1 apart.
- Late-stamp-only events (Δt ≥ 1.5×median with no loss): 16 621.
- Hidden-loss test: 3 077 borderline intervals checked. The samples after them sit at a median offset of −0.001 slot, and 100 % are within 0.3 slot of no loss, so there are no hidden 1-sample losses.
- The old edge estimate was biased (est − exact: mean +3.4, sd 9.2, p99 +41, max 168). It is replaced by exact counts in the index.

| date | samples (M) | events | missing | 1–5 | 100–106 | other | % missing |
|---|---|---|---|---|---|---|---|
| 02-04 | 74.8 | 651 | 67 944 | 2 | 648 | 1 | 0.091 |
| 02-05 | 208.6 | 2 524 | 264 102 | 11 | 2 508 | 5 | 0.126 |
| 02-06 | 207.3 | 16 690 | 1 586 639 | 1 553 | 15 136 | 1 | 0.760 |
| 02-07 | 206.3 | 28 450 | 2 612 952 | 3 621 | 24 822 | 7 | 1.251 |
| 02-08 | 206.3 | 30 489 | 2 700 211 | 4 830 | 25 656 | 3 | 1.292 |
| 02-09 | 206.3 | 29 425 | 2 570 268 | 5 034 | 24 391 | 0 | 1.231 |
| 02-10 | 205.5 | 35 526 | 3 200 351 | 5 029 | 30 497 | 0 | 1.533 |
| 02-11 | 141.2 | 26 565 | 2 437 319 | 3 396 | 23 164 | 5 | 1.697 |
| 02-12 | 81.0 | 857 | 13 614 | 724 | 124 | 9 | 0.017 |
| 02-13 | 208.7 | 3 817 | 45 837 | 3 396 | 405 | 16 | 0.022 |
| 02-14 | 143.2 | 6 586 | 284 356 | 3 776 | 2 781 | 29 | 0.198 |
| 02-18 | 111.1 | 1 086 | 113 285 | 1 | 1 085 | 0 | 0.102 |
| 02-19 | 208.6 | 1 220 | 127 721 | 0 | 1 219 | 1 | 0.061 |
| 02-20 | 127.5 | 433 | 45 274 | 0 | 432 | 1 | 0.035 |
| 02-24 | 78.9 | 893 | 92 793 | 0 | 892 | 1 | 0.117 |
| 02-25 | 208.4 | 929 | 96 428 | 0 | 929 | 0 | 0.046 |
| 02-26 | 208.5 | 841 | 87 077 | 0 | 840 | 1 | 0.042 |
| 02-27 | 109.4 | 1 120 | 116 519 | 1 | 1 119 | 0 | 0.106 |

Dates are assigned by the wall time of each event.

**Per-run patterns**
- **Run 0204**: 100–106-sample dropouts dominate. From Feb 6 onwards there are 16k to 36k events per day, i.e. one every 2.4 to 5 s.
- **Run 0212**: 1–5-sample losses dominate.
- **Runs 0218 and 0224**: about 1 000 dropouts per day, i.e. one every ~90 s.

### 5.2 Slot-estimator validation (`synthetic_slot_test.py` / `_stdout.txt`)

**Generator**
- Grid period P ∈ {414, 419, 435} ticks, with ±0.3-tick linear drift within the chunk.
- 2 ticks rms white jitter, stamps rounded to integers.
- Late first stamp after a dropout: U(0, 0.9) slot.
- Isolated dropouts of 0, 1, 2, 5 and 105 samples.
- Bursts of 2 to 4 dropouts spaced 1, 2, 3, 4, 8, 12, 16, 24, 28 or 40 samples apart.
- A realistic run-0204 mix.
- 60 chunks per scenario, about 36k events in total.

**Results**
- **v2**: in every scenario except spacing 1, 0 per-event errors, 0 total error, and 0 mis-slotted samples. At spacing 1 the total is exact, but the sample sandwiched between the two dropouts is ±1 slot (640 of 2 671 events, 6e-5 of samples).
- **v1**: at spacing 1 the total is off by up to 5 and 65 % of samples are mis-slotted. v1 is otherwise exact.

Spacing 1 never occurs in the real data (minimum 4), so the real-data slot error is expected to be **0**.

On real chunks, ticks fitted against slots leave 0.022 to 0.030 slot rms residual. Only the late stamps exceed 0.5 slot (≤1.3e-4 of samples).

### 5.3 Clocks and contiguity

- The wall-clock fit residual is 0.350 s rms, max 1.75 s. This is consistent with 1 s filename quantisation (uniform σ = 0.29 s) plus write latency. **Caveat:** the residual is measured on the fit's own training points, so it is not an independent accuracy test. The mains check below is the independent test.
- `load_span` over 2 h of heavy dropouts (Feb 10, 09:00 to 11:00) and 1.5 h in run 0212: 0 samples dropped at joins, minimum Δt 413 µs (`load_span_check_stdout.txt`).

### 5.4 Sampling rate: mains check with an uncertainty budget (`mains_check_v2_stdout.txt`)

| sample | n | fs_mains mean ± sd | fs_mains − fs_wall | half-to-half diff sd | slope fs_mains/fs_wall [95 % CI] | r |
|---|---|---|---|---|---|---|
| A: stratified (run × 3 h block × 3 chunks) | 96 | 2415.59 ± 2.73 | −0.15 ± 1.17 Hz | 0.58 Hz | 0.99 [0.91, 1.09] | 0.90 |
| B: every 25th chunk, Feb 4 16:00 to Feb 8 16:00 | 332 | 2417.92 ± 2.78 | −0.05 ± 1.09 Hz | 0.55 Hz | 0.96 [0.91, 1.00] | 0.92 |

- **Day (10–17 h) vs night:**
  - Sample A: mains 2413.7 vs 2416.5, PC clock 2413.7 vs 2416.7, device ticks 2353.9 vs 2362.5.
  - Sample B: mains 2415.2 vs 2419.3, PC clock 2415.1 vs 2419.4.
- **Long term, Feb 4 to 8:** the PC-clock mean is 2417.955 Hz (±0.007 from quantisation). The mains mean is 2417.916 ± 0.152 Hz (standard error). The difference is 0.04 ± 0.15 Hz, so the PC-clock rate matches the grid's long-term average to within about 150 ppm (2σ).
- **Attribution of the diurnal dip.** The PC oscillator and the power grid are independent references, and both see the same 3 to 4 Hz daytime drop (slope ≈ 1). For the grid to produce it alone, it would need a deviation of −0.06 to −0.08 Hz held for about 7 h every day. That is outside normal Continental-Europe operation (±0.05 Hz band, frequency restored within minutes) — this is general knowledge, not measured grid data. A PC-clock artefact would not appear in the mains measurement. So the ADC oscillator itself slows during the day, probably from heating. The device tick oscillator drifts separately and more.

## 6. Uncertainty budget

| Quantity | Error source | Size (1σ) | How obtained |
|---|---|---|---|
| fs_local per chunk | Filename quantisation (1 s) in the 28-min knot fit | ≈0.2–0.5 Hz | Analytic, σ = 0.29 s over ~1680 s |
| | PC-clock rate error | ≤0.15 Hz (≤60 ppm) | Long-term mains vs PC clock: 0.04 ± 0.15 Hz |
| | Grid wander at the time of the mains measurement | ≈1 Hz (≈0.02 Hz of grid frequency) | Remainder of the 1.1–1.2 Hz spread after the other terms |
| | Mains peak estimator plus 20 s of grid wander | ≈0.4 Hz per half-chunk | Half-to-half sd 0.55–0.58 Hz / √2 |
| | **Combined, conservative** | **±1.2 Hz (0.05 %)** | Whole fs_mains − fs_wall sd is attributed to fs_local |
| fs, dataset level | Real ADC variation | 2407–2421 Hz (sample-weighted mean 2416.3) | fs_local across all chunks |
| Absolute sample time from `load_span` | Anchor from the wall-clock fit | ≈0.35 s rms, max ≈1.8 s | Fit residual (not independent, see §5.3) |
| Relative time within a chunk | fs error × duration | ≤0.05 % (≤21 ms over 42 s) | From fs_local ±1.2 Hz |
| Slot index | Estimator | 0 samples (±1 only at spacing-1 bursts, which never occur) | Synthetic test + real-data minimum spacing |
| Dropout counts | Exact per chunk | 0 | Full scan |

Rounded: **fs = 2416 ± 3 Hz** over the whole dataset (real variation, 1σ = 2.7 Hz). For a given chunk, **fs_local ± 1.2 Hz**. Using 2400 is a 0.7 % bias; using a constant 2416 is up to a 0.35 % bias.

## 7. Coverage per calendar day (`coverage_per_day.csv`)

| date | run | hours covered | first | last | largest hole (h) | % dropped |
|---|---|---|---|---|---|---|
| 02-04 | 0204 | 8.60 | 15:24:13 | 24:00 | 15.40 | 0.091 |
| 02-05 | 0204 | 24.00 | | | 0 | 0.126 |
| 02-06 | 0204 | 24.00 | | | 0 | 0.760 |
| 02-07 | 0204 | 24.00 | | | 0 | 1.250 |
| 02-08 | 0204 | 24.00 | | | 0 | 1.292 |
| 02-09 | 0204 | 24.00 | | | 0 | 1.231 |
| 02-10 | 0204 | 24.00 | | | 0 | 1.533 |
| 02-11 | 0204 | 16.50 | | 16:30:16 | 7.50 | 1.697 |
| 02-12 | 0212 | 9.30 | 14:41:49 | | 14.70 | 0.017 |
| 02-13 | 0212 | 24.00 | | | 0 | 0.022 |
| 02-14 | 0212 | 16.51 | | 16:30:31 | 7.49 | 0.198 |
| 02-15 to 02-17 | none | 0 | | | 24 | |
| 02-18 | 0218 | 12.79 | 11:12:41 | | 11.21 | 0.102 |
| 02-19 | 0218 | 24.00 | | | 0 | 0.061 |
| 02-20 | 0218 | 14.67 | | 14:40:28 | 9.33 | 0.035 |
| 02-21 to 02-23 | none | 0 | | | 24 | |
| 02-24 | 0224 | 9.08 | 14:55:20 | | 14.92 | 0.118 |
| 02-25 | 0224 | 24.00 | | | 0 | 0.046 |
| 02-26 | 0224 | 24.00 | | | 0 | 0.042 |
| 02-27 | 0224 | 12.60 | | 12:35:56 | 11.40 | 0.106 |

- Total: 340.05 h across 10 full and 8 partial days.
- "Hours covered" includes time lost to dropouts. Hours with actual samples = hours × (1 − % dropped); the worst case is Feb 11, 16.50 h → 16.22 h.
- Weekend data exists only on Feb 8–9 (Sat/Sun), and 7 of the 10 full days come from the single degraded run 0204. This severely limits any weekday, weekend or time-of-day analysis in workstreams C and D.

## 8. Anomalies

| # | Anomaly | Where | Handling |
|---|---|---|---|
| A1 | Time ticks are not µs (+1 to +5 %, drifting) | All | Never use Time/1e6. Use slot/fs_local. |
| A2 | fs ≈ 2416 (2407–2421), not 2400, with a daily ADC drift of about 0.15–0.2 % | All | Use `fs_local` (now the default in `load_chunk`). |
| A3 | Dropouts: 83 % are 100–106 samples, 17 % are 1–5; max 108 | Run 0204: 1.05 %, one event every 2.4–5 s from Feb 6. Run 0212: mostly 1–5-sample losses. Others: one event per ~90 s. | Mask or split (§10). |
| A4 | Settling transient at run start (baseline 11.87M → 11.56M over ~1 h; the dataset's only spike) | Run 0212, chunks 0 to ~90 | Exclude from normalisation and baseline statistics. |
| A5 | Truncated last chunk (640 samples); data ends about 2 h before the file name's Stop time | Run 0224 #6052 | `truncated` flag in the index. |
| A6 | Unflushed tail in the Data Buffer files | Runs 0204, 0218 | Ignored. |
| A7 | 50 Hz hum, ~3k counts peak-to-peak | All | Notch at 50 Hz; the notch must be tuned with fs_local. |
| A8 | Baseline wander of about 0.4M counts over the 14 days | All | Use a rolling baseline. |
| A9 | The first stamp after a dropout is late by 0–0.87 slot; 16 621 late stamps have no loss at all | At dropouts | Handled by estimator v2. |

**Correction to revision 1:** the first stamp after a dropout was described there as "early". It is late.

No file needs to be excluded for format reasons. This now rests on the full scan, not on a sample.

## 9. Run order (reproducibility)

All scripts are in this folder. Run them with `OMP_NUM_THREADS=1 ../../../.venv/bin/python <script>`. The dataset path is hard-coded as `DEFAULT_ROOT` in `bsd_io.py` and `ROOT` in `scan_headers.py`. The seeds are fixed. The stdout of each step is saved as `<name>_stdout.txt`.

1. `scan_headers.py` → `raw_header_scan.parquet`
2. `scan_time_edges.py` → `time_edge_scan.parquet`
3. `sample_chunks.py`, `sanity_scan.py` (exploratory; superseded for counts)
4. `python bsd_io.py --refresh` → `chunk_index.parquet` (edge estimates only on a fresh checkout)
5. `full_scan.py` (≈4 min, 5 processes) → `full_scan_chunks.parquet`, `full_scan_gaps.parquet`
6. `python -c "import bsd_io; bsd_io.list_chunks(refresh=True)"` merges the exact counts into the index
7. `measure_postgap.py` → `postgap_model.csv`; `synthetic_slot_test.py` → `synthetic_slot_test.csv`; `check_ambiguous.py`
8. `full_scan_summary.py` → `dropouts_per_day_exact.csv`, `dropout_size_histogram_exact.csv`
9. `mains_check_v2.py` → `mains_check_v2.csv` (`mains_check.py` is v1, superseded)
10. `coverage_and_figures.py` → `coverage_per_day.csv`, fig1 to fig6
11. `python bsd_io.py` → self-test, 46/46 (`selftest_stdout.txt`)

## 10. Recommendations

1. **Load through `bsd_io` only.** `load_chunk` now defaults to fs_local. `load_span` stitches exactly at joins, returns `datetime64` times, reports every gap including any samples dropped at joins, and returns a sample-weighted fs. **Other workstreams must use `fs_local`**, never 2400 or Time/1e6.
2. **Dropouts.** Never interpolate across gaps of 5 or more samples for features.
   - Split the stream at those gaps, or mask any window that overlaps one, and keep a `has_dropout` flag per event.
   - Fill 1–4-sample gaps linearly only where needed.
   - Quantified confound: a 1 s event window contains a dropout with probability ≈1 − exp(−1.04/2.4) ≈ **35 %** in run 0204 from Feb 6 on, but only ≈**1 %** in runs 0218 and 0224.
   - For contrastive learning or federated splits, drop or down-weight windows with dropouts, and check that embeddings cannot predict the run from dropout artefacts.
3. **Rate-dependent processing** (50 Hz notch, PSD bins, durations): use the per-chunk fs_local. A constant 2416 Hz leaves up to a 0.35 % error; 2400 leaves 0.7 %.
4. **Units**: counts minus a rolling baseline (10–60 s median). No physical units.
5. **Exclusions**: the first ~60 min of run 0212 from normalisation statistics. Treat run 0204 from Feb 6 onward as "degraded" and keep it, but evaluate separately.
6. **Absolute time** is good to about ±0.35 s rms (±1.8 s max). That is fine for time of day, but too coarse to align with any external event log below a second.

## 11. Files

**Code**
- `bsd_io.py`
- Scripts from §9

**Data**
- `chunk_index.parquet`
- `full_scan_chunks.parquet`, `full_scan_gaps.parquet`
- `dropouts_per_day_exact.csv`, `dropout_size_histogram_exact.csv`
- `coverage_per_day.csv`
- `mains_check_v2.csv`, `mains_check.csv`
- `synthetic_slot_test.csv`
- `postgap_model.csv`
- Earlier parquet outputs

**Logs**
- All `*_stdout.txt` files

**Figures**
- `fig1_clocks.png`
- `fig2_diurnal_clock.png`
- `fig3_dropouts_coverage.png` (exact counts)
- `fig4_dropout_example.png`
- `fig5_baseline.png`
- `fig6_largest_event.png`
