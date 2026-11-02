# Workstream A: Data integrity and I/O

_Saved by the orchestrator from the agent's final hand-back. The subagent could not write report files itself._

## Summary (confidence in brackets)
- **F1: Values format.** Little-endian `<u4`, exactly 4n bytes, no header. The low 24 bits are used and the top byte is always 0. Each full chunk is 400000 B = 100000 samples. (very high)
- **F2: Time format.** `<u8`, exactly 8n bytes: device ticks counted from logger start (first sample at 4.5–16.1 s). Strictly monotonic within and across chunks, no wrap. (very high)
- **F3: Ticks are not µs.** 1.007–1.051 ticks per wall-clock µs, varying by run (≈1.05 in run 0212) and diurnally (~1 %). Never use `Time/1e6` as seconds. (very high)
- **F4: The ADC grid is uniform at about 2416 Hz, not 2400** (range 2407–2421, ±0.2 % daily drift).
  - Wall-clock fit: piecewise linear, knots every ~28 min, residual 0.35 s rms, max 1.8 s.
  - Independent mains check (60 chunks): 2416.8 ± 3.0 Hz, differing from the wall-clock fit by −0.08 ± 1.3 Hz (r = 0.90).
  - Assuming 2400 Hz puts mains at 49.67 Hz.
  - The device-tick rate is off by +39 ± 25 Hz. (high)
- **F5: Dropouts inside chunks.**
  - Sizes: mostly 104–105 samples (each ≈44.5 ms), often in bursts 12–40 samples apart, plus 1–5-sample losses.
  - Run 0204: 1.07 % lost, 13035 of 14563 chunks affected, worsening from ~0.1 %/day (Feb 4–5) to 1.7 %/day (Feb 11). Other runs lose ≈0.07–0.08 %.
  - Evidence the gaps are real: the value jump across a gap matches the lag-106 difference (965 vs 848; lag-1 is 111). Counting the missing slots flattens the rate against the wall clock (std 14.5 → 2.2 Hz; run 0204 residual goes from −483..+1064 s to −44..+38 s).
  - Largest gap is 45.1 ms, from 484 fully read chunks.
  - Confidence: high for rates, medium for the maximum gap.
- **F6: Contiguous chunks.**
  - Indices run 0..N−1, with no unpaired, duplicate or missing files. The brief's 14563 vs 14564 count difference was the Data Buffer file.
  - Consecutive chunks join exactly one sample apart, except 3 joins that fall on a dropout. Filename `Stop[i]` equals `Start[i+1]` (±1 s).
  - The only truncated file is run 0224 #6052 (640 samples). The run ends at 12:35:56, about 2 h before its folder name says.
  - The Data Buffer files hold the unflushed stream tail: run 0204 has 80896 samples (33.5 s), run 0218 has 46080 (19.5 s). Ignored. (very high)
- **F7: Clean values** (484 chunks).
  - Codes 11.19M–11.97M, far from the rails. No clipping, sentinels or spikes; at most 3 identical values in a row; low 4 bits uniform (χ² ≈ 15).
  - White noise σ ≈ 62 counts (≈18 effective bits).
  - 50 Hz hum of about 3k counts p-p.
  - Baseline ranges 11.46M–11.87M, with diurnal wander and steps.
  - Run 0212 settles over its first hour.
  - Largest event seen: 783k counts p-p (run 0212 #15). (high)
- **F8: Undeterminable.** Offset-binary vs two's-complement (the code never crosses 0x800000, so the two differ only by a constant), and volts, Tesla or differential polarity (no Vref given). Work in counts minus a local baseline. (high)
- **Config.** `Hardware.cfg` is identical in all runs: gain 8, 2400 Hz nominal, DIFFERENTIAL, D0 on. `Repository.cfg` says 100000 points per file.

## Methods
- `scan_headers.py`: names, sizes, and first/last sample of every file.
- `scan_time_edges.py`: 2×2000 timestamps per Time file. Gives the base tick interval and the missing-sample estimate, validated at −22..+41 error with median 0.
- `sample_chunks.py` (73 stratified chunks) and `sanity_scan.py` (384 random chunks plus the 100 worst): full reads.
- Slot reconstruction: each sample advances 1 slot. At a gap the jump is the median of the next 8 samples; the first post-gap timestamp is 0.3–0.7 slot early. The tick residual is then 0.02–0.03 slot rms, and the Δt lag-1 autocorrelation of −0.49 indicates white read-out jitter on a uniform grid.
- `mains_check.py`: independent rate check from the 50 Hz line.
- `coverage_and_figures.py`: coverage table and figures 1–6.
- `bsd_io.py` self-test: 38/38 pass.

## Coverage (hours; % samples dropped)
- **Run 0204:** 02-04 8.60 h (0.09 %); 02-05 to 02-10 each 24 h (0.13, 0.76, 1.26, 1.30, 1.24, 1.54 %); 02-11 16.50 h (1.70 %).
- **Run 0212:** 02-12 9.30 h (0.02 %); 02-13 24 h (0.03 %); 02-14 16.51 h (0.20 %).
- No data 02-15 to 02-17.
- **Run 0218:** 02-18 12.79 h (0.10 %); 02-19 24 h (0.06 %); 02-20 14.67 h (0.04 %).
- No data 02-21 to 02-23.
- **Run 0224:** 02-24 9.08 h (0.12 %); 02-25 and 02-26 24 h each (0.05, 0.04 %); 02-27 12.60 h (0.11 %).
- **Total 340.0 h:** 10 full and 8 partial days. Weekend data exists only on Feb 8–9.

See `coverage_per_day.csv`.

## Loader (`bsd_io.py`)
- `list_chunks(root)` returns a DataFrame and caches it to `chunk_index.parquet`. The key columns are the paired and truncated flags, the missing-sample estimate, the wall time of each chunk's first sample, and `fs_local`.
- `load_chunk(values_path, time_path)` returns time in seconds on a uniform sample grid with holes at dropouts, plus the values as float counts.
- `load_span(root, start, end)` stitches chunks exactly and returns wall-clock times (about 0.35 s absolute uncertainty) plus a gaps table.

## Anomalies
1. Ticks are not µs.
2. fs ≈ 2416 Hz, not 2400.
3. ~105-sample dropouts, heavy in run 0204 and worsening.
4. Run 0212 settles over its first hour.
5. Run 0224's last chunk is truncated and the logger idled 2 h.
6. The Data Buffer tails are ignored.
7. 50 Hz hum.
8. Baseline wander.
9. The first post-gap timestamp is early; the slot estimator handles it.

No files need to be excluded for format reasons.

## Recommendations
1. Load through `bsd_io`, and use the slot index or wall-clock mapping as the clock.
2. Use `fs_local` (or 2415 Hz) wherever rates matter.
3. Never interpolate across gaps of 5 or more samples. Split or mask windows that overlap one, keep a `has_dropout` flag per event, and fill only 1–4-sample gaps.
4. Work in counts minus a rolling baseline.
5. Exclude the first ~60 min of run 0212 from normalisation statistics.
6. Mark run 0204 after Feb 6 as degraded, and watch for representations learning run-specific artefacts (dropout rate, clock) in the federated splits.
7. Notch 50 Hz before event detection.
