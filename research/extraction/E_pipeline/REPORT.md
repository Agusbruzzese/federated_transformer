# Workstream E: vehicle-event extraction pipeline

_E agent, 2026-09-30. Implements `../SYNTHESIS.md` §3. Code: `extraction/` (repo root). Data: `data/events/` (gitignored). Scripts, tables, figures and logs: this folder. Every number below comes from a file listed in §8. Saved by the orchestrator from the agent's hand-back._

_**Orchestrator verification** (2026-09-30), all matching this report:_
- _re-ran `python -m extraction.tests`: 33/33 pass;_
- _`manifest.json` totals: 15 260 kept / 4 375 with SNR ≥ 20 / 3 534 with SNR ≥ 30 / 923 excluded / 1 truncated / 1 duplicate removed;_
- _recomputed from the meta parquets: the same counts, fs_eff 49.816–50.113 Hz, window arrays float32 [N, 1050] and finite;_
- _`data/events/` is 98 MB._

## 0. Summary

| | |
|---|---|
| **Output** | 15 260 events kept (14 241 with SNR ≥ 5, **4 375 with SNR ≥ 20**, **3 534 with SNR ≥ 30**) over 340.1 recorded hours. 923 excluded for mask_frac > 0.2 (873 in run 0204; 49 with SNR ≥ 20). 1 cross-block duplicate removed. |
| **Per event** | 1050-sample float32 window (≈ 21 s at the effective rate fs_local·25/1208 = 49.82–50.11 Hz), centred on C's energy centroid; [onset, offset] mask; dropout/padding mask; 19 morph scalars on the 21 s window (`m_*`) and on the central 402-sample (≈ 8 s) crop (`m8_*`); metadata. |
| **Runtime** | 20.3 min wall with 16 processes. 18 840 CPU-s user + 482 CPU-s system (≈ 55 CPU-s per recorded hour). ≈ 1 000× real time; 33 GB read; peak RSS 0.94 GB per worker. Deterministic: all output files of run 1 bit-identical across two independent runs. |
| **Tests** | `python -m extraction.tests`: 33/33 pass (`selftest_stdout.txt`). |
| **Rates vs C** | 44.9 / 12.9 / 10.4 per h (all / SNR ≥ 20 / SNR ≥ 30) vs C's 53.0 / 16.3 / 12.1. **Explained, not a pipeline error.** On C's own sampled windows E finds 48.9 / 13.6 / 10.8 per h. The same events get ≈ 8 % lower SNR with long context; C had sampled 4-min blocks. Standalone C with long context matches E event for event (§4e). |
| **Diurnal** | SNR ≥ 20 and ≥ 30 look like traffic: weekday peak 08–10 h, secondary 13–15 h, night ≤ 3/h, weekend flat and low. Night/peak ratio 0.013 (SNR ≥ 30), 0.076 (SNR ≥ 20). The SNR ≥ 5 tier is flat (≈ 20–30/h at night), consistent with C's finding that most low-SNR detections are not vehicles. |
| **Leakage (morph)** | SNR ≥ 20: date **0.024**, run 0.144 (0.036 without run 0204), 6-h bin 0.35, has_dropout 0.14 (null 0.13). SNR ≥ 30: date 0.010, run 0.105. **All-SNR: date 0.18, run 0.30**, caused by a run-0204 population of short blips associated with logger dropouts, at SNR < 20 (§4d). |
| **Cross-checks** | 2-h spans: 156 vs 157 and 103 vs 104 events; every SNR ≥ 20 event matched; median centroid difference 2–3 ms. Morph port vs D's `features.py` on the same windows agrees to within quantisation (§4e). |

**Recommendations for phase F**
1. Use the SNR ≥ 20 (or ≥ 30) tier, as pre-registered. Don't train on the all-SNR tier without a blip filter.
2. Use `m8_dur10`, not `m_dur10`, for the P1 `log_dur` target (§5, deviation 9).
3. Quote tier sizes from this report, not from C's sampled rates.

## 1. What was built

```
extraction/
  run.py       entry point: python -m extraction.run --out data/events [--procs 16]
  pipeline.py  per block: C detection -> B feature path -> windows, masks, morph, metadata
  featpath.py  B's chain on A's run-global slot grid (phase-exact resampling, LP at the true fs_local)
  blockio.py   stitched block loading (A's slots); SlotClock: slot -> chunk, fs_local, wall time
  morph.py     D's morph_feats ported to the true time base (changes E1-E4 in its docstring)
  params.py    all parameters (copied verbatim into manifest.json)
  tests/       python -m extraction.tests
  _vendor/     bsd_io.py (A), detector.py (C), preproc.py (B), features.py (D): verbatim copies with a
               provenance header and the original sha256. One edited line, marked E-PATCH: bsd_io INDEX_PATH
               points to A's folder.
```

**Blocks**
- Each run is processed in blocks of 60 core chunks (≈ 41 min) plus 3 context chunks (124 s ≥ 60 s) on each side.
- A block emits only events whose centroid lies in its core.
- A post-pass merges consecutive events from different blocks whose [onset, offset] overlap. It keeps the copy from the block that owns the midpoint of the two centroids.
- This found one duplicate in the whole dataset.

**Time base**
- Every sample carries A's exact run-global slot index.
- The feature grid is run-global: output sample j sits at slot j·1208/25. The effective rate is fs_local·25/1208 per chunk.
- Wall time = wall_start + (slot − slot_start)/fs_local, from A's clock fit (±0.35 s).
- `dur_s` and all morph time features use the event's `fs_eff`.

**Rule for windows spanning two fs_local values** (1.3 % of windows; largest spread 0.11 Hz, i.e. 0.22 %)
- Resampling happens in the slot domain, so sample placement is exact whatever the fs.
- A segment's low-pass filter is designed at its slot-weighted mean fs_local.
- `fs_eff` is the slot-weighted mean over the window; `fs_eff_min` and `fs_eff_max` are stored.
- Wall time uses the fs_local of the chunk containing the centroid.

**Outputs** (`data/events/`, 98 MB)

| File | Contents |
|---|---|
| `run{r}_{date}_meta.parquet` | one row per event; the column `row` indexes the arrays |
| `_windows.npy` | float32 [N, 1050] |
| `_evmask.npy` | bool: C's [onset, offset] |
| `_repmask.npy` | bool: bridged dropout hole, within 0.15 s of a split, or padding |
| `excluded.parquet` | the excluded events |
| `blocks.parquet` | per-block statistics |
| `manifest.json` | code identity, parameters, vendored sha256s, dataset fingerprint, counts, runtime, environment |

`manifest.json` holds the git HEAD plus a package sha256 (`5c0043…`), because the code is uncommitted. The dataset fingerprint is the chunk-index md5 `f568550e…`.

Metadata columns:
- identity and time: event_id, run, run_id, chunk, slot_centroid, slot_onset, slot_offset, wall_time, date, hour;
- detection: snr, peak_z, dur_s, has_dropout, mask_frac, win_mask_frac, n_neighbours_in_window;
- rate: fs_eff, fs_eff_min, fs_eff_max, fs_local;
- window geometry: truncated, n_outside, win_start_j, centre_offset_samples, ev_i0, ev_i1, near_run_edge;
- tiers: snr_ge20, snr_ge30;
- features: `m_*` ×19 and `m8_*` ×19.

## 2. Design check: how to repair dropouts (`dropout_repair_test.py`)

**Question.** B's `to50` calls `resample_poly` with its default zero padding, and B's §4 recommends masking ±0.15 s around each split. Are these the best repair choices?

**Method.** 20 real high-SNR events from runs 0218 and 0224 (SNR 35–976). A 106-sample dropout was deleted at 9 offsets (−4…+4 s from the centroid), and the feature series was compared with the intact one within ±6 s.

| Resample padding | Bridged | Median max\|err\|/σ | p90 /σ | Median /peak | p90 /peak |
|---|---|---|---|---|---|
| constant (B's default) | hole only | 4.3 | 24.1 | 0.7 % | 4.7 % |
| constant | ±0.15 s | 9.4 | 76.6 | 1.6 % | 17.1 % |
| **line (E)** | **hole only (E)** | **2.7** | **12.2** | **0.4 %** | **3.0 %** |
| line | ±0.15 s | 9.4 | 76.8 | 1.6 % | 17.2 % |

**Decision**
- Use `line` padding.
- Bridge only the missing samples.
- Keep the ±0.15 s zone as a **flag**, following B's definition. It feeds `mask_frac` and `repmask` but is not bridged.

## 3. Tests (33 checks, all pass)

| Area | What is checked |
|---|---|
| 24-bit loader and legacy regression | Values == uint32 & 0xFFFFFF, top byte 0, codes 11.5–11.6 M. On a chunk crossing an int16 wrap point (65536k + 32768), the legacy `'<i2'[0::2]` reader jumps 65 487 counts, while the loader's largest step is 3 060. The legacy bug is reproduced bit-exactly. |
| Effective fs | `resample_to_grid(slot 0, constant)` equals scipy's `resample_poly(25, 1208)`. Phase-exact placement at arbitrary starts (error 7.5e-4 on a 1.3 Hz sine). A sine's period is recovered from `fs_eff` at fs_local 2407.1, 2416 and 2421.4 Hz. The slot-weighted fs rule and the wall-time formula are checked. All `fs_eff` values lie in 49.82–50.11 Hz. Morph FWHM, t_pp and t_spread come out in true seconds. D's smoother, fixed for 100 Hz, applied at 50 Hz under-reads a 0.19 s pulse by Δlog_amp −0.10 (this documents change E2). |
| Synthetic injection | Pulses of 15 k and 3 k counts at 3 sites in real run-0212 background, one site on a chunk boundary. All 6 recovered exactly once. Centroid error ≤ 0.11 s; window-centre error ≤ 0.105 s (spec: ±0.5 s); peak within 7 %. |
| Block overlaps | 6 blocks with 10-chunk cores (5 internal boundaries) vs one 60-chunk block: no overlapping events after dedupe; 31 vs 30 events; all SNR ≥ 20 events matched. The one unmatched event has SNR 16.4, a context effect. The dedupe rule is also unit-checked. |
| Dropout masking (run 0204, chunk 9011) | Splits equal A's exact count of ≥ 5-sample dropouts (7). Fills equal A's count of 1–4-sample dropouts (3). Every sample inside a hole is bridged. ±0.15 s is flagged, and nothing beyond 0.2 s is. The output is finite. C's `has_dropout` is set on every detection that contains a dropout. |
| Window and stream edge | Shape (1050,), float32, finite; bool masks. An event 4 s after the run start is `truncated`, and its zero padding is flagged in `repmask`. |

## 4. Validation

### a. Counts (`tab_tiers.csv`, `tab_runs.csv`, `tab_days.csv`, `tab_hourly.csv`)

| Tier | n | Per h | C's estimate | has_dropout | Share 07–09 h | Share by run 0204 / 0212 / 0218 / 0224 |
|---|---|---|---|---|---|---|
| all detections | 15 260 | 44.9 | 53.0 | 17 % | 12 % | 41 / 18 / 18 / 23 % |
| SNR ≥ 5 | 14 241 | 41.9 | – | 18 % | 12 % | 42 / 17 / 18 / 22 % |
| SNR ≥ 20 | 4 375 | 12.9 | 16.3 | 36 % | 20 % | 41 / 16 / 19 / 24 % |
| SNR ≥ 30 | 3 534 | 10.4 | 12.1 | 38 % | 23 % | 41 / 15 / 19 / 25 % |

- The run shares match C's (42/17/17/24 % at SNR ≥ 20).
- SNR ≥ 20 events per hour, by run: 0204 10.1, 0212 16.8, 0218 16.3, 0224 15.2. Run 0204 is lower because it holds the weekend (Feb 8: 1.9/h, Feb 9: 1.5/h) and most of the excluded events.
- Counting the excluded events too: 47.6 / 13.0 / 10.5 per h.

**Why the rates are below C's** (`rate_vs_C_sample.py`)
- C's figures come from 1 227 six-chunk blocks (≈ 4 min each) with 30 s trimmed at each end.
- On exactly those windows E finds 48.9 / 13.6 / 10.8 per h.
- For matched events, the median E/C SNR ratio is 0.92 at SNR ≥ 20 and 0.93 at SNR ≥ 30. Only 85 % of C's matched SNR ≥ 20 events stay above 20 in E.
- Standalone C, given the ≥ 60 s context it requires, agrees with E (§4e).
- **Conclusion:** short-block SNRs run about 8 % high, so C's tier rates are overestimates. Tier membership depends on how σ is estimated, and phase F should know it.

### b. Diurnal curve (`fig_diurnal.png`, `fig_days.png`)

Weekdays, SNR ≥ 20 and ≥ 30:
- a sharp peak at 08–10 h (51–95/h at SNR ≥ 20);
- a dip at 12 h, then a second hump at 13–15 h;
- 0.1–1/h at SNR ≥ 30 between 00 and 06 h.

The weekend (Feb 8–9) is flat at 1–4/h. This is consistent with commuter traffic. The SNR ≥ 5 tier stays at 17–30/h through the night, so it is dominated by non-vehicle detections.

### c. Gallery (`fig_gallery.png`, `gallery_events.csv`): what the agent saw

30 random SNR ≥ 20 windows, stratified by run × 6-h bin (seed 0).

- **Centring:** correct in every window. The dominant bipolar or tripolar signature sits at t = 0, inside C's [onset, offset]. Neighbours appear only at the edges, where `n_neighbours_in_window` > 0.
- **Truncation:** none in the gallery; one event in the whole dataset.
- **Artefact in run 0204:** windows such as `20250204_01246744037` and `…01296041702` show blips of 0.5–1.1 k counts, about 0.2 s long, **at the dropouts**. `dropout_spike_check.py` (`fig_dropout_spikes.png`) shows these are in the raw data, not made by the repair: a 20 ms moving average of the raw codes, which does no filtering across the gap, shows the same blips, with a median excursion of 0.5–0.6 k counts. This is B's "blips associated with logger dropout load". They are flagged only within ±0.15 s.
- **Low-SNR members of the SNR ≥ 20 tier:** telegraph-like steps (`20250212_00324887600`), a single spike on a periodic pulser (`20250204_00333487568`), and barely-above-noise events (`20250224_00524851388`, `…00299687749`). These are C's known failure modes, not extraction errors. They are why the tier needs human labels.

### d. Morph leakage (`tab_leakage.csv`)

- **Model:** HistGradientBoosting (200 iterations), 5-fold GroupKFold, grouped by 30-min block (by day for has_dropout).
- **Metric:** chance-corrected balanced accuracy (skill).
- **Null:** labels shuffled at the group level (within day for has_dropout).

| Features / subset | n | date | run | 6-h bin | has_dropout (null) |
|---|---|---|---|---|---|
| `m_`, all | 15 260 | **0.184** | **0.303** | 0.334 | 0.083 (0.007) |
| `m_`, SNR ≥ 20 | 4 375 | **0.024** | 0.144 | 0.347 | 0.142 (0.129) |
| `m_`, SNR ≥ 30 | 3 534 | 0.010 | 0.105 | 0.124 | 0.075 (0.094) |
| `m_`, SNR ≥ 20, events without has_dropout | 2 808 | 0.007 | 0.090 | 0.347 | – |
| `m_`, SNR ≥ 20, runs 0212–0224 only | 2 575 | 0.012 | 0.036 | – | – |
| `m8_`, SNR ≥ 20 | 4 375 | 0.018 | 0.122 | 0.322 | 0.120 (0.086) |

All date, run and hour nulls are within ±0.04.

**Reading**
- **Training tiers (SNR ≥ 20 and ≥ 30):** date skill is about 0, as D found.
- **Run skill:** it comes from run 0204. At SNR ≥ 20, the 21 s `m_dur10` has median 8.1 s in run 0204 vs 4.6–6.4 s elsewhere. The long window picks up the blips. `m8_dur10` is flat across runs (3.1–3.4 s).
- **has_dropout:** skill equals its null. Features that identify run 0204 also predict each day's dropout prevalence, so there is no evidence that morph "sees" individual dropouts.
- **Hour bin:** skill tracks the genuine hourly traffic mix.
- **Investigated: why the all-SNR tier leaks** (`leakage_investigate.py`, `tab_leakage_investigate.csv`, `tab_leakage_lobe_by_run.csv`):
  - It is not the noise level: σ alone gives run skill 0.02, SNR alone 0.007.
  - It lives at SNR < 20: run skill is 0.35 in both the < 10 and 10–20 bands, vs 0.14 at ≥ 20.
  - It is carried by main-lobe timing: single-feature run skill 0.15–0.17 for `dur50_main`, `rise_main` and `fall_main`.
  - Run 0204's low-SNR events are mostly short: FWHM ≤ 0.2 s in 68 % of them at SNR < 10 and 46 % at SNR 10–20, vs 13–24 % in the other runs. These are the same blips as in §4c.
  - Date skill within a single run is 0.28–0.31, so the low-SNR non-vehicle population also varies day to day.
  - **Conclusion:** a detection-population confound, not a pipeline defect.

### e. Cross-checks

**(e1) Standalone C vs the pipeline** (`crosscheck_span.json`). C's original `detector.py` was run on C's own I/O (device ticks) with 3 context chunks, and its events were mapped to A's slots.

| Span | Block boundaries inside | C standalone | Pipeline (incl. excluded) | Matched < 0.5 s | SNR ≥ 20: C / pipeline / matched | Median \|Δcentroid\| | Median \|ΔSNR\|/SNR |
|---|---|---|---|---|---|---|---|
| 0218, Feb 19, 07–09 h | 3 | 156 | 157 | 153 | 88 / 88 / 88 | 1.7 ms | 4.6 % |
| 0204, Feb 6, 16–18 h (dropout-dense) | 3 | 103 | 104 | 98 | 20 / 22 / 20 | 3.3 ms | 2.7 % |

All unmatched events have SNR < 17 on both sides.

**(e2) Morph port vs D's `features.py`** on 5 SNR ≥ 20 windows (`crosscheck_morph.csv`).
- Method: D's function hard-codes a 100 Hz smoother, so each E window was upsampled ×2 and D was called with fs = 2·fs_eff (this is the fs correction). D's log_energy was compared after subtracting log10(2·fs_eff).
- Amplitude, area and energy features agree to ≤ 0.005. Time features agree to within one sample.
- One event differs more (`20250212_00182484465`: neg_frac 0.12, kurt_env 0.17, t_pp 0.06 s). There, a near-tie between two extrema resolves differently at the two sampling rates.
- The stored `m_*` values equal a fresh recomputation.

## 5. Deviations

1. **C's detector gets A's exact slot index as `t`**, instead of device ticks. C's dropout re-insertion therefore uses A's validated slots, and its times (slot/2416) map back to slots exactly. `detector.py` itself is unchanged, so its internal constants remain ≤ 0.3 % off in true time. All outputs are converted via fs_local. Effect: §4e1.
2. **B's low-pass is designed at the segment's fs_local**, not at 2416 Hz.
3. **Phase-exact resampling onto a run-global grid.** Same filter as `resample_poly`, via `upfirdn`. B rounded each segment start to 20 ms, which could misalign segments across a dropout by up to one sample. Identical to `resample_poly` when the phases coincide (tested).
4. **`line` padding in the resampler instead of zero padding** (§2).
5. **The ±0.15 s gap guard is flagged, not bridged** (§2).
6. **The pass-2 baseline event mask** is the union of B's pass-1 5σ detections and C's detections, each ±2 s. B ran pass 1 on its 5 Hz detection path; here it runs on the 8 Hz feature path.
7. **`mask_frac` is computed over C's [onset, offset]**, using the flag mask (holes plus ±0.15 s). 923 events were excluded (median mask_frac 0.42) and saved.
8. **Morph port changes:**
   - E1: the true fs is used;
   - E2: the 5 Hz smoother is designed at the true fs;
   - E3: `log_energy` = log10(Σx²/fs + 1), a time integral, **≈ 2.003 below D's value**;
   - E4: a guard for the case where no sample is above the threshold.
9. **Morph on the 21 s window, as specified, differs from D's validation.** D used 8 s, 100 Hz windows. On 21 s, `m_dur10` degenerates for SNR < 20 (median ≈ 20.8 s, i.e. the whole window) and absorbs neighbours and blips. **Added beyond the spec:** `m8_*`, the same 19 features on the central ~8 s. **Recommendation:** F's `log_dur` should use `m8_dur10`. This needs an orchestrator decision, because the pre-registration says only "dur10".
10. **The window is a fixed 1050 samples**, so its true length is 20.95–21.08 s. `win_start_j` and `fs_eff` give exact times.
11. **Windows spanning two fs_local values** (1.3 %) use the slot-weighted `fs_eff`, with min and max stored.
12. **1 019 detections have C's snr < 5 and are kept** ("keep all detections"). C triggers on its high-passed channel but measures SNR on the waveform channel. The SNR ≥ 5 tier is a column filter.
13. **Truncated windows are zero-padded** (zero is the baseline), and the padding is flagged in `repmask`.
14. **The leakage null permutes labels per CV group** (per day for has_dropout), not per event.
15. **Code identity:** the repo has only its initial commit and `extraction/` is not committed. The manifest records git HEAD plus a package sha256.
16. **C's expected rates were not reproduced**, for the documented reason (§4a).
17. **Tests are plain `python -m extraction.tests` asserts**, because pytest is not installed.

## 6. Limitations

- Precision still rests on no human labels. The SNR ≥ 20 tier visibly contains non-vehicles.
- Tier boundaries are sensitive to the σ context (≈ 8 % SNR shift).
- Run 0204's blips remain in the windows, flagged only within ±0.15 s of dropouts. This is a route to run identity for an encoder, so P2 should report run 0204 separately.
- Wall time is accurate to ±0.35 s.

## 7. Reproduction

```
.venv/bin/python -m extraction.tests
.venv/bin/python -m extraction.run --out data/events --procs 16        # ~20 min
cd research/extraction/E_pipeline
python validate.py counts|gallery|leakage|crosscheck
python rate_vs_C_sample.py; python leakage_investigate.py
python dropout_repair_test.py; python dropout_spike_check.py
```

## 8. Files (this folder)

| File | Purpose |
|---|---|
| `validate.py` | validation (a)–(e) |
| `rate_vs_C_sample.py` | rate discrepancy vs C |
| `leakage_investigate.py` | all-tier leakage diagnosis |
| `dropout_repair_test.py` | repair design check |
| `dropout_spike_check.py` | run-0204 blips are in the raw data |
| `fig_diurnal.png`, `fig_days.png`, `fig_gallery.png`, `fig_dropout_spikes.png` | figures |
| `tab_*.csv`, `crosscheck_span.json`, `crosscheck_morph.csv`, `gallery_events.csv`, `dropout_repair_test.csv`, `rate_vs_C_sample.csv` | tables |
| `*_stdout.txt`, `run_stdout.txt`, `selftest_stdout.txt` | logs |
