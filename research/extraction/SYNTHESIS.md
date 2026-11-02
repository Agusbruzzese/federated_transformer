# Extraction re-design: synthesis

_Orchestrator, 2026-09-30. Sources: workstream reports A–D (`*/REPORT.md`), their audits (`*/AUDIT_*.md`),
`PROTOCOL.md` and `ORCHESTRATION_LOG.md`. **Status: proposal awaiting the user's approval.** Nothing in the
project's `client/` or `server/` code has been changed._

## 1. Headline findings

1. **The previous pipeline's input was corrupt.** `comparison/central_pipeline.py` (commit 6d63a85, line 94)
   reads the 24-bit sensor codes as the low 16 bits, as signed int16. The signal therefore jumps by 65 536
   counts in about 10 % of chunks and about 41 % of event windows. [very high; D1, reproduced independently by the
   orchestrator on 60 chunks]
   - **Consequence (inferred):** earlier results built on the 70 legacy features are **untrustworthy**, including
     the federation runs in `federation_run*.log` (loss 0.48 → 0.35). About 41 % of event windows are
     corrupted, and `t_hours` leaks date (item 2). The size of the damage was not quantified, because the
     pipeline is being rebuilt anyway.
2. **The legacy features also encode time, not vehicles.** `t_hours` is cumulative hours since Feb 4, so it
   predicts the run with skill 1.00 and the date with 0.98. Removing it still leaves date skill 0.96 on the
   holdout, because of the wrapped amplitudes. [very high; D2]
3. **The file format differs from what the code assumed** [very high; A]:
   - The sampling rate is **≈2416 Hz**, not 2400 (per chunk 2407–2421; use `fs_local`).
   - The Time file holds **device ticks, not µs**.
   - There are **188k dropouts** of about 105 samples, mostly in run 0204 from Feb 6.
4. **EMD/HHT is unsuitable** for this signal. [high; D3]
   - On raw data, IMFs 1–5 are mains and noise.
   - The vehicle IMF index wanders between events.
   - The IMF count changes in 34–51 % of events under a 0.1 s shift or added noise.
5. **Vehicles are slow, low-frequency signatures.**
   - Large events peak at about 20k counts, mostly bipolar, and last 2–12 s.
   - 99 % of their energy lies below 2.5 Hz.
   - 50 Hz mains (about 1.4k counts) dominates the raw signal.
   - [high; B, D agree]
6. **The old fixed-window "DC-relative" detection misses almost everything.** Its reconstruction scores
   F1 0.006 on injections. Dividing by the baseline is pointless (the divisor varies by only ±1.6 %);
   subtracting it is essential. [high for subtract vs divide (B, D); the legacy F1 is illustrative, since it
   comes from a reconstruction (C)]

## 2. Cross-workstream reconciliation

| Question | A | B | C | D | Decision |
|---|---|---|---|---|---|
| Sampling rate | 2416 ± 3 Hz; fs_local per chunk | 2416.3 | 2414–2418 | used 2400 (−0.69 % deviation, flagged) | **fs_local from A's index.** D's time features must be rescaled when ported. |
| Clock | ticks ≠ µs; slot index | slot index | sample clock | nominal | **Slot index × 1/fs_local.** Wall time from A's fit (±0.35 s). |
| Dropouts | fill 1–4 samples, split at ≥5, flag | same, mask ±0.15 s | mask 50 Hz samples whose FIR support touches a gap | ignored (flagged) | **A/B rule + C's masking.** Keep `has_dropout` and `mask_frac`. **Repair events with mask_frac ≤ 0.2**, don't drop them (B, C). |
| Mains | notch 50 Hz | regression, 50 Hz + 20 harmonics | FIR LP, no regression | LP 20 Hz | **LP ≤ 8 Hz makes a notch unnecessary** (≥100 dB). B's regression is kept only where a raw-bandwidth path is needed. |
| Detection front end | – | LP 5 Hz, 50 Hz, 2-pass 20 s median | FIR→50.33 Hz, 0.5 s median, 2-pass 60 s median, 4 s HP | provisional | **C's `detector.py`** (HYST‑BP, 5σ/2σ). On injections it beats a fair fixed-window method by ΔF1 +0.16 on a *weakly* held-out Run 3 (touched once during exploration), and the ranking is robust to independent templates. Real-data precision and recall are not yet measured with human labels. |
| Feature front end | – | LP 8 Hz, resample to exactly 50 Hz, 2-pass 20 s median | – | LP sweep 5–20 Hz: morph and waveform insensitive | **B's chain with LP 8 Hz** (D's sweep backs up that the choice is safe). |
| Window | – | 12–15 s | rule gives **21 s**; 12 s is core-only | used 8 s; 12 vs 21 not evaluated | **Extract 21 s centred on the energy centroid, with the [onset, offset] mask.** This is a superset: 8 or 12 s crops are derived later, so the choice waits for the model phase. |
| Training tier | – | large events above the ~5k valley | SNR ≥ 30 clean (4.1k events, 48 % at 07–09 h) or SNR ≥ 20 (5.5k) | both tiers stable, top-2 identical | **Provisional.** Extraction keeps **every** detection (SNR ≥ 5) with its SNR, so no tier is baked in. SNR ≥ 20 is the *provisional* default (less morning bias than ≥ 30, +35 % events). The final threshold is gated on human labels (§5). |
| Representation | – | – | – | morph top-2 in all 13 settings; waveform top-2 on high-SNR tiers but untested with an encoder | **Extract both:** 19 morphology scalars (on the true fs) and a fixed-length amplitude-normalised waveform. The encoder choice moves to the model phase. |
| Excluded | – | divide by DC | legacy DC-relative | t_hours, absolute levels, EMD/HHT, spectral > 5 Hz | **Excluded as inputs.** `has_dropout`, run, date and hour are **metadata only**, never inputs. |
| Normalisation | – | subtract baseline | – | global fixed transforms; per-client z-scoring makes the client identifiable (0.02 → 0.73) | **Fixed physical transforms** (log amplitude/energy; seconds on the true clock). Per-event max-normalisation for the waveform. No per-client scaling. |

**Real disagreements, and how they were resolved:**
- **Detection front end, B's chain vs C's.** C benchmarked B's detector (BDET) and HYST‑BP won by +0.13 F1,
  so C's is used for detection.
- **Window length, B 12–15 s vs C 21 s vs D 8 s.** Resolved by extracting a superset: 21 s with a mask.
- **Training tier, C's clean SNR ≥ 30 vs broader.** Resolved in favour of SNR ≥ 20 on D's stability evidence
  and C's bias analysis. Both tiers are tagged, so either can be selected later.

## 3. Proposed extraction pipeline (to be implemented after approval)

```
for each run: stitch chunks (A: bsd_io.load_span), keep ≥60 s context
  ├─ detection path:  C detector.detect_stream(x, t)
  │     → events with onset, offset, centroid, SNR, has_dropout, mask_frac, n_neighbours
  └─ feature path (B preproc): fill 1–4 / split ≥5 at dropouts → LP 8 Hz (4th-order, zero-phase,
        even padding, per segment) → resample_poly(25,1208) → 2-pass 20 s median, subtracted
        NOTE: 25/1208 gives exactly 50 Hz only when fs = 2416. Record the effective rate
        fs_local·25/1208 (49.8–50.1 Hz) per segment and use it for every time feature.
per event: 21 s window centred on the energy centroid (1050 samples at 50 Hz) + [onset, offset] mask
  outputs: waveform window (float32), 19 morph scalars (true fs), metadata
          (run, wall time, SNR, tier, has_dropout, mask_frac, n_neighbours) — metadata never fed to the model
storage: one Parquet/NPZ per client split, plus a manifest (dataset md5, code version, parameters)
```

Cost: detection is measured by C at about 2 CPU-s per hour of data. The feature path is **not yet measured**,
and the full pipeline must be timed on a stratified sample before any cost is quoted.

**Mandatory validation after implementation** (the audit pointed out that D's results were **not recomputed**
on this pipeline):
1. Recompute D's leakage (date, run, hour, `has_dropout`) and invariance on the *actual* pipeline outputs.
   Use C's events, B's chain and fs_local; group the CV by day; include repaired-dropout events and report
   them separately.
2. Use correlation-preserving nulls for the waveform (e.g. shift- or phase-randomised).
3. Build a second holdout that was never touched: one weekend span (Feb 8–9) and one dropout-dense span.

## 4. Limitations carried forward

- **Precision rests on AI labels only.** No human has labelled any event. C's precision figures
  (0.96 clean tier, 0.40 raw) are provisional (§5).
- **True recall is unknown.** Injection templates are high-SNR real events scaled down, so recall on
  small, distant vehicles is optimistic.
- **The holdout (Run 3) has no weekend and no dense-dropout regime.** It was touched once during C's
  exploration. Weekday vs weekend is testable only on Feb 8–9.
- **Run 0204 is a confound for the federated split.** Its dropout density differs by 10–20×, and
  `has_dropout` alone predicts the run with skill 0.34. If clients are split by date (README), client
  identity and dropout artefacts are entangled. This must be checked in the model phase.
- **The representation ranking is formally inconclusive under PROTOCOL** (ρ 0.77 < 0.8). The audit also found
  the waveform's cluster advantage partly inflated by a weak null. Only "morph is top-2" is robust.
- **KMeans pseudo-labels vs SimCLR pairs.** D's evidence against KMeans is suggestive (weak structure over the
  null; clusters track dropouts), not conclusive. It is a model-phase decision, but the legacy KMeans design
  should not be reused as is.
- **Origins unknown:** the small (~1k count) events; the 0.73, 1.98 and 9.45 Hz tones; the slow baseline ramps.

## 5. Decisions needed from the user

1. **Approve the pipeline in §3.** It would be implemented as a module (e.g. `extraction/`) in the repo, reusing
   `bsd_io.py`, C's `detector.py` and B's `preproc.py`, with tests. Then all 14 days are run once.
2. **Human labels (a gate for thresholds, not optional for finalising them).** Extraction can run
   before this, because it emits every event with its SNR, but no tier or threshold is final until then. Label the 80-plot blind pack in
   `C_detection/figs/human_pack/` (about 20–30 min). Codes are in its README; fill in
   `C_detection/data/human_pack_labels.csv`. This turns the precision and tier thresholds from AI judgement
   into measured values.
3. **Next phase (model):** train an encoder on SimCLR augmentation pairs; compare waveform vs morphology
   vs both with grouped CV and correlation-preserving nulls; choose the window crop; then revisit the
   federated client split and the FedProx server. The old `client/` and `server/` code needs rewriting to
   match, because its 70-feature input no longer exists.

## 6. Audit points rejected or deferred by the orchestrator (for independent review)

| Workstream / audit issue | Orchestrator's decision | Evidence |
|---|---|---|
| A2 #2, "fs contradiction" | rejected | A slowly drifting uniform grid is consistent. Mains vs wall clock agree to −0.1 ± 1.2 Hz across 428 chunks (A §5.4). |
| A2 #5, "the join clamp hides overlaps" | checked, no issue | The unclamped join distribution over 29 413 joins has a minimum of +0.42 slot and no overlaps (LOG). |
| B #2 / B2 #9, "fs circular" | rejected | Same as A2 #2. The absolute rate uncertainty is ±0.1 %, and per-chunk variation is ±0.3 %, handled by fs_local. |
| B2 #6, "out_* files missing" | rejected | `B_signal/data/out_baseline.txt` and `out_v2_analysis.txt` exist. The auditor omitted them for size (>8 k chars), so this is a transmission gap, not a refutation of their content. |
| B2 #7, "filling 1–4-sample gaps" | rejected | A's recommendation, based on a full-scan size histogram (17 % of dropouts are 1–5 samples). |
| B2 #1, "8 Hz is circular" | deferred to D | D's LP sweep (5/8/10/20 Hz) shows morph and waveform are insensitive, and the top-2 is unchanged. |
| C #1–3 (REJECT) | partly accepted; C revised | The headline was moved to FIXEDLOC, templates were made independent, the pre-registered window rule was applied, and the human pack was built. |
| C2 #12, "repair function missing" | rejected | `detector.front_end` and `fill_gaps` mask and bridge the samples around dropouts. |
| D #2–4, #6, #13, #15 (ranking method) | deferred to the model phase | The pipeline extracts the superset (morph + waveform), so the ranking doesn't change what is extracted. It will be recomputed on the pipeline outputs with an encoder, grouped CV and correlation-preserving nulls (§3). |
| Orchestrator audit 1 #1, "serialise A first" | not acted on (cost) | **Lesson recorded:** the parallel start cost rework in B, C and D (the 2400 Hz and µs assumptions). The model phase will serialise its dependencies. |
