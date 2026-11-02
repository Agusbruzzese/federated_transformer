# Addendum 1 to PREREG_central.md (v2, sha256 prefix d322a199b4b88aa5)

_Orchestrator, 2026-09-30. Written **after extraction (E) and before any encoder was trained or evaluated**.
No P1–P5 outcome was known when this was written. The frozen v2 file is unchanged. These are pre-training
deviations, prompted by E's report (deviation 9) and its audit (issue 4)._

**A1. P1 targets come from the central ~8 s crop (`m8_*`), not the 21 s window.**
- **Evidence** (`E_pipeline/tab_dur10_by_run_orchestrator.csv`, SNR ≥ 20): the 21 s `m_dur10` median varies by run,
  8.08 s in run 0204 vs 4.61–6.40 s elsewhere. That tracks the run-0204 dropout blips, a known artefact.
  `m8_dur10` is flat, 3.10–3.38 s.
- **Consistency:** D validated its morphology on 8 s windows, and the encoder sees a 12 s crop centred near the
  event. Using the 21 s window would let neighbours and blips outside the encoder's view define the targets.
- **Targets:**
  - `log_amp` = `m8_log_amp`;
  - `log_dur` = log10(`m8_dur10`);
  - `polarity` = `m8_polarity`;
  - `n_lobes` = `m8_n_lobes`, capped at 4.
- The 21 s (`m_*`) versions are reported as a secondary analysis.
- **Limitation:** below SNR 20, `m8_dur10` also saturates (median 7.94 s ≈ the whole crop). For the all-SNR
  secondary tier, `log_dur` is therefore reported but flagged as degenerate.

**A2. The tier is defined by E's `snr` column** (long-context σ, ≥ 60 s). Primary training tier: `snr ≥ 20`,
which is **4 375 events** in total before the split. C's short-block SNR runs about 8 % high (E §4a), so C's
tier sizes do not apply.

**A3. P2 reports run 0204 separately**, as E §6 recommends: it has blips at dropouts in the raw data, which
are a route to run identity.
