# Data intake kit

Takes a **new recording folder** through the validated pipeline and the pre-registered external test of the frozen study models, with one command. Design, validation and frozen hashes: `reports/50_*`.

```bash
.venv/bin/python -m intake.run --raw "/path/to/new_recordings" --name <short_name> [--procs 16]
```

## Input

The `--raw` folder must contain run folders exactly like the reference dataset:

```
Run YYYYMMDD HHhMMmSSs - YYYYMMDD HHhMMmSSs/
    S0C0 BIN Values - 000001 Start@ YYYYMMDD - HHhMMmSSs Stop@ ... .bsd
    S0C0 BIN Time   - 000001 Start@ ... .bsd
```

Use the same sensor format: 24-bit values in `<u4`, device ticks in `<u8`, about 2.4 kHz.

## Output, under `data/intake/<name>/` (never overwritten; rerunning resumes)

| Stage | What | Output |
|---|---|---|
| 1 integrity | chunk index + full scan of every file (workstream A code): pairing, 24-bit values, sampling rate 2400–2430 Hz, monotonic timestamps, dropouts | `A_index/`, `.done_integrity` |
| 2 extract | the validated extraction (`extraction/`, unchanged) | `events/` (same format as `data/events/`) |
| 3 quality | events per day, SNR tiers, dropout share, morphology drift vs the reference (two-sample KS, flag above 0.15), usable days (at least 30 events with SNR ≥ 20) | `.done_quality` |
| 4 external | frozen study models (central C256, federated F-nat, server-pretrained S+F; 10 seeds; **never trained on the new data**) scored on the new SNR ≥ 20 events. Probes are cross-fitted by day (alternating halves). The study's margins and rules apply, with validity controls C1 (independent central models equivalent) and C3 (untrained network inferior). | `external_scores.parquet`, `.done_external` |
| 5 report | plain-text summary + full JSON | `INTAKE_REPORT.txt`, `intake_<time>.json` |

## Reading the result

- **Read the external test only if it says `procedure VALID`.**
- **Comparisons and what the study found:**
  - federated vs central: the study found EQUIVALENT;
  - pretraining vs federated: the study found HELPS.
- **With fewer than about 6 usable days the intervals are wide.** Expect INCONCLUSIVE; it means "not enough data", not "refuted".
- **External test vs full replication.** The external test only evaluates models trained on the reference data. A full replication, retraining on the new data with the pre-registered pipeline of reports 41–48, is the next step if the new data are large enough: at least 14 days, so that 10 day-clients plus evaluation days exist.
- **Labels.** Score with SNR ≥ 20 events only. Low-SNR labels compress differences (reports 44–46); the kit enforces this.

## Self-test

```bash
.venv/bin/python -m intake.run --name selftest_<label> --selftest 20250224
```

This links one reference run and checks that the extraction reproduces `data/events` for that run. The external test on reference data is in-sample and is marked "NOT evidence".
