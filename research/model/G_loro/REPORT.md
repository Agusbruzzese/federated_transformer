# Workstream G: central SimCLR under PREREG v3 (leave-one-run-out, tightened)

_Saved by the orchestrator from the G agent's hand-back (the harness blocked the subagent from writing it). The complete agent text is preserved in the orchestrator's session; this file holds the verified essentials and points to the tables._

_**Orchestrator verification** (2026-10-01):_
- _PREREG v3 sha256 prefix `e6424ecd1c0a8abe` ✓._
- _`logs/hashes_at_freeze.txt` (11:06:47) and `hashes_at_end.txt` (14:03:01) differ only in their timestamp line._
- _Diagnostic D1 is present in the frozen `logs/src_at_freeze/config.py` (line 60), so it was declared before results._
- _`tab_P2_criteria_snr20.csv`, `tab_P2_within_space.csv` and `tab_P1_tests_snr20.csv` match the numbers below._

_**Orchestrator's admission:** P2a–c as written in v3 are **unpassable by design**._
- _Pooling out-of-fold embeddings from 4 different networks makes the network, and therefore the run, identifiable: an untrained pooled control already gives run skill 1.000._
- _This is the orchestrator's error in v3, not a property of the encoders._

## Verdicts (v3 wording, as pre-registered)

| | CNN | Transformer |
|---|---|---|
| P1 targets passing (Holm p < 0.05 and ≥ 3/4 runs positive) | 0/4 | 3/4 (amp_resid, polarity, n_lobes) |
| P2a date / P2b run excl. 0204 / P2c is-0204 / P2d has_dropout within 0204 | 0.154 ✗ / 1.000 ✗ / 1.000 ✗ / 0.118 ✓ | 0.139 ✗ / 0.999 ✗ / 0.998 ✗ / 0.099 ✓ |
| **Verdict** | **No evidence of learning** | **Learns but leaks** |

**CNN vs transformer.** By the rule, the CNN is preferred on pooled P1 mean, +0.015 [+0.008, +0.022]. In *gain over random-init* the transformer is clearly ahead: the CNN's random-init already scores 0.580, the transformer's 0.452.

## Pre-declared diagnostic D1 (descriptive)
- **P2, fold-init:** pooled untrained fold-specific networks give date 0.13 / 0.12 and run 1.000 / 1.000. So the P2a–c failures come from the pooling.
- **P2 within a single model's space** (40 models): date ≤ 0.021, run excl. 0204 ≤ 0.007. **No leakage.**
- **P1, SSL − fold-init** (pooling matched on both sides):
  - CNN: +0.059 / +0.011 (n.s.) / +0.047 / +0.140 (amp_resid / log_dur / polarity / n_lobes), so 3/4.
  - TF: +0.117 / +0.121 / +0.279 / +0.196, so 4/4.
- **The P1 pooling cost against single-network random-init** is −0.04 to −0.14 per target, so the pre-registered P1 test is biased against SSL.

## Other results
- **amp_resid sanity:** SNR alone gives R² ≈ 0 (linear) and 0.03 (kNN). On raw log_amp, SNR alone gives 0.846, which was the leak.
- **SSL vs SNR alone and vs PCA:** both SSL models beat SNR alone on all targets in all runs, and beat PCA-64.
- **vs morph8:** both stay below the hand-crafted morph8 (circular ceiling).
- **No-stretch CNN ablation:** log_dur Δ −0.000 [−0.008, +0.009]. **The duration hypothesis is NOT supported;** removing stretch only hurts the other targets.
- **P3 retrieval / P4 cross-seed Jaccard:** consistent with F (CNN 0.908 / 0.545; TF 0.826 / 0.524).
- **Selection:** margins ≤ 0.009, i.e. noise-level.
- **Tiers:** at SNR ≥ 30 the transformer passes 2/4 and the CNN 0/4. At all-SNR, has_dropout skill is ≈ 0.44 even for random-init (the blip population).

## Deviations
1. HGB date leakage was not run (time budget, after a 1.7 h network outage). P2b–d HGB was run with 10 permutations.
2. An extra selection assert was added before the freeze.
3. Two failed launches were re-run unchanged.
4. Choices J1–J17 were written before training.
5. D1 is an addition beyond v3, declared before results.
6. P5 was not run.
7. Training used 3 parallel GPU processes, so runs are not bit-reproducible.

## Limitations
- The pooled-OOF design flaw (above).
- Noise-level selection.
- Circular targets vs morph8.
- Run 0204 dominates the pooled metrics (1 800 / 4 375 events).
- No human labels.
- The CNN's duration loss is unexplained.

Runtime: 3.3 h end to end, about 1.6 h active (84 trainings, 112 GPU-process-min; P2 LR 12 min).

Tables: `tab_selection.csv`, `tab_P1_{values,tests,cnn_vs_tf}_*.csv`, `tab_P2_*.csv`, `tab_P2_within_space.csv`, `tab_P2_hgb_snr20.csv`, `tab_P3.csv`, `tab_P4.csv`, `results_verdict.json`. Code: `ssl_central_v3/`. Reproduce: `run_all.sh`.
