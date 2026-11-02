# H results: SimCLR, train runs 0204/0212/0224 → test run 0218 (SNR ≥ 20, 837 test events)

Scoring was fixed in PLAN.md (sha256 prefix 835b789c1e0af032) before training. The dry run (`dryrun.json`) passed:
- planted signal: morph8 polarity 1.00;
- noise null: ≈ 0;
- planted leak: L1 = 1.00.

All 10 trainings converged: CNN final loss 2.42–2.47 (T 0.2), transformer 1.52–1.62 (T 0.1).

| Score (test run) | CNN trained | CNN untrained | CNN gain [CI] | TF trained | TF untrained | TF gain [CI] |
|---|---|---|---|---|---|---|
| S1 polarity (skill) | 0.838 | 0.791 | +0.046 [+0.003, +0.090] | 0.830 | 0.577 | **+0.253** [+0.164, +0.342] |
| S2 lobe count (skill) | 0.490 | 0.341 | +0.149 [+0.091, +0.206] | 0.437 | 0.239 | **+0.197** [+0.168, +0.227] |
| S3 duration (R²) | 0.735 | 0.775 | **−0.040** [−0.076, −0.003] | 0.787 | 0.660 | +0.128 [+0.089, +0.166] |
| S4 magnitude group (skill) | 0.844 | 0.773 | +0.070 [+0.053, +0.088] | 0.854 | 0.521 | **+0.333** [+0.275, +0.391] |
| S5 4 groups (skill) | 0.803 | 0.745 | +0.057 [+0.051, +0.063] | 0.814 | 0.464 | **+0.350** [+0.284, +0.416] |
| L1 is-0204 leakage (skill; limit untrained + 0.05) | 0.140 | 0.060 | +0.080 ✗ | 0.099 | 0.029 | +0.070 ✗ |
| L2 date within the test run (skill; limit 0.03) | −0.087 | – | ✓ | −0.138 | – | ✓ |

**Reference ceiling** (hand-made morph8 features, which contain the targets): polarity 1.00, lobes 0.73, duration 0.91, magnitude 0.91, 4-group 0.93.

**By the plan's rules**
- **Learned something:** CNN 4/5 scores (duration worse), transformer 5/5.
- **Leakage OK:** **no** for both. Run-0204 identifiability rises 0.07–0.08 above untrained, just over the 0.05 limit. Absolute skill is low (0.10–0.14), and date leakage is ≈ 0.

**Deviation:** the plan said block-bootstrap CIs. The CIs above are a seed-level t-interval (5 seeds, df 4), which is simpler and reflects the training randomness only. Per-seed numbers are in `scores_per_seed.csv`.

**Caveat (stated in the plan):** S4 and S5 come from amplitude, but the encoder sees max-normalised input. It recovers magnitude indirectly, via shape and the relative noise floor.
