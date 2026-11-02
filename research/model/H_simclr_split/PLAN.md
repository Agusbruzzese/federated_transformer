# H: SimCLR central model, simple train/test split. Scoring fixed before running.

_Orchestrator, 2026-10-01. Written before any H model was trained. It applies the lessons from G and the critique:
single embedding space, comparison against the model's own untrained weights, no amplitude/SNR target, and
fixed hyperparameters._

## Split
- **Train:** runs 0204, 0212, 0224, minus Feb 26 (3 172 events).
- **Test:** run 0218 (837 events), never used for training or any choice. Tier SNR ≥ 20.
- Feb 26 is unused, so the code is the same as phase F's.

## Models
- SimCLR, existing `ssl_central` code, 100 epochs.
- Fixed hyperparameters from phase F:
  - CNN: lr 3e-4, T 0.2;
  - transformer: lr 3e-4, T 0.1.
- 5 seeds each.

## Score (test run only; frozen embeddings, deterministic central 12 s crop)
- **Probe:** kNN, k = 15, z-scored, 5-fold CV grouped by 30-min block **within the test run**.
- **Same space:** every metric is computed for the trained encoder and for **the same network's initial weights**. The gain = trained − untrained, paired by seed.

| Score | Metric | What it tells you |
|---|---|---|
| S1 polarity | skill (chance-corrected balanced accuracy) | does the embedding know up-first vs down-first? |
| S2 lobe count | skill (capped at 4) | the shape complexity |
| S3 duration | R² of log10 `m8_dur10` | how long the passage is |
| S4 magnitude group | skill, small (< 5 000 counts) vs large | the user's "events of similar magnitude" |
| S5 4-group | skill on magnitude × polarity | the natural groups found in the magnitude analysis |
| L1 leakage, run 0204 | skill for is-0204, all runs, one space, logistic regression and kNN (worse one reported) | must not exceed the untrained network by more than 0.05 |
| L2 leakage, date | date skill within the test run's 3 days | should be ≈ 0 |

**Learned something:** the gain is > 0 with its 95 % block-bootstrap CI excluding 0, on at least 3 of S1–S5.
**Leakage OK:** L1 ≤ untrained + 0.05 and L2 ≤ 0.03.

**Caveat known in advance:** S4 and S5 are derived from amplitude. The encoder sees max-normalised input, so magnitude can only reach it indirectly (noise floor and shape). A gain there is shape-linked, not a direct read of amplitude.

## Dry run before the real models (lesson from G)
- **Planted signal:** the morph8 features as the "embedding" must score high on S1–S3, because they are circular. This checks the pipeline can pass.
- **Planted leak:** an embedding with a one-hot run column appended must fail L1. This checks the pipeline can fail.
- **Null:** Gaussian noise embeddings must give gains ≈ 0.
