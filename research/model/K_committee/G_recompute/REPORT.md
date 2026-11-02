# Member G (independent recomputation, step 12), 2026-10-12

_Saved verbatim by the orchestrator from the agent's hand-back. Orchestrator check: merging `g_scores.parquet` with the protocol's `scores_combined_20261012_211200.parquet` on (arm, seed, fold, day, metric) gives 900 cells, max |diff| = 0.0. `integrity_checks.json` present (fold, hash and day checks per run)._

## 1. Verdict: REPRODUCED

With my own embedding, probe and statistics code, I get every Q-F, Q-S, interaction and S+F vs C256 number in `results_main_20261012_211202.json` exactly, to every printed digit, along with the same outcomes and labels. All 900 per-cell values match the protocol table with a difference of exactly 0.0. In the gate spot check, 4 or 5 of the 6 Q values match exactly (depending on solver setting). The rest differ by at most 0.0015, which is one prediction flipping because of solver tolerance. This has no effect on any decision.

## 2. My numbers vs the protocol's

Each cell is G / Protocol. Only one value is shown where they agree.

| Contrast | Metric | Delta | SE | df | 90% CI | Outcome | Label |
|---|---|---|---|---|---|---|---|
| **Q-F** F-nat vs C256 | LOBES | -0.02281 / same | 0.00656 / same | 23.508 / same | [-0.03404, -0.01158] / same | EQ / EQ | EQUIVALENT / EQUIVALENT |
| | POLARITY | -0.00970 / same | 0.00313 / same | 25.860 / same | [-0.01504, -0.00436] / same | EQ / EQ | |
| S+F vs F-nat | LOBES | 0.04856 / same | 0.00846 | 24.058 | [0.03410, 0.06303] | NI / NI | NON-INFERIOR / same |
| | POLARITY | 0.00431 / same | 0.00336 | 25.527 | [-0.00142, 0.01004] | EQ / EQ | |
| S+F vs F-X+P | LOBES | 0.03831 / same | 0.00703 | 23.591 | [0.02627, 0.05035] | NI / NI | NON-INFERIOR / same |
| | POLARITY | 0.00404 / same | 0.00292 | 24.632 | [-0.00096, 0.00904] | EQ / EQ | |
| S+F vs C256 | LOBES | 0.02576 / same | 0.00862 | 17.725 | [0.01080, 0.04071] | NI / NI | NON-INFERIOR / same |
| | POLARITY | -0.00540 / same | 0.00329 | 20.840 | [-0.01106, 0.00027] | EQ / EQ | |
| S+C vs C256 (extra) | LOBES / POLARITY | 0.01082 / 0.00027, same | | | | EQ, EQ | EQUIVALENT / same |

**Q-S, one-sided p-values (mine):**

| Test | p | Holm reject (G) | Holm reject (P) |
|---|---|---|---|
| S+F > F-nat, LOBES | 3.19e-6 | yes | yes |
| S+F > F-X+P, LOBES | 7.10e-6 | yes | yes |
| S+F > F-X+P, POLARITY | 0.0899 | no | no |
| S+F > F-nat, POLARITY | 0.1055 | no | no |

- For LOBES, both tests reject and both Deltas (0.0486 and 0.0383) are at least m/2 = 0.02.
- Verdict: **HELPS** in both my run and the protocol's.

**Interaction (S+F − F-nat) − (S+C − C256), report 14 A4:**
- **LOBES:** Delta 0.03775, SE 0.00958, df 15.214, CI [0.02096, 0.05453], p 0.00064. Identical to the protocol.
- **POLARITY:** 0.00403, CI [-0.00232, 0.01038]. Identical.

I also ran my statistics code on the protocol's own score table. The maximum difference on Delta, SE, df, L and U was 0.0 for all 8 contrast-metric pairs. So the statistics code agrees on its own, separately from the scoring.

## 3. Per-cell agreement

- **What I scored:** 5 arms × 10 seeds × 3 folds × 3 days × 2 metrics = 900 cells, each checkpoint only on its own fold.
- **Embedding:** done per run on the GPU with `eval_view` and `make("tf")`, weights from `final.pt["enc"]`.
- **Probes:**
  - kNN with k=15.
  - LogisticRegression with C=1, tol 1e-8 and max_iter 1e5, both with StandardScaler.
- **Score:** chance-corrected balanced accuracy over the classes present that day.
- **Result:** the maximum absolute difference against `scores_combined_20261012_211200.parquet` is **exactly 0.0** for every arm (C256, F-nat, S+F, F-X+P, S+C) and both metrics. No cell differs by more than 1e-6.
  - This is expected. The skill is a ratio of counts, so if the predictions are identical the values match bit for bit.
- **Convergence:** all 300 of my logistic fits converged (at most 555 iterations).

## 4. Gate spot check (fold 0, Q only, no permutation null)

Features: PCA-19 of the eval embeddings, the 19 `m8_*` columns, a 4-bin one-hot of `hbin`, and `has_dropout`. Classifier: LR(C=1) with StandardScaler, GroupKFold(5) over `m.blk`. Reference skill is 0.33716 for every row.

| Arm, seed | Q (G, tol 1e-8) | Q (G, protocol solver setting) | Q (protocol) |
|---|---|---|---|
| C256 s0 | -0.003234 | -0.001833 | -0.003234 |
| C256 s1 | -0.032389 | -0.033790 | -0.033790 |
| C256 s2 | 0.001628 | 0.001628 | 0.001628 |
| F-nat s0 | -0.011607 | -0.009940 | -0.009940 |
| F-nat s1 | -0.003361 | -0.003361 | -0.003361 |
| F-nat s2 | -0.017001 | -0.017001 | -0.017001 |

- **Fitting PCA inside the CV folds, or on standardised embeddings:** both give clearly different values, so the protocol fits PCA once on the raw eval embeddings. That is a legitimate reading of report 09 §6.2.
- **Cause of the leftover differences:** the protocol's gate uses `LogisticRegression(C=1, max_iter=2000)` with the default tol 1e-4, not 1e-8. With that loose tolerance a single prediction can flip.
  - Each flip is about 0.0015 of skill, one event out of about 350 per class.
  - Re-embedding with the protocol's batching changes the embeddings by about 1e-6 and leaves Q unchanged, so batching is not the cause.
- **Impact:** this precision is about 1/20 of m_leak (0.03), and the gate upper bounds are around -0.004. No gate label can change.

## 5. Integrity checks

- **Folds:** all 150 checkpoints were checked. `summary.json` `args.fold` equals the directory's fold in 150 of 150, and the trained `clients_days` equal that fold's client days in 150 of 150.
- **Evaluation days:**
  - In every fold they are disjoint from the client days, from the server days (Feb 18, 20) and from the tuning day (Feb 10).
  - The client days are also disjoint from the server and tuning days.
  - The 9 evaluation days are all distinct.
- **Hashes:** `sha256(final.pt)` equals `provenance.final` for **all 150** runs, not just a sample of 10.
- **S-arm initialisation:** every S+F and S+C run's `provenance.init` equals `server_hashes.json[seed]`.
- **Server checkpoint:** `v21_server_s0` was trained only on Feb 18 and 20, and its hash matches.

## 6. Issues found

- **No reproducibility issues.**
- **Minor, documentation only:** the gate's logistic regression runs at default tolerance (1e-4, max_iter 2000), while the probes use tol 1e-8. Report 09 §6.2 does not specify a tolerance, so this is not a deviation. It does mean an individual Q value is only reproducible to about ±0.0015 under a different solver setting. No decision depends on that precision.
- **Interpretation, not an error:** Q-F comes out EQUIVALENT even though F-nat's LOBES interval excludes zero ([-0.034, -0.012]). The federated arm is measurably worse on lobes, by about 0.023, but inside the 0.04 margin. That is how the frozen TOST rule works, and it is worth stating plainly when reporting the result.

## 7. Files created

All in this folder. No existing file was modified and nothing was committed.

- `g_embed.py` – embeddings, fold and hash checks
- `emb/v21_<ARM>_f<f>_s<s>.npz` – 150 embedding files
- `integrity_checks.json`
- `g_score.py` → `g_scores.parquet`
- `g_stats.py` → `g_results.json`
- `g_gate.py` → `g_gate.csv`
- `g_gate_defaulttol.py` → `g_gate_defaulttol.csv`
- `g_days_check.txt`

Two throwaway diagnostic scripts are in the session scratchpad: `g32.py` and `g_batch.py`.
