# Reviewer F: confirmation check of step 5b (reports 14 and 15), 2026-10-11

_Saved verbatim by the orchestrator. Orchestrator check of F1: `analyze.controls()` computes `ALL_PASS = all(crit.values())`, which includes L1. `main_analysis` receives only `m_leak`. Confirmed._

## Verdict: FREEZE AFTER 3 small fixes (no blocker, no major)

R1, R2 and R3 and my earlier MINOR items are done in the code, not only in the text. I checked this by reading the code and recomputing independently. The code I checked is byte-identical to the code that `test_step5b` tested: all 10 hashes match `test_step5b_20261011_094902.json`.

Three small gaps remain, each a few lines. Two make the code disagree with report 14 (F1, F3); one is a weakness in the code (F2).

**Fix before the freeze:**
1. **F1. Make the code follow report 14 A2's L1 rule.** It currently treats an L1-only failure as a stop, and still reports gate labels.
2. **F2. Check provenance at scoring time.** `score_all.py` only checks the fold; it should check every checkpoint against its manifest entry.
3. **F3. State the scope of the sensitivity analyses** in the frozen text.

## Findings

### MINOR, fix before the freeze

**F1. If only L1 fails, the code does not do what report 14 A2 says.**
- **What A2 says:** an L1 failure alone does not stop the dry run. The gate becomes UNCALIBRATED and is reported without gate labels.
- **What the code does:** `analyze.controls()` puts L1 inside `ALL_PASS` (`analyze.py:186-188`). When I planted a case where only L1 fails, it returned `ALL_PASS = False` with `gate_status = "UNCALIBRATED"` (`confirm_L1only.json`: C1, C3, C4, C5 and L2 true, L1 false). An operator applying "stop if not ALL_PASS" would stop when A2 says not to.
- **Second gap:** `main_analysis` reads only `m_leak` from the controls file (`analyze.py:199`). It always produces gate labels (l.146, l.150), even when the gate is UNCALIBRATED.
- **Fix:**
  - `ALL_PASS = C1 ∧ C3 ∧ C4 ∧ C5 ∧ L2`, reporting `gate_status` separately.
  - In `main`, read `gate_status` from the controls JSON; if it is UNCALIBRATED, write gate results as descriptive with `label = None`.
  - Add one unit test.

**F2. Scoring checks only the fold.**
- `score_all.emb_of` and `evaluator2.score_run` assert the fold and that a final hash exists. They do not compare `summary.json` with the manifest entry.
- So a run made outside `launch.py` would be scored silently, for example one with `--max-rounds`, a different `--lr` or a different `--init`. (`--max-rounds` still exists and has a default; `rounds_run` is never checked.)
- `launch.py`'s docstring says it checks rounds, but `check_plan` asserts only K, the batch rule and examples. The examples tolerance does catch a wrong R for the server: 409 600 / 423 936 = 0.966.
- **Fix:** in `emb_of`, assert all of the following:
  - `args` equal the resolved manifest CLI, with the learning rate taken from `tuned_lr.json`;
  - `rounds_run == R`;
  - `provenance.init` equals `server_hashes.json`;
  - `provenance.final == sha256(final.pt)`.

**F3. The scope of the sensitivity analyses is not stated.**
- In the code, sensitivity (a), (b) and (d) apply to Q-F only, and (c) to Q-F and Q-S (`analyze.py:155-161`).
- Report 14 A3 says only "sensitivity analyses (a)-(d)".
- (b) drops a seed from the whole contrast if it collapsed in any fold. This is necessary because the grid needs complete rows, but it is not stated.
- The collapse list does not cover S0 or server checkpoints (report 09 §3.5).
- **Fix:** state all of this in a sentence in the freeze record, or extend the code.

### MINOR, optional or log only

- **Convergence counts are unreliable in threads.** `warnings.catch_warnings` is not thread-safe, so the convergence counts (`lin_converged`, `gate_fits_nonconverged`) may be wrong when `score_all` runs in a thread pool. The scores themselves are unaffected, and these counts are diagnostics only.
- **`score_all.dryrun` and `main_mode` have never run end to end**, because no study checkpoints exist yet. On reading I found no silent defect:
  - tags match the manifest: all 330 scored tags are present;
  - C4 and C5 noise seeds are consistent between the bisection and the final pass;
  - the L2 arm names match the parsing in `analyze`.

  Any bug found at step 8 is fixable before a main model is scored, but the fix would be a logged deviation. Optionally, run a smoke test on dummy tags in a scratch `RUNS` folder.
- **`analyze` and `evaluator2` do not assert the pinned `folds.json` hash.** fedsim3 does. Low risk.

## What I confirmed (question by question)

1. **R1–R4 in the code.**
   - fedsim3: every arm argument is required, there is no FedAdam choice, the folds and pool hashes are asserted, the guard checks both name and content, the learning rate is logged per client per round, the final hash is stored, and `summary.json` is opened with `"x"`.
   - The test output records the code hashes. `safe_metric` asserts at least 2 classes per day. The gate counts non-converged fits: 0 of 60 in my check.
   - Report 14 A5 corrects the text: the file names, L1, the IID seed and the warm-up overshoot.
2. **Controls match A2.**
   - All controls are 10 vs 10: C256rep seeds 10–19 against C256 seeds 0–9.
   - C3 uses INIT seeds 0–9. `init_encoder(s)` is bit-identical to `fedsim3.build(s)` for s = 0 and 7.
   - C4 uses τ = 0.5. C5 uses 8 bisection steps on fold 0, targeting LOBES Δ = −0.05, against the C256 fold-0 mean, then applies τ* to all folds.
   - m_leak is computed on C256rep vs C256. κ* is the smallest κ detected in at least 50 % of the 30 seed-folds, and L2 requires gate FAIL at κ*.
   - `gate_core2` gives exactly the same output as `gate_core`.
   - L1's pre-stated response is only partly implemented (F1).
3. **`main_analysis` against my own formulas.**
   - I recomputed two-stratum, seed-paired (A4) and interaction (A4) on 200 random datasets. The maximum difference was 3.6e-15.
   - Q-S (Holm over 4 tests, both reject and Δ ≥ m/2, DOES NOT HELP iff EQUIVALENT) had 0 mismatches in 5 000 random scenarios.
   - The attribution rules had 0 mismatches over all 1 250 outcome combinations.
   - Collapse uses 0.5 × the C256 median in each fold.
   - Sensitivity (a) uses the 3 fold-0 days, (d) the 8 days without Feb 7, (b) the seeds left after collapse, and (c) the paired method.
4. **The manifest reproduces A1.**
   - All 338 entries match the A1 table on arm, kind, batch, R, budget, seed, fold and tag (0 mismatches).
   - Learning-rate inheritance:

     | Arm(s) | Learning rate |
     |---|---|
     | F-IID, both ablations | TUNED:F-nat |
     | C256rep | TUNED:C256 |
     | Server | 3e-4, not tuned |

   - Server initialisation is SERVER:s for S+F and S+C, and SERVER:0 for the S tuning runs.
   - Group cells are exact: tuning 7 arms × 4 learning rates, all at fold 0 and seed 0; ablations use seeds 0–4.
   - `launch.py check` passes for 41 configurations. Launch refuses S arms before the server hashes are pinned and tuned arms before `tuned_lr.json` exists; both are tested.
   - `tune.select` edges are correct: exactly +0.02 keeps 3e-4, +0.0201 switches, and a lower learning rate can win.
5. **Blockers:** none.

## Checks run

GPU use was under 1 minute (model construction and one embedding); GPU memory was back at 15 MiB afterwards. CPU use was about 2 minutes.
- `confirm_step5b.py` → `confirm_step5b_20261011_100258.json`: hashes, manifest against A1, recomputed formulas, Q-S and attribution checks, `tune.select`, INIT equality, gate equality, folds pin.
- `confirm_L1only.py` → `confirm_L1only.json`: the L1-only failure case.
- `launch.py check` (plan-only; it writes nothing).

## Files created

All in this folder:
- `confirm_step5b.py`, `confirm_step5b_20261011_100258.json`
- `confirm_L1only.py`, `confirm_L1only.json`

No other file was modified and nothing was committed.
