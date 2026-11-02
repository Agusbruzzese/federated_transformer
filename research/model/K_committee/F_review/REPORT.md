# Reviewer F: final review of protocol v2 (report 09) and its code, 2026-10-11

_Saved verbatim by the orchestrator from the agent's hand-back. Orchestrator checks:_
- _MAJOR-1 confirmed. The defaults are rounds 200, batch 64, examples X (`fedsim2.py:286-287`); server with R = 200 gives 409 600 examples (recomputed)._
- _MAJOR-2 misalignment confirmed. Permuting one arm's day columns changes the SE (0.0101 → 0.0120) with no error raised (independent toy test)._
- _All I1–I5 entries in `test_review_fedsim_20261011_091643.json` are true._

## 1. Verdict: FREEZE AFTER REQUIRED CHANGES

I found no blocker. Every simulator path I tested behaves as report 09 says. That covers FedAvg, fresh optimiser and schedule after `--init`, budgets, firewall, determinism, and that monitoring does not change training. The evaluator's formulas match sections 5–6. I also reproduced the gate independently: report 12's per-seed Z values, to every digit.

The protocol is still not ready to freeze by hash, for three reasons. The arm configurations live only in a test file. The step-11 analysis and the tuning scorer do not exist yet. And N_SEEDS = 10 has made the controls section inconsistent.

**Required changes, most important first:**
1. **R1 (MAJOR-1).** Freeze a run manifest and a launcher.
2. **R2 (MAJOR-2).** Write, test and freeze the analysis code (`analyze.py`) and the tuning scorer (`tune.py`), and specify the two missing formulas.
3. **R3 (MAJOR-3).** Redesign C1 and the other controls for 10 seeds (exact design under MAJOR-3).
4. **R4.** Do the text and provenance fixes (MINOR list).

## 2. Findings

### MAJOR-1. Arm definitions are not in the frozen code, and the CLI defaults are unsafe

**Evidence.**
- `fedsim2.py` encodes arms only through CLI flags. The one place where R, examples, batch and kind are bound to each arm name is the `ARMS` dict in `test_fedsim2.py:15-23`.
- The defaults (`fedsim2.py:284-291`) give the wrong arms silently:
  - `--arm server --examples P` with the default R = 200 gives 409 600 examples instead of P = 423 936. I ran this (`I6_server_default_rounds_examples`).
  - `--arm central` defaults to batch 64 (`I6_central_default_batch`), i.e. C64, not C256.
  - F-X+P needs `--rounds 233 --examples X+P` typed by hand.
- The default tag `{arm}_f{fold}_{kind}_b{batch}_s{seed}` (l.228) is the same for F-nat, F-X+P and S+F, and for every tuning learning rate. Runs are protected from overwriting, but they need hand-made tags.
- `--server fedadam`, `--opt reset` and `--noise m4` are also reachable. None of them is a protocol arm, apart from the reset ablation.

**Fix.**
- Add `manifest.json`, frozen with the code. One entry per (arm, fold, seed), giving the full CLI (including the tuned `--lr` once step 7 is done), the tag and the server checkpoint path.
- Add `launch.py`, which runs only manifest entries and checks the plan before training: examples_total within the stated tolerance of the budget, K, b, R, and `init` hash equal to the pinned server hash.

### MAJOR-2. The decision procedure is only partly implemented; two formulas are unspecified

**Evidence.**
- `evaluator.py` provides building blocks only: `two_stratum`, `outcome`, `contrast_label`, `gate_test`, `m_leak_rule`, `holm`, `heterogeneity`.
- No code does any of the following:
  - assemble Y(s, d) into [seed × 9 days] with keys. Day order and `fold_of_day` are built ad hoc in `test_evaluator.py:76`;
  - compute Zbar(s);
  - apply the Q-S rule (Holm over 4 tests AND both point estimates ≥ m/2, "does not help" iff EQUIVALENT);
  - apply the three attribution rules;
  - apply the collapse rule. Effective rank is not computed in `evaluator.py` at all;
  - run sensitivity analyses (a)–(d);
  - score tuning on Feb 10: `fold_sets` only returns folds.json evaluation days.
- Section 5.8's interaction CI, (S+F − F-nat) − (S+C − C256), and section 5.3/5.9(c)'s "seed-paired version" have no formula in report 09.
- **Silent-misalignment risk, measured.** If one arm's columns are permuted against the other's, `two_stratum` raises no error. Delta is unchanged, the SE changes (×1.28 in my test), and the per-fold Deltas and heterogeneity flag become wrong. When both arms are permuted the same way, the result is unchanged, as it should be (`check_data.json`).
- `score()` never checks which fold a checkpoint was trained on. I scored one checkpoint on folds 0 and 1 without any error (I7). Scoring a fold-1 model on fold 0 would leak evaluation days silently.

**Fix.**
- Write `analyze.py` and freeze it before step 7. Its inputs are tidy rows keyed (arm, seed, fold, day, metric). It should:
  - pivot the rows and assert the full 10 × 9 grid with day order taken from folds.json;
  - derive `fold_of_day` from folds.json;
  - implement every rule in sections 5.7–5.9 and 3.5.
- In `score()`, read `summary.json` next to the checkpoint and assert `args.fold == fold`, or `arm == server` for S0.
- Specify the interaction in report 09 like this:
  - per-seed I(s) = M_SF(s) − M_F(s) − M_SC(s) + M_C(s), with seeds shared, used as a paired seed stratum, var/n, df n − 1;
  - day stratum = var over d of the 4-arm day contrast / D;
  - Satterthwaite df.
- Specify the seed-paired version as: seed stratum var_s(M_A(s) − M_B(s))/n, df n − 1, with the same day stratum.
- Add unit tests: misaligned input must raise; one known toy dataset must give known labels for Q-F, Q-S, the attributions and collapse.

### MAJOR-3. N_SEEDS = 10 makes sections 5.8, 6.3, 7, 8.2 and 10 inconsistent

**Evidence.**
- C1, C4, C5, L1, L2, the m_leak rule and the 15-NN noise floor are all defined on C256 seeds 0–4 vs 5–9. Every one of those seeds is now a main-analysis C256 seed.
- The controls therefore run at n = 5 vs 5, while the decisions run at n = 10 vs 10. Results at one sample size do not carry over to the other.
- m_leak's term 2·SE is computed at 5 vs 5. That SE is about √2 times the SE of the 10 vs 10 gate test, so the margin is calibrated more leniently than the test it governs.

**Fix: the design I recommend.** Train **C256 seeds 10–19 × 3 folds** in the dry run, as a replicate reference used in no main contrast. Cost: 30 runs × 3.1 min ≈ 1.6 h GPU. Then run every control at the deployed size:

| Control | Comparison | Required result |
|---|---|---|
| C1 | C256(10–19) vs C256(0–9) | EQUIVALENT |
| C3 | init(0–9) vs C256(0–9) | INFERIOR |
| C4 | C256(10–19) + noise τ = 0.5 vs C256(0–9) | INFERIOR |
| C5 | τ* by bisection on fold 0, C256(10–19) vs C256(0–9) | not EQUIVALENT |
| m_leak | max(0.03, 2·\|Δ_leak(10–19 vs 0–9)\|, 2·SE at 10 vs 10) | — |
| L1 | gate on the same pair | PASS |
| L2 | planted offsets on C256(10–19) vs C256(0–9) | FAIL |

- The 15-NN noise floor becomes the seed pairs (s, s + 10).
- Fix the wording "C256 seeds 5–9 also serve C1".
- Name the seeds used by the 5-seed ablations (0–4).
- Update the section 10 costs: main runs ≈ 20.2 h; +1.6 h for this design.

**Cheaper alternative**, if the 1.6 h is refused: keep a split-half 0–4 vs 5–9. Label it explicitly as an n = 5 calibration, and accept that m_leak's SE term is lenient for the 10 vs 10 test.

### MINOR

1. **Wrong file names in report 09.**
   - §1.4 names `C_federated/fedsim.py` as "the simulator", and §8.1 cites `test_fedavg.py`. The real files are `protocol/fedsim2.py` and `test_fedsim2.py`.
   - `folds.py` says "PROTOCOL v1", and it rewrites `folds.json` with mode `"w"`. Make it refuse to overwrite.
   - The freeze list must name the right files.
2. **L1 is not "PASS by construction" (§8.2).** Without the 0.03 floor, Δ = SE gives U = SE·(1 + t) > 2·SE = m_leak. Simulated false-FAIL rates of the rule (n = 5 vs 5):
   - Z per-seed SD 0.02: 0.26 with the floor, 0.42 without it;
   - SD 0.012: 0.019;
   - measured SD of about 0.008: ≈ 0.

   The practical risk is low, but correct the text, and decide in advance what happens if L1 fails.
3. **L2 has no operational definition.**
   - "The smallest strength the permutation null detects" is not defined. The test only requires FAIL for some κ (`test_evaluator.py:122`).
   - My rerun: the per-(seed, fold) null "detects" (Q > null 95th percentile) 1/15 runs at κ = 0, 0/15 at κ = 0.01 and 1/15 at κ = 0.02. So κ ≤ 0.02 is undetectable, which is consistent with the gate passing there.
   - Define κ* as the smallest κ with ≥ 50 % of seed-folds detected, and require gate FAIL at κ*.
4. **Learning rate is not logged.** §3.6 promises a learning-rate trace, and E's B1 fix (d) asked for one, but `rounds.jsonl` has no learning rate. E's B1 test (b), identical clients reproduce the single-client run, was also not added. Proof of B1 still holds through C0a/b/c plus my I1/I5.
5. **Provenance gaps.**
   - Neither test output records the hash of the code it tested. Rerun both tests on the frozen hashes, or record the hashes in the outputs.
   - The noise-pool hash is recorded only after training (l.274). Assert the pinned hash in `day_pool`.
   - The guard is name-based (l.43), so a copy or symlink at another path bypasses it.
   - The final checkpoint hash is stored as `final=None`.
6. **Skill fails on single-class days.** `m4.skill` returns −inf when a day has a single class. No day is affected now: all 10 evaluation and tuning days contain lobe classes 1–4 and both polarities, the smallest class having 11 events. Add an assertion.
7. **Small code issues.**
   - The `_cv_skill` gate fits never check convergence.
   - The `score()` docstring gives the wrong signature for `transform`.
   - With fractional steps, warm-up overshoots the peak learning rate by up to 0.17 % (C64, F-X+P). This is negligible.
   - State that the IID permutation (seed 1000 + fold) is fixed across init seeds.
8. **The day stratum treats the 9 days as independent.** With an arm × fold interaction, the 3 days of a fold are correlated, so SE is understated. The protocol already acknowledges this (heterogeneity flag only). Keep the caveat in the claims.

### Did report 09 resolve E's points? (checked against the code)

- **B1 resolved.** `load_vec` copies (l.152-159). C0 passed, and my I1, I4 and I5 pass.
- **M1 resolved.** Per-day pools cover all 18 days. The pool hash matches report 10, and every client, server and tuning day has a pool (minimum Feb 11, 35 segments). The per-fold firewall is in place, and the guard is present but name-based (MINOR 5).
- **M2 resolved in formula.** `two_stratum` matches §5.3: ddof 1, pooled df nA + nB − 2, Satterthwaite. The assembly is missing (MAJOR-2).
- **M3 resolved.** The gate matches §6.2: PCA-19, conditional minus reference, GroupKFold(5) over absolute 30-min blocks (0 blocks span two days), 50 block-label permutations, Z = Q − Q0. The gate test is Welch, one-sided 95 %. The L2 definition is vague (MINOR 3).
- **M4 resolved in the text.** The tuning code does not exist yet (MAJOR-2).
- **M5 resolved in the text.** `outcome` and `contrast_label` match §5.6–5.7 exactly. The Q-S and attribution rules are not coded (MAJOR-2).
- **m1–m8 resolved in the text.** I recomputed m2: P = 423 936, and central exposure 994.9 / 1 009.1 / 926.5. Collapse (m5) has no code.
- **m9 only partly resolved.** MINOR 1 lists what is still wrong.

## 3. Checks I ran

All GPU use was under 2 minutes and the GPU was back at its 110 MiB baseline afterwards. CPU use was about 12 minutes.

- **`check_data.py` → `check_data.json`:**
  - code hashes match reports 10 and 11 (fedsim2 638febb1, folds 79f3b1ff, pool 07b52c9a);
  - class counts per evaluation and tuning day are as in MINOR 6;
  - no block spans two days; date, hour and wall_time agree; hour bins are 0–3;
  - 19–46 blocks per evaluation day;
  - pool coverage complete; P recomputed;
  - simulated L1 failure rates (MINOR 2);
  - misalignment test (MAJOR-2).
- **`test_review_fedsim.py` → `test_review_fedsim_20261011_091643.json`.** Tiny runs, redirected to `F_review/_runs_*`.

| Check | Result |
|---|---|
| I1: S+C and every S+F client start bit-identical to the server checkpoint | PASS |
| I2: AdamW state empty at the first step after init; S+F moments persist into round 1 | PASS |
| I3: learning-rate trace equals peak·lam(i) from i = 0, i.e. a fresh schedule per phase (S+C, S+F) | PASS |
| I4: monitoring on vs off gives bit-identical weights (central K = 1 and fed) | PASS |
| I5: fed IID rerun bit-identical; seed 1 differs | PASS |
| I6: full-length schedules, warm-up fraction | 0.0495–0.0501 for all arms and folds |
| I6: full-length schedules, last learning rate | ≤ 2.7e-6 × peak |
| I6: defaults hazard | found (MAJOR-1) |
| I7: S0 path scores a fedsim2 checkpoint; no fold check | confirmed (MAJOR-2) |

- **`check_L2.py` → `check_L2.json`:**
  - reproduced report 12's Zbar for M5 seeds 0–4 exactly;
  - permutation-null detection rate at κ = 0 / 0.01 / 0.02 is 1/15, 0/15, 1/15.

## 4. Files created

All in this folder:
- `check_data.py`, `check_data.json`
- `test_review_fedsim.py`, `test_review_fedsim_20261011_091643.json`
- `check_L2.py`, `check_L2.json`, `check_L2.log`
- `_runs_20261011_091643/`: tiny test runs, 8.6 MB, safe to delete

No other file was modified. Nothing was committed. Importing the protocol modules may have refreshed `protocol/__pycache__`.
