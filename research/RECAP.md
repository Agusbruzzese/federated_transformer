# Project recap (as of 2026-10-01)

Spintronic (TMR) vehicle sensor, 14 unlabeled days, aiming at federated self-supervised learning.
This file is the entry point for a new session. Details are in the linked reports.

## 1. Starting point
- The repo had a Flower + FedProx federation (15 clients, NT-Xent on KMeans pseudo-labels, 70 EMD/HHT features).
- The feature-extraction code (`pipeline.py`, `clustering.py`) was missing from the repo.
- The user asked to redo everything "with a scientific disposition", starting with extraction, with Claude as orchestrator of agents and an external low-cost auditor checking every step.

## 2. How the work is organised
- **Agents:** one per workstream. Each writes `REPORT.md`; Claude verifies key numbers independently, then audits it.
- **External auditor:** `research/audit/auditor.py` (Sonnet 5.5, medium effort). Spend so far ≈ $2. Usage is in `research/audit/usage.jsonl`. The API key is at `~/.config/federated_transformer/anthropic_api_key`, outside the repo. **Rotate it:** it was pasted in chat.
- **Rules:**
  - `research/extraction/BRIEFS.md` (tasks) and `PROTOCOL.md` (Run 3 = holdout, nulls, bootstrap CIs).
  - Every decision, audit response and rejected audit point is logged in `research/extraction/ORCHESTRATION_LOG.md`.
  - **Stopping rule:** a workstream closes when there are no BLOCKERs and every remaining MAJOR is irrelevant to the pipeline or verified by Claude.
- **Environment:**
  - `uv` in `~/.local/bin`, `.venv` with numpy/scipy/sklearn/PyEMD/pywt/torch 2.14 (CUDA, RTX 4060 8 GB).
  - Lockfile `research/requirements.lock`.
  - `.gitignore` excludes the dataset, `data/`, `.venv`.
  - **Nothing is committed yet**; the repo still has only its initial commit.

## 3. Phase 1: extraction re-design (workstreams A–D, all closed)

| WS | Question | Key results |
|---|---|---|
| **A** integrity (`A_integrity/`) | file format, clock, gaps | Values are `<u4` holding 24-bit codes. Time is device **ticks, not µs**. **fs ≈ 2416 Hz** (2407–2421 per chunk; use `fs_local`), not 2400. 188k dropouts (mostly ~105 samples), concentrated in run 0204 from Feb 6 (~1 %). 340 h of data, no missing files, no clipping. Loader `bsd_io.py` (46 self-tests). |
| **B** signal (`B_signal/`) | what vehicles look like | Vehicles are slow (energy < 2.5 Hz), mostly bipolar, 2–12 s. 50 Hz mains ≈ 1.4k counts. The baseline wanders strongly. **Subtract the baseline, never divide** ("DC-relative" division is pointless). Chain: mains regression → LP (detection 5 Hz, features 8 Hz) → 50 Hz → two-pass 20 s median. |
| **C** detection (`C_detection/`) | which detector | **HYST‑BP** wins: ΔF1 +0.16 over fair fixed windows on the holdout, and the ranking is robust to independent injection templates. The old DC-relative fixed windows score F1 0.006 (a reconstruction). An SNR tier is needed for purity. 21 s windows. The precision figures rest on **AI labels only**. |
| **D** features (`D_features/`) | what to feed the model | **The old pipeline read the data wrong:** `comparison/central_pipeline.py:94` (branch `origin/comparison-update`) took the low 16 bits as int16, so about 41 % of event windows are corrupted. Claude reproduced this independently. `t_hours` leaks the date (skill 0.98). **EMD/HHT is unsuitable** (IMF index unstable, mains in the IMFs). Morphology (19 scalars) is robustly in the top-2. Use SimCLR pairs, not KMeans pseudo-labels; normalise globally. |

**Synthesis:** `research/extraction/SYNTHESIS.md`. It reconciles the workstreams, proposes the pipeline, and in §6 tables the audit points that were rejected, with evidence.
**Consequence:** the earlier federation results (`federation_run*.log`, loss 0.48 → 0.35) are untrustworthy.

## 4. Phase E: extraction pipeline built and run (closed)
- **Code:** `extraction/` package. It reuses A's loader, C's detector and B's chain. Run it with `python -m extraction.run --out data/events`. Self-tests: `python -m extraction.tests`, 33/33 pass, including a regression test for the int16 bug.
- **Output:** `data/events/` (98 MB): 15 260 events, of which **4 375 have SNR ≥ 20** and 3 534 have SNR ≥ 30.
  - Per event: a 21 s window at ~50 Hz, the onset/offset mask, 19 morph features on 21 s (`m_*`) and on the central 8 s (`m8_*`), and metadata.
  - 20 min on 16 processes; deterministic.
- **Validation:**
  - The diurnal curve looks like traffic.
  - Morph features at SNR ≥ 20 don't leak the date (skill 0.02).
  - The all-SNR tier leaks the run because of the run-0204 dropout blips, so **train on SNR ≥ 20**.
- Report: `research/extraction/E_pipeline/REPORT.md`.

## 5. Phase F: central self-supervised model (done)
- **Pre-registered** before training: `research/model/PREREG_central.md` v2 (sha256 d322a199…) plus `PREREG_central_addendum1.md` (94297f25…).
- **Code:** `ssl_central/`. Checkpoints and embeddings in `data/ssl_central/`.
- **Setup:**
  - SimCLR on SNR ≥ 20 events; Runs 1, 2, 4 for training, Feb 26 for validation, **Run 3 as holdout** (weakly held out).
  - A CNN and a patch transformer (~210k parameters each), 3 seeds each, compared against their own random-init networks.
  - Training takes ~5 min on the GPU; evaluation took ~6 h.
- **Results** (report: `research/model/F_central/REPORT.md`):
  - **Verdict "Positive" for both,** by the pre-registered rules.
  - **Real gains over random-init:** polarity and lobe count; duration for the transformer only.
  - **No date or run leakage** outside run 0204.
  - **CNN vs transformer:** the CNN wins narrowly on the rule (P1 mean 0.723 vs 0.705, overlapping CIs). The transformer is cleaner: it improves on all four targets, keeps duration, and leaks less.
  - **Caveats:**
    - The amplitude "gain" equals SNR alone, so it is leakage.
    - The CNN loses duration information.
    - SSL training doubles run-0204 identifiability.
    - Both encoders stay below the hand-crafted morphology on every physical target.
    - There are no natural clusters.
  - **The audit found the pre-registration criteria were too loose** (Claude's design error). The verdict is kept as the rules produce it; the fixes go into the next pre-registration.

## 6. Open items and next options
- **Decide:**
  1. Tighten the evaluation and rerun: residualise log_amp on SNR; stricter leakage tests, including run 0204; more seeds; leave-one-run-out; a faster leakage probe.
  2. Go federated with the transformer, using this central model as the benchmark.
  3. Get labels.
- **Human labels:** a blind 80-plot pack is in `research/extraction/C_detection/figs/human_pack/` (README with the codes; fill in `C_detection/data/human_pack_labels.csv`). It checks vehicle vs non-vehicle only. Car-model recognition would need outside ground truth.
- **Pending:** commit `extraction/` and `ssl_central/` to a branch; rotate the API key.
- **Lessons:**
  - Run dependent phases serially.
  - Subagents can't write `REPORT.md` (Claude saves it from their hand-back).
  - HistGradientBoosting leakage probes with ≥ 20 permutations are very slow on this laptop.

## 7. Phase G: tightened rerun plus independent critique (2026-10-01)
- **Rules and data:** PREREG v3 (`research/model/PREREG_central_v3.md`). G report: `research/model/G_loro/REPORT.md`. Critique: `research/model/critique/CRITIQUE.md`. Human report: a Claude Doc, "Spintronic SSL: tightened evaluation report".
- **Pre-registered verdicts** (CNN "No evidence of learning", TF "Learns but leaks") are **invalid** because of a v3 design error. Pooling embeddings from 4 networks makes the network identifiable, so the leakage tests were unpassable.
- **The fair single-space test** (each fold model vs its own init, on its held-out run): both learn shape structure.
  - Transformer: polarity +0.26, lobes +0.17, duration +0.07.
  - CNN: lobes +0.12, polarity +0.04, no duration loss.
- **Leakage:** run 0204 leaks a little (+0.02 to +0.09). Date and other runs don't.
- **amp_resid** turned out to be the noise floor (r = 0.997), so it is dropped.
- **Next:** outside labels → a v4 single-space evaluation, dry-run on planted controls first → federated with the transformer.

## 8. H / H2: SimCLR central model and the transformer claim (2026-10-01)
- **H** (simple split, train 0204/0212/0224 → test 0218): `research/model/H_simclr_split/`. The transformer learned 5/5 shape scores vs its untrained weights.
- **H2** (pre-registered, LORO 4 runs × 5 seeds, planted-control dry run, independent review): `research/model/H2_transformer_claim/`.
  - **Verdict:** the transformer learns shape and beats PCA and raw waveforms, but **is NOT better than a matched CNN**.
    - Lobe count: CNN better in 4/4 runs.
    - Polarity: tie.
    - Duration: depends on the probe.
  - Run-0204 leakage is +0.04 to +0.07, depending on the probe.
  - Human report: a Claude Doc, "Transformer vs CNN: does self-supervised training really learn event shape?".
- **Next candidate:** a transformer patch-size ablation (0.5 s → 0.1–0.2 s patches).

## 9. M4: four SSL methods compared (2026-10-02)
- **Files:** `research/model/M4_methods/` (PREREG, m4.py, results, review/REVIEW.md). In-distribution split: test days Feb 7/13/19/25, validation Feb 10. 5 seeds each.
- **Rank:** SimCLR-CNN > SimCLR-TF > VICReg-TF > MAE-TF.
- **Review's narrower reading:**
  - Supported: SimCLR beats VICReg on lobes; the CNN gives the best embeddings (its untrained version is already strong).
  - **The MAE never trained.** Exclude it from claims.
  - **The leakage limit is uncalibrated,** so its flags are not evidence.
  - **SSL vs random or raw features is not established.**
  - Training was short: ~1 000 steps.
- **NEXT (when the user resumes):** propose the review's experiment:
  - 10k steps plus warm-up;
  - an MAE gate against linear interpolation;
  - random/raw baselines in the ranking;
  - leakage calibrated against morph8.

  Ask the user before starting.

## 10. M5: 10k-step SSL, gated MAE, random/raw baselines (2026-10-02)
- **Files:** `research/model/M5_long/` (PREREG + amendments C1–C10, m5.py, results, review/REVIEW.md). All results are exploratory (test days already seen in M4).
- **Results:**
  - The SimCLR-TF now ties with the CNN, closing M4's lobe gap; VICReg is also tied; MAE trains (corrected gate) but is weakest.
  - SSL ≫ raw/PCA.
  - **vs random networks:** SSL adds lobe-count information (+0.07–0.17, robust), but nothing beyond an untrained CNN on polarity/duration (saturated). The strict rule says "no evidence"; the review notes it had ~5 % power.
  - **Leakage:** SimCLR-CNN carries some extra run-0204 information (beats the permutation null); SimCLR-TF does not (inside the null); VICReg is clean.
- **Next step (reviewer):** external, non-morph labels (vehicle class/axles/speed) for label-efficiency curves vs random networks. Otherwise a fresh evaluation with ≥ 8 held-out days.

## 11. C1: clustering events by magnetic signature (2026-10-02)
- **Files:** `research/model/C1_clustering/` (PREREG v4 with D1–D11, REPORT.md, results.json, cluster_labels.csv, galleries, review/, audits). The SSL checkpoints are in `data/c1/ckpt/`; SimCLR-TF was trained on all 4 375 SNR≥20 events from all days.
- **Dry run:** D9 failed (low SNR and white-noise singletons). D10 is a post-hoc realistic dry run, which passed.
- **Pre-registered result:** R3 (morph8) is the only valid representation. On inspection, though, it is an artefact of the discrete features (polarity × first-lobe sign × n_lobes): a coarse shape taxonomy that is 3.9× wide in amplitude.
- **Best look-alike groups:** R2 SimCLR seeds 0, 1 and 3 (17 clusters, peak p90/p10 2.8×, ARI ~0.9 across days and runs, 33–40 % coverage). Seeds 2 and 4 collapsed. R1 AWD is the tightest in shape but covers only 16 %.
- **Next:** C2 (exploratory), a consensus over the non-collapsed seeds plus hierarchical sub-splitting to ≤ 1.5× amplitude; needs its own pre-registration.

## 12. C2: look-alike groups (2026-10-02, exploratory)
- **Files:** `research/model/C2_lookalikes/` (REPORT.md has a CORRECTIONS section and a FINAL VERDICT). The LORO SSL models are in `data/c2/`.
- **Tolerance:** corr ≥ 0.9, amplitude within 1.5×, duration within 1.5×. Complete linkage (C2a) gives 106 groups covering 38.5 %, every pair within tolerance.
  - Group boundaries are arbitrary tiles (LORO 0.38; the shuffled-shape null still gives 93 groups).
- **Coarse level:** about **7 shape families** (Louvain on corr ≥ 0.9), covering 88 %, reproducible across runs (LORO 0.58–0.70). Amplitude and duration vary continuously inside each.
- **SSL:** C2b was inconclusive; there is no evidence SSL helps with look-alike finding.
- **Recommended:** a two-level index (family, then a continuous amplitude/duration look-alike query) or a prototype library.
- **Rule:** NEVER commit unless the user explicitly asks.

## 13. Reports folder (2026-10-07)
- Plain-text reports now live in `reports/` (index: `reports/00_INDEX.txt`, naming `NN_YYYY-MM-DD_name.txt`, never overwrite). 01 extraction, 02 central model, 03 KMeans→SimCLR, 04 research plan (ground truth first; test battery T1–T12), 05 label-free plan (no labels available: federated vs central as main question, dipole simulator, time-locked recurrence).
- 06 synthetic event generator (`synth/`, 2026-10-08): per family×polarity registered template + PCA + smoothed bootstrap + coloured real noise; congruent in distribution (morph KS = real-vs-real), not indistinguishable (C2ST emb 0.60, PSD 0.90).
- 07 dataset vs 14-day generator run (`synth/compare.py`, `synth/timeline.py`): cross-fitted run matches daily counts (r 0.975), family mix (TVD 0.016), waveforms (median corr ≥ 0.994); joint Louvain gives 0 synthetic-only and 0 real-only clusters; unclustered 12 % not imitated.
- 08 committee + protocol v1 DRAFT (2026-10-10): `research/model/K_committee/` (A model, B stats, C federated simulator bit-identical to M5, D validity). Server = run 0218, tuning Feb 10, 3-fold cross-fit over 9 eval days, 10 clients/fold, 7 arms, TOST margins lobes 0.04 / polarity 0.02, leakage gate 0.03. Pending: independent review, evaluator, dry run, user approval of ~13 h GPU.
- E review (2026-10-10): accept with required changes. B1 fedsim aliasing bug (clients trained sequentially) FIXED + control C0 (test_fedavg.py) PASS; selftest still bit-identical. Pending for protocol v2: noise pools for Feb 7/13/19/25, nested-seed statistics, leakage gate redefinition, symmetric tuning, complete decision table.
- 09 protocol v2 DRAFT (2026-10-10): all review points mapped (sec. 0). Open user parameters: N_SEEDS (5/10), PRIVACY (S-PRIV yes/no). Next: step 2 noise pools (rule: strict, add relaxed if < 50 segments).
- Step 2 done (2026-10-10): per-day noise pool v2 for all 18 days, data/k_committee/protocol/bgpool_day_v2.npy (7 199 seg, sha 07b52c9a), Feb 7 & 10 relaxed, 36 duplicates removed, Feb 11 = 35 (< N_MIN, reported). Next: step 3 simulator (folds, arms, firewall, raw-pool assertion).
- Step 3 done (2026-10-10, report 11): protocol/fedsim2.py (sha 638febb1) — folds, all arms, guard, firewall; tests C0/C0prime/G/W/P/smoke ALL PASS. Next: step 4 evaluator. Rule from now: one report per step, never overwrite.
- Step 4 done (2026-10-10, report 12): protocol/evaluator.py + test_evaluator.py ALL PASS (type I 0.033-0.038; C1 EQUIVALENT, C3/C4 INFERIOR, C5 INCONCLUSIVE, gate detects offset >= 3 %). Next: step 5 independent re-review.
- Step 5 done (2026-10-11, report 13): reviewer F = FREEZE AFTER REQUIRED CHANGES, no blocker. R1 manifest+launcher, R2 analyze.py/tune.py, R3 controls at 10v10 (C256 seeds 10-19, +1.6 h), R4 minor. User: N_SEEDS=10, PRIVACY=no. Next: step 5b implement R1-R4 as new files.
- Step 5b done (2026-10-11, reports 14-15): protocol v2.1 amendment; fedsim3/evaluator2/analyze/tune/manifest(338 runs)/launch/score_all; test_step5b ALL_PASS (run 1 caught a zbar bug, fixed). User: controls = replicate C256 seeds 10-19. Next: F confirmation check, then step 6 freeze.
- Steps 5c+6 done (2026-10-11, reports 16-17): F confirmation fixes F1-F3, end-to-end scoring smoke ALL_PASS; PROTOCOL v2.1 FROZEN (protocol/freeze_v2_1.json sha 7f3754d0; check with verify_freeze.py). Next: step 7 (server + tuning, ~3 h GPU) — needs user go.
- Step 7 done (2026-10-11, report 18): 10 servers OK + pinned; tuned lr C256 1.5e-4, S+F 1e-3, CJ 1e-3, others 3e-4 (tuned_lr.json sha 0d84d000). Incident: tune.py run from wrong cwd, rerun unchanged from repo root. Parallel gives no GPU speed-up: ~26 h wall left. Next: step 8 dry run (~4.5 h) after user go.
- Step 8 done (2026-10-11, report 19): dry run ALL_PASS, gate CALIBRATED, m_leak 0.03, kappa* 0.05 (results_controls_20261011_231043.json). C1 lobes +0.003 [-0.008,+0.014]. GO for step 9 main (210 runs, ~19.3 h) after user go.
- Step 9 done (2026-10-12, report 20): 210/210 main runs rc 0 (~16 h, detached launch). Monitoring healthy, no collapse. Next: step 10 ablations (~2.9 h) then step 11 scoring+analysis once.
- Steps 10-11 done (2026-10-12, reports 21-22): ablations EQ to F-nat. RESULTS: Q-F F-nat vs C256 EQUIVALENT (lobes -0.023 [-0.034,-0.012], pol -0.010 [-0.015,-0.004]), gate PASS; Q-S HELPS on lobes (S+F +0.049 vs F-nat, +0.038 vs F-X+P); caveat: C64 central is +0.035 above C256. results_main_20261012_211202.json. Next: step 12 independent recomputation.
- Steps 12-13 done (2026-10-12, reports 23-24): independent recomputation REPRODUCED exactly; FINAL REPORT 24. Study complete.
- Option A (2026-10-13, reports 25-26): real Flower system in fl_system/ (.venv-fl, Flower 1.39 compat API). T1 real system BIT-IDENTICAL to fedsim3; T3 isolation, T4 fault+exact resume, T6 PASS. Running T2 (3 full S+F runs fold 0).
- Option A done (2026-10-13, report 27): real FL system VALIDATED; T2 full 200-round S+F runs bit-identical to the study (seeds 0-2); T5 wire overhead 1.1%; ~5 min/run; runbook fl_system/README.md.
- Containers (2026-10-13, reports 28-29): FL system as 11 Docker containers (fl_system/containers/), C1a/C1b bit-identical to study (full 200 rounds), C2 isolation by container FS, C3 docker-kill + resume exact (found+fixed server hang after refused round), C4 1.2% overhead.
- Option A M1+M2 done (2026-10-14, reports 31-32): fl_system2 on current Flower runtime bit-identical to study (full run, 2 platform failures replayed exactly); TLS+auth bit-identical, intruders refused. Next: M3 SecAgg+ (fl_system3).
- Option A M3 done (2026-10-14, report 33): SecAgg+ (fl_system3) M3a decoded err <= bound PASS, M3b |corr| <= 0.005 PASS; M3c FAIL (seed 1 LOBES -0.0186 > 0.01; others within); exploratory (not prereg) paired CIs within study margins; +29 % round time. Next: user decides M4 (DP) and/or M3c' prereg 10-seed equivalence.
- M3c' + M4 launched (2026-10-14, reports 34-35): M3c' pre-registered SecAgg+ vs plain, seeds 3-9 (fl_system3/run_M3cp.sh, ~15.5 h, analysis analyze_M3cp.py frozen). M4 site-level DP: fl_system4 (dp.py, dpsim.py exact sim of real SecAgg+ system, 3-round dev check bit-identical; unit tests U1-U4 PASS); chain fl_system4/run_M4.sh waits for M3c', then V1/calib/V2/curve 8 z x 3 seeds/analysis (~4-5 h). Note: report 35 sec 2 noise/signal used 896/5412 (test recipe); full recipe 2176/12730 -> ~79 z, not 77 z; correct in results report.
- M3c' + M4 done (2026-10-15, reports 36-37): M3c' prereg PRIMARY EQUIVALENT (LOBES +0.0005 [-0.014,+0.015], POL -0.002 [-0.007,+0.003]); M3 accepted. M4: dpsim bit-identical to real system (V1 200 rounds, V2 real DP run); C=0.3975; z*=None; z>=0.3 (eps<=1311) worse than init; site-level DP not recommended at 10 sites. Option A complete.
- Release test (2026-10-15, report 38): fl_system5 full stack (TLS+auth+SecAgg+, 11 containers, netem WAN). Dev found DEFECT: Flower SecAgg+ workflow skips a failed round then hangs; fixed by round-continuity guard (fl_system5/app); kill+resume bit-identical to dpsim. Chain fl_system5/run_rel.sh running (R3 intruders, R4 fault, R1 full 200 rounds) -> check_rel -> report 39; then consolidated report 40 + runbook (option 2). Hourly progress line requested by user.
- Release test + consolidation done (2026-10-15, reports 39-40): RELEASE_PASS (R1 full 200-round full-stack run bit-identical to m3c_SpF_s0; R2 isolation; R3 intruders; R4 kill+guard+resume exact). R5: 0.97/1.90 MB per site-round, 40.9 s/round. Report 40 = consolidated final report; runbook fl_system5/RUNBOOK.md.
- Low-SNR replication launched (2026-10-15 ~14:20, report 41 prereg): no new recordings exist; replicating QF/QS/S+F-vs-central/SecAgg on the never-used 10<=SNR<20 tier (5284 events) via research/model/R_replication/rep.py (fills m4 cache; frozen protocol unchanged). Chain run_rep.sh (log data/k_committee/replication_chain_*.log, resumable), ~16-17 h -> results_rep_*.json -> report 42.
- Correction to the previous line: the replication chain started 2026-10-15 19:25 (not ~14:20); ETA about 11:00 on 2026-10-16.
- Low-SNR replication done (2026-10-16, report 42): PROCEDURE_VALID; SF (S+F vs C256) EQUIVALENT + SecAgg EQUIVALENT -> REPLICATED; QF F-nat vs C256 NON-INFERIOR (lobes +0.023 [+0.005,+0.041], fed better, gate PASS -0.099); QS server pretraining DOES NOT HELP (S+F vs F-nat -0.020 EQ) -> NOT REPLICATED; C64 caveat reversed (-0.039). Report 40 point 2 restricted to SNR>=20.
- Pretraining diagnostic launched (2026-10-16 20:22, report 43 prereg): H-data (Sclean+F: study clean servers -> low-SNR clients), H-lr (re-tune F-nat/S+F by study rule), H-client. Chain research/model/R_replication/run_rep2.sh (log data/k_committee/rep2_chain_*.log, resumable) -> results_rep2_*.json -> report 44.
- 2026-10-17 08:30: diagnostic chain paused overnight because the PC suspended at 22:10 (no failure; resumed on wake). Stage A: study rule re-tunes S+F to 3e-4 (F-nat stays 3e-4) -> H-lr tested with 30 extra runs (60 stage-B runs). Sleep inhibitor (systemd-inhibit, tied to the chain PID) added. ETA ~18:30.
- Pretraining diagnostic done (2026-10-17, report 44): H-data no (Sclean+F vs F-nat -0.012 EQ; clean vs noisy server +0.007 p=0.097), H-lr no (S+F@3e-4 vs F-nat -0.002 EQ), H-client SUPPORTED. Descriptive: server model alone (S0) already = F-nat on low tier (0.42-0.43) -> possible label-noise ceiling. Proposed next: score low-tier models with clean-tier (SNR>=20) probes/labels (~40 min, prereg report 45).
- Ceiling test done (2026-10-18, reports 45-46): low-SNR-trained models scored with clean labels -> PRIMARY HELPS (S+F vs F-nat +0.029 p=2e-4, vs F-X+P +0.023) = reading B (noisy labels hid the gain). F-nat vs C256 -0.018 EQUIVALENT with clean labels (report-42 'fed better' was a label artifact). Untrained->central gap 0.26 vs 0.10 on noisy labels. lr 1e-3 better than re-tuned 3e-4 on clean labels.
- Training-tier study launched (2026-10-18, report 47 prereg): T10 (SNR>=10) runs for C256/F-nat/S+F vs study T20 and replication TLO, clean-label scoring; primary S+F T10 vs T20 + gate -> data rule. Chain run_rep4.sh under systemd-inhibit (log data/k_committee/rep4_chain_*.log) -> report 48.
- 2026-10-18 21:58: tier chain died at 16:51 (PC rebooted; user session killed); 56/100 done; resumed with run_rep4.sh (t10_F-nat_f1_s6 folder kept as .interrupted). ETA ~03:30.
- Tier study done (2026-10-19, report 48): rule output NO FILTER NEEDED (S+F T10 vs T20 EQUIVALENT, lobes -0.017 [-0.032,-0.003], gate PASS); C256/F-nat T10 = T20; noisy-only INFERIOR (-0.055..-0.075). Pre-reg rule text ambiguous for 'EQ with CI<0' (code took NO FILTER); practical rule: S+F default SNR>=20, others may use SNR>=10, never noisy-only.
- Report 49 (2026-10-19): consolidation addendum (updated bottom line, fairness section, figure index) + reports/figures/49_fig1/fig2 + fl_system5/RUNBOOK_v2.md (data rule, lr 1e-3, clean-label scoring).
- Report 50 (2026-10-19): intake kit intake/run.py (+README): integrity -> extraction -> quality/drift -> external test of frozen study models -> report. Self-test on run 20250224 bit-identical to data/events after fixing a thread-pinning defect (numpy imported before extraction's OMP=1).
- Report 51 (2026-10-20): fairness audit central vs federated (user request). Equal in data/model/init/loss/aug/compute/tuning/eval; differs in batch size and batch composition (controls C64, F-IID). New descriptive number: F-nat vs C64 -0.057 [-0.075,-0.040] lobes (INC).
