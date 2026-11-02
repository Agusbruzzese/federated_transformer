# Independent critique: phases F and G (central SimCLR), the v2 → v3 changes, and the orchestration

_Critic agent, 2026-10-01. I had no part in this work. I modified no existing file. Scripts and tables are in this folder:_
- `check_within_space.py` produces `ws_p1_summary.csv`, `ws_p1_per_seed.csv`, `ws_p2_summary.csv` and `ws_p2_per_seed.csv`.
- `check_extra.py` produces `ws_cnn_vs_tf.csv`, `ws_reference_reps.csv` and `ws_morph8_explains_embedding.csv`.

_Compute: about 25 s of wall time on 8 processes._

## Bottom line

Neither pre-registered verdict, F's "Positive ×2" or G's "CNN: no evidence / TF: learns but leaks", describes the models correctly.

**My fair comparison.** I compared each fold's SSL model with *its own initial weights*, in a single embedding space, on the run it never saw. On that comparison, both encoders learn shape structure beyond an untrained network:
- **Transformer:** gains on all 4 targets in all 4 held-out runs, 20/20 seed×run pairs positive on 3 targets and 18/20 on the fourth (log_dur). Polarity skill rises by +0.19 to +0.26 and lobe count by +0.17 to +0.24.
- **CNN:** a gain on lobe count (+0.12 to +0.16) and a small one on polarity (+0.03 to +0.04). It shows **no duration loss** (Δ −0.005 [−0.020, +0.011]).

So the "learns shape structure" conclusion survives. G's pre-registered CNN verdict is an artefact of pooling.

**What does not survive:**
- **"No leakage within a single space."** G measured date and run-excluding-0204 there, but never is-0204. Measured fairly, SSL raises is-0204 skill above its own init by +0.06 to +0.09 for the CNN and +0.02 to +0.06 for the transformer. The CNN rise exceeds v3's own +0.05 tolerance even in the fold where 0204 was held out, and the transformer narrowly exceeds it in 2 of 4 folds.
- **The v3 replacement target `amp_resid`.** It is the log of the noise floor (r = 0.997 with log σ). It measures the recording environment, not the vehicle.

None of this evidence concerns vehicles as such; it concerns waveform morphology, and polarity probably reflects direction of travel. The process (pre-registration, hashing, logging) is better than usual. But the orchestrator's verification is clerical: it checks that numbers match tables, and it does not test whether a design can fail or pass. Two avoidable design errors went through two audits.

---

## A. Performance: what the encoders actually learned

### A1. The fair comparison (my required check)

**Design**
- **SSL:** for held-out run k and seed s, SSL = `ws_f{k}_{arch}_s{s}` restricted to run k. The fold-k model never saw run k.
- **Control:** `{arch}_rand_s{s}`, which is *the same model's initial weights*. `info.json` confirms that the init is identical across folds.
- **Probes:**
  - **within-run:** kNN k = 15, z-scored, GroupKFold(5) by 30-min block, run k only;
  - **transfer:** the same kNN fitted on the 3 training runs' events in the same space, scored on run k.
- **CI:** block bootstrap within each run, runs weighted equally, 5 seeds averaged. `ws_p1_summary.csv`.

| Arch | Target | Δ within-run [95 % CI] | runs > 0 | seed×run > 0 | Δ transfer [95 % CI] | runs > 0 |
|---|---|---|---|---|---|---|
| CNN | amp_resid (see A3) | +0.032 [+0.018, +0.052] | 4/4 | 20/20 | +0.020 [−0.000, +0.038] | 4/4 |
| CNN | log_dur | **−0.005 [−0.020, +0.011]** | 1/4 | 8/20 | −0.002 [−0.013, +0.009] | 1/4 |
| CNN | polarity | +0.041 [+0.029, +0.052] | 4/4 | 18/20 | +0.027 [+0.014, +0.039] | 4/4 |
| CNN | n_lobes | **+0.118 [+0.101, +0.138]** | 4/4 | 20/20 | **+0.163 [+0.134, +0.194]** | 4/4 |
| TF | amp_resid | +0.102 [+0.071, +0.158] | 4/4 | 20/20 | +0.128 [+0.092, +0.179] | 4/4 |
| TF | log_dur | +0.074 [+0.049, +0.104] | 4/4 | 18/20 | +0.070 [+0.044, +0.096] | 4/4 (0212 ≈ 0) |
| TF | polarity | **+0.256 [+0.236, +0.276]** | 4/4 | 20/20 | **+0.188 [+0.169, +0.209]** | 4/4 |
| TF | n_lobes | **+0.168 [+0.148, +0.187]** | 4/4 | 20/20 | **+0.240 [+0.214, +0.267]** | 4/4 |

**Reading.** Under a fair, non-pooled comparison:
- the CNN passes 3/4 by v3's per-target rule (2/4 if amp_resid is excluded as it should be);
- the transformer passes 4/4.

The CNN's gain is essentially *one* target: lobe count. Its polarity gain is about 1/6 of the transformer's.

This broadly replicates F's single-space result on run 0218:
- F's CNN n_lobes was +0.17 and its polarity +0.06.
- F's "CNN loses duration" (−0.046) does **not** replicate across runs. On 0218 alone I get −0.022 [−0.062, +0.014]. G's pooled −0.053 is pooling cost.

G therefore spent an ablation explaining a duration loss that probably isn't there. The ablation's conclusion ("stretch is not the cause") is fine. The G limitation "the CNN's duration loss is unexplained" should instead read "not established".

**G's D1 "matched" fold-init control is biased in favour of SSL.** The SSL fold models share one init, while D1's fold-init networks are 4 *independent* random nets (seed 100 + 10k + s). The fold-init pool is therefore more heterogeneous than the SSL pool, and pays a larger pooling penalty. D1's SSL − fold-init exceeds my fair Δ on every target:

| | amp_resid | log_dur | polarity | n_lobes |
|---|---|---|---|---|
| CNN, D1 | +0.059 | +0.011 | +0.047 | +0.140 |
| CNN, fair | +0.032 | −0.005 | +0.041 | +0.118 |
| TF, D1 | +0.117 | +0.121 | +0.279 | +0.196 |
| TF, fair | +0.102 | +0.074 | +0.256 | +0.168 |

The qualitative conclusion is unchanged, but "pooling matched on both sides" (G REPORT, ORCHESTRATION_LOG l.140) is not true.

### A2. Absolute levels and context (transfer probe, mean over runs; `ws_reference_reps.csv`)

| Representation | amp_resid R² | log_dur R² | polarity skill | n_lobes skill |
|---|---|---|---|---|
| CNN SSL | 0.170 | 0.823 | 0.880 | 0.614 |
| TF SSL | 0.145 | 0.838 | 0.892 | 0.523 |
| CNN random-init | 0.150 | 0.825 | 0.853 | 0.451 |
| TF random-init | 0.018 | 0.768 | 0.704 | 0.283 |
| PCA-64 | −0.019 | 0.355 | 0.689 | 0.197 |
| SNR alone | −0.003 | 0.586 | 0.059 | 0.074 |
| morph8 (circular ceiling) | 0.181 | 0.953 | 1.000 | 0.842 |

- **An untrained CNN already gets most of the way.** It reaches 0.85 polarity skill and 0.83 duration R². The CNN architecture is a strong prior; SSL adds little to it except on lobe count.
- **The transformer's large "gains" are mainly repair of a weak initialisation.** Its untrained version is poor: polarity 0.70, amp_resid ≈ 0. After training it reaches the CNN's level, not beyond it. F audit #3 said exactly this; it is still true.
- **Morph8 + log SNR explain the embeddings linearly about as well as they explain random-init embeddings** (ridge, fit on training runs, R² on the held-out run; `ws_morph8_explains_embedding.csv`):

  | | SSL | random-init |
  |---|---|---|
  | CNN | 0.645 | 0.664 |
  | TF | 0.681 | 0.598 |

  SSL does not make the space markedly more "morphology-like". About a third of the variance is not linearly explained by 19 shape scalars plus SNR. Whether that third is vehicle information or nuisance is unknown.

### A3. `amp_resid` is a noise-floor target, not an amplitude target

- `amp_resid` = m8_log_amp − (1.928 + 1.037·log10 SNR). Since SNR = peak/σ_long, it correlates **r = 0.997 with log10 σ_long**, the event's background noise level.
- The 6-hour bin alone explains 17.7 % of its variance (in-sample bin means), and date explains 7.7 %.
- Every model's held-out R² on it is 0.12–0.18. That is roughly what time of day alone gives.

**v3 therefore replaced a target that leaked SNR with a target that *is* the environment.** Gains on it (TF +0.10, CNN +0.03) are, if anything, a mild leakage signal. The encoder input is max-normalised, so the only way to predict σ is from the noise texture. The orchestrator's sanity check (`sanity_amp_resid.json`) only verified that SNR cannot predict it, which is true by construction. It never asked what the residual *is*.

### A4. CNN vs transformer

On the fair comparison, in absolute held-out values (`ws_cnn_vs_tf.csv`):
- **The CNN is better on n_lobes:**
  - transfer: +0.091, CNN better in 20/20 seed×run pairs;
  - within-run: +0.038, 15/20.
- **The CNN is "better" on amp_resid** (+0.02), which is not a merit (A3).
- **The transformer is slightly better on polarity and log_dur:** −0.013 and −0.015 in CNN − TF terms, with the transformer better in 15/20 pairs each.
- **The CNN leaks more run 0204 identity** (A5): 0.12–0.14 vs 0.05–0.09.

**There is no defensible single winner.** The pre-registered rule (a pooled mean that mixes R² and skill, including the noise-floor target) gives "CNN +0.015" for reasons that are partly artefactual. Saying "the transformer is clearly ahead in gain" (G REPORT) praises it for overcoming a worse init.

**My reading:**
- *For federation:* the transformer has the better leakage profile and comparable shape probes, so it is the reasonable default.
- *For representation quality alone:* call it a tie. The CNN's edge is on lobe count.

### A5. Leakage measured fairly (single space; `ws_p2_summary.csv`)

**Run excluding 0204 (LR).** SSL −0.027 to +0.000 against init −0.016: **no leakage**. This agrees with G's D1.

**is-0204 skill, by the fold whose held-out run is listed**

| Fold held out | CNN LR, SSL (init 0.053) | CNN Δ | CNN Δ, kNN probe | TF LR, SSL (init 0.023) | TF Δ |
|---|---|---|---|---|---|
| 0204 (0204 unseen) | 0.117 | **+0.064** (seed range +0.014 to +0.091) | +0.074 | 0.045 | +0.022 |
| 0212 | 0.139 | +0.086 | +0.096 | 0.087 | +0.064 |
| 0218 | 0.133 | +0.080 | +0.091 | 0.063 | +0.040 |
| 0224 | 0.133 | +0.080 | +0.094 | 0.079 | +0.056 |

- On the kNN probe the transformer's Δ is +0.050 to +0.086.
- **v3's P2c, fairly applied** (≤ init + 0.05 and ≤ morph8 LR 0.079):
  - **CNN: fails**, in every fold, including the fold that never saw 0204.
  - **Transformer:** passes in 2 folds (0204, 0218) and narrowly fails in 2 (0212: Δ +0.064, abs 0.087; 0224: Δ +0.056).
- F audit #6 ("SSL doubles is-0204") is therefore **confirmed, not resolved**. G's statement "Within a single model's space … No leakage" covers only date and run-excluding-0204, and omits the one leakage axis that had previously failed.
- **In absolute terms the leak is small:** skill ≈ 0.1. Some of it may be genuine traffic differences in early February rather than dropout artefacts. But it is a real training-induced effect, and it matters for a federated split by run or day.

### A6. Is any of this about vehicles?

**No direct evidence.** All four targets are hand-crafted descriptors of the same 8 s window. They are generic waveform properties that any decent shape representation captures, and morph8 encodes them perfectly by construction.

**Polarity is probably direction of travel or lane, not vehicle type.** The positive-polarity fraction is 13 % at night vs 35 % in the 06–12 h bin. That pattern is consistent with commuting flow by direction. It may be useful traffic information, but it is not vehicle recognition.

**Lobe count and duration plausibly depend on the vehicle's magnetic signature and on length/speed.** But no analysis separates "vehicle" from "speed" or "lateral offset".

**Without labels, the honest claim is narrow:** "SSL embeddings organise events by waveform morphology at least as well as an untrained CNN, and better than an untrained transformer, PCA or SNR, without date leakage. They modestly increase run-0204 identifiability." Nothing more.

---

## B. The v2 → v3 changes

| Change | Verdict | Why |
|---|---|---|
| log_amp → `amp_resid` | **Wrong (harmful)** | It removes SNR leakage by creating a noise-floor target (A3). The right fix was to drop amplitude from P1 and add log σ_noise to P2 as a leakage axis. The auditor proposed the residualisation; the orchestrator accepted it without checking what the residual is. |
| 5 seeds, seed-level Δ ranges | Right | Seed SD of Δ is up to 0.05 (TF polarity), so 3 seeds were inadequate. |
| LORO | Right in intent, **wrong in implementation** | Pooling OOF embeddings from 4 networks into one probe matrix is the central flaw. It made P2a–c unpassable (an untrained pooled control gets run skill 1.000). It also biased P1 *against* SSL, since the single-network random-init doesn't pay the pooling cost (−0.04 to −0.14). F's single-space design was fair; v3 replaced it with an unfair one. Per-fold single-space evaluation plus meta-analysis was the obvious alternative. |
| Absolute P2 thresholds (≤ 0.03) | Right for single-space; fatal with pooling | Also: no positive or negative control was run to show that the criteria are passable and failable before freezing. |
| P2c is-0204 vs own init | **Right criterion**; never fairly evaluated | When evaluated fairly, it fails for the CNN (A5). |
| P2d has_dropout within 0204, within-block null | Mostly right | The null is still positive (0.06–0.09) because prevalence varies within blocks. Fine as a paired test. |
| LR leakage probe (fast) instead of HGB | **Questionable** | It is a weaker adversary. morph8's is-0204 skill is 0.079 under LR vs 0.351 under HGB in F. "≤ morph8" thresholds became laxer while the headline said "tightened". HGB date was dropped (deviation 1), and HGB on pooled embeddings is meaningless (skill 0.999). |
| Selection by leave-one-day-out CV | Right in principle, wasted in practice | Margins ≤ 0.009. Selection embeds the SSL model's *own training events*, so it is in-sample for the encoder. 32 grid trainings (4 folds × 2 archs × 4 cells) bought nothing. The hyperparameters should have been fixed. |
| Secondary tiers descriptive only | Right | |
| Hashing all source files at freeze and end | Right, and verified | Credit to the orchestrator. |
| CNN no-stretch ablation | Fine as executed | It tests an effect that the fair analysis does not show exists. |
| CNN-vs-TF on a pooled P1 mean that mixes R² and skill | Wrong | Mixed units, includes the noise-floor target, and pooled. |

**F/G audit concerns still unaddressed**
1. **Target circularity:** the crop is aligned to the detector, and the targets are the detector's own morph features (F audit #11). There is still no external target.
2. **Additional baselines** (F audit #3): a trained-on-shuffled or phase-randomised encoder, or a supervised-on-morph8 encoder. Without these, it is unclear whether a CNN gain of +0.12 on lobes is large or trivial.
3. **Block-size skew** (F audit #9, max 75 events per block): no capped or weighted sensitivity analysis.
4. **is-0204 within a single space**: computed here for the first time; it fails for the CNN.
5. **Human labels:** the 80-item pack is still unlabelled after two phases. It is the only external check available, and it costs the user perhaps 30 minutes.
6. **Weakly held-out data:** C's detector design saw run 0218. LORO dilutes this but doesn't remove it.

---

## C. The orchestrator's work

### C1. Process: what is genuinely good
- **Pre-registration was real.** SHA-256s were logged, and the deviation lists are honest. The F verdict was **kept even after the orchestrator knew the criteria were too loose** (LOG l.112–113). That is correct and uncommon discipline.
- **Every audit point is logged** with an accept/reject reason, and SYNTHESIS §6 tables the rejections with evidence. Spot-checked, the rejections are mostly sound (A2 #2 fs, B2 #6 missing files, C2 #12 repair function).
- **Some verification was substantive.** In phase 1 the orchestrator independently reproduced the int16 bug (LOG l.70) and the dropout totals (l.31), and checked joins (l.33).
- **Errors were admitted openly** (the F criteria; the v3 pooling).

### C2. Process: weaknesses
1. **In phase F/G, verification became clerical.** Every orchestrator check on F and G (LOG l.108–110 and l.137; G REPORT header) is a hash match or "the table matches the text". None recomputed a metric or probed a claim. A one-line check would have shown amp_resid ≈ log σ. A 20-second check would have shown is-0204 within a single space. Neither was done before "No leakage" was written.
2. **No dry run of the frozen evaluation on null and control representations.** Running v3's P2 on pooled random-init embeddings, before freezing, would have shown run skill 1.000 in seconds. The G *agent*, not the orchestrator, foresaw this (D1 in `config.py`, l.60). The orchestrator should adopt this as a rule: every criterion must be shown failable by a planted-leak positive control and passable by a clean negative control before freeze.
3. **The stopping rule lets the orchestrator self-certify:** "every remaining MAJOR … verified by Claude". Combined with clerical verification, MAJORs close on assertion.
4. **The auditor cannot catch design flaws of this kind.** It is Sonnet at medium effort, with truncated inputs (B2 #6 was an omission due to size) and about $0.16 per audit. It reviews text, never runs code, and missed the pooling flaw in v3 (LOG l.134: "no design blocker against v3"). Treating its silence as clearance is a process error. Its *suggestions* (residualise; absolute thresholds) were also adopted without the orchestrator checking their consequences.

### C3. Decisions it got wrong
- **v3 design:** pooled OOF evaluation (B). This is the costliest error: it invalidated the entire primary analysis of a 3.3 h run.
- **amp_resid** (A3).
- **The LR leakage probe** as primary while claiming tightening (B).
- **Selection CV:** effort spent on a noise-level choice (B).
- **Moving to v3 without labels.** RECAP §6 listed "get labels" as option 3. The user chose tightening, but the orchestrator never argued that tightening proxy targets cannot answer the question that matters (vehicles).
- **Phase-1 parallelisation against audit 1 #1:** admitted later as costly (SYNTHESIS §6).

### C4. Criticism handled wrongly
**Accepted but invalid, or accepted uncritically:**
- F audit #2's "SNR-residualised probe" became the noise-floor target.
- The auditor's absolute-threshold advice was adopted without considering the evaluation geometry.

**Valid but left unacted** (logged as "carried", then not carried):
- F audit #3's additional baselines;
- F audit #9's block-size sensitivity;
- F audit #11's circularity;
- audit 10-01 #7, follow-up 7 (human labels).

**Rejections:** I found no clearly invalid rejection. The 10-01 BLOCKER ("F's Positive is not supported") was treated as framing (LOG l.135). That is defensible, because the verdict was kept as the rule output with an annotation.

### C5. Framing to the user
**Over-claims:**
- **G REPORT / LOG l.139:** "Within a single model space, leakage is ≈ 0 … No leakage." is-0204 was not measured, and it is elevated (A5).
- **"Against the matched fold-init control: CNN 3/4, TF 4/4."** The control is not matched (A1). The fair count happens to be similar, but the stated Δs are inflated.
- **RECAP §5 for F:** "Real gains … duration for the transformer only" and "No date or run leakage outside run 0204". Run 0204 is 41 % of the events and the main federated concern, so "outside run 0204" carries a lot of weight.
- **"amp_resid sanity" presented as validation**, when it only checks a tautology.
- **"The duration hypothesis is NOT supported" as the headline ablation finding.** It hides that the effect being explained is itself not established.

**Under-claims:**
- G's headline table leads with "CNN: No evidence of learning". Although it is flagged as a design artefact, a reader skimming sees the CNN as failed. The fair analysis shows a clear lobe-count gain.

**Framing in general:** the orchestrator is candid about its errors *after* they surface, and that is to its credit. But it consistently writes the most favourable reading of diagnostics that were not pre-registered: D1 "matched", "No leakage".

### C6. Wasted effort
- **F:** about 6 h of HGB leakage probes (RECAP lessons). Secondary tiers with 2–5 permutations, which were then declared not evaluable.
- **G:**
  - 32 grid trainings plus LODO selection for noise-level differences;
  - LR P2 with 50 permutations, HGB P2 and the tiers, all run on pooled embeddings and therefore uninterpretable;
  - the no-stretch ablation, chasing a non-replicating effect.

  A large share of G's 1.6 active hours produced primary numbers that the report itself calls invalid.
- **A progressive Claude Doc report** was being filled while the results were invalid by design. Its framing must now be revised.

---

## D. The single most important next step, and what v4 must look like

**Most important next step: get an external target.**

*Minimum:* have the user label the existing 80-item blind pack (vehicle vs non-vehicle).

*Better:* label 200–500 events for **direction/lane and coarse class** (car / van / truck / bus) from a camera or a site log for a few hours on two runs.

Why: every further tightening of proxy targets refines an answer to "does SSL encode the detector's own morphology features?". That question is now answered: yes, modestly, and with the CNN largely free from its architecture. No v4 on proxies can be decisive about *vehicles*, which is what the federated system is for.

**v4 evaluation (decisive, and cheap with the existing checkpoints):**
1. **Unit of analysis: one model, one space, per fold.**
   - For held-out run k, the fold-k model embeds everything.
   - The probe is fitted on the training runs and scored on run k (transfer). Within-run CV is secondary.
   - The control is *that model's own init*.
   - Aggregate over 4 folds × 5 seeds. Pass if the block-bootstrap CI excludes 0 **and** ≥ 3/4 runs are positive **and** Δ ≥ a pre-stated minimum effect (e.g. +0.03 skill / R²).
   - This also matches the federated setting (clients ≈ runs/days). `check_within_space.py` here is a working template.
2. **Targets:**
   - Primary: external labels (V/N AUC; direction; class if available).
   - Secondary proxies: polarity, n_lobes, log_dur.
   - **Drop amp_resid** and move log σ_noise to the leakage family.
3. **Leakage, in the same single space:** is-0204, date, run-excluding-0204, hour bin and log σ_noise.
   - Probes: LR **and** kNN (and HGB if affordable). Use the strongest probe's value.
   - Criterion: SSL − own init ≤ +0.05 with the CI upper bound reported.
4. **Freeze only after a dry run.** Run the entire pipeline on:
   - (i) random-init embeddings;
   - (ii) a planted-leak control (random-init + a small run one-hot), which must **fail** P2;
   - (iii) a planted-signal control (random-init + noisy target), which must **pass** P1.

   Log these outputs with the hash.
5. **Baselines:**
   - morph8 (ceiling), PCA, SNR, own init;
   - **an SSL encoder trained on phase-randomised or shuffled-crop events** (the "trivial invariance" null);
   - a supervised-on-morph8 encoder (an upper reference for what the architecture can extract).
6. **CNN vs transformer:** per target, absolute held-out values, paired by seed and run. No mixed-unit means. Leakage is a co-primary criterion for the federated choice.
7. **Hyperparameters fixed** from F/G (selection is noise). Spend the compute on seeds and controls instead.
8. **Report a block-size-capped sensitivity analysis.** Verification by the orchestrator must include recomputing at least one primary Δ and one leakage number independently, as was done here in about 25 s.
