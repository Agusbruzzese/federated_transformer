# C2: groups of look-alike events (exploratory report)

_Orchestrator, 2026-10-02._
- PREREG v3: v1 sha c33c1dcb, then E1–E7 after the audit, then F1 after the failed dry run.
- Code pins are in `pins_at_run.txt`; the last line is the run that produced these results.
- Status: **exploratory, post-hoc** (designed after C1).

## Definition used (the user's, made explicit)
Two events are look-alikes when all three hold:
- aligned **shape correlation ≥ 0.90**, allowing a shift of up to 1 s;
- **peak amplitude within 1.5×**;
- **duration within 1.5×**.

C2a groups events by complete linkage on this rule, so **every pair inside a group satisfies all three**. This was verified independently: worst within-group amplitude ratio 1.494, duration ratio 1.490, minimum correlation 0.900.

## Dry run
- **v2 FAILED** (`dryrun_v2_FAILED.json`, ARI 0.58). The planted amplitude spread (1.56×) exceeded the tolerance, which forces splits. This was a specification error, fixed by amendment F1 before rerunning.
- **v3 PASS:** ARI 0.90, homogeneity 0.97, singletons 1.00 not grouped, null 0 groups.
- **SNR 20–40 variant:** ARI 0.73.
- **Wider amplitude spreads:** templates are **split, not merged**; homogeneity 0.97 at ×0.8–1.25 and 0.87 at ×0.5–2.

## Results (`results.json`, `c2_labels.csv`, `profiles_*.csv`, `gallery_*.png`)

| | Groups | Coverage | Amplitude p90/p10 | Duration p90/p10 | Median corr | LORO ARI (mean / min) | LORO frac joint | Beats random every fold | Reliable |
|---|---|---|---|---|---|---|---|---|---|
| **C2a direct** | 106 | 0.385 | 1.24 | 1.22 | 0.971 | **0.378 / 0.325** | **0.16** | yes (random p95 ≤ 0.02) | **no** |
| C2b SSL consensus | 38 | 0.140 | 1.22 | 1.23 | 0.975 | **0.080 / 0.000** | **0.01** | no | **no** |

**Nulls:**
- phase-randomised waveforms: 0 groups (10 draws);
- background-noise windows: 0 groups;
- time-reversed copies (descriptive): 84 of 223 groups mix originals with reversed events, so many prints are nearly time-symmetric.

**Sensitivity (C2a):**
- strict (0.95, 1.25×): 26 groups, 7 % coverage;
- loose (0.85, 2×): 126 groups, 64 %;
- 8 s crop: 121 groups, 46 % coverage, ARI 0.57 against the main crop;
- SNR ≥ 30: 95 groups, 43 % coverage.

No group is dominated (> 80 %) by a single run.

**Look-alikes are abundant but not partitionable.** These numbers come from `verify.py` → `verify.json`. 88 % of events have at least one look-alike under the main tolerance, 78 % have ≥ 10, and the median event has 89. The look-alike relation is not transitive (A≈B and B≈C does not give A≈C), so a population with a continuous spread is cut into tolerance-width bins. The galleries show this: groups 90, 92, 93, 87 and 84 have the same slow negative-dip shape at 2.7k–4.0k counts. That is why the **group boundaries** reproduce only at ARI ~0.38 across runs, although each group is genuinely tight and the nulls form no groups.

## C2b and the "does SSL help?" question
- **The C2b LORO is largely degenerate.** On a single held-out run, the fold models' HDBSCAN clustering collapses, leaving ≤ 1 kept seed in folds 0, 1 and 3. The pre-registered fallback then sets those folds to ARI 0.
- **`ssl_helps = true` in `results.json` is INVALID; disregard it.** The random control was not matched (coverage 0.004 vs C2b's consensus 0.35) because of integer truncation, and refinement inside random partitions finds almost nothing by construction (audit #1).
- **Verdict on SSL: inconclusive** (revised after audit #2). C2b's LORO measures the collapse of HDBSCAN on single-run embeddings, not what SSL can do. On the full data C2b covers 14 % vs 38.5 % for C2a, with no gain in tightness. So SSL is **not shown** to help, and this test cannot say it doesn't.

## Verdict
- **High confidence:** groups of near-identical prints exist (0 groups under all nulls), and the C2a groups meet the user's definition by construction. They are visually convincing; see `gallery_C2a_random.png`.
- **Not supported:** a stable partition of events into discrete look-alike classes. Look-alikes form a continuum of overlapping neighbourhoods, and fixed group boundaries are arbitrary (LORO ARI 0.38 < 0.50).
- **No evidence that SSL helps** this specific task.
- **Practical recommendation** (untested, for the user to decide):
  - represent similarity as a **look-alike graph or query**: for any event, list all events within tolerance (all pairs are already computable);
  - or report groups from C2a as one convenient cover, stating that the boundaries are arbitrary.

## Non-partition reproducibility: do prints recur? (`verify.json`; added after audit #3)
- **87 %** of events have a look-alike **in a different run** (95 % CI 0.86–0.88, bootstrap over events).
  - 88 % have one on a different day; 88 % have one more than 1 h apart.
  - Per run: 0.88, 0.84, 0.86, 0.88.
- **Shape is a real constraint:** only **22 %** of the pairs that match in amplitude and duration also match in shape.
- **So the prints recur across runs and days;** it is only the *partition* into fixed groups that does not reproduce.

## Audit 124215 responses
- **Accepted:**
  - #1, `ssl_helps` invalid;
  - #2, C2b inconclusive;
  - #3, recurrence measure added;
  - #4, numbers now traceable via `verify.py`;
  - #5, dry-run PASS is conditional on amendment F1 (the original spec gave 0.58; the `amp0.8_1.25` entry is that spec);
  - #9, the time-reversal result is descriptive only (aligned with chunk 8 for memory);
  - #10, "visually convincing" is a subjective judgement;
  - #11, the SNR ≥ 20 bias applies to all conclusions.
- **#6, seed text:** E7 said `rng(2)`, but the dry-run synthetic data used `rng(1)`, the same set as C1 D10, while the nulls used `rng(2)`. This is disclosed, not changed.
- **#7, feature paths:** already checked before the run. `feats()` on crop[99:501] reproduces the stored m8 log_amp and dur10 (r = 1.000, median |Δ| ≤ 0.001).
- **#8:** consensus memory was 4375² float32, about 77 MB (not 5 GB); no issue.
- **#12:** there are no CIs on the LORO ARI, since only 4 folds exist. Fold values are listed in `results.json`.
- **Deferred to a possible C3:** re-running C2b with fold-adapted HDBSCAN parameters.

## CORRECTIONS after independent review (`review/`). These override the claims above.
1. **"Nulls give 0 groups" is trivial.** Phase-randomised crops are noise. A **shuffled-shape null** keeps each event's amplitude and duration and swaps its shape with another event in the same amplitude×duration decile. It still gives **93 groups, 34.9 % coverage** (orchestrator, seed 11); the reviewer got 100–105 groups and 37–38 % (real: 106 and 38.5 %).
   - **So C2a groups are tiles of real shape families cut by amplitude and duration, not distinct joint prints.** They still satisfy the user's tolerance pairwise.
2. **"Prints recur across runs" is withdrawn.**
   - Shuffled-shape recurrence is 0.867 against 0.869 observed.
   - Look-alike pairs come from the same run at 0.302 vs 0.288 for all pairs (orchestrator check).
   - What the data shows is that **the population of shapes is stable over 18 days.** It does not show that individual prints or vehicles repeat.
3. **Continuum vs discrete is refined.** At the fine level (shape + amplitude + duration) the look-alike graph behaves like a continuum: one component holds 85 % of events, and Louvain reproducibility is like a cluster-free point cloud. **At the coarse level of shape alone, there are a few stable families** (below).
4. **Correlation definition.** The correlation is normalised by the full-window norm, not the overlap's (up to 0.18 lower at large shifts, so conservative). It is correctly called "aligned normalised cross-correlation", not Pearson.
5. **Transfer method.** LORO ARI is 0.38 with medoid transfer and 0.46 with the reviewer's neighbour vote; the verdict is unchanged.

## Shape families (post-hoc, exploratory; `families.py` → `families.json`, `families.csv`, `gallery_families.png`)
- **Method:** Louvain on the shape-only graph (corr ≥ 0.9), keeping families of ≥ 20 events. This gives **7 families covering 87.7 %** of events.
- **Stability (orchestrator's own check):**
  - seed ARI 0.91–1.00;
  - leave-one-run-out ARI 0.64, 0.70, 0.68 and 0.58 (mean 0.65), with 76–86 % of events transferred;
  - the reviewer found 9 families with similar stability.
- **Inside a family**, amplitude varies 3–5.6× (p90/p10) and duration varies continuously. C2a groups nest inside families: 77 % of groups sit ≥ 90 % inside one family.

## FINAL VERDICT (supersedes the earlier verdict)
- **Reliable:**
  - **≈ 7 shape families** (exploratory but reproducible across runs);
  - inside them, look-alikes form a **continuum along amplitude and duration**;
  - C2a groups meet the user's tolerance pairwise but are **arbitrary tiles** of that continuum;
  - look-alikes are abundant (88 % of events have one).
- **Not reliable:** fixed fine-group boundaries; repeat-vehicle claims; any benefit from SSL (inconclusive).
- **Recommended deliverable** (reviewer + orchestrator; untested as a product): a **two-level index**:
  1. the shape family;
  2. inside it, continuous amplitude and duration, plus a **look-alike query** listing every event within tolerance.
- **Optional prototype library:** 10 prototypes cover 48 % of events, 100 cover 82 % (reviewer `r4.json`). Two events matched to the same prototype can differ by up to 2× the tolerance.
