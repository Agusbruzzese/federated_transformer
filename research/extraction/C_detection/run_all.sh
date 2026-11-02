#!/usr/bin/env bash
# Regenerates every number in REPORT.md (order matters). Seeds are fixed inside the scripts.
# Hand labels (data/hand_labels.csv, pf_labels_*.csv, bdet_labels_hold.csv) are human inputs and are NOT regenerated;
# the candidate files they refer to are deterministic (seeds 0/1/2/3) -- do not re-run the `make` steps unless relabelling.
set -euo pipefail
cd "$(dirname "$0")"
PY=../../../.venv/bin/python
$PY scripts/validate_io.py                 # cio.py vs A_integrity/bsd_io.py
$PY scripts/build_spans.py                 # 12 tuning + 10 holdout 20-min spans -> data/span_*.npz
$PY scripts/templates.py                   # 61 injection templates from tuning runs only
$PY scripts/dropout_test.py                # dropout handling (concat / raw interp / 50 Hz mask)
$PY scripts/inject_eval.py tune            # injection + surrogate sweep (Runs 1,2,4)
$PY scripts/analyze_inj.py tune            # operating points -> data/frozen_params.json
$PY scripts/inject_eval.py holdout         # frozen settings on Run 3
$PY scripts/analyze_inj.py holdout
$PY scripts/stability.py
$PY scripts/diurnal.py                     # ~1/4 of all chunks, 1227 blocks
$PY scripts/summarize.py
$PY scripts/segment_repr.py               # incl. pre-registered rule as written -> data/tab_segrepr_rule.csv
# audit follow-up (2026-09-30)
$PY scripts/templates.py bdet ; $PY scripts/templates.py wavelet
for s in bdet wavelet; do for sp in tune holdout; do TSET=$s $PY scripts/inject_eval.py $sp; done; done
$PY scripts/ranking.py                     # ranking per template set
$PY scripts/poisson_ci.py                  # exact CIs for zero/low counts
$PY scripts/bias.py                        # clean-tier selection bias
# $PY scripts/human_pack.py                # blind human pack (already generated; re-running reproduces it, seed 7)
# $PY scripts/handcheck.py make ; $PY scripts/postfilter.py make_tune ; make_hold ; $PY scripts/tier_compare.py make
$PY scripts/handcheck.py score
$PY scripts/postfilter.py fit              # rule on tuning labels only (already frozen in data/postfilter_rule.json)
$PY scripts/postfilter.py score
$PY scripts/tier_compare.py score
