#!/bin/bash
# Queue 2: check (b) on workstream C's FINAL detector + C design questions (SNR tiers) + div test.
set -x
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
PY=../../../.venv/bin/python
while pgrep -f "run_queue.sh" > /dev/null; do sleep 20; done
$PY -W ignore check_C.py extract
PREFIX=C_ N_EVAL_DEV=700 N_EVAL_HOLD=500 $PY -W ignore compute_all.py
$PY -W ignore check_C.py evaluate
$PY -W ignore evaluate.py --C-tier 20
$PY -W ignore evaluate.py --C-tier 30
$PY -W ignore div_test.py
echo QUEUE2_DONE
