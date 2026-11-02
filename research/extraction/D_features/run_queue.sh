#!/bin/bash
# Sequential queue (<=4 worker processes at a time). Reproduces all secondary evaluations.
set -x
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
PY=../../../.venv/bin/python
PREFIX=C_ N_EVAL_DEV=700 N_EVAL_HOLD=500 $PY -W ignore compute_all.py
$PY -W ignore check_C.py evaluate
$PY -W ignore evaluate.py --large
for c in 5 8 10; do LP_CUT=$c $PY -W ignore compute_all.py; $PY -W ignore evaluate.py --lp $c; done
$PY -W ignore confound_test.py
$PY -W ignore norm_test.py
$PY -W ignore div_test.py
echo QUEUE_DONE
