#!/usr/bin/env bash
# T2 chain (report 25): 3 full real-system S+F runs (fold 0, seeds 0-2), then the comparison with the study's S+F runs.
set -e; cd "$(dirname "$0")/../.."
for s in 0 1 2; do
  .venv-fl/bin/python fl_system/run_local.py --fed data/fl_system/fold0_v2/federation.json --run t2_SpF_s$s --seed $s --rounds 200 \
     --examples 2560000 --batch 64 --lr 1e-3 --init data/fl_system/fold0_v2/server/server_init_s$s.pt --port 810$s
done
.venv/bin/python fl_system/tests/test_T2_fullrun.py
echo CHAIN_EXIT 0
